"""SOTA training + W&B-instrumented diagnostic primitives, shared across
baselines (HyperIV, GNO, ...).

Pulls together the boilerplate that every training loop in this repo wants:

  * ``TrainerConfig`` — dataclass with optimization / W&B / AMP fields.
  * ``init_wandb(config, model, extra_config=None)`` — initialises a run
    targeting ``maths-ox/research-iv`` by default, tagging by
    ``config.variant`` and registering ``wandb.watch(model)`` for grad
    histograms. No-ops cleanly when wandb is unavailable or disabled.
  * ``set_seed`` / ``device`` — single source of truth.
  * ``AverageMeter`` — online mean (no per-batch list-then-concatenate).
  * ``grad_stats`` / ``grouped_grad_stats`` / ``omega_block_grad_stats`` —
    per-step diagnostics. ``grouped_grad_stats`` consults
    ``model.gradient_components()`` if defined (HyperIV variants), falling
    back to top-level named children otherwise (GNO etc.).
  * ``maybe_scrub_nans`` / ``clip_grads_to_norm`` — replacements for the
    bespoke nan_to_num + manual clip patterns previously duplicated.
  * ``amp_context`` — bf16 autocast on CUDA when enabled, ``nullcontext``
    otherwise. Note: any autograd-through-autograd construction (e.g.
    HyperIV's ``aux_loss``) should NOT be wrapped — it's fp32-stable in
    a way bf16 isn't.

This module deliberately does *not* expose a generic ``run_epoch`` /
``fit`` — the per-batch step shape differs between HyperIV (dense (z, X)
batches) and GNO (per-surface ``Data`` graphs), so the per-baseline
``train.py`` keeps its loop and uses these primitives à la carte.
"""

from __future__ import annotations

import math
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn

try:
    import wandb  # type: ignore
    _HAS_WANDB = True
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore
    _HAS_WANDB = False


# =====================================================================
#  Device + seeding.
# =====================================================================


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
#  Online metric meter.
# =====================================================================


class AverageMeter:
    __slots__ = ("sum", "count")

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


# =====================================================================
#  Trainer config.
# =====================================================================


@dataclass
class TrainerConfig:
    # Optimization
    epochs: int = 100
    lr: float = 1e-3
    eta_min: float = 1e-5
    weight_decay: float = 0.0
    grad_clip: float | None = None        # None disables; e.g. 1.0 enables
    nan_to_num_grad: bool = True          # NaN/Inf grads -> 0 in-place
    use_amp: bool = False                 # bf16 autocast on CUDA when True

    # Logging
    log_grad_every_step: int = 50         # 0 disables per-step grad logging
    log_param_hist_every_epoch: int = 25  # 0 disables param histogram dump
    track_predictions: bool = True

    # Run identity
    variant: str = "vanilla"
    run_name: str | None = None
    seed: int = 0
    extra_tags: tuple[str, ...] = ()

    # W&B
    wandb_entity: str | None = None
    wandb_project: str | None = "pivot-anon"
    wandb_mode: str = "disabled"          # "online" | "offline" | "disabled"
    wandb_watch: str | None = "gradients"  # "gradients" | "parameters" | "all" | None
    wandb_watch_freq: int = 200

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =====================================================================
#  W&B initialisation.
# =====================================================================


def init_wandb(
    config: TrainerConfig,
    model: nn.Module,
    extra_config: Mapping[str, Any] | None = None,
) -> Any | None:
    """Initialise a W&B run for this training session.

    Returns the run handle (or ``None`` when wandb is unavailable / disabled).
    Safe to call even without wandb installed — caller treats ``None`` as
    "no logger".
    """
    if not _HAS_WANDB or config.wandb_mode == "disabled":
        return None
    cfg = config.to_dict()
    cfg.update(dict(extra_config or {}))
    cfg["model_class"] = type(model).__name__
    cfg["param_count_total"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cfg["param_count_all"] = sum(p.numel() for p in model.parameters())
    if hasattr(model, "omega_dim"):
        cfg["omega_dim"] = int(model.omega_dim)

    tags = list(config.extra_tags) + [config.variant]
    run = wandb.init(
        entity=config.wandb_entity,
        project=config.wandb_project,
        name=config.run_name,
        tags=tags,
        config=cfg,
        mode=config.wandb_mode,
    )
    if config.wandb_watch is not None:
        wandb.watch(
            model,
            log=config.wandb_watch,
            log_freq=max(config.wandb_watch_freq, 1),
            log_graph=False,
        )
    return run


def log_param_histograms(logger: Any | None, model: nn.Module, step: int | None = None) -> None:
    """Dump a histogram-per-named-parameter to W&B (no-op if logger is None)."""
    if logger is None or not _HAS_WANDB:
        return
    payload: dict[str, Any] = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            payload[f"params/{name}"] = wandb.Histogram(p.detach().float().cpu().numpy())
    if step is not None:
        logger.log(payload, step=step)
    else:
        logger.log(payload)


# =====================================================================
#  Gradient diagnostics + scrubbing + clipping.
# =====================================================================


def maybe_scrub_nans(model: nn.Module) -> int:
    """Replace NaN/Inf grads with zeros in-place. Returns count of scrubbed
    elements (0 if all grads were already finite).
    """
    n = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        bad = ~torch.isfinite(p.grad)
        if bad.any():
            n += int(bad.sum().item())
            p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
    return n


def clip_grads_to_norm(model: nn.Module, max_norm: float | None) -> torch.Tensor | None:
    """Wrapper around ``clip_grad_norm_`` that no-ops on ``None``. Returns the
    pre-clip total norm (or ``None`` when disabled)."""
    if max_norm is None:
        return None
    return torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        max_norm=max_norm,
        error_if_nonfinite=False,
    )


def is_finite_params(model: nn.Module) -> bool:
    """True iff every parameter tensor is finite. Useful for catching
    parameter-side overflow that didn't fully NaN the loss this step."""
    return all(torch.isfinite(p).all() for p in model.parameters())


def grad_stats(model: nn.Module) -> dict[str, float]:
    """Aggregate gradient statistics across all trainable params."""
    total_sq = 0.0
    max_abs = 0.0
    n_params = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total_sq += float(g.pow(2).sum().item())
        max_abs = max(max_abs, float(g.abs().max().item()))
        n_params += g.numel()
    return {
        "grad/total_norm": math.sqrt(total_sq),
        "grad/max_abs": max_abs,
        "grad/n_params_with_grad": float(n_params),
    }


def grouped_grad_stats(model: nn.Module) -> dict[str, float]:
    """Per-group grad / param norms.

    Group source preference:
      1. ``model.gradient_components()`` if defined (HyperIV variants).
      2. ``model.named_children()`` — one group per top-level child.
    """
    if hasattr(model, "gradient_components") and callable(model.gradient_components):
        groups: dict[str, list[nn.Parameter]] = model.gradient_components()
    else:
        groups = {}
        for name, child in model.named_children():
            params = [p for p in child.parameters() if p.requires_grad]
            if params:
                groups[name] = params

    out: dict[str, float] = {}
    for gname, params in groups.items():
        gsq = 0.0
        psq = 0.0
        for p in params:
            if p.grad is not None:
                gsq += float(p.grad.detach().pow(2).sum().item())
            psq += float(p.detach().pow(2).sum().item())
        gn = math.sqrt(gsq)
        pn = math.sqrt(psq)
        out[f"grad/group/{gname}/grad_norm"] = gn
        out[f"grad/group/{gname}/param_norm"] = pn
        out[f"grad/group/{gname}/grad_to_param"] = gn / max(pn, 1e-12)
    return out


def omega_block_grad_stats(model: nn.Module) -> dict[str, float]:
    """Per-h_omega-block grad norms on the *generated* omega vector.

    Only meaningful for hypernetwork-style models that expose
    ``omega_block_grad_norms()`` (HyperIV variants). Empty dict otherwise.
    """
    if hasattr(model, "omega_block_grad_norms") and callable(model.omega_block_grad_norms):
        return {f"omega_grad/{n}": float(v) for n, v in model.omega_block_grad_norms().items()}
    return {}


def collect_grad_diagnostics(model: nn.Module) -> dict[str, float]:
    """Bundle ``grad_stats`` + ``grouped_grad_stats`` + ``omega_block_grad_stats``."""
    out: dict[str, float] = {}
    out.update(grad_stats(model))
    out.update(grouped_grad_stats(model))
    out.update(omega_block_grad_stats(model))
    return out


# =====================================================================
#  AMP context.
# =====================================================================


def amp_context(use_amp: bool, dev: torch.device) -> Any:
    """Return ``torch.autocast(bfloat16)`` on CUDA when ``use_amp`` else
    ``nullcontext()``. CPU/MPS callers always get ``nullcontext()`` because
    bf16 autocast there is either unsupported or pessimal."""
    if use_amp and dev.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


# =====================================================================
#  Backwards-compat aliases (private leading-underscore names used by
#  models/hyperiv/trainer_util.py before the lift).
# =====================================================================

# These are kept so HyperIV's trainer_util can re-export under its
# pre-existing names without breaking the (notebook-level) import contract.
_maybe_scrub_nans = maybe_scrub_nans
