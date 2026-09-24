"""Aux-loss formulas + training-helper aliases used by every variant.

The two auxiliary-loss formulas (``price_aux_loss`` and the sentinel-protected,
vega-gated ``rt_aux_loss``) are the central methodological contribution of
these experiments and were previously duplicated across the per-experiment
``train.py`` files. Defining them once here keeps them numerically and
behaviourally identical across baselines.

Device / seed / NaN-scrub / finite-param helpers are re-exported from the
shared SOTA training primitives in ``experiments.utils.training`` so there
is exactly one canonical implementation; ``sanitize_grads`` is an
``maybe_scrub_nans``-backed alias preserved for legacy callers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch

# Make the shared training module importable regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from experiments.utils.training import (  # noqa: E402
    device,
    set_seed,
    is_finite_params,
    maybe_scrub_nans,
)


def sanitize_grads(model: torch.nn.Module) -> bool:
    """Replace non-finite grads in-place with zeros. Returns True if any
    were found (legacy bool API; new code should use
    ``experiments.utils.training.maybe_scrub_nans`` which returns a count).
    """
    return maybe_scrub_nans(model) > 0


def price_aux_loss(price_hat: torch.Tensor, mid: torch.Tensor) -> torch.Tensor:
    """Batch-mean-abs-scaled MSE between model price and market mid."""
    scale = mid.abs().mean().clamp_min(1e-3)
    return ((price_hat - mid) / scale).pow(2).mean()


def rt_aux_loss(
    price_hat: torch.Tensor,
    vega_hat: torch.Tensor,
    is_call: torch.Tensor,
    F: torch.Tensor,
    K: torch.Tensor,
    tau_yr: torch.Tensor,
    r_rate: torch.Tensor,
    sigma_star: torch.Tensor,
    vega_floor: float,
    use_gate: bool,
    use_sentinel: bool,
) -> Tuple[torch.Tensor, int]:
    """Sentinel-protected, vega-gated IV-roundtrip loss.

    Returns ``(loss, n_unsafe)`` where ``n_unsafe`` is the count of rows
    whose ``|vega|`` was below ``vega_floor`` (and whose ``price_for_iv``
    was therefore replaced by the analytic Black-76 sentinel).
    """
    # Local imports keep this module importable in environments where
    # fast-vollib isn't on sys.path yet (e.g. unit-tested in isolation).
    from fast_vollib.jackel.differentiable import implied_volatility_autograd  # type: ignore
    from fast_vollib.backends.torch_backend import _price_vega_d1d2_t  # type: ignore

    if use_sentinel:
        with torch.no_grad():
            unsafe = vega_hat.detach().abs() <= vega_floor
            sentinel_price, _, _, _ = _price_vega_d1d2_t(
                is_call, F, K, tau_yr, r_rate, sigma_star, r_rate,
            )
        price_for_iv = torch.where(unsafe, sentinel_price.detach(), price_hat)
        n_unsafe = int(unsafe.sum().item())
    else:
        price_for_iv = price_hat
        n_unsafe = 0

    sigma_rt = implied_volatility_autograd(
        price_for_iv, F, K, tau_yr, r_rate, is_call, model="black",
    )
    per_row = torch.nan_to_num(
        (sigma_rt - sigma_star) ** 2, nan=0.0, posinf=0.0, neginf=0.0,
    )
    if use_gate:
        v2 = vega_hat.detach() ** 2
        w = v2 / (v2 + vega_floor ** 2)
        loss = (w * per_row).mean()
    else:
        loss = per_row.mean()
    return loss, n_unsafe


def direct_iv_aux_loss(
    vega_hat: torch.Tensor,
    sigma_hat: torch.Tensor,
    sigma_star: torch.Tensor,
    vega_floor: float,
    use_gate: bool,
    use_sentinel: bool,
) -> Tuple[torch.Tensor, int]:
    """Algebraically matched *direct* gated IV loss.

    Mirrors ``rt_aux_loss`` term by term -- the same detached-vega smooth gate
    ``w = v^2 / (v^2 + tau^2)``, the same ``nan_to_num`` scrub, and the same
    failure semantics (rows with ``|vega| <= vega_floor`` contribute zero,
    exactly as sentinel replacement forces ``sigma_rt == sigma_star`` there) --
    but skips the Jaeckel inversion entirely: the per-row error is
    ``(sigma_hat - sigma_star)^2`` instead of
    ``(J(P(sigma_hat)) - sigma_star)^2``.

    Returns ``(loss, n_unsafe)`` with the same contract as ``rt_aux_loss``.
    """
    if use_sentinel:
        with torch.no_grad():
            unsafe = vega_hat.detach().abs() <= vega_floor
        n_unsafe = int(unsafe.sum().item())
    else:
        unsafe = None
        n_unsafe = 0

    per_row = torch.nan_to_num(
        (sigma_hat - sigma_star) ** 2, nan=0.0, posinf=0.0, neginf=0.0,
    )
    if unsafe is not None:
        per_row = torch.where(unsafe, torch.zeros_like(per_row), per_row)
    if use_gate:
        v2 = vega_hat.detach() ** 2
        w = v2 / (v2 + vega_floor ** 2)
        loss = (w * per_row).mean()
    else:
        loss = per_row.mean()
    return loss, n_unsafe


def direct_rt_equiv_stats(
    price_hat: torch.Tensor,
    vega_hat: torch.Tensor,
    is_call: torch.Tensor,
    F: torch.Tensor,
    K: torch.Tensor,
    tau_yr: torch.Tensor,
    r_rate: torch.Tensor,
    sigma_hat: torch.Tensor,
    sigma_star: torch.Tensor,
    vega_floor: float,
    use_gate: bool,
    use_sentinel: bool,
) -> dict:
    """Loss/gradient agreement between the PIVOT roundtrip loss and the
    direct gated IV loss on one identical batch.

    Both losses are evaluated on the same tensors; gradients are taken w.r.t.
    ``sigma_hat`` via ``torch.autograd.grad(..., retain_graph=True)`` so the
    surrounding training graph is unaffected. Feeds the
    "max |L_direct - L_pivot| and gradient agreement on identical batches"
    reporting requirement.
    """
    rt_loss, _ = rt_aux_loss(
        price_hat, vega_hat, is_call, F, K, tau_yr, r_rate, sigma_star,
        vega_floor=vega_floor, use_gate=use_gate, use_sentinel=use_sentinel,
    )
    dir_loss, _ = direct_iv_aux_loss(
        vega_hat, sigma_hat, sigma_star,
        vega_floor=vega_floor, use_gate=use_gate, use_sentinel=use_sentinel,
    )
    (g_rt,) = torch.autograd.grad(rt_loss, sigma_hat, retain_graph=True)
    (g_dir,) = torch.autograd.grad(dir_loss, sigma_hat, retain_graph=True)
    gd = (g_rt - g_dir).abs()
    denom = g_dir.abs().clamp_min(1e-30)
    return {
        "equiv_loss_rt": float(rt_loss.detach()),
        "equiv_loss_direct": float(dir_loss.detach()),
        "equiv_loss_absdiff": float((rt_loss - dir_loss).abs().detach()),
        "equiv_grad_max_absdiff": float(gd.max().detach()),
        "equiv_grad_mean_absdiff": float(gd.mean().detach()),
        "equiv_grad_max_reldiff": float((gd / denom).max().detach()),
    }


__all__ = [
    "device", "set_seed", "is_finite_params", "maybe_scrub_nans",
    "sanitize_grads", "price_aux_loss", "rt_aux_loss",
    "direct_iv_aux_loss", "direct_rt_equiv_stats",
]
