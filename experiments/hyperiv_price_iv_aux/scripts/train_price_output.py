"""Price-output model where an IV-space objective requires inversion.

The model predicts option PRICES directly (softplus-normalized by the forward
F), so an IV-space supervision term cannot be evaluated without mapping
predicted prices back to implied volatility.

Variants (identical architecture, data, seeds, gate semantics, lambda):
    price_only     - P0: scaled price MSE only (control).
    pivot_iv       - P1: price MSE + lambda_iv * gated IV MSE through PIVOT
                     (Jaeckel forward, implicit backward; invalid rows -> NaN,
                     masked and counted).
    unrolled_iv    - P2: same IV term through a fixed-iteration Newton solver
                     differentiated through its iterations.
    generic_ift_iv - P3: same IV term through a plain Newton forward with a
                     generic implicit-function backward (autograd d P/d sigma).
    jackel_ift_iv  - P4: CONTROL that isolates the backward mechanism. Forward
                     is the *same trusted Jaeckel root* PIVOT uses; only the
                     backward differs -- a generic implicit-function inversion
                     of an autograd-obtained dP/dsigma with a plain clamp, with
                     no invalid-domain mask and no conditioning contract.
                     P1 vs P4 therefore attributes any difference to PIVOT's
                     backward contract rather than to forward-solver quality
                     (P3 conflates the two).

The gate mirrors aux_losses.rt_aux_loss: w = v^2/(v^2+tau^2) with v the
(detached) vega at the inverted sigma; non-finite inversions contribute zero
and are counted. Evaluation inverts predicted prices with float64 Jaeckel
(no-grad) for IV metrics regardless of the training variant.

Run:
    python train_price_output.py --config experiments/configs/hyperiv_price_iv_aux.toml \
        --variant pivot_iv --lambda-iv 0.1 --tau 1e-6 --seed 1 --epochs 100 \
        --tag pivot_liv0.1_tau1e-6_s1 --data-tag spx_1day
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from _common import EXPT_ROOT, load_config, runs_dir, prepared_dir  # noqa: F401
from _dataset import OptionPriceIVDataset
from _model import build_model
from aux_losses import (  # type: ignore
    device as _device,
    set_seed as _set_seed,
    price_aux_loss as _price_aux_loss,
    sanitize_grads,
    is_finite_params,
)
from experiments.utils.training import (  # type: ignore
    TrainerConfig,
    init_wandb,
    grad_stats,
)
from fast_vollib.backends.torch_backend import _price_vega_d1d2_t  # type: ignore
from fast_vollib.jackel.differentiable import implied_volatility_autograd  # type: ignore

NEWTON_ITERS = 20


# ----------------------------- inversion arms -----------------------------

def _newton_body(price, F, K, t, r, is_call, differentiable: bool):
    sigma = torch.full_like(price, 0.2)
    for _ in range(NEWTON_ITERS):
        p, vega, _, _ = _price_vega_d1d2_t(is_call, F, K, t, r, sigma, r)
        step = (price - p) / vega.clamp_min(1e-12)
        sigma = (sigma + step).clamp(1e-4, 5.0)
        if not differentiable:
            sigma = sigma.detach()
    return sigma


class _GenericIFT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, price, F, K, t, r, is_call):
        with torch.no_grad():
            sigma = _newton_body(price, F, K, t, r, is_call, differentiable=False)
        ctx.save_for_backward(sigma, F, K, t, r, is_call)
        return sigma

    @staticmethod
    def backward(ctx, grad_out):
        sigma, F, K, t, r, is_call = ctx.saved_tensors
        sigma_v = sigma.detach().requires_grad_(True)
        with torch.enable_grad():
            p, _, _, _ = _price_vega_d1d2_t(is_call, F, K, t, r, sigma_v, r)
            dp = torch.autograd.grad(p.sum(), sigma_v)[0]
        return grad_out / dp.clamp_min(1e-14), None, None, None, None, None


class _JackelGenericIFT(torch.autograd.Function):
    """Trusted Jaeckel forward + GENERIC implicit-function backward.

    Forward: the converged Jaeckel root (identical to PIVOT's forward, incl.
    its NaN-on-invalid output). Backward: dP/dsigma obtained by autograd on the
    pricing map at that root and inverted with a plain clamp -- i.e. what a
    generic implicit-diff wrapper (jaxopt/theseus/DEQ style) yields when handed
    a good solver but no domain knowledge. No invalid mask is propagated and no
    low-vega conditioning contract is applied inside the layer.
    """

    @staticmethod
    def forward(ctx, price, F, K, t, r, is_call):
        with torch.no_grad():
            sigma = implied_volatility_autograd(
                price, F, K, t, r, is_call, model="black")
        ctx.save_for_backward(sigma, F, K, t, r, is_call)
        return sigma

    @staticmethod
    def backward(ctx, grad_out):
        sigma, F, K, t, r, is_call = ctx.saved_tensors
        sigma_v = sigma.detach().requires_grad_(True)
        with torch.enable_grad():
            p, _, _, _ = _price_vega_d1d2_t(is_call, F, K, t, r, sigma_v, r)
            dp = torch.autograd.grad(p.sum(), sigma_v)[0]
        return grad_out / dp.clamp_min(1e-14), None, None, None, None, None


def invert(variant, price, F, K, t, r, is_call):
    if variant == "pivot_iv":
        return implied_volatility_autograd(price, F, K, t, r, is_call, model="black")
    if variant == "unrolled_iv":
        return _newton_body(price, F, K, t, r, is_call, differentiable=True)
    if variant == "generic_ift_iv":
        return _GenericIFT.apply(price, F, K, t, r, is_call)
    if variant == "jackel_ift_iv":
        return _JackelGenericIFT.apply(price, F, K, t, r, is_call)
    raise ValueError(variant)


# ----------------------------- config -----------------------------

@dataclass
class RunConfig:
    variant: str
    lambda_iv: float
    tau: float
    seed: int
    epochs: int
    batch_size: int
    contracts_per_interval: int
    learning_rate: float
    tag: str
    train_path: str
    test_path: str


def gated_iv_loss(sigma_inv, sigma_star, is_call, F, K, t, r, tau):
    """Matched gate/failure semantics across all inversion arms."""
    finite = torch.isfinite(sigma_inv)
    sigma_safe = torch.where(finite, sigma_inv, torch.full_like(sigma_inv, 0.2))
    with torch.no_grad():
        _, vega, _, _ = _price_vega_d1d2_t(is_call, F, K, t, r, sigma_safe.detach(), r)
        v2 = vega ** 2
        w = v2 / (v2 + tau ** 2)
        w = torch.where(finite, w, torch.zeros_like(w))
    per_row = torch.nan_to_num((sigma_safe - sigma_star) ** 2,
                               nan=0.0, posinf=0.0, neginf=0.0)
    n_invalid = int((~finite).sum().item())
    return (w * per_row).mean(), n_invalid


# ----------------------------- train / eval -----------------------------

def train_one_epoch(model, loader, optimizer, device, run, epoch, logger):
    model.train()
    tot = {"price_loss": 0.0, "iv_loss": 0.0, "total": 0.0}
    nan_grad = nonfinite = n_batches = 0
    invalid_rows = 0
    step_idx = 0
    for z, X, y, side in loader:
        z, X, y, side = (t.to(device, non_blocking=True) for t in (z, X, y, side))
        F = side[..., 0]; K = side[..., 1]; tau_t = side[..., 2]; r = side[..., 3]
        is_call = side[..., 4] > 0.5; mid = side[..., 5]
        optimizer.zero_grad()
        raw = model(z, X).squeeze(-1)
        # Base network ends in Softplus (raw > 0). Linear 0.1*F scale keeps the
        # initial prediction near typical OTM mids and gradients unsaturated
        # (v1's softplus(raw)*F double-softplus pinned the model at ~0.69*F).
        price_hat = raw * 0.1 * F
        loss = _price_aux_loss(price_hat, mid)
        iv_val = torch.tensor(0.0, device=device)
        if run.variant != "price_only":
            sigma_inv = invert(run.variant, price_hat, F, K, tau_t, r, is_call)
            iv_val, n_inv = gated_iv_loss(sigma_inv, y, is_call, F, K, tau_t, r, run.tau)
            invalid_rows += n_inv
            loss = loss + run.lambda_iv * iv_val
        p_loss = loss - run.lambda_iv * iv_val if run.variant != "price_only" else loss
        loss.backward()
        if sanitize_grads(model):
            nan_grad += 1
        if logger is not None and step_idx % 100 == 0:
            try:
                logger.log({"step/loss": float(loss.detach()),
                            "step/iv_loss": float(iv_val.detach()),
                            "step/epoch": epoch,
                            **{f"step/{k}": v for k, v in grad_stats(model).items()}})
            except Exception:
                pass
        step_idx += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        if not is_finite_params(model):
            nonfinite += 1
        tot["price_loss"] += float(p_loss.detach())
        tot["iv_loss"] += float(iv_val.detach())
        tot["total"] += float(loss.detach())
        n_batches += 1
    out = {k: v / max(n_batches, 1) for k, v in tot.items()}
    out.update({"nan_grad_steps": nan_grad, "nonfinite_parameter_steps": nonfinite,
                "invalid_inversion_rows": invalid_rows})
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    pr_abs, pr_sq, iv_abs, iv_sq = [], [], [], []
    n_invalid = n_total = 0
    for z, X, y, side in loader:
        z, X, y, side = (t.to(device) for t in (z, X, y, side))
        F = side[..., 0]; K = side[..., 1]; tau_t = side[..., 2]; r = side[..., 3]
        is_call = side[..., 4] > 0.5; mid = side[..., 5]
        raw = model(z, X).squeeze(-1)
        price_hat = raw * 0.1 * F
        d_pr = (price_hat - mid).abs()
        pr_abs.append(d_pr.cpu().numpy().reshape(-1))
        pr_sq.append((d_pr ** 2).cpu().numpy().reshape(-1))
        # IV metrics via trusted float64 Jaeckel inversion of predicted prices
        sig = implied_volatility_autograd(
            price_hat.double(), F.double(), K.double(), tau_t.double(),
            r.double(), is_call, model="black")
        finite = torch.isfinite(sig)
        n_invalid += int((~finite).sum().item()); n_total += int(sig.numel())
        d_iv = (sig[finite] - y.double()[finite]).abs()
        iv_abs.append(d_iv.cpu().numpy().reshape(-1))
        iv_sq.append((d_iv ** 2).cpu().numpy().reshape(-1))
    fp, fps = np.concatenate(pr_abs), np.concatenate(pr_sq)
    fi, fis = np.concatenate(iv_abs), np.concatenate(iv_sq)
    return {
        "test_price_mae": float(fp.mean()),
        "test_price_rmse": float(np.sqrt(fps.mean())),
        "test_iv_mae": float(fi.mean()),
        "test_iv_rmse": float(np.sqrt(fis.mean())),
        "test_invalid_price_fraction": float(n_invalid / max(n_total, 1)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", required=True,
                    choices=["price_only", "pivot_iv", "unrolled_iv",
                             "generic_ift_iv", "jackel_ift_iv"])
    ap.add_argument("--lambda-iv", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data-tag", default="spx_1day")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--contracts-per-interval", type=int, default=None)
    ap.add_argument("--wandb-mode", default="disabled",
                    choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb-project", default="pivot-anon")
    args = ap.parse_args()

    cfg = load_config(args.config)
    training = cfg["training"]
    epochs = args.epochs if args.epochs is not None else int(training["epochs"])
    batch_size = args.batch_size or int(training["batch_size"])
    n_contracts = args.contracts_per_interval or int(training["contracts_per_interval"])
    prep = prepared_dir(cfg)
    train_path = prep / f"{args.data_tag}_train.parquet"
    test_path = prep / f"{args.data_tag}_test.parquet"
    run = RunConfig(args.variant, float(args.lambda_iv), float(args.tau),
                    int(args.seed), int(epochs), int(batch_size), int(n_contracts),
                    float(training["learning_rate"]), args.tag,
                    str(train_path), str(test_path))
    _set_seed(run.seed)
    device = _device()
    print(f"device={device} variant={run.variant} tag={run.tag} epochs={run.epochs}")

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    tl = DataLoader(OptionPriceIVDataset(df_train, n_samples=run.contracts_per_interval,
                                         sample=True, seed=run.seed),
                    batch_size=run.batch_size, shuffle=True, num_workers=0)
    el = DataLoader(OptionPriceIVDataset(df_test, n_samples=run.contracts_per_interval,
                                         sample=False, seed=run.seed),
                    batch_size=1, shuffle=False, num_workers=0)

    model, _ = build_model()
    _set_seed(run.seed)  # build_model resets global RNG; re-seed so per-seed
    # stochasticity (shuffling/sampling) actually differs across seeds.
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=run.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(run.epochs, 1), eta_min=run.learning_rate * 0.05)

    out_dir = runs_dir(cfg) / run.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer_cfg = TrainerConfig(
        epochs=run.epochs, lr=run.learning_rate, weight_decay=0.0, grad_clip=10.0,
        nan_to_num_grad=True, variant=f"priceout_{run.variant}", run_name=run.tag,
        seed=run.seed, extra_tags=("priceout", args.data_tag),
        wandb_entity=None, wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode, log_grad_every_step=100,
        log_param_hist_every_epoch=0,
    )
    wb = init_wandb(trainer_cfg, model,
                    extra_config={"lambda_iv": run.lambda_iv, "tau": run.tau})

    history = []
    nan_total = nonfinite_total = invalid_total = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    best = float("inf"); best_state = None; best_epoch = 0
    for ep in range(run.epochs):
        ep_t = time.time()
        m = train_one_epoch(model, tl, optimizer, device, run, ep, wb)
        scheduler.step()
        nan_total += m["nan_grad_steps"]; nonfinite_total += m["nonfinite_parameter_steps"]
        invalid_total += m["invalid_inversion_rows"]
        m.update({"epoch": ep + 1, "epoch_seconds": round(time.time() - ep_t, 2)})
        history.append(m)
        if wb is not None:
            wb.log({f"train/{k}": v for k, v in m.items() if isinstance(v, (int, float))})
        if m["total"] < best and ep >= 5:
            best = m["total"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = ep + 1
        if (ep + 1) % max(1, run.epochs // 10) == 0 or ep == 0:
            print(f"epoch {ep+1:>3}/{run.epochs} total={m['total']:.5f} "
                  f"price={m['price_loss']:.5f} iv={m['iv_loss']:.5f} "
                  f"invalid_rows={m['invalid_inversion_rows']} "
                  f"nan_grad={nan_total} ({m['epoch_seconds']:.1f}s)")
    wall = time.time() - t0
    peak_mem = (torch.cuda.max_memory_allocated() / 2**20
                if device.type == "cuda" else float("nan"))
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "run_config": asdict(run),
                "best_epoch": best_epoch}, out_dir / "model.pt")
    pd.DataFrame(history).to_csv(out_dir / "epoch_history.csv", index=False)

    t1 = time.time()
    test = evaluate(model, el, device)
    summary = {**asdict(run), **test,
               "train_wall_seconds": round(wall, 1),
               "test_wall_seconds": round(time.time() - t1, 1),
               "peak_gpu_mem_mb": round(peak_mem, 1),
               "nan_grad_steps_total": nan_total,
               "nonfinite_parameter_steps": nonfinite_total,
               "train_invalid_inversion_rows_total": invalid_total,
               "best_epoch": best_epoch}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(test, indent=2))
    if wb is not None:
        wb.log({f"test/{k}": v for k, v in test.items()})
        wb.summary.update({k: v for k, v in summary.items()
                           if isinstance(v, (int, float, str))})
        wb.finish()


if __name__ == "__main__":
    main()
