"""Train HyperIV-style model with optional price-space and gated IV-roundtrip auxiliaries.

Variants:
    vanilla        - IV MSE + arbitrage auxiliaries (cal, g, integral).
    price_aux      - vanilla + lambda_price * scaled_MSE(price_hat, price_market).
    price_rt_aux   - price_aux + lambda_rt * mean(gate(vega_hat) * (sigma_rt - sigma*)^2)
                     with sentinel-replaced price_for_iv on low-vega rows.
    price_direct_iv_aux - price_aux + lambda_rt * mean(gate(vega_hat) * (sigma_hat - sigma*)^2)
                     (algebraically matched control, no inversion).
    unsafe_rt_control - diagnostic-only: ungated, no sentinel — for NaN-grad metric.

Run from the repository root:
    uv run python experiments/hyperiv_price_iv_aux/scripts/train.py \
        --config experiments/configs/hyperiv_price_iv_aux.toml \
        --variant price_rt_aux --seed 1 --epochs 100 --tag rt_s1 --no-wandb
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from _common import EXPT_ROOT, load_config, runs_dir, prepared_dir
from _dataset import OptionPriceIVDataset
from _model import build_model
from aux_losses import (  # type: ignore
    device as _device,
    set_seed as _set_seed,
    price_aux_loss as _price_aux_loss,
    rt_aux_loss as _rt_aux_loss,
    direct_iv_aux_loss as _direct_iv_aux_loss,
    direct_rt_equiv_stats as _direct_rt_equiv_stats,
    sanitize_grads,
    is_finite_params,
)

# Shared training primitives + W&B-instrumented diagnostics, copied from the
# GNO trainer's pattern so HyperIV runs land in the same anonymous-org/pivot-experiments
# project under their own asset/variant tags.
from experiments.utils.training import (  # type: ignore
    TrainerConfig,
    init_wandb,
    log_param_histograms,
    grad_stats,
    grouped_grad_stats,
)

# fast-vollib differentiable IV path (still imported here for the eval loop's
# direct use of _price_vega_d1d2_t outside the shared helpers).
from fast_vollib.backends.torch_backend import _price_vega_d1d2_t  # type: ignore


@dataclass
class RunConfig:
    variant: str
    lambda_price: float
    lambda_rt: float
    tau: float
    seed: int
    epochs: int
    batch_size: int
    contracts_per_interval: int
    learning_rate: float
    tag: str
    use_price_aux: bool
    use_roundtrip_aux: bool
    use_vega_gate: bool
    use_sentinel: bool
    use_direct_iv_aux: bool
    diagnostic_only: bool
    equiv_check_every: int
    train_path: str
    test_path: str


# --------- arbitrage auxiliaries (mirrors models/hyperiv/trainer_util.py) ---------

def _aux_loss_grid(model, z_batch: torch.Tensor,
                    k_start: float = -1.5, k_end: float = 0.5, k_num: int = 71,
                    t_start: float = 0.01, t_end: float = 2.0, t_num: int = 17):
    batch_size = z_batch.shape[0]
    k_samples = torch.linspace(k_start, k_end, k_num, device=z_batch.device, requires_grad=True)
    t_samples = torch.linspace(t_start, t_end, t_num, device=z_batch.device, requires_grad=True)
    kt = torch.cartesian_prod(k_samples, t_samples)
    kt = torch.tile(kt, (batch_size, 1, 1))
    pred = model(z_batch, kt)
    grad = torch.autograd.grad(pred, kt, torch.ones_like(pred), create_graph=True)[0]
    grad_k, grad_t = grad[..., 0], grad[..., 1]
    grad2 = torch.autograd.grad(grad_k, kt, grad_outputs=torch.ones_like(grad_k),
                                 create_graph=True, retain_graph=True)[0]
    grad2_kk = grad2[..., 0]
    k = kt[..., 0]
    t = kt[..., 1]
    s = pred[..., 0]
    s_k = grad_k
    s_t = grad_t
    s_kk = grad2_kk

    cal_cond = s + 2 * t * s_t
    cal_loss = torch.relu(-cal_cond).mean()

    g = (1 - k * s_k / s) ** 2 - ((0.5 * t * s * s_k) ** 2) + t * s * s_kk
    g_loss = torch.relu(-g).mean()

    d_minus = (-k - 0.5 * s ** 2 * t) / (s * t ** 0.5)
    p = g * torch.exp(-0.5 * d_minus ** 2) / (s * torch.sqrt(2 * torch.pi * torch.as_tensor(1.0, device=z_batch.device) * t))
    integral = torch.trapz(p.view(-1, k_num, t_num), k.view(-1, k_num, t_num), dim=1)
    integral_loss = ((integral - 1.0) ** 2).mean()
    return cal_loss, g_loss, integral_loss


# ---------------- training step ----------------


def _train_one_epoch(model, loader, optimizer, device, run: RunConfig,
                     mse_only: bool = False, *,
                     epoch: int = 0,
                     logger=None,
                     log_grad_every_step: int = 50) -> dict:
    model.train()
    total_iv_mse = total_cal = total_g = total_intg = 0.0
    total_price_aux = total_rt_aux = total_direct_aux = 0.0
    equiv_max: dict = {}
    total_loss = 0.0
    nan_grad_steps = 0
    nonfinite_param_steps = 0
    n_batches = 0
    step_idx = 0

    for z, X, y, side in loader:
        z = z.to(device, non_blocking=True)
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        side = side.to(device, non_blocking=True)

        optimizer.zero_grad()

        sigma_hat = model(z, X).squeeze(-1)
        iv_mse = ((sigma_hat - y) ** 2).mean()
        cal_loss, g_loss, integral_loss = _aux_loss_grid(model, z)

        # Side tensors (B, M, 7) -> flatten last two dims for the price head.
        # side columns: [F, K, tau, r, is_call, mid, vega_star]
        F = side[..., 0]
        K = side[..., 1]
        tau_t = side[..., 2]
        r = side[..., 3]
        is_call_t = (side[..., 4] > 0.5)
        mid = side[..., 5]

        loss = iv_mse + cal_loss + g_loss + integral_loss
        price_aux_val = torch.tensor(0.0, device=device)
        rt_aux_val = torch.tensor(0.0, device=device)
        direct_aux_val = torch.tensor(0.0, device=device)

        if run.use_price_aux or run.use_roundtrip_aux or run.use_direct_iv_aux:
            price_hat, vega_hat, _, _ = _price_vega_d1d2_t(
                is_call_t, F, K, tau_t, r, sigma_hat, r,
            )

            if run.use_price_aux:
                price_aux_val = _price_aux_loss(price_hat, mid)
                loss = loss + run.lambda_price * price_aux_val

            if run.use_roundtrip_aux:
                rt_aux_val, _ = _rt_aux_loss(
                    price_hat, vega_hat, is_call_t, F, K, tau_t, r, y,
                    vega_floor=run.tau, use_gate=run.use_vega_gate,
                    use_sentinel=run.use_sentinel,
                )
                loss = loss + run.lambda_rt * rt_aux_val

            if run.use_direct_iv_aux:
                direct_aux_val, _ = _direct_iv_aux_loss(
                    vega_hat, sigma_hat, y,
                    vega_floor=run.tau, use_gate=run.use_vega_gate,
                    use_sentinel=run.use_sentinel,
                )
                loss = loss + run.lambda_rt * direct_aux_val

            if (run.equiv_check_every > 0
                    and (step_idx % run.equiv_check_every == 0)
                    and (run.use_roundtrip_aux or run.use_direct_iv_aux)):
                try:
                    stats = _direct_rt_equiv_stats(
                        price_hat, vega_hat, is_call_t, F, K, tau_t, r,
                        sigma_hat, y, vega_floor=run.tau,
                        use_gate=run.use_vega_gate,
                        use_sentinel=run.use_sentinel,
                    )
                    for k in ("equiv_loss_absdiff", "equiv_grad_max_absdiff",
                              "equiv_grad_max_reldiff"):
                        equiv_max[k] = max(equiv_max.get(k, 0.0), stats[k])
                    if logger is not None:
                        logger.log({f"step/{k}": v for k, v in stats.items()})
                except Exception:
                    pass

        loss.backward()
        if sanitize_grads(model):
            nan_grad_steps += 1
        # Per-step grad/param diagnostics (HyperIV-grain, mirroring GNO).
        if logger is not None and log_grad_every_step > 0 and (step_idx % log_grad_every_step == 0):
            try:
                payload = {f"step/{k}": v for k, v in grad_stats(model).items()}
                payload.update({f"step/{k}": v for k, v in grouped_grad_stats(model).items()})
                payload["step/loss"] = float(loss.detach())
                payload["step/iv_mse"] = float(iv_mse.detach())
                payload["step/price_aux"] = float(price_aux_val.detach())
                payload["step/rt_aux"] = float(rt_aux_val.detach())
                payload["step/direct_aux"] = float(direct_aux_val.detach())
                payload["step/epoch"] = epoch
                payload["step/nan_grad_steps_epoch"] = nan_grad_steps
                logger.log(payload)
            except Exception:
                pass
        step_idx += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        if not is_finite_params(model):
            nonfinite_param_steps += 1

        total_iv_mse += float(iv_mse.detach())
        total_cal += float(cal_loss.detach())
        total_g += float(g_loss.detach())
        total_intg += float(integral_loss.detach())
        total_price_aux += float(price_aux_val.detach())
        total_rt_aux += float(rt_aux_val.detach())
        total_direct_aux += float(direct_aux_val.detach())
        total_loss += float(loss.detach())
        n_batches += 1

    return {
        "iv_mse": total_iv_mse / max(n_batches, 1),
        "cal_loss": total_cal / max(n_batches, 1),
        "g_loss": total_g / max(n_batches, 1),
        "integral_loss": total_intg / max(n_batches, 1),
        "price_aux": total_price_aux / max(n_batches, 1),
        "rt_aux": total_rt_aux / max(n_batches, 1),
        "direct_aux": total_direct_aux / max(n_batches, 1),
        "total_loss": total_loss / max(n_batches, 1),
        **equiv_max,
        "nan_grad_steps": nan_grad_steps,
        "nonfinite_param_steps": nonfinite_param_steps,
    }


@torch.no_grad()
def _eval(model, loader, device, run: RunConfig) -> dict:
    """Compute test IV/price diagnostics and per-row vega bins."""
    model.eval()
    iv_abs = []
    iv_sq = []
    pr_abs = []
    pr_sq = []
    spread_norm = []
    vega_log_bins = [-np.inf, math.log(1e-8), math.log(1e-6), math.log(1e-4), math.log(1e-2), np.inf]
    bin_iv_abs = [[] for _ in range(5)]
    bin_pr_abs = [[] for _ in range(5)]
    n_low_vega = 0
    n_total = 0

    for z, X, y, side in loader:
        z = z.to(device); X = X.to(device); y = y.to(device); side = side.to(device)
        sigma_hat = model(z, X).squeeze(-1)
        F = side[..., 0]; K = side[..., 1]; tau_t = side[..., 2]; r = side[..., 3]
        is_call_t = (side[..., 4] > 0.5); mid = side[..., 5]
        bid = mid - 0.5 * (mid.abs() * 0)  # placeholder

        # Recompute bid/offer-spread from side: we did not propagate them; use mid only.
        # (Kept for spread_normalized_price_mae we approximate spread as max(|mid|*0.01, 0.05)).
        approx_spread = torch.clamp(mid.abs() * 0.02, min=0.05)

        price_hat, vega_hat, _, _ = _price_vega_d1d2_t(
            is_call_t, F, K, tau_t, r, sigma_hat, r,
        )
        diff_iv = (sigma_hat - y).abs()
        diff_pr = (price_hat - mid).abs()

        iv_abs.append(diff_iv.detach().cpu().numpy().reshape(-1))
        iv_sq.append((diff_iv ** 2).detach().cpu().numpy().reshape(-1))
        pr_abs.append(diff_pr.detach().cpu().numpy().reshape(-1))
        pr_sq.append((diff_pr ** 2).detach().cpu().numpy().reshape(-1))
        spread_norm.append((diff_pr / approx_spread).detach().cpu().numpy().reshape(-1))

        v_abs = vega_hat.abs().clamp_min(1e-30).detach().cpu().numpy().reshape(-1)
        log_v = np.log(v_abs)
        n_low_vega += int((vega_hat.abs() <= 1e-14).sum().item())
        n_total += int(v_abs.size)
        diff_iv_np = diff_iv.detach().cpu().numpy().reshape(-1)
        diff_pr_np = diff_pr.detach().cpu().numpy().reshape(-1)
        for bi in range(5):
            mask = (log_v > vega_log_bins[bi]) & (log_v <= vega_log_bins[bi + 1])
            if mask.any():
                bin_iv_abs[bi].append(diff_iv_np[mask])
                bin_pr_abs[bi].append(diff_pr_np[mask])

    flat_iv = np.concatenate(iv_abs)
    flat_pr = np.concatenate(pr_abs)
    flat_iv_sq = np.concatenate(iv_sq)
    flat_pr_sq = np.concatenate(pr_sq)
    flat_spread = np.concatenate(spread_norm)
    metrics = {
        "test_iv_mae": float(flat_iv.mean()),
        "test_iv_rmse": float(np.sqrt(flat_iv_sq.mean())),
        "test_price_mae": float(flat_pr.mean()),
        "test_price_rmse": float(np.sqrt(flat_pr_sq.mean())),
        "spread_normalized_price_mae": float(flat_spread.mean()),
        "fraction_low_vega_1e14": float(n_low_vega / max(n_total, 1)),
    }
    bin_metrics = []
    bin_labels = ["leq_1e-8", "1e-8_to_1e-6", "1e-6_to_1e-4", "1e-4_to_1e-2", "gt_1e-2"]
    for bi, label in enumerate(bin_labels):
        if bin_iv_abs[bi]:
            iv_arr = np.concatenate(bin_iv_abs[bi])
            pr_arr = np.concatenate(bin_pr_abs[bi])
            bin_metrics.append({
                "bin": label,
                "n": int(iv_arr.size),
                "iv_mae": float(iv_arr.mean()),
                "price_mae": float(pr_arr.mean()),
            })
        else:
            bin_metrics.append({"bin": label, "n": 0, "iv_mae": None, "price_mae": None})
    metrics["vega_bins"] = bin_metrics
    return metrics


def _make_loader(df, run: RunConfig, sample: bool, batch_size: int) -> DataLoader:
    ds = OptionPriceIVDataset(df, n_samples=run.contracts_per_interval, sample=sample, seed=run.seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=sample, num_workers=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", required=True,
                        choices=["vanilla", "price_aux", "price_rt_aux",
                                 "price_direct_iv_aux", "unsafe_rt_control"])
    parser.add_argument("--lambda-price", type=float, default=0.1)
    parser.add_argument("--lambda-rt", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=1e-6, help="vega gate threshold")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--data-tag", type=str, default="spx_1day")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--contracts-per-interval", type=int, default=None)
    parser.add_argument("--equiv-check-every", type=int, default=0,
                        help="If >0, every N steps log direct-vs-roundtrip "
                             "loss/gradient agreement on the current batch.")
    parser.add_argument("--wandb-mode", type=str, default=None,
                        choices=["online", "offline", "disabled"],
                        help="Override wandb mode (default: cfg.wandb.mode or 'disabled').")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="Override wandb project (default: pivot-anon).")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Shortcut for --wandb-mode disabled.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    training = cfg["training"]
    variants = cfg["variants"][args.variant]
    epochs = args.epochs if args.epochs is not None else int(training["epochs"])
    batch_size = args.batch_size if args.batch_size is not None else int(training["batch_size"])
    n_contracts = args.contracts_per_interval if args.contracts_per_interval is not None else int(training["contracts_per_interval"])

    prep_dir = prepared_dir(cfg)
    train_path = prep_dir / f"{args.data_tag}_train.parquet"
    test_path = prep_dir / f"{args.data_tag}_test.parquet"
    if not train_path.exists() or not test_path.exists():
        raise SystemExit(f"Missing prepared data at {prep_dir} (looked for {train_path.name}, {test_path.name})")

    run = RunConfig(
        variant=args.variant,
        lambda_price=float(args.lambda_price if variants.get("use_price_aux", False) else 0.0),
        lambda_rt=float(args.lambda_rt if (variants.get("use_roundtrip_aux", False)
                        or variants.get("use_direct_iv_aux", False)) else 0.0),
        tau=float(args.tau),
        seed=int(args.seed),
        epochs=int(epochs),
        batch_size=int(batch_size),
        contracts_per_interval=int(n_contracts),
        learning_rate=float(training["learning_rate"]),
        tag=str(args.tag),
        use_price_aux=bool(variants.get("use_price_aux", False)),
        use_roundtrip_aux=bool(variants.get("use_roundtrip_aux", False)),
        use_vega_gate=bool(variants.get("use_vega_gate", False)),
        use_sentinel=bool(variants.get("sentinel_replacement", False)),
        use_direct_iv_aux=bool(variants.get("use_direct_iv_aux", False)),
        diagnostic_only=bool(variants.get("diagnostic_only", False)),
        equiv_check_every=int(args.equiv_check_every),
        train_path=str(train_path),
        test_path=str(test_path),
    )

    _set_seed(run.seed)
    device = _device()
    print(f"device={device} variant={run.variant} tag={run.tag} epochs={run.epochs}")

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    print(f"train rows={len(df_train)} dates={df_train['date'].nunique()}")
    print(f"test  rows={len(df_test)} dates={df_test['date'].nunique()}")

    train_loader = _make_loader(df_train, run, sample=True, batch_size=run.batch_size)
    test_loader = _make_loader(df_test, run, sample=False, batch_size=1)

    model, _ = build_model()
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"hypernetwork params: {n_params}, base mlp params: {model.base_param_count}")

    optimizer = torch.optim.Adam(model.parameters(), lr=run.learning_rate)
    # Cosine LR schedule helps stop late-epoch divergence on this very small h_omega.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(run.epochs, 1), eta_min=run.learning_rate * 0.05
    )

    out_dir = runs_dir(cfg) / run.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- W&B init (shared anonymous-org/pivot-experiments project) ----------
    wandb_cfg_block = cfg.get("wandb", {}) or {}
    wandb_mode = (
        "disabled" if args.no_wandb
        else (args.wandb_mode or wandb_cfg_block.get("mode", "disabled"))
    )
    log_grad_every_step = int(wandb_cfg_block.get("log_grad_every_step", 50))
    log_param_hist_every = int(wandb_cfg_block.get("log_param_hist_every_epoch", 25))
    trainer_cfg = TrainerConfig(
        epochs=run.epochs,
        lr=run.learning_rate,
        weight_decay=0.0,
        grad_clip=10.0,
        nan_to_num_grad=True,
        variant=run.variant,
        run_name=run.tag,
        seed=run.seed,
        extra_tags=("hyperiv", args.data_tag),
        wandb_entity=wandb_cfg_block.get("entity"),
        wandb_project=str(args.wandb_project or wandb_cfg_block.get("project", "pivot-anon")),
        wandb_mode=wandb_mode,
        log_grad_every_step=log_grad_every_step,
        log_param_hist_every_epoch=log_param_hist_every,
    )
    extra_wandb = {
        "lambda_price": run.lambda_price,
        "lambda_rt": run.lambda_rt,
        "tau": run.tau,
        "data_tag": args.data_tag,
        "use_price_aux": run.use_price_aux,
        "use_roundtrip_aux": run.use_roundtrip_aux,
        "use_vega_gate": run.use_vega_gate,
        "use_sentinel": run.use_sentinel,
        "use_direct_iv_aux": run.use_direct_iv_aux,
        "equiv_check_every": run.equiv_check_every,
        "diagnostic_only": run.diagnostic_only,
    }
    wb_run = init_wandb(trainer_cfg, model, extra_config=extra_wandb)

    history: list[dict] = []
    nan_grad_total = 0
    nonfinite_total = 0
    t_train = time.time()
    best_train_iv_mse = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    for ep in range(run.epochs):
        ep_start = time.time()
        ep_metrics = _train_one_epoch(
            model, train_loader, optimizer, device, run,
            epoch=ep, logger=wb_run, log_grad_every_step=log_grad_every_step,
        )
        scheduler.step()
        nan_grad_total += ep_metrics["nan_grad_steps"]
        nonfinite_total += ep_metrics["nonfinite_param_steps"]
        ep_metrics["epoch"] = ep + 1
        ep_metrics["epoch_seconds"] = round(time.time() - ep_start, 2)
        ep_metrics["lr"] = float(optimizer.param_groups[0]["lr"])
        history.append(ep_metrics)
        if wb_run is not None:
            wb_run.log(
                {f"train/{k}": v for k, v in ep_metrics.items()
                 if isinstance(v, (int, float))}
                | {"epoch": ep + 1, "lr": ep_metrics["lr"],
                   "nan_grad_total": nan_grad_total,
                   "nonfinite_param_total": nonfinite_total}
            )
            if log_param_hist_every > 0 and (ep % max(log_param_hist_every, 1) == 0):
                log_param_histograms(wb_run, model)
        # Select by training IV MSE, never by the held-out test split.
        # Keep the five-epoch warmup for full runs; short smoke runs must
        # save trained weights rather than restoring the initialization.
        if ep_metrics["iv_mse"] < best_train_iv_mse and ep >= min(5, run.epochs - 1):
            best_train_iv_mse = ep_metrics["iv_mse"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = ep + 1
        if (ep + 1) % max(1, run.epochs // 10) == 0 or ep == 0:
            print(
                f"epoch {ep+1:>3}/{run.epochs}  iv_mse={ep_metrics['iv_mse']:.5f}"
                f"  cal={ep_metrics['cal_loss']:.4f}  g={ep_metrics['g_loss']:.4f}"
                f"  intg={ep_metrics['integral_loss']:.4f}"
                f"  pr_aux={ep_metrics['price_aux']:.4f}  rt_aux={ep_metrics['rt_aux']:.4f}"
                f"  dir_aux={ep_metrics['direct_aux']:.4f}"
                f"  nan_grad={nan_grad_total}  lr={ep_metrics['lr']:.2e}"
                f"  ({ep_metrics['epoch_seconds']:.1f}s)"
            )
    train_wall = time.time() - t_train

    # Restore best-train weights for evaluation.
    model.load_state_dict(best_state)
    print(f"restoring best-train weights from epoch {best_epoch} (train iv_mse={best_train_iv_mse:.5f})")
    ckpt_path = out_dir / "model.pt"
    torch.save({"state_dict": best_state, "run_config": asdict(run), "n_params": n_params,
                "best_epoch": best_epoch, "best_train_iv_mse": best_train_iv_mse}, ckpt_path)
    pd.DataFrame(history).to_csv(out_dir / "epoch_history.csv", index=False)

    # Eval
    t_eval = time.time()
    test_metrics = _eval(model, test_loader, device, run)
    eval_wall = time.time() - t_eval

    # Persist run summary.
    summary = {
        **asdict(run),
        **test_metrics,
        "train_wall_seconds": round(train_wall, 1),
        "test_wall_seconds": round(eval_wall, 1),
        "nan_grad_steps_total": nan_grad_total,
        "nonfinite_parameter_steps": nonfinite_total,
        "final_train_iv_mse": history[-1]["iv_mse"],
        "best_train_iv_mse": best_train_iv_mse,
        "best_epoch": best_epoch,
        "final_train_cal_loss": history[-1]["cal_loss"],
        "final_train_g_loss": history[-1]["g_loss"],
        "final_train_integral_loss": history[-1]["integral_loss"],
        "final_train_price_aux": history[-1]["price_aux"],
        "final_train_rt_aux": history[-1]["rt_aux"],
        "final_train_direct_aux": history[-1]["direct_aux"],
        "equiv_loss_absdiff_max": max((h.get("equiv_loss_absdiff", 0.0) for h in history), default=0.0),
        "equiv_grad_max_absdiff_max": max((h.get("equiv_grad_max_absdiff", 0.0) for h in history), default=0.0),
        "equiv_grad_max_reldiff_max": max((h.get("equiv_grad_max_reldiff", 0.0) for h in history), default=0.0),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out_dir / 'summary.json'}")
    print(json.dumps({k: v for k, v in test_metrics.items() if k != "vega_bins"}, indent=2))

    if wb_run is not None:
        wb_run.log({f"test/{k}": v for k, v in test_metrics.items()
                    if isinstance(v, (int, float))})
        wb_run.summary.update({k: v for k, v in summary.items()
                              if isinstance(v, (int, float, str))})
        wb_run.finish()


if __name__ == "__main__":
    main()
