"""Matched forward+backward GPU systems benchmark for differentiable IV inversion.

Compare under identical
batches, dtype, masks, and timing methodology:

  pivot        : Jäckel/LBR forward (trusted, branch-heavy) + custom implicit
                 backward 1/vega with invalid-domain and low-vega contract
                 (fast_vollib.jackel.differentiable.implied_volatility_autograd).
  unrolled     : fixed-iteration Newton solver, differentiated THROUGH the
                 iterations by autograd (the standard "just unroll it" baseline).
  generic_ift  : plain Newton forward under no_grad + generic implicit-function
                 backward where dP/dsigma is obtained by autograd on the pricing
                 map (not hand-derived) — the "generic implicit-root" baseline.
  jackel_ift   : CONTROL isolating the backward mechanism. Same trusted Jaeckel
                 forward as pivot; only the backward differs (generic autograd
                 dP/dsigma + plain clamp, no invalid mask, no conditioning
                 contract). pivot vs jackel_ift attributes differences to the
                 backward contract; generic_ift conflates forward quality with
                 backward mechanism.
  finite_diff  : central finite-difference gradient (small batches only;
                 validation, not a throughput competitor).

Reported per (method, batch, dtype): forward ms, forward+backward ms (median of
--repeats, synchronized, after warmup), peak GPU memory, forward IV error vs a
float64 Jäckel reference, gradient error vs the analytic implicit derivative
1/vega on well-conditioned rows, and non-finite/failure counts on invalid and
low-vega rows.

Batch composition mirrors the plan: mostly well-conditioned rows plus explicit
low-vega (deep-OTM short-dated) and invalid (below-intrinsic) slices.

Usage:
    python bench_differentiable_iv.py [--smoke] [--device cuda] \
        [--batches 1000 100000 1000000 10000000] [--repeats 7] \
        [--out benchmark.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# The artifact vendors the exact operator source under the repository's src/.
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from fast_vollib.backends.torch_backend import _price_vega_d1d2_t  # noqa: E402
from fast_vollib.jackel.differentiable import implied_volatility_autograd  # noqa: E402

LOW_VEGA_THRESHOLD = 1e-14
NEWTON_ITERS = 20


# --------------------------------------------------------------------------
# batch construction
# --------------------------------------------------------------------------

def make_batch(n: int, device, dtype, seed: int = 0,
               frac_low_vega: float = 0.02, frac_invalid: float = 0.01):
    g = torch.Generator(device="cpu").manual_seed(seed)
    F = torch.full((n,), 100.0, dtype=dtype)
    k = torch.exp(torch.randn(n, generator=g, dtype=dtype) * 0.25) * 100.0
    t = torch.rand(n, generator=g, dtype=dtype) * 1.9 + 0.02
    r = torch.full((n,), 0.02, dtype=dtype)
    sigma = torch.rand(n, generator=g, dtype=dtype) * 0.45 + 0.05
    n_lv = int(n * frac_low_vega)
    n_inv = int(n * frac_invalid)
    # low-vega slice: deep OTM, short-dated, low vol
    if n_lv:
        k[:n_lv] = F[:n_lv] * 20.0
        t[:n_lv] = 0.02
        sigma[:n_lv] = 0.05
    is_call = k >= F
    price, vega = bs_price_vega(is_call, F, k, t, r, sigma)
    # invalid slice: below-intrinsic / non-positive prices
    if n_inv:
        price[n_lv:n_lv + n_inv] = -1.0
    return (x.to(device) for x in (price, F, k, t, r, sigma, is_call))


def bs_price_vega(is_call, F, k, t, r, sigma):
    sqrt_t = torch.sqrt(t)
    d1 = (torch.log(F / k) + 0.5 * sigma**2 * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    disc = torch.exp(-r * t)
    ncdf = lambda x: 0.5 * (1 + torch.erf(x / 2**0.5))
    npdf = lambda x: torch.exp(-0.5 * x**2) / (2 * torch.pi) ** 0.5
    call = disc * (F * ncdf(d1) - k * ncdf(d2))
    put = disc * (k * ncdf(-d2) - F * ncdf(-d1))
    return torch.where(is_call, call, put), disc * F * npdf(d1) * sqrt_t


# --------------------------------------------------------------------------
# methods under test
# --------------------------------------------------------------------------

def iv_pivot(price, F, k, t, r, is_call):
    return implied_volatility_autograd(price, F, k, t, r, is_call, model="black")


def _newton_body(price, F, k, t, r, is_call, differentiable: bool):
    sigma = torch.full_like(price, 0.2)
    for _ in range(NEWTON_ITERS):
        p, vega, _, _ = _price_vega_d1d2_t(is_call, F, k, t, r, sigma, r)
        step = (price - p) / vega.clamp_min(1e-12)
        sigma = (sigma + step).clamp(1e-4, 5.0)
        if not differentiable:
            sigma = sigma.detach()
    return sigma


def iv_unrolled(price, F, k, t, r, is_call):
    return _newton_body(price, F, k, t, r, is_call, differentiable=True)


class _GenericIFT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, price, F, k, t, r, is_call):
        with torch.no_grad():
            sigma = _newton_body(price, F, k, t, r, is_call, differentiable=False)
        ctx.save_for_backward(sigma, F, k, t, r, is_call)
        return sigma

    @staticmethod
    def backward(ctx, grad_out):
        sigma, F, k, t, r, is_call = ctx.saved_tensors
        sigma_v = sigma.detach().requires_grad_(True)
        with torch.enable_grad():
            p, _, _, _ = _price_vega_d1d2_t(is_call, F, k, t, r, sigma_v, r)
            dp_dsigma = torch.autograd.grad(p.sum(), sigma_v)[0]
        grad_price = grad_out / dp_dsigma.clamp_min(LOW_VEGA_THRESHOLD)
        return grad_price, None, None, None, None, None


def iv_generic_ift(price, F, k, t, r, is_call):
    return _GenericIFT.apply(price, F, k, t, r, is_call)


class _JackelGenericIFT(torch.autograd.Function):
    """Trusted Jaeckel forward + generic implicit-function backward."""

    @staticmethod
    def forward(ctx, price, F, k, t, r, is_call):
        with torch.no_grad():
            sigma = implied_volatility_autograd(
                price, F, k, t, r, is_call, model="black")
        ctx.save_for_backward(sigma, F, k, t, r, is_call)
        return sigma

    @staticmethod
    def backward(ctx, grad_out):
        sigma, F, k, t, r, is_call = ctx.saved_tensors
        sigma_v = sigma.detach().requires_grad_(True)
        with torch.enable_grad():
            p, _, _, _ = _price_vega_d1d2_t(is_call, F, k, t, r, sigma_v, r)
            dp = torch.autograd.grad(p.sum(), sigma_v)[0]
        return grad_out / dp.clamp_min(LOW_VEGA_THRESHOLD), None, None, None, None, None


def iv_jackel_ift(price, F, k, t, r, is_call):
    return _JackelGenericIFT.apply(price, F, k, t, r, is_call)


METHODS = {
    "pivot": iv_pivot,
    "jackel_ift": iv_jackel_ift,
    "unrolled": iv_unrolled,
    "generic_ift": iv_generic_ift,
}


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench_method(name, fn, batch, device, repeats):
    price, F, k, t, r, sigma_true, is_call = batch
    fwd_ms, fb_ms = [], []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    out = None
    for i in range(repeats + 2):  # 2 warmup
        p_in = price.clone().requires_grad_(True)
        _sync(device); t0 = time.perf_counter()
        out = fn(p_in, F, k, t, r, is_call)
        _sync(device); t1 = time.perf_counter()
        finite = torch.isfinite(out)
        out.masked_fill(~finite, 0.0).sum().backward()
        _sync(device); t2 = time.perf_counter()
        if i >= 2:
            fwd_ms.append((t1 - t0) * 1e3)
            fb_ms.append((t2 - t0) * 1e3)
        grad = p_in.grad
    med = lambda xs: sorted(xs)[len(xs) // 2]
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
    return out.detach(), grad.detach(), med(fwd_ms), med(fb_ms), peak_mb


def accuracy_row(out, grad, batch):
    price, F, k, t, r, sigma_true, is_call = batch
    # reference: float64 PIVOT forward (Jäckel) as ground truth; analytic 1/vega
    with torch.no_grad():
        p64, F64, k64, t64, r64, s64 = (x.double() for x in (price, F, k, t, r, sigma_true))
        ref = implied_volatility_autograd(p64, F64, k64, t64, r64, is_call, model="black")
        _, vega64 = bs_price_vega(is_call, F64, k64, t64, r64, s64)
        valid_ref = torch.isfinite(ref)
        well = valid_ref & (vega64 > 1e-4)
        finite_out = torch.isfinite(out)
        both = well & finite_out
        iv_err = (out.double()[both] - ref[both]).abs()
        grad_err = (grad.double()[both] - 1.0 / vega64[both]).abs() * vega64[both]  # relative
        low_vega = valid_ref & (vega64 <= 1e-4)
        return {
            "valid_rows": int(valid_ref.sum()),
            "finite_outputs_on_valid": int((finite_out & valid_ref).sum()),
            "nonfinite_on_valid": int((~finite_out & valid_ref).sum()),
            "invalid_rows_flagged_nonfinite": int((~finite_out & ~valid_ref).sum()),
            "iv_max_abs_err": float(iv_err.max()) if both.any() else None,
            "iv_mean_abs_err": float(iv_err.mean()) if both.any() else None,
            "grad_max_rel_err": float(grad_err.max()) if both.any() else None,
            "grad_mean_rel_err": float(grad_err.mean()) if both.any() else None,
            "low_vega_rows": int(low_vega.sum()),
            "low_vega_nonfinite_grad": int((~torch.isfinite(grad) & low_vega).sum()),
        }


def finite_diff_check(batch, device, n_check=2048, h_rel=1e-6):
    price, F, k, t, r, sigma_true, is_call = batch
    n = min(n_check, price.shape[0])
    sl = slice(0, n)
    p = price[sl].double(); Fd, kd, td, rd = (x[sl].double() for x in (F, k, t, r))
    ic = is_call[sl]
    with torch.no_grad():
        h = (p.abs() * h_rel).clamp_min(1e-10)
        up = implied_volatility_autograd(p + h, Fd, kd, td, rd, ic, model="black")
        dn = implied_volatility_autograd(p - h, Fd, kd, td, rd, ic, model="black")
        fd = (up - dn) / (2 * h)
        _, vega = bs_price_vega(ic, Fd, kd, td, rd, sigma_true[sl].double())
        ok = torch.isfinite(fd) & (vega > 1e-4)
        rel = ((fd[ok] - 1.0 / vega[ok]).abs() * vega[ok])
    return {"n": int(ok.sum()), "fd_vs_analytic_max_rel": float(rel.max()),
            "fd_vs_analytic_mean_rel": float(rel.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtypes", nargs="+", default=["float32", "float64"])
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[1_000, 100_000, 1_000_000, 10_000_000])
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU-friendly run: batches 1e3/1e4, 3 repeats")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.smoke:
        args.batches = [1_000, 10_000]
        args.repeats = 3
    device = torch.device(args.device)
    print(f"device={device} torch={torch.__version__} "
          f"gpu={torch.cuda.get_device_name(0) if device.type=='cuda' else 'n/a'}")

    results = []
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        for n in args.batches:
            batch = tuple(make_batch(n, device, dtype, seed=1))
            for name, fn in METHODS.items():
                try:
                    out, grad, fwd, fb, mem = bench_method(name, fn, batch, device, args.repeats)
                    row = {"method": name, "batch": n, "dtype": dtype_name,
                           "fwd_ms_median": round(fwd, 3),
                           "fwd_bwd_ms_median": round(fb, 3),
                           "peak_mem_mb": round(mem, 1),
                           **accuracy_row(out, grad, batch)}
                except RuntimeError as e:  # OOM etc.
                    row = {"method": name, "batch": n, "dtype": dtype_name,
                           "error": str(e)[:200]}
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                results.append(row)
                print(json.dumps(row))
        # finite-difference validation on a small batch (float64 path)
        small = tuple(make_batch(10_000, device, torch.float64, seed=2))
        fd = {"method": "finite_diff_validation", "dtype": "float64",
              **finite_diff_check(small, device)}
        results.append(fd)
        print(json.dumps(fd))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "device": str(device),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "newton_iters": NEWTON_ITERS,
            "results": results,
        }, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
