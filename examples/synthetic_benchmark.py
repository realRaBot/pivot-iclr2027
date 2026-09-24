"""Licensed-data-free synthetic benchmark for the PIVOT operator.

Generates synthetic Black-76 option batches, then measures:
  1. forward IV round-trip accuracy sigma -> P -> J(P) vs sigma;
  2. backward gradient accuracy vs the analytic implicit derivative 1/vega;
  3. synchronized forward and forward+backward wall time across batch sizes;
  4. invalid-input and low-vega failure accounting.

Run: python examples/synthetic_benchmark.py [--device cuda] [--dtype float64]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fast_vollib.jackel.differentiable import implied_volatility_autograd  # noqa: E402


def make_batch(n: int, device, dtype, seed: int = 0, low_vega_frac: float = 0.02):
    g = torch.Generator(device="cpu").manual_seed(seed)
    F = torch.full((n,), 100.0, dtype=dtype)
    k = torch.exp(torch.randn(n, generator=g, dtype=dtype) * 0.25) * 100.0
    t = torch.rand(n, generator=g, dtype=dtype) * 1.9 + 0.02
    r = torch.full((n,), 0.02, dtype=dtype)
    sigma = torch.rand(n, generator=g, dtype=dtype) * 0.45 + 0.05
    # Push a slice deep OTM/short-dated to stress the low-vega regime.
    m = int(n * low_vega_frac)
    if m:
        k[:m] = F[:m] * torch.exp(torch.tensor(3.0, dtype=dtype))
        t[:m] = 0.02
        sigma[:m] = 0.05
    is_call = k >= F
    return (x.to(device) for x in (F, k, t, r, sigma, is_call))


def bs_price_vega(is_call, F, k, t, r, sigma):
    sqrt_t = torch.sqrt(t)
    d1 = (torch.log(F / k) + 0.5 * sigma**2 * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    disc = torch.exp(-r * t)
    ncdf = lambda x: 0.5 * (1 + torch.erf(x / 2**0.5))
    npdf = lambda x: torch.exp(-0.5 * x**2) / (2 * torch.pi) ** 0.5
    call = disc * (F * ncdf(d1) - k * ncdf(d2))
    put = disc * (k * ncdf(-d2) - F * ncdf(-d1))
    price = torch.where(is_call, call, put)
    vega = disc * F * npdf(d1) * sqrt_t
    return price, vega


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[1_000, 100_000, 1_000_000, 10_000_000])
    ap.add_argument("--repeats", type=int, default=7)
    args = ap.parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    print(f"device={device} dtype={dtype}")

    # --- accuracy + gradient check on a moderate batch ---
    F, k, t, r, sigma, is_call = make_batch(200_000, device, dtype)
    price, vega = bs_price_vega(is_call, F, k, t, r, sigma)
    price_in = price.clone().requires_grad_(True)
    sigma_hat = implied_volatility_autograd(price_in, F, k, t, r, is_call, model="black")
    valid = torch.isfinite(sigma_hat)
    rt_err = (sigma_hat[valid] - sigma[valid]).abs()
    print(f"valid_fraction={valid.float().mean().item():.6f}")
    print(f"roundtrip_max_abs_err={rt_err.max().item():.3e}  mean={rt_err.mean().item():.3e}")

    (g,) = torch.autograd.grad(sigma_hat[valid].sum(), price_in)
    analytic = torch.where(valid, 1.0 / vega, torch.zeros_like(vega))
    well = valid & (vega > 1e-4)
    gerr = (g[well] - analytic[well]).abs() / analytic[well].abs()
    print(f"grad_max_rel_err_wellcond={gerr.max().item():.3e}  mean={gerr.mean().item():.3e}")

    # --- timing across batch sizes ---
    print("batch | fwd_ms(med) | fwd+bwd_ms(med) | peak_mem_MB")
    for n in args.batches:
        F, k, t, r, sigma, is_call = make_batch(n, device, dtype, seed=1)
        price, _ = bs_price_vega(is_call, F, k, t, r, sigma)
        fwd_times, fb_times = [], []
        for _ in range(args.repeats):
            p = price.clone().requires_grad_(True)
            if device.type == "cuda":
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            s = implied_volatility_autograd(p, F, k, t, r, is_call, model="black")
            if device.type == "cuda": torch.cuda.synchronize()
            t1 = time.perf_counter()
            s[torch.isfinite(s)].sum().backward()
            if device.type == "cuda": torch.cuda.synchronize()
            t2 = time.perf_counter()
            fwd_times.append((t1 - t0) * 1e3); fb_times.append((t2 - t0) * 1e3)
        med = lambda xs: sorted(xs)[len(xs) // 2]
        mem = (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else float("nan")
        print(f"{n:>9,} | {med(fwd_times):10.2f} | {med(fb_times):10.2f} | {mem:10.1f}")


if __name__ == "__main__":
    main()
