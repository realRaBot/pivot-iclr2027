"""Create a tiny licensed-data-free fixture for downstream smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr


def make_split(n_dates: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for day in range(n_dates):
        date = pd.Timestamp("2022-01-03") + pd.Timedelta(days=day)
        n = 48
        log_m = np.linspace(-0.35, 0.20, n)
        tau = np.tile(np.array([7, 30, 90, 180], dtype=float) / 365.0, n // 4)
        forward = 100.0 + 0.15 * day
        strike = forward * np.exp(log_m)
        sigma = 0.18 + 0.08 * np.abs(log_m) + 0.02 * np.sqrt(tau)
        sigma += rng.normal(0.0, 0.001, size=n)
        rate = np.full(n, 0.02)
        is_call = strike >= forward
        sqrt_t = np.sqrt(tau)
        d1 = (np.log(forward / strike) + 0.5 * sigma**2 * tau) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        disc = np.exp(-rate * tau)
        call = disc * (forward * ndtr(d1) - strike * ndtr(d2))
        put = disc * (strike * ndtr(-d2) - forward * ndtr(-d1))
        price = np.where(is_call, call, put)
        vega = disc * forward * np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi) * sqrt_t
        ref_idx = np.linspace(0, n - 1, 9, dtype=int)
        for i in range(n):
            rows.append({
                "date": date,
                "is_ref": int(i in ref_idx),
                "log_moneyness": float(log_m[i]),
                "tau": float(tau[i]),
                "sigma_star": float(sigma[i]),
                "F": float(forward),
                "K": float(strike[i]),
                "r": float(rate[i]),
                "is_call": int(is_call[i]),
                "mid": float(price[i]),
                "vega_star": float(vega[i]),
            })
    return pd.DataFrame(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    make_split(24, 7).to_parquet(out / "synthetic_train.parquet", index=False)
    make_split(6, 11).to_parquet(out / "synthetic_test.parquet", index=False)
    print(f"wrote synthetic fixture to {out}")


if __name__ == "__main__":
    main()
