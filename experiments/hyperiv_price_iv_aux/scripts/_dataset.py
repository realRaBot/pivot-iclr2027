"""Per-date HyperIV dataset that also returns side tensors needed for price/RT loss."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

REF_FEATURES = ["log_moneyness", "tau", "sigma_star"]
QUERY_FEATURES = ["log_moneyness", "tau"]
SIDE_FEATURES = ["F", "K", "tau", "r", "is_call", "mid", "vega_star"]


class OptionPriceIVDataset(Dataset):
    """Returns (z, X, y, side) per date.

    z   : (N=9, 3)        reference set Z = [(k, t, sigma_star)]_i.
    X   : (M=N_sample, 2) sampled (k, t) for surface-network query.
    y   : (M,)            sigma_star labels for those samples.
    side: (M, 7)          [F, K, tau, r, is_call, mid, vega_star] aligned with X.
    """

    def __init__(self, df: pd.DataFrame, n_samples: int = 1024, sample: bool = True,
                 seed: int = 0) -> None:
        # Use sort-then-slice; O(N log N) instead of O(N*D).
        df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
        ref_df = df[df["is_ref"] == 1]

        # Build per-date row-slice indices for both full and ref tables.
        full_dates = df["date"].to_numpy()
        ref_dates = ref_df["date"].to_numpy()

        # Get unique dates in same order; np.unique gives sorted unique.
        unique_dates, full_starts = np.unique(full_dates, return_index=True)
        full_ends = np.append(full_starts[1:], len(full_dates))
        ref_unique, ref_starts = np.unique(ref_dates, return_index=True)
        ref_ends = np.append(ref_starts[1:], len(ref_dates))
        ref_index = {d: (s, e) for d, s, e in zip(ref_unique, ref_starts, ref_ends)}

        # Pre-extract numpy matrices once; index by slice in __getitem__.
        ref_mat = ref_df[REF_FEATURES].to_numpy(dtype=np.float32)
        xy_mat = df[QUERY_FEATURES + ["sigma_star"]].to_numpy(dtype=np.float32)
        side_mat = df[SIDE_FEATURES].to_numpy(dtype=np.float32)

        self._dates = unique_dates
        self._ref_mat = ref_mat
        self._xy_mat = xy_mat
        self._side_mat = side_mat
        # Aligned slice arrays
        full_idx = []
        for d, s, e in zip(unique_dates, full_starts, full_ends):
            r = ref_index.get(d)
            if r is None:
                # No reference set for this date; skip via 0-len.
                full_idx.append((s, e, 0, 0))
            else:
                full_idx.append((s, e, r[0], r[1]))
        self._slices = np.array(full_idx, dtype=np.int64)

        self.n_samples = int(n_samples)
        self.sample = bool(sample)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._dates)

    def __getitem__(self, idx: int):
        fs, fe, rs, re_ = self._slices[idx]
        ref_arr = self._ref_mat[rs:re_]
        xy = self._xy_mat[fs:fe]
        side = self._side_mat[fs:fe]
        n = xy.shape[0]
        if self.sample and n > 0:
            ix = self._rng.integers(0, n, size=self.n_samples)
            xy_s = xy[ix]
            side_s = side[ix]
        else:
            xy_s = xy
            side_s = side
        X = xy_s[:, :2]
        y = xy_s[:, 2]
        return (
            torch.from_numpy(ref_arr.copy()),
            torch.from_numpy(X.copy()),
            torch.from_numpy(y.copy()),
            torch.from_numpy(side_s.copy()),
        )
