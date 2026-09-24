"""HyperIV-style architecture used by the price/IV auxiliary experiment.

Mirrors `models/hyperiv/hyperiv_util.py` (read-only) without modifying it. The
base IV-surface MLP follows the paper's "2 hidden layers x 16 neurons, ~337
parameters" sketch with a softplus output to keep sigma positive.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _BaseSurfaceMLP(nn.Module):
    """h_omega(.): tiny MLP (k, t) -> sigma > 0.

    2 hidden layers of 16 neurons. Output uses softplus so sigma > 0 without
    requiring positive weights. Total params ≈ 2*16 + 16 + 16*16 + 16 + 16 + 1
    = 337.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return torch.nn.functional.softplus(x) + 1e-4


class SetEmbeddingNetwork(nn.Module):
    """g_theta(.): set encoder Z -> flat omega vector.

    Permutation-invariant Transformer-without-positional-embedding + mean pool.
    """

    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 2,
                 num_layers: int = 2, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                batch_first=True, dropout=0, activation="relu",
            )
            for _ in range(num_layers)
        ])
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        for layer in self.attention_layers:
            x = layer(x)
        x = x.mean(dim=1)
        return self.fc2(x)


class HyperNetwork(nn.Module):
    """Composition f_theta(k, t | Z) wiring g_theta into h_omega via vmap+functional_call."""

    def __init__(self, hyper: nn.Module, base: nn.Module) -> None:
        super().__init__()
        self._hyper = hyper
        params_lookup: list[list] = [[None, None, 0, 0]]
        for name, param in base.named_parameters():
            s = params_lookup[-1][-1]
            params_lookup.append([name, param.shape, s, s + int(np.prod(param.shape))])
        self._params_lookup = params_lookup[1:]
        buffers: dict = {}
        self._call = lambda params, data: torch.func.functional_call(
            base, (params, buffers), (data,)
        )

    @property
    def base_param_count(self) -> int:
        return self._params_lookup[-1][3] if self._params_lookup else 0

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        params = self._hyper(x)
        params_dict: dict[str, torch.Tensor] = {}
        for entry in self._params_lookup:
            name, shape, lo, hi = entry
            params_dict[name] = params[:, lo:hi].reshape(-1, *shape)
        return torch.func.vmap(self._call, in_dims=(0, 0))(params_dict, z)


def build_model() -> tuple[HyperNetwork, _BaseSurfaceMLP]:
    base = _BaseSurfaceMLP()
    n_params = sum(p.numel() for p in base.parameters())
    hyper = SetEmbeddingNetwork(input_dim=3, output_dim=n_params)
    return HyperNetwork(hyper, base), base
