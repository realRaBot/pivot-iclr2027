"""HyperIV experiment-local shim around `experiments/scripts/`.

Keeps the original `from _common import ...` API while delegating all
shared logic to ``experiments/scripts`` and resolving the bundled source
from the repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

EXPT_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = EXPT_ROOT.parent / "scripts"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from paths import (  # type: ignore  # noqa: E402
    REPO_ROOT,
    add_sys_path,
    expt_figures_dir as _shared_expt_figures_dir,
    expt_outputs_dir as _shared_expt_outputs_dir,
    load_config,
    logs_dir as _shared_logs_dir,
    prepared_dir as _shared_prepared_dir,
    runs_dir as _shared_runs_dir,
)

# This anonymous artifact vendors the exact source snapshot under ``src/``.
FAST_VOLLIB_SRC = REPO_ROOT / "src"
add_sys_path(FAST_VOLLIB_SRC)

# Make `from experiments.utils.<x> import ...` resolvable regardless of cwd.
add_sys_path(REPO_ROOT)


def expt_outputs_dir() -> Path:
    return _shared_expt_outputs_dir(EXPT_ROOT)


def expt_figures_dir() -> Path:
    return _shared_expt_figures_dir(EXPT_ROOT)


def prepared_dir(cfg: dict[str, Any]) -> Path:
    return _shared_prepared_dir(EXPT_ROOT, cfg)


def runs_dir(cfg: dict[str, Any]) -> Path:
    return _shared_runs_dir(EXPT_ROOT, cfg)


def logs_dir(cfg: dict[str, Any]) -> Path:
    return _shared_logs_dir(EXPT_ROOT, cfg)


__all__ = [
    "EXPT_ROOT", "REPO_ROOT", "FAST_VOLLIB_SRC",
    "load_config", "expt_outputs_dir", "expt_figures_dir",
    "prepared_dir", "runs_dir", "logs_dir",
]
