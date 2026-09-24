"""Path/config helpers shared by every experiment under `experiments/`.

Centralizes TOML loading, local-source bootstrapping, and resolution of the
writable directories an experiment uses (`prepared/`, `runs/`, `logs/`,
`outputs/`, `figures/`). Optional external storage overrides are accepted
through `[paths.external]` in the TOML config.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from typing import Any

# Repository root (this file lives at experiments/scripts/paths.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_SCRIPTS_DIR = Path(__file__).resolve().parent


def add_sys_path(p: Path | str) -> None:
    """Insert ``p`` at the head of ``sys.path`` if it isn't already there."""
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / cfg_path).resolve()
    with cfg_path.open("rb") as f:
        return tomllib.load(f)


def expt_root_for(common_file: str | Path) -> Path:
    """Return the experiment root given the path of an experiment's `_common.py`.

    Each experiment's `_common.py` lives at
    ``experiments/<expt>/scripts/_common.py``; the experiment root is its
    grandparent.
    """
    return Path(common_file).resolve().parents[1]


def _external_path(cfg: dict[str, Any], key: str) -> Path | None:
    p = cfg.get("paths", {}).get("external", {}).get(key)
    if not p:
        return None
    p = Path(p)
    try:
        p.mkdir(parents=True, exist_ok=True)
        return p
    except OSError:
        return None


def _expt_or_external(expt_root: Path, cfg: dict[str, Any], external_key: str,
                     local_subdir: str) -> Path:
    p = _external_path(cfg, external_key)
    if p is not None:
        return p
    p = expt_root / local_subdir
    p.mkdir(parents=True, exist_ok=True)
    return p


def expt_outputs_dir(expt_root: Path) -> Path:
    p = expt_root / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def expt_figures_dir(expt_root: Path) -> Path:
    p = expt_root / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p


def prepared_dir(expt_root: Path, cfg: dict[str, Any]) -> Path:
    return _expt_or_external(expt_root, cfg, "prepared_dir", "prepared")


def runs_dir(expt_root: Path, cfg: dict[str, Any]) -> Path:
    return _expt_or_external(expt_root, cfg, "runs_dir", "runs")


def logs_dir(expt_root: Path, cfg: dict[str, Any]) -> Path:
    return _expt_or_external(expt_root, cfg, "logs_dir", "logs")
