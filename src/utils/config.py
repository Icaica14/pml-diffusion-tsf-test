"""Config loading.

One config = one run (CONTRIBUTING.md). Configs are plain YAML under `configs/`;
this module just reads them into a dict. Kept deliberately thin — no schema magic,
no defaults hidden in code — so the YAML stays the single source of truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} did not parse to a mapping (got {type(cfg)}).")
    return cfg
