"""Deterministic seeding.

One call seeds every RNG we might touch (Python, NumPy, and — if installed —
PyTorch). The seed is read from the config and logged into the run manifest, so
every result in `results/registry.csv` is reproducible (CONTRIBUTING.md: "Seed
set and logged").
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> int:
    """Seed Python, NumPy, and (if available) PyTorch. Returns the seed used."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Torch is part of the heavy/Colab group; seed it only if it's importable so
    # the light local env has no hard torch dependency.
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    return seed
