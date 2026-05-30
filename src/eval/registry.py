"""The results registry (plan §8.5) — the single source of truth for numbers.

Every experiment appends **one row** to ``results/registry.csv``; tables and figures
for the slides are *generated* from this file, never typed by hand. Centralizing the
append here keeps the schema consistent and makes a run reproducible from its row
(dataset, model, split, seed, and the knobs that define the comparison).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def append_result(path: str | Path, row: dict[str, Any]) -> Path:
    """Append ``row`` to the CSV at ``path`` (created with a header if absent).

    New keys that were not in earlier rows become new columns (older rows get blank
    cells), so the schema can grow as models add metrics without breaking old rows.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new], ignore_index=True, sort=False)
    else:
        combined = new
    combined.to_csv(path, index=False)
    return path
