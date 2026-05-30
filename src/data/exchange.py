"""Exchange loader — the iteration dataset (plan §4.2–4.3).

8 currencies vs USD, daily cadence, ~7588 steps, from the LSTNet multivariate
benchmark (the same lineage TimeGrad uses). Tiny and fast: we bring the whole
pipeline green here before scaling to the primary dataset.

This module wires the generic data contract (`contract.py`, `scaling.py`) to the
specific Exchange source. Public entry point: :func:`load_exchange`.

Run as a script for a smoke test of shapes::

    python -m src.data.exchange
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contract import ForecastDataset, temporal_split
from .scaling import Scaler


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------
def download(url: str, raw_dir: str | Path, filename: str) -> Path:
    """Idempotently fetch the raw gzipped file into ``raw_dir``.

    Skips the download if the file already exists. ``data/raw/`` is gitignored, so
    the file lives only on the local machine. Network access may be sandboxed; in
    that case download once outside the sandbox and the cached file is reused.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / filename
    if dest.exists():
        return dest

    import requests  # local import: only needed on the cache-miss path

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def load_raw(path: str | Path) -> pd.DataFrame:
    """Parse the gzipped, header-less CSV into a tidy ``(L, 8)`` DataFrame.

    The file has no timestamps; columns are bare floats. We return a plain
    integer-indexed frame here — the nominal calendar index is attached later in
    :func:`load_exchange` from the config, since it is for covariates only.
    """
    path = Path(path)
    with gzip.open(path, "rt") as fh:
        df = pd.read_csv(fh, header=None)
    df.columns = [f"c{i}" for i in range(df.shape[1])]
    return df


def _attach_calendar(df: pd.DataFrame, start_date: str, freq: str) -> pd.DataFrame:
    """Give the values a *nominal* DatetimeIndex (for calendar features only)."""
    idx = pd.date_range(start=start_date, periods=len(df), freq=freq)
    out = df.copy()
    out.index = idx
    return out


# ---------------------------------------------------------------------------
# The contract builder
# ---------------------------------------------------------------------------
def load_exchange(config: dict[str, Any]) -> ForecastDataset:
    """Build the :class:`ForecastDataset` for Exchange from a parsed config dict.

    Pipeline: download -> parse -> nominal calendar -> temporal split (no leakage)
    -> fit scaler on TRAIN ONLY -> transform every split. Returns the single object
    every model consumes.
    """
    src = config["source"]
    raw_path = download(src["url"], src["raw_dir"], src["filename"])
    df = load_raw(raw_path)
    df = _attach_calendar(df, src["start_date"], src["freq"])

    values = df.to_numpy(dtype=np.float64)  # (L, D), time-major
    raw_splits = temporal_split(values, tuple(config["split"]["ratios"]))

    # Fit on train only — the leakage guard (plan §4.4).
    scaler = Scaler(method=config["scaling"]["method"]).fit(raw_splits["train"])
    splits = {k: scaler.transform(v) for k, v in raw_splits.items()}

    meta = {
        "D": values.shape[1],
        "freq": src["freq"],
        "start_date": src["start_date"],
        "context_length": config["window"]["context_length"],
        "horizon": config["window"]["horizon"],
        "stride": config["window"].get("stride", 1),
        "scaling": config["scaling"]["method"],
        "split_ratios": list(config["split"]["ratios"]),
        "seed": config.get("seed"),
    }
    return ForecastDataset(
        name=config["name"],
        splits=splits,
        raw_splits=raw_splits,
        scaler=scaler,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _smoke_test() -> None:
    """Load Exchange from the committed config and print shapes (manual sanity check)."""
    from ..utils.config import load_config
    from ..utils.seeds import set_seed

    repo_root = Path(__file__).resolve().parents[2]
    cfg = load_config(repo_root / "configs" / "data_exchange.yaml")
    set_seed(cfg.get("seed", 42))

    ds = load_exchange(cfg)
    print(f"dataset      : {ds.name}  (D={ds.D}, H={ds.H}, tau={ds.tau})")
    for split in ("train", "val", "test"):
        raw = ds.raw_splits[split]
        ctx, tgt = ds.windows(split)
        print(
            f"  {split:5s}: series {raw.shape}  ->  "
            f"contexts {ctx.shape}  targets {tgt.shape}"
        )
    # Leakage sanity: scaled train should be ~zero-mean / unit-std per channel.
    tr = ds.splits["train"]
    print(f"scaled train : mean|max|={np.abs(tr.mean(0)).max():.2e}  std~{tr.std(0).mean():.3f}")


if __name__ == "__main__":
    _smoke_test()
