"""The data contract (plan §4.5).

A single object, :class:`ForecastDataset`, that every model in the ladder consumes
unchanged. It bundles the three time-ordered splits, the train-only scaler, and the
metadata (``D, freq, H, tau``) so the head-to-head comparison is *fair by
construction*: nobody can accidentally use a different split, scaling, horizon, or
set of test windows.

Conventions
-----------
* Arrays are **time-major**: shape ``(L, D)`` — rows are time steps, columns are
  channels. A context window is ``(H, D)``; a target window is ``(tau, D)``.
* Splits are **contiguous in time** (no shuffling); test is the most-recent slice.
* Windows are built **within** a single split, so no window straddles a boundary
  and there is no leakage across train/val/test by construction (plan §4.4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .scaling import Scaler


# ---------------------------------------------------------------------------
# Splitting & windowing — pure functions, easy to unit-test
# ---------------------------------------------------------------------------
def temporal_split(
    values: np.ndarray,
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> dict[str, np.ndarray]:
    """Split a time-major ``(L, D)`` array into contiguous train/val/test.

    Time order is preserved (never shuffle a time series): the test split is the
    most-recent slice. Boundaries are floor-based on cumulative ratios; any rounding
    remainder is absorbed by the test split so all rows are used exactly once.
    """
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"Expected (L, D) array, got shape {values.shape}.")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios} (sum={sum(ratios)}).")
    L = values.shape[0]
    n_train = int(np.floor(ratios[0] * L))
    n_val = int(np.floor(ratios[1] * L))
    return {
        "train": values[:n_train],
        "val": values[n_train : n_train + n_val],
        "test": values[n_train + n_val :],
    }


def make_windows(
    values: np.ndarray,
    context_length: int,
    horizon: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice sliding ``(context, target)`` windows out of one split.

    Parameters
    ----------
    values : np.ndarray
        Time-major ``(L, D)`` array for a **single** split.
    context_length, horizon : int
        ``H`` and ``tau``. Each window needs ``H + tau`` consecutive steps.
    stride : int
        Step between successive window starts.

    Returns
    -------
    contexts : np.ndarray, shape ``(N, H, D)``
    targets  : np.ndarray, shape ``(N, tau, D)``
        Built only from inside ``values``, so no window crosses a split boundary.
        ``N == 0`` (returned as correctly-shaped empty arrays) if the split is
        shorter than ``H + tau``.
    """
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"Expected (L, D) array, got shape {values.shape}.")
    L, D = values.shape
    span = context_length + horizon
    contexts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for start in range(0, L - span + 1, stride):
        ctx_end = start + context_length
        contexts.append(values[start:ctx_end])
        targets.append(values[ctx_end : ctx_end + horizon])
    if not contexts:
        return (
            np.empty((0, context_length, D), dtype=values.dtype),
            np.empty((0, horizon, D), dtype=values.dtype),
        )
    return np.stack(contexts), np.stack(targets)


# ---------------------------------------------------------------------------
# The contract object
# ---------------------------------------------------------------------------
@dataclass
class ForecastDataset:
    """The single object every model consumes (plan §4.5).

    ``splits`` holds the **scaled**, time-major arrays for ``train``/``val``/``test``.
    ``raw_splits`` keeps the original-scale arrays (GluonTS scales internally, and
    metrics must be read on the original scale). ``scaler`` is fit on
    train only. ``meta`` records everything the comparison must hold fixed.
    """

    name: str
    splits: dict[str, np.ndarray]
    raw_splits: dict[str, np.ndarray]
    scaler: Scaler
    meta: dict[str, Any]

    # -- convenience accessors ------------------------------------------------
    @property
    def D(self) -> int:
        return int(self.meta["D"])

    @property
    def H(self) -> int:
        return int(self.meta["context_length"])

    @property
    def tau(self) -> int:
        return int(self.meta["horizon"])

    def windows(self, split: str, scaled: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """``(contexts, targets)`` for a split — scaled by default, raw if asked."""
        source = self.splits if scaled else self.raw_splits
        return make_windows(
            source[split],
            self.meta["context_length"],
            self.meta["horizon"],
            self.meta.get("stride", 1),
        )

    # -- GluonTS bridge (M2 DeepAR / M3 TimeGrad live in the heavy env) --------
    def to_gluonts(self, split: str = "train"):
        """Build a GluonTS ``ListDataset`` (one univariate entry per channel).

        Uses the **raw** (un-scaled) series: GluonTS applies its own per-series
        scaling internally, so feeding it pre-scaled values would double-scale.
        Import is guarded so the light/local env needs no GluonTS install.
        """
        try:
            from gluonts.dataset.common import ListDataset
        except ImportError as exc:  # pragma: no cover - heavy/Colab env only
            raise ImportError(
                "to_gluonts() needs the heavy/Colab group (gluonts). It is not "
                "installed in the light local env (see requirements.txt)."
            ) from exc

        arr = self.raw_splits[split]  # (L, D)
        freq = self.meta["freq"]
        start = self.meta.get("start_date")
        entries = [
            {"start": start, "target": arr[:, d].astype("float32")}
            for d in range(arr.shape[1])
        ]
        return ListDataset(entries, freq=freq)

    def to_gluonts_multivariate(self, split: str = "train"):
        """Build a GluonTS multivariate ``ListDataset`` — **one** entry, target ``(D, L)``.

        Where :meth:`to_gluonts` emits ``D`` independent univariate series (M2 DeepAR
        models each channel on its own), **M3 TimeGrad models the ``D`` channels
        *jointly***: its diffusion step denoises the whole ``D``-vector at once, so it
        needs the multivariate layout GluonTS encodes as a single entry whose ``target``
        has shape ``(D, L)`` with ``one_dim_target=False``.

        Uses the **raw** (un-scaled) series — TimeGrad/GluonTS scale internally, exactly
        as in :meth:`to_gluonts`. Import is guarded so the light env needs no GluonTS.
        """
        try:
            from gluonts.dataset.common import ListDataset
        except ImportError as exc:  # pragma: no cover - heavy/Colab env only
            raise ImportError(
                "to_gluonts_multivariate() needs the heavy/Colab group (gluonts). It is "
                "not installed in the light local env (see requirements.txt)."
            ) from exc

        arr = self.raw_splits[split]  # (L, D)
        freq = self.meta["freq"]
        start = self.meta.get("start_date")
        target = arr.T.astype("float32")  # (D, L) — channels-first for multivariate
        entry = {"start": start, "target": target}
        return ListDataset([entry], freq=freq, one_dim_target=False)

    # -- provenance -----------------------------------------------------------
    def manifest(self) -> dict[str, Any]:
        """A JSON-serializable record of how this dataset was built.

        Captures split sizes, scaler stats, and the windowing knobs so any result
        can be traced back to an exact data build (plan §4.4 "Determinism").
        """
        return {
            "name": self.name,
            "meta": self.meta,
            "split_sizes": {k: int(v.shape[0]) for k, v in self.raw_splits.items()},
            "scaler": self.scaler.to_dict(),
        }

    def save_manifest(self, path: str | Path) -> Path:
        """Write :meth:`manifest` to ``path`` as pretty JSON; returns the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.manifest(), fh, indent=2)
        return path
