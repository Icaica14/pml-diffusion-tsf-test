"""Per-channel standardization, fit on TRAIN ONLY.

Scaling leakage is the #1 silent bug in a forecasting comparison (plan §4.4): if
the scaler sees val/test statistics, every downstream metric is quietly optimistic.
This `Scaler` is fit on the training split alone and then *applied* to val/test, so
the leak is impossible by construction.

Convention: arrays are time-major, shape ``(L, D)`` — rows are time steps, columns
are channels. Statistics are computed over time (axis=0), one mean/std per channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Scaler:
    """Per-channel z-score scaler. Fit on train, transform everything.

    Attributes
    ----------
    method : str
        Only ``"standardize"`` for now (z-score). Kept as a field so the config's
        ``scaling.method`` is recorded and future methods slot in here.
    mean_, std_ : np.ndarray
        Per-channel statistics, shape ``(D,)``. Set by :meth:`fit`.
    """

    method: str = "standardize"
    mean_: np.ndarray = field(default=None, repr=False)
    std_: np.ndarray = field(default=None, repr=False)

    def fit(self, train: np.ndarray) -> "Scaler":
        """Compute per-channel statistics from the training split only."""
        if self.method != "standardize":
            raise NotImplementedError(f"Unknown scaling method: {self.method!r}")
        train = np.asarray(train, dtype=np.float64)
        if train.ndim != 2:
            raise ValueError(f"Expected (L, D) array, got shape {train.shape}.")
        self.mean_ = train.mean(axis=0)
        std = train.std(axis=0)
        # Guard zero-variance channels: a constant column would divide by zero and
        # poison every downstream value with inf/nan. Treat std==0 as 1 (no scaling).
        std[std == 0.0] = 1.0
        self.std_ = std
        return self

    def _check_fitted(self) -> None:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler.transform called before fit().")

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Standardize an ``(L, D)`` array using the fitted train statistics."""
        self._check_fitted()
        x = np.asarray(x, dtype=np.float64)
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Map standardized values back to the original scale (for metrics/plots)."""
        self._check_fitted()
        x = np.asarray(x, dtype=np.float64)
        return x * self.std_ + self.mean_

    def to_dict(self) -> dict:
        """Serializable stats for the run manifest (reproducibility audit)."""
        self._check_fitted()
        return {
            "method": self.method,
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
        }
