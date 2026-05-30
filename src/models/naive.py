"""M0 — the seasonal-naive baseline (plan Part 5, the "honesty anchor").

Every model in the ladder must beat this. The seasonal-naive forecast simply
**repeats the last observed season**: the prediction for horizon step ``h`` is the
value seen ``m`` steps earlier, where ``m`` is the seasonal period. With ``m = 1``
this is the *random-walk / persistence* forecast — "tomorrow equals today" — which,
as the Exchange EDA showed (near-unit-root levels, lag-1 autocorrelation ≈ 0.999),
is a genuinely hard baseline to beat on that dataset. For a strongly seasonal series
(e.g. hourly electricity) one sets ``m = 24`` or ``168`` instead.

Making it *probabilistic*
-------------------------
A bare naive forecast is a single line; to score it on CRPS / coverage it needs a
predictive *distribution*. We attach one the classical, assumption-light way — a
**residual bootstrap**: collect the naive model's forecast errors on the training
windows, then form predictive samples by adding resampled error trajectories to the
point forecast. Because the errors of a random-walk naive grow with the horizon, the
resulting bands automatically widen with ``h`` — the honest shape for this data. We
resample **whole error vectors** (one per training window) rather than per-step, so
the cross-horizon and cross-channel correlation of the errors is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def seasonal_naive_point(
    contexts: np.ndarray, season_length: int, horizon: int
) -> np.ndarray:
    """Seasonal-naive point forecast for a batch of context windows.

    Parameters
    ----------
    contexts : ``(N, H, D)`` — N context windows (most recent step is ``[:, -1]``).
    season_length : the naive lag ``m`` (``m = 1`` ⇒ repeat the last value).
    horizon : forecast length ``tau``.

    Returns
    -------
    ``(N, tau, D)`` forecast: step ``h`` copies the value ``m`` steps before the
    matching phase, i.e. ``contexts[:, H - m + (h mod m), :]``.
    """
    contexts = np.asarray(contexts, dtype=np.float64)
    if contexts.ndim != 3:
        raise ValueError(f"Expected (N, H, D) contexts, got shape {contexts.shape}.")
    H = contexts.shape[1]
    if season_length < 1:
        raise ValueError(f"season_length must be >= 1, got {season_length}.")
    if season_length > H:
        raise ValueError(
            f"season_length ({season_length}) exceeds context length H ({H})."
        )
    idx = H - season_length + (np.arange(horizon) % season_length)
    return contexts[:, idx, :]


@dataclass
class SeasonalNaiveForecaster:
    """Seasonal-naive point forecast + residual-bootstrap predictive samples.

    Parameters
    ----------
    season_length : naive lag ``m`` (1 = persistence, the Exchange default).
    horizon : forecast length ``tau``.
    n_samples : number ``S`` of predictive trajectories drawn per window.
    seed : RNG seed for the bootstrap (logged for reproducibility).

    Notes
    -----
    :meth:`predict` allocates an ``(N, S, tau, D)`` sample tensor. That is cheap for
    Exchange (this sandbox's only dataset) but would be large for a wide series; in
    that case score the test split in chunks.
    """

    season_length: int
    horizon: int
    n_samples: int = 100
    seed: int = 0
    residual_pool_: np.ndarray = field(default=None, repr=False)

    # -- fit / predict --------------------------------------------------------
    def fit(self, train_contexts: np.ndarray, train_targets: np.ndarray) -> "SeasonalNaiveForecaster":
        """Collect the naive model's training forecast errors (the bootstrap pool).

        ``train_contexts`` : ``(M, H, D)`` and ``train_targets`` : ``(M, tau, D)`` are
        the training windows on the **original** data scale. The residual pool stored
        is ``targets - naive_point(contexts)``, shape ``(M, tau, D)``.
        """
        train_contexts = np.asarray(train_contexts, dtype=np.float64)
        train_targets = np.asarray(train_targets, dtype=np.float64)
        if train_contexts.shape[0] == 0:
            raise ValueError("No training windows to fit the residual bootstrap.")
        point = seasonal_naive_point(train_contexts, self.season_length, self.horizon)
        self.residual_pool_ = train_targets - point
        return self

    def point_forecast(self, contexts: np.ndarray) -> np.ndarray:
        """The deterministic ``(N, tau, D)`` seasonal-naive forecast."""
        return seasonal_naive_point(contexts, self.season_length, self.horizon)

    def predict(self, contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(point, samples)`` for a batch of context windows.

        ``point``   : ``(N, tau, D)`` seasonal-naive forecast.
        ``samples`` : ``(N, S, tau, D)`` = point + bootstrapped training residuals.
        """
        if self.residual_pool_ is None:
            raise RuntimeError("predict() called before fit().")
        point = self.point_forecast(contexts)  # (N, tau, D)
        N = point.shape[0]
        M = self.residual_pool_.shape[0]
        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, M, size=(N, self.n_samples))  # (N, S)
        drawn = self.residual_pool_[idx]  # (N, S, tau, D)
        samples = point[:, None, :, :] + drawn
        return point, samples
