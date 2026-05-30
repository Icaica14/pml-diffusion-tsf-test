"""Forecast evaluation metrics (plan Part 6) — point *and* probabilistic.

Why two families? A point forecast collapses the future to one number, so it can
only be judged on *distance to the truth* (MAE, RMSE). A probabilistic forecast
emits a whole predictive distribution, so it must also be judged on whether that
distribution is *well-calibrated and sharp* (CRPS, interval coverage). The headline
of this project lives in the probabilistic family — that is where a diffusion model
can beat a point model even when their point accuracy ties.

Array conventions (match the data contract, plan §4.5)
------------------------------------------------------
* ``y_true``  : ``(N, tau, D)`` — N forecast windows, horizon tau, D channels.
* ``point``   : ``(N, tau, D)`` — a single predicted trajectory per window.
* ``samples`` : ``(N, S, tau, D)`` — S sampled trajectories per window (the
  Monte-Carlo stand-in for the predictive distribution).
All metrics are computed on the **original (un-scaled) data scale** so the numbers
are interpretable and comparable across models (plan §4.4).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

ArrayLike = np.ndarray


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------
def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Error — average L1 distance to the truth."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.abs(y_true - y_pred).mean())


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root Mean Squared Error — penalizes large misses more than MAE."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def seasonal_naive_scale(train_series: ArrayLike, season_length: int = 1) -> np.ndarray:
    """MASE denominator: in-sample mean absolute seasonal-naive error, per channel.

    MASE scales the forecast error by how hard the series is to predict with the
    *naive* model on the training data, so a value < 1 means "better than naive" and
    the metric is comparable across series of wildly different magnitudes.

    Parameters
    ----------
    train_series : ``(L, D)`` time-major training split (original scale).
    season_length : the naive lag ``m`` (1 = persistence/random-walk naive).

    Returns
    -------
    scale : ``(D,)`` per-channel mean ``|y_t - y_{t-m}|`` over the training split.
        Zero scales (a constant channel) are floored to a tiny epsilon to avoid
        division by zero.
    """
    x = np.asarray(train_series, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected (L, D) train series, got shape {x.shape}.")
    if x.shape[0] <= season_length:
        raise ValueError("Training series shorter than the seasonal lag.")
    diffs = np.abs(x[season_length:] - x[:-season_length])  # (L-m, D)
    scale = diffs.mean(axis=0)
    scale[scale == 0.0] = np.finfo(np.float64).eps
    return scale


def mase(y_true: ArrayLike, y_pred: ArrayLike, scale: ArrayLike) -> float:
    """Mean Absolute Scaled Error — MAE divided by the in-sample naive MAE.

    ``scale`` is the per-channel ``(D,)`` denominator from
    :func:`seasonal_naive_scale`. MASE < 1 ⇒ beats the naive baseline on its own
    training-difficulty yardstick; MASE == 1 ⇒ ties it.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scale = np.asarray(scale, dtype=np.float64)
    abs_err = np.abs(y_true - y_pred)  # (N, tau, D)
    return float((abs_err / scale).mean())


# ---------------------------------------------------------------------------
# Probabilistic metrics
# ---------------------------------------------------------------------------
def crps_ensemble(y_true: ArrayLike, samples: ArrayLike) -> float:
    """Continuous Ranked Probability Score from an ensemble of samples (lower = better).

    CRPS is the *generalization of MAE to distributions*: for a degenerate forecast
    (all samples identical) it reduces exactly to ``|x - y|``. It rewards forecasts
    that are both **calibrated** (mass in the right place) and **sharp** (not need-
    lessly wide).

    We use the **fair / almost-unbiased** ensemble estimator (Zamo & Naveau 2018):

        CRPS = (1/S) Σ_i |x_i - y|  −  1/(S(S-1)) Σ_i (2i - S - 1) x_(i)

    where ``x_(i)`` are the samples sorted ascending. The second term estimates
    ``½·E|X - X'|`` with the ``S-1`` correction that removes the small-ensemble bias
    of the naive ``1/S²`` form, and is computed in ``O(S log S)`` via the sort — no
    ``S×S`` pairwise matrix is ever allocated.

    Parameters
    ----------
    y_true  : ``(N, tau, D)``.
    samples : ``(N, S, tau, D)`` — S Monte-Carlo trajectories per window.

    Returns
    -------
    Mean CRPS over all ``(N, tau, D)`` positions.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != y_true.ndim + 1:
        raise ValueError(
            f"samples must be (N, S, tau, D) and y_true (N, tau, D); "
            f"got {samples.shape} and {y_true.shape}."
        )
    S = samples.shape[1]
    y = np.expand_dims(y_true, axis=1)  # (N, 1, tau, D)
    term1 = np.abs(samples - y).mean(axis=1)  # (N, tau, D) — E|X - y|
    if S < 2:
        return float(term1.mean())
    xs = np.sort(samples, axis=1)  # ascending along the sample axis
    # weights (2i - S - 1) for i = 1..S, broadcast over the sample axis
    shape = [1] * samples.ndim
    shape[1] = S
    weights = (2 * np.arange(1, S + 1) - S - 1).reshape(shape)
    term2 = (weights * xs).sum(axis=1) / (S * (S - 1))  # (N, tau, D) — ½ E|X - X'|
    return float((term1 - term2).mean())


def _central_quantiles(level: float) -> tuple[float, float]:
    """Lower/upper quantile levels for a central interval of coverage ``level``."""
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}.")
    alpha = 1.0 - level
    return alpha / 2.0, 1.0 - alpha / 2.0


def interval_coverage(y_true: ArrayLike, samples: ArrayLike, level: float = 0.9) -> float:
    """Empirical coverage of the central ``level`` predictive interval.

    A perfectly calibrated 90% interval contains the truth 90% of the time. We read
    the interval edges off the sample quantiles, then report the fraction of
    ``(N, tau, D)`` truths that land inside. Compare against ``level``: above ⇒ the
    forecast is under-confident (too wide); below ⇒ over-confident (too narrow).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    samples = np.asarray(samples, dtype=np.float64)
    lo_q, hi_q = _central_quantiles(level)
    lo = np.quantile(samples, lo_q, axis=1)
    hi = np.quantile(samples, hi_q, axis=1)
    inside = (y_true >= lo) & (y_true <= hi)
    return float(inside.mean())


def interval_width(samples: ArrayLike, level: float = 0.9) -> float:
    """Mean width of the central ``level`` interval — the **sharpness** of the forecast.

    Read alongside :func:`interval_coverage`: the goal is the *narrowest* interval
    that still achieves nominal coverage. A wide interval can hit coverage trivially
    but is uninformative.
    """
    samples = np.asarray(samples, dtype=np.float64)
    lo_q, hi_q = _central_quantiles(level)
    lo = np.quantile(samples, lo_q, axis=1)
    hi = np.quantile(samples, hi_q, axis=1)
    return float((hi - lo).mean())


def pinball_loss(
    y_true: ArrayLike,
    samples: ArrayLike,
    quantiles: Iterable[float] = tuple(np.round(np.arange(0.05, 1.0, 0.05), 2)),
) -> float:
    """Average pinball (quantile) loss over a grid of quantile levels.

    The pinball loss ``ρ_q`` is the loss whose minimizer is the ``q``-quantile; it
    penalizes under- and over-prediction asymmetrically by ``q`` vs ``1-q``.
    Because ``CRPS = 2 ∫₀¹ ρ_q dq``, averaging ``ρ_q`` over a dense grid of ``q``
    approximates **half the CRPS** — so ``2 × pinball ≈ CRPS``, a useful cross-check
    that the two probabilistic numbers tell the same story.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    samples = np.asarray(samples, dtype=np.float64)
    qs = np.asarray(list(quantiles), dtype=np.float64)
    total = 0.0
    for q in qs:
        pred_q = np.quantile(samples, q, axis=1)  # (N, tau, D)
        diff = y_true - pred_q
        total += np.maximum(q * diff, (q - 1.0) * diff).mean()
    return float(total / len(qs))


# ---------------------------------------------------------------------------
# Convenience aggregator
# ---------------------------------------------------------------------------
def evaluate_forecast(
    y_true: ArrayLike,
    point: ArrayLike,
    samples: ArrayLike,
    mase_scale: ArrayLike,
    levels: Iterable[float] = (0.5, 0.9),
) -> dict[str, float]:
    """Compute every metric in one call and return a flat ``{name: value}`` dict.

    ``levels`` is the set of central-interval coverages to report (default 50% and
    90%). Keys are emitted as ``cov50``/``width50``/``cov90``/``width90`` so they
    slot straight into the results registry as columns.
    """
    out: dict[str, float] = {
        "MAE": mae(y_true, point),
        "RMSE": rmse(y_true, point),
        "MASE": mase(y_true, point, mase_scale),
        "CRPS": crps_ensemble(y_true, samples),
        "pinball": pinball_loss(y_true, samples),
    }
    for level in levels:
        pct = int(round(level * 100))
        out[f"cov{pct}"] = interval_coverage(y_true, samples, level)
        out[f"width{pct}"] = interval_width(samples, level)
    return out
