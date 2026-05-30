"""Offline invariants for the evaluation metrics (plan Part 6).

Pin the properties the head-to-head comparison relies on — run in seconds, no
network, no heavy deps:

    python -m pytest tests/
    python tests/test_metrics.py        # plain-python fallback
"""

from __future__ import annotations

import sys
from math import erf, pi, sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.metrics import (  # noqa: E402
    crps_ensemble,
    interval_coverage,
    mae,
    mase,
    pinball_loss,
    rmse,
    seasonal_naive_scale,
)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return np.exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _crps_normal_closed_form(y: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """CRPS of N(mu, sigma) at y — Gneiting & Raftery (2007)."""
    w = (y - mu) / sigma
    return sigma * (w * (2 * _norm_cdf(w) - 1) + 2 * _norm_pdf(w) - 1.0 / sqrt(pi))


def test_point_metrics_basic() -> None:
    y = np.zeros((4, 3, 2))
    yhat = np.ones((4, 3, 2))
    assert abs(mae(y, yhat) - 1.0) < 1e-12
    assert abs(rmse(y, yhat) - 1.0) < 1e-12
    # Perfect forecast → zero error.
    assert mae(y, y) == 0.0 and rmse(y, y) == 0.0


def test_crps_degenerate_equals_mae() -> None:
    """An ensemble of identical samples is a point forecast: CRPS must equal MAE."""
    rng = np.random.default_rng(0)
    y = rng.standard_normal((5, 4, 3))
    point = rng.standard_normal((5, 4, 3))
    samples = np.repeat(point[:, None, :, :], 50, axis=1)  # all S identical
    assert abs(crps_ensemble(y, samples) - mae(y, point)) < 1e-10


def test_crps_perfect_is_zero() -> None:
    y = np.random.default_rng(1).standard_normal((3, 4, 2))
    samples = np.repeat(y[:, None, :, :], 10, axis=1)
    assert crps_ensemble(y, samples) < 1e-12


def test_crps_matches_normal_closed_form() -> None:
    """Large ensemble from N(0,1) reproduces the analytic CRPS at y=0."""
    rng = np.random.default_rng(7)
    samples = rng.standard_normal((1, 40000, 1, 1))
    y = np.zeros((1, 1, 1))
    est = crps_ensemble(y, samples)
    assert abs(est - _crps_normal_closed_form(0.0)) < 0.01


def test_coverage_is_calibrated() -> None:
    """Truth and predictive samples from the same law → coverage ≈ nominal level."""
    rng = np.random.default_rng(3)
    y = rng.standard_normal((4000, 1, 1))
    samples = rng.standard_normal((4000, 600, 1, 1))
    assert abs(interval_coverage(y, samples, 0.9) - 0.9) < 0.03
    assert abs(interval_coverage(y, samples, 0.5) - 0.5) < 0.03


def test_mase_against_naive_scale() -> None:
    """A forecast that errs by exactly the naive scale gets MASE == 1."""
    rng = np.random.default_rng(5)
    train = np.cumsum(rng.standard_normal((400, 2)), axis=0)  # random walk
    scale = seasonal_naive_scale(train, season_length=1)
    y = np.zeros((6, 3, 2))
    yhat = np.broadcast_to(scale, (6, 3, 2))  # |y - yhat| == scale everywhere
    assert abs(mase(y, yhat, scale) - 1.0) < 1e-12


def test_pinball_approximates_half_crps() -> None:
    """CRPS = 2∫ρ_q dq, so 2 × (mean pinball over a quantile grid) ≈ CRPS."""
    rng = np.random.default_rng(9)
    y = rng.standard_normal((200, 2, 2))
    samples = rng.standard_normal((200, 400, 2, 2))
    assert abs(2.0 * pinball_loss(y, samples) - crps_ensemble(y, samples)) < 0.03


def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failures}/{len(fns)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
