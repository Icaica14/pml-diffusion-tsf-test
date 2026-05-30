"""Offline invariants for the M0 seasonal-naive forecaster (plan Part 5).

    python -m pytest tests/
    python tests/test_naive.py          # plain-python fallback
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.naive import SeasonalNaiveForecaster, seasonal_naive_point  # noqa: E402


def _toy_windows(N: int = 12, H: int = 10, tau: int = 4, D: int = 3):
    rng = np.random.default_rng(0)
    ctx = rng.standard_normal((N, H, D))
    tgt = rng.standard_normal((N, tau, D))
    return ctx, tgt


def test_persistence_repeats_last_value() -> None:
    """m=1 is the random-walk forecast: every horizon step equals the last context value."""
    ctx, _ = _toy_windows()
    point = seasonal_naive_point(ctx, season_length=1, horizon=4)
    last = ctx[:, -1, :]  # (N, D)
    for h in range(4):
        assert np.array_equal(point[:, h, :], last)


def test_seasonal_phase_is_repeated() -> None:
    """m>1 copies the matching phase from the last season of the context."""
    ctx, _ = _toy_windows(H=10, tau=5)
    m = 3
    point = seasonal_naive_point(ctx, season_length=m, horizon=5)
    H = ctx.shape[1]
    for h in range(5):
        expected = ctx[:, H - m + (h % m), :]
        assert np.array_equal(point[:, h, :], expected)


def test_predict_shapes_and_centering() -> None:
    ctx, tgt = _toy_windows()
    model = SeasonalNaiveForecaster(season_length=1, horizon=4, n_samples=64, seed=0).fit(ctx, tgt)
    point, samples = model.predict(ctx)
    assert point.shape == (12, 4, 3)
    assert samples.shape == (12, 64, 4, 3)
    # Residual pool has one error vector per training window.
    assert model.residual_pool_.shape == (12, 4, 3)


def test_predict_is_reproducible() -> None:
    ctx, tgt = _toy_windows()
    a = SeasonalNaiveForecaster(1, 4, n_samples=32, seed=42).fit(ctx, tgt).predict(ctx)[1]
    b = SeasonalNaiveForecaster(1, 4, n_samples=32, seed=42).fit(ctx, tgt).predict(ctx)[1]
    assert np.array_equal(a, b)
    c = SeasonalNaiveForecaster(1, 4, n_samples=32, seed=43).fit(ctx, tgt).predict(ctx)[1]
    assert not np.array_equal(a, c)


def test_samples_equal_point_plus_a_training_residual() -> None:
    """Every drawn sample is the point forecast plus some training residual vector."""
    ctx, tgt = _toy_windows()
    model = SeasonalNaiveForecaster(1, 4, n_samples=20, seed=1).fit(ctx, tgt)
    point, samples = model.predict(ctx)
    deltas = samples - point[:, None, :, :]  # (N, S, tau, D) — must be drawn residuals
    pool = model.residual_pool_
    # Each delta vector matches at least one row of the residual pool.
    for n in range(ctx.shape[0]):
        for s in range(samples.shape[1]):
            assert np.any(np.all(np.isclose(pool, deltas[n, s]), axis=(1, 2)))


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
