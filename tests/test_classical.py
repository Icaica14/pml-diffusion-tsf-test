"""Offline invariants for the M1 ARIMA forecaster (plan Part 5).

Skipped automatically if statsmodels is absent, so the light env that lacks it still
runs the rest of the suite green.

    python -m pytest tests/
    python tests/test_classical.py      # plain-python fallback
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import statsmodels  # noqa: F401

    HAVE_STATSMODELS = True
except ImportError:  # pragma: no cover - exercised only in the stripped env
    HAVE_STATSMODELS = False

from src.models.classical import ClassicalForecaster  # noqa: E402


def _toy(D: int = 2, L: int = 300, seed: int = 0):
    """A pair of independent Gaussian random walks (Exchange-like integrated series)."""
    rng = np.random.default_rng(seed)
    train = np.cumsum(rng.standard_normal((L, D)), axis=0)
    # Context windows continued from the training tail so apply() has a sensible state.
    N, H = 6, 40
    ctx = np.cumsum(rng.standard_normal((N, H, D)), axis=1) + train[-1]
    return train, ctx


def test_predict_shapes() -> None:
    if not HAVE_STATSMODELS:
        print("SKIP (statsmodels not installed)")
        return
    train, ctx = _toy()
    model = ClassicalForecaster(horizon=5, n_samples=32, seed=0).fit(train)
    point, samples = model.predict(ctx)
    assert point.shape == (6, 5, 2)
    assert samples.shape == (6, 32, 5, 2)
    assert len(model.orders_) == 2 and len(model.results_) == 2
    assert np.isfinite(point).all() and np.isfinite(samples).all()


def test_point_equals_predictive_mean() -> None:
    """The returned point forecast is the centre of the Gaussian sample cloud."""
    if not HAVE_STATSMODELS:
        print("SKIP (statsmodels not installed)")
        return
    train, ctx = _toy()
    model = ClassicalForecaster(horizon=6, n_samples=4000, seed=3).fit(train)
    point, samples = model.predict(ctx)
    # Monte-Carlo mean of the draws must converge to the analytic point forecast.
    assert np.abs(samples.mean(axis=1) - point).max() < 0.15


def test_uncertainty_grows_with_horizon() -> None:
    """ARIMA forecast SE accumulates: later horizon steps are wider than earlier ones."""
    if not HAVE_STATSMODELS:
        print("SKIP (statsmodels not installed)")
        return
    train, ctx = _toy(D=1, seed=1)
    model = ClassicalForecaster(horizon=10, order=(0, 1, 0), n_samples=2000, seed=0).fit(train)
    _, samples = model.predict(ctx)
    spread = samples.std(axis=1).mean(axis=(0, 2))  # mean sample std per horizon step
    assert spread[-1] > spread[0]
    assert np.all(np.diff(spread) > -1e-6)  # monotone non-decreasing (random walk)


def test_fixed_order_is_used_for_all_channels() -> None:
    if not HAVE_STATSMODELS:
        print("SKIP (statsmodels not installed)")
        return
    train, _ = _toy(D=3)
    model = ClassicalForecaster(horizon=4, order=(1, 1, 1), n_samples=8, seed=0).fit(train)
    assert model.orders_ == [(1, 1, 1), (1, 1, 1), (1, 1, 1)]


def test_predict_is_reproducible() -> None:
    """Same seed → identical Gaussian draws; different seed → different draws."""
    if not HAVE_STATSMODELS:
        print("SKIP (statsmodels not installed)")
        return
    train, ctx = _toy()
    a = ClassicalForecaster(horizon=5, order=(0, 1, 0), n_samples=16, seed=42).fit(train).predict(ctx)[1]
    b = ClassicalForecaster(horizon=5, order=(0, 1, 0), n_samples=16, seed=42).fit(train).predict(ctx)[1]
    c = ClassicalForecaster(horizon=5, order=(0, 1, 0), n_samples=16, seed=43).fit(train).predict(ctx)[1]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_predict_before_fit_raises() -> None:
    model = ClassicalForecaster(horizon=4)
    try:
        model.predict(np.zeros((2, 10, 1)))
    except RuntimeError:
        return
    raise AssertionError("predict() before fit() should raise RuntimeError.")


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
