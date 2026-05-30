"""Offline invariants for the M3 TimeGrad wrapper (plan Part 5).

The contract checks that don't need PyTorchTS run everywhere; the one that trains a
tiny TimeGrad is **skipped automatically when ``pts`` is absent**, so the light local
env (which deliberately omits the heavy group) still runs green.

    python -m pytest tests/
    python tests/test_timegrad.py        # plain-python fallback
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.timegrad import TimeGradForecaster  # noqa: E402

HAVE_PTS = importlib.util.find_spec("pts") is not None


def test_predict_before_fit_raises() -> None:
    """Contract guard — runs without PyTorchTS installed."""
    model = TimeGradForecaster(target_dim=2, horizon=4, context_length=10, freq="D")
    try:
        model.predict(np.zeros((2, 10, 2)))
    except RuntimeError:
        return
    raise AssertionError("predict() before fit() should raise RuntimeError.")


def test_predict_channel_mismatch_raises() -> None:
    """A context with the wrong number of channels must be rejected (no PyTorchTS needed)."""
    model = TimeGradForecaster(target_dim=2, horizon=4, context_length=10, freq="D")
    model.predictor_ = object()  # bypass the fit-guard to reach the D check
    try:
        model.predict(np.zeros((2, 10, 3)))  # D=3 but target_dim=2
    except ValueError as exc:
        assert "target_dim" in str(exc)
        return
    raise AssertionError("predict() with mismatched D should raise ValueError.")


def test_import_error_is_clear_without_pts() -> None:
    """In the light env, fit() must fail with a message that names the heavy group."""
    if HAVE_PTS:
        print("SKIP (pts present — light-env path not exercised)")
        return
    model = TimeGradForecaster(target_dim=2, horizon=4, context_length=10, freq="D")
    try:
        model.fit(object())  # dummy: import should fail before it is ever used
    except ImportError as exc:
        assert "heavy group" in str(exc).lower() or "pytorchts" in str(exc).lower()
        return
    raise AssertionError("fit() without pts should raise a clear ImportError.")


def test_tiny_train_predict_shapes() -> None:
    """End-to-end shape contract on a 1-epoch TimeGrad (heavy env only)."""
    if not HAVE_PTS:
        print("SKIP (pytorchts not installed)")
        return
    from gluonts.dataset.common import ListDataset

    rng = np.random.default_rng(0)
    D, L, H, tau, N, S = 2, 200, 20, 5, 3, 8
    series = np.cumsum(rng.standard_normal((L, D)), axis=0)
    # Multivariate training entry: target is channels-first (D, L).
    train_ds = ListDataset(
        [{"start": "2000-01-01", "target": series.T.astype("float32")}],
        freq="D", one_dim_target=False,
    )
    model = TimeGradForecaster(
        target_dim=D, horizon=tau, context_length=H, freq="D", start="2000-01-01",
        num_cells=8, num_layers=1, diff_steps=10, max_epochs=1, num_batches_per_epoch=2,
        batch_size=4, n_samples=S, device="cpu", seed=0,
    ).fit(train_ds)

    ctx = np.cumsum(rng.standard_normal((N, H, D)), axis=1) + series[-1]
    point, samples = model.predict(ctx)
    assert point.shape == (N, tau, D)
    assert samples.shape == (N, S, tau, D)
    assert np.isfinite(point).all() and np.isfinite(samples).all()
    # The reported point is the per-position sample mean.
    assert np.allclose(samples.mean(axis=1), point)


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
