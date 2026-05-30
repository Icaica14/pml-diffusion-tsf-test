"""Offline invariants for the M2 DeepAR wrapper (plan Part 5).

The contract checks that don't need GluonTS run everywhere; the ones that train a
tiny DeepAR are **skipped automatically when GluonTS/Lightning are absent**, so the
light local env (which deliberately omits the heavy group) still runs green.

    python -m pytest tests/
    python tests/test_deepar.py         # plain-python fallback
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.deepar import DeepARForecaster  # noqa: E402

HAVE_GLUONTS = (
    importlib.util.find_spec("gluonts") is not None
    and importlib.util.find_spec("lightning") is not None
)


def test_predict_before_fit_raises() -> None:
    """Contract guard — runs without GluonTS installed."""
    model = DeepARForecaster(horizon=4, context_length=10, freq="D")
    try:
        model.predict(np.zeros((2, 10, 1)))
    except RuntimeError:
        return
    raise AssertionError("predict() before fit() should raise RuntimeError.")


def test_import_error_is_clear_without_gluonts() -> None:
    """In the light env, fit() must fail with a message that names the heavy group."""
    if HAVE_GLUONTS:
        print("SKIP (gluonts present — light-env path not exercised)")
        return
    model = DeepARForecaster(horizon=4, context_length=10, freq="D")
    try:
        model.fit(object())  # dummy: import should fail before it is ever used
    except ImportError as exc:
        assert "heavy group" in str(exc).lower() or "gluonts" in str(exc).lower()
        return
    raise AssertionError("fit() without gluonts should raise a clear ImportError.")


def test_tiny_train_predict_shapes() -> None:
    """End-to-end shape contract on a 1-epoch DeepAR (heavy env only)."""
    if not HAVE_GLUONTS:
        print("SKIP (gluonts/lightning not installed)")
        return
    from gluonts.dataset.common import ListDataset

    rng = np.random.default_rng(0)
    D, L, H, tau, N, S = 2, 200, 20, 5, 3, 16
    series = np.cumsum(rng.standard_normal((L, D)), axis=0)
    train_ds = ListDataset(
        [{"start": "2000-01-01", "target": series[:, d].astype("float32")} for d in range(D)],
        freq="D",
    )
    model = DeepARForecaster(
        horizon=tau, context_length=H, freq="D", start="2000-01-01",
        hidden_size=8, num_layers=1, max_epochs=1, num_batches_per_epoch=3,
        batch_size=8, n_samples=S, accelerator="cpu", seed=0,
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
