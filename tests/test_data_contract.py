"""Offline invariants for the data contract (plan §4.4–4.5).

These run on a tiny synthetic array — NO network, NO heavy deps — so any teammate
can verify the leakage guards in seconds:

    python -m pytest tests/                 # if pytest is installed
    python tests/test_data_contract.py      # plain-python fallback

They pin the three things the code-review checklist (CONTRIBUTING.md) cares about:
contiguous time-ordered splits, train-only scaling, and no window straddling a
split boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.contract import make_windows, temporal_split  # noqa: E402
from src.data.scaling import Scaler  # noqa: E402


def _toy(L: int = 500, D: int = 8) -> np.ndarray:
    """A reproducible (L, D) array with a per-channel offset so leakage would show."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((L, D))
    # Give the most-recent (test) region a different mean+scale: if the scaler ever
    # peeked at it, the "train-only" assertion below would fail loudly.
    base[int(0.8 * L):] = base[int(0.8 * L):] * 5.0 + 20.0
    return base


def test_split_is_contiguous_and_ordered() -> None:
    x = _toy()
    s = temporal_split(x, (0.7, 0.1, 0.2))
    assert s["train"].shape[0] == 350
    assert s["val"].shape[0] == 50
    assert s["test"].shape[0] == 100
    # Every row used exactly once, in order: concatenation reconstructs the input.
    recon = np.concatenate([s["train"], s["val"], s["test"]], axis=0)
    assert np.array_equal(recon, x)


def test_scaler_fits_on_train_only() -> None:
    x = _toy()
    s = temporal_split(x, (0.7, 0.1, 0.2))
    scaler = Scaler().fit(s["train"])
    # Stats must equal the TRAIN stats, never the whole array's.
    assert np.allclose(scaler.mean_, s["train"].mean(0))
    assert not np.allclose(scaler.mean_, x.mean(0))  # the inflated test region differs
    # Scaled train is ~zero-mean / unit-std; scaled test is NOT (proves no peeking).
    tr = scaler.transform(s["train"])
    te = scaler.transform(s["test"])
    assert np.abs(tr.mean(0)).max() < 1e-12
    assert np.allclose(tr.std(0), 1.0, atol=1e-9)
    assert np.abs(te.mean(0)).max() > 1.0


def test_scaler_inverse_roundtrips() -> None:
    x = _toy()
    scaler = Scaler().fit(x[:350])
    z = scaler.transform(x)
    assert np.allclose(scaler.inverse_transform(z), x, atol=1e-9)


def test_scaler_handles_constant_channel() -> None:
    x = _toy()
    x[:, 0] = 7.0  # zero-variance channel -> std guard must avoid div-by-zero
    scaler = Scaler().fit(x[:350])
    z = scaler.transform(x)
    assert np.isfinite(z).all()
    assert np.allclose(z[:, 0], 0.0)  # constant maps to 0, not nan


def test_windows_shapes_and_count() -> None:
    x = _toy(L=500, D=8)
    H, tau = 60, 30
    ctx, tgt = make_windows(x, H, tau, stride=1)
    assert ctx.shape == (500 - (H + tau) + 1, H, 8)
    assert tgt.shape == (500 - (H + tau) + 1, tau, 8)


def test_windows_do_not_straddle_split_boundary() -> None:
    """Windows built per-split can never cross train/val/test — the core no-leak proof."""
    x = _toy(L=500, D=8)
    H, tau = 60, 30
    s = temporal_split(x, (0.7, 0.1, 0.2))
    total = 0
    for split in ("train", "val", "test"):
        ctx, tgt = make_windows(s[split], H, tau, stride=1)
        total += ctx.shape[0]
        # Each window is fully inside its split: reconstructed (H+tau) rows must
        # match a contiguous slice of that split alone.
        if ctx.shape[0] > 0:
            window0 = np.concatenate([ctx[0], tgt[0]], axis=0)
            assert np.array_equal(window0, s[split][: H + tau])
    # Building windows globally would yield MORE windows (the straddling ones);
    # the per-split count must be strictly smaller -> straddlers are excluded.
    global_ctx, _ = make_windows(x, H, tau, stride=1)
    assert total < global_ctx.shape[0]


def test_short_split_returns_empty_windows() -> None:
    x = _toy(L=50, D=8)  # shorter than H+tau=90
    ctx, tgt = make_windows(x, 60, 30, stride=1)
    assert ctx.shape == (0, 60, 8)
    assert tgt.shape == (0, 30, 8)


def _run_all() -> int:
    """Plain-python runner so the file works without pytest installed."""
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
