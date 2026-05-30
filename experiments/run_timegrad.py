"""Run the M3 TimeGrad model and append its row to the results registry.

The fourth rung of the ladder and the **conditional-diffusion centerpiece** — the model
the whole project builds toward. It follows the exact conventions of ``run_naive.py`` /
``run_arima.py`` / ``run_deepar.py`` — build the :class:`ForecastDataset` from a config,
train on the **train** split, forecast the **test** windows on the **original** scale,
score with the shared metrics module, append one provenance-tagged row — so the
M0/M1/M2/M3 numbers are comparable cell-for-cell.

Unlike M2 (D univariate DeepARs), TimeGrad is **multivariate**: it denoises the joint
D-vector, so it consumes ``ds.to_gluonts_multivariate("train")``. Per-window forecasts
reuse the trained predictor with no refit (leakage-free, identical to M1/M2).

Usage (on the Colab/GPU box with the pinned heavy group — see the M3 notebook)::

    python -m experiments.run_timegrad                       # Exchange, default budget
    python -m experiments.run_timegrad --epochs 30 --samples 100 --device cuda
    python -m experiments.run_timegrad --input-size 200      # pin if auto-detect misfires
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.exchange import load_exchange  # noqa: E402
from src.eval.metrics import evaluate_forecast, seasonal_naive_scale  # noqa: E402
from src.eval.registry import append_result  # noqa: E402
from src.models.timegrad import TimeGradForecaster  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seeds import set_seed  # noqa: E402


def _resolve_device(choice: str) -> str:
    """Map ``--device auto`` to 'cuda' when a GPU is visible, else 'cpu'."""
    if choice != "auto":
        return choice
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the M3 TimeGrad model.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "data_exchange.yaml"))
    parser.add_argument("--epochs", type=int, default=20, help="pts Trainer epochs.")
    parser.add_argument("--batches", type=int, default=50, help="num_batches_per_epoch.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-cells", type=int, default=64, help="Conditioning RNN width.")
    parser.add_argument("--layers", type=int, default=2, help="Conditioning RNN layers.")
    parser.add_argument("--cell-type", default="GRU", help="RNN cell: 'GRU' or 'LSTM'.")
    parser.add_argument("--diff-steps", type=int, default=100, help="Diffusion steps T.")
    parser.add_argument("--beta-end", type=float, default=0.1, help="Noise-schedule beta_end.")
    parser.add_argument("--beta-schedule", default="linear", help="'linear' or 'cosine'.")
    parser.add_argument("--samples", type=int, default=100, help="Sampled trajectories per window.")
    parser.add_argument("--input-size", type=int, default=None,
                        help="Pin TimeGrad's RNN input width (else auto-detected).")
    parser.add_argument("--device", default="auto",
                        help="'auto' (GPU when present), 'cuda', or 'cpu'.")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    parser.add_argument("--registry", default=str(REPO_ROOT / "results" / "registry.csv"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)
    device = _resolve_device(args.device)

    ds = load_exchange(cfg)

    # Train on the raw multivariate train series via the contract's multivariate bridge
    # (TimeGrad scales internally). Metrics read the raw test set on the original scale.
    train_ds = ds.to_gluonts_multivariate("train")
    te_ctx, te_tgt = ds.windows("test", scaled=False)

    model = TimeGradForecaster(
        target_dim=ds.D,
        horizon=ds.tau,
        context_length=ds.H,
        freq=ds.meta["freq"],
        start=ds.meta.get("start_date"),
        num_cells=args.num_cells,
        num_layers=args.layers,
        cell_type=args.cell_type,
        diff_steps=args.diff_steps,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        max_epochs=args.epochs,
        num_batches_per_epoch=args.batches,
        batch_size=args.batch_size,
        n_samples=args.samples,
        input_size=args.input_size,
        device=device,
        seed=seed,
    )

    t_fit = time.perf_counter()
    model.fit(train_ds)
    fit_s = time.perf_counter() - t_fit

    t0 = time.perf_counter()
    point, samples = model.predict(te_ctx)
    predict_s = time.perf_counter() - t0

    # MASE denominator is a data-level constant: identical to M0/M1/M2 for comparability.
    scale = seasonal_naive_scale(ds.raw_splits["train"], season_length=1)
    metrics = evaluate_forecast(te_tgt, point, samples, scale, levels=(0.5, 0.9))

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": ds.name,
        "model": "timegrad",
        "split": "test",
        "n_windows": int(te_tgt.shape[0]),
        "H": ds.H,
        "tau": ds.tau,
        "D": ds.D,
        "n_samples": args.samples,
        "epochs": args.epochs,
        "diff_steps": args.diff_steps,
        "num_cells": args.num_cells,
        "layers": args.layers,
        "input_size": model.input_size_,
        "seed": seed,
        **{k: round(v, 6) for k, v in metrics.items()},
        "fit_s": round(fit_s, 4),
        "predict_s": round(predict_s, 4),
        "platform": platform.platform(),
    }
    append_result(args.registry, row)

    # Console summary.
    print(f"M3 TimeGrad  ·  {ds.name}  ·  test split")
    print(f"  windows={row['n_windows']}  H={ds.H}  tau={ds.tau}  D={ds.D}  "
          f"S={args.samples}  seed={seed}  device={device}")
    print(f"  model   : {args.cell_type} cells={args.num_cells} layers={args.layers}  "
          f"diff_steps={args.diff_steps} beta_end={args.beta_end}  "
          f"input_size={model.input_size_}")
    print(f"  train   : epochs={args.epochs} x {args.batches} batches")
    print("  point   : "
          f"MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  MASE={metrics['MASE']:.4f}")
    print("  prob    : "
          f"CRPS={metrics['CRPS']:.4f}  pinball={metrics['pinball']:.4f}")
    print("  calib   : "
          f"cov50={metrics['cov50']:.3f} (width {metrics['width50']:.4f})  "
          f"cov90={metrics['cov90']:.3f} (width {metrics['width90']:.4f})")
    print(f"  time    : fit {fit_s:.1f}s  predict {predict_s:.1f}s  "
          f"->  appended to {Path(args.registry).name}")


if __name__ == "__main__":
    main()
