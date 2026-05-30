"""Run the M2 DeepAR baseline and append its row to the results registry.

The third rung of the model ladder, and the first that needs the **heavy / Colab**
environment (GluonTS + Lightning). It follows the exact conventions of
``run_naive.py`` / ``run_arima.py`` — build the :class:`ForecastDataset` from a config,
train on the **train** split, forecast the **test** windows on the **original** scale,
score with the shared metrics module, append one provenance-tagged row — so the
M0/M1/M2 numbers are comparable cell-for-cell.

DeepAR training is one global model over the channels (GluonTS scales each series
internally); per-window forecasts reuse the trained predictor with no refit.

Usage (on a machine with the heavy group installed)::

    python -m experiments.run_deepar                     # Exchange, default budget
    python -m experiments.run_deepar --epochs 30 --samples 200
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
from src.models.deepar import DeepARForecaster  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seeds import set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the M2 DeepAR baseline.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "data_exchange.yaml"))
    parser.add_argument("--epochs", type=int, default=20, help="Lightning max_epochs.")
    parser.add_argument("--batches", type=int, default=50, help="num_batches_per_epoch.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=40, help="LSTM hidden size.")
    parser.add_argument("--layers", type=int, default=2, help="LSTM layers.")
    parser.add_argument("--samples", type=int, default=100, help="Sampled trajectories per window.")
    parser.add_argument("--accelerator", default="auto",
                        help="Lightning accelerator: 'auto' (GPU on Colab), 'gpu', or 'cpu'.")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    parser.add_argument("--registry", default=str(REPO_ROOT / "results" / "registry.csv"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)

    ds = load_exchange(cfg)

    # Train on the raw train series via the contract's GluonTS bridge (GluonTS scales
    # internally, so it consumes the un-scaled series). Metrics read the raw test set.
    train_ds = ds.to_gluonts("train")
    te_ctx, te_tgt = ds.windows("test", scaled=False)

    model = DeepARForecaster(
        horizon=ds.tau,
        context_length=ds.H,
        freq=ds.meta["freq"],
        start=ds.meta.get("start_date"),
        num_layers=args.layers,
        hidden_size=args.hidden,
        max_epochs=args.epochs,
        num_batches_per_epoch=args.batches,
        batch_size=args.batch_size,
        n_samples=args.samples,
        accelerator=args.accelerator,
        seed=seed,
    )

    t_fit = time.perf_counter()
    model.fit(train_ds)
    fit_s = time.perf_counter() - t_fit

    t0 = time.perf_counter()
    point, samples = model.predict(te_ctx)
    predict_s = time.perf_counter() - t0

    # MASE denominator is a data-level constant: identical to M0/M1 for comparability.
    scale = seasonal_naive_scale(ds.raw_splits["train"], season_length=1)
    metrics = evaluate_forecast(te_tgt, point, samples, scale, levels=(0.5, 0.9))

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": ds.name,
        "model": "deepar",
        "split": "test",
        "n_windows": int(te_tgt.shape[0]),
        "H": ds.H,
        "tau": ds.tau,
        "D": ds.D,
        "n_samples": args.samples,
        "epochs": args.epochs,
        "hidden": args.hidden,
        "layers": args.layers,
        "seed": seed,
        **{k: round(v, 6) for k, v in metrics.items()},
        "fit_s": round(fit_s, 4),
        "predict_s": round(predict_s, 4),
        "platform": platform.platform(),
    }
    append_result(args.registry, row)

    # Console summary.
    print(f"M2 DeepAR  ·  {ds.name}  ·  test split")
    print(f"  windows={row['n_windows']}  H={ds.H}  tau={ds.tau}  D={ds.D}  "
          f"S={args.samples}  seed={seed}")
    print(f"  model   : LSTM layers={args.layers} hidden={args.hidden}  "
          f"epochs={args.epochs} x {args.batches} batches")
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
