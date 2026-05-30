"""Run the M0 seasonal-naive baseline and append its row to the results registry.

This is the first rung of the model ladder and the first real entry in
``results/registry.csv``. It establishes the conventions every later experiment
follows: build the :class:`ForecastDataset` from a config, forecast the **test**
windows on the **original** scale, score with the shared metrics module, and append
exactly one provenance-tagged row.

Usage::

    python -m experiments.run_naive                     # Exchange, persistence (m=1)
    python -m experiments.run_naive --season 5 --samples 200
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
from src.models.naive import SeasonalNaiveForecaster  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seeds import set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the M0 seasonal-naive baseline.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "data_exchange.yaml"))
    parser.add_argument(
        "--season",
        type=int,
        default=1,
        help="Seasonal lag m. Default 1 (persistence) — the honest naive for the "
        "near-random-walk Exchange series (see docs/EDA_EXCHANGE.md).",
    )
    parser.add_argument("--samples", type=int, default=100, help="Bootstrap samples per window.")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    parser.add_argument("--registry", default=str(REPO_ROOT / "results" / "registry.csv"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)

    ds = load_exchange(cfg)

    # Metrics are read on the ORIGINAL scale → use raw (un-scaled) windows.
    tr_ctx, tr_tgt = ds.windows("train", scaled=False)
    te_ctx, te_tgt = ds.windows("test", scaled=False)

    model = SeasonalNaiveForecaster(
        season_length=args.season,
        horizon=ds.tau,
        n_samples=args.samples,
        seed=seed,
    ).fit(tr_ctx, tr_tgt)

    t0 = time.perf_counter()
    point, samples = model.predict(te_ctx)
    predict_s = time.perf_counter() - t0

    scale = seasonal_naive_scale(ds.raw_splits["train"], args.season)
    metrics = evaluate_forecast(te_tgt, point, samples, scale, levels=(0.5, 0.9))

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": ds.name,
        "model": "seasonal_naive",
        "season_length": args.season,
        "split": "test",
        "n_windows": int(te_tgt.shape[0]),
        "H": ds.H,
        "tau": ds.tau,
        "D": ds.D,
        "n_samples": args.samples,
        "seed": seed,
        **{k: round(v, 6) for k, v in metrics.items()},
        "predict_s": round(predict_s, 4),
        "platform": platform.platform(),
    }
    append_result(args.registry, row)

    # Provenance: pin exactly how the data was built alongside the numbers.
    ds.save_manifest(REPO_ROOT / "results" / f"manifest_{ds.name}.json")

    # Console summary.
    print(f"M0 seasonal-naive (m={args.season})  ·  {ds.name}  ·  test split")
    print(f"  windows={row['n_windows']}  H={ds.H}  tau={ds.tau}  D={ds.D}  "
          f"S={args.samples}  seed={seed}")
    print("  point   : "
          f"MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  MASE={metrics['MASE']:.4f}")
    print("  prob    : "
          f"CRPS={metrics['CRPS']:.4f}  pinball={metrics['pinball']:.4f}")
    print("  calib   : "
          f"cov50={metrics['cov50']:.3f} (width {metrics['width50']:.4f})  "
          f"cov90={metrics['cov90']:.3f} (width {metrics['width90']:.4f})")
    print(f"  predict : {predict_s:.3f}s  ->  appended to {Path(args.registry).name}")


if __name__ == "__main__":
    main()
