"""Run the M1 ARIMA baseline and append its row to the results registry.

The second rung of the model ladder (after M0 seasonal-naive). It follows exactly
the conventions ``run_naive.py`` established — build the :class:`ForecastDataset` from
a config, fit on the **train** split, forecast the **test** windows on the
**original** scale, score with the shared metrics module, append one
provenance-tagged row — so the M0/M1 numbers are comparable cell-for-cell.

ARIMA orders are selected per channel by AIC (unless ``--order`` pins one for all
channels) and logged both to the console and to ``results/arima_orders_exchange.json``.

Usage::

    python -m experiments.run_arima                      # Exchange, auto order per channel
    python -m experiments.run_arima --order 1,1,1        # fix ARIMA(1,1,1) everywhere
    python -m experiments.run_arima --samples 200
"""

from __future__ import annotations

import argparse
import json
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
from src.models.classical import ClassicalForecaster  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seeds import set_seed  # noqa: E402


def _parse_order(text: str | None) -> tuple[int, int, int] | None:
    """Parse ``"p,d,q"`` into a 3-tuple, or ``None`` for auto-selection."""
    if text is None:
        return None
    parts = tuple(int(p) for p in text.split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--order must be 'p,d,q', e.g. 1,1,1")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the M1 ARIMA baseline.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "data_exchange.yaml"))
    parser.add_argument(
        "--order",
        type=str,
        default=None,
        help="Fixed ARIMA p,d,q for every channel (e.g. 1,1,1). Default: auto-select "
        "each channel's order by AIC.",
    )
    parser.add_argument("--samples", type=int, default=100, help="Gaussian predictive samples per window.")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    parser.add_argument("--registry", default=str(REPO_ROOT / "results" / "registry.csv"))
    args = parser.parse_args()

    order = _parse_order(args.order)

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)

    ds = load_exchange(cfg)

    # Metrics are read on the ORIGINAL scale → fit and forecast on raw values.
    train_series = ds.raw_splits["train"]
    te_ctx, te_tgt = ds.windows("test", scaled=False)

    t_fit = time.perf_counter()
    model = ClassicalForecaster(
        horizon=ds.tau,
        order=order,
        n_samples=args.samples,
        seed=seed,
    ).fit(train_series)
    fit_s = time.perf_counter() - t_fit

    t0 = time.perf_counter()
    point, samples = model.predict(te_ctx)
    predict_s = time.perf_counter() - t0

    # MASE denominator is a data-level constant: identical to M0 for comparability.
    scale = seasonal_naive_scale(train_series, season_length=1)
    metrics = evaluate_forecast(te_tgt, point, samples, scale, levels=(0.5, 0.9))

    # Compact, human-readable summary of the per-channel orders for the registry row.
    orders_str = ";".join(f"{p}.{d}.{q}" for (p, d, q) in model.orders_)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": ds.name,
        "model": "arima",
        "order": "auto" if order is None else f"{order[0]}.{order[1]}.{order[2]}",
        "orders_per_channel": orders_str,
        "split": "test",
        "n_windows": int(te_tgt.shape[0]),
        "H": ds.H,
        "tau": ds.tau,
        "D": ds.D,
        "n_samples": args.samples,
        "seed": seed,
        **{k: round(v, 6) for k, v in metrics.items()},
        "fit_s": round(fit_s, 4),
        "predict_s": round(predict_s, 4),
        "platform": platform.platform(),
    }
    append_result(args.registry, row)

    # Provenance: pin the exact per-channel orders next to the numbers.
    orders_path = REPO_ROOT / "results" / f"arima_orders_{ds.name}.json"
    with orders_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {"dataset": ds.name, "seed": seed,
             "orders_per_channel": [list(o) for o in model.orders_]},
            fh, indent=2,
        )

    # Console summary.
    sel = "fixed" if order is not None else "AIC-selected"
    print(f"M1 ARIMA ({sel})  ·  {ds.name}  ·  test split")
    print(f"  windows={row['n_windows']}  H={ds.H}  tau={ds.tau}  D={ds.D}  "
          f"S={args.samples}  seed={seed}")
    print(f"  orders  : {orders_str}")
    print("  point   : "
          f"MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  MASE={metrics['MASE']:.4f}")
    print("  prob    : "
          f"CRPS={metrics['CRPS']:.4f}  pinball={metrics['pinball']:.4f}")
    print("  calib   : "
          f"cov50={metrics['cov50']:.3f} (width {metrics['width50']:.4f})  "
          f"cov90={metrics['cov90']:.3f} (width {metrics['width90']:.4f})")
    print(f"  time    : fit {fit_s:.2f}s  predict {predict_s:.2f}s  "
          f"->  appended to {Path(args.registry).name}")


if __name__ == "__main__":
    main()
