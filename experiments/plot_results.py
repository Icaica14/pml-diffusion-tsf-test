"""Regenerate the M0–M3 comparison figures from the results registry.

Mirrors the repo convention that *figures are never typed by hand*: every PNG under
``figures/cmp_*.png`` is produced here from ``results/registry.csv`` so the internal
report (``docs/RENDICONTO_SANDBOX_IT.md``) always shows the committed numbers.

Figures produced (all on the Exchange sandbox, test split):
  cmp_mase.png          point accuracy (MASE, lower = better) — bars
  cmp_crps.png          probabilistic accuracy (CRPS, lower = better) — bars
  cmp_calibration.png   reliability: nominal vs empirical coverage at 50% / 90%
  cmp_intervals.png     coverage (vs nominal) and interval width at 50% / 90%
  cmp_cost.png          train (fit_s) and inference (predict_s, log) wall-clock
  cmp_quality_cost.png  the "no free lunch" summary: CRPS vs predict_s (log-x)

Run::

    python -m experiments.plot_results
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "results" / "registry.csv"
FIGDIR = REPO_ROOT / "figures"

# Fixed model order + a stable colour/label per model, so a reader learns the code
# once ("red = TimeGrad") and reads every figure the same way.
ORDER = ["seasonal_naive", "arima", "deepar", "deepar_notf", "timegrad"]
CODE = {
    "seasonal_naive": "M0",
    "arima": "M1",
    "deepar": "M2",
    "deepar_notf": "M2-nf",
    "timegrad": "M3",
}
LABEL = {
    "seasonal_naive": "M0  seasonal-naive",
    "arima": "M1  ARIMA",
    "deepar": "M2  DeepAR",
    "deepar_notf": "M2-nf  DeepAR (no time-feat)",
    "timegrad": "M3  TimeGrad",
}
COLOR = {
    "seasonal_naive": "#7f7f7f",
    "arima": "#1f77b4",
    "deepar": "#ff7f0e",
    "deepar_notf": "#ffbb78",
    "timegrad": "#d62728",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})


def _load() -> pd.DataFrame:
    df = pd.read_csv(REGISTRY)
    df = df[df["model"].isin(ORDER)].copy()
    df["model"] = pd.Categorical(df["model"], categories=ORDER, ordered=True)
    df = df.sort_values("model").reset_index(drop=True)
    return df


def _codes(df: pd.DataFrame) -> list[str]:
    return [CODE[m] for m in df["model"]]


def _colors(df: pd.DataFrame) -> list[str]:
    return [COLOR[m] for m in df["model"]]


def _bar_labels(ax, bars, values, fmt="{:.3f}", dy=0.0):
    for b, v in zip(bars, values):
        if not np.isfinite(v):
            continue
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt.format(v),
                ha="center", va="bottom", fontsize=9)


def fig_mase(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    vals = df["MASE"].to_numpy()
    bars = ax.bar(_codes(df), vals, color=_colors(df), edgecolor="black", linewidth=0.6)
    _bar_labels(ax, bars, vals, fmt="{:.2f}", dy=0.05)
    naive = float(df.loc[df["model"] == "seasonal_naive", "MASE"].iloc[0])
    ax.axhline(naive, ls="--", color="#7f7f7f", lw=1.2)
    ax.text(len(df) - 0.5, naive, "  persistence reference", color="#555",
            va="bottom", ha="right", fontsize=9)
    ax.set_ylabel("MASE  (lower = better)")
    ax.set_title("Point accuracy — Mean Absolute Scaled Error")
    ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout()
    fig.savefig(FIGDIR / "cmp_mase.png")
    plt.close(fig)


def fig_crps(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    vals = df["CRPS"].to_numpy()
    bars = ax.bar(_codes(df), vals, color=_colors(df), edgecolor="black", linewidth=0.6)
    _bar_labels(ax, bars, vals, fmt="{:.4f}", dy=vals.max() * 0.01)
    ax.set_ylabel("CRPS  (lower = better)")
    ax.set_title("Probabilistic accuracy — Continuous Ranked Probability Score")
    ax.set_ylim(0, vals.max() * 1.18)
    fig.tight_layout()
    fig.savefig(FIGDIR / "cmp_crps.png")
    plt.close(fig)


def fig_calibration(df: pd.DataFrame) -> None:
    """Reliability plot: nominal coverage on x, empirical on y, diagonal = perfect."""
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    nominal = np.array([0.5, 0.9])
    ax.plot([0, 1], [0, 1], ls="--", color="black", lw=1.0, label="perfect calibration")
    for _, row in df.iterrows():
        emp = np.array([row["cov50"], row["cov90"]])
        ax.plot(nominal, emp, "-o", color=COLOR[row["model"]], lw=1.8, ms=8,
                label=LABEL[row["model"]])
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks([0.5, 0.9])
    ax.set_xlabel("nominal coverage (target)")
    ax.set_ylabel("empirical coverage (achieved)")
    ax.set_title("Calibration — below diagonal = over-confident")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(FIGDIR / "cmp_calibration.png")
    plt.close(fig)


def fig_intervals(df: pd.DataFrame) -> None:
    """Two panels: coverage vs nominal (left) and interval width / sharpness (right)."""
    fig, (axc, axw) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    x = np.arange(len(df))
    w = 0.38
    # Coverage panel
    axc.bar(x - w / 2, df["cov50"], w, color=_colors(df), edgecolor="black",
            linewidth=0.6, label="50%")
    axc.bar(x + w / 2, df["cov90"], w, color=_colors(df), edgecolor="black",
            linewidth=0.6, alpha=0.55, label="90%")
    axc.axhline(0.5, ls="--", color="#333", lw=1.0)
    axc.axhline(0.9, ls="--", color="#333", lw=1.0)
    axc.text(len(df) - 0.5, 0.5, " nominal 50%", va="bottom", ha="right", fontsize=8, color="#333")
    axc.text(len(df) - 0.5, 0.9, " nominal 90%", va="bottom", ha="right", fontsize=8, color="#333")
    axc.set_xticks(x)
    axc.set_xticklabels(_codes(df))
    axc.set_ylabel("empirical coverage")
    axc.set_title("Coverage vs nominal (dashed)")
    axc.set_ylim(0, 1.05)
    axc.legend(title="interval", fontsize=9)
    # Width panel
    axw.bar(x - w / 2, df["width50"], w, color=_colors(df), edgecolor="black",
            linewidth=0.6, label="50%")
    axw.bar(x + w / 2, df["width90"], w, color=_colors(df), edgecolor="black",
            linewidth=0.6, alpha=0.55, label="90%")
    axw.set_xticks(x)
    axw.set_xticklabels(_codes(df))
    axw.set_ylabel("mean interval width (original scale)")
    axw.set_title("Sharpness — narrower = sharper")
    axw.legend(title="interval", fontsize=9)
    fig.suptitle("Calibration is coverage AND sharpness together", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGDIR / "cmp_intervals.png")
    plt.close(fig)


def fig_cost(df: pd.DataFrame) -> None:
    fig, (axf, axp) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    fit = df["fit_s"].fillna(0.0).to_numpy()
    pred = df["predict_s"].to_numpy()
    bf = axf.bar(_codes(df), fit, color=_colors(df), edgecolor="black", linewidth=0.6)
    _bar_labels(axf, bf, fit, fmt="{:.0f}s", dy=max(fit) * 0.01)
    axf.set_ylabel("train wall-clock (s)")
    axf.set_title("Training cost (fit_s)")
    bp = axp.bar(_codes(df), pred, color=_colors(df), edgecolor="black", linewidth=0.6)
    axp.set_yscale("log")
    for b, v in zip(bp, pred):
        axp.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}s", ha="center",
                 va="bottom", fontsize=9)
    axp.set_ylabel("inference wall-clock (s, log scale)")
    axp.set_title("Inference cost (predict_s) — note log axis")
    fig.suptitle("Cost — the diffusion sampling price", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGDIR / "cmp_cost.png")
    plt.close(fig)


def fig_quality_cost(df: pd.DataFrame) -> None:
    """The summary: probabilistic quality (CRPS) against inference cost (predict_s)."""
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for _, row in df.iterrows():
        ax.scatter(row["predict_s"], row["CRPS"], s=140, color=COLOR[row["model"]],
                   edgecolor="black", linewidth=0.8, zorder=3)
        ax.annotate(CODE[row["model"]], (row["predict_s"], row["CRPS"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=10, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("inference cost — predict_s (s, log scale)")
    ax.set_ylabel("CRPS  (lower = better)")
    ax.set_title("No free lunch — quality vs cost\n(bottom-left is best; M3 is far right)")
    # legend mapping codes to names
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=COLOR[m],
                          markeredgecolor="black", label=LABEL[m]) for m in ORDER]
    ax.legend(handles=handles, fontsize=8.5, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "cmp_quality_cost.png")
    plt.close(fig)


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    df = _load()
    fig_mase(df)
    fig_crps(df)
    fig_calibration(df)
    fig_intervals(df)
    fig_cost(df)
    fig_quality_cost(df)
    made = sorted(p.name for p in FIGDIR.glob("cmp_*.png"))
    print(f"Wrote {len(made)} figures to {FIGDIR}/:")
    for name in made:
        print(f"  {name}")


if __name__ == "__main__":
    main()
