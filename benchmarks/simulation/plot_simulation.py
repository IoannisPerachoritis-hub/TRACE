"""Simulation result plots: null QQ, lambda boxplot, window sensitivity."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.pub_theme import apply_matplotlib_theme
apply_matplotlib_theme()

PAPER_FIGURES = ROOT / "paper" / "figures"

TOOL_COLORS = {
    "platform_loco": "#2563eb",
    "platform_global": "#FF9800",
    "gapit": "#16a34a",
    "rmvp": "#ea580c",
}
TOOL_LABELS = {
    "platform_loco": "TRACE-LOCO",
    "platform_global": "TRACE-Global",
    "gapit": "GAPIT3",
    "rmvp": "rMVP",
}


def plot_null_qq(
    qq_path: str | None = None,
    output_dir: str | None = None,
):
    """QQ plot from null (permuted) phenotypes."""
    if qq_path is None:
        qq_path = ROOT / "benchmarks" / "simulation" / "sim_summary" / "null_qq_data.csv"
    if output_dir is None:
        output_dir = ROOT / "benchmarks" / "simulation" / "sim_summary"
    output_dir = Path(output_dir)

    if not Path(qq_path).exists():
        print(f"QQ data not found: {qq_path}")
        return

    df = pd.read_csv(qq_path)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df["expected"], df["observed"], s=2, alpha=0.3, color="#2563eb")

    max_val = max(df["expected"].max(), df["observed"].max()) + 0.5
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1, alpha=0.7)

    ax.set_xlabel("Expected $-\\log_{10}(p)$")
    ax.set_ylabel("Observed $-\\log_{10}(p)$")
    ax.set_title("QQ Plot — Null Calibration (100 Permutations)")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect("equal")

    # Add lambda GC annotation
    lambda_path = ROOT / "benchmarks" / "simulation" / "sim_summary" / "null_lambda_gc.csv"
    if Path(lambda_path).exists():
        lambdas = pd.read_csv(lambda_path)["lambda_gc"].values
        ax.text(
            0.05, 0.95,
            f"$\\lambda_{{GC}}$ = {np.mean(lambdas):.3f} $\\pm$ {np.std(lambdas):.3f}\n"
            f"(n={len(lambdas)} permutations)",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    fig.tight_layout()
    out_path = output_dir / "null_qq_plot.png"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(PAPER_FIGURES / "figure_s1a_null_qq.png", bbox_inches="tight")
    fig.savefig(PAPER_FIGURES / "figure_s1a_null_qq.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path} and .pdf")


def plot_null_lambda_boxplot(
    lambda_path: str | None = None,
    output_dir: str | None = None,
):
    """Boxplot of lambda GC across null permutations."""
    if lambda_path is None:
        lambda_path = ROOT / "benchmarks" / "simulation" / "sim_summary" / "null_lambda_gc.csv"
    if output_dir is None:
        output_dir = ROOT / "benchmarks" / "simulation" / "sim_summary"
    output_dir = Path(output_dir)

    if not Path(lambda_path).exists():
        print(f"Lambda data not found: {lambda_path}")
        return

    lambdas = pd.read_csv(lambda_path)["lambda_gc"].values

    fig, ax = plt.subplots(figsize=(3, 5))
    bp = ax.boxplot(lambdas, patch_artist=True)
    bp["boxes"][0].set_facecolor("#dbeafe")
    bp["boxes"][0].set_edgecolor("#2563eb")
    bp["medians"][0].set_color("#dc2626")

    ax.axhline(1.0, color="#16a34a", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylabel("$\\lambda_{GC}$")
    ax.set_title("Genomic Inflation\nUnder Null")
    ax.set_xticks([])

    fig.tight_layout()
    out_path = output_dir / "null_lambda_boxplot.png"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(PAPER_FIGURES / "figure_s1b_null_lambda.png", bbox_inches="tight")
    fig.savefig(PAPER_FIGURES / "figure_s1b_null_lambda.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path} and .pdf")


def plot_window_sensitivity(
    ws_path: str | None = None,
    output_dir: str | None = None,
):
    """Line plot of power vs TP-window size, faceted by scenario."""
    if ws_path is None:
        ws_path = ROOT / "benchmarks" / "simulation" / "sim_summary" / "window_sensitivity.csv"
    if output_dir is None:
        output_dir = ROOT / "benchmarks" / "simulation" / "sim_summary"
    output_dir = Path(output_dir)

    if not Path(ws_path).exists():
        print(f"Window sensitivity data not found: {ws_path}")
        return

    df = pd.read_csv(ws_path)
    if len(df) == 0:
        print("Empty window sensitivity data!")
        return

    h2_levels = sorted(df["h2"].unique())
    qtn_levels = sorted(df["n_qtns"].unique())
    tools = [t for t in TOOL_COLORS if t in df["tool"].unique()]

    fig, axes = plt.subplots(
        len(h2_levels), len(qtn_levels),
        figsize=(4 * len(qtn_levels), 3.5 * len(h2_levels)),
        squeeze=False,
    )

    for i, h2 in enumerate(h2_levels):
        for j, nq in enumerate(qtn_levels):
            ax = axes[i][j]
            sub = df[(df["h2"] == h2) & (df["n_qtns"] == nq)]

            for tool in tools:
                tsub = sub[sub["tool"] == tool].sort_values("window_kb")
                if len(tsub) == 0:
                    continue

                ax.plot(
                    tsub["window_kb"], tsub["power_mean"],
                    color=TOOL_COLORS[tool],
                    label=TOOL_LABELS[tool],
                    marker="o", markersize=4, linewidth=1.5,
                )
                ax.fill_between(
                    tsub["window_kb"],
                    tsub["power_ci_lo"],
                    tsub["power_ci_hi"],
                    color=TOOL_COLORS[tool],
                    alpha=0.15,
                )

            ax.set_xlabel("TP Window (kb)")
            ax.set_ylabel("Power")
            ax.set_title(f"h$^2$={h2}, QTNs={nq}")
            ax.set_ylim(-0.05, 1.05)
            ax.set_xticks([50, 100, 250, 500])

    # Collect legend handles from last axes
    ws_handles, ws_labels = axes[0][-1].get_legend_handles_labels()
    fig.legend(ws_handles, ws_labels, fontsize=8, loc="upper center",
               ncol=len(tools), bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(
        "Power Sensitivity to TP-Classification Window Size",
        fontsize=14, y=1.05,
    )
    fig.tight_layout()
    out_path = output_dir / "window_sensitivity.png"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(PAPER_FIGURES / "figure_s2_window_sensitivity.png", bbox_inches="tight")
    fig.savefig(PAPER_FIGURES / "figure_s2_window_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path} and .pdf")


if __name__ == "__main__":
    print("Generating simulation plots...")
    plot_null_qq()
    plot_null_lambda_boxplot()
    plot_window_sensitivity()
    print("Done.")
