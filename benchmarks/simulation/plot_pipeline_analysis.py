"""Plots for the QTN-level recovery analysis.

Stacked bar charts of LOCO-exclusive vs competitor-exclusive QTNs,
effect-size-stratified recovery curves, and polygenic detection
rate comparison.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from utils.pub_theme import apply_matplotlib_theme
apply_matplotlib_theme()

SIM_SUMMARY = ROOT / "benchmarks" / "simulation" / "sim_summary"
OUT_DIR = SIM_SUMMARY

TOOL_COLORS = {
    "platform_loco": "#1f77b4",
    "platform_global": "#aec7e8",
    "gapit": "#ff7f0e",
    "rmvp": "#2ca02c",
    "farmcpu_loco": "#9467bd",
    "gapit_farmcpu": "#d62728",
}
TOOL_LABELS = {
    "platform_loco": "TRACE-LOCO",
    "platform_global": "TRACE-Global",
    "gapit": "GAPIT3",
    "rmvp": "rMVP",
}


def plot_qtn_set_differences(df: pd.DataFrame, out_dir: Path = OUT_DIR):
    """Stacked bar chart of QTN set differences per scenario.

    For each scenario, shows: LOCO-only, Shared, Competitor-only, Neither.
    Faceted by comparison (vs GAPIT3, vs rMVP).
    """
    comparisons = [
        ("TRACE-LOCO vs GAPIT3", "GAPIT3"),
        ("TRACE-LOCO vs rMVP", "rMVP"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, (comp_label, short) in zip(axes, comparisons):
        sub = df[df["comparison"] == comp_label].copy()
        if len(sub) == 0:
            continue

        # Aggregate per scenario
        agg = sub.groupby(["h2", "n_qtns"]).agg({
            "n_only_a": "mean",
            "n_only_b": "mean",
            "n_both": "mean",
            "n_neither": "mean",
        }).reset_index()

        # Sort by h2 then n_qtns
        agg = agg.sort_values(["h2", "n_qtns"])
        labels = [f"h$^2$={r['h2']}\n{int(r['n_qtns'])} QTNs" for _, r in agg.iterrows()]
        x = np.arange(len(agg))
        w = 0.6

        # Stacked bars
        ax.bar(x, agg["n_only_a"], w, label="TRACE-LOCO-only", color="#1f77b4")
        ax.bar(x, agg["n_both"], w, bottom=agg["n_only_a"],
               label="Shared", color="#7fbf7f")
        ax.bar(x, agg["n_only_b"], w,
               bottom=agg["n_only_a"] + agg["n_both"],
               label="Competitor-only", color="#ff7f0e")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(f"vs {short}", fontweight="bold")
        ax.set_ylabel("Mean QTNs detected" if ax == axes[0] else "")

    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("QTN-Level Detection: Set Differences",
                 fontweight="bold", y=1.02)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"qtn_set_differences.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: qtn_set_differences.png/pdf")


def plot_effect_size_recovery(df: pd.DataFrame, out_dir: Path = OUT_DIR):
    """Recovery rate vs effect size for MLM tools."""
    tools = ["platform_loco", "platform_global", "gapit", "rmvp"]
    # Focus on 5-QTN scenarios for clarity
    sub = df[(df["tool"].isin(tools)) & (df["n_qtns"] == 5)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    h2_vals = [0.3, 0.5, 0.8]

    for ax, h2 in zip(axes, h2_vals):
        h2_sub = sub[sub["h2"] == h2]

        for tool in tools:
            t_sub = h2_sub[h2_sub["tool"] == tool]
            if len(t_sub) == 0:
                continue

            # Order bins
            bin_order = ["<0.1", "0.1-0.3", "0.3-0.5", "0.5-1.0", "1.0-2.0", ">2.0"]
            t_sub = t_sub.set_index("abs_beta_bin").reindex(bin_order).dropna(subset=["recovery_rate"])

            ax.plot(
                range(len(t_sub)), t_sub["recovery_rate"],
                marker="o", markersize=4, linewidth=1.5,
                color=TOOL_COLORS[tool],
                label=TOOL_LABELS[tool],
            )

            if len(t_sub) > 0:
                ax.set_xticks(range(len(t_sub)))
                ax.set_xticklabels(t_sub.index, fontsize=7, rotation=30)

        ax.set_title(f"h$^2$ = {h2}", fontweight="bold")
        ax.set_xlabel("|$\\beta$| (effect size)")
        ax.set_ylabel("Recovery rate" if ax == axes[0] else "")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("QTN Recovery by Effect Size (5-QTN Scenarios)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"qtn_effect_size_recovery.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: qtn_effect_size_recovery.png/pdf")


def plot_pve_recovery(df: pd.DataFrame, out_dir: Path = OUT_DIR):
    """Recovery rate vs per-QTN PVE for MLM tools.

    Same as plot_effect_size_recovery but with a variance-explained
    x-axis that normalizes across beta and MAF.
    """
    tools = ["platform_loco", "platform_global", "gapit", "rmvp"]
    # Focus on 5-QTN scenarios for clarity
    sub = df[(df["tool"].isin(tools)) & (df["n_qtns"] == 5)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    h2_vals = [0.3, 0.5, 0.8]
    bin_order = ["<0.5%", "0.5-1%", "1-2%", "2-5%", "5-10%", "10-20%", ">20%"]

    for ax, h2 in zip(axes, h2_vals):
        h2_sub = sub[sub["h2"] == h2]

        for tool in tools:
            t_sub = h2_sub[h2_sub["tool"] == tool]
            if len(t_sub) == 0:
                continue

            t_sub = t_sub.set_index("pve_bin").reindex(bin_order).dropna(subset=["recovery_rate"])

            ax.plot(
                range(len(t_sub)), t_sub["recovery_rate"],
                marker="o", markersize=4, linewidth=1.5,
                color=TOOL_COLORS[tool],
                label=TOOL_LABELS[tool],
            )

            if len(t_sub) > 0:
                ax.set_xticks(range(len(t_sub)))
                ax.set_xticklabels(t_sub.index, fontsize=7, rotation=30)

        ax.set_title(f"h$^2$ = {h2}", fontweight="bold")
        ax.set_xlabel("Per-QTN PVE (% phenotypic variance)")
        ax.set_ylabel("Recovery rate" if ax == axes[0] else "")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("QTN Recovery by Per-QTN Variance Explained (5-QTN Scenarios)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"qtn_pve_recovery.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: qtn_pve_recovery.png/pdf")


def plot_exclusive_detection_rate(df: pd.DataFrame, out_dir: Path = OUT_DIR):
    """Bar chart: fraction of reps where LOCO finds extra QTNs.

    Per scenario, shows % of reps where LOCO detects at least one
    QTN that the competitor misses entirely.
    """
    comp = "TRACE-LOCO vs GAPIT3"
    sub = df[df["comparison"] == comp].copy()

    agg = sub.groupby(["h2", "n_qtns"]).apply(
        lambda g: pd.Series({
            "pct_reps_loco_exclusive": (g["n_only_a"] > 0).mean(),
            "pct_reps_gapit_exclusive": (g["n_only_b"] > 0).mean(),
            "mean_loco_exclusive": g["n_only_a"].mean(),
        }),
        include_groups=False,
    ).reset_index()
    agg = agg.sort_values(["h2", "n_qtns"])

    labels = [f"h$^2$={r['h2']}, {int(r['n_qtns'])}Q"
              for _, r in agg.iterrows()]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(agg))
    w = 0.35

    ax.bar(x - w/2, agg["pct_reps_loco_exclusive"], w,
           label="LOCO finds extra QTNs", color="#1f77b4")
    ax.bar(x + w/2, agg["pct_reps_gapit_exclusive"], w,
           label="GAPIT3 finds extra QTNs", color="#ff7f0e")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Fraction of replicates")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend()
    ax.set_title("How Often Does Each Tool Find Exclusive QTNs? (vs GAPIT3)",
                 fontweight="bold")
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"qtn_exclusive_detection.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: qtn_exclusive_detection.png/pdf")


def plot_polygenic_advantage(df_recovery: pd.DataFrame, out_dir: Path = OUT_DIR):
    """Mean QTNs detected vs number of simulated QTNs.

    Compares LOCO vs competitors at h2=0.8 across increasing
    QTN counts (5, 15, 50).
    """
    tools = ["platform_loco", "gapit", "rmvp"]
    sub = df_recovery[(df_recovery["h2"] == 0.8) & (df_recovery["tool"].isin(tools))]

    fig, ax = plt.subplots(figsize=(6, 4))

    for tool in tools:
        t = sub[sub["tool"] == tool]
        agg = t.groupby("n_qtns").agg(
            power=("detected", "mean"),
        ).reset_index()
        agg["n_qtns_detected"] = agg["power"] * agg["n_qtns"]

        ax.plot(agg["n_qtns"], agg["n_qtns_detected"],
                marker="o", linewidth=2, markersize=6,
                color=TOOL_COLORS[tool],
                label=TOOL_LABELS[tool])

    ax.set_xlabel("Number of simulated QTNs")
    ax.set_ylabel("Mean QTNs detected")
    ax.set_xticks([5, 15, 50])
    ax.legend()
    ax.set_title("QTNs Detected vs Genetic Complexity (h$^2$ = 0.8)",
                 fontweight="bold")
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"qtn_polygenic_advantage.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: qtn_polygenic_advantage.png/pdf")


def main():
    print("Generating QTN Recovery Figures")
    print("=" * 40)

    # Load data
    set_diff = pd.read_csv(SIM_SUMMARY / "qtn_set_differences.csv")
    es_recovery = pd.read_csv(SIM_SUMMARY / "qtn_effect_size_recovery.csv")
    qtn_recovery = pd.read_csv(SIM_SUMMARY / "qtn_recovery.csv")

    plot_qtn_set_differences(set_diff)
    plot_effect_size_recovery(es_recovery)
    pve_recovery = pd.read_csv(SIM_SUMMARY / "qtn_pve_recovery.csv")
    plot_pve_recovery(pve_recovery)
    plot_exclusive_detection_rate(set_diff)
    plot_polygenic_advantage(qtn_recovery)

    print("\nDone! All figures saved to sim_summary/")


if __name__ == "__main__":
    main()
