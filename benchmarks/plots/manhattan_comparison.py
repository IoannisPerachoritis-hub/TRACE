"""Cross-tool Manhattan plot comparison for tomato locule number.

Generates a three-panel stacked Manhattan plot (TRACE, GAPIT3, rMVP)
showing concordant detection of chr2 (lc/SlWUS) and chr11 (fas/SlCLV3)
signals. Supplementary Figure S10 for the TRACE Application Note.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.pub_theme import (
    apply_matplotlib_theme, MANHATTAN_COLORS, SIG_LINE_COLOR, PALETTE,
)
from gwas.plotting import compute_cumulative_positions

apply_matplotlib_theme()

# ── Data paths ────────────────────────────────────────────────
TRACE_CSV = (ROOT / "benchmarks" / "qc_data" / "tomato_locule_number"
             / "platform_GWAS_locule_number.csv")
GAPIT_CSV = (ROOT / "benchmarks" / "results" / "gapit"
             / "tomato_locule_number" / "gapit_results_standardized.csv")
RMVP_CSV = (ROOT / "benchmarks" / "results" / "rmvp"
            / "tomato_locule_number" / "rmvp_results_standardized.csv")

# ── Thresholds ────────────────────────────────────────────────
N_SNPS = 43_974
BONF_P = 0.05 / N_SNPS
BONF_LOD = -np.log10(BONF_P)
SUGGEST_LOD = 4.0  # -log10(1e-4)

# ── Key loci ──────────────────────────────────────────────────
CHR2_LEAD = "SL25ch02p47366168"
CHR11_LEAD = "SL25ch11p55055059"


def load_all() -> dict[str, pd.DataFrame]:
    """Load and standardize results from all three tools."""
    trace = pd.read_csv(TRACE_CSV)
    trace = trace[["SNP", "Chr", "Pos", "PValue"]].dropna(subset=["PValue"])

    gapit = pd.read_csv(GAPIT_CSV).dropna(subset=["PValue"])
    rmvp = pd.read_csv(RMVP_CSV).dropna(subset=["PValue"])

    return {"TRACE": trace, "GAPIT3": gapit, "rMVP": rmvp}


def _chr_cumpos_bounds(df, chr_col="Chr"):
    """Return dict mapping chromosome → (min_cumpos, max_cumpos)."""
    bounds = {}
    for ch in df[chr_col].unique():
        mask = df[chr_col] == ch
        cmin = df.loc[mask, "CumPos"].min()
        cmax = df.loc[mask, "CumPos"].max()
        bounds[str(ch)] = (cmin, cmax)
    return bounds


def plot_manhattan_comparison(output_path: Path | None = None):
    """Create three-panel stacked Manhattan comparison figure."""
    if output_path is None:
        output_path = ROOT / "benchmarks" / "plots" / "manhattan_comparison.png"

    datasets = load_all()

    # Compute cumulative positions from TRACE (same coordinates for all)
    ref_df, tick_positions, tick_labels = compute_cumulative_positions(
        datasets["TRACE"].copy()
    )
    cumpos_map = ref_df.set_index("SNP")["CumPos"]

    # Apply cumulative positions to all datasets and compute -log10(p)
    for name in datasets:
        df = datasets[name]
        df["CumPos"] = df["SNP"].map(cumpos_map)
        df["-log10p"] = -np.log10(df["PValue"].clip(1e-300))
        datasets[name] = df.dropna(subset=["CumPos"])

    # Global y-axis max
    ymax_global = max(df["-log10p"].max() for df in datasets.values())
    ymax = ymax_global * 1.08

    # Chromosome shading bounds (from reference)
    chr_bounds = _chr_cumpos_bounds(ref_df)

    # ── Figure ────────────────────────────────────────────────
    fig, axes = plt.subplots(
        3, 1, figsize=(10, 9), sharex=True,
        gridspec_kw={"hspace": 0.06},
    )

    tools = ["TRACE", "GAPIT3", "rMVP"]
    panel_labels = ["A", "B", "C"]

    for idx, (tool, label) in enumerate(zip(tools, panel_labels)):
        ax = axes[idx]
        df = datasets[tool]

        # Chromosome ordering
        def chr_key(x):
            x = str(x).replace("chr", "").replace("Chr", "")
            try:
                return (0, int(x))
            except ValueError:
                return (1, x)

        chroms_unique = sorted(df["Chr"].astype(str).unique(), key=chr_key)

        # Chr2 and Chr11 shading
        for highlight_chr, color in [("2", "#FFE0B2"), ("11", "#E1BEE7")]:
            if highlight_chr in chr_bounds:
                lo, hi = chr_bounds[highlight_chr]
                pad = (hi - lo) * 0.02
                ax.axvspan(lo - pad, hi + pad, alpha=0.35, color=color,
                           zorder=0, linewidth=0)

        # Scatter by chromosome
        for i, ch in enumerate(chroms_unique):
            mask = df["Chr"].astype(str) == ch
            ax.scatter(
                df.loc[mask, "CumPos"], df.loc[mask, "-log10p"],
                s=6, color=MANHATTAN_COLORS[i % 2],
                zorder=1, rasterized=True,
            )

        # Bonferroni threshold (solid red)
        ax.axhline(BONF_LOD, linestyle="-", color=SIG_LINE_COLOR,
                    linewidth=0.9, zorder=2, alpha=0.8)

        # Suggestive threshold (dashed grey)
        ax.axhline(SUGGEST_LOD, linestyle="--", color=PALETTE["grey"],
                    linewidth=0.7, zorder=2, alpha=0.6)

        # Highlight significant SNPs above Bonferroni
        sig_mask = df["-log10p"] >= BONF_LOD
        if sig_mask.any():
            ax.scatter(
                df.loc[sig_mask, "CumPos"], df.loc[sig_mask, "-log10p"],
                s=16, color=SIG_LINE_COLOR, edgecolors="white",
                linewidths=0.3, zorder=3,
            )

        # Annotate chr2 lead SNP
        lead2 = df[df["SNP"] == CHR2_LEAD]
        if not lead2.empty:
            lx = lead2["CumPos"].iloc[0]
            ly = lead2["-log10p"].iloc[0]
            ax.annotate(
                CHR2_LEAD, xy=(lx, ly), xytext=(lx, ly + ymax * 0.06),
                fontsize=7, fontweight="bold", ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
            )

        # Annotate chr11 SNP (TRACE lead; #2 in GAPIT3, #3 in rMVP)
        lead11 = df[df["SNP"] == CHR11_LEAD]
        if not lead11.empty:
            lx = lead11["CumPos"].iloc[0]
            ly = lead11["-log10p"].iloc[0]
            ax.annotate(
                CHR11_LEAD, xy=(lx, ly), xytext=(lx, ly + ymax * 0.06),
                fontsize=7, fontweight="bold", ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
            )

        # Panel label
        ax.text(
            0.01, 0.95, f"{label}  {tool}",
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="top", ha="left",
        )

        ax.set_ylim(0, ymax)
        ax.set_ylabel(r"$-\log_{10}(p)$")

        # Remove x tick labels for top panels
        if idx < 2:
            ax.tick_params(axis="x", labelbottom=False)

    # Bottom panel: chromosome tick labels
    axes[2].set_xticks(tick_positions)
    axes[2].set_xticklabels(tick_labels, fontsize=8)
    axes[2].set_xlabel("Chromosome")

    # Legend for threshold lines (top panel — TRACE)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=SIG_LINE_COLOR, linewidth=0.9, linestyle="-",
               label=f"Bonferroni (p = {BONF_P:.2e})"),
        Line2D([0], [0], color=PALETTE["grey"], linewidth=0.7, linestyle="--",
               label="Suggestive (p = 1e-4)"),
    ]
    axes[0].legend(handles=legend_elements, loc="upper right", fontsize=8,
                   framealpha=0.9)

    fig.suptitle(
        "Cross-tool Manhattan comparison — Tomato locule number (MLM)",
        fontsize=13, fontweight="bold", y=0.995,
    )

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()

    # ── Save ──────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=600)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")

    # Also save to paper/figures/
    paper_dir = ROOT / "paper" / "figures"
    paper_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper_dir / "figure_s10_manhattan_comparison.png",
                bbox_inches="tight", dpi=600)
    fig.savefig(paper_dir / "figure_s10_manhattan_comparison.pdf",
                bbox_inches="tight")

    plt.close(fig)
    print(f"Saved: {output_path}")
    print(f"Saved: {paper_dir / 'figure_s10_manhattan_comparison.png'}")

    # Print key SNP p-values across tools
    print("\nKey SNP p-values:")
    print(f"  {'Tool':<8s} {'chr2 lead (lc)':<20s} {'chr11 lead (fas)':<20s}")
    print("-" * 50)
    for tool in tools:
        df = datasets[tool]
        p2 = df.loc[df["SNP"] == CHR2_LEAD, "PValue"]
        p11 = df.loc[df["SNP"] == CHR11_LEAD, "PValue"]
        p2_str = f"{p2.iloc[0]:.2e}" if not p2.empty else "N/A"
        p11_str = f"{p11.iloc[0]:.2e}" if not p11.empty else "N/A"
        print(f"  {tool:<8s} {p2_str:<20s} {p11_str:<20s}")
    print(f"\n  Bonferroni threshold: {BONF_P:.2e}")


if __name__ == "__main__":
    plot_manhattan_comparison()
