"""Generate p-value concordance scatter plots across tools.

Creates a 4x3 grid of -log10(p) scatter plots for all dataset/tool pairs.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.pub_theme import apply_matplotlib_theme
apply_matplotlib_theme()

QC_DIR = ROOT / "benchmarks" / "qc_data"
GAPIT_DIR = ROOT / "benchmarks" / "results" / "gapit"
RMVP_DIR = ROOT / "benchmarks" / "results" / "rmvp"

RUNS = {
    "pepper_FWe": {"trait": "FWe", "species": "Pepper"},
    "pepper_BX": {"trait": "BX", "species": "Pepper"},
    "tomato_weight_g": {"trait": "weight_g", "species": "Tomato"},
    "tomato_locule_number": {"trait": "locule_number", "species": "Tomato"},
}

TOOLS = ["TRACE", "GAPIT3", "rMVP"]
PAIRS = [("TRACE", "GAPIT3"), ("TRACE", "rMVP"), ("GAPIT3", "rMVP")]


def load_results(run_name: str) -> dict[str, pd.DataFrame]:
    """Load standardized results from all tools for a given run."""
    results = {}

    # TRACE
    qc = QC_DIR / run_name
    csvs = list(qc.glob("platform_GWAS_*.csv"))
    mlm_csvs = [c for c in csvs if "MLMM" not in c.name
                and "FarmCPU" not in c.name and "Haplotype" not in c.name]
    if mlm_csvs:
        df = pd.read_csv(mlm_csvs[0])
        results["TRACE"] = df[["SNP", "Chr", "Pos", "PValue"]].dropna(subset=["PValue"])

    # GAPIT3
    gf = GAPIT_DIR / run_name / "gapit_results_standardized.csv"
    if gf.exists():
        results["GAPIT3"] = pd.read_csv(gf).dropna(subset=["PValue"])

    # rMVP
    rf = RMVP_DIR / run_name / "rmvp_results_standardized.csv"
    if rf.exists():
        results["rMVP"] = pd.read_csv(rf).dropna(subset=["PValue"])

    return results


def plot_scatter_grid(output_path: str | None = None):
    """Generate 4x3 grid of -log10(p) scatter plots."""
    if output_path is None:
        output_path = ROOT / "benchmarks" / "plots" / "pvalue_scatter_grid.png"

    fig, axes = plt.subplots(4, 3, figsize=(12, 14))

    for row_idx, run_name in enumerate(RUNS):
        info = RUNS[run_name]
        results = load_results(run_name)

        for col_idx, (tool_a, tool_b) in enumerate(PAIRS):
            ax = axes[row_idx][col_idx]

            if tool_a not in results or tool_b not in results:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="gray")
                ax.set_title(f"{tool_a} vs {tool_b}")
                continue

            df_a = results[tool_a]
            df_b = results[tool_b]

            # Merge on SNP
            merged = pd.merge(
                df_a[["SNP", "PValue"]].rename(columns={"PValue": "P_a"}),
                df_b[["SNP", "PValue"]].rename(columns={"PValue": "P_b"}),
                on="SNP", how="inner",
            )

            if len(merged) < 10:
                ax.text(0.5, 0.5, f"<10 shared\nSNPs", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="gray")
                continue

            x = -np.log10(merged["P_a"].values.clip(1e-300))
            y = -np.log10(merged["P_b"].values.clip(1e-300))

            ax.scatter(x, y, s=1, alpha=0.15, color="#2563eb", rasterized=True)

            # Identity line
            max_val = max(x.max(), y.max()) * 1.05
            ax.plot([0, max_val], [0, max_val], "r--", linewidth=0.8, alpha=0.5)

            # Spearman correlation
            rho, pval = stats.spearmanr(merged["P_a"], merged["P_b"])
            ax.text(
                0.05, 0.95,
                f"$\\rho$ = {rho:.3f}\nn = {len(merged):,}",
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

            ax.set_xlim(0, max_val)
            ax.set_ylim(0, max_val)
            ax.set_aspect("equal")

            if row_idx == 3:
                ax.set_xlabel(f"$-\\log_{{10}}(p)$ {tool_a}")
            if col_idx == 0:
                ax.set_ylabel(f"$-\\log_{{10}}(p)$ {tool_b}")

            if row_idx == 0:
                ax.set_title(f"{tool_a} vs {tool_b}", fontsize=10)

        # Row label
        axes[row_idx][0].annotate(
            f"{info['species']}\n{info['trait']}",
            xy=(-0.45, 0.5), xycoords="axes fraction",
            fontsize=9, fontweight="bold",
            ha="center", va="center",
            rotation=90,
        )

    fig.suptitle("P-value Concordance Across GWAS Tools", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    fig.savefig(Path(output_path).with_suffix(".pdf"), bbox_inches="tight", dpi=150)
    paper_fig = ROOT / "paper" / "figures"
    fig.savefig(paper_fig / "figure_s4_pvalue_concordance.png", bbox_inches="tight", dpi=150)
    fig.savefig(paper_fig / "figure_s4_pvalue_concordance.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path} and paper/figures/")


def print_concordance_summary():
    """Print concordance summary table."""
    print("\n" + "=" * 70)
    print("  P-VALUE CONCORDANCE SUMMARY")
    print("=" * 70)
    print(f"  {'Dataset':<25s} {'Pair':<22s} {'Spearman':>8s} {'n_shared':>8s}")
    print("-" * 70)

    for run_name, info in RUNS.items():
        results = load_results(run_name)
        for tool_a, tool_b in PAIRS:
            if tool_a not in results or tool_b not in results:
                continue
            merged = pd.merge(
                results[tool_a][["SNP", "PValue"]].rename(columns={"PValue": "P_a"}),
                results[tool_b][["SNP", "PValue"]].rename(columns={"PValue": "P_b"}),
                on="SNP", how="inner",
            )
            if len(merged) > 10:
                rho, _ = stats.spearmanr(merged["P_a"], merged["P_b"])
                print(f"  {run_name:<25s} {tool_a+' vs '+tool_b:<22s} "
                      f"{rho:>8.4f} {len(merged):>8,}")


if __name__ == "__main__":
    print("Generating p-value scatter plots...")
    plot_scatter_grid()
    print_concordance_summary()
