"""FarmCPU concordance benchmark: TRACE FarmCPU vs GAPIT3 FarmCPU.

Orchestrates:
1. Running GAPIT3 FarmCPU on all 4 real datasets
2. Loading TRACE FarmCPU + GAPIT3 FarmCPU results
3. Computing concordance metrics (Spearman, Jaccard, top-hit overlap)
4. Generating p-value scatter plots
5. Saving comparison table
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.pub_theme import apply_matplotlib_theme
apply_matplotlib_theme()

QC_DIR = ROOT / "benchmarks" / "qc_data"
GAPIT_FARMCPU_DIR = ROOT / "benchmarks" / "results" / "gapit_farmcpu"

RSCRIPT = r"C:\Program Files\R\R-4.5.2\bin\Rscript.exe"

RUNS = {
    "pepper_FWe": {"trait": "FWe", "species": "Pepper"},
    "pepper_BX": {"trait": "BX", "species": "Pepper"},
    "tomato_weight_g": {"trait": "weight_g", "species": "Tomato"},
    "tomato_locule_number": {"trait": "locule_number", "species": "Tomato"},
}


# ── Data loading ─────────────────────────────────────────────

def load_trace_farmcpu(run_name: str) -> pd.DataFrame | None:
    """Load TRACE FarmCPU results from extracted CSV."""
    qc = QC_DIR / run_name
    csvs = list(qc.glob("platform_GWAS_FarmCPU_*.csv"))
    if not csvs:
        return None
    df = pd.read_csv(csvs[0])
    cols = ["SNP", "Chr", "Pos", "PValue"]
    for c in cols:
        if c not in df.columns:
            return None
    return df[cols].dropna(subset=["PValue"])


def load_gapit_farmcpu(run_name: str) -> pd.DataFrame | None:
    """Load standardized GAPIT3 FarmCPU results."""
    f = GAPIT_FARMCPU_DIR / run_name / "gapit_farmcpu_results_standardized.csv"
    if not f.exists():
        return None
    return pd.read_csv(f).dropna(subset=["PValue"])


# ── Metrics ──────────────────────────────────────────────────

def lambda_gc(pvalues: np.ndarray) -> float:
    chisq = stats.chi2.ppf(1 - pvalues, df=1)
    return float(np.median(chisq) / stats.chi2.ppf(0.5, df=1))


def jaccard(set_a: set, set_b: set) -> float:
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    union = set_a | set_b
    if len(union) == 0:
        return 0.0
    return len(set_a & set_b) / len(union)


# ── Run GAPIT3 FarmCPU ───────────────────────────────────────

def run_gapit_farmcpu(run_name: str) -> bool:
    """Run GAPIT3 FarmCPU for a single dataset. Returns True on success."""
    out_dir = GAPIT_FARMCPU_DIR / run_name
    result_file = out_dir / "gapit_farmcpu_results_standardized.csv"

    if result_file.exists():
        print(f"  [skip] {run_name}: results already exist")
        return True

    print(f"  Running GAPIT3 FarmCPU on {run_name}...")
    cmd = [RSCRIPT, str(ROOT / "benchmarks" / "run_gapit_farmcpu.R"), run_name]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            cwd=str(ROOT),
        )
        if result.returncode != 0:
            print(f"  [FAIL] {run_name}: return code {result.returncode}")
            if result.stderr:
                # Print last 20 lines of stderr
                lines = result.stderr.strip().split("\n")
                for line in lines[-20:]:
                    print(f"    {line}")
            return False
        print(f"  [OK] {run_name}")
        return True
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {run_name}: exceeded 10 min")
        return False
    except FileNotFoundError:
        print(f"  [ERROR] Rscript not found at {RSCRIPT}")
        return False


def run_all_gapit_farmcpu():
    """Run GAPIT3 FarmCPU on all 4 datasets."""
    print("\n" + "=" * 60)
    print("  Phase 1: Running GAPIT3 FarmCPU")
    print("=" * 60)

    results = {}
    for run_name in RUNS:
        results[run_name] = run_gapit_farmcpu(run_name)

    ok = sum(results.values())
    print(f"\n  {ok}/{len(RUNS)} datasets completed successfully")
    return results


# ── Concordance analysis ─────────────────────────────────────

def compare_run(run_name: str) -> dict:
    """Compare TRACE FarmCPU vs GAPIT3 FarmCPU for a single dataset."""
    info = RUNS[run_name]
    result = {"run": run_name, "trait": info["trait"], "species": info["species"]}

    trace = load_trace_farmcpu(run_name)
    gapit = load_gapit_farmcpu(run_name)

    if trace is None:
        print(f"  {run_name}: TRACE FarmCPU results not found")
        result["status"] = "missing_trace"
        return result
    if gapit is None:
        print(f"  {run_name}: GAPIT3 FarmCPU results not found")
        result["status"] = "missing_gapit"
        return result

    result["status"] = "ok"
    result["n_snps_TRACE"] = len(trace)
    result["n_snps_GAPIT3"] = len(gapit)

    # Lambda GC
    result["lambda_TRACE"] = round(lambda_gc(trace["PValue"].values), 4)
    result["lambda_GAPIT3"] = round(lambda_gc(gapit["PValue"].values), 4)

    # Significant SNPs (Bonferroni)
    n_trace = len(trace)
    n_gapit = len(gapit)
    sig_trace = set(trace.loc[trace["PValue"] < 0.05 / n_trace, "SNP"])
    sig_gapit = set(gapit.loc[gapit["PValue"] < 0.05 / n_gapit, "SNP"])
    result["sig_bonf_TRACE"] = len(sig_trace)
    result["sig_bonf_GAPIT3"] = len(sig_gapit)
    result["jaccard_sig"] = round(jaccard(sig_trace, sig_gapit), 4)

    # Merge on SNP for rank correlation
    merged = pd.merge(
        trace[["SNP", "PValue"]].rename(columns={"PValue": "P_trace"}),
        gapit[["SNP", "PValue"]].rename(columns={"PValue": "P_gapit"}),
        on="SNP", how="inner",
    )
    result["n_shared_snps"] = len(merged)

    if len(merged) > 10:
        rho, pval = stats.spearmanr(merged["P_trace"], merged["P_gapit"])
        result["spearman_rho"] = round(rho, 4)
    else:
        result["spearman_rho"] = np.nan

    # Top-10 and top-20 hit overlap
    for top_n in [10, 20]:
        top_trace = set(trace.nsmallest(top_n, "PValue")["SNP"])
        top_gapit = set(gapit.nsmallest(top_n, "PValue")["SNP"])
        overlap = top_trace & top_gapit
        result[f"top{top_n}_overlap"] = len(overlap)

    return result


def run_concordance_analysis():
    """Compare all 4 datasets and save results."""
    print("\n" + "=" * 60)
    print("  Phase 2: FarmCPU Concordance Analysis")
    print("=" * 60)

    all_results = []
    for run_name in RUNS:
        info = RUNS[run_name]
        print(f"\n  {run_name} ({info['species']} — {info['trait']})")
        result = compare_run(run_name)
        all_results.append(result)

        if result.get("status") == "ok":
            print(f"    Lambda: TRACE={result['lambda_TRACE']:.4f}, "
                  f"GAPIT3={result['lambda_GAPIT3']:.4f}")
            print(f"    Spearman rho: {result.get('spearman_rho', 'N/A')}")
            print(f"    Top-10 overlap: {result.get('top10_overlap', 'N/A')}/10")
            print(f"    Sig (Bonf): TRACE={result['sig_bonf_TRACE']}, "
                  f"GAPIT3={result['sig_bonf_GAPIT3']}")

    df = pd.DataFrame(all_results)
    out_path = ROOT / "benchmarks" / "results" / "farmcpu_comparison_table.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    return df


# ── Scatter plots ────────────────────────────────────────────

def plot_farmcpu_scatter():
    """Generate 2x2 grid of TRACE FarmCPU vs GAPIT3 FarmCPU scatter plots."""
    print("\n" + "=" * 60)
    print("  Phase 3: Generating FarmCPU Scatter Plots")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes_flat = axes.flatten()

    for idx, run_name in enumerate(RUNS):
        info = RUNS[run_name]
        ax = axes_flat[idx]

        trace = load_trace_farmcpu(run_name)
        gapit = load_gapit_farmcpu(run_name)

        if trace is None or gapit is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_title(f"{info['species']} — {info['trait']}")
            continue

        merged = pd.merge(
            trace[["SNP", "PValue"]].rename(columns={"PValue": "P_trace"}),
            gapit[["SNP", "PValue"]].rename(columns={"PValue": "P_gapit"}),
            on="SNP", how="inner",
        )

        if len(merged) < 10:
            ax.text(0.5, 0.5, "<10 shared\nSNPs", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
            ax.set_title(f"{info['species']} — {info['trait']}")
            continue

        x = -np.log10(merged["P_trace"].values.clip(1e-300))
        y = -np.log10(merged["P_gapit"].values.clip(1e-300))

        ax.scatter(x, y, s=2, alpha=0.2, color="#2563eb", rasterized=True)

        # Identity line
        max_val = max(x.max(), y.max()) * 1.05
        ax.plot([0, max_val], [0, max_val], "r--", linewidth=0.8, alpha=0.5)

        # Spearman correlation
        rho, _ = stats.spearmanr(merged["P_trace"], merged["P_gapit"])
        ax.text(
            0.05, 0.95,
            f"$\\rho$ = {rho:.3f}\nn = {len(merged):,}",
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_aspect("equal")
        ax.set_xlabel("$-\\log_{10}(p)$ TRACE FarmCPU", fontsize=8)
        ax.set_ylabel("$-\\log_{10}(p)$ GAPIT3 FarmCPU", fontsize=8)
        ax.set_title(f"{info['species']} — {info['trait']}", fontsize=10)

    fig.suptitle("FarmCPU Concordance: TRACE vs GAPIT3", fontsize=13, y=1.01)
    fig.tight_layout()

    out_png = ROOT / "benchmarks" / "plots" / "farmcpu_concordance.png"
    out_pdf = ROOT / "benchmarks" / "plots" / "farmcpu_concordance.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")

    # Also save to paper/figures/
    paper_fig = ROOT / "paper" / "figures"
    paper_fig.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper_fig / "figure_s9_farmcpu_concordance.png",
                dpi=300, bbox_inches="tight")
    fig.savefig(paper_fig / "figure_s9_farmcpu_concordance.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png}")


# ── Summary ──────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    """Print a readable summary table."""
    print("\n" + "=" * 60)
    print("  FARMCPU CONCORDANCE SUMMARY")
    print("=" * 60)

    ok = df[df["status"] == "ok"]
    if ok.empty:
        print("  No successful comparisons.")
        return

    print(f"\n  {'Dataset':<25s} {'Spearman':>8s} {'Top-10':>6s} {'Top-20':>6s} "
          f"{'lam_TRACE':>9s} {'lam_GAPIT':>9s}")
    print("-" * 72)
    for _, row in ok.iterrows():
        print(f"  {row['run']:<25s} {row['spearman_rho']:>8.4f} "
              f"{row['top10_overlap']:>6.0f} {row['top20_overlap']:>6.0f} "
              f"{row['lambda_TRACE']:>9.4f} {row['lambda_GAPIT3']:>9.4f}")

    rho_vals = ok["spearman_rho"].dropna()
    if len(rho_vals) > 0:
        print(f"\n  Spearman rho range: {rho_vals.min():.3f} to {rho_vals.max():.3f}")


# ── Main ─────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  FarmCPU Concordance Benchmark")
    print("  TRACE FarmCPU vs GAPIT3 FarmCPU")
    print("=" * 60)

    # Phase 1: Run GAPIT3 FarmCPU
    gapit_results = run_all_gapit_farmcpu()

    # Phase 2: Concordance analysis
    df = run_concordance_analysis()

    # Phase 3: Scatter plots
    plot_farmcpu_scatter()

    # Summary
    print_summary(df)

    print("\n\nDone.")


if __name__ == "__main__":
    main()
