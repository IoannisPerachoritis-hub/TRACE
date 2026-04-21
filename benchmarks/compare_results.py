"""Compare GWAS results across platform, GAPIT3, and rMVP.

Computes:
- Lambda GC comparison
- Jaccard overlap of significant SNPs
- Rank correlation of p-values
- Top-hit concordance
- Runtime comparison
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
QC_DIR = ROOT / "benchmarks" / "qc_data"
GAPIT_DIR = ROOT / "benchmarks" / "results" / "gapit"
RMVP_DIR = ROOT / "benchmarks" / "results" / "rmvp"

RUNS = {
    "pepper_FWe": {"trait": "FWe", "species": "pepper"},
    "pepper_BX": {"trait": "BX", "species": "pepper"},
    "tomato_weight_g": {"trait": "weight_g", "species": "tomato"},
    "tomato_locule_number": {"trait": "locule_number", "species": "tomato"},
}


def load_platform_results(run_name: str) -> pd.DataFrame | None:
    """Load platform MLM results from extracted CSV."""
    qc = QC_DIR / run_name
    csvs = list(qc.glob("platform_GWAS_*.csv"))
    # Prefer the primary MLM (not MLMM/FarmCPU)
    mlm_csvs = [c for c in csvs if "MLMM" not in c.name and "FarmCPU" not in c.name
                 and "Haplotype" not in c.name]
    if not mlm_csvs:
        return None
    df = pd.read_csv(mlm_csvs[0])
    return df[["SNP", "Chr", "Pos", "PValue"]].dropna(subset=["PValue"])


def load_gapit_results(run_name: str) -> pd.DataFrame | None:
    """Load standardized GAPIT results."""
    f = GAPIT_DIR / run_name / "gapit_results_standardized.csv"
    if not f.exists():
        return None
    return pd.read_csv(f).dropna(subset=["PValue"])


def load_rmvp_results(run_name: str) -> pd.DataFrame | None:
    """Load standardized rMVP results."""
    f = RMVP_DIR / run_name / "rmvp_results_standardized.csv"
    if not f.exists():
        return None
    return pd.read_csv(f).dropna(subset=["PValue"])


def lambda_gc(pvalues: np.ndarray) -> float:
    """Compute genomic inflation factor."""
    chisq = stats.chi2.ppf(1 - pvalues, df=1)
    return float(np.median(chisq) / stats.chi2.ppf(0.5, df=1))


def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity coefficient."""
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    union = set_a | set_b
    if len(union) == 0:
        return 0.0
    return len(set_a & set_b) / len(union)


def compare_run(run_name: str) -> dict:
    """Compare all three tools for a single run."""
    info = RUNS[run_name]
    print(f"\n{'='*60}")
    print(f"  {run_name} ({info['species']} — {info['trait']})")
    print(f"{'='*60}")

    platform = load_platform_results(run_name)
    gapit = load_gapit_results(run_name)
    rmvp = load_rmvp_results(run_name)

    result = {"run": run_name, "trait": info["trait"], "species": info["species"]}

    tools = {"TRACE": platform, "GAPIT3": gapit, "rMVP": rmvp}
    available = {k: v for k, v in tools.items() if v is not None}

    if not available:
        print("  No results available.")
        return result

    # ── Lambda GC comparison ──────────────────────────────
    print("\n  Lambda GC:")
    for name, df in available.items():
        lam = lambda_gc(df["PValue"].values)
        result[f"lambda_{name}"] = round(lam, 4)
        print(f"    {name:12s}: {lam:.4f}")

    # ── Number of SNPs tested ─────────────────────────────
    print("\n  SNPs tested:")
    for name, df in available.items():
        result[f"n_snps_{name}"] = len(df)
        print(f"    {name:12s}: {len(df):,}")

    # ── Significant SNPs (Bonferroni) ─────────────────────
    print("\n  Significant SNPs (Bonferroni 0.05/n):")
    sig_sets = {}
    for name, df in available.items():
        n = len(df)
        threshold = 0.05 / n
        sig = set(df.loc[df["PValue"] < threshold, "SNP"].tolist())
        sig_sets[name] = sig
        result[f"sig_bonf_{name}"] = len(sig)
        print(f"    {name:12s}: {len(sig):4d}  (threshold={threshold:.2e})")

    # ── Jaccard overlap ───────────────────────────────────
    pairs = []
    tool_names = list(sig_sets.keys())
    print("\n  Jaccard overlap (significant SNPs):")
    for i in range(len(tool_names)):
        for j in range(i + 1, len(tool_names)):
            a_name, b_name = tool_names[i], tool_names[j]
            j_val = jaccard(sig_sets[a_name], sig_sets[b_name])
            key = f"jaccard_{a_name}_vs_{b_name}"
            result[key] = round(j_val, 4)
            pairs.append((a_name, b_name, j_val))
            print(f"    {a_name} vs {b_name}: {j_val:.4f}")

    # ── P-value rank correlation ──────────────────────────
    print("\n  Spearman rank correlation (p-values):")
    for i in range(len(tool_names)):
        for j in range(i + 1, len(tool_names)):
            a_name, b_name = tool_names[i], tool_names[j]
            df_a, df_b = available[a_name], available[b_name]
            merged = pd.merge(
                df_a[["SNP", "PValue"]].rename(columns={"PValue": "P_a"}),
                df_b[["SNP", "PValue"]].rename(columns={"PValue": "P_b"}),
                on="SNP", how="inner"
            )
            if len(merged) > 10:
                rho, p = stats.spearmanr(merged["P_a"], merged["P_b"])
                key = f"spearman_{a_name}_vs_{b_name}"
                result[key] = round(rho, 4)
                print(f"    {a_name} vs {b_name}: rho={rho:.4f} "
                      f"(n={len(merged):,}, p={p:.2e})")
            else:
                print(f"    {a_name} vs {b_name}: <10 shared SNPs")

    # ── Pearson correlation on -log10(p) ──────────────────
    print("\n  Pearson correlation (-log10 p-values):")
    for i in range(len(tool_names)):
        for j in range(i + 1, len(tool_names)):
            a_name, b_name = tool_names[i], tool_names[j]
            df_a, df_b = available[a_name], available[b_name]
            merged = pd.merge(
                df_a[["SNP", "PValue"]].rename(columns={"PValue": "P_a"}),
                df_b[["SNP", "PValue"]].rename(columns={"PValue": "P_b"}),
                on="SNP", how="inner"
            )
            if len(merged) > 10:
                log_a = -np.log10(merged["P_a"].clip(lower=1e-300))
                log_b = -np.log10(merged["P_b"].clip(lower=1e-300))
                r, p = stats.pearsonr(log_a, log_b)
                key = f"pearson_neglog10p_{a_name}_vs_{b_name}"
                result[key] = round(r, 4)
                print(f"    {a_name} vs {b_name}: r={r:.4f} "
                      f"(n={len(merged):,}, p={p:.2e})")
            else:
                print(f"    {a_name} vs {b_name}: <10 shared SNPs")

    # ── Top-k hit concordance at multiple k ───────────────
    top_k_values = [10, 50, 100, 500]
    for k in top_k_values:
        top_sets_k = {}
        for name, df in available.items():
            top_k = set(df.nsmallest(k, "PValue")["SNP"].tolist())
            top_sets_k[name] = top_k

        if k == 10:
            print(f"\n  Top-{k} hit overlap:")
        for i in range(len(tool_names)):
            for j in range(i + 1, len(tool_names)):
                a_name, b_name = tool_names[i], tool_names[j]
                overlap = top_sets_k[a_name] & top_sets_k[b_name]
                result[f"top{k}_overlap_{a_name}_vs_{b_name}"] = len(overlap)
                if k == 10:
                    print(f"    {a_name} vs {b_name}: {len(overlap)}/{k} shared")

    return result


def load_timing():
    """Load timing results from GAPIT and rMVP."""
    rows = []
    for tool_dir, tool_name in [(GAPIT_DIR, "GAPIT3"), (RMVP_DIR, "rMVP")]:
        for run_name in RUNS:
            tfile = tool_dir / run_name / "timing.csv"
            if tfile.exists():
                rows.append(pd.read_csv(tfile).iloc[0].to_dict())
    return pd.DataFrame(rows) if rows else None


def main():
    print("=" * 60)
    print("  GWAS Benchmarking — 3-Tool Comparison")
    print("  TRACE (FastLMM) vs GAPIT3 vs rMVP")
    print("=" * 60)

    all_results = []
    for run_name in RUNS:
        result = compare_run(run_name)
        all_results.append(result)

    # ── Save comparison table ─────────────────────────────
    comparison = pd.DataFrame(all_results)
    out_path = ROOT / "benchmarks" / "results" / "comparison_table.csv"
    comparison.to_csv(out_path, index=False)
    print(f"\n\nComparison table saved to: {out_path}")

    # ── Timing comparison ─────────────────────────────────
    timing = load_timing()
    if timing is not None:
        print("\n\nRuntime Comparison:")
        print(timing.to_string(index=False))

    # ── Overall summary ───────────────────────────────────
    print("\n\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    # Check which tools have results
    for run_name in RUNS:
        gapit_ok = (GAPIT_DIR / run_name / "gapit_results_standardized.csv").exists()
        rmvp_ok = (RMVP_DIR / run_name / "rmvp_results_standardized.csv").exists()
        print(f"  {run_name:25s}  GAPIT3={'OK' if gapit_ok else 'MISSING':7s}  "
              f"rMVP={'OK' if rmvp_ok else 'MISSING':7s}")


if __name__ == "__main__":
    main()
