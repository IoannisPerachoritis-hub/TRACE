"""Compare Platform-LOCO vs Platform-Global vs GAPIT3 vs rMVP on real data.

Shows that Platform-Global matches GAPIT3/rMVP closely (confirming the
LOCO divergence explanation), and quantifies LOCO vs Global differences.
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent

DATASETS = {
    "tomato_locule_number": {},
    "tomato_weight_g": {},
    "pepper_FWe": {},
    "pepper_BX": {},
}


def lambda_gc(pvals):
    """Compute genomic inflation factor."""
    pvals = pvals[(pvals > 0) & (pvals < 1)]
    if len(pvals) < 10:
        return float("nan")
    chisq = stats.chi2.ppf(1 - pvals, df=1)
    return float(np.median(chisq) / stats.chi2.ppf(0.5, df=1))


def load_platform_global_results(dataset_name):
    """Load Platform-Global (--no-loco) results from ZIP."""
    result_dir = ROOT / "benchmarks" / "loco_sensitivity" / dataset_name
    if not result_dir.exists():
        return None

    zip_paths = list(result_dir.glob("GWAS_*.zip"))
    if not zip_paths:
        return None

    zpath = max(zip_paths, key=lambda p: p.stat().st_mtime)
    with zipfile.ZipFile(zpath) as zf:
        csv_names = [n for n in zf.namelist()
                    if n.startswith("tables/GWAS_") and n.endswith(".csv")
                    and "QC_" not in n and "PC_" not in n]
        if csv_names:
            with zf.open(csv_names[0]) as f:
                return pd.read_csv(f)
    return None


def load_gapit_results(dataset_name):
    """Load GAPIT3 standardized results."""
    path = ROOT / "benchmarks" / "results" / "gapit" / dataset_name / "gapit_results_standardized.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def load_rmvp_results(dataset_name):
    """Load rMVP standardized results."""
    path = ROOT / "benchmarks" / "results" / "rmvp" / dataset_name / "rmvp_results_standardized.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def load_platform_loco_results(dataset_name):
    """Load Platform-LOCO results from qc_data."""
    qc_dir = ROOT / "benchmarks" / "qc_data" / dataset_name
    # Find the MLM file
    mlm_files = list(qc_dir.glob("platform_GWAS_locule_number.csv")) + \
                list(qc_dir.glob("platform_GWAS_weight_g.csv")) + \
                list(qc_dir.glob("platform_GWAS_FWe.csv")) + \
                list(qc_dir.glob("platform_GWAS_BX.csv")) + \
                list(qc_dir.glob("platform_GWAS_MLM_*.csv")) + \
                list(qc_dir.glob("platform_GWAS_*.csv"))

    # Filter to MLM only (no FarmCPU, MLMM)
    mlm_files = [f for f in mlm_files
                 if "FarmCPU" not in f.name and "MLMM" not in f.name]
    if mlm_files:
        return pd.read_csv(mlm_files[0])
    return None


def compare_tools(ds_name, loco_df, global_df, gapit_df, rmvp_df):
    """Compute pairwise Spearman correlations and lambda GC."""
    tools = {}
    if loco_df is not None:
        # Standardize column names
        pcol = "PValue" if "PValue" in loco_df.columns else "pvalue"
        scol = "SNP" if "SNP" in loco_df.columns else "SNP_ID"
        if pcol not in loco_df.columns:
            for c in loco_df.columns:
                if "pval" in c.lower() or "p.val" in c.lower() or c == "P":
                    pcol = c
                    break
        if scol not in loco_df.columns:
            for c in loco_df.columns:
                if "snp" in c.lower():
                    scol = c
                    break
        tools["Platform-LOCO"] = loco_df[[scol, pcol]].rename(
            columns={scol: "SNP", pcol: "PValue"})

    if global_df is not None:
        pcol = "PValue" if "PValue" in global_df.columns else "pvalue"
        scol = "SNP" if "SNP" in global_df.columns else "SNP_ID"
        if pcol not in global_df.columns:
            for c in global_df.columns:
                if "pval" in c.lower() or "p.val" in c.lower() or c == "P":
                    pcol = c
                    break
        if scol not in global_df.columns:
            for c in global_df.columns:
                if "snp" in c.lower():
                    scol = c
                    break
        tools["Platform-Global"] = global_df[[scol, pcol]].rename(
            columns={scol: "SNP", pcol: "PValue"})

    if gapit_df is not None:
        pcol = "PValue" if "PValue" in gapit_df.columns else "P.value"
        scol = "SNP" if "SNP" in gapit_df.columns else "SNP_ID"
        if pcol not in gapit_df.columns:
            for c in gapit_df.columns:
                if "pval" in c.lower() or "p.val" in c.lower():
                    pcol = c
                    break
        if scol not in gapit_df.columns:
            for c in gapit_df.columns:
                if "snp" in c.lower():
                    scol = c
                    break
        tools["GAPIT3"] = gapit_df[[scol, pcol]].rename(
            columns={scol: "SNP", pcol: "PValue"})

    if rmvp_df is not None:
        pcol = "PValue" if "PValue" in rmvp_df.columns else "P.value"
        scol = "SNP" if "SNP" in rmvp_df.columns else "SNP_ID"
        if pcol not in rmvp_df.columns:
            for c in rmvp_df.columns:
                if "pval" in c.lower() or "p.val" in c.lower():
                    pcol = c
                    break
        if scol not in rmvp_df.columns:
            for c in rmvp_df.columns:
                if "snp" in c.lower():
                    scol = c
                    break
        tools["rMVP"] = rmvp_df[[scol, pcol]].rename(
            columns={scol: "SNP", pcol: "PValue"})

    if len(tools) < 2:
        return None

    # Compute lambda GC for each
    lambdas = {}
    for name, df in tools.items():
        lambdas[name] = lambda_gc(df["PValue"].dropna().values)

    # Pairwise Spearman on -log10(p)
    pairs = []
    tool_names = list(tools.keys())
    for i in range(len(tool_names)):
        for j in range(i + 1, len(tool_names)):
            t1, t2 = tool_names[i], tool_names[j]
            merged = tools[t1].merge(tools[t2], on="SNP", suffixes=("_1", "_2"))
            merged = merged.dropna()
            if len(merged) < 100:
                continue
            log1 = -np.log10(merged["PValue_1"].clip(lower=1e-300))
            log2 = -np.log10(merged["PValue_2"].clip(lower=1e-300))
            rho, pval = stats.spearmanr(log1, log2)
            pairs.append({
                "dataset": ds_name,
                "tool_1": t1,
                "tool_2": t2,
                "spearman_rho": round(rho, 4),
                "n_snps": len(merged),
                "lambda_gc_1": round(lambdas[t1], 4),
                "lambda_gc_2": round(lambdas[t2], 4),
            })

    return pairs


def run_comparison():
    """Run full LOCO sensitivity comparison."""
    all_pairs = []

    for ds_name in DATASETS:
        print(f"\n{'='*50}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*50}")

        # Load Platform-LOCO (original runs)
        loco_df = load_platform_loco_results(ds_name)
        if loco_df is not None:
            print(f"  Platform-LOCO: {len(loco_df)} SNPs")
        else:
            print("  Platform-LOCO: NOT FOUND")

        # Load Platform-Global (--no-loco runs)
        global_df = load_platform_global_results(ds_name)
        if global_df is not None:
            print(f"  Platform-Global: {len(global_df)} SNPs")
        else:
            print("  Platform-Global: NOT FOUND")

        # Load GAPIT3
        gapit_df = load_gapit_results(ds_name)
        if gapit_df is not None:
            print(f"  GAPIT3: {len(gapit_df)} SNPs")
        else:
            print("  GAPIT3: NOT FOUND")

        # Load rMVP
        rmvp_df = load_rmvp_results(ds_name)
        if rmvp_df is not None:
            print(f"  rMVP: {len(rmvp_df)} SNPs")
        else:
            print("  rMVP: NOT FOUND")

        pairs = compare_tools(ds_name, loco_df, global_df, gapit_df, rmvp_df)
        if pairs:
            all_pairs.extend(pairs)
            print("\n  Pairwise Spearman -log10(p):")
            for p in pairs:
                print(f"    {p['tool_1']:20s} vs {p['tool_2']:20s}: "
                      f"rho={p['spearman_rho']:.4f}")

    if all_pairs:
        df = pd.DataFrame(all_pairs)
        out_path = ROOT / "benchmarks" / "loco_sensitivity" / "loco_comparison.csv"
        df.to_csv(out_path, index=False)
        print(f"\n\nComparison saved to {out_path}")
        print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    run_comparison()
