"""Re-run TRACE FarmCPU on all 4 datasets with OLS final scan (standard FarmCPU).

Loads QC data from benchmarks/qc_data/, auto-selects PCs via lambda_GC
scan (matching the platform's auto-PC logic), runs run_farmcpu(), and
saves updated results.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gwas.utils import PhenoData, CovarData, _mean_impute_cols
from gwas.kinship import _build_grm_from_Z, _standardize_geno_for_grm
from gwas.models import run_farmcpu
from gwas.plotting import compute_lambda_gc
from annotation import canon_chr

QC_DIR = ROOT / "benchmarks" / "qc_data"

RUNS = {
    "pepper_FWe": {"trait": "FWe"},
    "pepper_BX": {"trait": "BX"},
    "tomato_weight_g": {"trait": "weight_g"},
    "tomato_locule_number": {"trait": "locule_number"},
}

MAX_PCS_SCAN = 10


def load_qc_data(run_name: str):
    """Load QC'd genotype, SNP map, and phenotype for a dataset."""
    qc = QC_DIR / run_name
    geno_df = pd.read_csv(qc / "QC_genotype_matrix.csv", index_col=0)
    snp_map = pd.read_csv(qc / "QC_snp_map.csv")
    pheno_df = pd.read_csv(qc / "QC_phenotype.csv")
    return geno_df, snp_map, pheno_df


def _auto_select_pcs(geno_imputed, sid, chroms, chroms_num, positions,
                     iid, pheno_reader, K0, pcs_full,
                     band_lo=0.95, band_hi=1.05, parsimony_tol=0.02):
    """Scan k=0..MAX_PCS_SCAN PCs, pick k using band strategy.

    Band: smallest k where band_lo <= lambda_GC <= band_hi.
    Fallback: closest to 1.0 with adaptive parsimony tolerance.
    """
    max_k = min(MAX_PCS_SCAN, pcs_full.shape[1])
    lambdas = []
    for k in range(max_k + 1):
        covar_k = None
        if k > 0:
            covar_k = CovarData(
                iid=iid, val=pcs_full[:, :k],
                names=[f"PC{i+1}" for i in range(k)],
            )
        try:
            df_k, _, _ = run_farmcpu(
                geno_imputed=geno_imputed, sid=sid, chroms=chroms,
                chroms_num=chroms_num, positions=positions, iid=iid,
                pheno_reader=pheno_reader, K0=K0, covar_reader=covar_k,
                verbose=False,
            )
            lam = compute_lambda_gc(df_k["PValue"].values, trim=False)
        except Exception:
            lam = np.nan
        lambdas.append(lam)
        print(f"    k={k}: lambda={lam:.4f}" if np.isfinite(lam) else f"    k={k}: lambda=NaN")

    deltas = [abs(v - 1.0) if np.isfinite(v) else np.inf for v in lambdas]
    valid = [np.isfinite(v) for v in lambdas]

    # Band strategy: smallest k in [band_lo, band_hi]
    in_band = [i for i in range(len(lambdas))
               if valid[i] and band_lo <= lambdas[i] <= band_hi]
    if in_band:
        best_k = in_band[0]
    else:
        # Fallback: closest to 1.0 with adaptive parsimony tolerance
        valid_deltas = [deltas[i] for i in range(len(deltas)) if valid[i]]
        best_delta = min(valid_deltas) if valid_deltas else 0
        tol = max(parsimony_tol, best_delta * 0.15)
        near_best = [i for i in range(len(deltas))
                     if valid[i] and deltas[i] <= best_delta + tol]
        best_k = near_best[0] if near_best else 0

    print(f"  Auto-PC (band): selected k={best_k} (lambda={lambdas[best_k]:.4f})")
    return best_k


def run_single(run_name: str):
    """Run TRACE FarmCPU on a single dataset and save results."""
    info = RUNS[run_name]
    trait = info["trait"]

    print(f"\n{'='*60}")
    print(f"  {run_name} (trait={trait})")
    print(f"{'='*60}")

    geno_df, snp_map, pheno_df = load_qc_data(run_name)

    # Align samples
    common = sorted(set(geno_df.index) & set(pheno_df["SampleID"]))
    geno_df = geno_df.loc[common]
    pheno_df = pheno_df.set_index("SampleID").loc[common].reset_index()

    geno_imputed = _mean_impute_cols(geno_df.values.astype(np.float32))
    sid = np.array(geno_df.columns, dtype=str)
    n_samples, n_snps = geno_imputed.shape
    print(f"  Samples: {n_samples}, SNPs: {n_snps}")

    # SNP metadata
    snp_order = {s: i for i, s in enumerate(snp_map["SNP_ID"].values)}
    idx_order = [snp_order[s] for s in sid]
    chroms_raw = snp_map["Chr"].values[idx_order].astype(str)
    positions_raw = snp_map["Pos"].values[idx_order].astype(int)
    chroms_canon = np.array([canon_chr(c) for c in chroms_raw], dtype=str)

    # Filter out ALT / unplaced chromosomes (non-numeric)
    numeric_mask = np.array([c.isdigit() for c in chroms_canon])
    if not numeric_mask.all():
        n_alt = (~numeric_mask).sum()
        print(f"  Filtering {n_alt} ALT/unplaced SNPs")
        geno_imputed = geno_imputed[:, numeric_mask]
        sid = sid[numeric_mask]
        chroms_raw = chroms_raw[numeric_mask]
        positions_raw = positions_raw[numeric_mask]
        chroms_canon = chroms_canon[numeric_mask]
        n_snps = geno_imputed.shape[1]
        print(f"  SNPs after filter: {n_snps}")

    chroms = chroms_raw
    positions = positions_raw
    chroms_num = chroms_canon.astype(int)

    # IID
    iid = np.array([[s, s] for s in common], dtype=str)

    # Phenotype reader
    y = pheno_df[trait].values.astype(float)
    pheno_reader = PhenoData(iid=iid, val=y.reshape(-1, 1))

    # Kinship (GRM)
    Z_grm = _standardize_geno_for_grm(geno_imputed)
    K = _build_grm_from_Z(Z_grm)
    from pysnptools.kernelreader import KernelData
    K0 = KernelData(iid=iid, val=K)

    # PCA — compute max PCs, then auto-select
    from sklearn.decomposition import PCA
    max_k = min(MAX_PCS_SCAN, n_samples - 1, n_snps)
    pca = PCA(n_components=max_k)
    pcs_full = pca.fit_transform(geno_imputed)

    print(f"  Auto-selecting PCs for FarmCPU (scanning k=0..{max_k})...")
    n_pcs = _auto_select_pcs(
        geno_imputed, sid, chroms, chroms_num, positions,
        iid, pheno_reader, K0, pcs_full,
    )

    covar_reader = None
    if n_pcs > 0:
        covar_reader = CovarData(
            iid=iid,
            val=pcs_full[:, :n_pcs],
            names=[f"PC{i+1}" for i in range(n_pcs)],
        )

    # Final FarmCPU run with selected PCs
    print(f"  Running FarmCPU (MLM final scan, {n_pcs} PCs)...")
    farmcpu_df, pseudo_qtn_table, convergence_info = run_farmcpu(
        geno_imputed=geno_imputed,
        sid=sid,
        chroms=chroms,
        chroms_num=chroms_num,
        positions=positions,
        iid=iid,
        pheno_reader=pheno_reader,
        K0=K0,
        covar_reader=covar_reader,
        final_scan="mlm",
        verbose=False,
    )

    print(f"  Converged: {convergence_info['converged']}, "
          f"Iterations: {convergence_info['n_iterations']}, "
          f"Pseudo-QTNs: {convergence_info['n_pseudo_qtns']}")

    # Lambda GC
    lam = compute_lambda_gc(farmcpu_df["PValue"].values)
    print(f"  Lambda GC: {lam:.4f}")

    # Save
    out_path = QC_DIR / run_name / f"platform_GWAS_FarmCPU_{trait}.csv"
    farmcpu_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    return farmcpu_df


def main():
    print("Re-running TRACE FarmCPU with OLS final scan + auto-PC selection")
    print("=" * 60)

    for run_name in RUNS:
        run_single(run_name)

    print("\n\nDone. Now run: python benchmarks/run_farmcpu_benchmarks.py")


if __name__ == "__main__":
    main()
