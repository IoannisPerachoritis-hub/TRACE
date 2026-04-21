"""Re-run TRACE MLM GWAS on all 4 benchmark QC datasets.

Generates platform_GWAS_<trait>.csv files in benchmarks/qc_data/<run>/
using the current paper repo code, with the same auto-PC band strategy
used in production.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from annotation import canon_chr
from gwas.kinship import (
    _standardize_geno_for_grm,
    _build_grm_from_Z,
    _ld_prune_for_grm_by_chr_bp,
    _build_loco_kernels_impl,
)
from gwas.models import _run_gwas_impl, auto_select_pcs
from gwas.utils import PhenoData

QC_DIR = ROOT / "benchmarks" / "qc_data"

RUNS = {
    "pepper_FWe": "FWe",
    "pepper_BX": "BX",
    "tomato_weight_g": "weight_g",
    "tomato_locule_number": "locule_number",
}


def load_qc_dataset(run_name):
    """Load QC'd dataset from benchmarks/qc_data/."""
    qc = QC_DIR / run_name
    geno_df = pd.read_csv(qc / "QC_genotype_matrix.csv", index_col=0)
    snp_map = pd.read_csv(qc / "QC_snp_map.csv")
    pheno_df = pd.read_csv(qc / "QC_phenotype.csv", index_col=0)

    geno = geno_df.values.astype(np.float32)
    col_means = np.nanmean(geno, axis=0)
    nan_mask = np.isnan(geno)
    if nan_mask.any():
        geno[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    sid = snp_map["SNP_ID"].values if "SNP_ID" in snp_map.columns else snp_map.iloc[:, 0].values
    chroms_raw = snp_map["Chr"].values if "Chr" in snp_map.columns else snp_map.iloc[:, 1].values
    chroms_str = np.array([str(canon_chr(c)) for c in chroms_raw])

    chr_map = {}
    for c in np.unique(chroms_str):
        try:
            chr_map[c] = int(c)
        except ValueError:
            chr_map[c] = hash(c) % 1000
    chroms_num = np.array([chr_map[c] for c in chroms_str])

    positions = snp_map["Pos"].values.astype(float) if "Pos" in snp_map.columns else snp_map.iloc[:, 2].values.astype(float)

    trait_col = [c for c in pheno_df.columns if c.lower() not in ("fid", "iid", "sampleid")][0]
    y = pheno_df[trait_col].values.astype(float)

    geno_df.index = geno_df.index.astype(str)
    pheno_df.index = pheno_df.index.astype(str)
    common = geno_df.index.intersection(pheno_df.index)
    geno_idx = [i for i, x in enumerate(geno_df.index) if x in set(common)]
    pheno_idx = pheno_df.index.get_indexer(geno_df.index[geno_idx])

    geno = geno[geno_idx, :]
    y = y[pheno_idx]

    valid = np.isfinite(y)
    geno = geno[valid, :]
    y = y[valid]

    sample_ids = geno_df.index[geno_idx].values[valid].astype(str)
    iid = np.column_stack([sample_ids, sample_ids])

    return geno, y, sid, chroms_str, chroms_num, positions, iid, trait_col


def run_trace_mlm(run_name, trait_name):
    """Run TRACE MLM with auto-PC on one dataset."""
    print(f"\n{'='*60}")
    print(f"  {run_name} ({trait_name})")
    print(f"{'='*60}")

    geno, y, sid, chroms_str, chroms_num, positions, iid, trait_col = load_qc_dataset(run_name)
    n, m = geno.shape
    print(f"  Loaded: {n} samples, {m} SNPs")

    # Build GRM
    Z_pruned, keep_mask = _ld_prune_for_grm_by_chr_bp(
        chroms_str, positions, geno, return_mask=True,
    )
    chroms_pruned = chroms_str[keep_mask]
    Z_grm = _standardize_geno_for_grm(Z_pruned)
    K_base = _build_grm_from_Z(Z_grm)

    eigvals, eigvecs = eigh(K_base)
    idx_sort = np.argsort(eigvals)[::-1]
    pcs_full = eigvecs[:, idx_sort[:10]]

    # Auto-PC selection
    result_df = auto_select_pcs(
        geno_imputed=geno,
        y=y,
        sid=sid,
        chroms=chroms_str,
        chroms_num=chroms_num,
        positions=positions,
        iid=iid,
        Z_grm=Z_grm,
        chroms_grm=chroms_pruned,
        K_base=K_base,
        pcs_full=pcs_full,
        max_pcs=10,
        strategy="band",
    )
    rec = result_df[result_df["recommended"] == "\u2605"]
    n_pcs = int(rec["n_pcs"].values[0]) if len(rec) > 0 else 0
    print(f"  Auto-PC selected: k = {n_pcs}")

    # Build LOCO kernels
    K0, K_by_chr, _ = _build_loco_kernels_impl(
        iid=iid, Z_grm=Z_grm, chroms_grm=chroms_pruned, K_base=K_base,
    )
    pheno_reader = PhenoData(iid=iid, val=y)

    # Run final LOCO MLM with selected PCs
    gwas_df = _run_gwas_impl(
        geno_imputed=geno,
        y=y,
        pcs_full=pcs_full,
        n_pcs=n_pcs,
        sid=sid,
        positions=positions,
        chroms=chroms_str,
        chroms_num=chroms_num,
        iid=iid,
        _K0=K0,
        _K_by_chr=K_by_chr,
        pheno_reader=pheno_reader,
    )

    # Save results
    out_path = QC_DIR / run_name / f"platform_GWAS_{trait_name}.csv"
    gwas_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    print(f"  Results: {len(gwas_df)} SNPs, min p = {gwas_df['PValue'].min():.2e}")

    return gwas_df


if __name__ == "__main__":
    for run_name, trait_name in RUNS.items():
        run_trace_mlm(run_name, trait_name)
    print("\nAll datasets complete.")
