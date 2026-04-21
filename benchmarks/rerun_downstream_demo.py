"""Re-run LD block detection + haplotype GWAS on tomato locule number.

Uses the MLM GWAS results from rerun_trace_mlm.py (must run first).
Generates platform_Haplotype_GWAS_MLM.csv used by figure3 scripts.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from annotation import canon_chr
from gwas import ld
from gwas.haplotype import run_haplotype_block_gwas
from gwas.kinship import (
    _standardize_geno_for_grm,
    _build_grm_from_Z,
    _ld_prune_for_grm_by_chr_bp,
)

QC_DIR = ROOT / "benchmarks" / "qc_data" / "tomato_locule_number"
TRAIT = "locule_number"


def main():
    print("=" * 60)
    print("  Downstream Demo: tomato locule number")
    print("=" * 60)

    # ── Load QC data ──
    geno_df = pd.read_csv(QC_DIR / "QC_genotype_matrix.csv", index_col=0)
    snp_map = pd.read_csv(QC_DIR / "QC_snp_map.csv")
    pheno_df = pd.read_csv(QC_DIR / "QC_phenotype.csv", index_col=0)

    geno = geno_df.values.astype(np.float32)
    col_means = np.nanmean(geno, axis=0)
    nan_mask = np.isnan(geno)
    if nan_mask.any():
        geno[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    sid = snp_map["SNP_ID"].values
    chroms_raw = snp_map["Chr"].values
    chroms_str = np.array([str(canon_chr(c)) for c in chroms_raw])
    positions = snp_map["Pos"].values.astype(int)

    # Align samples
    geno_df.index = geno_df.index.astype(str)
    pheno_df.index = pheno_df.index.astype(str)
    common = geno_df.index.intersection(pheno_df.index)
    geno_idx = [i for i, x in enumerate(geno_df.index) if x in set(common)]
    pheno_idx = pheno_df.index.get_indexer(geno_df.index[geno_idx])

    geno = geno[geno_idx, :]
    pheno_aligned = pheno_df.iloc[pheno_idx].copy()
    pheno_aligned.index = geno_df.index[geno_idx]

    # Rebuild geno_df aligned
    geno_df_aligned = pd.DataFrame(
        geno, index=geno_df.index[geno_idx], columns=geno_df.columns,
    )

    n, m = geno.shape
    print(f"  Loaded: {n} samples, {m} SNPs")

    # ── Load new MLM GWAS results ──
    gwas_path = QC_DIR / "platform_GWAS_locule_number.csv"
    gwas_df = pd.read_csv(gwas_path)
    print(f"  GWAS results: {len(gwas_df)} SNPs, min p = {gwas_df['PValue'].min():.2e}")

    # ── LD decay estimation ──
    print("  Estimating LD decay...")
    ld_distances = []
    for ch in np.unique(chroms_str):
        ch_mask = chroms_str == ch
        ch_idx = np.where(ch_mask)[0]
        if len(ch_idx) < 50:
            continue
        geno_ch = geno[:, ch_idx].astype(float)
        pos_ch = positions[ch_idx]
        # Subsample large chromosomes (same as CLI, line 927)
        if len(pos_ch) > 1500:
            sub_idx = np.linspace(0, len(pos_ch) - 1, 1500, dtype=int)
            pos_ch, geno_ch = pos_ch[sub_idx], geno_ch[:, sub_idx]
        r2_mat = ld.pairwise_r2(geno_ch)
        dk, _, _ = ld.ld_decay(pos_ch, r2_mat, ld_threshold=0.2, max_dist_kb=5000)
        if np.isfinite(dk):
            ld_distances.append(dk)

    if ld_distances:
        ld_decay_kb = float(np.median(ld_distances))
        ld_flank_kb = int(2 * ld_decay_kb)
    else:
        ld_decay_kb = 150.0
        ld_flank_kb = 300
    print(f"  LD decay ~ {ld_decay_kb:.0f} kb -> flank = {ld_flank_kb} kb")

    # ── LD block detection ──
    print("  Detecting LD blocks...")
    ld_blocks = ld.find_ld_clusters_genomewide(
        gwas_df=gwas_df, chroms=chroms_str, positions=positions,
        geno_imputed=geno.astype(float), sid=sid,
        ld_threshold=0.6, flank_kb=ld_flank_kb,
        ld_decay_kb=ld_decay_kb, min_snps=3,
        top_n=10, sig_thresh=1e-5,
    )
    ld_blocks, _ = ld.filter_contained_blocks(ld_blocks, min_contained=2)
    print(f"  {len(ld_blocks)} LD blocks detected")

    if ld_blocks.empty:
        print("  WARNING: No LD blocks — skipping haplotype GWAS")
        return

    # Print block summary
    for _, blk in ld_blocks.iterrows():
        start_col = "Start (bp)" if "Start (bp)" in blk.index else "Start"
        end_col = "End (bp)" if "End (bp)" in blk.index else "End"
        print(f"    Chr{blk['Chr']} {blk[start_col]:,}-{blk[end_col]:,} "
              f"({blk.get('n_snps', '?')} SNPs)")

    # ── Haplotype GWAS ──
    print("  Running haplotype GWAS (Freedman-Lane, 500 perms)...")
    # Rename block columns for haplotype function
    haplo_df = ld_blocks.copy()
    if "Start (bp)" in haplo_df.columns:
        haplo_df = haplo_df.rename(columns={
            "Start (bp)": "Start", "End (bp)": "End",
        })

    trait_col = TRAIT
    if trait_col not in pheno_aligned.columns:
        # Try to find the trait column
        for c in pheno_aligned.columns:
            if c.lower() not in ("fid", "iid", "sampleid"):
                trait_col = c
                break

    hap_results, _ = run_haplotype_block_gwas(
        haplo_df=haplo_df,
        chroms=chroms_str,
        positions=positions,
        geno_imputed=geno.astype(float),
        sid=sid,
        geno_df=geno_df_aligned,
        pheno_df=pheno_aligned,
        trait_col=trait_col,
        pcs=None,  # k=0 PCs
        n_perm=500,
        n_pcs_used=0,
    )

    if hap_results is not None and not hap_results.empty:
        out_path = QC_DIR / "platform_Haplotype_GWAS_MLM.csv"
        hap_results.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")
        print(f"  {len(hap_results)} blocks with haplotype results")

        # Summary
        for _, row in hap_results.iterrows():
            f_val = row.get("F_param", row.get("F_perm", "?"))
            eta2 = row.get("eta2", "?")
            pval = row.get("PValue", "?")
            print(f"    Chr{row['Chr']} {row['Start']:,}-{row['End']:,}: "
                  f"F={f_val:.2f}, eta2={eta2:.3f}, p={pval}")
    else:
        print("  WARNING: No haplotype results returned")


if __name__ == "__main__":
    main()
