"""T-61 — Reconstruct the block-detection parameters that produced Table S8.

Confirms which flank window reproduces the six published tomato locule-number LD
blocks (Supp Table S8), and records ``median_decay`` — the quantity ``cli.py``
computes for the flank derivation but that no committed artifact stores.

This does NOT re-run the GWAS: it consumes the committed MLM result table
``platform_GWAS_locule_number.csv`` and mirrors the detection path of
``benchmarks/rerun_downstream_demo.py`` exactly (same QC loading, same sample
alignment, same fixed parameters). The reproducing configuration is the one that
already produced the committed ``platform_Haplotype_GWAS_MLM.csv`` (verified to
match Table S8 to three decimals); this script re-confirms it, records the flank
derivation, and shows that the alternative fixed-flank grid cells do NOT reproduce.

Rule (measure, never predict): no parameter is adjusted to force a match. If the
grid does not reproduce, that is the finding.

Run:  .venv/Scripts/python benchmarks/reconstruct_table_s8_params.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from annotation import canon_chr  # noqa: E402
from gwas import ld  # noqa: E402

QC_DIR = ROOT / "benchmarks" / "qc_data" / "tomato_locule_number"

# Table S8's six blocks, as printed at Mb precision (Supp p8) and — for an exact
# check — the bp intervals of the committed reproducing run
# (platform_Haplotype_GWAS_MLM.csv). Detection echoes these coordinates.
TABLE_S8_MB = {
    (46.46, 46.58), (46.97, 47.04), (47.02, 47.13),
    (47.09, 47.17), (47.24, 47.33), (47.30, 47.66),
}
TABLE_S8_BP = [
    (46461623, 46582239), (46973281, 47038294), (47017547, 47129438),
    (47092604, 47168061), (47239566, 47332608), (47301921, 47657766),
]

# Fixed parameters held constant across the whole grid (T-61 / rerun_downstream_demo
# defaults): ld_threshold=0.6, adj_r2_min=0.2, min_snps=2, top_n=10, sig_thresh=1e-5,
# gap_factor=10.0, merge_iou=0.3, min_pair_n=20 (all detector defaults except top_n).


def _load():
    geno_df = pd.read_csv(QC_DIR / "QC_genotype_matrix.csv", index_col=0)
    snp_map = pd.read_csv(QC_DIR / "QC_snp_map.csv")
    pheno_df = pd.read_csv(QC_DIR / "QC_phenotype.csv", index_col=0)

    geno = geno_df.values.astype(np.float32)
    col_means = np.nanmean(geno, axis=0)
    nan_mask = np.isnan(geno)
    if nan_mask.any():
        geno[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    sid = snp_map["SNP_ID"].values
    chroms_str = np.array([str(canon_chr(c)) for c in snp_map["Chr"].values])
    positions = snp_map["Pos"].values.astype(int)

    geno_df.index = geno_df.index.astype(str)
    pheno_df.index = pheno_df.index.astype(str)
    common = geno_df.index.intersection(pheno_df.index)
    geno_idx = [i for i, x in enumerate(geno_df.index) if x in set(common)]
    geno = geno[geno_idx, :]
    return geno, sid, chroms_str, positions


def _median_decay(geno, chroms_str, positions):
    """Mirror rerun_downstream_demo.py:73-98 exactly."""
    ld_distances = []
    for ch in np.unique(chroms_str):
        ch_idx = np.where(chroms_str == ch)[0]
        if len(ch_idx) < 50:
            continue
        geno_ch = geno[:, ch_idx].astype(float)
        pos_ch = positions[ch_idx]
        if len(pos_ch) > 1500:
            sub = np.linspace(0, len(pos_ch) - 1, 1500, dtype=int)
            pos_ch, geno_ch = pos_ch[sub], geno_ch[:, sub]
        r2_mat = ld.pairwise_r2(geno_ch)
        dk, _, _ = ld.ld_decay(pos_ch, r2_mat, ld_threshold=0.2, max_dist_kb=5000)
        if np.isfinite(dk):
            ld_distances.append(dk)
    if ld_distances:
        return float(np.median(ld_distances))
    return 150.0  # rerun_downstream_demo fallback


def _detect(gwas_df, chroms_str, positions, geno, sid, *, flank_kb, ld_decay_kb):
    blocks = ld.find_ld_clusters_genomewide(
        gwas_df=gwas_df, chroms=chroms_str, positions=positions,
        geno_imputed=geno.astype(float), sid=sid,
        ld_threshold=0.6, flank_kb=flank_kb, ld_decay_kb=ld_decay_kb,
        min_snps=2, top_n=10, sig_thresh=1e-5,
    )
    blocks, _ = ld.filter_contained_blocks(blocks, min_contained=2)
    return blocks


def _intervals_mb(blocks):
    if blocks is None or blocks.empty:
        return set()
    return {(round(int(r["Start (bp)"]) / 1e6, 2), round(int(r["End (bp)"]) / 1e6, 2))
            for _, r in blocks.iterrows()}


def _exact_bp(blocks):
    if blocks is None or blocks.empty:
        return []
    return sorted((int(r["Start (bp)"]), int(r["End (bp)"])) for _, r in blocks.iterrows())


def main():
    print("=" * 70)
    print("  T-61: reconstruct Table S8 block-detection parameters")
    print("=" * 70)

    geno, sid, chroms_str, positions = _load()
    gwas_df = pd.read_csv(QC_DIR / "platform_GWAS_locule_number.csv")
    print(f"  geno {geno.shape}, {len(gwas_df)} GWAS rows, "
          f"min p = {gwas_df['PValue'].min():.2e}")

    median_decay = _median_decay(geno, chroms_str, positions)
    flank_2x = int(2 * median_decay)
    print(f"\n  >>> median_decay = {median_decay:.3f} kb   "
          f"2x(unclipped) = {flank_2x} kb   "
          f"clip(2x,200,2000) = {int(np.clip(flank_2x, 200, 2000))} kb <<<")

    # --- primary: the exact rerun_downstream_demo config (flank=2x, ld_decay set) ---
    print("\n  PRIMARY — exact reproduction config "
          f"(flank_kb={flank_2x}, ld_decay_kb={median_decay:.3f}):")
    prim = _detect(gwas_df, chroms_str, positions, geno, sid,
                   flank_kb=flank_2x, ld_decay_kb=median_decay)
    prim_mb = _intervals_mb(prim)
    prim_bp = _exact_bp(prim)
    mb_match = prim_mb == TABLE_S8_MB
    bp_match = prim_bp == sorted(TABLE_S8_BP)
    print(f"    {len(prim)} blocks | Mb-match Table S8: {mb_match} | "
          f"exact-bp-match: {bp_match}")
    for s, e in prim_bp:
        tag = "" if (round(s / 1e6, 2), round(e / 1e6, 2)) in TABLE_S8_MB else "  <-- NOT in S8"
        print(f"      {s:,} - {e:,}  ({s/1e6:.2f}-{e/1e6:.2f} Mb){tag}")

    # --- grid: vary flank_kb, ld_decay_kb held at the data-derived value ---
    grid = [200, 300, flank_2x, int(np.clip(flank_2x, 200, 2000)), 400, 500, 600, 620]
    grid = sorted(set(grid))
    print("\n  GRID (ld_decay_kb = median_decay held fixed; vary flank_kb):")
    print(f"    {'flank_kb':>9} {'n_blocks':>9} {'Mb-match':>9}")
    reproducing = []
    for fk in grid:
        b = _detect(gwas_df, chroms_str, positions, geno, sid,
                    flank_kb=fk, ld_decay_kb=median_decay)
        ok = _intervals_mb(b) == TABLE_S8_MB
        if ok:
            reproducing.append(fk)
        star = " *2x*" if fk == flank_2x else ""
        print(f"    {fk:>9} {len(b):>9} {str(ok):>9}{star}")

    print("\n  CONCLUSION")
    print(f"    median_decay = {median_decay:.3f} kb; reproducing flank_kb = {flank_2x} "
          f"(2x median_decay, unclipped), with ld_decay_kb = {median_decay:.3f}.")
    print(f"    Flank cells reproducing all six Table S8 intervals: {reproducing}")
    print(f"    Primary config reproduces Table S8: Mb={mb_match}, exact-bp={bp_match}")
    print("    eta2 producer = library gwas/haplotype.py (n_perm=1000); confirmed "
          "elsewhere against the committed platform_Haplotype_GWAS_MLM.csv.")


if __name__ == "__main__":
    main()
