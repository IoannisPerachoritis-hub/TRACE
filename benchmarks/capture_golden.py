"""T-94 — Published-output regression harness: golden capture.

Freezes the four downstream artefacts of the tomato locule-number demo
(`ld_blocks.csv`, `haplotype_blocks.csv`, `annotated_blocks.csv`,
`run_manifest.json`) from the *real* Varitome QC inputs, so that
`tests/test_golden_published.py` can prove no published number moved. The
pipeline here mirrors `benchmarks/rerun_downstream_demo.py` exactly — the same
loading, the same producers, the same k=0 PCs — so the frozen tables are what the
shipped code emits, not a re-derivation.

`flank_kb` is a **required** keyword with no default: §0.1 of the coverage spec
establishes that no single flank value is currently authoritative across the CLI
/ GUI paths, so a default here would silently enshrine whichever one the
implementer guessed. The value that reproduces Table S8 is determined by T-61
(Batch B) and passed in explicitly; the golden CSVs and their ±10 %-flank /
min_snps=2 perturbation demonstration are produced at that point.

Because `benchmarks/qc_data/` is gitignored, the consuming test is marked
`@pytest.mark.golden` and excluded from CI (`-m "not golden"`). Do not commit
panel genotypes to make CI green.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation import annotate_ld_blocks, canon_chr, load_gene_annotation  # noqa: E402
from gwas import ld  # noqa: E402
from gwas.haplotype import run_haplotype_block_gwas  # noqa: E402

# Exact frozen column sets (coverage_and_scope.md T-60/T-94). hap_stats_json is
# excluded (non-contractual serialised dict); MidPos and -log10p are derived.
LD_BLOCK_COLS = ["Chr", "Start (bp)", "End (bp)", "lead_snp", "lead_snp_pvalue", "SNP_IDs"]
HAP_COLS = [
    "Chr", "Start", "End", "lead_snp", "n_samples_block", "n_snps", "n_haplotypes",
    "n_tested_haplotypes", "df1", "df2", "n_permutations", "F_param", "PValue_param",
    "F_perm", "P_perm", "PValue", "eta2",
]
ANN_COLS = [
    "Chr", "Start", "End", "n_genes_overlapping", "overlapping_genes", "annotation_status",
    "upstream_gene_1", "upstream_dist_1", "upstream_gene_2", "upstream_dist_2",
    "downstream_gene_1", "downstream_dist_1", "downstream_gene_2", "downstream_dist_2",
]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rename_start_end(df):
    ren = {}
    if "Start (bp)" in df.columns:
        ren["Start (bp)"] = "Start"
    if "End (bp)" in df.columns:
        ren["End (bp)"] = "End"
    return df.rename(columns=ren) if ren else df


def _freeze(df, cols):
    """Select the declared frozen columns (after normalising Start/End names).
    Returns (frozen_df, missing_cols) — a missing column is a contract change and
    is surfaced in the manifest, never silently dropped."""
    df = _rename_start_end(df.copy())
    present = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    return df[present].reset_index(drop=True), missing


def _load_qc(qc_dir):
    qc_dir = Path(qc_dir)
    geno_df = pd.read_csv(qc_dir / "QC_genotype_matrix.csv", index_col=0)
    snp_map = pd.read_csv(qc_dir / "QC_snp_map.csv")
    pheno_df = pd.read_csv(qc_dir / "QC_phenotype.csv", index_col=0)

    geno = geno_df.values.astype(np.float64)
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
    pheno_idx = pheno_df.index.get_indexer(geno_df.index[geno_idx])
    geno = geno[geno_idx, :]
    pheno_aligned = pheno_df.iloc[pheno_idx].copy()
    pheno_aligned.index = geno_df.index[geno_idx]
    geno_df_aligned = pd.DataFrame(geno, index=geno_df.index[geno_idx], columns=geno_df.columns)
    return geno, geno_df_aligned, pheno_aligned, sid, chroms_str, positions


def capture_golden(
    qc_dir: Path,
    gwas_csv: Path,
    out_dir: Path,
    *,
    flank_kb: float,               # REQUIRED — no default (see module docstring / T-61)
    ld_decay_kb: float | None = None,
    ld_threshold: float = 0.6,
    adj_r2_min: float = 0.2,
    min_snps: int = 3,
    top_n: int = 10,
    sig_thresh: float = 1e-5,
    gap_factor: float = 10.0,
    merge_iou: float = 0.3,
    min_contained: int = 2,
    n_perm: int = 1000,
    min_hap_count: int = 5,
    min_group_size: int = 3,
    genes_csv: Path | None = None,
    descriptions_csv: Path | None = None,
    trait_col: str | None = None,
) -> dict:
    """Freeze the downstream golden tables. Returns a dict of written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    geno, geno_df_aligned, pheno_aligned, sid, chroms_str, positions = _load_qc(qc_dir)
    gwas_df = pd.read_csv(gwas_csv)

    # --- LD blocks (detection -> containment filter) ---
    ld_blocks = ld.find_ld_clusters_genomewide(
        gwas_df=gwas_df, chroms=chroms_str, positions=positions,
        geno_imputed=geno.astype(float), sid=sid,
        ld_threshold=ld_threshold, flank_kb=flank_kb, ld_decay_kb=ld_decay_kb,
        min_snps=min_snps, top_n=top_n, sig_thresh=sig_thresh,
        adj_r2_min=adj_r2_min, merge_iou=merge_iou, gap_factor=gap_factor,
    )
    ld_blocks, _ = ld.filter_contained_blocks(ld_blocks, min_contained=min_contained)
    written = {}
    ld_out = ld_blocks[[c for c in LD_BLOCK_COLS if c in ld_blocks.columns]].reset_index(drop=True)
    ld_out.to_csv(out_dir / "ld_blocks.csv", index=False)
    written["ld_blocks"] = out_dir / "ld_blocks.csv"

    # --- Haplotype GWAS (k=0 PCs, matching the demo) ---
    hap_missing = []
    if not ld_blocks.empty:
        haplo_df = _rename_start_end(ld_blocks.copy())
        tcol = trait_col
        if tcol is None or tcol not in pheno_aligned.columns:
            tcol = next(c for c in pheno_aligned.columns
                        if c.lower() not in ("fid", "iid", "sampleid"))
        hap_results, _ = run_haplotype_block_gwas(
            haplo_df=haplo_df, chroms=chroms_str, positions=positions,
            geno_imputed=geno.astype(float), sid=sid,
            geno_df=geno_df_aligned, pheno_df=pheno_aligned, trait_col=tcol,
            pcs=None, n_perm=n_perm, n_pcs_used=0,
            min_hap_count=min_hap_count, min_group_size=min_group_size,
        )
        hap_frozen, hap_missing = _freeze(hap_results, HAP_COLS)
        hap_frozen.to_csv(out_dir / "haplotype_blocks.csv", index=False)
        written["haplotype_blocks"] = out_dir / "haplotype_blocks.csv"

    # --- Annotation ---
    ann_missing = []
    if genes_csv is not None and not ld_blocks.empty:
        genes = load_gene_annotation(str(genes_csv),
                                     str(descriptions_csv) if descriptions_csv else None)
        annotated = annotate_ld_blocks(ld_blocks.copy(), genes, n_flank=2, max_flank_dist_bp=500_000)
        ann_frozen, ann_missing = _freeze(annotated, ANN_COLS)
        ann_frozen.to_csv(out_dir / "annotated_blocks.csv", index=False)
        written["annotated_blocks"] = out_dir / "annotated_blocks.csv"

    # --- Manifest ---
    n = geno.shape[0]
    m = geno.shape[1]
    meff_value = n_sig_meff = n_sig_bonf = None
    try:
        from gwas.plotting import compute_meff_li_ji
        _meff, _eigs = compute_meff_li_ji(geno.astype(float))  # returns (meff, eigenvalues)
        meff_value = float(_meff)
        if meff_value and np.isfinite(meff_value):
            n_sig_meff = int((gwas_df["PValue"] < (0.05 / meff_value)).sum())
        n_sig_bonf = int((gwas_df["PValue"] < (0.05 / m)).sum())
    except Exception as exc:  # noqa: BLE001 — provenance only; recorded, not fatal
        meff_value = f"uncomputed: {type(exc).__name__}"

    manifest = {
        "n_samples": int(n), "n_snps": int(m),
        "flank_kb": flank_kb, "ld_decay_kb": ld_decay_kb, "ld_threshold": ld_threshold,
        "adj_r2_min": adj_r2_min, "min_snps": min_snps, "top_n": top_n,
        "sig_thresh": sig_thresh, "gap_factor": gap_factor, "merge_iou": merge_iou,
        "min_contained": min_contained, "n_perm": n_perm,
        "min_hap_count": min_hap_count, "min_group_size": min_group_size,
        "n_significant_meff": n_sig_meff, "n_significant_bonf": n_sig_bonf,
        "meff_value": meff_value,
        "gwas_csv": str(gwas_csv), "gwas_csv_sha256": _sha256(gwas_csv),
        "genes_csv": str(genes_csv) if genes_csv else None,
        "descriptions_csv": str(descriptions_csv) if descriptions_csv else None,
        "trait_col": trait_col,
        "n_blocks": int(len(ld_blocks)),
        "haplotype_missing_columns": hap_missing,
        "annotated_missing_columns": ann_missing,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    written["run_manifest"] = out_dir / "run_manifest.json"
    return written


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Capture the published-output golden (T-94).")
    ap.add_argument("--qc-dir", required=True, type=Path)
    ap.add_argument("--gwas-csv", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--flank-kb", required=True, type=float,
                    help="REQUIRED — the flank that reproduces Table S8 (from T-61)")
    ap.add_argument("--ld-decay-kb", type=float, default=None)
    ap.add_argument("--genes-csv", type=Path, default=None)
    ap.add_argument("--descriptions-csv", type=Path, default=None)
    ap.add_argument("--trait-col", default=None)
    a = ap.parse_args()
    paths = capture_golden(
        a.qc_dir, a.gwas_csv, a.out_dir, flank_kb=a.flank_kb, ld_decay_kb=a.ld_decay_kb,
        genes_csv=a.genes_csv, descriptions_csv=a.descriptions_csv, trait_col=a.trait_col,
    )
    for k, v in paths.items():
        print(f"  wrote {k}: {v}")
