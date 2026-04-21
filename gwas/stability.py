from datetime import datetime
import sys
import numpy as np
import pandas as pd
import scipy
try:
    import streamlit as st
except ImportError:
    st = None
from gwas.utils import CovarData, resolve_trait_column, align_pheno_to_geno
from gwas.models import _ols_scan_pvals_fast
from gwas.ld import (
    find_ld_clusters_genomewide,
    _blocks_to_interval_set,
    _interval_iou_bp,
    _split_leads,
)

def _stability_screen_topk(
    geno_imputed,
    y_vec,
    sid,
    covar_reader,
    iid_full=None,
    n_reps=100,
    test_frac=0.2,
    top_k=50,
    seed=0,
):
    """
    Stability screen:
    - Repeated random splits
    - On each split: OLS p-value scan in TRAIN only
    - Track how often each SNP appears in top_k.
    This avoids recomputing GRM/LOCO per split.
    """

    # SAFETY: covariate ↔ genotype alignment
    if covar_reader is not None and getattr(covar_reader, "val", None) is not None:
        if covar_reader.val.shape[0] != geno_imputed.shape[0]:
            raise ValueError(
                "Covariates and genotypes are misaligned in stability screen (top-k)."
            )

    # SAFETY: iid_full ↔ genotype alignment
    if iid_full is not None and iid_full.shape[0] != geno_imputed.shape[0]:
        raise ValueError(
            "iid_full rows do not match genotype rows in stability screen (top-k)."
        )

    rng = np.random.default_rng(seed)
    n, m = geno_imputed.shape

    counts = np.zeros(m, dtype=int)

    for _ in range(int(n_reps)):
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = max(1, int(round(test_frac * n)))
        train_idx = idx[n_test:]

        G_tr = geno_imputed[train_idx, :]
        y_tr = y_vec[train_idx]

        cov_tr = None
        if covar_reader is not None and getattr(covar_reader, "val", None) is not None:
            if iid_full is None:
                raise ValueError(
                    "iid_full must be provided when covariates are used in stability screen."
                )

            cov_tr = CovarData(
                iid=iid_full[train_idx, :],
                val=np.asarray(covar_reader.val)[train_idx, :],
                names=list(covar_reader.sid) if hasattr(covar_reader, "sid") else None,
            )

        pvals = _ols_scan_pvals_fast(G_tr, y_tr, cov_tr)
        top_idx = np.argsort(pvals)[:min(int(top_k), m)]
        counts[top_idx] += 1

    df = pd.DataFrame({
        "SNP": sid.astype(str),
        "StabilityCount": counts,
        "StabilityFreq": counts / float(n_reps),
    }).sort_values(
        ["StabilityFreq", "StabilityCount"],
        ascending=False
    ).reset_index(drop=True)

    return df

def _make_run_manifest(
    trait_col: str,
    pheno_label: str,
    vcf_name: str,
    phe_name: str,
    maf_thresh: float,
    mac_thresh: float,
    miss_thresh: float,
    ind_miss_thresh: float,
    drop_alt: bool,
    norm_option: str,
    n_pcs: int,
    sig_rule: str,
    lambda_gc: float,
    n_samples: int,
    n_snps: int,
    qc_snp: dict | None,
    pheno_zero_handling: dict | None = None,
    pheno_transformations: dict | None = None,
    kinship_model: str | None = None,
    loco_fallback_by_chr: dict | None = None,
):
    # Include subsampling summary if available
    subsampling_summary = (
        st.session_state.get("subsampling_gwas_summary", None)
        if st is not None else None
    )
    return {
        "timestamp_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "software_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "analysis_type": "GWAS_panel_LOCO_FaSTLMM",
        "phenotype_file_label": pheno_label,
        "trait": trait_col,
        "inputs": {"vcf_filename": vcf_name, "phenotype_filename": phe_name},
        "qc_thresholds": {
            "maf_thresh": float(maf_thresh),
            "mac_thresh": float(mac_thresh),
            "snp_missingness_max": float(miss_thresh),
            "individual_missingness_max": float(ind_miss_thresh),
            "drop_alt": bool(drop_alt),
        },
        "phenotype_model_normalization": str(norm_option),
        "phenotype_preprocessing": {
            "zero_handling": pheno_zero_handling or {},
            "transformations_section_2b": pheno_transformations or {},
        },
        "structure_covariates": {"n_pcs": int(n_pcs)},
        "significance_rule": str(sig_rule),
        "diagnostics": {
            "lambda_gc_bulk_5_95": None if lambda_gc is None or not np.isfinite(lambda_gc) else float(lambda_gc),
            "lambda_gc_definition": "median chi-square using 5–95% bulk of p-values",
            "kinship_model": kinship_model,
            "loco_fallback_by_chr": loco_fallback_by_chr or {},
            "n_samples_used": int(n_samples),
            "n_snps_used": int(n_snps),
            "qc_snp_breakdown": qc_snp or {},
        },
        "subsampling_gwas_resampling": subsampling_summary,
    }


def ld_block_stability_screen(
    gwas_df,
    chroms,
    positions,
    geno_imputed,
    sid,
    ld_threshold=0.6,
    flank_kb=300,
    min_snps=3,
    top_n=10,
    sig_thresh=1e-5,
    max_dist_bp=None,
    ld_decay_kb=None,
    adj_r2_min=0.2,
    n_reps=50,
    sample_frac=0.8,
    iou_match=0.5,
    seed=0,
    gap_factor=10.0,
    pheno_df=None,
    geno_df=None,
    trait_col=None,
):
    """
    Stability screen for LD blocks using subsampling.

    Parameters
    ----------
    gap_factor : float
        Gap factor for block detection.
    pheno_df : pd.DataFrame or None
        Phenotype table for sample alignment.
    geno_df : pd.DataFrame or None
        Genotype DataFrame for index alignment.
    trait_col : str or None
        Trait column name.
    """

    rng = np.random.default_rng(int(seed))

    # Reference blocks on full data
    ref = find_ld_clusters_genomewide(
        gwas_df=gwas_df,
        chroms=chroms,
        positions=positions,
        geno_imputed=np.asarray(geno_imputed, float),
        sid=sid,
        ld_threshold=ld_threshold,
        flank_kb=flank_kb,
        min_snps=min_snps,
        top_n=top_n,
        sig_thresh=sig_thresh,
        max_dist_bp=max_dist_bp,
        ld_decay_kb=ld_decay_kb,
        adj_r2_min=adj_r2_min,
        gap_factor=float(gap_factor),
    )

    if ref is None or ref.empty:
        return pd.DataFrame()

    ref_int = _blocks_to_interval_set(ref)

    counts = np.zeros(len(ref_int), dtype=int)
    lead_retention = np.zeros(len(ref_int), dtype=int)

    ref_leads = [
        set(_split_leads(r.get("Lead SNP", "")))
        for _, r in ref.iterrows()
    ]

    # Align genotype to phenotype if provided
    if (pheno_df is not None and isinstance(pheno_df, pd.DataFrame) and not pheno_df.empty
            and geno_df is not None and isinstance(geno_df, pd.DataFrame) and not geno_df.empty):
        if trait_col is None:
            trait_col = pheno_df.columns[0]

        trait_col = resolve_trait_column(trait_col, pheno_df)
        _, _, keep_mask = align_pheno_to_geno(pheno_df, geno_df, trait_col)
        geno_imputed = np.asarray(geno_imputed)[keep_mask, :]

    n = geno_imputed.shape[0]
    k = max(20, int(round(sample_frac * n)))
    k = min(k, n)

    if k < 2:
        return pd.DataFrame()

    for _ in range(int(n_reps)):
        idx = rng.choice(n, size=k, replace=False)
        geno_sub = geno_imputed[idx, :]

        blk = find_ld_clusters_genomewide(
            gwas_df=gwas_df,
            chroms=chroms,
            positions=positions,
            geno_imputed=geno_sub,
            sid=sid,
            ld_threshold=ld_threshold,
            flank_kb=flank_kb,
            min_snps=min_snps,
            top_n=top_n,
            sig_thresh=sig_thresh,
            max_dist_bp=max_dist_bp,
            ld_decay_kb=ld_decay_kb,
            adj_r2_min=adj_r2_min,
            gap_factor=float(gap_factor),
        )

        blk_int = _blocks_to_interval_set(blk)

        for i, (ch, s, e) in enumerate(ref_int):
            matched = False

            for (ch2, s2, e2) in blk_int:
                if ch2 != ch:
                    continue

                if _interval_iou_bp(s, e, s2, e2) >= float(iou_match):
                    matched = True

                    # Lead SNP retention
                    mask = (
                        (np.asarray(chroms).astype(str) == ch2)
                        & (np.asarray(positions) >= s2)
                        & (np.asarray(positions) <= e2)
                    )
                    snps_here = set(np.asarray(sid)[mask])

                    if ref_leads[i].intersection(snps_here):
                        lead_retention[i] += 1

                    break

            if matched:
                counts[i] += 1

    out = ref.copy()
    out["StabilityCount"] = counts
    out["StabilityFreq"] = counts / float(n_reps)
    out["LeadRetentionFreq"] = lead_retention / float(n_reps)

    return out.sort_values(
        ["StabilityFreq", "StabilityCount"],
        ascending=False,
    ).reset_index(drop=True)