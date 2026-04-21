
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2
from gwas.utils import PhenoData, CovarData
from gwas.kinship import _build_grm_from_Z


# ============================================================
# Single subsampling iteration (module-level for picklability)
# ============================================================
def _subsample_one_rep(
    rep: int,
    idx: np.ndarray,
    geno_imputed: np.ndarray,
    y: np.ndarray,
    iid: np.ndarray,
    Z_for_grm: np.ndarray,
    sid: np.ndarray,
    pos_arr: np.ndarray,
    n_pcs: int,
    discovery_thresh: float,
    use_loco: bool = True,
    chroms_grm: np.ndarray | None = None,
    chroms: np.ndarray | None = None,
):
    """Run a single subsampling GWAS iteration. Returns (rep, pvals, metadata).

    PCs are re-estimated from ``Z_for_grm[idx]`` (the subsample's LD-pruned
    genotype matrix) so that the covariate structure reflects the subsample,
    matching the per-subsample GRM recomputation.
    """
    from fastlmm.association import single_snp
    from pysnptools.snpreader import SnpData
    from pysnptools.kernelreader import KernelData as PSKernelData

    n_snps = geno_imputed.shape[1]
    k_sub = len(idx)

    G_sub = geno_imputed[idx, :]
    y_sub = y[idx]
    iid_sub = iid[idx, :]
    Z_sub = Z_for_grm[idx, :]

    pheno_sub = PhenoData(iid=iid_sub, val=y_sub)

    # Re-estimate PCs from the subsample's LD-pruned Z matrix
    covar_sub = None
    if int(n_pcs) > 0:
        from gwas.kinship import _compute_pcs_full_impl
        pcs_sub, _ = _compute_pcs_full_impl(Z_sub, int(n_pcs))
        if pcs_sub is not None:
            covar_sub = CovarData(
                iid=iid_sub,
                val=pcs_sub,
                names=[f"PC{i+1}" for i in range(int(n_pcs))],
            )

    try:
        if use_loco and chroms_grm is not None and chroms is not None:
            # LOCO: rebuild per-chromosome kernels from subsample
            from gwas.kinship import _build_loco_kernels_impl
            K0_sub, K_by_chr_sub, _ = _build_loco_kernels_impl(
                iid_sub, Z_sub, chroms_grm,
            )
            # Run per-chromosome (like _run_gwas_impl)
            res_parts = []
            for ch in np.unique(chroms):
                mask_ch = chroms == ch
                snps_ch = SnpData(
                    iid=iid_sub, sid=sid[mask_ch],
                    val=G_sub[:, mask_ch], pos=pos_arr[mask_ch],
                )
                K_use = K_by_chr_sub.get(str(ch), K0_sub)
                res_ch = single_snp(
                    test_snps=snps_ch, pheno=pheno_sub,
                    K0=K_use, covar=covar_sub,
                    leave_out_one_chrom=False,
                )
                res_parts.append(res_ch)
            res = pd.concat(res_parts, ignore_index=True).copy()
        else:
            # Global kinship (default)
            K_sub = _build_grm_from_Z(Z_sub)
            K0_sub = PSKernelData(iid=iid_sub, val=K_sub)
            snp_data = SnpData(iid=iid_sub, sid=sid, val=G_sub, pos=pos_arr)
            res = single_snp(
                test_snps=snp_data,
                pheno=pheno_sub,
                K0=K0_sub,
                covar=covar_sub,
                leave_out_one_chrom=False,
            ).copy()
    except Exception as e:
        return (
            rep,
            np.full(n_snps, np.nan, dtype=np.float64),
            {
                "rep": int(rep),
                "n_samples": int(k_sub),
                "status": f"failed: {str(e)[:100]}",
                "lambda_gc": np.nan,
                "n_discoveries": 0,
            },
        )

    snp_col = next(
        (c for c in ["SNP", "sid", "rsid", "Id"] if c in res.columns),
        None,
    )
    if snp_col is None:
        return (
            rep,
            np.full(n_snps, np.nan, dtype=np.float64),
            {
                "rep": int(rep),
                "n_samples": int(k_sub),
                "status": "failed: no SNP column",
                "lambda_gc": np.nan,
                "n_discoveries": 0,
            },
        )
    if snp_col != "SNP":
        res = res.rename(columns={snp_col: "SNP"})

    res["SNP"] = res["SNP"].astype(str)
    res["PValue"] = pd.to_numeric(res["PValue"], errors="coerce")
    res = res.dropna(subset=["SNP", "PValue"])
    res = res.drop_duplicates(subset=["SNP"], keep="first")

    pval_map = dict(zip(res["SNP"].values, res["PValue"].values))
    pvals_rep = np.array(
        [pval_map.get(s, np.nan) for s in sid],
        dtype=np.float64,
    )

    discovered = np.isfinite(pvals_rep) & (pvals_rep < float(discovery_thresh))

    pv_clean = pvals_rep[np.isfinite(pvals_rep) & (pvals_rep > 0) & (pvals_rep <= 1)]
    if pv_clean.size > 50:
        chisq = chi2.isf(np.clip(pv_clean, 1e-300, 1.0), df=1)
        lam = float(np.median(chisq) / 0.454936423119572)
    else:
        lam = np.nan

    meta = {
        "rep": int(rep),
        "n_samples": int(k_sub),
        "status": "ok",
        "lambda_gc": lam,
        "n_discoveries": int(discovered.sum()),
    }
    return (rep, pvals_rep, meta)


# ============================================================
# SUBSAMPLING GWAS STABILITY
#
# Re-runs MLM association on random 80% subsamples with
# per-subsample GRM recomputation. Records per-SNP and
# per-LD-block discovery frequency.
#
# This is a TRUE stability assessment: kinship, PCs, phenotype
# vector, and test statistics all change per iteration.
# ============================================================
def subsample_gwas_resampling(
    geno_imputed: np.ndarray,
    y: np.ndarray,
    sid: np.ndarray,
    chroms: np.ndarray,
    chroms_num: np.ndarray,
    positions: np.ndarray,
    iid: np.ndarray,
    Z_for_grm: np.ndarray,
    pcs_full: np.ndarray | None = None,  # noqa: ARG001 — deprecated, kept for API compat
    n_pcs: int = 4,
    n_reps: int = 50,
    sample_frac: float = 0.80,
    discovery_thresh: float = 1e-4,
    seed: int = 0,
    progress_callback=None,
    n_jobs: int = 1,
    use_loco: bool = True,
    chroms_grm: np.ndarray | None = None,
):
    """
    Stability GWAS via random subsampling (without replacement) and
    per-subsample GRM recomputation.

    Note: This performs subsampling without replacement (not bootstrap).
    Without-replacement subsampling is appropriate for
    assessing signal stability in small panels.

    For each iteration:
      1. Subsample sample_frac of individuals (without replacement)
      2. Recompute GRM from the LD-pruned Z basis for the subsample
      3. Run single_snp MLM (global kinship by default; optional LOCO)
      4. Record which SNPs pass discovery_thresh

    Parameters
    ----------
    geno_imputed : ndarray (n_samples, n_snps)
        Full imputed genotype matrix.
    y : ndarray (n_samples, 1)
        Phenotype vector.
    sid, chroms, chroms_num, positions : arrays
        SNP metadata.
    iid : ndarray (n_samples, 2)
        Sample IDs for FastLMM.
    Z_for_grm : ndarray (n_samples, m_grm)
        LD-pruned standardized genotype matrix for GRM construction.
        Also used for per-subsample PCA (GRM standardization is the
        correct basis for population-structure PCs; Price et al. 2006).
    pcs_full : ndarray or None
        **Deprecated / ignored.** PCs are now re-estimated from
        ``Z_for_grm`` per subsample. Kept for API compatibility.
    n_pcs : int
        Number of PCs to use as covariates.
    n_reps : int
        Number of subsampling iterations.
    sample_frac : float
        Fraction of samples per iteration.
    discovery_thresh : float
        P-value threshold for "discovery" (relaxed for subsamples).
    seed : int
        Random seed.
    progress_callback : callable or None
        Called with (iteration, n_reps) for progress updates.
        Only used when n_jobs=1 (sequential mode).
    n_jobs : int
        Number of parallel workers. 1 = sequential (default).
        -1 = use all available cores.
    use_loco : bool
        Rebuild per-chromosome LOCO kinship for each subsample.
        More accurate but ~12x slower per iteration.
    chroms_grm : ndarray or None
        Chromosome labels aligned to ``Z_for_grm`` columns.
        Required when ``use_loco=True``.

    Returns
    -------
    discovery_df : DataFrame
        Per-SNP discovery frequency + summary statistics.
    raw_pvals : ndarray (n_reps, n_snps)
        Full p-value matrix across iterations (for downstream analysis).
    rep_metadata : list[dict]
        Per-iteration metadata (n_samples, lambda_gc, n_discoveries).
    """
    rng = np.random.default_rng(int(seed))

    geno_imputed = np.asarray(geno_imputed, dtype=np.float32, order="C")
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    sid = np.asarray(sid).astype(str)
    chroms = np.asarray(chroms).astype(str)
    chroms_num = np.asarray(chroms_num).astype(int)
    positions = np.asarray(positions).astype(int)
    iid = np.asarray(iid).astype(str)
    Z_for_grm = np.asarray(Z_for_grm, dtype=np.float32)

    if iid.ndim == 1:
        iid = np.c_[iid, iid]

    n_total, n_snps = geno_imputed.shape
    k_sub = max(20, int(round(sample_frac * n_total)))
    k_sub = min(k_sub, n_total)

    # Pre-build pos array for FastLMM SnpData
    pos_arr = np.c_[
        chroms_num.astype(float),
        np.zeros(n_snps, float),
        positions.astype(float),
    ]

    # Pre-generate all subsample indices for reproducibility
    all_indices = [
        np.sort(rng.choice(n_total, size=k_sub, replace=False))
        for _ in range(int(n_reps))
    ]

    # Storage
    raw_pvals = np.full((int(n_reps), n_snps), np.nan, dtype=np.float64)
    discovery_counts = np.zeros(n_snps, dtype=np.int32)
    rep_metadata = []

    if n_jobs == 1:
        # ---- Sequential path (supports progress callback) ----
        for rep in range(int(n_reps)):
            if progress_callback is not None:
                progress_callback(rep, int(n_reps))

            rep_idx, pvals_rep, meta = _subsample_one_rep(
                rep, all_indices[rep], geno_imputed, y, iid, Z_for_grm,
                sid, pos_arr, n_pcs, discovery_thresh,
                use_loco=use_loco, chroms_grm=chroms_grm, chroms=chroms,
            )
            raw_pvals[rep_idx, :] = pvals_rep
            if meta["status"] == "ok":
                discovered = np.isfinite(pvals_rep) & (pvals_rep < float(discovery_thresh))
                discovery_counts += discovered.astype(np.int32)
            rep_metadata.append(meta)
    else:
        # ---- Parallel path (joblib) ----
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_subsample_one_rep)(
                rep, all_indices[rep], geno_imputed, y, iid, Z_for_grm,
                sid, pos_arr, n_pcs, discovery_thresh,
                use_loco, chroms_grm, chroms,
            )
            for rep in range(int(n_reps))
        )

        for rep_idx, pvals_rep, meta in results:
            raw_pvals[rep_idx, :] = pvals_rep
            if meta["status"] == "ok":
                discovered = np.isfinite(pvals_rep) & (pvals_rep < float(discovery_thresh))
                discovery_counts += discovered.astype(np.int32)
            rep_metadata.append(meta)

    # ============================================================
    # Build per-SNP summary
    # ============================================================
    n_successful = sum(1 for m in rep_metadata if m["status"] == "ok")
    n_successful = max(n_successful, 1)  # avoid /0

    discovery_freq = discovery_counts.astype(float) / float(n_successful)

    # Median and mean p-value across reps (ignoring NaN)
    with np.errstate(all="ignore"):
        median_p = np.nanmedian(raw_pvals, axis=0)
        mean_p = np.nanmean(raw_pvals, axis=0)
        sd_logp = np.nanstd(-np.log10(np.clip(raw_pvals, 1e-300, 1.0)), axis=0)

    # Mean rank across reps
    rank_mat = np.full_like(raw_pvals, np.nan)
    for r in range(raw_pvals.shape[0]):
        row = raw_pvals[r, :]
        finite = np.isfinite(row)
        if finite.sum() > 0:
            rank_mat[r, finite] = stats.rankdata(row[finite], method="average")

    mean_rank = np.nanmean(rank_mat, axis=0)

    discovery_df = pd.DataFrame({
        "SNP": sid,
        "Chr": chroms,
        "Pos": positions,
        "DiscoveryCount": discovery_counts,
        "DiscoveryFreq": discovery_freq,
        "MedianPValue": median_p,
        "MeanPValue": mean_p,
        "SD_log10P": sd_logp,
        "MeanRank": mean_rank,
    }).sort_values("DiscoveryFreq", ascending=False).reset_index(drop=True)

    return discovery_df, raw_pvals, rep_metadata


def aggregate_subsampling_to_ld_blocks(
    discovery_df: pd.DataFrame,
    ld_blocks_df: pd.DataFrame,
    raw_pvals: np.ndarray,
    sid: np.ndarray,
    chroms: np.ndarray,
    positions: np.ndarray,
    discovery_thresh: float = 1e-4,
):
    """
    Aggregate per-SNP subsampling results to LD block level.

    For each LD block:
      - Block discovery frequency: fraction of reps where ANY SNP
        in the block passes discovery_thresh
      - Best SNP discovery frequency within the block
      - Lead SNP stability: does the same lead SNP appear as
        the best SNP across reps?
      - Median of per-rep min-P within the block
      - Number of consistently discovered SNPs (freq > 0.5)

    Parameters
    ----------
    discovery_df : DataFrame from subsample_gwas_resampling
    ld_blocks_df : DataFrame with Chr, Start (bp), End (bp), Lead SNP
    raw_pvals : ndarray (n_reps, n_snps)
    sid, chroms, positions : arrays aligned to raw_pvals columns

    Returns
    -------
    block_stability : DataFrame with block-level stability metrics
    """
    from gwas.haplotype import normalize_ld_blocks_schema

    if ld_blocks_df is None or ld_blocks_df.empty:
        return pd.DataFrame()

    ld_blocks_df = normalize_ld_blocks_schema(ld_blocks_df.copy())

    sid = np.asarray(sid).astype(str)
    chroms = np.asarray(chroms).astype(str)
    positions = np.asarray(positions).astype(int)
    raw_pvals = np.asarray(raw_pvals, dtype=np.float64)

    # Build SNP lookup
    disc_map = dict(zip(
        discovery_df["SNP"].values,
        discovery_df["DiscoveryFreq"].values,
    ))

    results = []

    from gwas.ld import get_block_snp_mask

    for _, block in ld_blocks_df.iterrows():
        ch = str(block["Chr"])
        s = int(block["Start (bp)"])
        e = int(block["End (bp)"])
        lead = str(block.get("Lead SNP", ""))

        # SNPs in this block
        mask = get_block_snp_mask(block, chroms, positions, sid)
        snp_idx = np.where(mask)[0]

        if snp_idx.size == 0:
            results.append({
                "Chr": ch,
                "Start (bp)": s,
                "End (bp)": e,
                "Lead SNP": lead,
                "n_snps_in_block": 0,
                "BlockDiscoveryFreq": 0.0,
                "BestSNP_DiscoveryFreq": 0.0,
                "BestSNP_ID": "",
                "LeadSNP_DiscoveryFreq": 0.0,
                "n_consistent_snps_50pct": 0,
                "MedianMinP_block": np.nan,
                "MeanMinP_block": np.nan,
            })
            continue

        block_sids = sid[snp_idx]
        block_pvals = raw_pvals[:, snp_idx]  # (n_reps, n_block_snps)

        # Block discovery: any SNP in block passes threshold in each rep
        with np.errstate(all="ignore"):
            block_min_p = np.nanmin(block_pvals, axis=1)  # (n_reps,)

        block_discovered = np.isfinite(block_min_p) & (block_min_p < float(discovery_thresh))
        n_valid_reps = np.sum(np.isfinite(block_min_p))
        block_disc_freq = float(block_discovered.sum()) / max(1, float(n_valid_reps))

        # Best SNP by discovery frequency
        snp_freqs = np.array([disc_map.get(s, 0.0) for s in block_sids])
        best_idx = int(np.argmax(snp_freqs))
        best_snp = str(block_sids[best_idx])
        best_freq = float(snp_freqs[best_idx])

        # Lead SNP stability
        lead_freq = float(disc_map.get(lead, 0.0))

        # Consistently discovered SNPs (freq > 50%)
        n_consistent = int(np.sum(snp_freqs > 0.5))

        # Median of per-rep block min-P
        med_minp = float(np.nanmedian(block_min_p))
        mean_minp = float(np.nanmean(block_min_p))

        results.append({
            "Chr": ch,
            "Start (bp)": s,
            "End (bp)": e,
            "Lead SNP": lead,
            "n_snps_in_block": int(snp_idx.size),
            "BlockDiscoveryFreq": block_disc_freq,
            "BestSNP_DiscoveryFreq": best_freq,
            "BestSNP_ID": best_snp,
            "LeadSNP_DiscoveryFreq": lead_freq,
            "n_consistent_snps_50pct": n_consistent,
            "MedianMinP_block": med_minp,
            "MeanMinP_block": mean_minp,
        })

    return pd.DataFrame(results).sort_values(
        "BlockDiscoveryFreq", ascending=False
    ).reset_index(drop=True)