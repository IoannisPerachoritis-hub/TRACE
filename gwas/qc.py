import logging
import numpy as np
import pandas as pd
try:
    import streamlit as st
except ImportError:
    st = None
from sklearn.impute import SimpleImputer
import allel
from gwas.io import load_vcf_cached, _clean_chr_series, _extract_info_scores

log = logging.getLogger(__name__)


def _log_or_warn(msg: str) -> None:
    """Emit a Streamlit warning if available, otherwise log."""
    if st is not None:
        try:
            st.warning(msg)
        except Exception:
            log.warning(msg)
    else:
        log.warning(msg)


def allele_freq_from_called_dosage(G_dos):
    """
    Compute allele frequency p from CALLED genotypes only.
    G_dos is dosage matrix (n x m) with NaNs allowed.
    Returns p (length m), with NaN where no calls exist.
    """
    G = np.asarray(G_dos, float)
    G = np.clip(G, 0.0, 2.0)  # defensive: ensure dosage in [0,2]
    called = np.isfinite(G)
    n_called = called.sum(axis=0).astype(float)
    AC = np.nansum(G, axis=0)  # sum dosage (ALT allele count across samples)
    AN = 2.0 * n_called
    p = np.divide(AC, AN, out=np.full_like(AC, np.nan, dtype=float), where=(AN > 0))
    return p

def valid_af_mask_from_called_dosage(G_dos: np.ndarray) -> np.ndarray:
    """
    Returns boolean mask of SNPs with valid allele frequency estimates from called genotypes:
      - finite p
      - 0 < p < 1
      - at least 2 called genotypes (AN >= 4) (conservative; avoids weird edge cases)
    """
    G = np.asarray(G_dos, float)
    called = np.isfinite(G)
    n_called = called.sum(axis=0).astype(float)
    AN = 2.0 * n_called

    p = allele_freq_from_called_dosage(G)
    valid = np.isfinite(p) & (p > 0.0) & (p < 1.0) & (AN >= 4.0)
    return valid


# ============================================================
# Pipeline sub-functions (pure computation, no Streamlit)
# ============================================================

def _pipeline_parse_vcf_biallelic(callset):
    """
    Parse VCF callset and enforce biallelic sites.

    Returns
    -------
    genotypes : ndarray (n_samples, n_snps), float32 dosage with NaN for missing
    samples   : ndarray of str
    chroms    : ndarray of str
    positions : ndarray of int
    vid, ref, alt : ndarray or None (filtered metadata)
    info_scores   : ndarray float32 or None (imputation quality per SNP)
    info_field    : str or None (name of the INFO field found)
    """
    if (callset is None
            or callset.get("calldata/GT") is None
            or callset["calldata/GT"].shape[0] == 0):
        raise ValueError(
            "VCF file contains no variant records. "
            "Check that the file has data lines below the #CHROM header."
        )

    samples = callset["samples"]
    chroms = callset["variants/CHROM"]
    positions = callset["variants/POS"]
    gt = callset["calldata/GT"]

    alt_raw = callset.get("variants/ALT", None)

    def _count_nonempty(row):
        k = 0
        for a in row:
            if a is None:
                continue
            if isinstance(a, (bytes, bytearray)):
                if len(a) == 0:
                    continue
            else:
                s = str(a).strip()
                if s in ("", ".", "none", "None", "nan"):
                    continue
            k += 1
        return k

    if alt_raw is not None:
        alt_arr = np.asarray(alt_raw)

        if alt_arr.ndim == 1:
            def _is_one_alt(a):
                if a is None:
                    return False
                if isinstance(a, (bytes, bytearray)):
                    a = a.decode("utf-8", errors="ignore")
                s = str(a).strip()
                if s in ("", ".", "none", "None", "nan"):
                    return False
                return ("," not in s)

            is_biallelic = np.array([_is_one_alt(a) for a in alt_arr], dtype=bool)
        else:
            n_alt = np.array([_count_nonempty(r) for r in alt_arr], dtype=int)
            is_biallelic = (n_alt == 1)
    else:
        is_biallelic = np.ones(len(chroms), dtype=bool)

    chroms = chroms[is_biallelic]
    positions = positions[is_biallelic]
    gt = gt[is_biallelic]

    G = allel.GenotypeArray(gt).to_n_alt().astype("float32")
    G[G < 0] = np.nan

    genotypes = G.T
    samples = samples.astype(str)
    chroms = chroms.astype(str)
    positions = positions.astype(int)

    vid = callset.get("variants/ID", None)
    ref = callset.get("variants/REF", None)
    alt = callset.get("variants/ALT", None)

    if vid is not None:
        vid = np.asarray(vid)[is_biallelic]
    if ref is not None:
        ref = np.asarray(ref)[is_biallelic]
    if alt is not None:
        alt = np.asarray(alt)[is_biallelic]

    # Extract imputation quality scores (if present in VCF)
    info_scores, info_field = _extract_info_scores(callset)
    if info_scores is not None:
        info_scores = info_scores[is_biallelic]

    return genotypes, samples, chroms, positions, vid, ref, alt, info_scores, info_field


def _pipeline_harmonize_ids(genotypes, samples, chroms, positions, vid, ref, alt,
                            pheno, info_scores=None):
    """
    Build SNP IDs, create genotype DataFrame, deduplicate split
    multi-allelic records, and harmonize genotype/phenotype sample IDs.

    Returns
    -------
    geno_df      : DataFrame (samples x SNPs)
    pheno        : DataFrame (aligned to geno_df)
    chroms       : ndarray of str
    positions    : ndarray of int
    sid          : ndarray of str
    info_scores  : ndarray float32 or None (filtered to match surviving SNPs)
    allele_map   : dict {sid_str: (ref_str, alt_str)}
    """
    sid = []
    _ref_list = []
    _alt_list = []
    for i, (c, p) in enumerate(zip(chroms, positions)):
        rs = None
        if vid is not None:
            v = vid[i]
            if isinstance(v, (bytes, bytearray)):
                v = v.decode("utf-8", errors="ignore")
            if v is not None:
                v = str(v).strip()
                if v != "" and v != "." and v.lower() != "none":
                    rs = v

        # Extract ref/alt alleles for this variant
        _r = "."
        _a = "."
        if ref is not None:
            rr = ref[i]
            if isinstance(rr, (bytes, bytearray)):
                rr = rr.decode("utf-8", errors="ignore")
            _r_tmp = str(rr).strip()
            if _r_tmp not in ("", ".", "None", "none"):
                _r = _r_tmp
        if alt is not None:
            aa = alt[i]
            for x in np.atleast_1d(aa):
                if x is None:
                    continue
                if isinstance(x, (bytes, bytearray)):
                    if len(x) == 0:
                        continue
                    x = x.decode("utf-8", errors="ignore")
                s_alt = str(x).strip()
                if s_alt not in ("", ".", "None", "none", "nan"):
                    _a = s_alt
                    break

        if rs is not None:
            sid.append(rs)
        else:
            r = _r if _r != "." else "N"
            a = _a if _a != "." else "N"
            sid.append(f"{c}_{p}_{r}_{a}_v{i}")
        _ref_list.append(_r)
        _alt_list.append(_a)

    sid = np.asarray(sid, dtype=str)
    if len(np.unique(sid)) != len(sid):
        sid = np.array([f"{s}_{i}" for i, s in enumerate(sid)], dtype=str)
        if pd.Series(sid).duplicated().any():
            sid = np.asarray([f"{s}__v{i}" for i, s in enumerate(sid)], dtype=str)

    geno_df = pd.DataFrame(genotypes, index=samples, columns=sid)

    # Filter split multi-allelic records (keep highest MAF per chr:pos)
    chrom_pos_key = pd.Series(
        [f"{c}:{p}" for c, p in zip(chroms, positions)],
        index=geno_df.columns,
    )
    if chrom_pos_key.duplicated().any():
        _n_before_multi = len(chrom_pos_key)
        dos_tmp = geno_df.astype(float).values
        n_called = np.sum(np.isfinite(dos_tmp), axis=0).astype(float)
        ac = np.nansum(dos_tmp, axis=0)
        an = 2.0 * n_called
        af = np.divide(ac, an, out=np.full_like(ac, np.nan, dtype=float), where=(an > 0))
        maf_tmp = np.minimum(af, 1.0 - af)

        keep_idx = []
        for key in chrom_pos_key.unique():
            mask = np.where(chrom_pos_key.values == key)[0]
            if len(mask) == 1:
                keep_idx.append(mask[0])
            else:
                best = mask[np.argmax(maf_tmp[mask])]
                keep_idx.append(best)

        keep_idx = np.sort(keep_idx)
        _n_multi_removed = _n_before_multi - len(keep_idx)
        if _n_multi_removed > 0:
            _log_or_warn(
                f"{_n_multi_removed} multi-allelic records collapsed "
                f"(kept highest-MAF allele per chr:pos)."
            )
        geno_df = geno_df.iloc[:, keep_idx]
        chroms = chroms[keep_idx]
        positions = positions[keep_idx]
        sid = sid[keep_idx]
        _ref_list = [_ref_list[j] for j in keep_idx]
        _alt_list = [_alt_list[j] for j in keep_idx]
        if info_scores is not None:
            info_scores = info_scores[keep_idx]

    # Harmonize sample IDs
    geno_ids = pd.Series(geno_df.index.astype(str)).str.strip()
    pheno_ids = pd.Series(pheno.index.astype(str)).str.strip()

    if geno_ids.duplicated().any():
        geno_df = geno_df[~geno_ids.duplicated()].copy()
        geno_ids = pd.Series(geno_df.index.astype(str))

    if pheno_ids.duplicated().any():
        pheno = pheno.loc[~pheno_ids.duplicated()].copy()

    # Force numpy string dtype on indices — pd.Series.values can return
    # ArrowStringArray on newer pandas+pyarrow, which propagates through
    # gwas_pipeline()'s @st.cache_data return and breaks downstream caching.
    geno_df.index = np.asarray(geno_ids.to_numpy(dtype=object), dtype=str)
    pheno_orig = pheno.copy()  # keep before intersection for fallback matching

    common_ids = geno_df.index.intersection(pheno.index)

    if len(common_ids) == 0:
        # Try normalizing numeric IDs (strip leading zeros)
        geno_norm = geno_ids.str.lstrip("0").replace("", "0")
        pheno_norm = pheno_ids.str.lstrip("0").replace("", "0")
        geno_df_try = geno_df.copy()
        geno_df_try.index = np.asarray(geno_norm.to_numpy(dtype=object), dtype=str)
        pheno_try = pheno_orig.copy()
        pheno_try.index = np.asarray(pheno_norm.to_numpy(dtype=object), dtype=str)
        common_norm = geno_df_try.index.intersection(pheno_try.index)
        if len(common_norm) > 0:
            _log_or_warn(
                "Sample IDs matched after stripping leading zeros "
                f"(e.g., VCF '{geno_ids.iloc[0]}' -> phenotype "
                f"'{pheno_ids.iloc[0]}'). "
                f"Proceeding with {len(common_norm)} matched samples."
            )
            geno_df = geno_df_try.loc[common_norm]
            pheno = pheno_try.loc[common_norm]
            common_ids = common_norm
        else:
            raise ValueError(
                "No overlapping sample IDs between genotype and phenotype files. "
                "Check that accession names match exactly (case-sensitive). "
                f"VCF samples (first 5): {geno_ids.head().tolist()} | "
                f"Phenotype IDs (first 5): {pheno_ids.head().tolist()}"
            )
    else:
        geno_df = geno_df.loc[common_ids]
        pheno = pheno.loc[common_ids]

    # Partial overlap warning
    n_geno = len(geno_ids)
    n_pheno = len(pheno_ids)
    n_common = len(common_ids)
    if n_common < min(n_geno, n_pheno):
        pct = n_common / max(n_geno, n_pheno) * 100
        _log_or_warn(
            f"Partial sample overlap: {n_common} of {n_geno} VCF samples matched "
            f"{n_common} of {n_pheno} phenotype samples ({pct:.0f}% overlap)."
        )

    allele_map = {s: (_ref_list[j], _alt_list[j]) for j, s in enumerate(sid)}
    return geno_df, pheno, chroms, positions, sid, info_scores, allele_map


def _pipeline_phenotype_qc(geno_df, pheno, trait_col, norm_option, ind_miss_thresh):
    """
    Extract phenotype vector, filter NaN samples, apply sample-level
    missingness threshold, and normalize.

    Returns
    -------
    geno_df : DataFrame (filtered)
    pheno   : DataFrame (filtered, aligned)
    y       : ndarray (n, 1)
    """
    # Early detection of non-numeric trait values
    raw_values = pheno[trait_col]
    _n_numeric = pd.to_numeric(raw_values, errors="coerce").notna().sum()
    if _n_numeric == 0:
        raise ValueError(
            f"Trait column '{trait_col}' contains no numeric values "
            f"(first 5: {raw_values.head().tolist()}). "
            "Ensure the trait column contains numeric measurements, not categories."
        )
    if _n_numeric < len(raw_values):
        _n_non = len(raw_values) - _n_numeric
        _log_or_warn(
            f"{_n_non} non-numeric values in trait '{trait_col}' will be treated as missing."
        )

    y = pd.to_numeric(pheno[trait_col], errors="coerce").values.reshape(-1, 1)

    # Drop samples with NaN phenotype first, then check variance
    keep_non_nan = ~np.isnan(y).ravel()
    y = y[keep_non_nan]
    geno_df = geno_df.iloc[keep_non_nan, :]
    pheno = pheno.iloc[keep_non_nan, :]

    if np.nanstd(y) < 1e-8:
        raise ValueError(
            f"Trait '{trait_col}' has near-zero variance after filtering. "
            "GWAS is not meaningful."
        )

    pheno = pheno.reindex(geno_df.index)

    sample_missing = geno_df.isna().mean(axis=1)
    keep_samples = sample_missing < ind_miss_thresh

    geno_df = geno_df.loc[keep_samples]
    pheno = pheno.loc[keep_samples]

    if geno_df.shape[0] == 0:
        raise ValueError(
            f"All samples removed by missingness filter (threshold={ind_miss_thresh}). "
            "Try a less strict threshold."
        )

    # Rebuild phenotype vector from aligned dataframe
    y = pd.to_numeric(pheno[trait_col], errors="coerce").to_numpy().reshape(-1, 1)

    # Drop any phenotype NaNs introduced by reindex/filtering
    keep_y = ~np.isnan(y).ravel()
    geno_df = geno_df.iloc[keep_y, :]
    pheno = pheno.iloc[keep_y, :]
    y = y[keep_y, :]

    if not geno_df.index.equals(pheno.index):
        raise RuntimeError("Genotype and phenotype indices misaligned after filtering.")
    if geno_df.shape[0] != y.shape[0]:
        raise RuntimeError("Genotype and phenotype lengths differ after filtering.")

    # Normalization
    if norm_option == "Z-score (mean=0, sd=1)":
        y_mean = float(np.nanmean(y))
        y_std = float(np.nanstd(y)) or 1.0
        y = (y - y_mean) / y_std
    elif norm_option == "Min–Max scaling (0–1)":
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
        if y_max > y_min:
            y = (y - y_min) / (y_max - y_min)
    elif norm_option == "Log transform (if positive)" and np.all(y > 0):
        y = np.log10(y)
    elif norm_option == "Yeo–Johnson (robust Box–Cox)":
        from scipy.stats import yeojohnson
        y_trans, _ = yeojohnson(y.flatten())
        y = y_trans.reshape(-1, 1)
    elif norm_option == "Rank-based inverse normal (INT)":
        from gwas.utils import _rank_int_1d
        y = _rank_int_1d(y.ravel()).reshape(-1, 1)

    return geno_df, pheno, y


def _pipeline_snp_qc(geno_df, chroms, positions, sid,
                      maf_thresh, miss_thresh, mac_thresh, drop_alt,
                      info_scores=None, info_thresh=0.0,
                      canonical=None):
    """
    SNP-level QC: MAF, missingness, MAC, imputation quality;
    chromosome label cleaning; optional ALT chromosome removal;
    genomic coordinate sorting.

    Returns
    -------
    geno_df        : DataFrame (filtered, sorted)
    chroms         : ndarray of str
    chroms_num     : ndarray of int
    positions      : ndarray of int
    sid            : ndarray of str
    n_initial_snps : int
    qc_snp         : dict with QC breakdown
    info_scores    : ndarray float32 or None (filtered, sorted)
    """
    n_initial_snps = geno_df.shape[1]

    missing_rate = geno_df.isna().mean(axis=0)

    dos = geno_df.astype(float).values
    dos = np.where(np.isfinite(dos), np.clip(dos, 0.0, 2.0), np.nan)
    AC = np.nansum(dos, axis=0)
    n_called = np.sum(~np.isnan(dos), axis=0)
    AN = 2.0 * n_called
    AF = np.divide(AC, AN, out=np.full_like(AC, np.nan, dtype=float), where=(AN > 0))
    maf = np.minimum(AF, 1.0 - AF)
    mac = np.minimum(AC, AN - AC)

    # Imputation quality filter
    has_info = (info_scores is not None and info_thresh > 0
                and len(info_scores) == len(maf))
    if has_info:
        info_fail = ~np.isfinite(info_scores) | (info_scores < info_thresh)
    else:
        info_fail = np.zeros(len(maf), dtype=bool)

    qc_snp = {
        "Total SNPs": int(len(maf)),
        "Fail MAF": int((maf <= maf_thresh).sum()),
        "Fail Missingness": int((missing_rate >= miss_thresh).sum()),
        "Fail MAC": int((mac < mac_thresh).sum()),
        "Fail INFO": int(info_fail.sum()),
    }

    keep_mask = (
            (maf > maf_thresh) &
            (missing_rate < miss_thresh) &
            (mac >= mac_thresh) &
            (~info_fail)
    )
    keep_mask = np.asarray(keep_mask, dtype=bool)
    qc_snp["Fail ANY"] = int((~keep_mask).sum())
    qc_snp["Pass ALL"] = int(keep_mask.sum())

    if keep_mask.sum() == 0:
        raise ValueError("All SNPs failed QC with current thresholds.")

    geno_df = geno_df.loc[:, keep_mask]
    chroms = chroms[keep_mask]
    positions = positions[keep_mask]
    sid = sid[keep_mask]
    if info_scores is not None:
        info_scores = info_scores[keep_mask]

    chroms_raw = chroms.copy()
    chr_label, chr_num, chr_order_labels, chr_order_map = _clean_chr_series(chroms, canonical=canonical)
    chroms = chr_label.astype(str)
    chroms_num = chr_num.astype(int)

    # ── Guard: unrecognized chromosome names ──────────────────
    n_alt = int((chroms == "ALT").sum())
    qc_snp["ALT chromosomes"] = n_alt

    if n_alt > 0 and n_alt == len(chroms):
        sample = list(pd.Series(chroms_raw).unique()[:10])
        raise ValueError(
            "None of the chromosome names in your VCF were recognized as "
            "numeric chromosomes. TRACE strips common species prefixes and "
            "keeps entries that resolve to integers (e.g., 'chr1' -> '1', "
            "'SL4.0ch01' -> '1').\n\n"
            f"Your VCF contains these CHROM values: {sample}\n\n"
            "If your species uses non-numeric chromosome names (e.g., wheat "
            "'1A'/'1B'/'1D'), recode them to integers before loading."
        )

    if drop_alt:
        keep = chroms != "ALT"
        geno_df = geno_df.loc[:, keep]
        chroms = chroms[keep]
        chroms_num = chroms_num[keep]
        positions = positions[keep]
        sid = sid[keep]
        if info_scores is not None:
            info_scores = info_scores[keep]

        if len(chroms) == 0:
            sample = list(pd.Series(chroms_raw).unique()[:10])
            raise ValueError(
                "No SNPs remain after dropping non-numeric chromosomes. "
                "All variants were on unrecognized chromosome names.\n\n"
                f"Original CHROM values (sample): {sample}\n\n"
                "Uncheck 'Drop non-numeric chromosomes' to keep them, "
                "or recode your VCF chromosomes to integers."
            )

    # Sort by ChrNum / Pos
    ord_idx = np.lexsort((positions.astype(int), chroms_num.astype(int)))
    geno_df = geno_df.iloc[:, ord_idx]
    chroms = chroms[ord_idx]
    chroms_num = chroms_num[ord_idx]
    positions = positions[ord_idx]
    sid = sid[ord_idx]
    if info_scores is not None:
        info_scores = info_scores[ord_idx]

    return geno_df, chroms, chroms_num, positions, sid, n_initial_snps, qc_snp, info_scores


def _pipeline_build_geno_matrices(geno_df):
    """
    Build raw dosage matrix (with NaN for LD/haplotype), imputation rate,
    mean-imputed matrix (for GWAS), and IID array.

    Returns
    -------
    geno_dosage_raw    : ndarray float32 (n_samples, n_snps), NaN for missing
    snp_imputation_rate: ndarray float32 (n_snps,)
    geno_imputed       : ndarray float32 (n_samples, n_snps), mean-imputed
    iid                : ndarray str (n_samples, 2)
    """
    geno_dosage_raw = geno_df.astype("float32").to_numpy()
    snp_imputation_rate = geno_df.isna().mean(axis=0).values.astype(np.float32)

    # Mean imputation ONLY for GWAS mixed model
    # NOTE: mean imputation biases effect estimates toward zero for high-missingness SNPs.
    # For publication, consider upstream imputation (Beagle/STITCH) for panels with >5% missingness.
    imputer = SimpleImputer(strategy="mean")
    geno_imputed = imputer.fit_transform(geno_df).astype("float32")

    iid = np.c_[geno_df.index.values, geno_df.index.values].astype(str)

    return geno_dosage_raw, snp_imputation_rate, geno_imputed, iid



def _pipeline_build_kinship(geno_df, geno_imputed, chroms, positions):
    """
    Compute allele frequencies from called genotypes, standardize,
    LD-prune, and build GRM via blockwise Z @ Z.T / m.

    Returns
    -------
    K              : ndarray float32 (n, n) kinship matrix
    Z_for_pca      : ndarray float32 (n, m_pruned) LD-pruned Z matrix
    chroms_grm     : ndarray of str (m_pruned,)
    positions_grm  : ndarray of int (m_pruned,)
    kinship_model  : str
    """
    G_called = geno_df.astype("float32").to_numpy()
    p = allele_freq_from_called_dosage(G_called)
    valid_af = valid_af_mask_from_called_dosage(G_called)

    geno_for_grm = geno_imputed[:, valid_af]
    p_grm = p[valid_af]

    denom = np.sqrt(2.0 * p_grm * (1.0 - p_grm))
    denom[~np.isfinite(denom) | (denom == 0)] = 1.0

    Z = (geno_for_grm - 2.0 * p_grm) / denom
    Z = np.nan_to_num(Z, nan=0.0).astype(np.float32, copy=False)

    chroms_for_grm = np.asarray(chroms)[valid_af]
    positions_for_grm = np.asarray(positions)[valid_af]

    from gwas.kinship import _ld_prune_for_grm_by_chr_bp
    Z_pruned, keep_prune = _ld_prune_for_grm_by_chr_bp(
        chroms_for_grm, positions_for_grm, Z,
        r2_thresh=0.2, window_bp=500_000, step_bp=100_000,
        return_mask=True
    )

    var_ok = np.nanvar(Z_pruned, axis=0) > 0
    Z_grm = Z_pruned[:, var_ok].astype(np.float32, copy=False)

    if not np.isfinite(Z_grm).all():
        Z_grm = np.nan_to_num(Z_grm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    chroms_pruned = np.asarray(chroms_for_grm)[keep_prune]
    positions_pruned = np.asarray(positions_for_grm)[keep_prune]
    chroms_grm = chroms_pruned[var_ok]
    positions_grm = positions_pruned[var_ok]

    m = int(Z_grm.shape[1])
    if m < 10 or not np.isfinite(m):
        K = np.eye(Z_grm.shape[0], dtype=np.float32)
        kinship_model = "identity_GRM_low_snp_count"
    else:
        from gwas.kinship import _build_grm_from_Z
        K = _build_grm_from_Z(Z_grm)
        kinship_model = "LD_pruned_Z_over_m"

    diag_mean = float(np.nanmean(np.diag(K)))
    if np.isfinite(diag_mean) and diag_mean > 1e-8:
        K /= diag_mean

    return K, Z_grm, chroms_grm, positions_grm, kinship_model


# ============================================================
# Cached orchestrator (thin Streamlit wrapper)
# ============================================================

def gwas_pipeline(
    vcf_hash: str,
    pheno_hash: str,
    trait_col,
    maf_thresh,
    miss_thresh,
    norm_option,
    drop_alt,
    ind_miss_thresh,
    mac_thresh,
    info_thresh=0.0,
    canonical=None,
):
    """
    Cached preprocessing pipeline.
    Inputs are retrieved from st.session_state via stable hashes
    to avoid caching on huge bytes/DataFrames.
    """
    # -- Streamlit coupling: resolve session state --
    vcf_bytes = st.session_state.get(f"VCF_BYTES::{vcf_hash}", None)
    pheno = st.session_state.get(f"PHENO_DF::{pheno_hash}", None)

    if vcf_bytes is None:
        raise RuntimeError("VCF bytes missing from session_state (hash key not found).")
    if pheno is None:
        raise RuntimeError("Phenotype table missing from session_state (hash key not found).")

    # -- Pure computation stages --
    callset = load_vcf_cached(vcf_bytes)

    genotypes, samples, chroms, positions, vid, ref, alt, info_scores, info_field = \
        _pipeline_parse_vcf_biallelic(callset)

    geno_df, pheno, chroms, positions, sid, info_scores, _ = \
        _pipeline_harmonize_ids(genotypes, samples, chroms, positions,
                                vid, ref, alt, pheno, info_scores)

    geno_df, pheno, y = \
        _pipeline_phenotype_qc(geno_df, pheno, trait_col,
                               norm_option, ind_miss_thresh)

    geno_df, chroms, chroms_num, positions, sid, n_initial_snps, qc_snp, info_scores = \
        _pipeline_snp_qc(geno_df, chroms, positions, sid,
                         maf_thresh, miss_thresh, mac_thresh, drop_alt,
                         info_scores, info_thresh, canonical=canonical)

    geno_dosage_raw, snp_imputation_rate, geno_imputed, iid = \
        _pipeline_build_geno_matrices(geno_df)

    K, Z_for_pca, chroms_grm, positions_grm, kinship_model = \
        _pipeline_build_kinship(geno_df, geno_imputed, chroms, positions)

    return {
        "geno_imputed": geno_imputed,
        "geno_dosage_raw": geno_dosage_raw,
        "snp_imputation_rate": snp_imputation_rate,
        "chroms": chroms,
        "chroms_num": chroms_num,
        "positions": positions,
        "sid": sid,
        "iid": iid,
        "K": K,
        "pcs": None,
        "geno_df": geno_df,
        "pheno": pheno,
        "y": y,
        "n_snps_raw": int(n_initial_snps),
        "qc_snp": qc_snp,
        "Z_for_pca": Z_for_pca,
        "chroms_grm": chroms_grm,
        "positions_grm": positions_grm,
        "kinship_model": kinship_model,
        "info_scores": info_scores,
        "info_field": info_field,
    }

if st is not None:
    gwas_pipeline = st.cache_data(
        show_spinner="Preparing data & running GWAS (cached)…", max_entries=8
    )(gwas_pipeline)
