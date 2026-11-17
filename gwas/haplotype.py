import json
import logging
import numpy as np
import pandas as pd
try:
    import streamlit as st
except ImportError:
    st = None
from gwas.utils import _ensure_2d, stable_seed
from gwas.utils import _mean_impute_cols
from gwas.models import _ols_fit, _nested_f_test, _one_hot_drop_first
from gwas.ld import get_block_snp_mask
from gwas.genostore import as_columns as _as_columns   # the table kept on disk

def _inv_simpson_from_counts(counts) -> float:
    """1 / sum(p_i^2) on a value_counts Series (T-31). NaN when empty.

    Inverse Simpson on the pre-collapse MLG frequency vector: the number of
    equally-frequent MLGs that would produce the same homozygosity. Insensitive
    to the singleton tail, unlike the raw MLG count.
    """
    if counts is None or len(counts) == 0:
        return float("nan")
    total = float(np.asarray(counts, dtype=float).sum())
    if total <= 0:
        return float("nan")
    p = np.asarray(counts, dtype=float) / total
    denom = float(np.sum(p * p))
    return (1.0 / denom) if denom > 0 else float("nan")


def _eta2_confint(F, df1, df2, conf=0.95):
    """95% CI for the (partial) haplotype eta^2 via noncentral-F inversion
    (Steiger 2004; MBESS conf.limits.ncf).

    Uses the PARAMETRIC F (F_param), because the inversion assumes the parametric
    noncentral-F sampling distribution; the permutation F cannot supply this interval
    (permutation destroys the association, so its eta^2 sits at the null floor
    ~= df1/(N-1), not around the effect). lam_lo/lam_hi invert ncf.cdf at 0.975/0.025;
    eta^2 = lam / (lam + df1 + df2 + 1). Returns (lo, hi); (nan, nan) when undefined.
    """
    if not (np.isfinite(F) and F > 0 and df1 >= 1 and df2 >= 1):
        return np.nan, np.nan
    from scipy.stats import ncf, f as _f
    from scipy.optimize import brentq
    M = df1 + df2 + 1
    a = 1.0 - conf
    cdf0 = _f.cdf(F, df1, df2)                 # central-F cdf (noncentrality 0)

    def _lam(target):
        if cdf0 < target:                     # even nc=0 is already below the target
            return 0.0
        hi = 1.0
        while ncf.cdf(F, df1, df2, hi) > target:   # ncf.cdf decreases in lam
            hi *= 2.0
            if hi > 1e8:
                return hi
        return brentq(lambda lam: ncf.cdf(F, df1, df2, lam) - target, 0.0, hi, xtol=1e-6)

    lam_lo = _lam(1.0 - a / 2.0)               # 0.975 -> lower eta^2 bound
    lam_hi = _lam(a / 2.0)                     # 0.025 -> upper eta^2 bound
    return lam_lo / (lam_lo + M), lam_hi / (lam_hi + M)


def _omega2_from_f(F, df1, df2):
    """Omega-squared (variance explained, less upward-biased than eta^2 as the group
    count grows): df1*(F-1) / (df1*F + df2 + 1). Uses the parametric F, consistent with
    eta2. NOT clamped -- a block with F < 1 legitimately yields a negative value (the
    haplotype grouping explains less than chance). Reported as a column, never a
    replacement for eta2 (the k-dependent gap is far below the eta2 CI width here)."""
    if not (np.isfinite(F) and df1 >= 1 and df2 >= 1):
        return np.nan
    denom = df1 * F + df2 + 1.0
    return (float(df1) * (F - 1.0)) / denom if denom > 0 else np.nan


def _lead_snp_partial_r2(lead_snp, sid, geno_imputed, sample_ids, test_samples,
                         y_resid, X_pcs):
    """Additive (1 df) partial R^2 of the block's lead SNP dosage on the PC-residualised
    phenotype, over the SAME retained samples used for eta2. 1 df makes it comparable
    across blocks where eta2 (df1 = groups-1) is not. The lead SNP may be a NON-member
    of the block (it is the seeding SNP), so it is looked up in the FULL sid array.
    Returns nan when the lead is empty/absent from sid/monomorphic in the retained set
    or n < 4."""
    if not lead_snp:
        return np.nan
    idx = np.flatnonzero(np.asarray(sid, dtype=str) == str(lead_snp))
    if idx.size == 0:
        return np.nan
    dose = (pd.Series(np.asarray(geno_imputed[:, idx[0]], dtype=float),
                      index=np.asarray(sample_ids, dtype=str))
            .reindex(np.asarray(test_samples, dtype=str)).to_numpy(dtype=float))
    y = np.asarray(y_resid, dtype=float)
    ok = np.isfinite(dose) & np.isfinite(y)
    if int(ok.sum()) < 4:
        return np.nan
    dose, y = dose[ok], y[ok]
    if X_pcs is not None:                      # partial out the same PCs eta2 removed
        Xp = np.asarray(X_pcs, dtype=float)[ok]
        dose = dose - Xp @ np.linalg.lstsq(Xp, dose, rcond=None)[0]
    if np.std(dose) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    r = np.corrcoef(y, dose)[0, 1]
    return float(r * r) if np.isfinite(r) else np.nan


def run_haplotype_block_gwas(
    haplo_df,
    chroms,
    positions,
    geno_imputed,
    sid,
    geno_df,
    pheno_df,
    trait_col,
    pcs=None,
    min_hap_count=5,
    min_group_size=3,
    n_perm=1000,
    geno_encoding="dosage012",
    n_pcs_used=None,
    user_covar=None,
):
    """
    Genome-wide haplotype/MLG association per LD block.

    The parametric test, the Freedman-Lane permutation and the effect sizes all use one
    design: an intercept, the population-structure PCs and the user covariates
    (``user_covar``, aligned to ``geno_df``'s rows; a tested sample without a value raises).
    Effect sizes are reported on the **residualised phenotype** (the phenotype after that
    design is regressed out), consistent with the nested F-test:
    ``eta2`` (variance explained by the haplotype grouping), its noncentral-F 95% CI
    (``eta2_ci_low``/``eta2_ci_high``, from the parametric F), ``omega2`` (a less
    upward-biased variance-explained estimate), and ``lead_snp_r2`` (the 1-df additive
    R^2 of the block's lead SNP, comparable across blocks where eta2 is not because its
    df is fixed at 1).

    Two caveats apply to every haplotype effect size here: (1) ``eta2`` and ``omega2``
    **grow with the number of haplotype groups** (more groups can only raise the
    variance a categorical factor explains), so they are not directly comparable across
    blocks with different group counts; (2) **neither estimate corrects for selection** --
    blocks are tested because they were formed around GWAS-significant seeds, so the
    reported effect sizes are conditional on that selection and are upward-biased as
    estimates of a randomly-chosen block's effect.
    """
    hap_tables = {}
    # ------------------------------------------------------------
    # SAFETY: phenotype / trait validation
    # ------------------------------------------------------------
    if pheno_df is None or not isinstance(pheno_df, pd.DataFrame):
        raise ValueError("pheno_df must be a pandas DataFrame")

    if trait_col not in pheno_df.columns:
        raise KeyError(
            f"Trait '{trait_col}' not found in pheno_df columns.\n"
            f"Available traits: {list(pheno_df.columns)}"
        )

    # --- STRICT genotype–phenotype alignment (safe) ---
    geno_df = geno_df.copy()
    pheno_df = pheno_df.copy()

    geno_df.index = geno_df.index.astype(str)
    pheno_df.index = pheno_df.index.astype(str)

    common_ids = geno_df.index.intersection(pheno_df.index)
    if len(common_ids) == 0:
        raise ValueError("No overlapping genotype/phenotype IDs.")

    keep_mask = geno_df.index.isin(common_ids)

    geno_df = geno_df.loc[keep_mask]
    pheno_df = pheno_df.reindex(geno_df.index)

    geno_imputed = _as_columns(geno_imputed)[keep_mask, :]
    if pcs is not None:
        pcs = np.asarray(pcs)
        if pcs.shape[0] != geno_imputed.shape[0]:
            raise ValueError("PC matrix does not match genotype rows after filtering.")

    # --------------------------------------------------------
    # SAFETY: enforce array types for genomic coordinates
    # --------------------------------------------------------
    if isinstance(chroms, pd.Series):
        chroms = chroms.values
    elif isinstance(chroms, pd.DataFrame):
        chroms = chroms.iloc[:, 0].values
    else:
        chroms = np.asarray(chroms)

    if isinstance(positions, pd.Series):
        positions = positions.values.astype(float)
    elif isinstance(positions, pd.DataFrame):
        positions = positions.iloc[:, 0].values.astype(float)
    else:
        positions = np.asarray(positions, dtype=float)

    if isinstance(sid, pd.Series):
        sid = sid.values
    elif isinstance(sid, pd.DataFrame):
        sid = sid.iloc[:, 0].values
    else:
        sid = np.asarray(sid)

    if haplo_df is None or len(haplo_df) == 0:
        return pd.DataFrame(
            columns=[
                "Chr", "Start", "End", "lead_snp",
                "n_snps", "n_haplotypes", "n_tested_haplotypes",
                "df1", "df2",
                "F_param", "PValue_param",
                "F_perm", "P_perm",
                "PValue",
                "MidPos", "-log10p",
                "mlg_n_observed", "mlg_top1_freq", "mlg_eff_inv_simpson",
                "mlg_n_other", "mlg_frac_retained", "mlg_min_group_n", "n_samples_tested",
            ]
        ), {}

    # Normalize LD block schema
    haplo_df = normalize_ld_blocks_schema(haplo_df)
    # P16: refuse unanchored (ALT) blocks -- their members span concatenated
    # unplaced scaffolds, so asserting linkage/haplotype structure across the bin
    # has no basis (align with MLM/FarmCPU/kinship, which all exclude ALT).
    if "Chr" in haplo_df.columns:
        haplo_df = haplo_df[haplo_df["Chr"].astype(str) != "ALT"].reset_index(drop=True)

    chroms = np.asarray(chroms)
    positions = np.asarray(positions)
    sid = np.asarray(sid).astype(str)

    sample_ids = geno_df.index.astype(str)

    results = []
    # why each block that is not tested was left out ({"Chr:Start-End": reason}); kept in
    # the result's attrs for the GUI's block table, never written to a file
    untested = {}

    # These two lines prevent any UnboundLocalError even in edge cases
    hap_labels = pd.Series(dtype=str)
    hap_counts = pd.Series(dtype=int)

    for _, block in haplo_df.iterrows():

        chr_sel = str(block["Chr"])
        start_bp = int(block["Start (bp)"])
        end_bp = int(block["End (bp)"])
        lead_snp = str(block.get("lead_snp", ""))
        _bkey = f"{chr_sel}:{start_bp}-{end_bp}"

        region_mask = get_block_snp_mask(block, chroms, positions, sid)

        if region_mask.sum() < 2:
            untested[_bkey] = "fewer than 2 SNPs"
            continue

        block_sids = sid[region_mask].astype(str)
        G_block = np.asarray(geno_imputed[:, region_mask], float)

        # drop SNP columns with too many missing
        min_col_n = max(10, int(0.05 * G_block.shape[0]))
        col_ok = np.isfinite(G_block).sum(axis=0) >= min_col_n
        G_block = G_block[:, col_ok]
        block_sids = block_sids[col_ok]

        if G_block.shape[1] < 2:
            untested[_bkey] = "fewer than 2 SNPs pass the call-rate filter"
            continue

        # Allow missingness within block; drop only very incomplete samples
        # (keeps power and reduces structure bias from complete-case filtering)
        # For haplotype string construction, moderate missingness is tolerable
        # since remaining NaNs are mean-imputed before rounding to integer alleles.
        max_missing_frac = 0.30  # 30% missing allowed per sample in the block
        row_missing_frac = np.mean(~np.isfinite(G_block), axis=1)
        row_ok = row_missing_frac <= max_missing_frac

        G_block = G_block[row_ok, :]
        sample_block = sample_ids[row_ok]

        # For remaining NaNs, mean-impute per SNP WITHIN the block
        # (haplotype strings are still hard-called later; this avoids losing samples)
        G_block = _mean_impute_cols(G_block)

        n_samples_block_used = int(G_block.shape[0])

        if G_block.shape[0] < 5:
            untested[_bkey] = "fewer than 5 samples with genotypes"
            continue

        enc = str(geno_encoding)

        if enc == "dosage012":
            G_block_int = np.rint(G_block).astype(np.int8)
        elif enc == "binary01":
            Gb = np.rint(G_block)
            if not np.all(np.isin(Gb[np.isfinite(Gb)], [0, 1])):
                raise ValueError("binary01 encoding has values outside {0,1}.")
            G_block_int = Gb.astype(np.int8)
        elif enc == "dosage02":
            Gb = np.rint(G_block)
            if not np.all(np.isin(Gb[np.isfinite(Gb)], [0, 2])):
                raise ValueError("dosage02 encoding has values outside {0,2}.")
            G_block_int = Gb.astype(np.int8)
        else:
            raise ValueError(f"Unsupported geno_encoding: {enc}")

        # ---- thin redundant SNPs ----
        def _thin_block_snps(Gint, r2_thresh=0.9):
            Gint = np.asarray(Gint, np.int8)
            m = Gint.shape[1]
            if m <= 2:
                return np.arange(m, dtype=int)

            Zb = Gint.astype(float)
            Zb -= Zb.mean(axis=0, keepdims=True)

            v = np.var(Zb, axis=0)
            ok = v > 1e-8
            if ok.sum() < 2:
                return np.arange(min(2, m), dtype=int)

            Zb = Zb[:, ok]
            map_back = np.where(ok)[0]

            C = np.corrcoef(Zb, rowvar=False)
            if not np.isfinite(C).all():
                return map_back[:max(2, len(map_back))]

            R2 = C * C
            keep = np.ones(Zb.shape[1], dtype=bool)

            for i in range(R2.shape[0]):
                if not keep[i]:
                    continue
                high = np.where(R2[i, :] >= r2_thresh)[0]
                high = high[high > i]
                keep[high] = False

            idx = map_back[np.where(keep)[0]]
            if idx.size < 2:
                idx = np.arange(min(2, m), dtype=int)
            return idx

        thin_idx = _thin_block_snps(G_block_int, r2_thresh=0.9)
        G_block_int = G_block_int[:, thin_idx]
        block_sids = block_sids[thin_idx]
        n_snps_block = int(G_block_int.shape[1])

        if n_snps_block < 2:
            untested[_bkey] = "fewer than 2 SNPs after thinning"
            continue

        # haplotype strings
        hap_strings = pd.Series(
            pd.DataFrame(G_block_int, index=sample_block)
            .astype(str)
            .agg("|".join, axis=1)
            .values,
            index=sample_block,
        )

        # align phenotype to haplotype-complete samples
        ph = pheno_df.reindex(hap_strings.index)
        y_vec = pd.to_numeric(ph[trait_col], errors="coerce").astype(float)

        keep = np.isfinite(y_vec.values)
        hap_strings = hap_strings.iloc[keep]
        y_vec = y_vec.iloc[keep]

        if hap_strings.empty:
            untested[_bkey] = "no phenotyped samples"
            continue

        # counts AFTER phenotype filtering
        hap_counts = hap_strings.value_counts()
        keep_haps = hap_counts[hap_counts >= int(min_hap_count)].index
        if len(keep_haps) == 0:
            untested[_bkey] = f"no haplotype in at least {int(min_hap_count)} samples"
            continue

        hap_sorted = hap_counts.loc[keep_haps].sort_values(ascending=False).index

        hap_labels = hap_strings.map(
            {h: f"H{i + 1}" for i, h in enumerate(hap_sorted)}
        ).fillna("Other")

        if hap_labels.empty:
            untested[_bkey] = "no phenotyped samples"
            continue

        # store once (for expression selection)
        mlg_df = pd.DataFrame({
            "Sample": hap_labels.index.astype(str),
            "Haplotype": hap_labels.values.astype(str),
            "Allele_sequence": hap_strings.values,
        })
        block_key = f"{chr_sel}:{start_bp}-{end_bp}"
        hap_tables[block_key] = mlg_df.copy()
        # the markers of Allele_sequence, in its order (after the call-rate filter and
        # the r2 >= 0.9 thinning); read by haplotype_assignments
        hap_tables[block_key]["SNP_order"] = "|".join(block_sids.astype(str))

        # phenotype frame
        haplo_pheno_df = pd.DataFrame({
            "Sample": hap_strings.index,
            "MLG": hap_labels.values,
            trait_col: y_vec.values,
        })

        haplo_pheno_df = haplo_pheno_df.replace({trait_col: {np.inf: np.nan, -np.inf: np.nan}})
        haplo_pheno_df = haplo_pheno_df.dropna(subset=[trait_col])

        if haplo_pheno_df.empty:
            untested[_bkey] = "no phenotyped samples"
            continue

        # test frame
        df_test = haplo_pheno_df.loc[:, ["Sample", "MLG", trait_col]].copy()
        df_test[trait_col] = pd.to_numeric(df_test[trait_col], errors="coerce").astype(float)
        df_test = df_test.replace({trait_col: {np.inf: np.nan, -np.inf: np.nan}})
        df_test = df_test.dropna(subset=[trait_col])

        if df_test.empty:
            untested[_bkey] = "no phenotyped samples"
            continue

        counts = df_test["MLG"].astype(str).value_counts()
        valid_haps = [
            h for h in counts.index
            if (counts[h] >= int(min_group_size) and str(h) != "Other")
        ]
        if len(valid_haps) < 2:
            untested[_bkey] = (f"fewer than 2 haplotype groups of at least "
                               f"{int(min_group_size)} samples")
            continue

        df_test = df_test[df_test["MLG"].isin(valid_haps)].copy()

        # Warn when rare-haplotype filtering discards too many samples
        frac_retained = np.nan  # T-31: always bound (drives the warning; reported as mlg_frac_retained)
        if n_samples_block_used > 0:
            frac_retained = len(df_test) / n_samples_block_used
            if frac_retained < 0.75:
                logging.warning(
                    "Block %s:%s-%s: only %.0f%% of samples retained "
                    "after rare-haplotype filtering (%d/%d).",
                    chr_sel, start_bp, end_bp, frac_retained * 100,
                    len(df_test), n_samples_block_used,
                )

        y_test = df_test[trait_col].values.astype(float)
        # Force numpy str dtype — pd.Series.values can return ArrowStringArray
        # on pandas>=2.0 with pyarrow, which Streamlit cache_data cannot hash.
        g_test = np.asarray(
            df_test["MLG"].astype(str).to_numpy(dtype=object), dtype=str,
        )

        # PCs aligned to df_test samples
        pcs_use = None
        k_used = 0
        if pcs is not None:
            pcs = np.asarray(pcs, float)
            pcs_df = pd.DataFrame(pcs, index=geno_df.index.astype(str))

            pcs_block = pcs_df.reindex(df_test["Sample"].astype(str))
            pcs_block = pcs_block.dropna(axis=0, how="any")

            if pcs_block.shape[0] != df_test.shape[0]:
                df_test = df_test.set_index("Sample").loc[pcs_block.index].reset_index()
                y_test = df_test[trait_col].values.astype(float)
                g_test = np.asarray(
                    df_test["MLG"].astype(str).to_numpy(dtype=object), dtype=str,
                )

            k_target = pcs_block.shape[1] if n_pcs_used is None else int(n_pcs_used)
            k_used = int(min(k_target, pcs_block.shape[1]))
            pcs_use = pcs_block.iloc[:, :k_used].values if k_used > 0 else None

        # Which labelled samples the test below takes (df_test is final here), and why
        # the others are left out; read by haplotype_assignments, changes nothing else.
        _ht = hap_tables[block_key]
        _in_test = _ht["Sample"].astype(str).isin(set(df_test["Sample"].astype(str))).to_numpy()
        _ht["in_haplotype_test"] = _in_test
        _ht["exclusion_reason"] = np.select(
            [_in_test, (_ht["Haplotype"] == "Other").to_numpy(),
             ~_ht["Haplotype"].isin(valid_haps).to_numpy()],
            ["", "rare haplotype (Other)", "group below minimum size"],
            default="missing covariate",
        )

        # Append user covariates AFTER the PC truncation above (n_pcs_used slices only the
        # PC columns, so covariates stacked before it would be sliced off). Aligned to the
        # block's final samples, PCs first then covariates (the _make_covar convention).
        # The parametric test, the permutation, eta2 and lead_snp_r2 all use this one
        # design, [1, PCs, covariates]; with user_covar None it is the PCs alone, as before.
        # The callers drop the samples without a covariate value before the test, so a
        # missing value here is an error, never a silent fall back to the PCs alone.
        _cov_use = pcs_use
        if user_covar is not None:
            _uc = pd.DataFrame(np.asarray(user_covar, float),
                               index=geno_df.index.astype(str))
            _uc_mat = _uc.reindex(df_test["Sample"].astype(str)).to_numpy(dtype=float)
            if not np.isfinite(_uc_mat).all():
                raise ValueError(
                    f"Block {_bkey}: {int((~np.isfinite(_uc_mat)).any(axis=1).sum())} tested "
                    "sample(s) have no value for a user covariate. Samples without a covariate "
                    "value must be dropped before the haplotype test.")
            _cov_use = (np.column_stack([pcs_use, _uc_mat])
                        if pcs_use is not None else _uc_mat)

        try:
            F_param, pval_param, df1, df2 = block_test_lm_with_pcs(
                y=y_test,
                groups=g_test,
                pcs=_cov_use,
            )

            # Seed the Freedman-Lane permutation from the block's coordinates only.
            # The SNP id was dropped (P3): it is a reproducibility handle, not a
            # statistical one (every seed gives an unbiased p-value), and it must not
            # reference the lead_snp column (a p-value-derived quantity) -- otherwise
            # any QC/normalisation change would silently reseed every permutation.
            block_seed = stable_seed(chr_sel, start_bp, end_bp, trait_col, "FL")
            F_obs, pval_perm = freedman_lane_perm_pvalue(
                y=y_test,
                groups=g_test,
                pcs=_cov_use,
                n_perm=int(n_perm),
                seed=block_seed,
            )

        except Exception:
            untested[_bkey] = "the haplotype test failed"
            continue

        # η² must be on the residuals of the test's own design ([1, PCs, covariates]) to be
        # consistent with the F-test
        if _cov_use is not None and _cov_use.shape[0] == len(y_test):
            _X_pcs = np.column_stack([np.ones(len(y_test)), _cov_use])
            _beta  = np.linalg.lstsq(_X_pcs, y_test, rcond=None)[0]
            _y_for_effects = y_test - _X_pcs @ _beta
        else:
            _X_pcs = None
            _y_for_effects = y_test
        hap_effects = compute_haplotype_effects(_y_for_effects, g_test)

        # Effect-size statistics (additive; parametric; on the SAME residualised phenotype
        # + retained samples as eta2). With the intercept alone as the design (no PCs, no
        # covariates) eta2 == df1*F_param/(df1*F_param+df2); these reuse F_param/df1/df2.
        _eta2_lo, _eta2_hi = _eta2_confint(F_param, df1, df2)
        _om2 = _omega2_from_f(F_param, df1, df2)
        _lead_r2 = _lead_snp_partial_r2(
            lead_snp, sid, geno_imputed, sample_ids,
            df_test["Sample"], _y_for_effects, _X_pcs,
        )

        results.append(
            {
                "Chr": chr_sel,
                "Start": start_bp,
                "End": end_bp,
                "lead_snp": lead_snp,
                "n_samples_block": int(n_samples_block_used),
                "n_snps": int(n_snps_block),
                "n_haplotypes": int(hap_counts.shape[0]),
                "n_tested_haplotypes": int(len(valid_haps)),
                "df1": int(df1),
                "df2": int(df2),
                "n_permutations": int(n_perm),
                "permutation_type": "Freedman-Lane_residual_permutation_with_PCs",
                "F_param": float(F_param) if np.isfinite(F_param) else np.nan,
                "PValue_param": float(pval_param) if np.isfinite(pval_param) else np.nan,
                "F_perm": float(F_obs) if np.isfinite(F_obs) else np.nan,
                "P_perm": float(pval_perm) if np.isfinite(pval_perm) else np.nan,
                "PValue": float(pval_perm) if np.isfinite(pval_perm) else np.nan,
                "eta2": hap_effects["eta2"],
                # Effect-size additions (all additive; eta2 unchanged). eta2_ci_* =
                # noncentral-F 95% CI on eta2; omega2 = less-biased variance-explained
                # (not a replacement); lead_snp_r2 = 1-df additive R^2 of the lead SNP.
                "eta2_ci_low": _eta2_lo,
                "eta2_ci_high": _eta2_hi,
                "omega2": _om2,
                "lead_snp_r2": _lead_r2,
                "hap_stats_json": json.dumps(hap_effects["hap_stats"]),
                # T-31 Layer 2 — MLG fragmentation + sample retention. Additive:
                # every value reuses names already in scope; no new estimation, no
                # second genotype pass, exclusion chain untouched (§2.5 frozen).
                "mlg_n_observed": int(hap_counts.shape[0]),
                "mlg_top1_freq": float(hap_counts.iloc[0] / hap_counts.sum()) if hap_counts.size else np.nan,
                "mlg_eff_inv_simpson": _inv_simpson_from_counts(hap_counts),
                "mlg_n_other": int((hap_labels == "Other").sum()),
                "mlg_frac_retained": float(len(df_test) / n_samples_block_used) if n_samples_block_used > 0 else np.nan,
                "mlg_min_group_n": int(counts.loc[valid_haps].min()),
                "n_samples_tested": int(len(df_test)),
            }
        )

    if not results:
        empty = pd.DataFrame(
            columns=[
                "Chr", "Start", "End", "lead_snp",
                "n_snps", "n_haplotypes", "n_tested_haplotypes",
                "df1", "df2",
                "F_param", "PValue_param",
                "F_perm", "P_perm",
                "PValue",
                "MidPos", "-log10p",
                "mlg_n_observed", "mlg_top1_freq", "mlg_eff_inv_simpson",
                "mlg_n_other", "mlg_frac_retained", "mlg_min_group_n", "n_samples_tested",
            ]
        )
        empty.attrs["untested"] = untested
        return empty, {}

    df_res = pd.DataFrame(results)
    df_res["MidPos"] = (df_res["Start"] + df_res["End"]) / 2.0
    df_res["-log10p"] = -np.log10(np.clip(df_res["PValue"].values, 1e-300, 1.0))

    from statsmodels.stats.multitest import multipletests
    try:
        df_res["FDR_BH"] = multipletests(df_res["PValue"], method="fdr_bh")[1]
    except (ValueError, ZeroDivisionError):
        df_res["FDR_BH"] = np.nan
    df_res.attrs["untested"] = untested

    return df_res, hap_tables

def run_haplotype_block_gwas_cached(
    haplo_df,
    chroms,
    positions,
    geno_imputed,
    sid,
    geno_df,
    pheno_df,
    trait_col,
    pcs,
    min_hap_count,
    min_group_size,
    n_perm,
    geno_encoding,
    n_pcs_used,
    user_covar=None
):
    """
    Cached wrapper for haplotype/MLG GWAS.

    IMPORTANT:
    All parameters influencing results MUST be passed explicitly
    (avoid reading st.session_state inside cached functions).
    """
    return run_haplotype_block_gwas(
        haplo_df=haplo_df,
        chroms=chroms,
        positions=positions,
        geno_imputed=geno_imputed,
        sid=sid,
        geno_df=geno_df,
        pheno_df=pheno_df,
        trait_col=trait_col,
        pcs=pcs,
        min_hap_count=min_hap_count,
        min_group_size=min_group_size,
        n_perm=n_perm,
        geno_encoding=geno_encoding,
        n_pcs_used=n_pcs_used,
        user_covar=user_covar,
    )

if st is not None:
    run_haplotype_block_gwas_cached = st.cache_data(
        show_spinner=False, max_entries=5
    )(run_haplotype_block_gwas_cached)


# Columns of the sample-to-haplotype assignments table, in file order; "Phenotype_raw"
# follows "Phenotype_used" only when the test's phenotype is not the raw values.
HAP_ASSIGNMENT_COLUMNS = [
    "Trait", "Chr", "Block_start", "Block_end", "Lead_SNP", "Sample", "Haplotype",
    "Allele_sequence", "SNP_order", "in_haplotype_test", "exclusion_reason",
    "Phenotype_used",
]
# Why a sample of the panel is not in a block's test. A sample with neither a usable
# genotype nor a phenotype is reported as "missing phenotype".
HAP_EXCLUSION_REASONS = (
    # the first two do not occur in TRACE's own files: the callers pass the imputed
    # genotype matrix and the trait's analysis samples
    "missing genotype",          # over 30% of the block's markers missing
    "missing phenotype",         # no finite trait value
    "rare haplotype (Other)",    # its haplotype has fewer than min_hap_count samples
    "group below minimum size",  # its haplotype group has fewer than min_group_size
    "missing covariate",         # a PC value is missing (not seen with genotype PCs)
)


def haplotype_assignments(hap_result, hap_tables, geno_df, pheno_df, trait_col, *,
                          blocks=None, pheno_raw=None):
    """Sample-to-haplotype assignments: one row per sample of the panel per tested block.

    ``hap_result`` and ``hap_tables`` are the two return values of one
    run_haplotype_block_gwas call; ``geno_df``, ``pheno_df`` and ``trait_col`` are that
    call's arguments. The panel is the samples in both ``geno_df`` and ``pheno_df``, in
    genotype order: in TRACE's callers, the trait's analysis samples (after phenotype QC,
    --ind-miss and covariate drops), so a sample without a trait value is not listed.
    Blocks follow ``hap_result`` (the tested blocks); ``blocks`` keeps only those keys
    ("<Chr>:<Start>-<End>").

    Each row gives the sample's haplotype label (H1..Hk by descending count, or
    "Other"), its allele sequence (the rounded dosages the test used, "|"-separated, in
    the order of SNP_order) and whether the test took it. A sample without a haplotype
    (missing genotype or phenotype) has an empty Haplotype and Allele_sequence; every
    sample left out has its reason (HAP_EXCLUSION_REASONS). Phenotype_used is the trait
    value the test takes for the sample (blank when missing); only rows with
    in_haplotype_test True enter the test. ``pheno_raw`` (a Series by sample; a repeated
    sample keeps its first value, as the ID harmonisation does) adds Phenotype_raw when,
    for some sample of the panel, both values are present and differ, so the per-block
    and all-blocks tables always have the same columns. TRACE's callers pass none: the
    test uses the uploaded values. Inputs are not modified.
    """
    columns = list(HAP_ASSIGNMENT_COLUMNS)
    geno_ids = pd.Index(np.asarray(geno_df.index, dtype=str))
    pheno = pheno_df.copy()
    pheno.index = pheno.index.astype(str)
    panel = geno_ids[geno_ids.isin(pheno.index)]
    used = pd.to_numeric(pheno[trait_col], errors="coerce").astype(float).reindex(panel)
    used = used.where(np.isfinite(used.to_numpy()))
    raw = None
    if pheno_raw is not None:
        raw_s = pd.Series(pheno_raw).copy()
        raw_s.index = raw_s.index.astype(str)
        raw_s = raw_s[~raw_s.index.duplicated(keep="first")]
        raw = pd.to_numeric(raw_s, errors="coerce").astype(float).reindex(panel)
        _r, _u = raw.to_numpy(), used.to_numpy()
        _both = np.isfinite(_r) & np.isfinite(_u)
        if not (_r[_both] != _u[_both]).any():
            raw = None
        else:
            columns.append("Phenotype_raw")

    frames = []
    for _, row in (hap_result if hap_result is not None else pd.DataFrame()).iterrows():
        key = f"{row['Chr']}:{int(row['Start'])}-{int(row['End'])}"
        table = (hap_tables or {}).get(key)
        # a tested block's table always has the test's columns (run_haplotype_block_gwas)
        if (table is None or "in_haplotype_test" not in table.columns
                or (blocks is not None and key not in blocks)):
            continue
        tab = table.copy()
        tab["Sample"] = tab["Sample"].astype(str)
        tab = tab.set_index("Sample").reindex(panel)
        labelled = tab["Haplotype"].notna().to_numpy()
        reason = np.where(labelled, tab["exclusion_reason"].fillna("").astype(str).to_numpy(),
                          np.where(used.isna().to_numpy(), "missing phenotype", "missing genotype"))
        snp_order = table["SNP_order"].iloc[0] if len(table) else ""
        frame = pd.DataFrame({
            "Trait": str(trait_col),
            "Chr": str(row["Chr"]),
            "Block_start": np.int64(int(row["Start"])),
            "Block_end": np.int64(int(row["End"])),
            "Lead_SNP": str(row.get("lead_snp", "")),
            "Sample": panel.to_numpy(dtype=str),
            "Haplotype": tab["Haplotype"].fillna("").astype(str).to_numpy(),
            "Allele_sequence": tab["Allele_sequence"].fillna("").astype(str).to_numpy(),
            "SNP_order": str(snp_order),
            "in_haplotype_test": tab["in_haplotype_test"].astype("boolean").fillna(False)
                                 .to_numpy(dtype=bool),
            "exclusion_reason": reason.astype(str),
            "Phenotype_used": used.to_numpy(dtype=float),
        }, index=pd.RangeIndex(len(panel)))
        if raw is not None:
            frame["Phenotype_raw"] = raw.to_numpy(dtype=float)
        frames.append(frame)
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype=object) for c in columns})
    return pd.concat(frames, ignore_index=True)[columns]


def snp_block_assignments(blocks_df, assignments, chrom, pos, sample_ids):
    """The rows of a haplotype_assignments table for the LD block that contains a SNP.

    The block is the first LD block on the SNP's chromosome with Start <= pos <= End (the
    significant-SNP table's Block_ID rule). Returns ``(rows, block_key, reason)``:
    ``rows`` has ``Haplotype`` (blank when the sample has none) and ``in_haplotype_test``,
    aligned to ``sample_ids``; it is None, with ``reason`` "no_block" or "no_groups", when
    no block contains the SNP or its block has no tested haplotype group.
    """
    from annotation import canon_chr

    if blocks_df is None or len(blocks_df) == 0:
        return None, None, "no_block"
    b = normalize_ld_blocks_schema(blocks_df)
    c = canon_chr(str(chrom))
    hit = b[(b["Chr"].astype(str).map(canon_chr) == c)
            & (b["Start (bp)"].astype(int) <= int(pos)) & (b["End (bp)"].astype(int) >= int(pos))]
    if hit.empty:
        return None, None, "no_block"
    row = hit.iloc[0]
    start, end = int(row["Start (bp)"]), int(row["End (bp)"])
    key = f"{row['Chr']}:{start}-{end}"
    if assignments is None or len(assignments) == 0:
        return None, key, "no_groups"
    a = assignments[(assignments["Chr"].astype(str).map(canon_chr) == c)
                    & (assignments["Block_start"].astype(int) == start)
                    & (assignments["Block_end"].astype(int) == end)]
    if a.empty or not a["Haplotype"].astype(str).str.match(r"^H\d+$").any():
        return None, key, "no_groups"
    a = a.set_index(a["Sample"].astype(str))
    ids = np.asarray(sample_ids, dtype=str)
    rows = pd.DataFrame({
        "Haplotype": a["Haplotype"].reindex(ids).fillna("").astype(str).to_numpy(dtype=object),
        "in_haplotype_test": (a["in_haplotype_test"].reindex(ids).astype("boolean")
                              .fillna(False).to_numpy(dtype=bool)),
    }, index=pd.Index(ids, name="Sample"))
    return rows, key, None


def snp_haplotype_labels(blocks_df, assignments, chrom, pos, sample_ids):
    """The haplotype labels of the LD block that contains a SNP, for its box plot.

    The block is the first row of ``blocks_df`` (the LD blocks) on the SNP's chromosome
    with Start <= pos <= End, the rule the significant-SNP table's Block_ID uses. Its
    labels come from ``assignments`` (a haplotype_assignments table), aligned to
    ``sample_ids``; a sample without a haplotype there counts as "Other". Returns
    ``(labels, block_key, reason)``: ``labels`` is None and ``reason`` says why when
    the dots cannot be coloured: "no_block" (no LD block contains the SNP) or
    "no_groups" (its block has no tested haplotype group).
    """
    rows, key, reason = snp_block_assignments(blocks_df, assignments, chrom, pos, sample_ids)
    if rows is None:
        return None, key, reason
    labels = rows["Haplotype"].to_numpy(dtype=object)
    labels[labels == ""] = "Other"
    return labels, key, None


def block_test_lm_with_pcs(y, groups, pcs=None):
    """
    Parametric nested F-test for:
        y ~ intercept + PCs + haplotype(groups)
    Returns: F, p, df1, df2
    """
    y = np.asarray(y, float).reshape(-1)
    g = np.asarray(groups, object)

    if pcs is None:
        C = np.zeros((len(y), 0), float)
    else:
        C = _ensure_2d(pcs)
        if C.shape[0] != len(y):
            raise ValueError(f"PCs rows ({C.shape[0]}) != y length ({len(y)}).")

    H, keep_levels, all_levels = _one_hot_drop_first(g)
    if H is None or H.shape[1] < 1:
        return np.nan, np.nan, 0, int(max(1, len(y) - 1))

    X0 = np.column_stack([np.ones(len(y)), C])         # reduced
    X1 = np.column_stack([np.ones(len(y)), C, H])      # full

    _, _, sse0, df0 = _ols_fit(y, X0)
    _, _, sse1, df1 = _ols_fit(y, X1)

    F, p = _nested_f_test(sse0, sse1, df0, df1)
    df_num = int(df0 - df1)
    df_den = int(df1)
    return float(F), float(p), int(df_num), int(df_den)

def freedman_lane_perm_pvalue(y, groups, pcs=None, n_perm=1000, seed=0):
    """
    Freedman–Lane permutation (valid with covariates):
      1) fit reduced y ~ PCs
      2) permute residuals
      3) y* = yhat + r_perm
      4) recompute nested-F
    Returns: (F_obs, p_emp)
    """
    rng = np.random.default_rng(int(seed))
    y = np.asarray(y, float).reshape(-1)
    g = np.asarray(groups, object)

    if pcs is None:
        C = np.zeros((len(y), 0), float)
    else:
        C = _ensure_2d(pcs)
        if C.shape[0] != len(y):
            raise ValueError(f"PCs rows ({C.shape[0]}) != y length ({len(y)}).")

    H, _, _ = _one_hot_drop_first(g)
    if H is None or H.shape[1] < 1:
        return np.nan, np.nan

    X0 = np.column_stack([np.ones(len(y)), C])
    X1 = np.column_stack([np.ones(len(y)), C, H])

    # observed
    _, _, sse0, df0 = _ols_fit(y, X0)
    _, _, sse1, df1 = _ols_fit(y, X1)
    F_obs, _ = _nested_f_test(sse0, sse1, df0, df1)
    if not np.isfinite(F_obs):
        return np.nan, np.nan

    # reduced fit for FL
    beta0, resid0, _, _ = _ols_fit(y, X0)
    yhat0 = X0 @ beta0

    n_perm = int(n_perm)
    F_perm = np.zeros(n_perm, float)
    idx = np.arange(len(y))

    for b in range(n_perm):
        rng.shuffle(idx)
        y_star = yhat0 + resid0[idx]

        _, _, sse0b, df0b = _ols_fit(y_star, X0)
        _, _, sse1b, df1b = _ols_fit(y_star, X1)
        Fb, _ = _nested_f_test(sse0b, sse1b, df0b, df1b)
        F_perm[b] = 0.0 if not np.isfinite(Fb) else float(Fb)

    p_emp = (1.0 + float(np.sum(F_perm >= F_obs))) / (1.0 + float(n_perm))
    return float(F_obs), float(p_emp)

def compute_haplotype_effects(y_test, g_test):
    """
    Compute η² and per-haplotype summary statistics.

    Returns dict:
      'eta2'      : float  — proportion of variance explained by haplotype group
      'hap_stats' : dict   — {hap_label: {'n', 'mean', 'se', 'ci95_lo', 'ci95_hi'}}
    """
    import scipy.stats as _st
    y = np.asarray(y_test, dtype=float)
    g = np.asarray(g_test, dtype=object)

    valid = np.isfinite(y)
    y = y[valid]
    g = g[valid]

    if len(y) < 4:
        return {"eta2": np.nan, "hap_stats": {}}

    grand_mean = float(np.mean(y))
    ss_total = float(np.sum((y - grand_mean) ** 2))
    ss_between = 0.0
    hap_stats = {}

    for h in np.unique(g):
        mask = g == h
        y_h = y[mask]
        n = int(len(y_h))
        if n < 2:
            continue
        mean_h = float(np.mean(y_h))
        se_h = float(_st.sem(y_h))
        ss_between += n * (mean_h - grand_mean) ** 2
        hap_stats[str(h)] = {
            "n": n,
            "mean": mean_h,
            "se": se_h,
            "ci95_lo": mean_h - 1.96 * se_h,
            "ci95_hi": mean_h + 1.96 * se_h,
        }

    eta2 = float(ss_between / ss_total) if ss_total > 0 else np.nan
    if not np.isfinite(eta2):
        eta2 = np.nan

    return {"eta2": eta2, "hap_stats": hap_stats}


def normalize_ld_blocks_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize LD block dataframe to a canonical schema used across pages.

    Canonical output columns:
      - Chr
      - Start (bp)
      - End (bp)
      - lead_snp
    """
    if df is None or len(df) == 0:
        return df

    out = df.copy()

    # Standardize chromosome
    if "Chr" not in out.columns and "CHROM" in out.columns:
        out = out.rename(columns={"CHROM": "Chr"})

    # Start/End variants
    if "Start (bp)" not in out.columns and "Start" in out.columns:
        out = out.rename(columns={"Start": "Start (bp)"})
    if "End (bp)" not in out.columns and "End" in out.columns:
        out = out.rename(columns={"End": "End (bp)"})

    # lead_snp variants (canonical name = lead_snp; alias the legacy "Lead SNP"
    # spelling + older variants so pre-rename CSVs still load)
    if "lead_snp" not in out.columns:
        for _legacy in ("Lead SNP", "Lead_SNP", "Lead", "Representative SNP"):
            if _legacy in out.columns:
                out = out.rename(columns={_legacy: "lead_snp"})
                break

    # Enforce presence of required columns
    required = ["Chr", "Start (bp)", "End (bp)"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise KeyError(
            f"LD block table missing required columns {missing}. "
            f"Found columns: {list(out.columns)}"
        )

    # Coerce types
    out["Chr"] = out["Chr"].astype(str)
    out["Start (bp)"] = pd.to_numeric(out["Start (bp)"], errors="coerce").astype("Int64")
    out["End (bp)"] = pd.to_numeric(out["End (bp)"], errors="coerce").astype("Int64")

    # Drop unusable rows
    out = out.dropna(subset=["Start (bp)", "End (bp)"]).copy()
    out["Start (bp)"] = out["Start (bp)"].astype(int)
    out["End (bp)"] = out["End (bp)"].astype(int)

    if "lead_snp" not in out.columns:
        out["lead_snp"] = ""

    return out


# ============================================================
# MLG/haplotype utility functions (moved from pages/LD analysis.py)
# ============================================================

def attach_trait_to_hap_tables(hap_tables, trait_series, colname="Trait_adj"):
    """
    Ensure every hap_table contains phenotype column.

    trait_series:
        pd.Series indexed by sample ID (aligned phenotype).
    """
    out = {}
    for b, df in hap_tables.items():
        df2 = df.copy()

        if colname not in df2.columns:
            df2[colname] = (
                pd.Series(trait_series)
                .reindex(df2["Sample"].astype(str))
                .values
            )

        out[b] = df2

    return out


def mlg_labels_for_block(
    block_geno: np.ndarray,
    min_hap_count: int = 5,
    return_mask=False,
    strict_selfing: bool = False,
    geno_encoding: str | None = None,
    max_missing_frac: float = 0.30,
):
    """
    Build multi-locus genotype (MLG) labels.

    Behavior:
    - Hard-calls genotype dosages by rounding.
    - Optional strict selfing mode:
        masks heterozygotes (value==1) ONLY when encoding is dosage012.
    - Drops samples missing > max_missing_frac of SNPs in block.
    - Mean-imputes remaining per-SNP NaNs (matches GWAS page behavior).
    - Collapses rare haplotypes (<min_hap_count) into "Other".
    """

    G = np.asarray(block_geno, float)

    # ---- Hard-call dosages ----
    G = np.rint(G)
    # Only mask heterozygotes if:
    #   - user activated strict selfing
    #   - encoding is dosage012 (0/1/2 where 1=het)
    if strict_selfing and geno_encoding == "dosage012":
        G[G == 1] = np.nan

    # ---- Drop samples with too many missing SNPs ----
    if G.shape[1] > 0:
        missing_frac = np.mean(~np.isfinite(G), axis=1)
        keep = missing_frac <= float(max_missing_frac)
    else:
        keep = np.ones(G.shape[0], dtype=bool)
    G = G[keep, :]

    if G.shape[0] == 0:
        return (np.array([]), np.array([], dtype=bool)) if return_mask else np.array([])

    # ---- Mean-impute remaining NaNs per SNP (consistent with GWAS page) ----
    if np.any(~np.isfinite(G)):
        col_means = np.nanmean(G, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        nan_idx = np.where(~np.isfinite(G))
        G[nan_idx] = np.rint(col_means[nan_idx[1]])  # round imputed values to integer

    # ---- Build haplotype strings ----
    hap_strings = (
        pd.DataFrame(G.astype(int))
        .astype(str)
        .agg("|".join, axis=1)
        .values
    )

    # ---- Collapse rare haplotypes ----
    counts = pd.Series(hap_strings).value_counts()
    keep_haps = counts[counts >= min_hap_count].index
    labels = np.where(np.isin(hap_strings, keep_haps), hap_strings, "Other")

    if return_mask:
        return labels, keep
    return labels


def anova_F(groups_dict):
    """
    Compute one-way ANOVA F statistic for dict label -> 1d array.
    Returns np.nan if invalid.
    """
    from scipy import stats as sp_stats
    groups = [np.asarray(v, float) for v in groups_dict.values()]
    if len(groups) < 2:
        return np.nan
    try:
        F, _ = sp_stats.f_oneway(*groups)
        return float(F)
    except (ValueError, RuntimeError):
        return np.nan


def block_perm_pvalue_mlg_anova(
    block_geno: np.ndarray,
    y_vec: np.ndarray,
    min_hap_count: int = 5,
    min_group_size: int = 3,
    n_perm: int = 1000,
    seed: int = 0,
    pcs: np.ndarray | None = None,
):
    """
    Freedman-Lane permutation p-value for the block-level MLG test,
    conditioning on PCs (consistent with haplotype GWAS on the main page).

    If pcs is None, falls back to unconditional permutation.

    Returns:
      (p_empirical, F_obs, n_groups, n_used)
    """
    rng = np.random.default_rng(int(seed))
    y = np.asarray(y_vec, float).reshape(-1)

    # Step 1: filter samples with missing phenotype
    keep_y = np.isfinite(y)
    if keep_y.sum() < 5:
        return np.nan, np.nan, 0, int(keep_y.sum())

    G = np.asarray(block_geno, float)[keep_y, :]
    y = y[keep_y]

    # Step 2: build MLG labels WITH the same filtering as visualization
    mlg, keep_mlg = mlg_labels_for_block(
        G,
        min_hap_count=min_hap_count,
        return_mask=True,
    )

    # Apply the MLG sample mask to phenotype and genotype
    y = y[keep_mlg]
    G = G[keep_mlg, :]
    n = len(y)

    if n < 5:
        return np.nan, np.nan, 0, int(n)

    # Step 3: align PCs to the same final sample set
    C = np.ones((n, 1), float)  # intercept only
    if pcs is not None:
        pcs_sub = np.asarray(pcs, float)
        if pcs_sub.ndim == 1:
            pcs_sub = pcs_sub.reshape(-1, 1)
        # Apply both masks: keep_y (finite phenotype) then keep_mlg (rare-hap filter)
        if pcs_sub.shape[0] == len(keep_y):
            pcs_sub = pcs_sub[keep_y]
        if pcs_sub.shape[0] == len(keep_mlg):
            pcs_sub = pcs_sub[keep_mlg]
        if pcs_sub.shape[0] == n:
            C = np.column_stack([C, pcs_sub])

    # Step 4: identify valid test groups (same criteria as visualization)
    df_test = pd.DataFrame({"MLG": mlg, "y": y})
    counts = df_test["MLG"].value_counts()
    valid = [g for g in counts.index if (counts[g] >= int(min_group_size) and g != "Other")]

    if len(valid) < 2:
        return np.nan, np.nan, len(valid), int(n)

    # Step 5: observed F statistic
    groups_obs = {g: df_test.loc[df_test["MLG"] == g, "y"].values for g in valid}
    F_obs = anova_F(groups_obs)
    if not np.isfinite(F_obs):
        return np.nan, np.nan, len(valid), int(n)

    # Step 6: Freedman-Lane permutation
    beta_C, *_ = np.linalg.lstsq(C, y, rcond=None)
    yhat = C @ beta_C
    resid = y - yhat

    F_perm = np.zeros(int(n_perm), dtype=float)
    idx = np.arange(n)

    for b in range(int(n_perm)):
        rng.shuffle(idx)
        y_star = yhat + resid[idx]
        df_test["y_perm"] = y_star
        groups_perm = {g: df_test.loc[df_test["MLG"] == g, "y_perm"].values for g in valid}
        Fp = anova_F(groups_perm)
        F_perm[b] = 0.0 if not np.isfinite(Fp) else float(Fp)

    p_emp = (1.0 + np.sum(F_perm >= F_obs)) / (1.0 + float(n_perm))
    return float(p_emp), float(F_obs), int(len(valid)), int(n)


def build_two_panel_expression_lines(
    hap_tables: dict,
    blocks: list[str],
    trait_col: str = "Trait_adj",
    min_group_size: int = 3,
    n_per_group: int = 4,
    exclude_other: bool = True,
    hap_gwas_df: pd.DataFrame | None = None,
    weight_mode: str = "logp",
):
    """
    Select two line panels (High vs Low) across multiple blocks based on haplotype means.

    Requires that hap_tables[block] has:
      Sample, Haplotype, and trait_col (e.g. Trait_adj).
    """

    # -------------------------
    # 0) block weights (optional)
    # -------------------------
    block_w = {b: 1.0 for b in blocks}
    if hap_gwas_df is not None and isinstance(hap_gwas_df, pd.DataFrame) and not hap_gwas_df.empty:
        for _, r in hap_gwas_df.iterrows():
            b = f"{r['Chr']}:{int(r['Start'])}-{int(r['End'])}"
            if b in block_w and weight_mode == "logp":
                p = max(float(r.get("PValue", 1.0)), 1e-300)
                block_w[b] = float(-np.log10(p))

    # -------------------------
    # 1) choose contrast haplotypes per block
    # -------------------------
    block_targets = {}  # block -> (H_high, H_low)
    block_summaries = []

    for b in blocks:
        if b not in hap_tables:
            continue

        df = hap_tables[b].copy()
        if trait_col not in df.columns:
            raise ValueError(f"Block {b} missing required column '{trait_col}' in hap_tables.")

        df["Sample"] = df["Sample"].astype(str)
        df["Haplotype"] = df["Haplotype"].astype(str)
        df[trait_col] = pd.to_numeric(df[trait_col], errors="coerce").astype(float)
        df = df[np.isfinite(df[trait_col].values)].copy()

        if exclude_other:
            df = df[df["Haplotype"] != "Other"].copy()

        # enforce min_group_size
        cnt = df["Haplotype"].value_counts()
        valid = cnt[cnt >= int(min_group_size)].index.tolist()
        df = df[df["Haplotype"].isin(valid)].copy()

        if df["Haplotype"].nunique() < 2:
            continue

        means = df.groupby("Haplotype")[trait_col].mean().sort_values()
        H_low = means.index[0]
        H_high = means.index[-1]

        block_targets[b] = (H_high, H_low)

        block_summaries.append({
            "Block": b,
            "H_high": H_high,
            "mean_high": float(means.loc[H_high]),
            "H_low": H_low,
            "mean_low": float(means.loc[H_low]),
            "delta": float(means.loc[H_high] - means.loc[H_low]),
            "weight": float(block_w.get(b, 1.0)),
            "n_groups": int(df["Haplotype"].nunique()),
            "n_samples": int(df.shape[0]),
        })

    if len(block_targets) < 1:
        return [], [], pd.DataFrame(block_summaries), pd.DataFrame()

    block_summary_df = pd.DataFrame(block_summaries).sort_values(
        ["weight", "delta"], ascending=[False, False]
    )

    # -------------------------
    # 2) build line x block haplotype matrix + trait
    # -------------------------
    all_samples = set()
    for b in block_targets.keys():
        all_samples.update(hap_tables[b]["Sample"].astype(str).tolist())
    all_samples = sorted(all_samples)

    H = pd.DataFrame(index=all_samples)
    Y = pd.DataFrame(index=all_samples)

    for b in block_targets.keys():
        df = hap_tables[b].copy()
        df["Sample"] = df["Sample"].astype(str)
        df["Haplotype"] = df["Haplotype"].astype(str)
        df[trait_col] = pd.to_numeric(df[trait_col], errors="coerce").astype(float)

        df = df.drop_duplicates(subset=["Sample"], keep="first")

        H[b] = df.set_index("Sample")["Haplotype"].reindex(H.index)
        Y[b] = df.set_index("Sample")[trait_col].reindex(Y.index)

    y_mean = Y.mean(axis=1, skipna=True)

    # -------------------------
    # 3) score each sample for High vs Low panel
    # -------------------------
    def score_sample(sample, want_high=True):
        s = 0.0
        for b, (Hh, Hl) in block_targets.items():
            hb = H.loc[sample, b]
            if pd.isna(hb):
                continue
            w = float(block_w.get(b, 1.0))
            if want_high and hb == Hh:
                s += w
            elif (not want_high) and hb == Hl:
                s += w
        return s

    scores = pd.DataFrame({
        "Sample": H.index,
        "score_high": [score_sample(s, True) for s in H.index],
        "score_low":  [score_sample(s, False) for s in H.index],
        "trait_mean": y_mean.values,
    }).set_index("Sample")

    # -------------------------
    # 4) select panels greedily
    # -------------------------
    high_rank = scores.sort_values(["score_high", "trait_mean"], ascending=[False, False]).index.tolist()
    low_rank  = scores.sort_values(["score_low", "trait_mean"], ascending=[False, True]).index.tolist()

    group_high = []
    used = set()

    for s in high_rank:
        if len(group_high) >= int(n_per_group):
            break
        if s in used:
            continue
        if scores.loc[s, "score_high"] <= 0:
            continue
        group_high.append(s)
        used.add(s)

    group_low = []
    for s in low_rank:
        if len(group_low) >= int(n_per_group):
            break
        if s in used:
            continue
        if scores.loc[s, "score_low"] <= 0:
            continue
        group_low.append(s)
        used.add(s)

    # Audit
    audit = H.loc[group_high + group_low, list(block_targets.keys())].copy()
    audit.insert(0, "Panel", ["High"] * len(group_high) + ["Low"] * len(group_low))
    audit.insert(1, "trait_mean", scores.loc[group_high + group_low, "trait_mean"].values)
    audit.insert(2, "score_high", scores.loc[group_high + group_low, "score_high"].values)
    audit.insert(3, "score_low", scores.loc[group_high + group_low, "score_low"].values)

    return group_high, group_low, block_summary_df, audit