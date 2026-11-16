"""The GWAS page's computations on the genotype table, as importable functions.

The identity
    trait_identity      one string per trait table: the VCF's identity and every setting that changes
                        the genotype table or the phenotype vector y (QC and imputation settings, the
                        trait and its normalisation, the retained samples, the covariates and the
                        chromosome options). Every cache and session key that depends on the table is
                        built on it, so a changed setting can never be served another table's result.
    sample_set_hash     the retained samples, as gwas.qc names them for its imputation key

Caches keyed by the identity (no table is ever hashed by Streamlit; tables are passed as arguments
whose names start with an underscore, which Streamlit leaves out of the key)
    ld_decay            the page's per-chromosome LD decay (annotation.compute_ld_decay_by_chromosome)
    meff                M_eff (Li and Ji) by the CLI's function, gwas.plotting._compute_meff_li_ji_impl,
                        with its whole-number rule; kept in the session per identity

"""
import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError:                       # pragma: no cover - streamlit is a TRACE dependency
    st = None


# ------------------------------------------------------------------------------ the identity
def sample_set_hash(samples):
    """The retained samples (in any order), named as gwas.qc names them for its imputation key."""
    from gwas.utils import hash_bytes

    return hash_bytes(repr(tuple(sorted(pd.Index(samples).astype(str)))).encode())


def trait_identity(file_identity, *, pheno, trait, norm, maf, miss, ind_miss, mac, info, drop_alt,
                   impute, impute_k, impute_l, covar, canonical, samples):
    """The identity of one trait's genotype table and phenotype vector (gwas.guitable.table_identity):

    file_identity   the VCF's content hash (gwas.guitable.file_identity)
    pheno           the uploaded phenotype table's hash (its raw values)
    trait, norm     the trait and its normalisation (y)
    maf, miss, ind_miss, mac, info, drop_alt
                    the QC settings
    impute, impute_k, impute_l
                    the imputation method and LD-kNNi's k and l
    covar           the covariate table's hash, or None (its incomplete samples are dropped)
    canonical       the chromosome options: the canonical chromosome names, or None (the page has
                    no option for them; ALT markers are always dropped: drop_alt)
    samples         the retained sample set (sample_set_hash), after phenotype QC and the
                    covariate drop
    """
    from gwas.guitable import table_identity

    return table_identity(
        file_identity, pheno=pheno, trait=trait, norm=norm, maf=maf, miss=miss,
        ind_miss=ind_miss, mac=mac, info=info, drop_alt=drop_alt, impute=impute,
        impute_k=impute_k, impute_l=impute_l, covar=covar, canonical=canonical, samples=samples)


# ------------------------------------------------------------------------------- LD decay
def ld_decay(table_key, _geno_imputed=None, _chroms=None, _positions=None):
    """The page's LD decay: annotation.compute_ld_decay_by_chromosome with the CLI's settings
    (1,500 SNPs per chromosome, 5,000 kb, 40 bins); Post-GWAS shows the estimate in the LD Blocks
    and Haplotypes Settings. Returns (decay_df, summary_df, median_decay_kb); the median is None
    when no chromosome gave a decay value. Cached by ``table_key``, the trait table's identity."""
    from annotation import compute_ld_decay_by_chromosome

    if _geno_imputed is None:
        return None, None, None
    decay_df, summary_df = compute_ld_decay_by_chromosome(
        chroms=np.asarray(_chroms, dtype=str),
        positions=np.asarray(_positions, dtype=int),
        geno_imputed=_geno_imputed,
        max_snps_per_chr=1500,   # aligned to the CLI's per-chromosome subsample cap (cli.py, the `if len(pos_ch) > 1500`
        # in the "Estimating LD decay" loop, not ld_decay's MAX_LD_SNPS safety net)
        max_dist_kb=5000.0,
        n_bins=40,   # standardised onto the CLI ld_decay bin count (LD-decay Tier 1a)
    )
    median_decay = None
    if summary_df is not None and not summary_df.empty \
            and "decay_kb_r2_0.2" in summary_df.columns:
        _m = summary_df["decay_kb_r2_0.2"].median()
        if pd.notna(_m) and np.isfinite(_m):
            median_decay = float(_m)
    return decay_df, summary_df, median_decay


if st is not None:
    ld_decay = st.cache_data(show_spinner="Computing LD decay (per-chromosome)…",
                             max_entries=8)(ld_decay)


# ----------------------------------------------------------------------------------- M_eff
MEFF_KEY = "_trace_meff"


def meff(state, table_key, geno_imputed):
    """(M_eff, eigenvalues) of the trait table, by the CLI's function
    (gwas.plotting._compute_meff_li_ji_impl: the eigenvalues summed over blocks of SNP columns and
    the whole-number rule), computed once per identity and kept in the session ``state`` under
    MEFF_KEY (one entry: the current table's). The table is never hashed."""
    from gwas import plotting

    hit = state.get(MEFF_KEY)
    if hit is not None and hit[0] == table_key:
        return hit[1]
    out = plotting._compute_meff_li_ji_impl(geno_imputed)
    state[MEFF_KEY] = (table_key, out)
    return out

