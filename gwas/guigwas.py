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

The table, in memory or on disk (the same results either way)
    table               the server's table of the VCF (gwas.guitable's registry): the parsed
                        genotypes in memory, or the table file on disk
    preprocess          the page's preprocessing of one trait (gwas.qc._gwas_pipeline_stages: on
                        disk through gwas.storepipe.TraitStore, at the CLI's places), kept in the
                        session: one current preprocessing per session, replaced when a setting
                        changes; its warnings are shown again on every rerun
    column_means        the imputed table's per-SNP means (the QC report's MAF histogram)
    subsampling         the page's subsampling; in memory with several workers, the workers'
                        memory is checked first
"""
import numpy as np
import pandas as pd

from gwas.genostore import is_lazy

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


# ------------------------------------------------------------------------------ the table
def table(file_identity, mode, vcf_path, work_dir):
    """The server's table of ``vcf_path`` in ``mode`` (gwas.guitable.get_registry): the parsed
    genotypes (gwas.pipeline.ParsedGenotypes, read-only) in memory, or the table file
    (gwas.genostore.StoreParsed) in a folder of the work folder on disk. Shared by the sessions
    and never closed by a page: the registry owns it. A VCF the reader refuses raises the
    in-memory reader's error, in both modes."""
    from gwas import guitable
    from gwas.genostore import DeferredParseError

    try:
        return guitable.get_registry().get(file_identity, mode, vcf_path, work_dir)
    except DeferredParseError as exc:          # on disk: the in-memory parse's own stop
        raise exc.original from None


# ---------------------------------------------------------------------- the preprocessing
PREP_KEY = "_trace_gwas_prep"


def preprocess(state, key, parsed, pheno, trait_col, maf_thresh, miss_thresh, norm_option,
               drop_alt, ind_miss_thresh, mac_thresh, info_thresh=0.0, canonical=None,
               impute_method="mean", impute_k=5, impute_l=20, covar_df=None, show=None):
    """The page's preprocessing of ``trait_col`` on the server's table ``parsed``
    (gwas.qc._gwas_pipeline_stages), kept in the session ``state`` under PREP_KEY: one entry,
    the current one, reused while ``key`` (the inputs' identity and the mode) and the table
    object are the same, replaced when either changes. No table is put in a Streamlit cache.

    The warnings the preprocessing gives (_log_or_warn) are passed to ``show`` when they are
    made and again on every reuse, as Streamlit's cache replayed them at ff56e5c. A trait that
    fails (for example a phenotype-QC stop) raises, after its warnings, and is not kept."""
    from gwas import qc

    hit = state.get(PREP_KEY)
    if hit is not None and hit[0] == key and hit[1]["parsed"] is parsed:
        for msg in hit[1]["warnings"]:
            if show is not None:
                show(msg)
        return hit[1]
    messages = []
    try:
        with qc.collect_warnings(messages):
            out = qc._gwas_pipeline_stages(
                parsed, pheno, trait_col, maf_thresh, miss_thresh, norm_option, drop_alt,
                ind_miss_thresh, mac_thresh, info_thresh=info_thresh, canonical=canonical,
                impute_method=impute_method, impute_k=impute_k, impute_l=impute_l,
                covar_df=covar_df)
    finally:
        if show is not None:
            for msg in messages:
                show(msg)
    out["warnings"] = list(messages)
    state[PREP_KEY] = (key, out)
    return out


def n_snps(results):
    """The number of SNPs after QC: the genotype frame's columns in memory; on disk the frame
    holds the sample rows only, so the SNPs are counted from their IDs (the same number)."""
    return int(len(results["sid"]))


def column_means(geno_imputed, piece=4096):
    """``geno_imputed.mean(axis=0)`` of the page's float32 table (the QC report's MAF
    histogram). On disk the table is read in F-ordered float32 blocks of columns: each column
    is contiguous as in the whole F-ordered table, so numpy sums it in the same order."""
    if not is_lazy(geno_imputed):
        return geno_imputed.mean(axis=0)
    from gwas.genostore import piece as _piece

    m = geno_imputed.shape[1]
    out = np.empty(m, dtype=np.float32)
    step = _piece(piece)
    for a in range(0, m, step):
        b = min(m, a + step)
        out[a:b] = geno_imputed.block(np.arange(a, b), layout="F").mean(axis=0)
    return out


# ----------------------------------------------------------------------------- subsampling
def subsampling_workers_fit(n_samples, n_snps_, n_jobs, available=None):
    """(fits, need_bytes, free_bytes): whether ``n_jobs`` subsampling workers fit in the free
    memory. Each worker after the first is a process with its own share of the table:
    gwas.storeplan.estimate_memory_mib with ``boot_jobs`` minus the same with one worker (300 MiB
    and 12 bytes per genotype call each, batch 12's measurements). They fit when that is at most
    storeplan.AUTO_MEMORY_FRACTION of the free memory, the rule --geno-store auto applies."""
    from gwas import storeplan as sp

    est = {j: sp.estimate_memory_mib(n_samples, n_snps_, subsampling=True, boot_jobs=j)
           for j in (int(n_jobs), 1)}
    need = max(0.0, est[int(n_jobs)] - est[1]) * sp.MIB
    free = sp.available_memory_bytes() if available is None else int(available)
    return need <= sp.AUTO_MEMORY_FRACTION * free, int(need), int(free)


def subsampling_plan(geno_imputed, n_jobs, available=None):
    """How the page's subsampling runs: (workers, note). ``workers`` above 1: in that many processes
    (the table in memory, and the workers fit). 1: one replicate after another in this process, under
    the thread limits a joblib worker of ``n_jobs`` starts with when ``n_jobs`` is above 1 (with the
    table on disk always, inside subsample_gwas_resampling; in memory when the workers do not fit),
    which gives the parallel run's numbers. ``note`` is None or the sentence the page shows."""
    if int(n_jobs) <= 1 or is_lazy(geno_imputed):
        return 1, None
    n, m = geno_imputed.shape
    fits, need, free = subsampling_workers_fit(n, m, int(n_jobs), available=available)
    if fits:
        return int(n_jobs), None
    return 1, (f"The subsampling iterations run one after another instead of on {int(n_jobs)} cores: "
               f"the {int(n_jobs)} workers would need about {need / 1e9:.1f} GB more memory, and "
               f"{free / 1e9:.1f} GB are free. The results are the same.")


def subsampling(geno_imputed, y, sid, chroms, chroms_num, positions, iid, Z_for_grm, pcs_full,
                n_pcs, n_reps, sample_frac, discovery_thresh, seed, n_jobs, use_loco=False,
                chroms_grm=None, progress_callback=None, available=None, plan=None):
    """The page's subsampling (gwas.subsampling.subsample_gwas_resampling), run as
    subsampling_plan says (or ``plan``, its result); returns (discovery_df, raw_pvals, rep_metadata,
    note). The progress callback is used when the replicates run in this process."""
    from gwas.subsampling import subsample_gwas_resampling

    workers, note = plan if plan is not None else subsampling_plan(geno_imputed, n_jobs, available)
    args = dict(geno_imputed=geno_imputed, y=y, sid=sid, chroms=chroms, chroms_num=chroms_num,
                positions=positions, iid=iid, Z_for_grm=Z_for_grm, pcs_full=pcs_full, n_pcs=n_pcs,
                n_reps=int(n_reps), sample_frac=float(sample_frac),
                discovery_thresh=float(discovery_thresh), seed=int(seed), use_loco=use_loco,
                chroms_grm=chroms_grm)
    if workers > 1:
        out = subsample_gwas_resampling(**args, n_jobs=int(n_jobs), progress_callback=None)
    elif int(n_jobs) <= 1 or is_lazy(geno_imputed):
        out = subsample_gwas_resampling(**args, n_jobs=int(n_jobs), progress_callback=progress_callback)
    else:
        # the process-wide limits under one lock (gwas.provenance.limited_threads): another
        # session's subsampling waits for it instead of interleaving with it
        from gwas.provenance import limited_threads, worker_threadpool_limits

        with limited_threads(worker_threadpool_limits(int(n_jobs))):
            out = subsample_gwas_resampling(**args, n_jobs=1, progress_callback=progress_callback)
    return (*out, note)


# ------------------------------------------------------------------------------- One-Click
class PlanStop(Exception):
    """gwas.storeplan stopped a run before reading (str(): the stop in the page's words)."""


def one_click_table(file_identity, vcf_path, choice, work_dir, options, estimate=None):
    """One-Click's table and its run_metadata.json keys: (parsed, meta, plan).

    ``plan`` is gwas.storeplan.plan for One-Click's own options (its models, subsampling and its
    workers, imputation), as cli.py plans a run; ``parsed`` the server's table in that plan's mode
    (table(): the page's own table when the modes agree, as they do unless One-Click's options need
    more memory than the detailed view's); ``meta`` the plan's metadata for the genotypes as read
    (gwas.storeplan.Plan.metadata: the mode the table has, the CLI's estimate for these options and
    the call count), which run_trait_pipeline adds to run_metadata.json as cli.py's run does.
    ``estimate`` replaces the CLI's estimate for the choice only (the GUI's own rule); the metadata
    always records the CLI's. Raises PlanStop for a stop. The table is the registry's: One-Click
    never closes it."""
    from gwas import guitable, storeplan

    plan = storeplan.plan(vcf_path, choice=choice, store_base=work_dir,
                          models=tuple(getattr(options, "model", None) or ("mlm",)),
                          subsampling=bool(getattr(options, "subsampling", False)),
                          boot_jobs=getattr(options, "boot_jobs", 1),
                          export_qc=bool(getattr(options, "export_qc", False)),
                          impute=getattr(options, "impute", "mean"), estimate=estimate)
    if plan.stop:
        raise PlanStop(guitable.page_stop(plan, work_dir))
    parsed = table(file_identity, plan.mode, vcf_path, work_dir)
    meta = plan.metadata(len(parsed.samples), parsed.genotypes.shape[1])
    return parsed, meta, plan


# ------------------------------------------------------------------------- Clear session data
# The GWAS page's session keys "Clear session data" drops (gwas.guitable.clear_all adds the Post-GWAS
# page's): its inputs, the preprocessing and tables, the run's records and results.
SESSION_PREFIXES = ("ARRKEY::", "KERNELKEY::", "OBJKEY::", "gwas_zip_", "BOOT_GWAS::")
SESSION_KEYS = (
    "_persist_vcf_path", "_persist_pheno", "_persist_covar", "_vcf_upload_token", "_vcf_source",
    "_vcf_live", "_vcf_save_failed",
    "_persist_vcf_name", "gwas_df", "gwas_results", "gwas_results_by_model", "gwas_figures",
    "gwas_run_summary", "gwas_triggered", "geno_imputed", "geno_dosage_raw",
    "geno_df", "K", "K0", "K_by_chr", "Z_grm", "chroms_grm", "pcs", "pcs_full",
    "pheno_reader", "covar_reader", "pheno_aligned", "pheno", "y", "iid",
    "sid", "chroms", "chroms_num", "positions", "GWAS_CTX", "geno_table", "pheno_raw",
    "pheno_applied_transforms", "_pheno_raw_fp", "subsampling_gwas_summary", MEFF_KEY,
    PREP_KEY, "meff_val", "ld_decay_df", "ld_decay_summary", "ld_decay_computed", "ld_decay_kb",
    "ld_block_max_dist_bp", "downstream_model", "active_trait", "trait_col", "trait_col_locked",
    "geno_row_ids", "pheno_used_for_hap", "loco_diagnostics_by_chr", "n_snps_raw", "qc_snp",
    "one_click_results", "subsampling_block_stability", "_last_gwas_trait",
)
