"""One trait's GWAS pipeline, shared by the CLI and the One-Click Full Analysis.

run_trait_pipeline() runs everything after VCF parsing for ONE trait, on that
trait's own phenotyped samples: sample alignment, phenotype and SNP QC, imputation,
kinship and LOCO kernels, PCs, the GWAS models, significance, LD blocks, the
isolated-SNP rescue, the haplotype test, annotation, triage, subsampling, figures,
the HTML report and the results ZIP. Only the parsed VCF (read-only) and the gene
models are shared between traits; a multi-trait run calls this once per trait, in
sequence, and keeps only each trait's result.

This is the CLI's former per-trait code, moved unchanged. ``options`` is the CLI's
argparse namespace (the GUI builds one from its widgets); inside, it is ``args`` with
``args.trait`` set to the trait. The pipeline logs to the "cli" logger, as it always
has: tests and the runtime benchmark attach to that logger.
"""
import argparse
import logging
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from gwas.genostore import as_columns as _as_columns, is_lazy as _is_lazy   # the table kept on disk
from gwas import stagelog as _stagelog   # per-stage marks (TRACE_STAGE_LOG; off by default)
from gwas import taps as _taps           # digests of intermediate objects (TRACE_TAPS; off by default)
from gwas.provenance import thread_provenance as _thread_provenance

log = logging.getLogger("cli")


class TraitPipelineError(Exception):
    """A trait-level stop. ``exit_arg`` is what the single-trait CLI passes to
    sys.exit (the message, or a bare exit code), so its behaviour is unchanged."""

    def __init__(self, message, exit_arg=None):
        super().__init__(message)
        self.exit_arg = message if exit_arg is None else exit_arg


@dataclass(frozen=True)
class ParsedGenotypes:
    """The parsed VCF (biallelic filter), shared read-only by every trait of a run."""

    genotypes: np.ndarray
    samples: np.ndarray
    chroms: np.ndarray
    positions: np.ndarray
    vid: np.ndarray
    ref: np.ndarray
    alt: np.ndarray
    info_scores: object
    info_field: object

    @classmethod
    def from_callset(cls, callset):
        """Parse a scikit-allel callset (gwas.io.load_vcf_cached) once, read-only."""
        from gwas.qc import _pipeline_parse_vcf_biallelic

        parts = _pipeline_parse_vcf_biallelic(callset)
        for p in parts:
            if isinstance(p, np.ndarray):
                p.setflags(write=False)  # a trait must never modify the shared parse
        return cls(*parts)


@dataclass
class TraitRunResult:
    """What one trait's run produced. ``tables`` and ``figures`` (PNG bytes) are the
    ZIP's contents by file name; ``context`` is what cli.run_pipeline returns for a
    single trait (it holds the genotype matrix, so multi-trait loops drop it)."""

    trait: str
    zip_name: str
    zip_bytes: bytes
    tables: dict
    figures: dict
    context: dict
    summary: dict = field(default_factory=dict)  # lambda_gc, n_significant, n_ld_blocks, ...
    log: list = field(default_factory=list)


def cli_default_options(**overrides):
    """The CLI's defaults (cli._build_parser) as an options namespace, with
    ``overrides`` applied: how the GUI builds the options it passes to
    run_trait_pipeline, so every setting it does not name behaves as in the CLI."""
    from cli import _build_parser

    args = _build_parser().parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _zip_figures(zip_bytes):
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        return {n.split("/", 1)[1]: zf.read(n) for n in zf.namelist()
                if n.startswith("figures/") and not n.endswith("/")}


class _Collector(logging.Handler):
    """Collects this trait's "cli" log lines; forwards each to ``progress``.

    Only records from the thread that runs the trait are taken, so two GUI
    sessions running at once (one thread each) never see each other's lines."""

    def __init__(self, records, progress=None):
        import threading

        super().__init__(level=logging.INFO)
        self.records, self.progress = records, progress
        self.thread = threading.get_ident()

    def emit(self, record):
        if record.thread != self.thread:
            return
        try:
            msg = record.getMessage()
        except Exception:  # pragma: no cover - a bad format string must not stop a run
            msg = str(record.msg)
        self.records.append(f"[{record.levelname}] {msg}")
        if self.progress is not None:
            try:
                self.progress(msg)
            except Exception:  # pragma: no cover - a UI callback must not stop a run
                pass


# The CLI sets INFO via basicConfig; under Streamlit the "cli" logger would sit at
# WARNING and drop the step messages the collector and progress need. Runs can
# overlap (one thread per GUI session), so the level is lowered by the first run
# that needs it and restored by the last one to finish.
_LEVEL_LOCK = __import__("threading").Lock()
_level_state = {"runs": 0, "saved": None, "lowered": False}


def _info_level_enter():
    with _LEVEL_LOCK:
        if _level_state["runs"] == 0:
            _level_state["saved"] = log.level
            _level_state["lowered"] = log.getEffectiveLevel() > logging.INFO
            if _level_state["lowered"]:
                log.setLevel(logging.INFO)
        _level_state["runs"] += 1


def _info_level_exit():
    with _LEVEL_LOCK:
        _level_state["runs"] -= 1
        if _level_state["runs"] == 0 and _level_state["lowered"]:
            log.setLevel(_level_state["saved"])
            _level_state["lowered"] = False


def run_trait_pipeline(parsed_genotypes, pheno_df, trait, options, progress=None):
    """Run one trait end to end; return a TraitRunResult.

    ``parsed_genotypes`` is a ParsedGenotypes (shared, read-only). ``pheno_df`` is
    the phenotype table indexed by sample ID (as the CLI reads it, index_col=0), with
    ``trait`` among its columns. ``options`` is an argparse namespace with the CLI's
    fields; it is copied, never modified. ``progress``, if given, is called with each
    log message of this trait's run. Raises TraitPipelineError for a trait-level
    stop, and lets any other error propagate to the caller.

    GUI-only: ``options.norm_by_trait`` ({trait: --norm slug}, the phenotype QC
    panel's applied transformations) gives a trait its own normalisation in place of
    ``options.norm``. The CLI never sets it, so its ``--norm`` applies to every trait.

    The figures are drawn with matplotlib's original settings (those a CLI process
    has), whatever the calling process set: the GUI applies its publication theme
    globally, which would otherwise make One-Click's figures differ from the CLI's.
    """
    import matplotlib

    args = argparse.Namespace(**vars(options))
    args.trait = trait
    _norm_by_trait = getattr(args, "norm_by_trait", None)
    if _norm_by_trait is not None:
        args.norm = _norm_by_trait.get(trait, args.norm)
    records = []
    collector = _Collector(records, progress)
    log.addHandler(collector)
    _info_level_enter()
    try:
        with matplotlib.rc_context(rc=_cli_matplotlib_settings()):
            result = _run_trait(parsed_genotypes, pheno_df.copy(), args)
    finally:
        log.removeHandler(collector)
        _info_level_exit()
    result.log = records
    return result


def _cli_matplotlib_settings():
    """matplotlib's settings as loaded at import (the matplotlibrc values a CLI process
    draws with), without the keys a style may not set (backend, interactive, ...). In
    a CLI process they equal the current settings, so applying them changes nothing."""
    import matplotlib
    from matplotlib.style.core import STYLE_BLACKLIST

    return {k: v for k, v in matplotlib.rcParamsOrig.items() if k not in STYLE_BLACKLIST}


def run_traits(parsed_genotypes, pheno_df, traits, options, on_start=None, progress=None):
    """Run several traits one after another; yield ``(trait, result, error)``.

    ``result`` is the trait's TraitRunResult, or None when it failed; ``error`` is
    then the traceback text (else None). One trait's failure never stops the
    others. ``on_start(i, trait)`` is called before trait i (1-based). The caller
    keeps what it needs from each result (the CLI and One-Click keep the ZIP bytes)
    and drops the rest before the next trait runs.
    """
    import traceback

    # One gene-model load for the whole batch (the parsed VCF is shared the same way).
    options = argparse.Namespace(**vars(options))
    options.gene_model_memo = {}
    for i, trait in enumerate(traits, 1):
        if on_start is not None:
            on_start(i, trait)
        try:
            result = run_trait_pipeline(parsed_genotypes, pheno_df, trait, options,
                                        progress=progress)
        except (Exception, SystemExit) as exc:  # isolate the trait, keep the batch
            log.error("Trait %s failed: %s: %s", trait, type(exc).__name__, exc)
            yield trait, None, "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__))
            continue
        yield trait, result, None
        del result  # the caller has it; free it before the next trait runs


def failed_traits_report(traits, completed, failed):
    """Text of ``failed_traits.txt`` in a multi-trait run's combined ZIP: each
    trait's status, then the traceback of every trait that failed."""
    lines = [f"TRACE multi-trait run: {len(completed)} of {len(traits)} traits completed, "
             f"{len(failed)} failed.", ""]
    lines += [f"{t}: {'completed' if t in completed else 'FAILED'}" for t in traits]
    for trait, tb in failed.items():
        lines += ["", f"=== {trait} ===", str(tb).rstrip()]
    return "\n".join(lines) + "\n"


def _n_snps(geno_df, sid):
    """The number of SNP columns, geno_df.shape[1]. With the genotype table on disk the frame
    holds the sample index only, and the SNPs are counted from their IDs (the same number)."""
    from gwas.storepipe import is_store_frame
    return len(sid) if is_store_frame(geno_df) else geno_df.shape[1]


def _run_trait(parsed, pheno, args):
    from gwas.qc import (
        _pipeline_harmonize_ids,
        _pipeline_phenotype_qc,
        _pipeline_snp_qc,
        _pipeline_build_geno_matrices,
        _pipeline_build_kinship,
    )

    # With the genotype table on disk (gwas/genostore.py) the per-trait stages are the store's
    # (gwas/storepipe.py): the same arguments and results, called at the same places.
    _st = None
    if getattr(parsed, "is_store", False):
        from gwas.storepipe import TraitStore
        _st = TraitStore(parsed)
        _pipeline_harmonize_ids = _st.harmonize_ids
        _pipeline_phenotype_qc = _st.phenotype_qc
        _pipeline_snp_qc = _st.snp_qc
        _pipeline_build_geno_matrices = _st.build_geno_matrices
        _pipeline_build_kinship = _st.build_kinship

    vcf_path = Path(args.vcf)
    pheno_path = Path(args.pheno)
    genotypes, samples, chroms, positions = (
        parsed.genotypes, parsed.samples, parsed.chroms, parsed.positions)
    vid, ref, alt = parsed.vid, parsed.ref, parsed.alt
    info_scores, info_field = parsed.info_scores, parsed.info_field
    if _taps.on() and getattr(args, "output", None):
        _taps.start(Path(args.output).parent / f"taps_{args.trait}.json")

    _stagelog.mark("S05_ids")
    log.info("Harmonizing sample IDs…")
    geno_df, pheno, chroms, positions, sid, info_scores, allele_map = \
        _pipeline_harmonize_ids(genotypes, samples, chroms, positions,
                                vid, ref, alt, pheno, info_scores)
    log.info("  %d overlapping samples, %d SNPs", geno_df.shape[0], _n_snps(geno_df, sid))
    _taps.tap("S05_ids", sid=sid, samples=geno_df.index, chroms=chroms, positions=positions,
              info_scores=info_scores, allele_map=allele_map)

    _stagelog.mark("S06_pheno_qc")
    log.info("Phenotype QC…")
    geno_df, pheno, y = _pipeline_phenotype_qc(
        geno_df, pheno, args.trait, args.norm, args.ind_miss,
    )
    log.info("  %d samples after phenotype QC", geno_df.shape[0])
    _taps.tap("S06_pheno_qc", samples=geno_df.index, y=y, pheno=pheno)

    # ── Optional user covariates: load + align to the phenotype-QC'd sample set,
    #    dropping samples that lack a covariate value BEFORE PCA/kinship so every
    #    downstream array (iid/geno/K/PCs) is built on the covar-complete set ──
    _stagelog.mark("S07_covariates")
    covar_df_sel = None
    _covar_frame = getattr(args, "covar_frame", None)  # GUI: the uploaded table, already read
    _cov_dropped, _cov_n_before = [], None
    if getattr(args, "covar", None) or _covar_frame is not None:
        from gwas.covariates import (
            load_covariate_frame, select_covariate_columns, drop_covariate_missing,
            covariate_drop_message,
        )
        _cov_cols = ([c.strip() for c in args.covar_cols.split(",")]
                     if getattr(args, "covar_cols", None) else None)
        covar_df_sel = select_covariate_columns(
            _covar_frame if _covar_frame is not None else load_covariate_frame(args.covar),
            _cov_cols)
        _cov_names = list(covar_df_sel.columns)
        # the samples without a covariate value are dropped here, by the helper the GWAS
        # page's preprocessing calls at the same stage
        _cov_n_before = int(geno_df.shape[0])
        geno_df, pheno, y, _cov_dropped = drop_covariate_missing(covar_df_sel, geno_df, pheno, y)
        _n_drop = len(_cov_dropped)
        if _n_drop:
            log.warning("  --covar: %s", covariate_drop_message(_cov_dropped, _cov_n_before))
        if geno_df.shape[0] == 0:
            raise TraitPipelineError("No samples remain after dropping covariate-missing rows.")
        log.info("  --covar: %d covariate(s) [%s] on %d samples",
                 covar_df_sel.shape[1], ", ".join(_cov_names), geno_df.shape[0])

    _taps.tap("S07_covariates", samples=geno_df.index, y=y)
    _stagelog.mark("S08_snp_qc")
    log.info("SNP QC (MAF=%.3f, miss=%.2f, MAC=%d, INFO=%.2f)…",
             args.maf, args.miss, args.mac, args.info_thresh)
    canonical = (tuple(str(i) for i in range(1, args.n_chromosomes + 1))
                 if args.n_chromosomes else None)
    geno_df, chroms, chroms_num, positions, sid, n_raw, qc_snp, info_scores = \
        _pipeline_snp_qc(geno_df, chroms, positions, sid,
                         args.maf, args.miss, args.mac, True,  # P11: ALT always excluded
                         info_scores, args.info_thresh, canonical=canonical,
                         max_het=args.max_het, het_excess_p=args.het_excess_p)
    log.info("  QC: %s", qc_snp)
    _taps.tap("S08_snp_qc", sid=sid, chroms=chroms, chroms_num=chroms_num, positions=positions,
              n_raw=n_raw, qc_snp=dict(qc_snp), info_scores=info_scores, samples=geno_df.index)

    _stagelog.mark("S09_matrices")
    log.info("Building genotype matrices (impute=%s)…", args.impute)
    geno_dosage_raw, snp_imputation_rate, geno_imputed, iid = \
        _pipeline_build_geno_matrices(geno_df, method=args.impute,
                                      impute_k=args.impute_k, impute_l=args.impute_l)
    qc_snp["Imputation method"] = args.impute
    if _taps.on():
        _taps.tap("S09_matrices", raw=_taps.table(geno_dosage_raw), imputed=_taps.table(geno_imputed),
                  imputation_rate=snp_imputation_rate, iid=iid)

    _stagelog.mark("S10_kinship")
    log.info("Building kinship matrix…")
    K, Z_for_pca, chroms_grm, positions_grm, kinship_model = \
        _pipeline_build_kinship(geno_df, geno_imputed, chroms, positions)
    log.info("  Kinship model: %s", kinship_model)
    _taps.tap("S10_kinship", K=K, Z_for_pca=Z_for_pca, chroms_grm=chroms_grm,
              positions_grm=positions_grm, kinship_model=kinship_model)

    # ── PCA ──────────────────────────────────────────────
    _stagelog.mark("S11_pca")
    from gwas.kinship import _compute_pcs_full_impl
    pcs_full, pca_eigenvalues = _compute_pcs_full_impl(Z_for_pca, max_pcs=20)
    _max_avail = pcs_full.shape[1] if pcs_full is not None else 0
    n_pcs = min(args.n_pcs, _max_avail) if pcs_full is not None else 0
    _taps.tap("S11_pca", pcs_full=pcs_full, pca_eigenvalues=pca_eigenvalues)

    from gwas.plotting import (
        compute_cumulative_positions, plot_manhattan_static, plot_qq, compute_lambda_gc,
    )

    # ── LOCO kernels ─────────────────────────────────────
    _stagelog.mark("S12_loco")
    from gwas.kinship import _build_loco_kernels_impl
    K0, K_by_chr, _ = _build_loco_kernels_impl(iid, Z_for_pca, chroms_grm, K)
    if _taps.on():
        _taps.tap("S12_loco", K0=np.asarray(K0.val),
                  **{f"K_chr_{_c}": np.asarray(_k.val) for _c, _k in sorted(K_by_chr.items())})

    if getattr(args, "no_loco", False):
        K_by_chr = {ch: K0 for ch in K_by_chr}
        log.info("  --no-loco: using global kinship for all chromosomes")

    # ── Phenotype reader ─────────────────────────────────
    from gwas.utils import PhenoData, CovarData
    pheno_reader = PhenoData(iid, y)

    # Aligned user-covariate matrix (in iid order); None unless --covar was given.
    user_covar_mat = None
    user_covar_names = None
    if covar_df_sel is not None:
        from gwas.covariates import align_covariates as _align_cov
        user_covar_mat, user_covar_names, _ = _align_cov(covar_df_sel, iid[:, 0])

    def _make_covar(k):
        if user_covar_mat is None:
            # original path — byte-identical when no user covariates are supplied
            if pcs_full is None or k <= 0:
                return None
            return CovarData(iid, pcs_full[:, :k])
        _pc = pcs_full[:, :k] if (pcs_full is not None and k > 0) else None
        if _pc is None:
            return CovarData(iid, user_covar_mat, names=list(user_covar_names))
        return CovarData(iid, np.c_[_pc, user_covar_mat],
                         names=[f"PC{i + 1}" for i in range(int(k))] + list(user_covar_names))

    _stagelog.mark("S13_qc_report")
    extra_csvs = {}  # additional CSVs to include in ZIP
    figures = {}     # figures to include in ZIP

    # ── QC report (P1-P5): report-only statistics; never removes a sample or
    #    marker and never breaks the run (wrapped) ──
    try:
        from gwas import qc_report as _qcr
        _qc_trait = (pheno[args.trait].to_numpy()
                     if args.trait in getattr(pheno, "columns", []) else y.ravel())
        _qc_rep = (_st.qc_report if _st is not None else _qcr.compute_qc_report)(
            geno_df, _qc_trait, args.trait, qc_snp=qc_snp,
            geno_dosage_raw=geno_dosage_raw, geno_imputed=geno_imputed,
            impute_method=args.impute, impute_k=args.impute_k, impute_l=args.impute_l)
        extra_csvs.update(_qcr.qc_report_dataframes(_qc_rep))
        figures.update(_qcr.qc_report_figures(_qc_rep))
        log.info("  QC report: %d het-outlier(s) |z|>3, %d relatedness pair(s), median F_IS %.3f",
                 _qc_rep["sample_het"]["n_flagged"],
                 _qc_rep["dup_pairs"]["n_pairs_flagged"],
                 _qc_rep["variant_fis"]["median_fis"])
    except Exception as _qc_err:  # pragma: no cover
        log.warning("QC report skipped: %s", _qc_err)

    _stagelog.mark("S14_export_qc")
    _exp_imputed, _exp_raw = geno_imputed, geno_dosage_raw
    if getattr(args, "export_qc", False) and _st is not None:
        # The export writes the whole table as text. With the table on disk, up to the
        # one-piece size the two tables are read whole for it; above it the export is refused.
        from gwas.storepipe import one_piece_cells
        if geno_imputed.size > one_piece_cells():
            raise TraitPipelineError(
                "--export-qc writes the whole genotype table as text and is not available "
                "at this size with the genotype table on disk.")
        _all = np.arange(geno_imputed.shape[1])           # a whole read by design (logged)
        _exp_imputed = geno_imputed.block(_all, _whole=True)
        _exp_raw = geno_dosage_raw.block(_all, _whole=True)
    if getattr(args, "export_qc", False):
        log.info("Preparing QC'd genotype export for benchmarking…")
        geno_export = pd.DataFrame(
            _exp_imputed, index=list(geno_df.index), columns=sid,
        )
        geno_export.index.name = "SampleID"
        extra_csvs["QC_genotype_matrix.csv"] = geno_export.reset_index()
        # P3: also export the RAW dosage matrix (NaN preserved) -- the imputed
        # export destroys the missing-call evidence.  QC_genotype_matrix.csv is
        # unchanged; this is an additional file.
        geno_raw_export = pd.DataFrame(
            _exp_raw, index=list(geno_df.index), columns=sid,
        )
        geno_raw_export.index.name = "SampleID"
        extra_csvs["QC_genotype_raw.csv"] = geno_raw_export.reset_index()
        ref_vals = [allele_map.get(s, (".", "."))[0] for s in sid]
        alt_vals = [allele_map.get(s, (".", "."))[1] for s in sid]
        extra_csvs["QC_snp_map.csv"] = pd.DataFrame({
            "SNP_ID": sid, "Chr": chroms, "Pos": positions,
            "Ref": ref_vals, "Alt": alt_vals,
        })
        extra_csvs["QC_phenotype.csv"] = pd.DataFrame({
            "SampleID": list(geno_df.index), args.trait: y.ravel(),
        })
        log.info("  QC export: %d samples x %d SNPs (impute=%s; raw missing %.3f%%)",
                 geno_imputed.shape[0], geno_imputed.shape[1], args.impute,
                 float(np.isnan(_exp_raw).mean()) * 100.0)

    # Per-model PC counts (fall back to --n-pcs when not specified)
    n_pcs_mlm = min(args.n_pcs_mlm if args.n_pcs_mlm is not None else n_pcs, _max_avail)
    n_pcs_mlmm = min(args.n_pcs_mlmm if args.n_pcs_mlmm is not None else n_pcs_mlm, _max_avail)
    n_pcs_fc = min(args.n_pcs_farmcpu if args.n_pcs_farmcpu is not None else n_pcs, _max_avail)

    # ── PC-selection diagnostics (report-only; REPORTS, never selects) ──
    # The lambda-GC auto-PC selector was removed (D-FINAL): TRACE uses the fixed
    # --n-pcs default and reports the genotype-PCA eigenvalue spectrum so the user can
    # judge the choice.  No criterion, no recommended k; one eigendecomposition, no model fits.
    _stagelog.mark("S15_pc_diagnostics")
    _pc_diag = None
    try:
        from gwas.pc_diagnostics import compute_pc_diagnostics
        _pc_diag = compute_pc_diagnostics(
            Z_for_pca, n=int(Z_for_pca.shape[0]), m=int(Z_for_pca.shape[1]),
            prune_params={"r2": 0.2, "window_bp": 500_000, "step_bp": 100_000},
            spectrum_depth=20,
        )
        extra_csvs["PC_diagnostics_spectrum.csv"] = _pc_diag["spectrum"]
        log.info("  PC diagnostics: %d eigenvalues, trace/m=%.3f "
                 "(eigenvalue spectrum -> PC_diagnostics_spectrum.csv; TRACE uses the fixed --n-pcs)",
                 _pc_diag["meta"]["n_eigenvalues"], _pc_diag["meta"]["trace_over_m"])
    except Exception as _pcd_err:  # pragma: no cover
        log.warning("  PC diagnostics skipped: %s", _pcd_err)

    log.info("  PCs: MLM=%d, MLMM=%d, FarmCPU=%d", n_pcs_mlm, n_pcs_mlmm, n_pcs_fc)

    covar_mlm = _make_covar(n_pcs_mlm)
    covar_mlmm = _make_covar(n_pcs_mlmm)
    covar_fc = _make_covar(n_pcs_fc)

    # ── Run GWAS models ──────────────────────────────────
    from gwas.models import _run_gwas_impl, add_ols_effects_to_gwas
    from gwas.significance import add_significance_columns

    gwas_df = None
    extra_model_dfs = {}
    cofactor_tables = {}

    _stagelog.mark("S16_meff")
    # Compute M_eff (always, for reference columns)
    try:
        # the uncached function, in both modes (Streamlit's cache cannot hash a table on disk);
        # it sums its matrix in blocks of SNPs and gives the whole number (gwas/plotting.py)
        from gwas.plotting import _compute_meff_li_ji_impl
        _meff_val, _ = _compute_meff_li_ji_impl(geno_imputed)
        meff_thresh = 0.05 / _meff_val
        log.info("M_eff = %d independent tests (of %d SNPs)", _meff_val, _n_snps(geno_df, sid))
    except Exception:
        _meff_val = _n_snps(geno_df, sid)
        meff_thresh = 0.05 / _meff_val
        log.warning("M_eff computation failed; using naive Bonferroni")

    _taps.tap("S16_meff", meff=int(_meff_val), meff_thresh=float(meff_thresh))
    bonf_thresh_naive = 0.05 / _n_snps(geno_df, sid)

    # Primary threshold based on user choice
    sig_rule = getattr(args, "sig_thresh", "bonferroni")
    _sig_is_numeric = isinstance(sig_rule, (int, float)) and not isinstance(sig_rule, bool)
    if sig_rule == "bonferroni":
        primary_thresh = bonf_thresh_naive
        log.info("Significance: Bonferroni %.2e (%d SNPs)", primary_thresh, _n_snps(geno_df, sid))
    elif sig_rule == "fdr":
        primary_thresh = None  # FDR uses per-SNP q-values
        log.info("Significance: FDR q < 0.05")
    else:  # meff (default)
        primary_thresh = meff_thresh
        log.info("Significance: M_eff %.2e (M_eff=%d)", primary_thresh, _meff_val)

    if _sig_is_numeric:
        # a user-specified fixed p-value threshold overrides the named-rule value
        primary_thresh = float(sig_rule)
        log.info("Significance: p < %.2e (user-specified)", primary_thresh)

    # One rule for every model's table (MLM, MLMM, FarmCPU): MLM's Bonferroni, M_eff
    # and custom thresholds, with Benjamini-Hochberg FDR within each model's own p-values.
    _sig_cols = {"bonf_thresh": bonf_thresh_naive, "meff_thresh": meff_thresh,
                 "custom_thresh": primary_thresh if _sig_is_numeric else None}
    # The analysis rule, resolved once: the cross-model consensus, the isolated-SNP
    # rescue, the significant-SNP tables and the report read each model's own column.
    from gwas.significance import rule_from_cli_args
    _sig_rule_obj = rule_from_cli_args(args, _n_snps(geno_df, sid), _meff_val)

    if "mlm" in args.model:
        _stagelog.mark("S17_mlm")
        log.info("Running MLM (LOCO)…")
        gwas_df = _run_gwas_impl(
            geno_imputed, y, pcs_full, n_pcs_mlm, sid, positions,
            chroms, chroms_num, iid, K0, K_by_chr, pheno_reader, args.trait,
            user_covar=user_covar_mat, user_covar_names=user_covar_names,
        )
        gwas_df["PValue"] = np.clip(gwas_df["PValue"].astype(float), 1e-300, 1.0)
        gwas_df["-log10p"] = -np.log10(gwas_df["PValue"])
        try:
            _stagelog.mark("S18_ols_effects")
            gwas_df = add_ols_effects_to_gwas(gwas_df, geno_imputed, y, covar_mlm, sid)
        except Exception as e:
            log.warning("OLS effects failed: %s", e)
        _stagelog.mark("S18b_mlm_table")
        gwas_df = add_significance_columns(gwas_df, **_sig_cols)
        gwas_df["Model"] = "MLM"

        # Imputation rate
        _imp = pd.DataFrame({"SNP": sid.astype(str), "ImputationRate": snp_imputation_rate})
        gwas_df = gwas_df.merge(_imp, on="SNP", how="left")

        log.info("  MLM: %d significant (M_eff), %d (FDR<0.05)",
                 gwas_df["Significant_Meff"].sum(), gwas_df["Significant_FDR"].sum())

    # ── LD decay estimation (ahead of the models: MLMM takes the LD window as its
    #    cofactor window; the estimate needs only the genotypes) ──
    _stagelog.mark("S19_ld_decay")
    import gwas.ld as ld
    from gwas.windows import ld_window_kb
    ld_flank_kb = args.ld_flank_kb
    ld_decay_kb = None
    ld_blocks_mlm = None  # track MLM blocks for subsampling aggregation

    _ld_decay_n_censored = None  # LD-decay Tier 1b: grid-limited per-chr count (run_metadata)
    if ld_flank_kb is None:
        log.info("Estimating LD decay…")
        ld_distances = []
        _ld_decay_n_censored = 0
        _geno_float = geno_imputed.astype(float)
        for ch in np.unique(chroms):
            ch_mask = chroms == ch
            pos_ch = positions[ch_mask]
            if _is_lazy(_geno_float):
                # on disk: read only the columns the estimate uses (at most 1,500)
                if len(pos_ch) < 50:
                    continue
                _ci = np.flatnonzero(ch_mask)
                if len(pos_ch) > 1500:
                    idx = np.linspace(0, len(pos_ch) - 1, 1500, dtype=int)
                    pos_ch, _ci = pos_ch[idx], _ci[idx]
                geno_ch = _geno_float[:, _ci]
            else:
                geno_ch = _geno_float[:, ch_mask]
                if len(pos_ch) < 50:
                    continue
                if len(pos_ch) > 1500:
                    idx = np.linspace(0, len(pos_ch) - 1, 1500, dtype=int)
                    pos_ch, geno_ch = pos_ch[idx], geno_ch[:, idx]
            r2_mat = ld.pairwise_r2(geno_ch)
            dk, _, _dfld = ld.ld_decay(pos_ch, r2_mat, ld_threshold=0.2, max_dist_kb=5000)
            if np.isfinite(dk):
                ld_distances.append(dk)
                if _dfld.attrs.get("ld_decay_censored"):
                    _ld_decay_n_censored += 1
        if ld_distances:
            ld_decay_kb = float(np.median(ld_distances))
            ld_flank_kb = ld_window_kb(ld_decay_kb)    # int(2 x decay)
            # LD-decay Tier 1b: N grid-limited -> the median is partly a median of floors
            # (an upper bound). Display/logging only; ld_decay_kb / flank unchanged.
            _cens = (f" ({_ld_decay_n_censored} of {len(ld_distances)} chromosomes "
                     f"grid-limited <= floor)") if _ld_decay_n_censored else ""
            log.info("  LD decay ~ %.0f kb -> flank = %d kb%s", ld_decay_kb, ld_flank_kb, _cens)
        else:
            ld_flank_kb = ld_window_kb(None)            # 300 kb
            log.warning("  Could not estimate LD decay; using 300 kb")

    _stagelog.mark("S20_mlmm")
    if "mlmm" in args.model and gwas_df is not None:
        from gwas.models import run_mlmm_research_grade_fast
        log.info("Running MLMM…")
        # GUI-only settings (no CLI flag); absent -> the library defaults, as before
        _mlmm_opts = {k: v for k, v in (("p_enter", getattr(args, "mlmm_p_enter", None)),
                                         ("max_cof", getattr(args, "mlmm_max_cof", None)))
                      if v is not None}
        # the cofactor window is the run's LD window (--ld-flank-kb, else int(2 x decay),
        # else 300 kb), as the GWAS page's run passes it
        _mlmm_window_kb = int(ld_flank_kb)
        gwas_mlmm, cof_tbl = run_mlmm_research_grade_fast(
            geno_imputed, sid, chroms, chroms_num, positions, iid,
            pheno_reader, K0, covar_mlmm, verbose=False, window_kb=_mlmm_window_kb,
            **_mlmm_opts,
        )
        gwas_mlmm["PValue"] = np.clip(gwas_mlmm["PValue"].astype(float), 1e-300, 1.0)
        gwas_mlmm["-log10p"] = -np.log10(gwas_mlmm["PValue"])
        gwas_mlmm = add_significance_columns(gwas_mlmm, **_sig_cols)
        extra_model_dfs["MLMM"] = gwas_mlmm
        cofactor_tables["MLMM"] = cof_tbl
        log.info("  MLMM: %d cofactors selected", len(cof_tbl))
        log.info("  MLMM: %d significant (M_eff), %d (FDR<0.05)",
                 gwas_mlmm["Significant_Meff"].sum(), gwas_mlmm["Significant_FDR"].sum())

    _stagelog.mark("S21_farmcpu")
    if "farmcpu" in args.model and gwas_df is not None:
        from gwas.models import run_farmcpu
        log.info("Running FarmCPU…")
        gwas_farmcpu, pqtn_tbl, conv_info = run_farmcpu(
            geno_imputed, sid, chroms, chroms_num, positions, iid,
            pheno_reader, K0, covar_fc,
            final_scan="ols", verbose=False,  # D-103: FarmCPU MLM final scan frozen out (published OLS)
            use_loco=not getattr(args, "no_loco", False),
            **({"max_iterations": int(args.farmcpu_max_iter)}
               if getattr(args, "farmcpu_max_iter", None) else {}),  # GUI-only setting
        )
        gwas_farmcpu["PValue"] = np.clip(gwas_farmcpu["PValue"].astype(float), 1e-300, 1.0)
        gwas_farmcpu["-log10p"] = -np.log10(gwas_farmcpu["PValue"])
        gwas_farmcpu = add_significance_columns(gwas_farmcpu, **_sig_cols)
        extra_model_dfs["FarmCPU"] = gwas_farmcpu
        cofactor_tables["FarmCPU"] = pqtn_tbl
        log.info("  FarmCPU: %s, %d pseudo-QTNs",
                 "converged" if conv_info["converged"] else "max iter",
                 conv_info["n_pseudo_qtns"])
        log.info("  FarmCPU: %d significant (M_eff), %d (FDR<0.05)",
                 gwas_farmcpu["Significant_Meff"].sum(), gwas_farmcpu["Significant_FDR"].sum())

    if gwas_df is None:
        log.error("No primary GWAS model ran. At least 'mlm' is required.")
        raise TraitPipelineError("No primary GWAS model ran. At least 'mlm' is required.",
                                 exit_arg=1)

    # ── Cross-model consensus ───────────────────────────
    # Mirrors the Streamlit consensus block (pages/GWAS_analysis.py).
    # Only emitted when at least two of {MLM, MLMM, FarmCPU} ran.
    _stagelog.mark("S22_consensus")
    _consensus_models = [("MLM", gwas_df)]
    for _mname in ["MLMM", "FarmCPU"]:
        if _mname in extra_model_dfs:
            _consensus_models.append((_mname, extra_model_dfs[_mname]))

    if len(_consensus_models) >= 2:
        try:
            _sig_by_model = {}
            for _mname, _mdf in _consensus_models:
                # each model's own column for the analysis rule (add_significance_columns),
                # so a tie at a threshold is decided as in the tables (FDR: the
                # Benjamini-Hochberg decision, p <= rank/m x 0.05)
                _sig_by_model[_mname] = set(
                    _mdf.loc[_sig_rule_obj.significant_mask(_mdf), "SNP"].astype(str))

            _all_sig = sorted(set().union(*_sig_by_model.values()))
            if _all_sig:
                _rows = []
                # SNP -> best (smallest) p across models for sorting
                _best_p = {}
                for _mname, _mdf in _consensus_models:
                    _sub = _mdf[_mdf["SNP"].astype(str).isin(_all_sig)][["SNP", "PValue"]]
                    for _snp, _p in zip(_sub["SNP"].astype(str), _sub["PValue"]):
                        _best_p[_snp] = min(_best_p.get(_snp, np.inf), float(_p))
                for _snp in _all_sig:
                    _detected_by = sorted(
                        _mname for _mname, _s in _sig_by_model.items() if _snp in _s
                    )
                    _rows.append({
                        "SNP": _snp,
                        "N_models": len(_detected_by),
                        "Detected_by": ",".join(_detected_by),
                        "Best_PValue": _best_p.get(_snp, np.nan),
                    })
                _consensus_df = (
                    pd.DataFrame(_rows)
                    .sort_values(["N_models", "Best_PValue"], ascending=[False, True])
                    .reset_index(drop=True)
                )
                extra_csvs["CrossModel_Consensus.csv"] = _consensus_df
                _n_full = int((_consensus_df["N_models"] == len(_consensus_models)).sum())
                log.info("  Cross-model consensus: %d SNPs significant in %d/%d models",
                         _n_full, len(_consensus_models), len(_consensus_models))
        except Exception as e:
            log.warning("Cross-model consensus failed: %s", e)

    # ── Per-model post-GWAS: LD blocks → haplotype → annotation ──
    from gwas.haplotype import run_haplotype_block_gwas
    from annotation import canon_chr

    _stagelog.mark("S23_post_setup")
    post_gwas_models = [("MLM", gwas_df)]
    for mname in ["MLMM", "FarmCPU"]:
        if mname in extra_model_dfs:
            post_gwas_models.append((mname, extra_model_dfs[mname]))

    _geno_float = geno_imputed.astype(float)
    pheno_clean = pheno  # phenotype DataFrame for haplotype testing
    # LD heatmaps: (chr, start, end) -> the lead SNP of the first model that reports the
    # block with a significant lead, every model that does ("finders"), and whether the
    # block has the 2 SNPs a heatmap needs. One heatmap per block, drawn after the
    # per-model loop. _hm_sig_col is the analysis rule's column, in every model's table.
    _heatmap_blocks = {}
    _hm_sig_col = ("Significant_Custom" if _sig_is_numeric
                   else "Significant_Meff" if sig_rule == "meff"
                   else ("Significant_FDR" if sig_rule == "fdr" else "Significant_Bonf"))
    chroms_canon = np.array([canon_chr(str(c)) for c in chroms])

    # ── Isolated-SNP rescue setup (T-45) ──────────────────
    # Reporting rule + gene model resolved ONCE, before the per-model loop, so the
    # rescue can mine genes even for models with no LD blocks. The rescue is a
    # positional set difference against the FINAL emitted block table and NEVER
    # calls the block detector (gwas/isolated.py has no gwas.ld import).
    from gwas import isolated as _iso
    _do_rescue = not getattr(args, "no_isolated_rescue", False)
    _iso_edge_flank_bp = _iso.default_edge_flank_bp(
        getattr(args, "isolated_edge_flank_kb", None), ld_flank_kb)
    _iso_max_interval_bp = int(float(getattr(args, "isolated_max_interval_mb", 5.0)) * 1_000_000)
    _iso_low_res_bp = (int(args.isolated_low_res_kb) * 1000
                       if getattr(args, "isolated_low_res_kb", None) else None)
    _iso_ld_decay_bp = int(ld_decay_kb * 1000) if ld_decay_kb else None
    _iso_counts = {}

    # ── Gene model: resolved ONCE and loaded ONCE, for every consumer — the
    #    isolated-SNP rescue, the per-model block annotation, and the run
    #    manifest. One answer is what makes it structurally impossible for
    #    run_metadata.json to name a gene model that annotation never used.
    #    Gated on --no-annotation alone: opting out of the isolated rescue must
    #    not silently disable block annotation as well.
    from annotation import resolve_gene_model as _resolve_gm
    _gm_res = _resolve_gm(
        args.species,
        genome_build=getattr(args, "genome_build", "SL3"),
        override=args.gene_model,
    )
    _gm_status = "disabled (--no-annotation)" if args.no_annotation else _gm_res.status
    _iso_genes_df = None
    if not args.no_annotation:
        if not _gm_res.found:
            log.warning(
                "  Gene model unavailable (%s). Candidate-gene annotation is SKIPPED: "
                "no Isolated_SNP_candidate_genes_*.csv will be written and the LD-block "
                "tables will carry no gene columns.", _gm_res.status)
        else:
            try:
                from annotation import load_gene_annotation as _iso_load_genes
                _gm_key = (str(_gm_res.gene_model),
                           getattr(args, "gene_desc", None)  # GUI: uploaded descriptions
                           or (str(_gm_res.descriptions) if _gm_res.descriptions else None))
                # run_traits shares one loaded gene model across a batch's traits; each
                # trait gets its own copy, so none can change another's.
                _gm_memo = getattr(args, "gene_model_memo", None)
                if _gm_memo is not None and _gm_key in _gm_memo:
                    _iso_genes_df = _gm_memo[_gm_key].copy()
                else:
                    _iso_genes_df = _iso_load_genes(*_gm_key)
                    if _gm_memo is not None:
                        _gm_memo[_gm_key] = _iso_genes_df.copy()
            except Exception as e:
                log.warning("  Gene model load failed (%s: %s); candidate-gene annotation "
                            "is SKIPPED.", _gm_res.gene_model, e)
                _gm_status = f"load failed: {_gm_res.gene_model} ({e})"

    # ── Post-GWAS frames captured for the HTML report (T-80) ─────────────
    # The report previously omitted every post-GWAS table (defect 0.2). Capture
    # the primary model's frames (MLM if run, else the first model) plus each
    # model's annotation/haplotype. Additive: when a frame is absent the report
    # renders exactly as before (every report section is {% if %}-guarded).
    _primary_model = "MLM" if any(m == "MLM" for m, _ in post_gwas_models) else (
        post_gwas_models[0][0] if post_gwas_models else None)
    _report_sig_table = _report_unblocked = _report_isolated = None
    _report_ld_annotated = _report_hap_gwas = None
    _report_ld_blocks = _report_assignments = None   # the per-SNP plots' haplotype colours
    _per_model_post = {}

    # P14: LD-block seed mode (mirrors the GUI radio). "suggestive" (default) =
    # the --ld-seed-p threshold + the --ld-top-n floor -> byte-identical to the
    # prior CLI behaviour; "significant" = seeds restricted to the genome-wide
    # --sig-thresh, with NO top-N floor. FDR (no single p-threshold) falls back to
    # naive Bonferroni for the seed threshold, matching the GUI.
    _ld_gw_thresh = primary_thresh if primary_thresh is not None else bonf_thresh_naive
    if args.ld_seed_mode == "significant":
        _ld_seed_thresh, _ld_seed_top_n = _ld_gw_thresh, 0
    else:
        _ld_seed_thresh, _ld_seed_top_n = args.ld_seed_p, args.ld_top_n

    for model_name, model_df in post_gwas_models:
        _stagelog.mark(f"S24_ld_blocks:{model_name}")
        log.info("Post-GWAS: %s", model_name)
        m_hap_gwas = None
        m_hap_tables = {}   # the haplotype test's per-block sample tables
        m_ld_annotated = None
        m_ld_blocks = pd.DataFrame()   # always defined; empty => no blocks (rescue still runs)

        # LD block detection
        # P15: report both seed counts so the suggestive vs significant distinction
        # is visible regardless of --ld-seed-mode (the top-N floor can also form
        # non-significant blocks -- e.g. pepper forms 3 FarmCPU blocks this way).
        _n_sugg = int((model_df["PValue"] < args.ld_seed_p).sum())
        _n_sig = int((model_df["PValue"] < _ld_gw_thresh).sum())
        log.info("  %d seed SNPs at p < %.1e, of which %d pass the genome-wide "
                 "threshold (%.2e)", _n_sugg, args.ld_seed_p, _n_sig, _ld_gw_thresh)

        has_seeds = (model_df["PValue"] < _ld_seed_thresh).any()
        if not has_seeds:
            # Also try top-N seeding
            if _ld_seed_top_n > 0:
                log.info("  No seed SNPs at p < %.1e; using top-%d seeding.",
                         _ld_seed_thresh, _ld_seed_top_n)
            else:
                log.info("  No seed SNPs (p < %.1e) and no top-N seeding; no LD blocks "
                         "(isolated-SNP rescue still runs).", _ld_seed_thresh)

        try:
            m_ld_blocks = ld.find_ld_clusters_genomewide(
                gwas_df=model_df, chroms=chroms, positions=positions,
                geno_imputed=_geno_float, sid=sid,
                ld_threshold=args.ld_r2, flank_kb=ld_flank_kb,
                ld_decay_kb=ld_decay_kb, min_snps=2,
                top_n=_ld_seed_top_n, sig_thresh=_ld_seed_thresh,
                ld_merge_r2=args.ld_merge_r2,
            )
            m_ld_blocks, _ = ld.filter_contained_blocks(m_ld_blocks, min_contained=2)
        except Exception as e:
            log.warning("  LD block detection failed for %s: %s", model_name, e)
            m_ld_blocks = pd.DataFrame()

        log.info("  %d LD blocks detected", len(m_ld_blocks))

        if model_name == "MLM":
            ld_blocks_mlm = m_ld_blocks

        # ── Isolated-SNP rescue (T-45): runs for EVERY model, with or without
        #    blocks. Set difference vs the FINAL block table; the block detector
        #    is untouched. Emits parallel CSVs; never writes into m_ld_blocks. ──
        if _do_rescue:
            try:
                _stagelog.mark(f"S25_rescue:{model_name}")
                _rescue = _iso.run_isolated_snp_rescue(
                    model_df, m_ld_blocks, _sig_rule_obj, chroms, positions, sid,
                    genes=_iso_genes_df, seed_p_used=args.ld_seed_p, top_n_used=args.ld_top_n,
                    edge_flank_bp=_iso_edge_flank_bp, max_interval_bp=_iso_max_interval_bp,
                    low_res_bp=_iso_low_res_bp, ld_decay_bp=_iso_ld_decay_bp,
                )
                if _rescue.n_uncovered > 0:
                    extra_csvs[f"Isolated_SNP_intervals_{model_name}.csv"] = _rescue.intervals
                    if _rescue.genes_long is not None and not _rescue.genes_long.empty:
                        extra_csvs[f"Isolated_SNP_candidate_genes_{model_name}.csv"] = _rescue.genes_long
                    if model_name == _primary_model:
                        _report_isolated = _rescue.intervals
                _iso_counts[model_name] = {
                    "n_uncovered": _rescue.n_uncovered, "n_intervals": _rescue.n_intervals,
                    "n_seeding_path": _rescue.n_seeding_path, "n_block_path": _rescue.n_block_path,
                }
                log.info("  Isolated-SNP rescue: %d significant SNP(s) with no LD block "
                         "(%d seeding-threshold, %d block-formation) -> %d interval(s)",
                         _rescue.n_uncovered, _rescue.n_seeding_path,
                         _rescue.n_block_path, _rescue.n_intervals)
            except Exception as e:
                log.warning("  Isolated-SNP rescue failed for %s: %s", model_name, e)

        # ── Significant-SNP table (T-20): every reporting-significant SNP, marked
        #    in_block/unblocked with candidate interval + gene evidence. Same escape
        #    hatch as the rescue so --no-isolated-rescue stays byte-clean. Runs for
        #    every model (block or not) before the empty-block skip below. ──
        if _do_rescue:
            try:
                _stagelog.mark(f"S26_sigtable:{model_name}")
                from gwas.sigtable import build_significant_snp_table, project_unblocked
                _sigtab = build_significant_snp_table(
                    model_df, m_ld_blocks, _sig_rule_obj, chroms, positions, sid,
                    geno_dosage_raw=geno_dosage_raw, genes=_iso_genes_df,
                    seed_p_used=args.ld_seed_p, top_n_used=args.ld_top_n,
                    edge_flank_bp=_iso_edge_flank_bp, max_interval_bp=_iso_max_interval_bp,
                    low_res_bp=_iso_low_res_bp, ld_decay_bp=_iso_ld_decay_bp,
                    genome_build=getattr(args, "genome_build", "SL3"), species=args.species,
                )
                if not _sigtab.empty:
                    _unb = project_unblocked(_sigtab)
                    extra_csvs[f"Significant_SNPs_{model_name}.csv"] = _sigtab
                    extra_csvs[f"Unblocked_SNPs_{model_name}.csv"] = _unb
                    if model_name == _primary_model:
                        _report_sig_table, _report_unblocked = _sigtab, _unb
                    log.info("  Significant-SNP table: %d rows (%d unblocked) for %s",
                             len(_sigtab), int((_sigtab["Block_Status"] != "in_block").sum()), model_name)
            except Exception as e:
                log.warning("  Significant-SNP table failed for %s: %s", model_name, e)

        if m_ld_blocks.empty:
            continue

        # LD heatmaps for blocks whose lead SNP is significant in this model (its column
        # for the analysis rule). Each shows the block and its neighbouring SNPs
        # (gwas.plotting.plot_ld_block_heatmap, shared with the Post-GWAS Local LD tab).
        # They are drawn after this loop, one per block (chromosome, start, end) however
        # many models report it. The GUI's "Generate LD block heatmap PNGs" checkbox sets
        # no_ld_heatmaps; the CLI has no such flag (--no-plots).
        _stagelog.mark(f"S27_heatmap_setup:{model_name}")
        if not args.no_plots and not getattr(args, "no_ld_heatmaps", False):
            try:
                _sig_snps = set()
                if _hm_sig_col in model_df.columns:
                    _sig_snps = set(model_df.loc[model_df[_hm_sig_col], "SNP"].astype(str))
                for _, _brow in m_ld_blocks.iterrows():
                    _lead = str(_brow.get("lead_snp", "")).strip()
                    if _lead not in _sig_snps:
                        continue
                    _bchr = canon_chr(str(_brow["Chr"]))
                    _bstart = int(_brow.get("Start (bp)", _brow.get("Start", 0)))
                    _bend = int(_brow.get("End (bp)", _brow.get("End", 0)))
                    _block_key = (_bchr, _bstart, _bend)
                    _hm = _heatmap_blocks.get(_block_key)
                    if _hm is not None:  # an earlier model reports a block with these bounds
                        if model_name not in _hm["finders"]:
                            _hm["finders"].append(model_name)
                        continue
                    _in_block = (chroms_canon == _bchr) & (positions >= _bstart) & (positions <= _bend)
                    _heatmap_blocks[_block_key] = {"lead": _lead, "finders": [model_name],
                                                   "draw": bool(np.count_nonzero(_in_block) >= 2)}
            except Exception as e:
                log.warning("  LD heatmap generation failed for %s: %s", model_name, e)

        # Haplotype testing
        try:
            _stagelog.mark(f"S28_haplotype:{model_name}")
            _hap_pcs = pcs_full[:, :n_pcs_mlm] if pcs_full is not None and n_pcs_mlm > 0 else None
            m_hap_gwas, m_hap_tables = run_haplotype_block_gwas(
                haplo_df=m_ld_blocks, chroms=chroms, positions=positions,
                geno_imputed=_geno_float, sid=sid,
                geno_df=geno_df, pheno_df=pheno_clean, trait_col=args.trait,
                pcs=_hap_pcs, n_perm=args.hap_perms, n_pcs_used=n_pcs_mlm,
                min_hap_count=args.hap_min_count, min_group_size=args.hap_min_group_size,
                user_covar=user_covar_mat,
            )
            if m_hap_gwas is not None and not m_hap_gwas.empty:
                n_sig_hap = int((m_hap_gwas.get("FDR_BH", pd.Series(dtype=float)) < 0.05).sum())
                log.info("  Haplotype: %d/%d blocks significant", n_sig_hap, len(m_hap_gwas))
        except Exception as e:
            log.warning("  Haplotype testing failed for %s: %s", model_name, e)

        # Each block's candidate interval: from the marker before its first SNP to the
        # marker after its last, by the isolated-SNP rule (edge flank, 5 Mb clamp). The
        # block's own Start/End stay its member span, which the LD steps use.
        _stagelog.mark(f"S29_annotation:{model_name}")
        m_blocks_iv = m_ld_blocks
        try:
            _biv = _iso.block_candidate_intervals(
                m_ld_blocks, chroms, positions, sid,
                edge_flank_bp=_iso_edge_flank_bp, max_interval_bp=_iso_max_interval_bp)
            m_blocks_iv = m_ld_blocks.copy()
            _at = list(m_blocks_iv.columns).index("End (bp)") + 1
            m_blocks_iv.insert(_at, "Interval_start", _biv["Interval_start"].to_numpy())
            m_blocks_iv.insert(_at + 1, "Interval_end", _biv["Interval_end"].to_numpy())
        except Exception as e:
            log.warning("  Block candidate intervals failed for %s: %s", model_name, e)

        # Gene annotation over each block's candidate interval. The gene frame was
        # resolved and loaded once above the loop; when it is None the reason was
        # already reported there, loudly.
        if not args.no_annotation and _iso_genes_df is not None:
            try:
                from annotation import annotate_ld_blocks

                m_ld_annotated = annotate_ld_blocks(m_blocks_iv, _iso_genes_df)
            except Exception as e:
                log.warning("  Annotation failed for %s: %s", model_name, e)

        # Consolidate LD blocks + annotation + haplotype into one table
        from annotation import consolidate_ld_block_table
        try:
            m_consolidated = consolidate_ld_block_table(m_blocks_iv, m_hap_gwas, m_ld_annotated)
            if m_consolidated is not None and not m_consolidated.empty:
                extra_csvs[f"LD_blocks_annotated_{model_name}.csv"] = m_consolidated
        except Exception as e:
            log.warning("  LD block consolidation failed for %s: %s", model_name, e)

        # ── LD-quality triage (T-36): Layer 1 (predictive) + Layer 2 (mlg_*) ->
        #    router -> a NEW supplementary LD_triage_{model}.csv. Additive + gated
        #    by --no-triage (existing outputs byte-identical when off). Never a
        #    filter; changes no p/F/eta2/boundary. N1 safe: consolidate drops mlg_*. ──
        _stagelog.mark(f"S30_triage:{model_name}")
        if not args.no_triage and m_ld_blocks is not None and not m_ld_blocks.empty:
            try:
                import numpy as _np
                from annotation import canon_chr as _cc
                from gwas.ld import compute_block_ld_quality as _cblq, maf_from_matrix as _maf
                from gwas.triage import (TriageThresholds as _TT, triage_blocks as _tb,
                                         add_eta2_comparability as _eta)
                _ldq = _cblq(m_ld_blocks, chroms, positions, sid, _geno_float, model_df,
                             geno_dosage_raw=geno_dosage_raw, r2_coherent=args.ld_r2)
                _m = _ldq.copy()
                _m["_c"] = _m["Chr"].astype(str).map(_cc)
                _m["_s"] = _m["Start (bp)"].astype(int); _m["_e"] = _m["End (bp)"].astype(int)
                if m_hap_gwas is not None and not m_hap_gwas.empty:
                    _h = m_hap_gwas.rename(columns={"Start": "Start (bp)", "End": "End (bp)"}).copy()
                    _h["_c"] = _h["Chr"].astype(str).map(_cc)
                    _h["_s"] = _h["Start (bp)"].astype(int); _h["_e"] = _h["End (bp)"].astype(int)
                    _l2 = ["_c", "_s", "_e"] + [c for c in _h.columns if c.startswith("mlg_") or c in (
                        "eta2", "df1", "df2", "F_perm", "F_param", "n_samples_tested",
                        "n_samples_block", "n_tested_haplotypes")]
                    _m = _m.merge(_h[_l2], on=["_c", "_s", "_e"], how="left")
                _m = _m.drop(columns=["_c", "_s", "_e"])
                _m = _eta(_m)
                # per-row lead usability (n_lead_classes_ge, lead_maf) from raw dosage
                _sida = _np.asarray(sid).astype(str)
                _G = _as_columns(geno_dosage_raw, float) if geno_dosage_raw is not None else None
                _nc, _mf = [], []
                for _lead in _m["ldq_lead_snp"].astype(str):
                    _c1, _m1 = _np.nan, _np.nan
                    if _G is not None and _lead:
                        _ix = _np.where(_sida == _lead)[0]
                        if len(_ix):
                            _col = _G[:, int(_ix[0])]
                            _gg = _np.rint(_col[_np.isfinite(_col)])
                            _c1 = int(sum(int((_gg == _k).sum()) >= args.hap_min_group_size for _k in (0, 1, 2)))
                            _m1 = float(_maf(_G[:, [int(_ix[0])]], "dosage012")[0])
                    _nc.append(_c1); _mf.append(_m1)
                _m["n_lead_classes_ge"] = _nc
                _m["lead_maf"] = _mf
                _thr = _TT(
                    r2_coherent=(args.triage_r2_coherent if args.triage_r2_coherent is not None else args.ld_r2),
                    lead_r2_frac=args.triage_lead_r2_frac, min_group_n=args.hap_min_group_size, enabled=True)
                extra_csvs[f"LD_triage_{model_name}.csv"] = _tb(_m, _thr)
                log.info("  Triage: %d blocks -> LD_triage_%s.csv", len(_m), model_name)
            except Exception as e:
                log.warning("  LD-quality triage failed for %s: %s", model_name, e)

        # Sample-to-haplotype assignments of the primary model's tested blocks: one row
        # per sample per block (the same table as the Post-GWAS downloads)
        _stagelog.mark(f"S31_assignments:{model_name}")
        if model_name == _primary_model and m_hap_gwas is not None and not m_hap_gwas.empty:
            try:
                from gwas.haplotype import haplotype_assignments
                _assign = haplotype_assignments(m_hap_gwas, m_hap_tables, geno_df,
                                                pheno_clean, args.trait)
                _report_assignments = _assign
                if not _assign.empty:
                    extra_csvs[f"Haplotype_assignments_{args.trait}.csv"] = _assign
            except Exception as e:
                log.warning("  Haplotype assignments failed for %s: %s", model_name, e)
        # ... and of every other model's tested blocks, in a file named for the model
        # (as LD_triage_<model>.csv); the primary model's file above stays as it was
        elif m_hap_gwas is not None and not m_hap_gwas.empty:
            try:
                from gwas.haplotype import haplotype_assignments
                _assign_m = haplotype_assignments(m_hap_gwas, m_hap_tables, geno_df,
                                                  pheno_clean, args.trait)
                if not _assign_m.empty:
                    extra_csvs[f"Haplotype_assignments_{args.trait}_{model_name}.csv"] = _assign_m
            except Exception as e:
                log.warning("  Haplotype assignments failed for %s: %s", model_name, e)

        # Capture per-model annotation/haplotype frames for the HTML report (T-80)
        _mp = {}
        if m_ld_annotated is not None and not getattr(m_ld_annotated, "empty", True):
            _mp["ld_blocks_annotated_df"] = m_ld_annotated
        if m_hap_gwas is not None and not getattr(m_hap_gwas, "empty", True):
            _mp["haplotype_gwas_df"] = m_hap_gwas
        if _mp:
            _per_model_post[model_name] = _mp
        if model_name == _primary_model:
            _report_ld_annotated = m_ld_annotated
            _report_hap_gwas = m_hap_gwas
            _report_ld_blocks = m_ld_blocks

    # ── LD heatmaps: one per block whose lead SNP is significant in a model ──
    # A block is its chromosome, start and end. Each is drawn once, in the order the
    # models found the blocks (MLM, MLMM, FarmCPU), with the lead SNP of the first model
    # that reports it with a significant lead marked. The title names every model whose
    # own block with these bounds has a significant lead, or in which the marked lead SNP
    # is significant; when that is MLM alone it is the title the heatmap always had, so
    # an MLM-only run's heatmaps are unchanged. Another model can report the same lead
    # inside a block with other bounds; that block is its own heatmap.
    _stagelog.mark("S32_heatmaps")
    if any(_hm["draw"] for _hm in _heatmap_blocks.values()):
        from gwas.plotting import lead_snp_position, plot_ld_block_heatmap
        _sig_sets = {m: set(d.loc[d[_hm_sig_col], "SNP"].astype(str))
                     for m, d in post_gwas_models if _hm_sig_col in d.columns}
        _n_drawn, _n_named = 0, {}
        for (_bchr, _bstart, _bend), _hm in _heatmap_blocks.items():
            if not _hm["draw"]:
                continue
            _models = [m for m, _ in post_gwas_models
                       if m in _hm["finders"] or _hm["lead"] in _sig_sets.get(m, ())]
            try:
                _fig_hm, _ = plot_ld_block_heatmap(
                    _geno_float, chroms_canon, positions, _bchr, _bstart, _bend,
                    lead_pos=lead_snp_position(_hm["lead"], sid, chroms_canon, positions, _bchr),
                    models=None if _models == ["MLM"] else _models,
                )
            except Exception as e:
                log.warning("  LD heatmap failed for Chr%s %d-%d (%s): %s",
                            _bchr, _bstart, _bend, ", ".join(_models), e)
                continue
            figures[f"LD_heatmap_Chr{_bchr}_{_bstart}_{_bend}.png"] = _fig_hm
            _n_drawn += 1
            for m in _models:
                _n_named[m] = _n_named.get(m, 0) + 1
        if _n_drawn:
            log.info("LD heatmaps: %d drawn%s", _n_drawn,
                     "" if list(_n_named) == ["MLM"] else
                     " (" + ", ".join(f"{m} {k}" for m, k in _n_named.items()) + " significant)")

    # ── Subsampling GWAS ───────────────────────────────────
    _stagelog.mark("S33_subsampling")
    boot_disc_df = None
    if args.subsampling:
        from gwas.subsampling import subsample_gwas_resampling, aggregate_subsampling_to_ld_blocks
        log.info("Subsampling GWAS: %d reps, %.0f%% subsample…",
                 args.boot_reps, args.boot_frac * 100)

        _boot_use_loco = getattr(args, "boot_use_loco", None)  # GUI-only setting
        if _boot_use_loco is None:
            _boot_use_loco = not getattr(args, "no_loco", False)
        try:
            boot_disc_df, boot_raw_pvals, boot_meta = subsample_gwas_resampling(
                geno_imputed=geno_imputed, y=y, sid=sid,
                chroms=chroms, chroms_num=chroms_num, positions=positions,
                iid=iid, Z_for_grm=Z_for_pca, pcs_full=pcs_full,
                n_pcs=n_pcs_mlm,
                n_reps=args.boot_reps, sample_frac=args.boot_frac,
                discovery_thresh=args.boot_thresh, seed=args.seed,
                n_jobs=args.boot_jobs,
                use_loco=_boot_use_loco,
                chroms_grm=chroms_grm if _boot_use_loco else None,
            )

            extra_csvs["Subsampling_SNP_stability.csv"] = boot_disc_df
            boot_meta_df = pd.DataFrame(boot_meta)
            extra_csvs["Subsampling_rep_metadata.csv"] = boot_meta_df

            n_ok = int((boot_meta_df["status"] == "ok").sum())
            n_gt50 = int((boot_disc_df["DiscoveryFreq"] > 0.5).sum())
            log.info("  %d/%d reps OK, %d SNPs with freq > 50%%", n_ok, args.boot_reps, n_gt50)

            # Aggregate to LD blocks if available
            if ld_blocks_mlm is not None and not ld_blocks_mlm.empty:
                try:
                    block_stab = aggregate_subsampling_to_ld_blocks(
                        boot_disc_df, ld_blocks_mlm, boot_raw_pvals,
                        sid, chroms, positions, discovery_thresh=args.boot_thresh,
                    )
                    if not block_stab.empty:
                        extra_csvs["Subsampling_block_stability.csv"] = block_stab
                        log.info("  Block-level stability: %d blocks", len(block_stab))
                except Exception as e:
                    log.warning("  Subsampling block aggregation failed: %s", e)

            # Subsampling discovery frequency histogram
            if not args.no_plots:
                try:
                    import matplotlib.pyplot as plt
                    _freq_vals = boot_disc_df["DiscoveryFreq"].values
                    _fig_bh, _ax_bh = plt.subplots(figsize=(6, 4))
                    _ax_bh.hist(
                        _freq_vals[_freq_vals > 0], bins=30,
                        edgecolor="white", linewidth=0.5,
                    )
                    _ax_bh.set_xlabel("Discovery frequency")
                    _ax_bh.set_ylabel("Number of SNPs")
                    _ax_bh.set_title(
                        f"Subsampling GWAS: SNP discovery frequency ({args.trait})"
                    )
                    _ax_bh.axvline(0.5, color="red", linestyle="--", label="50% threshold")
                    _ax_bh.legend()
                    plt.tight_layout()
                    figures[f"Subsampling_discovery_freq_{args.trait}.png"] = _fig_bh
                    log.info("  Subsampling histogram generated")
                except Exception as e:
                    log.warning("  Subsampling histogram failed: %s", e)
        except Exception as e:
            log.error("  Subsampling GWAS failed: %s", e)

    # ── Plots ────────────────────────────────────────────
    _stagelog.mark("S34_lambda_gc")
    lambda_gc = compute_lambda_gc(gwas_df["PValue"].values)
    log.info("Lambda GC: %.3f", lambda_gc)

    if not args.no_plots:
        _stagelog.mark("S35_plots")
        log.info("Generating plots…")
        df_plot = gwas_df.copy()
        df_plot, tick_pos, tick_lab = compute_cumulative_positions(df_plot)

        import matplotlib.pyplot as plt

        _thresh_labels = {"meff": f"M_eff (M={_meff_val:,})", "bonferroni": "Bonferroni", "fdr": "FDR"}
        # a numeric --sig-thresh is the custom rule: its own label ("p < 1.0e-05"), as in the
        # report and the GUI (the bare number used to reach plot code that needs a string)
        _man_label = (_sig_rule_obj.label if _sig_is_numeric
                      else _thresh_labels.get(sig_rule, sig_rule))
        _man_lod = -np.log10(primary_thresh) if primary_thresh is not None else None
        # GUI-only option: a second, informational threshold line (Bonferroni and
        # M_eff shown together). It changes no significance call.
        _sec = {}
        if getattr(args, "secondary_threshold", False):
            if sig_rule == "bonferroni":
                _sec = {"secondary_lod": -np.log10(meff_thresh),
                        "secondary_label": f"M_eff (M={int(_meff_val):,})"}
            elif sig_rule == "meff":
                _sec = {"secondary_lod": -np.log10(bonf_thresh_naive),
                        "secondary_label": "Bonferroni α=0.05"}
        # the enlarged SNPs are the rule's own column, as in the tables (p < threshold)
        fig_man = plot_manhattan_static(df_plot, _man_lod, _man_label,
                                        f"Manhattan (MLM): {args.trait}",
                                        sig_mask=_sig_rule_obj.significant_mask(df_plot), **_sec)
        plt.figure(fig_man.number)
        plt.xticks(tick_pos, tick_lab, fontsize=8)
        figures["Manhattan_MLM.png"] = fig_man

        fig_qq = plot_qq(gwas_df["PValue"].values, lambda_gc_used=lambda_gc)
        figures["QQ_MLM.png"] = fig_qq

        # Per-model Manhattan + QQ (MLMM, FarmCPU)
        for _m_name, _m_df in extra_model_dfs.items():
            try:
                _mdf_plot = _m_df.copy()
                _mdf_plot, _mtp, _mtl = compute_cumulative_positions(_mdf_plot)
                _lam_m = compute_lambda_gc(_m_df["PValue"].values)
                _fig_m = plot_manhattan_static(
                    _mdf_plot, _man_lod, _man_label,
                    f"Manhattan ({_m_name}): {args.trait}",
                    sig_mask=_sig_rule_obj.significant_mask(_mdf_plot), **_sec,
                )
                plt.figure(_fig_m.number)
                plt.xticks(_mtp, _mtl, fontsize=8)
                figures[f"Manhattan_{_m_name}.png"] = _fig_m

                _fig_qq_m = plot_qq(_m_df["PValue"].values, lambda_gc_used=_lam_m)
                figures[f"QQ_{_m_name}.png"] = _fig_qq_m
                log.info("  %s plots generated", _m_name)
            except Exception as e:
                log.warning("  %s plots failed: %s", _m_name, e)

        if pcs_full is not None and n_pcs_mlm >= 2:
            from gwas.plotting import plot_pca_scatter
            fig_pca = plot_pca_scatter(
                pcs_full[:, :n_pcs_mlm], y=y.ravel(),
                title=f"PCA: {args.trait}",
                eigenvalues=pca_eigenvalues,
            )
            figures["PCA_scatter.png"] = fig_pca

        if pca_eigenvalues is not None:
            # P4: scree plot (eigenvalues already computed; parity with the GUI which
            # renders plot_pca_scree). Gated on availability, not n_pcs_mlm >= 2, so it
            # emits even at k=0 (e.g. the tomato deflation-guard panel).
            from gwas.plotting import plot_pca_scree
            figures["PCA_scree.png"] = plot_pca_scree(
                pca_eigenvalues, n_pcs_used=n_pcs_mlm, title="PCA Scree Plot")

    # ── Per-SNP effect boxplots + plotting ledger (T-21/T-84–86) ──────────
    # Primary model only. The significant-SNP table stays complete — this only
    # bounds how many figures the report embeds (--snp-plots / --max-snp-plots),
    # and --collapse-r2 thins near-redundant SNPs for PLOTTING only. Emits
    # SNP_view_index; gated by --no-isolated-rescue (no _report_sig_table).
    # (Block_sample_retention is deferred to the LD-triage sub-batch, where the
    # haplotype test's frac_retained is hoisted into its output frame.)
    _stagelog.mark("S36_snp_plots")
    _snp_boxplots = []
    if (_do_rescue and _report_sig_table is not None and not _report_sig_table.empty
            and geno_dosage_raw is not None):
        try:
            from gwas.snpplots import (build_snp_view_index, render_snp_boxplot,
                                       select_snps_for_plotting)
            from gwas.snpview import collapse_snps_for_plotting, effect_flag
            _collapse = collapse_snps_for_plotting(
                _report_sig_table, geno_dosage_raw, sid, r2_threshold=args.collapse_r2)
            _rep_ids = set(_collapse.loc[
                _collapse["SNP"] == _collapse["Representative_SNP"], "SNP"].astype(str))
            _rep_table = _report_sig_table[_report_sig_table["SNP"].astype(str).isin(_rep_ids)]
            _plot_ids = select_snps_for_plotting(
                _rep_table, mode=args.snp_plots, max_plots=args.max_snp_plots)
            extra_csvs[f"SNP_view_index_{_primary_model}.csv"] = build_snp_view_index(
                _report_sig_table, _plot_ids, _collapse)
            if _plot_ids and not args.no_report:
                from gwas.haplotype import snp_haplotype_labels
                from gwas.snpplots import (HAP_NOTE_NO_BLOCK, HAP_NOTE_NO_GROUPS,
                                           phenotype_axis_label)
                _sid_arr = np.asarray(sid).astype(str)
                _geno_raw = _as_columns(geno_dosage_raw, float)
                _yv = np.asarray(y, dtype=float).ravel()
                _samples = np.asarray(geno_df.index, dtype=str)    # the rows of y
                # the y axis names the values plotted: the GWAS's y, normalised as --norm
                _ylab = phenotype_axis_label(args.trait, args.norm)
                for _pid in _plot_ids:
                    _mi = np.where(_sid_arr == str(_pid))[0]
                    _rowsel = _report_sig_table[_report_sig_table["SNP"].astype(str) == str(_pid)]
                    if len(_mi) == 0 or _rowsel.empty:
                        continue
                    _row = _rowsel.iloc[0]
                    _bm = _row.get("Beta_MLM"); _bo = _row.get("Beta_OLS"); _se = _row.get("SE_MLM")
                    _bm = float(_bm) if pd.notna(_bm) else None
                    _bo = float(_bo) if pd.notna(_bo) else None
                    _se = float(_se) if pd.notna(_se) else None
                    _flag = effect_flag(_bm if _bm is not None else np.nan,
                                        _bo if _bo is not None else np.nan)
                    _cap = f"{_pid} · {_row.get('Block_Status', '')} · {_flag}"
                    # dots coloured by the haplotype of the LD block that contains the SNP
                    # (the Haplotype_assignments table's labels), else a note
                    try:
                        _labels, _bkey, _why = snp_haplotype_labels(
                            _report_ld_blocks, _report_assignments,
                            chroms[int(_mi[0])], positions[int(_mi[0])], _samples)
                        _hk = ({"hap_labels": _labels} if _labels is not None else
                               {"hap_note": HAP_NOTE_NO_BLOCK if _why == "no_block"
                                else HAP_NOTE_NO_GROUPS})
                    except Exception as e:
                        log.warning("  Haplotype colours for %s failed: %s", _pid, e)
                        _hk = {}
                    _fig = render_snp_boxplot(str(_pid), _geno_raw[:, int(_mi[0])], _yv,
                                              beta_mlm=_bm, se_mlm=_se, beta_ols=_bo,
                                              y_label=_ylab, **_hk)
                    _snp_boxplots.append((str(_pid), _fig, _cap))
        except Exception as e:
            log.warning("Per-SNP boxplots/ledger failed: %s", e)

    # ── Build HTML report (in-memory) ────────────────────
    # ── Run provenance (built ALWAYS; feeds the HTML report AND run_metadata.json) ──
    _stagelog.mark("S37_metadata")
    _gb_meta = getattr(args, "genome_build", "SL3")
    # Record what the run ACTUALLY used, not what the flags asked for. The gene
    # frame either loaded above or it did not; the manifest follows that fact, so
    # it can no longer assert a gene model that produced no candidate-gene output.
    if _iso_genes_df is None:
        _gm_meta = None
    elif args.gene_model:
        _gm_meta = getattr(args, "gene_model_label", None) or str(args.gene_model)
    else:
        _gm_meta = _gm_res.gene_model.name

    # GUI: name the uploaded gene model, not the temporary copy the run read.
    if getattr(args, "gene_model_label", None) and _gm_res.gene_model is not None:
        _gm_status = str(_gm_status).replace(str(_gm_res.gene_model), args.gene_model_label)

    # Addition B: block-detection provenance. Effective detection seed/top-N (P14) +
    # the find_ld_clusters_genomewide defaults for the params the call inherits.
    import inspect as _md_ins
    from gwas import ld as _md_ld
    _md_lddef = {k: p.default for k, p in
                 _md_ins.signature(_md_ld.find_ld_clusters_genomewide).parameters.items()}
    meta = {
        "VCF": str(vcf_path),
        "Phenotype": str(pheno_path),
        "Trait": args.trait,
        "Models": args.model,
        "Species": args.species,
        "Genome build": _gb_meta,
        "Gene model": _gm_meta,
        "Gene model status": _gm_status,
        "Covariate file (--covar)": str(args.covar) if getattr(args, "covar", None) else None,
        "Covariate columns": ", ".join(_cov_names) if getattr(args, "covar", None) else None,
        "Covariate samples dropped": _n_drop if getattr(args, "covar", None) else 0,
        "Covariate analysis N": int(geno_df.shape[0]) if getattr(args, "covar", None) else "N/A",
        "MAF threshold": args.maf,
        "MAC threshold": args.mac,
        "Missingness threshold": args.miss,
        "Info threshold": args.info_thresh,
        "Imputation method": (args.impute if args.impute == "mean"
                              else f"ldknni (k={args.impute_k}, l={args.impute_l})"),
        "Normalization": args.norm,
        "Significance": sig_rule,
        "PCs_MLM": n_pcs_mlm,
        "PCs_MLMM": n_pcs_mlmm if "mlmm" in args.model else "N/A",
        "PCs_FarmCPU": n_pcs_fc if "farmcpu" in args.model else "N/A",
        "Samples": int(geno_df.shape[0]),
        "SNPs (post-QC)": int(_n_snps(geno_df, sid)),
        "SNPs (raw)": int(n_raw),
        "Isolated SNPs (no LD block)": (
            "; ".join(
                f"{m}: {c['n_uncovered']} ({c['n_seeding_path']} seeding-threshold, "
                f"{c['n_block_path']} block-formation) -> {c['n_intervals']} interval(s)"
                for m, c in _iso_counts.items()
            ) if _iso_counts else ("off" if not _do_rescue else "0")
        ),
        "Lambda GC": round(lambda_gc, 4),
        "Kinship model": kinship_model,
        "LD decay (kb)": round(ld_decay_kb, 1) if ld_decay_kb else "N/A",
        "LD decay grid-limited (n_chr)": _ld_decay_n_censored,
        "LD flank (kb)": ld_flank_kb if ld_flank_kb else "N/A",
        "LD_r2 (--ld-r2)": args.ld_r2,
        "LD_seed_p (effective)": _ld_seed_thresh,
        "LD_top_n (effective)": _ld_seed_top_n,
        "LD_merge_r2 (--ld-merge-r2)": args.ld_merge_r2,
        "LD_merge_iou": _md_lddef.get("merge_iou"),
        "LD_adj_r2_min": _md_lddef.get("adj_r2_min"),
        "LD_gap_factor": _md_lddef.get("gap_factor"),
        "LD_min_snps": 2,
        "Hap_min_group_size (--hap-min-group-size)": args.hap_min_group_size,
        "Hap_n_perm": args.hap_perms,
        "LD blocks (MLM)": len(ld_blocks_mlm) if ld_blocks_mlm is not None else "N/A",
        "Subsampling reps": args.boot_reps if args.subsampling else "N/A",
        "PCs (fixed)": int(n_pcs),
        "PC diagnostics": (
            f"trace/m={_pc_diag['meta']['trace_over_m']}; see PC_diagnostics_spectrum.csv"
            if _pc_diag is not None else "n/a"),
        # the machine facts that decide the last bits of the numbers (gwas/provenance.py):
        # BLAS library and its threads, the cores, --boot-jobs
        **_thread_provenance(args),
    }

    # GUI-only provenance (absent in the CLI, whose metadata stays as it was): the run
    # label. "Normalization" above is this trait's own slug in the GUI too (norm_by_trait).
    # the descriptions file, named only when one was given (--gene-desc, or a One-Click
    # upload by its file name), after the gene model's status
    if getattr(args, "gene_desc", None) and _iso_genes_df is not None:
        _gd_meta = getattr(args, "gene_desc_label", None) or str(args.gene_desc)
        meta = {k: v for item in meta.items()
                for k, v in ([item, ("Gene descriptions", _gd_meta)]
                             if item[0] == "Gene model status" else [item])}
    if getattr(args, "run_label", None):
        meta["Pipeline"] = args.run_label
    # MLMM's cofactor window (the LD window), after the LD flank, when MLMM ran
    if "MLMM" in extra_model_dfs:
        meta = {k: v for item in meta.items()
                for k, v in ([item, ("MLMM cofactor window (kb)", _mlmm_window_kb)]
                             if item[0] == "LD flank (kb)" else [item])}

    _stagelog.mark("S38_report")
    report_html = None
    if not args.no_report:
        from gwas.reports import generate_gwas_report

        report_html = generate_gwas_report(
            trait_col=args.trait,
            qc_snp=qc_snp,
            gwas_df=gwas_df,
            figures=figures if not args.no_plots else None,
            metadata=meta,
            mlmm_df=cofactor_tables.get("MLMM"),
            farmcpu_df=cofactor_tables.get("FarmCPU"),
            ld_blocks_df=ld_blocks_mlm,
            ld_blocks_annotated_df=_report_ld_annotated,
            haplotype_gwas_df=_report_hap_gwas,
            per_model_post_gwas=_per_model_post or None,
            significant_snps_df=_report_sig_table,
            unblocked_snps_df=_report_unblocked,
            isolated_intervals_df=_report_isolated,
            snp_boxplots=_snp_boxplots or None,
            sig_label=_sig_rule_obj.label,
            # the primary model's significance column for the analysis rule, also when
            # the significant-SNP table is empty or was not built (--no-isolated-rescue);
            # the report's own fallback counts FDR<0.05 whatever the rule
            n_significant_override=int(_sig_rule_obj.significant_mask(gwas_df).sum()),
            # with several models, one count per model, each from its own column
            n_significant_by_model=(
                {m: int(_sig_rule_obj.significant_mask(d).sum()) for m, d in post_gwas_models}
                if len(post_gwas_models) > 1 else None),
            pc_selection_df=(_pc_diag["spectrum"] if _pc_diag is not None else None),
            lambda_gc=lambda_gc,
            n_samples=int(geno_df.shape[0]),
            n_snps=int(_n_snps(geno_df, sid)),
            info_field=info_field,
        )

    # ── ZIP archive (single output file) ──────────────────
    # The caller writes it (the CLI to --output; the GUI keeps it in memory).
    if getattr(args, "output", None):
        log.info("Saving results to %s", Path(args.output))
    _stagelog.mark("S39_zip_and_figures")
    from gwas.plotting import _build_gwas_results_zip

    zip_name, zip_buf = _build_gwas_results_zip(
        trait_col=args.trait,
        gwas_df=gwas_df,
        figures_dict=figures if not args.no_plots else None,
        pheno_label=pheno_path.stem,
        extra_model_dfs=extra_model_dfs or None,
        extra_tables=extra_csvs or None,
        report_html=report_html,
        # where the genotype table was kept: run_metadata.json only (the CLI's plan)
        metadata={**meta, **(getattr(args, "_geno_store_meta", None) or {})},
    )
    zip_bytes = zip_buf.getvalue()
    del zip_buf

    # Free figure memory: every figure this trait opened, the per-SNP boxplots and
    # (under --no-plots) the QC figure included, so a multi-trait run does not
    # accumulate them. The ZIP and report are already built.
    import matplotlib.pyplot as plt
    for fig in [*figures.values(), *(f for _, f, _ in _snp_boxplots)]:
        if hasattr(fig, "savefig"):
            plt.close(fig)

    _stagelog.mark("S40_result")
    tables = {f"GWAS_{args.trait}.csv": gwas_df}
    tables.update({f"GWAS_{m}_{args.trait}.csv": d for m, d in (extra_model_dfs or {}).items()})
    tables.update(extra_csvs)
    _sig_col_final = ("Significant_Custom" if _sig_is_numeric
                      else "Significant_Meff" if sig_rule == "meff"
                      else ("Significant_FDR" if sig_rule == "fdr" else "Significant_Bonf"))
    return TraitRunResult(
        trait=args.trait,
        zip_name=zip_name,
        zip_bytes=zip_bytes,
        tables=tables,
        figures=_zip_figures(zip_bytes),
        summary={
            "lambda_gc": float(lambda_gc),
            "n_significant": (int(gwas_df[_sig_col_final].sum())
                              if _sig_col_final in gwas_df.columns else None),
            "sig_label": _sig_rule_obj.label,
            "n_ld_blocks": len(ld_blocks_mlm) if ld_blocks_mlm is not None else None,
            "n_samples": int(geno_df.shape[0]),
            "n_snps": int(_n_snps(geno_df, sid)),
            "meff": int(_meff_val),
            # the samples dropped for a missing covariate value, for the GUI's warning
            "covariate_dropped": list(_cov_dropped),
            "covariate_n_before": _cov_n_before,
        },
        context={
            "gwas_df": gwas_df,
            "geno_imputed": geno_imputed,
            "y": y,
            "meff_val": _meff_val,
            "isolated_counts": _iso_counts,
        },
    )
