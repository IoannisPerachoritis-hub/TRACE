"""Memory and disk SessionTables of one trait, for the Post-GWAS tests (gwas.guipost).

memory  the GWAS page's memory path: gwas.qc._gwas_pipeline_impl without Streamlit's caches,
        with the page's default settings unless others are given. The SessionTable holds the
        page's arrays, as the GWAS page leaves them (gwas.guitable.SessionTable).
disk    gwas.genostore.ingest_vcf into a new folder, then gwas.storepipe.TraitStore's stages
        with the page's arguments, in the order gwas/pipeline.py:_run_trait calls them (IDs,
        phenotype QC, SNP QC, matrices). The SessionTable holds the view and raw LazyColumns,
        the trait's rows and the fill.

Both give the same identity (gwas.guitable.table_identity with the page's settings): it names
the table, not where it is kept.
"""
import contextlib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_V1 = ROOT / "tests" / "data" / "example_v1"       # the first example, frozen
EXAMPLE_V2 = ROOT / "examples"                             # the bundled example

# the GWAS page's defaults (QC preset "Standard GWAS", mean imputation, no transformation)
PAGE_DEFAULTS = dict(maf_thresh=0.05, miss_thresh=0.10, ind_miss_thresh=0.20, mac_thresh=5,
                     info_thresh=0.0, drop_alt=True, norm_option="none", canonical=None,
                     impute_method="mean", impute_k=5, impute_l=20)
_ACCESSION_COLS = {"accessions", "accession", "sample", "samples", "iid", "genotype", "id",
                   "sample_id", "sampleid"}


def read_pheno(path):
    """The phenotype table as the GWAS page prepares it (accession column as the index)."""
    pheno = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    pheno.columns = pheno.columns.astype(str).str.strip()
    acc = next((c for c in pheno.columns if c.lower().strip() in _ACCESSION_COLS), pheno.columns[0])
    pheno[acc] = pheno[acc].astype(str).str.strip()
    pheno = pheno.set_index(acc)
    pheno.index = pheno.index.astype(str).str.strip()
    pheno.columns = pheno.columns.str.strip()
    return pheno


def identity(vcf, pheno, trait, settings):
    """The page's table identity for these settings (pages/GWAS_analysis.py, geno_table)."""
    from gwas.guitable import file_identity, table_identity
    from gwas.utils import hash_df

    s = dict(PAGE_DEFAULTS, **settings)
    return table_identity(
        file_identity(vcf), pheno=hash_df(pheno), trait=trait, norm=s["norm_option"],
        maf=s["maf_thresh"], miss=s["miss_thresh"], ind_miss=s["ind_miss_thresh"],
        mac=s["mac_thresh"], info=s["info_thresh"], drop_alt=int(s["drop_alt"]),
        impute=s["impute_method"], impute_k=s["impute_k"], impute_l=s["impute_l"], covar=None)


@dataclass
class TraitTables:
    """One mode's SessionTable and what the page holds beside it."""

    table: object            # gwas.guitable.SessionTable
    geno_df: pd.DataFrame    # the genotype frame (memory) or the store frame (disk)
    pheno: pd.DataFrame      # the trait's analysis samples (pheno_used_for_hap)
    extra: dict = field(default_factory=dict)


@contextlib.contextmanager
def _uncached():
    """gwas.qc's Streamlit caches replaced by the functions they wrap, for the call."""
    from gwas import qc

    saved = (qc.load_vcf_cached, qc._impute_cached)
    qc.load_vcf_cached = getattr(saved[0], "__wrapped__", saved[0])
    qc._impute_cached = getattr(saved[1], "__wrapped__", saved[1])
    try:
        yield qc
    finally:
        qc.load_vcf_cached, qc._impute_cached = saved


def memory_tables(vcf, pheno, trait, **settings):
    """The memory SessionTable (the GWAS page's arrays) and the page's other objects."""
    from gwas.guitable import SessionTable, file_identity

    s = dict(PAGE_DEFAULTS, **settings)
    vcf = Path(vcf)
    with _uncached() as qc:
        res = qc._gwas_pipeline_impl(
            vcf.read_bytes(), pheno.copy(), trait, s["maf_thresh"], s["miss_thresh"],
            s["norm_option"], s["drop_alt"], s["ind_miss_thresh"], s["mac_thresh"],
            info_thresh=s["info_thresh"], canonical=s["canonical"],
            impute_method=s["impute_method"], impute_k=s["impute_k"], impute_l=s["impute_l"],
            covar_df=None, vcf_hash=file_identity(vcf))
    table = SessionTable(
        identity=identity(vcf, pheno, trait, settings), file_identity=file_identity(vcf),
        mode="memory", view=res["geno_imputed"], raw=res["geno_dosage_raw"], sid=res["sid"],
        chroms=res["chroms"], positions=res["positions"], iid=res["iid"])
    used = res["pheno"].copy()
    used.columns = used.columns.astype(str).str.strip()
    return TraitTables(table=table, geno_df=res["geno_df"], pheno=used, extra=res)


def disk_tables(vcf, pheno, trait, store_dir, **settings):
    """The disk SessionTable (the view and raw LazyColumns of a table built in ``store_dir``).
    Close it with close(tables)."""
    from gwas import genostore
    from gwas.guitable import SessionTable, file_identity
    from gwas.storepipe import TraitStore

    s = dict(PAGE_DEFAULTS, **settings)
    vcf = Path(vcf)
    parsed = genostore.ingest_vcf(vcf, Path(store_dir))
    try:
        ts = TraitStore(parsed)
        geno_df, ph, chroms, positions, sid, info_scores, _allele_map = ts.harmonize_ids(
            parsed.genotypes, parsed.samples, parsed.chroms, parsed.positions, parsed.vid,
            parsed.ref, parsed.alt, pheno.copy(), parsed.info_scores)
        geno_df, ph, y = ts.phenotype_qc(geno_df, ph, trait, s["norm_option"],
                                         s["ind_miss_thresh"])
        geno_df, chroms, _chroms_num, positions, sid, _n_raw, _qc_snp, _info = ts.snp_qc(
            geno_df, chroms, positions, sid, s["maf_thresh"], s["miss_thresh"], s["mac_thresh"],
            s["drop_alt"], info_scores, s["info_thresh"], canonical=s["canonical"])
        raw, _rate, imputed, iid = ts.build_geno_matrices(
            geno_df, method=s["impute_method"], impute_k=s["impute_k"], impute_l=s["impute_l"])
    except BaseException:
        genostore.remove_store(store_dir, parsed)
        raise
    table = SessionTable(
        identity=identity(vcf, pheno, trait, settings), file_identity=file_identity(vcf),
        mode="disk", parsed=parsed, view=imputed, raw=raw, rows=ts._rows(geno_df),
        fill=imputed.fill, sid=sid, chroms=chroms, positions=positions, iid=iid)
    used = ph.copy()
    used.columns = used.columns.astype(str).str.strip()
    return TraitTables(table=table, geno_df=geno_df, pheno=used,
                       extra={"y": y, "store_dir": Path(store_dir)})


def close(tables):
    """Close a disk table and delete its folder (memory tables: nothing to do)."""
    from gwas import genostore

    t = tables.table
    if t.mode == "disk" and t.parsed is not None:
        genostore.remove_store(tables.extra["store_dir"], t.parsed)
