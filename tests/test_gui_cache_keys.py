"""Every GUI cache key names everything that changes its result.

At ff56e5c the GWAS page keyed its caches by strings that named a few QC settings
(ARRKEY::GENO_IMPUTED::<vcf>|<pheno>|<maf>|<miss>|<ind_miss>|<mac>|<drop_alt>, KERNELKEY::K0::<vcf>|<pheno>,
ARRKEY::Y_VEC::<vcf>|<pheno>|<trait>|<norm>, ...) and by Streamlit's hash of array arguments, which reads
a 100,000-cell sample of an array of 500,000 cells or more. Now one identity per trait table
(gwas.guigwas.trait_identity) keys every cache that depends on the table, and the tables are
arguments Streamlit does not hash.

Four stale cases, each run as the page runs it (its preprocessing, then its cached computation with
the arguments the page passes, in one process so the caches are shared), and checked against the
same computation run fresh:

1. FarmCPU's key: LD-kNNi with k = 5, then k = 3. The tables differ; the samples and SNPs do not.
   At ff56e5c the second run returned the first table's p-values.
2. The sampled M_eff: two tables of 1,000 x 500 cells that differ in one cell outside Streamlit's
   sample (their Streamlit hashes are equal). At ff56e5c the second table was shown the first
   table's M_eff.
3. A change of imputation method (mean, then LD-kNNi): the LD decay and FarmCPU of the second run.
   At ff56e5c both returned the first table's results.
4. A change of INFO threshold (0, then 0.6, which removes SNPs): the table's key changes and every
   cached result is the new table's. At ff56e5c the table's key stayed the same.

The page's calls are in _Page. At ff56e5c the same scenarios ran with _Page replaced by the page's
code of that commit (the method and both runs: batch 13, track A notes and evidence).
"""
import hashlib
import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from gwas import guigwas, kinship, models, plotting, qc
from gwas.utils import PhenoData, hash_bytes, hash_df

N_PCS = 0                       # the page's default PC count
SETTINGS = dict(trait="T", norm="none", maf=0.05, miss=0.2, ind_miss=0.25, mac=5, info=0.0,
                drop_alt=True, impute="mean", impute_k=5, impute_l=20, covar=None)


# ------------------------------------------------------------------------------- the data
def _panel(n=60, per=120, chroms=("1", "2", "3"), miss=0.06, seed=11):
    """A VCF with LD along each chromosome, about 6% missing calls and an imputation-quality field
    (DR2), and a trait with one causal SNP."""
    rng = np.random.default_rng(seed)
    samples = [f"S{i:03d}" for i in range(n)]
    head = ["##fileformat=VCFv4.2",
            '##INFO=<ID=DR2,Number=1,Type=Float,Description="Dosage R-squared">',
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples)]
    rows, causal = [], None
    codes = np.array(["0/0", "0/1", "1/1"])
    for ch in chroms:
        hap = rng.binomial(1, 0.5, size=(2, n))
        for j in range(per):
            hap = np.where(rng.random((2, n)) < 0.08, 1 - hap, hap)
            g = hap.sum(axis=0)
            if ch == "1" and j == 40:
                causal = g.copy()
            gt = codes[g].astype(object)
            gt[rng.random(n) < miss] = "./."
            rows.append(f"{ch}\t{(j + 1) * 5000}\t{ch}_{j}\tA\tT\t.\tPASS\t"
                        f"DR2={rng.uniform(0.3, 1.0):.3f}\tGT\t" + "\t".join(gt))
    vcf = ("\n".join(head + rows) + "\n").encode()
    pheno = pd.DataFrame({"T": 1.2 * causal + rng.normal(0, 1, n)},
                         index=pd.Index(samples, name="Accession"))
    return vcf, pheno


_DATA = {}


def _prep(**changes):
    """The page's preprocessing for SETTINGS with ``changes``: (settings, results)."""
    if "panel" not in _DATA:
        _DATA["panel"] = _panel()
    vcf, pheno = _DATA["panel"]
    s = dict(SETTINGS, **changes, vcf_hash=hash_bytes(vcf), pheno_hash=hash_df(pheno))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = qc._gwas_pipeline_impl(
            vcf, pheno.copy(), s["trait"], s["maf"], s["miss"], s["norm"], s["drop_alt"],
            s["ind_miss"], s["mac"], info_thresh=s["info"], impute_method=s["impute"],
            impute_k=s["impute_k"], impute_l=s["impute_l"], vcf_hash=s["vcf_hash"])
    return s, r


def _two_tables():
    """Two 1,000 x 500 float32 tables (F-ordered, as the page's) that differ in one cell outside
    the 100,000 cells Streamlit's hash reads, so their Streamlit hashes are equal."""
    n, m = 1000, 500
    rng = np.random.default_rng(1)
    a = np.asfortranarray(rng.binomial(2, rng.uniform(0.2, 0.8, m), size=(n, m)).astype(np.float32))
    a[:, 8] = a[:, 7]
    sampled = set(np.random.RandomState(0).randint(0, n * m, size=100_000).tolist())  # hashing.py
    i0 = next(i for i in range(n) if i * m + 7 not in sampled and a[i, 7] != 2)
    b = a.copy(order="F")
    b[i0, 7] += 1
    return a, b


def _streamlit_hash(x):
    from streamlit.runtime.caching.cache_type import CacheType
    from streamlit.runtime.caching.hashing import update_hash

    h = hashlib.new("md5")
    update_hash(x, hasher=h, cache_type=CacheType.DATA)
    return h.hexdigest()


# ------------------------------------------------------------------------- the page's calls
class _Page:
    """The GWAS page's calls in this tree (pages/GWAS_analysis.py): the table's identity, then the
    identity-keyed caches with the tables as underscore arguments."""

    @staticmethod
    def clear():
        for f in (models.run_farmcpu_cached, models.run_gwas_cached, guigwas.ld_decay,
                  kinship.compute_pcs_full_cached, kinship.build_loco_kernels_cached):
            f.clear()

    @staticmethod
    def table_key(s, r):
        return guigwas.trait_identity(
            s["vcf_hash"], pheno=s["pheno_hash"], trait=s["trait"], norm=s["norm"], maf=s["maf"],
            miss=s["miss"], ind_miss=s["ind_miss"], mac=s["mac"], info=s["info"],
            drop_alt=int(s["drop_alt"]), impute=s["impute"], impute_k=s["impute_k"],
            impute_l=s["impute_l"], covar=s["covar"], canonical=None,
            samples=guigwas.sample_set_hash(r["geno_df"].index))

    @staticmethod
    def _inputs(state, s, r):
        key = _Page.table_key(s, r)
        pcs_full, _ = kinship.compute_pcs_full_cached(key, r["Z_for_pca"], max_pcs=20)
        K0, K_by_chr, _ = kinship.build_loco_kernels_cached(
            key, _iid=r["iid"], _Z_grm=r["Z_for_pca"],
            _chroms_grm=np.asarray(r["chroms_grm"], dtype=str), _K_base=r["K"])
        return key, pcs_full, K0, K_by_chr, PhenoData(iid=r["iid"], val=r["y"])

    @staticmethod
    def mlm(state, s, r):
        key, pcs_full, K0, K_by_chr, reader = _Page._inputs(state, s, r)
        return models.run_gwas_cached(
            key, N_PCS, use_loco=True, trait_name=s["trait"], _geno_imputed=r["geno_imputed"],
            _y=r["y"], _pcs_full=pcs_full, _sid=r["sid"], _positions=r["positions"],
            _chroms=np.asarray(r["chroms"], dtype=str), _chroms_num=r["chroms_num"],
            _iid=r["iid"], _K0=K0, _K_by_chr=K_by_chr, _pheno_reader=reader)

    @staticmethod
    def farmcpu(state, s, r):
        key, _pcs, K0, _kc, reader = _Page._inputs(state, s, r)
        return models.run_farmcpu_cached(
            key, N_PCS, max_iterations=10, final_scan="ols", use_loco=True,
            _geno_imputed=r["geno_imputed"], _sid=r["sid"],
            _chroms=np.asarray(r["chroms"], dtype=str), _chroms_num=r["chroms_num"],
            _positions=r["positions"], _iid=r["iid"], _K0=K0, _pheno_reader=reader,
            _covar_reader=None)

    @staticmethod
    def ld_decay(state, s, r):
        return guigwas.ld_decay(_Page.table_key(s, r), r["geno_imputed"],
                                np.asarray(r["chroms"], dtype=str), r["positions"])

    @staticmethod
    def meff(state, key, table):
        return guigwas.meff(state, key, table)

    @staticmethod
    def pcs(state, s, r):
        return _Page._inputs(state, s, r)[1]


# ----------------------------------------------------------------------- fresh references
def _fresh_kernels(r):
    return kinship._build_loco_kernels_impl(r["iid"], r["Z_for_pca"],
                                            np.asarray(r["chroms_grm"], dtype=str), r["K"])


def _fresh_farmcpu(r):
    K0, _kc, _d = _fresh_kernels(r)
    return models.run_farmcpu(
        np.asarray(r["geno_imputed"], dtype=np.float32, order="C"), np.asarray(r["sid"]).astype(str),
        np.asarray(r["chroms"]).astype(str), np.asarray(r["chroms_num"]).astype(int),
        np.asarray(r["positions"]).astype(int), np.asarray(r["iid"]).astype(str),
        PhenoData(iid=r["iid"], val=r["y"]), K0, None, max_iterations=10, final_scan="ols",
        verbose=False, use_loco=True)


def _fresh_mlm(s, r):
    K0, K_by_chr, _d = _fresh_kernels(r)
    pcs_full, _ = kinship._compute_pcs_full_impl(r["Z_for_pca"], max_pcs=20)
    return models._run_gwas_impl(r["geno_imputed"], r["y"], pcs_full, N_PCS, r["sid"], r["positions"],
                                 np.asarray(r["chroms"], dtype=str), r["chroms_num"], r["iid"], K0,
                                 K_by_chr, PhenoData(iid=r["iid"], val=r["y"]), s["trait"])


def _fresh_ld_decay(r):
    from annotation import compute_ld_decay_by_chromosome

    return compute_ld_decay_by_chromosome(np.asarray(r["chroms"], dtype=str),
                                          np.asarray(r["positions"], dtype=int), r["geno_imputed"],
                                          max_snps_per_chr=1500, max_dist_kb=5000.0, n_bins=40)


@pytest.fixture(autouse=True)
def _clean_caches():
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    _Page.clear()
    yield
    _Page.clear()


# ---------------------------------------------------------------------------------- the tests
def test_farmcpu_key_does_not_serve_another_tables_p_values():
    state = {}
    sa, ra = _prep(impute="ldknni", impute_k=5)
    sb, rb = _prep(impute="ldknni", impute_k=3)
    assert ra["geno_imputed"].shape == rb["geno_imputed"].shape          # same samples and SNPs
    fresh_a = _fresh_farmcpu(ra)[0]["PValue"].to_numpy()
    fresh_b = _fresh_farmcpu(rb)[0]["PValue"].to_numpy()
    assert not np.array_equal(fresh_a, fresh_b)                          # the tables' p-values differ
    got_a = _Page.farmcpu(state, sa, ra)[0]["PValue"].to_numpy()
    got_b = _Page.farmcpu(state, sb, rb)[0]["PValue"].to_numpy()
    np.testing.assert_array_equal(got_a, fresh_a)
    np.testing.assert_array_equal(got_b, fresh_b)                        # stale at ff56e5c: got_a


def test_meff_of_a_table_changed_outside_streamlits_sample():
    state = {}
    a, b = _two_tables()
    assert _streamlit_hash(a) == _streamlit_hash(b)                      # Streamlit cannot tell them apart
    fresh_a, fresh_b = plotting._compute_meff_li_ji_impl(a)[0], plotting._compute_meff_li_ji_impl(b)[0]
    assert fresh_a != fresh_b
    # the two tables come from two VCFs: each has its own identity
    keys = [guigwas.trait_identity(hash_bytes(t.tobytes()), pheno="p", trait="T", norm="none",
                                   maf=0.0, miss=1.0, ind_miss=1.0, mac=0, info=0.0, drop_alt=1,
                                   impute="mean", impute_k=5, impute_l=20, covar=None,
                                   canonical=None, samples="s") for t in (a, b)]
    assert _Page.meff(state, keys[0], a)[0] == fresh_a
    assert _Page.meff(state, keys[1], b)[0] == fresh_b                   # stale at ff56e5c: fresh_a


def test_a_change_of_imputation_method_gives_the_new_tables_results():
    state = {}
    sa, ra = _prep(impute="mean")
    sb, rb = _prep(impute="ldknni")
    assert ra["geno_imputed"].shape == rb["geno_imputed"].shape
    assert not np.array_equal(ra["geno_imputed"], rb["geno_imputed"])
    dec_a, dec_b = _fresh_ld_decay(ra), _fresh_ld_decay(rb)
    assert not dec_a[0].equals(dec_b[0])                                 # the decay curves differ
    fc_a = _fresh_farmcpu(ra)[0]["PValue"].to_numpy()
    fc_b = _fresh_farmcpu(rb)[0]["PValue"].to_numpy()
    assert not np.array_equal(fc_a, fc_b)
    got_a, got_b = _Page.ld_decay(state, sa, ra), _Page.ld_decay(state, sb, rb)
    pd.testing.assert_frame_equal(got_a[0], dec_a[0])
    pd.testing.assert_frame_equal(got_b[0], dec_b[0])                    # stale at ff56e5c: dec_a
    pd.testing.assert_frame_equal(got_b[1], dec_b[1])
    np.testing.assert_array_equal(_Page.farmcpu(state, sa, ra)[0]["PValue"].to_numpy(), fc_a)
    np.testing.assert_array_equal(_Page.farmcpu(state, sb, rb)[0]["PValue"].to_numpy(), fc_b)


def test_a_change_of_info_threshold_changes_the_key_and_gives_the_new_tables_results():
    state = {}
    sa, ra = _prep(info=0.0)
    sb, rb = _prep(info=0.6)
    assert rb["geno_imputed"].shape[1] < ra["geno_imputed"].shape[1]    # the threshold removes SNPs
    assert _Page.table_key(sa, ra) != _Page.table_key(sb, rb)           # the same at ff56e5c
    for s, r in ((sa, ra), (sb, rb)):
        mlm = _Page.mlm(state, s, r)
        pd.testing.assert_frame_equal(mlm, _fresh_mlm(s, r))
        np.testing.assert_array_equal(_Page.farmcpu(state, s, r)[0]["PValue"].to_numpy(),
                                      _fresh_farmcpu(r)[0]["PValue"].to_numpy())
        pd.testing.assert_frame_equal(_Page.ld_decay(state, s, r)[0], _fresh_ld_decay(r)[0])
        np.testing.assert_array_equal(_Page.pcs(state, s, r),
                                      kinship._compute_pcs_full_impl(r["Z_for_pca"], 20)[0])
