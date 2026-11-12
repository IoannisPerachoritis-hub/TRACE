"""The association models with the genotype table on disk (gwas.genostore.LazyColumns)
against the same models on today's in-memory table: bit-identical results.

MLM reads each chromosome's test SNPs through a pysnptools reader over the table on disk, the
per-copy SD and the OLS effects read today's blocks. Narrow operands get their own cases (a
chromosome of 1, 2 or 3 SNPs; an OLS chunk grid whose last chunk is 1, 2 or 3 SNPs wide),
because BLAS can take another code path for a narrow operand of another layout.
"""
import numpy as np
import pandas as pd
import pytest

from gwas.genostore import LazyColumns, write_store
from gwas.models import _run_gwas_impl, add_ols_effects_to_gwas, dosage_sd
from gwas.utils import CovarData, PhenoData

from tests.test_storepipe import _args, _cases, _memory, _store


def _mlm_inputs(out, kinship, n_pcs=3):
    from gwas.kinship import _build_loco_kernels_impl, _compute_pcs_full_impl

    s08, s09 = out["S08"], out["S09"]
    K, Z, chroms_grm, _pg, _model = kinship(s08["geno_df"], s09["imp"], s08["chroms"], s08["positions"])
    pcs, _eig = _compute_pcs_full_impl(Z, max_pcs=20)
    K0, K_by_chr, _ = _build_loco_kernels_impl(s09["iid"], Z, chroms_grm, K)
    y = out["S06"]["y"]
    return dict(y=y, pcs=pcs, n_pcs=min(n_pcs, pcs.shape[1]), sid=s08["sid"], positions=s08["positions"],
                chroms=s08["chroms"], chroms_num=s08["chroms_num"], iid=s09["iid"], K0=K0,
                K_by_chr=K_by_chr, pheno_reader=PhenoData(s09["iid"], y))


def _mlm(geno, i, trait):
    return _run_gwas_impl(geno, i["y"], i["pcs"], i["n_pcs"], i["sid"], i["positions"], i["chroms"],
                          i["chroms_num"], i["iid"], i["K0"], i["K_by_chr"], i["pheno_reader"], trait)


def _frames_identical(a, b):
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    for c in b.columns:
        if b[c].dtype.kind == "f":
            assert np.ascontiguousarray(a[c].to_numpy()).tobytes() == np.ascontiguousarray(b[c].to_numpy()).tobytes(), c


@pytest.mark.parametrize("case", ["example_v1", "example_v2", "edge"])
def test_mlm_and_ols_effects_equal_in_memory(tmp_path, case):
    from gwas.qc import _pipeline_build_kinship

    vcf, pheno, trait = _cases(tmp_path)[case]
    a = _args(trait=trait)
    mem = _memory(vcf, pheno, trait, a)
    sto, st, parsed = _store(vcf, pheno, trait, a, tmp_path / "s")
    try:
        im = _mlm_inputs(mem, _pipeline_build_kinship)
        is_ = _mlm_inputs(sto, st.build_kinship)
        gm = _mlm(mem["S09"]["imp"], im, trait)
        gs_ = _mlm(sto["S09"]["imp"], is_, trait)
        _frames_identical(gs_, gm)
        cov = CovarData(im["iid"], im["pcs"][:, :im["n_pcs"]])
        om = add_ols_effects_to_gwas(gm.copy(), mem["S09"]["imp"], im["y"], cov, im["sid"])
        os_ = add_ols_effects_to_gwas(gs_.copy(), sto["S09"]["imp"], is_["y"], cov, is_["sid"])
        _frames_identical(os_, om)
    finally:
        parsed.store.close()


def _narrow_chromosomes_vcf(tmp_path, n=40, seed=5):
    """chr1 with many SNPs, then chromosomes of 1, 2 and 3 SNPs."""
    rng = np.random.default_rng(seed)
    samples = [f"S{i:02d}" for i in range(n)]
    lines = ["##fileformat=VCFv4.2", '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">',
             "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples)]
    for ch, k in (("chr1", 120), ("chr2", 1), ("chr3", 2), ("chr4", 3)):
        for j in range(k):
            g = ["0/0", "0/1", "1/1"]
            p = rng.uniform(0.15, 0.5)
            gt = [g[int(rng.random() < p) + int(rng.random() < p)] for _ in range(n)]
            if rng.random() < 0.5:
                gt[int(rng.integers(n))] = "./."
            lines.append(f"{ch}\t{1000 + 5000 * j}\t.\tA\tT\t.\tPASS\t.\tGT\t" + "\t".join(gt))
    (tmp_path / "narrow.vcf").write_text("\n".join(lines) + "\n")
    y = rng.normal(size=n)
    pd.DataFrame({"T": y}, index=pd.Index(samples, name="id")).to_csv(tmp_path / "narrow_pheno.csv")
    return tmp_path / "narrow.vcf", tmp_path / "narrow_pheno.csv"


def test_mlm_on_chromosomes_of_one_two_three_snps(tmp_path):
    from gwas.qc import _pipeline_build_kinship

    vcf, pheno = _narrow_chromosomes_vcf(tmp_path)
    a = _args(trait="T", maf=0.01)
    mem = _memory(vcf, pheno, "T", a)
    sto, st, parsed = _store(vcf, pheno, "T", a, tmp_path / "s")
    try:
        assert set(np.unique(sto["S08"]["chroms"], return_counts=True)[1]) >= {1, 2, 3}
        im = _mlm_inputs(mem, _pipeline_build_kinship)
        is_ = _mlm_inputs(sto, st.build_kinship)
        _frames_identical(_mlm(sto["S09"]["imp"], is_, "T"), _mlm(mem["S09"]["imp"], im, "T"))
    finally:
        parsed.store.close()


@pytest.mark.parametrize("m", [2001, 2002, 2003, 4000])
def test_ols_effects_last_chunk_one_to_three_snps(tmp_path, m):
    """add_ols_effects_to_gwas scans chunks of 2,000 SNPs; the last chunk is 1, 2, 3 (or 2,000)
    SNPs wide. The table on disk must give the same bits as today's float64 table."""
    rng = np.random.default_rng(m)
    n = 30
    G = rng.integers(0, 3, size=(n, m)).astype(np.float32)
    G[rng.random((n, m)) < 0.03] = np.nan
    fill = np.nan_to_num(np.nanmean(G.astype(np.float64), axis=0)).astype(np.float32)
    imp = np.asfortranarray(np.where(np.isnan(G), fill, G).astype(np.float32))
    store = write_store(tmp_path / "s", G)
    try:
        L = LazyColumns(store, np.arange(m), fill=fill)
        sid = np.array([f"s{j}" for j in range(m)])
        df = pd.DataFrame({"SNP": sid, "PValue": rng.random(m)})
        iid = np.c_[np.arange(n).astype(str), np.arange(n).astype(str)]
        y = rng.normal(size=(n, 1))
        cov = CovarData(iid, rng.normal(size=(n, 2)))
        _frames_identical(add_ols_effects_to_gwas(df.copy(), L, y, cov, sid),
                          add_ols_effects_to_gwas(df.copy(), imp, y, cov, sid))
    finally:
        store.close()


@pytest.mark.parametrize("width", [1, 2, 3, 4097])
@pytest.mark.parametrize("rows", ["all", "subset"])
def test_dosage_sd_narrow_operands(tmp_path, width, rows):
    """The per-copy SD (mlm_effects_per_copy): F-ordered columns, or a row subset, which numpy
    makes C-ordered; 1, 2, 3 columns and a block grid with one column left over."""
    rng = np.random.default_rng(width)
    n = 25
    G = rng.integers(0, 3, size=(n, width)).astype(np.float32)
    G[rng.random((n, width)) < 0.05] = np.nan
    fill = np.nan_to_num(np.nanmean(G.astype(np.float64), axis=0)).astype(np.float32)
    imp = np.asfortranarray(np.where(np.isnan(G), fill, G).astype(np.float32))
    store = write_store(tmp_path / "s", G)
    try:
        L = LazyColumns(store, np.arange(width), fill=fill)
        if rows == "subset":
            r = np.ones(n, dtype=bool)
            r[[3, 7]] = False
            a, b = dosage_sd(L[r]), dosage_sd(imp[r])
        else:
            a, b = dosage_sd(L), dosage_sd(imp)
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
    finally:
        store.close()


# ------------------------------------------------------------- MLMM, FarmCPU, subsampling
@pytest.mark.parametrize("m", [1, 2, 3, 2001, 2002])
@pytest.mark.parametrize("piece", ["1", "2", "7", "2000"])
def test_streamed_standardisation_equals_todays(tmp_path, monkeypatch, m, piece):
    """LazyColumns.standardized: the per-SNP mean and SD from C-ordered blocks (never one column
    wide unless the table is) equal _standardize_genotypes_impl on the whole C-ordered table."""
    from gwas.models import _standardize_genotypes_impl

    monkeypatch.setenv("TRACE_STORE_PIECE", piece)
    rng = np.random.default_rng(m)
    n = 31
    G = rng.integers(0, 3, size=(n, m)).astype(np.float32)
    G[rng.random((n, m)) < 0.05] = np.nan
    if m > 3:
        G[:, 2] = 1.0                                   # a constant column: its SD is set to 1
    fill = np.nan_to_num(np.nanmean(G.astype(np.float64), axis=0)).astype(np.float32)
    imp = np.asfortranarray(np.where(np.isnan(G), fill, G).astype(np.float32))
    store = write_store(tmp_path / "s", G)
    try:
        Z_disk = _standardize_genotypes_impl(LazyColumns(store, np.arange(m), fill=fill))
        Z_today = _standardize_genotypes_impl(imp)
        got = np.asarray(Z_disk)
        assert got.dtype == Z_today.dtype and got.flags["C_CONTIGUOUS"] == Z_today.flags["C_CONTIGUOUS"]
        assert got.tobytes() == Z_today.tobytes()
        if m > 3:
            for sel in ([0], [5, 1], list(range(m - 3, m))):
                a, b = Z_disk[:, sel], Z_today[:, sel]
                assert a.tobytes() == np.ascontiguousarray(b).tobytes()
                assert a.flags["F_CONTIGUOUS"] == b.flags["F_CONTIGUOUS"]
    finally:
        store.close()


def _models_case(tmp_path, m=2001, n=60, seed=1106):
    """A panel whose post-QC SNP count leaves a one-column last chunk in the OLS scans
    (m mod 2,000 = 1), the last SNP being the QTL (the scaling audit's k2001_last)."""
    rng = np.random.default_rng(seed)
    samples = [f"K{i:03d}" for i in range(n)]
    lines = ["##fileformat=VCFv4.2", '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">',
             "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples)]
    G = np.zeros((n, m), dtype=int)
    for j in range(m):
        p = rng.uniform(0.2, 0.5)
        G[:, j] = (rng.random(n) < p).astype(int) + (rng.random(n) < p).astype(int)
    per = m // 4
    for j in range(m):
        ch = min(j // per, 3) + 1
        gt = "\t".join(["0/0", "0/1", "1/1"][g] for g in G[:, j])
        lines.append(f"chr{ch}\t{10_000 + 10_000 * (j - (ch - 1) * per)}\t.\tA\tT\t.\tPASS\t.\tGT\t{gt}")
    (tmp_path / "m.vcf").write_text("\n".join(lines) + "\n")
    y = 1.5 * G[:, -1] + rng.normal(0, 0.7, size=n)
    pd.DataFrame({"T": y}, index=pd.Index(samples, name="id")).to_csv(tmp_path / "m_pheno.csv")
    return tmp_path / "m.vcf", tmp_path / "m_pheno.csv"


def _model_inputs(tmp_path, n_pcs):
    from gwas.qc import _pipeline_build_kinship

    vcf, pheno = _models_case(tmp_path)
    a = _args(trait="T", maf=0.01)
    mem = _memory(vcf, pheno, "T", a)
    sto, st, parsed = _store(vcf, pheno, "T", a, tmp_path / "s")
    im = _mlm_inputs(mem, _pipeline_build_kinship, n_pcs=n_pcs)
    is_ = _mlm_inputs(sto, st.build_kinship, n_pcs=n_pcs)
    assert sto["S08"]["sid"].size % 2000 == 1
    return mem, sto, st, parsed, im, is_


@pytest.mark.parametrize("n_pcs", [0, 2])
def test_farmcpu_equal_in_memory(tmp_path, n_pcs):
    from gwas.models import run_farmcpu

    mem, sto, st, parsed, im, is_ = _model_inputs(tmp_path, n_pcs)
    try:
        def run(geno, i):
            cov = CovarData(i["iid"], i["pcs"][:, :i["n_pcs"]]) if i["n_pcs"] else None
            return run_farmcpu(geno, i["sid"], i["chroms"], i["chroms_num"], i["positions"], i["iid"],
                               i["pheno_reader"], i["K0"], cov, final_scan="ols", verbose=False)
        gm, qm, cm = run(mem["S09"]["imp"], im)
        gs_, qs, cs = run(sto["S09"]["imp"], is_)
        _frames_identical(gs_, gm)
        _frames_identical(qs, qm)
        assert cs == cm
    finally:
        parsed.store.close()


def test_mlmm_equal_in_memory(tmp_path):
    from gwas.models import run_mlmm_research_grade_fast

    mem, sto, st, parsed, im, is_ = _model_inputs(tmp_path, 2)
    try:
        def run(geno, i):
            return run_mlmm_research_grade_fast(geno, i["sid"], i["chroms"], i["chroms_num"], i["positions"],
                                                i["iid"], i["pheno_reader"], i["K0"],
                                                CovarData(i["iid"], i["pcs"][:, :2]), verbose=False)
        gm, cm = run(mem["S09"]["imp"], im)
        gs_, cs = run(sto["S09"]["imp"], is_)
        _frames_identical(gs_, gm)
        _frames_identical(cs, cm)
    finally:
        parsed.store.close()


@pytest.mark.parametrize("n_jobs", [1, 3])
def test_subsampling_on_disk_equals_todays_run(tmp_path, n_jobs):
    """On disk the replicates run one after another, each under the BLAS thread count a joblib
    worker of today's parallel run gets, so the results equal today's run with n_jobs."""
    from gwas.qc import _pipeline_build_kinship
    from gwas.subsampling import subsample_gwas_resampling

    mem, sto, st, parsed, im, is_ = _model_inputs(tmp_path, 2)
    try:
        def kin(fn, out):
            K, Z, chroms_grm, _p, _m = fn(out["S08"]["geno_df"], out["S09"]["imp"],
                                          out["S08"]["chroms"], out["S08"]["positions"])
            return Z, chroms_grm

        def run(geno, i, Z, chroms_grm):
            return subsample_gwas_resampling(
                geno_imputed=geno, y=i["y"], sid=i["sid"], chroms=i["chroms"], chroms_num=i["chroms_num"],
                positions=i["positions"], iid=i["iid"], Z_for_grm=Z, n_pcs=2, n_reps=3,
                sample_frac=0.8, discovery_thresh=1e-4, seed=42, n_jobs=n_jobs, use_loco=True,
                chroms_grm=chroms_grm)
        dm, pm, mm = run(mem["S09"]["imp"], im, *kin(_pipeline_build_kinship, mem))
        ds, ps, ms = run(sto["S09"]["imp"], is_, *kin(st.build_kinship, sto))
        _frames_identical(ds, dm)
        assert ps.tobytes() == pm.tobytes()
        assert ms == mm
    finally:
        parsed.store.close()
