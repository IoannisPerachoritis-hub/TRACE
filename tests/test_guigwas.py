"""The GWAS page's computations on the genotype table (gwas/guigwas.py), with the table in memory and on
disk: the same results, byte for byte, at function level.

The page's preprocessing (gwas.qc._gwas_pipeline_stages: on disk through gwas.storepipe.TraitStore), the
models as the page calls them (MLM, MLMM and FarmCPU through their cached wrappers, which pass the
LazyColumns on disk), OLS effects, LD decay, M_eff, the QC report's MAF histogram and subsampling, on the two
bundled examples; FarmCPU also on the audit's 2,001-SNP input, whose last OLS chunk is one column wide.
Also: one current preprocessing per session, its warnings shown again on every reuse, a phenotype-QC stop
recognised in both modes, and the subsampling workers' memory check (one worker, the same numbers).
Memory against ff56e5c (and tomato) is the harness's (batch 13, track A evidence).
"""
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gwas import guigwas, guitable, kinship, models
from gwas.genostore import is_lazy
from gwas.pipeline import is_phenotype_qc_stop
from gwas.utils import CovarData, PhenoData

REPO = Path(__file__).resolve().parents[1]
EX1 = (REPO / "tests" / "data" / "example_v1" / "example.vcf.gz",
       REPO / "tests" / "data" / "example_v1" / "example_pheno.csv")
EX2 = (REPO / "examples" / "example.vcf.gz", REPO / "examples" / "example_pheno.csv")
QC = dict(maf_thresh=0.05, miss_thresh=0.10, norm_option="none", drop_alt=True, ind_miss_thresh=0.20,
          mac_thresh=5)


@pytest.fixture(autouse=True)
def _quiet():
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _page_pheno(path):
    pheno = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    pheno.columns = pheno.columns.astype(str).str.strip()
    acc = pheno.columns[0]
    pheno[acc] = pheno[acc].astype(str).str.strip()
    pheno = pheno.set_index(acc)
    pheno.index = pheno.index.astype(str).str.strip()
    return pheno


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    """Each example in memory and on disk, in a registry of its own."""
    reg = guitable.TableRegistry(max_entries=4)
    work = tmp_path_factory.mktemp("guigwas_work")
    out = {}
    for name, (vcf, _ph) in (("ex1", EX1), ("ex2", EX2)):
        ident = guitable.file_identity(vcf)
        out[name] = {mode: reg.get(ident, mode, vcf, work) for mode in ("memory", "disk")}
    yield out
    reg.close_all()


def _prep(parsed, pheno, trait, covar_df=None, **qc):
    state = {}
    return guigwas.preprocess(state, "k", parsed, pheno.copy(), trait, **{**QC, **qc}, covar_df=covar_df)


def _eq_array(a, b, name):
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape, name
    assert (a.flags.c_contiguous, a.flags.f_contiguous) == (b.flags.c_contiguous, b.flags.f_contiguous), name
    assert a.tobytes() == b.tobytes(), name


def _covariates(pheno):
    rng = np.random.default_rng(7)
    c = pd.DataFrame({"batch": rng.integers(0, 3, len(pheno)).astype(float),
                      "env": np.round(rng.normal(0, 1, len(pheno)), 4)}, index=pheno.index)
    c.iloc[[1, 5, 9], 1] = np.nan
    return c


CASES = [("ex1", "Trait1", {}), ("ex2", "Trait2", {}), ("ex2", "Trait1", {"maf_thresh": 0.10, "mac_thresh": 3}),
         ("ex1", "Trait2", {"covar": True})]


@pytest.mark.parametrize("name,trait,extra", CASES)
def test_preprocessing_on_disk_equals_memory(tables, name, trait, extra):
    pheno = _page_pheno((EX1 if name == "ex1" else EX2)[1])
    extra = dict(extra)
    cov = _covariates(pheno) if extra.pop("covar", False) else None
    mem = _prep(tables[name]["memory"], pheno, trait, covar_df=cov, **extra)
    dsk = _prep(tables[name]["disk"], pheno, trait, covar_df=cov, **extra)
    assert mem["mode"] == "memory" and dsk["mode"] == "disk"
    assert is_lazy(dsk["geno_imputed"]) and is_lazy(dsk["geno_dosage_raw"])
    assert mem["fill"] is None and mem["rows"] is None
    assert dsk["fill"].dtype == np.float32 and dsk["rows"].size == len(dsk["iid"])
    # the tables (read whole here, in the test only): the in-memory arrays exactly
    _eq_array(mem["geno_imputed"], np.asarray(dsk["geno_imputed"]), "geno_imputed")
    _eq_array(mem["geno_dosage_raw"], np.asarray(dsk["geno_dosage_raw"]), "geno_dosage_raw")
    for k in ("snp_imputation_rate", "K", "Z_for_pca", "y", "iid", "sid", "chroms", "chroms_num", "positions",
              "chroms_grm", "positions_grm"):
        _eq_array(mem[k], dsk[k], k)
    assert mem["qc_snp"] == dsk["qc_snp"] and mem["allele_map"] == dsk["allele_map"]
    for k in ("n_snps_raw", "kinship_model", "info_field", "covar_dropped", "covar_n_before", "warnings"):
        assert mem[k] == dsk[k], k
    assert list(mem["geno_df"].index) == list(dsk["geno_df"].index)
    assert guigwas.n_snps(mem) == guigwas.n_snps(dsk) == mem["geno_df"].shape[1]
    assert mem["pheno"].to_csv() == dsk["pheno"].to_csv()
    qm, qd = mem["qc_report"], dsk["qc_report"]
    assert qm["sample_het"]["per_sample"].to_csv() == qd["sample_het"]["per_sample"].to_csv()
    assert qm["dup_pairs"] == qd["dup_pairs"] and qm["imputation"] == qd["imputation"]
    _eq_array(qm["variant_fis"]["fis"], qd["variant_fis"]["fis"], "fis")
    # the QC report's MAF histogram: the column means, read in pieces on disk
    _eq_array(guigwas.column_means(mem["geno_imputed"]), guigwas.column_means(dsk["geno_imputed"]), "means")
    _eq_array(guigwas.column_means(mem["geno_imputed"]), mem["geno_imputed"].mean(axis=0), "means vs numpy")


def _page_models(r, tid, n_pcs=2):
    """MLM, OLS effects, M_eff, LD decay, MLMM and FarmCPU as the page calls them."""
    pcs_full, _e = kinship.compute_pcs_full_cached(tid, r["Z_for_pca"], max_pcs=20)
    K0, K_by_chr, _d = kinship.build_loco_kernels_cached(
        tid, _iid=r["iid"], _Z_grm=r["Z_for_pca"], _chroms_grm=np.asarray(r["chroms_grm"], dtype=str),
        _K_base=r["K"])
    chroms = np.asarray(r["chroms"], dtype=str)
    reader = PhenoData(iid=r["iid"], val=r["y"])
    covar = CovarData(iid=r["iid"], val=pcs_full[:, :n_pcs], names=[f"PC{i + 1}" for i in range(n_pcs)])
    view = r["geno_imputed"]
    common = dict(_geno_imputed=view, _sid=r["sid"], _chroms=chroms, _chroms_num=r["chroms_num"],
                  _positions=r["positions"], _iid=r["iid"], _K0=K0, _pheno_reader=reader)
    mlm = models.run_gwas_cached(tid, n_pcs, use_loco=True, trait_name="t", _y=r["y"], _pcs_full=pcs_full,
                                 _K_by_chr=K_by_chr, **common)
    mlm = models.add_ols_effects_to_gwas(mlm, view, r["y"], covar, r["sid"])
    meff = guigwas.meff({}, tid, view)
    decay = guigwas.ld_decay(tid, view, chroms, r["positions"])
    mlmm = models.run_mlmm_core_cached(tid, n_pcs, p_enter=1e-3, max_cof=3, window_kb=200,
                                       _covar_reader=covar, **common)
    fc = models.run_farmcpu_cached(tid, n_pcs, max_iterations=10, final_scan="ols", use_loco=True,
                                   _covar_reader=covar, **common)
    return {"mlm": mlm.to_csv(index=False), "meff": (meff[0], meff[1].tobytes()),
            "decay": (decay[0].to_csv(index=False), decay[1].to_csv(index=False), decay[2]),
            "mlmm": (mlmm[0].to_csv(index=False), mlmm[1].to_csv(index=False)),
            "farmcpu": (fc[0].to_csv(index=False), fc[1].to_csv(index=False), repr(fc[2]))}


@pytest.mark.parametrize("name,trait", [("ex1", "Trait1"), ("ex2", "Trait2")])
def test_models_and_their_inputs_on_disk_equal_memory(tables, name, trait):
    pheno = _page_pheno((EX1 if name == "ex1" else EX2)[1])
    for f in (models.run_gwas_cached, models.run_mlmm_core_cached, models.run_farmcpu_cached, guigwas.ld_decay,
              kinship.compute_pcs_full_cached, kinship.build_loco_kernels_cached):
        f.clear()
    mem = _page_models(_prep(tables[name]["memory"], pheno, trait), f"{name}-memory")
    dsk = _page_models(_prep(tables[name]["disk"], pheno, trait), f"{name}-disk")
    for k in mem:
        assert mem[k] == dsk[k], k


def test_farmcpu_one_column_last_chunk_on_disk_equals_memory(tmp_path):
    """The audit's input whose last OLS chunk is one column wide (2,001 SNPs): FarmCPU through the
    page's wrapper gets, on disk, a strided view as that chunk, as the C-ordered copy's slice is in
    memory, and gives the same p-values and pseudo-QTNs."""
    from tests import _store_cases as sc

    argv = sc.farmcpu_case("h2001_null", tmp_path)
    vcf, ph = Path(argv[1]), Path(argv[3])
    reg = guitable.TableRegistry()
    try:
        ident = guitable.file_identity(vcf)
        pheno = _page_pheno(ph)
        out = {}
        for mode in ("memory", "disk"):
            r = _prep(reg.get(ident, mode, vcf, tmp_path / "work"), pheno, "Tnull")
            assert len(r["sid"]) % 2000 == 1                                # the last chunk: one column
            models.run_farmcpu_cached.clear()
            fc = models.run_farmcpu_cached(
                f"h2001-{mode}", 0, max_iterations=10, final_scan="ols", use_loco=True,
                _geno_imputed=r["geno_imputed"], _sid=r["sid"], _chroms=np.asarray(r["chroms"], dtype=str),
                _chroms_num=r["chroms_num"], _positions=r["positions"], _iid=r["iid"],
                _K0=kinship._build_loco_kernels_impl(r["iid"], r["Z_for_pca"], np.asarray(r["chroms_grm"],
                                                     dtype=str), r["K"])[0],
                _pheno_reader=PhenoData(iid=r["iid"], val=r["y"]), _covar_reader=None)
            out[mode] = (fc[0].to_csv(index=False), fc[1].to_csv(index=False))
            if mode == "disk":
                lazy_c = r["geno_imputed"].astype(np.float32).as_layout("C").astype(float)
                last = lazy_c[:, 2000:2001]
                mem_c = np.asarray(np.asarray(out["mem_table"]), dtype=np.float32, order="C")
                mem_c = np.asarray(mem_c, float)[:, 2000:2001]
                assert not last.flags.c_contiguous and not mem_c.flags.c_contiguous     # both strided
                assert last.tobytes() == np.ascontiguousarray(mem_c).tobytes()
            else:
                out["mem_table"] = r["geno_imputed"]
        assert out["memory"] == out["disk"]
    finally:
        reg.close_all()


def test_one_current_preprocessing_per_session(tables):
    pheno = _page_pheno(EX1[1])
    state = {}
    mem = tables["ex1"]["memory"]
    a = guigwas.preprocess(state, "key-a", mem, pheno.copy(), "Trait1", **QC)
    assert guigwas.preprocess(state, "key-a", mem, pheno.copy(), "Trait1", **QC) is a      # reused
    b = guigwas.preprocess(state, "key-b", mem, pheno.copy(), "Trait2", **QC)
    assert state[guigwas.PREP_KEY][0] == "key-b" and state[guigwas.PREP_KEY][1] is b       # one entry
    assert guigwas.preprocess(state, "key-a", mem, pheno.copy(), "Trait1", **QC) is not a  # replaced
    # another table object under the same key (the registry rebuilt it): computed again
    c = guigwas.preprocess(state, "key-c", tables["ex1"]["disk"], pheno.copy(), "Trait1", **QC)
    assert guigwas.preprocess(state, "key-c", tables["ex1"]["memory"], pheno.copy(), "Trait1", **QC) is not c


def test_a_failing_trait_keeps_the_current_preprocessing(tables):
    pheno = _page_pheno(EX1[1])
    pheno["Const"] = 1.0
    state = {}
    for mode in ("memory", "disk"):
        good = guigwas.preprocess(state, f"good-{mode}", tables["ex1"][mode], pheno.copy(), "Trait1", **QC)
        with pytest.raises(ValueError) as exc:
            guigwas.preprocess(state, f"const-{mode}", tables["ex1"][mode], pheno.copy(), "Const", **QC)
        assert is_phenotype_qc_stop(exc.value), mode           # the page skips this trait
        assert state[guigwas.PREP_KEY][1] is good
        with pytest.raises(ValueError, match="All samples removed by missingness filter"):
            guigwas.preprocess(state, f"miss-{mode}", tables["ex1"][mode], pheno.copy(), "Trait1",
                               **{**QC, "ind_miss_thresh": 0.0})


def test_the_preprocessings_warnings_are_shown_again_on_reuse(tmp_path):
    """A VCF with a split multi-allelic record: the collapse warning is shown when the preprocessing
    runs and again on every reuse (Streamlit's cache replayed it at ff56e5c), in both modes."""
    lines = ["##fileformat=VCFv4.2", '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">',
             "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(f"S{i}" for i in range(30))]
    rng = np.random.default_rng(3)
    for ch in ("1", "2"):
        for j in range(40):
            pos = (j + 1) * 1000 if not (ch == "1" and j == 5) else 5000      # j=4 and j=5 share 1:5000
            g = rng.choice(["0/0", "0/1", "1/1"], 30)
            lines.append(f"{ch}\t{pos}\t.\tA\tT\t.\tPASS\t.\tGT\t" + "\t".join(g))
    vcf = tmp_path / "dup.vcf"
    vcf.write_text("\n".join(lines) + "\n")
    pheno = pd.DataFrame({"T": rng.normal(size=30)}, index=[f"S{i}" for i in range(30)])
    reg = guitable.TableRegistry()
    try:
        ident = guitable.file_identity(vcf)
        for mode in ("memory", "disk"):
            state, shown = {}, []
            parsed = reg.get(ident, mode, vcf, tmp_path / "work")
            guigwas.preprocess(state, "k", parsed, pheno.copy(), "T", **QC, show=shown.append)
            guigwas.preprocess(state, "k", parsed, pheno.copy(), "T", **QC, show=shown.append)
            assert len(shown) == 2 and "multi-allelic records collapsed" in shown[0] and shown[0] == shown[1]
    finally:
        reg.close_all()


def test_subsampling_with_two_workers_falls_back_to_one_with_the_same_results(tables):
    pheno = _page_pheno(EX1[1])
    r = _prep(tables["ex1"]["memory"], pheno, "Trait1")
    d = _prep(tables["ex1"]["disk"], pheno, "Trait1")
    pcs_full, _e = kinship._compute_pcs_full_impl(r["Z_for_pca"], 20)

    def run(res, **kw):
        return guigwas.subsampling(
            res["geno_imputed"], res["y"], res["sid"], np.asarray(res["chroms"], dtype=str), res["chroms_num"],
            res["positions"], res["iid"], np.asarray(res["Z_for_pca"], dtype=np.float32), pcs_full, 1, 4, 0.8,
            1e-4, 42, 2, **kw)

    assert guigwas.subsampling_plan(r["geno_imputed"], 2) == (2, None)          # the workers fit
    assert guigwas.subsampling_plan(d["geno_imputed"], 2) == (1, None)          # on disk: one by one
    workers, note = guigwas.subsampling_plan(r["geno_imputed"], 2, available=1)
    assert workers == 1 and note.startswith("The subsampling iterations run one after another instead "
                                            "of on 2 cores") and note.endswith("The results are the same.")
    par = run(r)                                    # two processes
    seq = run(r, available=1)                       # one after another, under a worker's thread limits
    dsk = run(d)                                    # on disk: one after another
    assert par[3] is None and dsk[3] is None and seq[3] == note
    for out in (seq, dsk):
        assert out[0].to_csv(index=False) == par[0].to_csv(index=False)
        assert out[1].tobytes() == par[1].tobytes()
        assert pd.DataFrame(out[2]).to_csv(index=False) == pd.DataFrame(par[2]).to_csv(index=False)


def test_workers_fit_rule():
    fits, need, free = guigwas.subsampling_workers_fit(1000, 100_000, 4, available=100 * 2**30)
    # three workers after the first: 300 MiB and 12 bytes per call each
    assert fits and abs(need - (3 * 300 * 2**20 + 3 * 12 * 1000 * 100_000)) < 2**10
    assert not guigwas.subsampling_workers_fit(1000, 100_000, 4, available=need)[0]
    assert guigwas.subsampling_workers_fit(1000, 100_000, 1, available=1)[0]    # one worker: nothing extra


@pytest.mark.parametrize("choice", ["auto", "memory", "disk"])
def test_one_click_records_where_the_table_is_kept_as_the_cli_does(tmp_path, choice, monkeypatch):
    """One-Click's table is the registry's, in the mode of its own plan, and run_metadata.json gets the
    CLI's five keys for it (gwas.storeplan.Plan.metadata): the same values as a CLI run with
    --geno-store <choice> on the same file and options, and the same results in both modes. The free
    memory both plans read is pinned high, so Automatic keeps this small table in memory whatever
    else runs on the machine."""
    import io
    import json
    import zipfile

    import cli
    from gwas import pipeline as pl
    from gwas import storeplan
    from gwas.storeplan import MODE_KEYS

    monkeypatch.setattr(storeplan, "available_memory_bytes", lambda: 64 * 2 ** 30)

    vcf, ph = EX1
    opts = pl.cli_default_options(vcf=vcf.name, pheno=ph.name, output=None, no_plots=True, no_report=True,
                                  run_label="One-Click Full Analysis")
    reg = guitable.TableRegistry()
    try:
        import gwas.guitable as gt_mod
        real = gt_mod.get_registry
        gt_mod.get_registry = lambda: reg                       # this test's own registry
        try:
            parsed, meta, plan = guigwas.one_click_table(guitable.file_identity(vcf), str(vcf), choice,
                                                         tmp_path / "work", opts)
        finally:
            gt_mod.get_registry = real
        assert plan.mode == ("disk" if choice == "disk" else "memory")
        assert getattr(parsed, "is_store", False) == (plan.mode == "disk")
        assert set(meta) == set(MODE_KEYS)
        opts._geno_store_meta = meta
        [(trait, res, err)] = list(pl.run_traits(parsed, pd.read_csv(ph, index_col=0), ["Trait1"], opts))
        assert err is None
        gui_meta = json.loads(zipfile.ZipFile(io.BytesIO(res.zip_bytes)).read("run_metadata.json"))
        out = tmp_path / "cli"
        cli.run_pipeline(cli._build_parser().parse_args(
            ["--vcf", str(vcf), "--pheno", str(ph), "--trait", "Trait1", "--no-plots", "--no-report",
             "--geno-store", choice, "--output", str(out)]))
        (z,) = sorted(out.glob("*.zip"))
        cli_meta = json.loads(zipfile.ZipFile(z).read("run_metadata.json"))
        assert {k: gui_meta[k] for k in MODE_KEYS} == {k: cli_meta[k] for k in MODE_KEYS}
    finally:
        reg.close_all()


# ------------------------------------------------- a table file closed while a session holds it
@pytest.fixture()
def own_registry(monkeypatch):
    """A registry of the test's own, as the page's table() sees it (gwas.guitable.get_registry)."""
    reg = guitable.TableRegistry()
    monkeypatch.setattr(guitable, "get_registry", lambda: reg)
    yield reg
    reg.close_all()


def test_a_closed_table_is_dropped_and_read_again_on_the_next_run(tmp_path, own_registry):
    """Another session's Clear (or the registry making room) closes the session's table file: the
    page's next run drops what was built on it, takes a fresh table and preprocesses again, with
    the same values, instead of failing on the closed file."""
    vcf, ph = EX1
    ident = guitable.file_identity(vcf)
    pheno = _page_pheno(ph)
    state = {}
    p1 = guigwas.table(ident, "disk", vcf, tmp_path / "w")
    first = guigwas.preprocess(state, "k", p1, pheno.copy(), "Trait1", **QC)
    ref = np.asarray(first["geno_imputed"])
    state["geno_table"] = guitable.SessionTable(identity="i", file_identity=ident, mode="disk", parsed=p1,
                                                view=first["geno_imputed"])
    assert not guigwas.table_closed(state["geno_table"])
    own_registry.drop()
    assert guigwas.table_closed(state["geno_table"]) and guigwas.table_closed(p1)
    assert not guigwas.drop_closed_table(dict(state), "another file", "disk")     # not this VCF's
    assert guigwas.drop_closed_table(state, ident, "disk") and guigwas.PREP_KEY not in state
    p2 = guigwas.table(ident, "disk", vcf, tmp_path / "w")
    again = guigwas.preprocess(state, "k", p2, pheno.copy(), "Trait1", **QC)
    assert p2 is not p1 and not guigwas.table_closed(p2)
    _eq_array(np.asarray(again["geno_imputed"]), ref, "geno_imputed")
    mem = guitable.SessionTable(identity="i", file_identity=ident, mode="memory",
                                parsed=own_registry.get(ident, "memory", vcf))
    assert not guigwas.table_closed(mem)                                         # never closed


def test_a_table_closed_during_the_preprocessing_is_read_again_once(tmp_path, own_registry):
    vcf, ph = EX1
    ident = guitable.file_identity(vcf)
    pheno = _page_pheno(ph)
    ref = np.asarray(guigwas.preprocess({}, "r", guigwas.table(ident, "disk", vcf, tmp_path / "w"),
                                        pheno.copy(), "Trait1", **QC)["geno_imputed"])
    closed = guigwas.table(ident, "disk", vcf, tmp_path / "w")
    own_registry.drop()
    state, reads = {}, []

    def reread():
        reads.append(1)
        return guigwas.table(ident, "disk", vcf, tmp_path / "w")

    out, used, again = guigwas.preprocess_rereading(state, "k", closed, reread, pheno.copy(), "Trait1", **QC)
    assert again and reads == [1] and used is not closed and not guigwas.table_closed(used)
    _eq_array(np.asarray(out["geno_imputed"]), ref, "geno_imputed")
    pheno["Const"] = 1.0
    with pytest.raises(ValueError) as exc:                                       # another error: raised
        guigwas.preprocess_rereading(state, "c", used, reread, pheno.copy(), "Const", **QC)
    assert is_phenotype_qc_stop(exc.value) and reads == [1]


def test_one_click_runs_the_remaining_traits_again_on_a_fresh_table(tmp_path, own_registry, monkeypatch):
    """A table file closed while One-Click runs its first trait: that trait and the next run on a
    fresh table, with the results of a run that was not interrupted."""
    import io
    import zipfile

    from gwas import pipeline as pl

    vcf, ph = EX1
    ident = guitable.file_identity(vcf)
    pheno = pd.read_csv(ph, index_col=0)
    opts = pl.cli_default_options(vcf=vcf.name, pheno=ph.name, output=None, no_plots=True, no_report=True,
                                  run_label="One-Click Full Analysis")

    def tables(res):
        with zipfile.ZipFile(io.BytesIO(res.zip_bytes)) as zf:
            return {n: zf.read(n) for n in zf.namelist() if n.startswith("tables/")}

    plain = {t: tables(r) for t, r, _e in pl.run_traits(guigwas.table(ident, "disk", vcf, tmp_path / "w"),
                                                          pheno, ["Trait1", "Trait2"], opts)}
    p1 = guigwas.table(ident, "disk", vcf, tmp_path / "w")
    real, seen = pl.run_trait_pipeline, []

    def closing(parsed, pheno_df, trait, options, progress=None):
        seen.append((trait, parsed))
        if len(seen) == 1:
            own_registry.drop()                     # Clear in another session, during the first trait
        return real(parsed, pheno_df, trait, options, progress=progress)

    monkeypatch.setattr(pl, "run_trait_pipeline", closing)
    started = []
    out = list(guigwas.run_traits_rereading(p1, pheno, ["Trait1", "Trait2"], opts,
                                            lambda: guigwas.table(ident, "disk", vcf, tmp_path / "w"),
                                            on_start=lambda i, t: started.append((i, t))))
    assert [(t, err is None, again) for t, _r, err, again in out] == [("Trait1", True, True), ("Trait2", True, False)]
    assert [t for t, _p in seen] == ["Trait1", "Trait1", "Trait2"] and seen[1][1] is not p1
    assert started == [(1, "Trait1"), (1, "Trait1"), (2, "Trait2")]
    assert {t: tables(r) for t, r, _e, _a in out} == plain


def test_after_another_sessions_clear_an_uploaded_vcf_is_asked_for_plainly(tmp_path):
    """The round 1 re-check's r8, case 3: the session's VCF was an upload, and Clear session data in
    another session closed its table and deleted the saved upload. The session's next run says so
    (the page reads nothing), and a read that would read the VCF again (the setup's preprocessing,
    One-Click's next trait) raises VcfGone with that message, which the page shows, instead of a
    FileNotFoundError. A VCF given by its location is read again; one gone from its location is
    named."""
    import ast
    import io

    from gwas import pipeline as pl

    vcf, ph = EX1
    work = tmp_path / "w"
    pheno = _page_pheno(ph)
    opts = pl.cli_default_options(vcf=vcf.name, pheno=ph.name, output=None, no_plots=True, no_report=True,
                                  run_label="One-Click Full Analysis")
    upload = io.BytesIO(vcf.read_bytes())
    upload.name, upload.size, upload.file_id = vcf.name, vcf.stat().st_size, "u-1"
    state = {"_persist_pheno": "pheno"}
    try:
        guitable.vcf_input(state, "", upload, work)
        path = state["_persist_vcf_path"]
        ident = guitable.file_identity(path)
        held = guigwas.table(ident, "disk", path, work)
        guitable.clear_all({}, memo_folder=tmp_path / "memo")                  # another session's Clear
        assert guigwas.table_closed(held) and not Path(path).exists()
        nxt = guitable.vcf_input(state, "", None, work)                         # this session's next run
        assert nxt.warning == guitable.UPLOAD_REMOVED_MESSAGE == (
            "The uploaded VCF is no longer in the work folder. Clear session data (in any session) deletes "
            "saved uploads, and the system may clean its temporary folder, where the work folder is by "
            "default. Upload the VCF again.")
        assert not nxt.error and not guitable.inputs_kept(state)                # nothing is read
        with pytest.raises(guigwas.VcfGone) as gone:          # the setup's hash, the file deleted since
            guigwas.vcf_identity(path)                         # the page checked that it is there
        assert str(gone.value) == guitable.UPLOAD_REMOVED_MESSAGE
        assert guigwas.vcf_identity(vcf) == guitable.file_identity(vcf)        # a file there: its identity

        def reread():
            return guigwas.table(ident, "disk", path, work)

        with pytest.raises(guigwas.VcfGone) as gone:                            # the setup, mid-run
            guigwas.preprocess_rereading({}, "k", held, reread, pheno.copy(), "Trait1", **QC)
        assert str(gone.value) == guitable.UPLOAD_REMOVED_MESSAGE
        with pytest.raises(guigwas.VcfGone) as gone:                            # One-Click, mid-run
            list(guigwas.run_traits_rereading(held, pd.read_csv(ph, index_col=0), ["Trait1"], opts, reread))
        assert str(gone.value) == guitable.UPLOAD_REMOVED_MESSAGE

        loc = tmp_path / "loc" / "panel.vcf.gz"                                 # a location: read again
        loc.parent.mkdir()
        loc.write_bytes(vcf.read_bytes())
        lident = guitable.file_identity(loc)
        lheld = guigwas.table(lident, "disk", str(loc), work)
        guitable.clear_all({}, memo_folder=tmp_path / "memo")
        _out, used, again = guigwas.preprocess_rereading(
            {}, "k", lheld, lambda: guigwas.table(lident, "disk", str(loc), work), pheno.copy(), "Trait1", **QC)
        assert again and not guigwas.table_closed(used)
        guitable.get_registry().close_all()
        loc.unlink()
        with pytest.raises(guigwas.VcfGone, match="No file at"):
            guigwas.table(lident, "disk", str(loc), work)
    finally:
        guitable.get_registry().close_all()
        guitable.remove_saved_uploads()
    # the page shows VcfGone's message where it reads the VCF: the setup's hash and its table, One-Click's
    # table and its fresh table after a closed one; a Run that found its table closed says what happens
    page = ast.parse((REPO / "pages" / "GWAS_analysis.py").read_text(encoding="utf-8"))
    handlers = [h for h in ast.walk(page) if isinstance(h, ast.ExceptHandler) and h.type is not None
                and "_gg.VcfGone" in ast.unparse(h.type)]
    assert len(handlers) == 4
    assert "_gt.file_identity(vcf_path)" not in ast.unparse(page)            # the hash through vcf_identity
    assert "reads the VCF again" in guigwas.CLOSED_MESSAGE and "upload the VCF again" in guigwas.CLOSED_MESSAGE


def test_one_click_stopped_mid_way_keeps_the_traits_it_finished(tmp_path, own_registry, monkeypatch):
    """The round 2 re-check's q2 c: One-Click over two traits, the table on disk and the VCF an upload;
    Clear session data in another session while trait 2 runs closes the table and deletes the saved
    upload, so no fresh table can be read (VcfGone). The page stopped there with the message only, and
    trait 1's finished results were lost. Now trait 1 keeps them: its summary and the combined ZIP,
    whose failed_traits.txt names trait 2 with the plain message, as after a run with a failed trait;
    the page still stops plainly when no trait finished."""
    import ast
    import io
    import zipfile

    from gwas import pipeline as pl
    from gwas.plotting import _combine_gwas_zips

    # the page: the trait loop's stop handler stops only when no trait finished, and keeps the rest
    page = ast.parse((REPO / "pages" / "GWAS_analysis.py").read_text(encoding="utf-8"))
    (loop,) = [n for n in ast.walk(page) if isinstance(n, ast.Try)
               and any("_gg.run_traits_rereading(" in ast.unparse(s) for s in n.body)]
    (handler,) = loop.handlers
    assert ast.unparse(handler.type) == "(_gg.PlanStop, _gg.VcfGone)"
    stops = [n for n in ast.walk(handler) if isinstance(n, ast.Call) and ast.unparse(n.func) == "st.stop"]
    guards = [n for n in handler.body if isinstance(n, ast.If) and ast.unparse(n.test) == "not _done"]
    assert stops and len(guards) == 1 and all(any(s is n for n in ast.walk(guards[0])) for s in stops)
    assert "_not_run = _gg.traits_not_run(selected_traits, _done, _failed, _oc_stop)" in ast.unparse(handler)
    assert "_failed.update(_not_run)" in ast.unparse(handler)

    # q2 c, through the loop the page runs
    vcf, ph = EX1
    work = tmp_path / "w"
    opts = pl.cli_default_options(vcf=vcf.name, pheno=ph.name, output=None, no_plots=True, no_report=True,
                                  run_label="One-Click Full Analysis")
    traits = ["Trait1", "Trait2"]
    upload = io.BytesIO(vcf.read_bytes())
    upload.name, upload.size, upload.file_id = vcf.name, vcf.stat().st_size, "u-2"
    state, seen, real = {}, [], pl.run_trait_pipeline

    def clearing(parsed, pheno_df, trait, options, progress=None):
        seen.append(trait)
        if len(seen) == 2:
            guitable.clear_all({}, memo_folder=tmp_path / "memo")    # another session's Clear, during trait 2
        return real(parsed, pheno_df, trait, options, progress=progress)

    done, failed = {}, {}
    try:
        guitable.vcf_input(state, "", upload, work)
        path = state["_persist_vcf_path"]
        ident = guitable.file_identity(path)
        parsed = guigwas.table(ident, "disk", path, work)
        monkeypatch.setattr(pl, "run_trait_pipeline", clearing)
        with pytest.raises(guigwas.VcfGone) as stop:
            for t, res, err, _again in guigwas.run_traits_rereading(
                    parsed, pd.read_csv(ph, index_col=0), traits, opts,
                    lambda: guigwas.table(ident, "disk", path, work)):
                if res is None:
                    failed[t] = err
                else:
                    done[t] = res.zip_bytes
    finally:
        guitable.remove_saved_uploads()
    assert list(done) == ["Trait1"] and failed == {} and seen == ["Trait1", "Trait2"]
    not_run = guigwas.traits_not_run(traits, done, failed, stop.value)
    assert not_run == {"Trait2": guitable.UPLOAD_REMOVED_MESSAGE}
    failed.update(not_run)
    _name, buf = _combine_gwas_zips(done, pheno_label="example_pheno",
                                    extra_members={"failed_traits.txt": pl.failed_traits_report(traits, done, failed)})
    with zipfile.ZipFile(buf) as combined, zipfile.ZipFile(io.BytesIO(done["Trait1"])) as one:
        assert {n for n in combined.namelist() if n.startswith("Trait1/")} == {f"Trait1/{n}" for n in one.namelist()}
        assert not [n for n in combined.namelist() if n.startswith("Trait2/")]
        report = combined.read("failed_traits.txt").decode()
    assert "1 of 2 traits completed" in report and "Trait1: completed" in report and "Trait2: FAILED" in report
    assert guitable.UPLOAD_REMOVED_MESSAGE in report
    assert guigwas.traits_not_run(["A", "B", "C"], {"A": 1}, {"B": "its own error"}, "stopped") == {"C": "stopped"}


def test_a_new_tables_build_checks_the_plan_again_on_the_memory_and_disk_free_then(tmp_path, own_registry,
                                                                                  monkeypatch):
    """The server builds one table at a time, so a build may start after another session's: its check
    (build_check) makes the plan again on the memory and disk free then, by the rule that chose its mode,
    and stops in the page's words when that plan no longer holds, building nothing; One-Click's table too.
    Automatic in memory stops by Automatic's own rule (the estimate above 80% of the free memory), exactly
    when Automatic's next plan keeps the table on disk, and says so; In memory by its own (the estimate
    above the free memory). A table already there is not checked again. Every table the GWAS page reads
    passes the check."""
    import ast

    from gwas import pipeline as pl
    from gwas import storeplan

    vcf, ph = EX1
    ident = guitable.file_identity(vcf)
    work = tmp_path / "w"
    free = {"memory": [64 * 2**30], "disk": [100 * 2**30]}         # the last value stays
    monkeypatch.setattr(storeplan, "available_memory_bytes",
                        lambda: free["memory"].pop(0) if len(free["memory"]) > 1 else free["memory"][0])
    monkeypatch.setattr(storeplan, "free_disk_bytes",
                        lambda folder: free["disk"].pop(0) if len(free["disk"]) > 1 else free["disk"][0])
    gui = dict(estimate=guitable.gui_estimate_mib)
    plans = {c: storeplan.plan(vcf, choice=c, store_base=work, models=("mlm", "mlmm", "farmcpu"),
                               size_of=guitable.vcf_size_of(ident), **gui) for c in ("auto", "memory", "disk")}
    assert [plans[c].mode for c in ("auto", "memory", "disk")] == ["memory", "memory", "disk"]
    checks = {c: guigwas.build_check(vcf, p, work, ident, **gui) for c, p in plans.items()}
    for check in checks.values():
        check()                                                      # nothing changed: no stop
    need = plans["auto"].estimate_mib * storeplan.MIB

    def automatic_next_plan_mode():
        return storeplan.plan(vcf, choice="auto", store_base=work, models=("mlm", "mlmm", "farmcpu"),
                              size_of=guitable.vcf_size_of(ident), **gui).mode

    # the round 2 re-check's q6: between Automatic's 80% and all the free memory, Automatic's next plan
    # keeps the table on disk, so its build stops and says so; In memory builds
    for fraction, stops in ((0.79, False), (0.81, True), (0.9, True)):
        free.update(memory=[int(need / fraction)], disk=[100 * 2**30])
        assert (automatic_next_plan_mode() == "disk") == stops
        checks["memory"]()                                           # In memory: it still fits
        if not stops:
            checks["auto"]()
            continue
        with pytest.raises(guigwas.PlanStop) as stop:
            guigwas.table(ident, "memory", vcf, work, check=checks["auto"])
        assert str(stop.value).startswith(guigwas.BUILD_CHECK_NOTE)
        assert str(stop.value).endswith("are free (Automatic keeps the table in memory up to 80% of the free "
                                        "memory). Run again: Automatic then keeps the table on disk.")
        assert own_registry.keys() == []                             # nothing built
    free.update(memory=[2**20], disk=[100 * 2**30])                 # another build used the memory meanwhile
    with pytest.raises(guigwas.PlanStop) as stop:
        guigwas.table(ident, "memory", vcf, work, check=checks["auto"])
    assert str(stop.value).endswith("Run again: Automatic then keeps the table on disk.")
    free.update(memory=[2**20], disk=[2**20])                        # ... and the disk
    said = {}
    for c, check in checks.items():
        with pytest.raises(guigwas.PlanStop) as stop:
            guigwas.table(ident, plans[c].mode, vcf, work, check=check)
        said[c] = str(stop.value)
    assert all(s.startswith(guigwas.BUILD_CHECK_NOTE) for s in said.values())
    assert "so Automatic would now keep the table on disk. The genotype table on disk needs" in said["auto"]
    assert "Choose On disk under Large datasets" in said["memory"]
    assert f"in {work}, and 0.0 GB are free there. Free some space" in said["disk"]
    assert said["auto"].endswith(said["disk"][len(guigwas.BUILD_CHECK_NOTE):])     # the disk's own stop
    assert own_registry.keys() == [] and not list(work.glob(f"{guitable.GUI_STORE_PREFIX}_*"))

    free.update(memory=[64 * 2**30], disk=[100 * 2**30])
    held = guigwas.table(ident, "disk", vcf, work, check=checks["disk"])
    free.update(disk=[2**20])
    assert guigwas.table(ident, "disk", vcf, work, check=checks["disk"]) is held   # there: not checked
    own_registry.close_all()

    opts = pl.cli_default_options(vcf=vcf.name, pheno=ph.name, output=None, no_plots=True, no_report=True,
                                  run_label="One-Click Full Analysis")
    free.update(disk=[100 * 2**30, 2**20])                           # room for its plan, none at its build
    with pytest.raises(guigwas.PlanStop, match="checked the free memory and disk again"):
        guigwas.one_click_table(ident, str(vcf), "disk", work, opts, **gui)
    assert own_registry.keys() == [] and not list(work.glob(f"{guitable.GUI_STORE_PREFIX}_*"))

    page = ast.parse((REPO / "pages" / "GWAS_analysis.py").read_text(encoding="utf-8"))
    reads = [n for n in ast.walk(page) if isinstance(n, ast.Call) and ast.unparse(n.func) == "_gg.table"]
    assert len(reads) == 2 and all("check" in {k.arg for k in n.keywords} for n in reads)


def test_the_page_says_when_it_waits_for_another_sessions_build(tmp_path, own_registry, monkeypatch):
    """The round 2 re-check's q1 m: a session waiting for another session's build saw "Reading the VCF…".
    The table's spinners now say "Waiting for another session's genotype table to be built…" during the
    wait and "Reading the VCF…" only during the session's own build (nothing for a table already there),
    and every table the GWAS page and One-Click read shows them, with no other spinner around."""
    import ast
    import contextlib
    import threading

    from gwas import genostore

    page = ast.parse((REPO / "pages" / "GWAS_analysis.py").read_text(encoding="utf-8"))
    spinners = [ast.unparse(n) for n in ast.walk(page) if isinstance(n, ast.Call) and ast.unparse(n.func) == "st.spinner"]
    assert not [s for s in spinners if "Reading the VCF" in s]               # none around a wait
    reads = [n for n in ast.walk(page) if isinstance(n, ast.Call)
             and ast.unparse(n.func) in ("_gg.table", "_gg.one_click_table")]
    assert len(reads) == 3
    assert all({ast.unparse(k.value) for k in n.keywords if k.arg == "spinner"} == {"st.spinner"} for n in reads)

    vcf, _ph = EX1
    ident = guitable.file_identity(vcf)
    work = tmp_path / "w"
    shown = []

    def spinner(text):
        shown.append(text)
        return contextlib.nullcontext()

    guigwas.table(ident, "disk", vcf, work, spinner=spinner)                  # its own build
    assert shown == ["Reading the VCF into the genotype table on disk…"] == [guigwas.READING_MESSAGES["disk"]]
    shown.clear()
    guigwas.table(ident, "disk", vcf, work, spinner=spinner)                  # there: nothing
    assert shown == []
    real_ingest, started, release = genostore.ingest_vcf, threading.Event(), threading.Event()

    def slow_ingest(*a, **k):
        started.set()
        release.wait(30)
        return real_ingest(*a, **k)

    monkeypatch.setattr(genostore, "ingest_vcf", slow_ingest)
    other = threading.Thread(target=guigwas.table, args=("another-vcf", "disk", vcf, work))
    other.start()
    assert started.wait(30)
    mine = threading.Thread(target=guigwas.table, args=(ident, "memory", vcf, work), kwargs={"spinner": spinner})
    mine.start()
    for _ in range(300):
        if shown:
            break
        mine.join(0.05)
    mine.join(0.3)
    assert shown == ["Waiting for another session's genotype table to be built…"] == [guigwas.WAITING_MESSAGE]
    release.set()
    other.join(60)
    mine.join(60)
    assert shown == [guigwas.WAITING_MESSAGE, "Reading the VCF…"]


def test_the_qc_reports_frequencies_and_pc_spectrum_are_kept_per_table(tables, monkeypatch):
    """The QC report's allele frequencies (the table read whole, in pieces on disk) and the PC
    spectrum (a PCA of the pruned table) are made once per table identity, not on every rerun, with
    the values the page computed on every rerun before."""
    from gwas import pc_diagnostics as pcd

    pheno = _page_pheno(EX1[1])
    r = _prep(tables["ex1"]["disk"], pheno, "Trait1")
    state, calls = {}, []
    real_means, real_pcd = guigwas.column_means, pcd.compute_pc_diagnostics
    monkeypatch.setattr(guigwas, "column_means", lambda g, piece=4096: calls.append("means") or real_means(g, piece))
    monkeypatch.setattr(pcd, "compute_pc_diagnostics", lambda *a, **k: calls.append("pca") or real_pcd(*a, **k))
    f1 = guigwas.allele_frequencies(state, "id-1", r["geno_imputed"])
    d1 = guigwas.pc_diagnostics(state, "id-1", r["Z_for_pca"])
    assert guigwas.allele_frequencies(state, "id-1", r["geno_imputed"]) is f1        # a rerun: kept
    assert guigwas.pc_diagnostics(state, "id-1", r["Z_for_pca"]) is d1
    assert calls == ["means", "pca"]
    _eq_array(f1, real_means(r["geno_imputed"]) / 2.0, "allele frequencies")
    assert not f1.flags.writeable
    ref = real_pcd(r["Z_for_pca"], n=int(r["Z_for_pca"].shape[0]), m=int(r["Z_for_pca"].shape[1]),
                   prune_params={"r2": 0.2, "window_bp": 500_000, "step_bp": 100_000}, spectrum_depth=20)
    assert d1["spectrum"].to_csv() == ref["spectrum"].to_csv()
    guigwas.allele_frequencies(state, "id-2", r["geno_imputed"])                     # another table
    guigwas.pc_diagnostics(state, "id-2", r["Z_for_pca"])
    assert calls == ["means", "pca", "means", "pca"]
    assert {guigwas.FREQ_KEY, guigwas.PC_DIAG_KEY} <= set(guigwas.SESSION_KEYS)       # Clear drops them
