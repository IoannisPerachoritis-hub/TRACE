"""gwas.pipeline: the per-trait pipeline shared by the CLI and One-Click.

The GUI-only options are read with getattr, so the CLI (which never sets them) is
unchanged. These tests run each one through the real pipeline (wrapping the library
call it feeds where its effect is not in the outputs), plus run_traits' memory
release and gene-model sharing, the log level under concurrent runs, and the CLI's
per-trait isolation of a ZIP that cannot be written.
"""
import gc
import io
import json
import weakref
import zipfile

import pandas as pd
import pytest

import gwas.pipeline as pl
from cli import _build_parser, run_pipeline
from gwas.io import load_vcf_cached
from tests import _multitrait_data as mt


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    base = tmp_path_factory.mktemp("trait_pipeline")
    vcf, pheno_csv, info = mt.write(base)
    parsed = pl.ParsedGenotypes.from_callset(load_vcf_cached(vcf.read_bytes()))
    pheno = pd.read_csv(pheno_csv, index_col=0)
    return {"base": base, "vcf": vcf, "pheno_csv": pheno_csv, "parsed": parsed,
            "pheno": pheno, "info": info}


def _opts(p, **kw):
    return pl.cli_default_options(vcf=str(p["vcf"]), pheno=str(p["pheno_csv"]), **kw)


def _zip(res):
    with zipfile.ZipFile(io.BytesIO(res.zip_bytes)) as zf:
        return {n: zf.read(n) for n in zf.namelist()}


def test_parsed_genotypes_are_read_only(planted):
    with pytest.raises(ValueError):
        planted["parsed"].genotypes[0, 0] = 1


def test_cli_default_options_are_the_parser_defaults():
    assert vars(pl.cli_default_options()) == vars(_build_parser().parse_args([]))
    assert pl.cli_default_options(ld_r2=0.3).ld_r2 == 0.3


def test_no_ld_heatmaps_drops_only_the_heatmaps(planted):
    with_hm = _zip(pl.run_trait_pipeline(planted["parsed"], planted["pheno"], "T_chr1", _opts(planted)))
    without = _zip(pl.run_trait_pipeline(planted["parsed"], planted["pheno"], "T_chr1",
                                         _opts(planted, no_ld_heatmaps=True)))
    hm = {n for n in with_hm if "LD_heatmap_" in n}
    assert hm and not any("LD_heatmap_" in n for n in without)
    assert set(with_hm) - hm == set(without)


def test_covariate_frame_matches_the_covariate_file(planted, tmp_path):
    ids = planted["pheno"].index.astype(str)
    cov = pd.DataFrame({"Accession": ids, "Batch": [(i % 3) * 0.5 for i in range(len(ids))]})
    path = tmp_path / "cov.csv"
    cov.to_csv(path, index=False)
    from gwas.covariates import load_covariate_frame, select_covariate_columns
    frame = select_covariate_columns(load_covariate_frame(path))
    by_file = pl.run_trait_pipeline(planted["parsed"], planted["pheno"], "T_chr1",
                                    _opts(planted, covar=str(path), no_plots=True))
    # as the GUI passes it: a label (no readable path) plus the table it already read
    by_frame = pl.run_trait_pipeline(planted["parsed"], planted["pheno"], "T_chr1",
                                     _opts(planted, covar="covariates", covar_frame=frame,
                                           no_plots=True))
    none = pl.run_trait_pipeline(planted["parsed"], planted["pheno"], "T_chr1",
                                 _opts(planted, no_plots=True))
    gwas = "GWAS_T_chr1.csv"
    assert by_file.tables[gwas].equals(by_frame.tables[gwas])
    assert not by_frame.tables[gwas]["PValue"].equals(none.tables[gwas]["PValue"])


def test_refuse_policy_stops_a_trait_with_covariate_missing_samples(planted, tmp_path):
    ids = list(planted["pheno"].index.astype(str))
    batch = [(i % 3) * 0.5 for i in range(len(ids) - 3)] + [None] * 3
    cov = pd.DataFrame({"Accession": ids, "Batch": batch})
    path = tmp_path / "cov_missing.csv"
    cov.to_csv(path, index=False)
    with pytest.raises(pl.TraitPipelineError, match="3 of 120 analysis samples without covariate"):
        pl.run_trait_pipeline(planted["parsed"], planted["pheno"], "T_chr1",
                              _opts(planted, covar=str(path), covar_missing="refuse", no_plots=True))
    dropped = pl.run_trait_pipeline(planted["parsed"], planted["pheno"], "T_chr1",
                                    _opts(planted, covar=str(path), no_plots=True))  # CLI default
    assert dropped.summary["n_samples"] == 117


def test_gui_provenance_is_recorded_in_the_metadata(planted):
    res = pl.run_trait_pipeline(
        planted["parsed"], planted["pheno"], "T_chr1",
        _opts(planted, no_plots=True, run_label="One-Click Full Analysis",
              pheno_transform={"method": "Rank-based INT", "columns": ["T_chr1"]}))
    meta = json.loads(_zip(res)["run_metadata.json"])
    assert meta["Pipeline"] == "One-Click Full Analysis"
    assert meta["Normalization"].startswith("Rank-based INT")
    other = pl.run_trait_pipeline(
        planted["parsed"], planted["pheno"], "T_chr2",
        _opts(planted, no_plots=True,
              pheno_transform={"method": "Rank-based INT", "columns": ["T_chr1"]}))
    assert json.loads(_zip(other)["run_metadata.json"])["Normalization"] == "none"


def test_trait_log_lines_reach_progress_even_when_the_logger_is_at_warning(planted):
    import logging
    seen = []
    cli_log = logging.getLogger("cli")
    old = cli_log.level
    cli_log.setLevel(logging.WARNING)  # as under Streamlit, where nothing sets INFO
    try:
        res = pl.run_trait_pipeline(planted["parsed"], planted["pheno"], "T_chr1",
                                    _opts(planted, no_plots=True), progress=seen.append)
        level_after = cli_log.level
    finally:
        cli_log.setLevel(old)
    assert any("Phenotype QC" in m for m in seen)
    assert any("Phenotype QC" in line for line in res.log)
    assert level_after == logging.WARNING, "the pipeline must restore the logger's level"


def test_info_level_is_held_until_the_last_overlapping_run_ends():
    import logging
    cli_log = logging.getLogger("cli")
    old = cli_log.level
    cli_log.setLevel(logging.WARNING)
    try:
        pl._info_level_enter()          # session A
        pl._info_level_enter()          # session B overlaps
        pl._info_level_exit()           # A ends first
        mid = cli_log.getEffectiveLevel()
        pl._info_level_exit()           # B ends
        end = cli_log.level
    finally:
        cli_log.setLevel(old)
    assert mid == logging.INFO and end == logging.WARNING


def _recorder(monkeypatch, module, name):
    """Wrap module.name so each call's keyword arguments are recorded."""
    real = getattr(module, name)
    calls = []

    def _wrapped(*a, **k):
        calls.append(k)
        return real(*a, **k)

    monkeypatch.setattr(module, name, _wrapped)
    return calls


def test_gui_model_settings_reach_mlmm_and_farmcpu(planted, monkeypatch):
    import gwas.models as gm
    mlmm = _recorder(monkeypatch, gm, "run_mlmm_research_grade_fast")
    fc = _recorder(monkeypatch, gm, "run_farmcpu")
    pl.run_trait_pipeline(planted["parsed"], planted["pheno"], "T_chr1",
                          _opts(planted, model=["mlm", "mlmm", "farmcpu"], no_plots=True,
                                mlmm_p_enter=1e-3, mlmm_max_cof=4, farmcpu_max_iter=7))
    assert mlmm[-1]["p_enter"] == 1e-3 and mlmm[-1]["max_cof"] == 4
    assert fc[-1]["max_iterations"] == 7
    pl.run_trait_pipeline(planted["parsed"], planted["pheno"], "T_chr1",
                          _opts(planted, model=["mlm", "mlmm", "farmcpu"], no_plots=True))
    assert "p_enter" not in mlmm[-1] and "max_iterations" not in fc[-1]  # CLI: library defaults


def test_secondary_threshold_line_reaches_the_manhattan_plots(planted, monkeypatch):
    import gwas.plotting as gp
    calls = _recorder(monkeypatch, gp, "plot_manhattan_static")
    pl.run_trait_pipeline(planted["parsed"], planted["pheno"], "T_chr1",
                          _opts(planted, sig_thresh="meff", secondary_threshold=True))
    assert calls and calls[-1]["secondary_label"] == "Bonferroni α=0.05"
    n = len(calls)
    pl.run_trait_pipeline(planted["parsed"], planted["pheno"], "T_chr1", _opts(planted))
    assert all("secondary_lod" not in k for k in calls[n:])


def test_subsampling_loco_setting_reaches_the_resampler(planted, monkeypatch):
    import gwas.subsampling as gs
    calls = _recorder(monkeypatch, gs, "subsample_gwas_resampling")
    for use_loco in (False, True):
        pl.run_trait_pipeline(planted["parsed"], planted["pheno"], "T_chr1",
                              _opts(planted, no_plots=True, subsampling=True, boot_reps=2,
                                    boot_use_loco=use_loco))
        assert calls[-1]["use_loco"] is use_loco


def test_a_batch_loads_its_gene_model_once_and_names_the_upload(planted, monkeypatch):
    import annotation
    loads = _recorder(monkeypatch, annotation, "load_gene_annotation")
    _gm_res = annotation.resolve_gene_model("tomato", genome_build="SL3", override=None)
    opts = _opts(planted, no_plots=True, gene_model=str(_gm_res.gene_model),
                 gene_model_label="my_genes.csv")
    results = [r for _t, r, _e in pl.run_traits(planted["parsed"], planted["pheno"],
                                                ["T_chr1", "T_chr2"], opts)]
    assert len(loads) == 1 and all(r is not None for r in results)
    meta = json.loads(_zip(results[0])["run_metadata.json"])
    assert meta["Gene model"] == "my_genes.csv"
    assert "my_genes.csv" in meta["Gene model status"]
    assert str(_gm_res.gene_model) not in meta["Gene model status"]


def test_run_traits_frees_each_result_before_the_next_trait(monkeypatch):
    refs = []

    class _Res:
        zip_bytes = b""

    def _fake(parsed, pheno_df, trait, options, progress=None):
        r = _Res()
        refs.append(weakref.ref(r))
        return r

    alive_at_start = []
    monkeypatch.setattr(pl, "run_trait_pipeline", _fake)

    def _on_start(i, trait):
        gc.collect()
        alive_at_start.append([ref() is not None for ref in refs])

    for _trait, res, _err in pl.run_traits(None, pd.DataFrame(), ["a", "b", "c"],
                                           pl.cli_default_options(), on_start=_on_start):
        del res
    assert alive_at_start == [[], [False], [False, False]]


def test_a_zip_that_cannot_be_written_fails_only_that_trait(planted, monkeypatch):
    import pathlib
    real = pathlib.Path.write_bytes

    def _flaky(self, data):
        if "__T_chr2__" in self.name:
            raise OSError(28, "No space left on device")
        return real(self, data)

    monkeypatch.setattr(pathlib.Path, "write_bytes", _flaky)
    out = planted["base"] / "flaky"
    args = _build_parser().parse_args(["--vcf", str(planted["vcf"]), "--pheno", str(planted["pheno_csv"]),
                                       "--trait", "T_chr1", "T_chr2", "--output", str(out), "--no-plots"])
    with pytest.raises(SystemExit) as ei:
        run_pipeline(args)
    assert "1 of 2 traits failed (T_chr2); the others completed" in str(ei.value.code)
    assert list(out.glob("GWAS_*__T_chr1__*.zip"))
    combined = list(out.glob("GWAS_*__MultiTrait_1traits__*.zip"))
    assert len(combined) == 1
    with zipfile.ZipFile(combined[0]) as zf:
        assert "No space left on device" in zf.read("failed_traits.txt").decode()
