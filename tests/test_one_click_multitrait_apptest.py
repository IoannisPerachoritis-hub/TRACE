"""One-Click Full Analysis runs several traits (reviewer point 2-INT.2).

Replaces the one-trait-per-run guard of 822610f. Until then the One-Click loop
resolved its genotypes and phenotype reader through session keys registered once
from the first selected trait, so every later trait was analysed with the first
trait's data. Now each trait runs through gwas.pipeline.run_trait_pipeline, the
CLI's per-trait code, with the GUI's options and its own phenotyped samples.

Planted causal SNPs, one per trait on a different chromosome (tests/_multitrait_data),
in a real AppTest run (nothing stubbed; gwas.pipeline.run_traits and
gwas.plotting._combine_gwas_zips are wrapped only to record what they produce):
A. each trait's ZIP equals a single-trait run with the same options, byte for byte
   apart from the report's "Generated" line (and equals a real single-trait One-Click
   run for the trait with missing samples); its top hit is on the planted chromosome;
B. traits with different missing samples are each analysed on their own samples;
C. a constant trait (selected first, so it also cannot be the detailed view's trait)
   fails without stopping the others; the page shows its error, and the combined ZIP
   holds the completed traits plus failed_traits.txt.
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import gwas.pipeline as _pl  # noqa: E402
import gwas.plotting as _gp  # noqa: E402
from tests import _gui_inputs as _gi  # noqa: E402
from tests import _multitrait_data as mt  # noqa: E402

# over about 30 s: run once per batch, not per task (-m 'not slow_gui')
pytestmark = pytest.mark.slow_gui

_PAGE = str(_ROOT / "pages" / "GWAS_analysis.py")
_GENERATED = b"<strong>Generated:</strong>"
_PLANTED = list(mt.CAUSAL)
# Taken at import, before any AppTest runs, so the wrappers always call the originals.
_REAL_RUN_TRAITS = _pl.run_traits
_REAL_COMBINE = _gp._combine_gwas_zips


def _strip(zip_bytes):
    """ZIP member -> bytes, with the report's 'Generated:' time line dropped."""
    out = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            data = zf.read(name)
            if name.endswith("report.html"):
                data = b"\n".join(line for line in data.split(b"\n") if _GENERATED not in line)
            out[name] = data
    return out


def _one_click(traits, rec):
    vcf_text, pheno, _info = mt.build()
    st.cache_data.clear()
    at = AppTest.from_file(_PAGE, default_timeout=900)
    at.session_state["_persist_vcf_path"] = _gi.vcf_file(vcf_text)
    at.session_state["_persist_pheno"] = pheno
    at.session_state["selected_traits_multiselect"] = list(traits)
    # all three models, so every model's p-values are compared
    at.session_state["pipe_models"] = ["MLMM (iterative cofactors)", "FarmCPU (multi-locus)"]
    at.session_state["pipe_subsampling"] = False
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    n_before = len(rec["runs"])
    at.button(key="run_full_pipeline").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(rec["runs"]) == n_before + 1, "One-Click did not call gwas.pipeline.run_traits"
    return at, rec["runs"][-1]


@pytest.fixture(scope="module")
def gui():
    rec = {"runs": [], "combined": []}

    def _run_traits(parsed, pheno_df, traits, options, on_start=None, progress=None):
        call = {"traits": list(traits), "options": options, "parsed": parsed,
                "pheno": pheno_df, "zips": {}, "failed": {}}
        rec["runs"].append(call)
        for trait, res, err in _REAL_RUN_TRAITS(parsed, pheno_df, traits, options,
                                                on_start=on_start, progress=progress):
            if res is None:
                call["failed"][trait] = err
            else:
                call["zips"][trait] = res.zip_bytes
            yield trait, res, err

    def _combine(per_trait_zips, pheno_label=None, extra_members=None):
        name, buf = _REAL_COMBINE(per_trait_zips, pheno_label=pheno_label,
                                  extra_members=extra_members)
        rec["combined"].append((name, buf.getvalue()))
        buf.seek(0)
        return name, buf

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_pl, "run_traits", _run_traits)
        mp.setattr(_gp, "_combine_gwas_zips", _combine)
        multi_at, multi = _one_click([mt.CONSTANT_TRAIT, *_PLANTED], rec)
        n_combined = len(rec["combined"])
        single_at, single = _one_click(["T_chr2"], rec)
        single_combined = rec["combined"][n_combined:]
    st.cache_data.clear()
    _vcf, _pheno, info = mt.build()
    return {"multi_at": multi_at, "multi": multi, "combined": rec["combined"][n_combined - 1],
            "single_at": single_at, "single": single, "single_combined": single_combined,
            "info": info}


def test_each_trait_matches_a_single_trait_run_with_the_same_options(gui):
    run = gui["multi"]
    assert run["traits"] == [mt.CONSTANT_TRAIT, *_PLANTED]
    for t in _PLANTED:
        alone = _pl.run_trait_pipeline(run["parsed"], run["pheno"], t, run["options"])
        got = _strip(run["zips"][t])
        assert got == _strip(alone.zip_bytes), t
        for table in (f"GWAS_{t}", f"GWAS_MLMM_{t}", f"GWAS_FarmCPU_{t}"):
            assert f"tables/{table}.csv" in got, table


def test_single_trait_one_click_matches_its_multi_trait_run(gui):
    # T_chr2 lacks 10 samples, so a mixed-up sample set would show here
    assert gui["single"]["traits"] == ["T_chr2"] and not gui["single"]["failed"]
    assert _strip(gui["single"]["zips"]["T_chr2"]) == _strip(gui["multi"]["zips"]["T_chr2"])
    assert not gui["single_combined"], "a single-trait run must not build a combined ZIP"


def test_each_trait_uses_its_own_samples_and_finds_its_own_causal_chromosome(gui):
    for t in _PLANTED:
        with zipfile.ZipFile(io.BytesIO(gui["multi"]["zips"][t])) as zf:
            gwas = pd.read_csv(io.BytesIO(zf.read(f"tables/GWAS_{t}.csv")))
            meta = json.loads(zf.read("run_metadata.json"))
        assert meta["Trait"] == t
        assert meta["Samples"] == gui["info"][t]["n"], t
        assert str(gwas.sort_values("PValue").iloc[0]["Chr"]) == gui["info"][t]["chrom"], t


def test_a_failing_trait_is_reported_and_does_not_stop_the_others(gui):
    run = gui["multi"]
    assert set(run["failed"]) == {mt.CONSTANT_TRAIT}
    assert set(run["zips"]) == set(_PLANTED)
    errors = " ".join(e.value for e in gui["multi_at"].error)
    assert mt.CONSTANT_TRAIT in errors and "near-zero variance" in errors
    name, combined = gui["combined"]
    assert "MultiTrait_3traits" in name
    with zipfile.ZipFile(io.BytesIO(combined)) as zf:
        names = zf.namelist()
        report = zf.read("failed_traits.txt").decode()
        for t in _PLANTED:
            mine = {n.split("/", 1)[1]: zf.read(n) for n in names if n.startswith(f"{t}/")}
            assert mine == {n: d for n, d in _members(run["zips"][t]).items()}, t
    assert not any(n.startswith(f"{mt.CONSTANT_TRAIT}/") for n in names)
    assert f"{mt.CONSTANT_TRAIT}: FAILED" in report and "near-zero variance" in report
    summary = gui["multi_at"].session_state["one_click_results"]
    assert set(summary["completed"]) == set(_PLANTED) and set(summary["failed"]) == {mt.CONSTANT_TRAIT}


def test_detailed_view_falls_back_past_a_trait_that_fails_qc(gui):
    warnings = " ".join(w.value for w in gui["multi_at"].warning)
    assert mt.CONSTANT_TRAIT in warnings and "T_chr1" in warnings


def test_gui_widgets_reach_the_pipeline_options(monkeypatch):
    """The One-Click model, PC, significance, LD, haplotype, heatmap and subsampling
    settings map onto the options the pipeline reads, each set to a non-default value
    (run_traits is stubbed: this checks the mapping, not the analysis)."""
    seen = []

    def _record(parsed, pheno_df, traits, options, on_start=None, progress=None):
        seen.append(options)
        return iter(())

    monkeypatch.setattr(_pl, "run_traits", _record)
    vcf_text, pheno, _info = mt.build()
    st.cache_data.clear()
    at = AppTest.from_file(_PAGE, default_timeout=900)
    at.session_state["_persist_vcf_path"] = _gi.vcf_file(vcf_text)
    at.session_state["_persist_vcf_name"] = "planted.vcf"
    at.session_state["_persist_pheno"] = pheno
    at.session_state["selected_traits_multiselect"] = ["T_chr1"]
    for key, value in {
        "pipe_models": ["MLMM (iterative cofactors)", "FarmCPU (multi-locus)"],
        "pipe_pcs_mlm": 2, "pipe_pcs_mlmm": 1, "pipe_pcs_farmcpu": 3,
        "pipe_use_loco": False,
        "pipe_sig_rule": "M_eff: Li & Ji (LD-aware Bonferroni)",
        "pipe_show_secondary_threshold": True,
        "pipe_ld_r2": 0.5, "pipe_ld_seed": "Significant only (chosen threshold)",
        "pipe_ld_flank_mode": "Manual (fixed kb)", "pipe_ld_flank": 200,
        "pipe_hap_perms": 500, "pipe_generate_ld_heatmaps": False,
        "pipe_hap_test": "PCs and covariates only (the earlier test)",
        "pipe_mlmm_p": 1e-3, "pipe_mlmm_max_cof": 5, "pipe_fc_max_iter": 20,
        "pipe_subsampling": True, "pipe_boot_reps": 20, "pipe_boot_frac": 0.7,
        "pipe_boot_thresh": 1e-3, "pipe_boot_loco": True,
    }.items():
        at.session_state[key] = value
    at.run()
    at.button(key="run_full_pipeline").click().run()
    st.cache_data.clear()
    assert not at.exception, [e.value for e in at.exception]
    assert len(seen) == 1
    o = seen[0]
    expected = {
        "model": ["mlm", "mlmm", "farmcpu"], "n_pcs_mlm": 2, "n_pcs_mlmm": 1,
        "n_pcs_farmcpu": 3, "no_loco": True, "sig_thresh": "meff",
        "secondary_threshold": True, "ld_r2": 0.5, "ld_seed_mode": "significant",
        "ld_seed_p": 1e-5, "ld_top_n": 10,          # the CLI's own values in this mode
        "ld_flank_kb": 200, "hap_perms": 500, "no_ld_heatmaps": True, "hap_test": "pcs",
        "mlmm_p_enter": 1e-3, "mlmm_max_cof": 5, "farmcpu_max_iter": 20,
        "subsampling": True, "boot_reps": 20, "boot_frac": 0.7, "boot_thresh": 1e-3,
        "boot_use_loco": True, "norm": "none",
        "norm_by_trait": {},                        # no transformation applied
        "run_label": "One-Click Full Analysis", "vcf": "planted.vcf", "output": None,
    }
    got = {k: getattr(o, k, "<missing>") for k in expected}
    assert got == expected
    # the QC settings left at their defaults equal the CLI's defaults
    cli_defaults = _pl.cli_default_options()
    for k in ("maf", "miss", "ind_miss", "mac", "info_thresh", "impute", "seed"):
        assert getattr(o, k) == getattr(cli_defaults, k), k


def _members(zip_bytes):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        return {n: zf.read(n) for n in zf.namelist()}
