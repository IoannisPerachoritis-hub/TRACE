"""AppTest coverage for the GWAS page (GUI).

The page is a large Streamlit script with no unit-testable seams; these tests
drive it via streamlit.testing.v1.AppTest, seeding session_state to exercise the
run-from-memory path (AppTest cannot perform real file uploads). The deep display
block hits an AppTest bare-mode limitation (a `url_pathname` KeyError) that does
NOT occur in the real app; the scan + result columns run before it, so tests
assert on the produced state and tolerate only that specific harness error.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402
import pytest  # noqa: E402

_PAGE = str(_ROOT / "pages" / "GWAS_analysis.py")


def _synth(n=50, per=40, chroms=("1", "2", "3"), seed=0):
    rng = np.random.default_rng(seed)
    samples = [f"S{i:03d}" for i in range(n)]
    lines = ["##fileformat=VCFv4.2",
             '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">',
             "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples)]
    for ch in chroms:
        for j in range(per):
            gt = ["0/0" if g == 0 else "0/1" if g == 1 else "1/1"
                  for g in rng.binomial(2, 0.3, n)]
            lines.append(f"{ch}\t{(j+1)*1000}\tchr{ch}_{j}\tA\tT\t.\tPASS\t.\tGT\t" + "\t".join(gt))
    vcf_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    pheno = pd.DataFrame({"Accession": samples,
                          "TestTrait": np.random.default_rng(seed + 1).normal(10, 2, n)})
    return vcf_bytes, pheno, samples


def _seed_from_memory(at, run=False, extra=None):
    """Seed the persisted inputs so the page enters the run block from memory."""
    vcf_bytes, pheno, samples = _synth()
    at.session_state["_persist_vcf_bytes"] = vcf_bytes
    at.session_state["_persist_is_gz"] = False
    at.session_state["_persist_pheno"] = pheno
    at.session_state["selected_traits_multiselect"] = ["TestTrait"]
    for k, v in (extra or {}).items():
        at.session_state[k] = v
    if run:
        at.session_state["gwas_triggered"] = True
        at.session_state["_last_gwas_trait"] = "TestTrait"
    return samples


def _only_harness_error(at):
    return all("url_pathname" in str(e.value) for e in at.exception)


def test_page_loads_cold():
    at = AppTest.from_file(_PAGE, default_timeout=120)
    at.run()
    assert not at.exception


@pytest.mark.slow_gui
def test_run_from_memory_produces_results():
    """T-09 regression: with the uploaded file gone but inputs persisted, a Run
    proceeds from memory and yields gwas_df (no re-upload)."""
    at = AppTest.from_file(_PAGE, default_timeout=300)
    _seed_from_memory(at, run=True)
    at.run()
    assert _only_harness_error(at)
    assert "gwas_df" in at.session_state and len(at.session_state["gwas_df"]) > 0
    # under the default (Bonferroni) rule FarmCPU, on by default, carries the same
    # significance columns as this view's MLM: no M_eff column, which needs the M_eff rule
    fc = at.session_state["gwas_results_by_model"]["TestTrait"]["FarmCPU"]
    assert [c for c in fc.columns if c == "FDR" or c.startswith("Significant_")] == [
        "FDR", "Significant_FDR", "Significant_Bonf"]


@pytest.mark.slow_gui
def test_covariate_from_memory_runs():
    """A covariate (seeded via _persist_covar) is aligned + applied in a
    from-memory run without crashing. Covariate correctness is covered by the
    unit tests (test_covariates, run_gwas_cached user_covar)."""
    at = AppTest.from_file(_PAGE, default_timeout=300)
    samples = _seed_from_memory(at, run=True)
    rng = np.random.default_rng(7)
    at.session_state["_persist_covar"] = pd.DataFrame(
        {"batch": rng.normal(0, 1, len(samples))}, index=samples
    )
    at.run()
    assert _only_harness_error(at)
    assert "gwas_df" in at.session_state and len(at.session_state["gwas_df"]) > 0


def test_clear_session_data_releases_memory():
    at = AppTest.from_file(_PAGE, default_timeout=200)
    _seed_from_memory(at)
    at.run()
    assert "_persist_vcf_bytes" in at.session_state
    btns = [b for b in at.button if "Clear session data" in b.label]
    assert btns, "Clear session data button not found"
    btns[0].click().run()
    assert "_persist_vcf_bytes" not in at.session_state
    assert "_persist_pheno" not in at.session_state


@pytest.mark.slow_gui
def test_custom_sig_thresh_runs_from_memory():
    """The Custom-p-value path (selectbox -> number_input -> Significant_Custom
    column -> active threshold) runs end-to-end without a real error. Column
    correctness is covered by the significance unit tests (rule_from_streamlit /
    SignificanceRule custom)."""
    at = AppTest.from_file(_PAGE, default_timeout=300)
    _seed_from_memory(at, run=True, extra={
        "sig_rule_select": "Custom p-value",
        "custom_sig_thresh_input": 5e-8,
    })
    at.run()
    assert _only_harness_error(at)
    assert "gwas_df" in at.session_state and len(at.session_state["gwas_df"]) > 0
    # the rule + custom threshold are propagated to downstream pages (LD tab)
    assert at.session_state["sig_rule"] == "Custom p-value"
    assert float(at.session_state["custom_sig_thresh"]) == 5e-8
    # FarmCPU (on by default) carries the custom column too. This noise trait has no
    # SNP below 5e-8; the values are checked on planted data in
    # test_detailed_view_significance_apptest.
    fc = at.session_state["gwas_results_by_model"]["TestTrait"]["FarmCPU"]
    assert [c for c in fc.columns if c == "FDR" or c.startswith("Significant_")] == [
        "FDR", "Significant_FDR", "Significant_Bonf", "Significant_Custom"]


def test_covar_persist_decision():
    """Commit 1 §1a: the covariate-source decision. AppTest cannot hold a live upload
    (vcf_file is always None there), so the removable-covariate branch is exercised by
    this pure helper: live session (running_from_memory=False) + a persisted covariate +
    an empty uploader -> 'clear' (the user removed it); rehydrate -> 'session' (keep)."""
    from pages.GWAS_analysis import _covar_persist_decision as d
    assert d(True, False, False) == "upload"      # a file is present
    assert d(True, True, True) == "upload"         # file present wins regardless
    assert d(False, True, True) == "session"       # rehydrate + persisted -> keep
    assert d(False, False, True) == "clear"        # live + uploader emptied -> remove
    assert d(False, True, False) == "none"         # nothing persisted
    assert d(False, False, False) == "none"


@pytest.mark.slow_gui
def test_stale_scan_notice():
    """Commit 2: after a completed run, an input change without re-clicking Run
    re-displays the last results WITH a staleness notice; an unchanged rerun shows none."""
    at = AppTest.from_file(_PAGE, default_timeout=300)
    samples = _seed_from_memory(at, run=True)   # RunA computes -> stores gwas_run_summary.scan_fp
    at.run()
    assert "gwas_df" in at.session_state and len(at.session_state["gwas_df"]) > 0
    assert "gwas_run_summary" in at.session_state
    assert "scan_fp" in at.session_state["gwas_run_summary"]
    _msg = "Current settings differ"
    # RunB: no input change, no Run-click -> guard fires, fingerprint matches -> NO notice
    at.run()
    assert not any(_msg in w.value for w in at.warning)
    # RunC: add a covariate (changes the fingerprint) without a Run-click -> stale notice
    rng = np.random.default_rng(3)
    at.session_state["_persist_covar"] = pd.DataFrame(
        {"batch": rng.normal(0, 1, len(samples))}, index=samples
    )
    at.run()
    assert any(_msg in w.value for w in at.warning)


@pytest.mark.slow_gui
def test_meff_failure_writes_no_meff_column_for_any_model(monkeypatch):
    """When M_eff cannot be computed the page warns and falls back to Bonferroni, and, as
    at 7062830 for MLM, writes no Significant_Meff column: for MLM and for FarmCPU (on by
    default) alike."""
    import gwas.plotting as gp

    def _fail(geno, block=None):
        raise RuntimeError("forced M_eff failure")

    monkeypatch.setattr(gp, "compute_meff_li_ji", _fail)
    monkeypatch.setattr(gp, "_compute_meff_li_ji_impl", _fail)   # what the shared pipeline calls
    at = AppTest.from_file(_PAGE, default_timeout=300)
    _seed_from_memory(at, run=True, extra={"sig_rule_select": "M_eff: Li & Ji (LD-aware Bonferroni)"})
    at.run()
    assert _only_harness_error(at)
    assert any("M_eff computation failed" in w.value for w in at.warning)
    by_model = at.session_state["gwas_results_by_model"]["TestTrait"]
    assert [c for c in by_model["FarmCPU"].columns if c == "FDR" or c.startswith("Significant_")] == [
        "FDR", "Significant_FDR", "Significant_Bonf"]
    import io
    import zipfile
    zips = [v for k, v in at.session_state.filtered_state.items()
            if k.startswith("gwas_zip_") and isinstance(v, dict) and "data" in v]
    assert len(zips) == 1
    with zipfile.ZipFile(io.BytesIO(zips[0]["data"])) as zf:
        mlm = pd.read_csv(io.BytesIO(zf.read("tables/GWAS_TestTrait.csv")))
    assert "Significant_Meff" not in mlm.columns and "Significant_Bonf" in mlm.columns
