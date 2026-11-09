"""Phenotype transformations are applied only on request (reviewer point 2-INT.7).

The GWAS page's "Phenotype QC and transformation" panel used to transform the trait it
previewed (rank-INT by default) for every run, even for users who never opened it.
Now the preview changes nothing; Apply records a method for a trait in
st.session_state["pheno_applied_transforms"] ({trait: CLI --norm slug}); each trait's
slug is that trait's normalisation, applied where the CLI applies --norm. A status line
above the panel says what is used.

Real AppTest sessions on the bundled example (Trait1, Trait2), compared with the CLI
itself (cli.run_pipeline, and gwas.pipeline.run_trait_pipeline for the phenotype
vector y): One-Click's ZIP for a trait equals the CLI's ZIP with the same options and
that trait's --norm, member for member, apart from the run's provenance (the input
file names and the "Pipeline" label in run_metadata.json and in the report's metadata
block) and the report's "Generated" time. The detailed view's y equals the CLI's y.
gwas.pipeline.run_traits is wrapped only to record what One-Click passes; it runs the
real analysis where a test needs the ZIPs.

Three sessions: the panel (A, B, then Apply: C, E, F), the list (D, G, the trait's own
samples, log10, an unselected trait) and the single-trait view (H, with subsampling).
The fixtures record and do not assert, so at the commit before the change the
untouched-panel tests (A, B, H's first run) fail on what the page computes; every test
that needs the Apply button errors there for its absence.
"""
import copy
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import cli  # noqa: E402
import gwas.pipeline as pl  # noqa: E402
import gwas.reports as reports  # noqa: E402
from gwas.io import load_vcf_cached  # noqa: E402
from pages._pheno_qc.stats import (  # noqa: E402
    TRANSFORM_INT,
    TRANSFORM_LOG10,
    TRANSFORM_NONE,
    TRANSFORM_YEOJOHNSON,
)

_PAGE = str(_ROOT / "pages" / "GWAS_analysis.py")
_VCF = _ROOT / "examples" / "example.vcf.gz"
_PHENO = _ROOT / "examples" / "example_pheno.csv"
RAW = "Phenotypes used: raw values for all traits"
STATE = "pheno_applied_transforms"
PANEL = "Phenotype QC and transformation"
APPLY, RESET = "gwas_pheno_qc_apply", "gwas_pheno_qc_reset_all"
# One-Click settings and the CLI arguments that match them (the rest are both defaults)
_PIPE = {"pipe_models": [], "pipe_subsampling": False, "pipe_generate_ld_heatmaps": True,
         "pipe_ld_r2": 0.1, "pipe_ld_flank_mode": "Manual (fixed kb)", "pipe_ld_flank": 200,
         "pipe_hap_perms": 100}
_CLI_ARGS = ["--ld-r2", "0.1", "--ld-flank-kb", "200", "--hap-perms", "100"]
_PROVENANCE = ("VCF", "Phenotype", "Pipeline")
_GENERATED = re.compile(rb"^[^\n]*<strong>Generated:</strong>[^\n]*\n", re.M)
_META_BLOCK = re.compile(rb'<div class="metadata">(.*?)</div>', re.S)
# Taken at import, before any AppTest runs, so the wrapper always calls the original.
_REAL_RUN_TRAITS = pl.run_traits


# ── helpers ────────────────────────────────────────────────────────────────

def _upload(table=None):
    """The phenotype as the page reads an uploaded file (sep sniffed, first row header)."""
    if table is None:
        return pd.read_csv(_PHENO, sep=None, engine="python", encoding="utf-8-sig")
    return table.copy()


def _extended():
    """The example plus 7 accessions that are not in the VCF (they must not change y)."""
    extra = pd.DataFrame({"SampleID": [f"NOT_GENOTYPED_{i}" for i in range(7)],
                          "Trait1": np.linspace(5.0, 11.0, 7), "Trait2": np.linspace(-9.0, -3.0, 7)})
    return pd.concat([_upload(), extra], ignore_index=True)


def _positive():
    """The example shifted to all-positive values (short decimals: exact in a CSV)."""
    t = _upload()
    t["Trait1"] = np.round(t["Trait1"] + 20.0, 6)
    t["Trait2"] = np.round(t["Trait2"] + 20.0, 6)
    return t


def _ok(at):
    # only AppTest's bare-mode page_link error (url_pathname) may occur
    assert all("url_pathname" in str(e.value) for e in at.exception), [e.value for e in at.exception]


def _status_lines(node):
    return [c.value for c in node.caption if c.value.startswith(("Phenotypes used:", "Applied:"))]


def _panel_status_lines(at):
    """Status lines inside the panel's expander (there must be none: it sits outside)."""
    return [line for e in at.expander if e.label == PANEL for line in _status_lines(e)]


def _applied(at):
    return dict(at.session_state[STATE]) if STATE in at.session_state else {}


def _y(at):
    return np.asarray(at.session_state["y"], dtype=float).ravel()


def _preview(at, trait, method):
    """Show (trait, method) in the panel; reruns only when it changes something."""
    if (at.selectbox(key="gwas_pheno_qc_trait").value, at.selectbox(key="gwas_pheno_qc_transform").value) \
            == (trait, method):
        return
    at.selectbox(key="gwas_pheno_qc_trait").set_value(trait)
    at.selectbox(key="gwas_pheno_qc_transform").set_value(method)
    at.run()
    _ok(at)


def _click(at, key):
    at.button(key=key).click().run()
    _ok(at)


def _apply(at, trait, method):
    _preview(at, trait, method)
    _click(at, APPLY)


def _select(at, traits):
    at.multiselect(key="selected_traits_multiselect").set_value(list(traits)).run()
    _ok(at)


class _Recorder:
    """Stands in for gwas.pipeline.run_traits: records what One-Click passes, and runs
    the real analysis (keeping each trait's ZIP) only when ``real`` is set."""

    def __init__(self):
        self.calls, self.real = [], False

    def __call__(self, parsed, pheno_df, traits, options, on_start=None, progress=None):
        call = {"traits": list(traits), "pheno": pheno_df.copy(), "options": copy.copy(options),
                "zips": {}, "failed": {}}
        self.calls.append(call)
        if not self.real:
            return iter(())
        return self._run(call, parsed, pheno_df, traits, options, on_start, progress)

    @staticmethod
    def _run(call, parsed, pheno_df, traits, options, on_start, progress):
        for trait, res, err in _REAL_RUN_TRAITS(parsed, pheno_df, traits, options,
                                                on_start=on_start, progress=progress):
            if res is None:
                call["failed"][trait] = err
            else:
                call["zips"][trait] = res.zip_bytes
            yield trait, res, err


def _one_click(at, rec, real):
    rec.real = real
    n = len(rec.calls)
    _click(at, "run_full_pipeline")
    rec.real = False
    assert len(rec.calls) == n + 1, "One-Click did not call gwas.pipeline.run_traits"
    return rec.calls[-1]


def _effective_norm(options, trait):
    """The normalisation run_trait_pipeline gives this trait."""
    return (getattr(options, "norm_by_trait", None) or {}).get(trait, options.norm)


def _session(rec, mp, traits=("Trait1",), table=None, extra=None):
    """A fresh GWAS page with the panel untouched (its preview opens on rank-INT)."""
    mp.setattr(pl, "run_traits", rec)
    st.cache_data.clear()
    at = AppTest.from_file(_PAGE, default_timeout=900)
    ss = at.session_state
    ss["_persist_vcf_bytes"] = _VCF.read_bytes()
    ss["_persist_is_gz"] = True
    ss["_persist_pheno"] = _upload(table)
    ss["selected_traits_multiselect"] = list(traits)
    for key, value in {**_PIPE, **(extra or {})}.items():
        ss[key] = value
    at.run()
    _ok(at)
    return at


_PARSED, _CLI = {}, {}


def _parsed():
    if "g" not in _PARSED:
        _PARSED["g"] = pl.ParsedGenotypes.from_callset(load_vcf_cached(_VCF.read_bytes()))
    return _PARSED["g"]


def _cli_table(table=None):
    """The phenotype as the CLI reads the file (cli.py: index_col=0)."""
    if table is None:
        return pd.read_csv(_PHENO, index_col=0, encoding="utf-8-sig")
    buf = io.StringIO()
    table.to_csv(buf, index=False)
    buf.seek(0)
    return pd.read_csv(buf, index_col=0, encoding="utf-8-sig")


def _cli(trait, norm, table=None, tag="example"):
    """The CLI's run for this trait and --norm (the shared pipeline), cached."""
    key = (trait, norm, tag)
    if key not in _CLI:
        res = pl.run_trait_pipeline(_parsed(), _cli_table(table), trait, pl.cli_default_options(
            vcf=str(_VCF), pheno=str(_PHENO), norm=norm, no_plots=True, no_report=True))
        _CLI[key] = {"y": np.asarray(res.context["y"], dtype=float).ravel(),
                     "mlm": res.tables[f"GWAS_{trait}.csv"]}
    return _CLI[key]


def _cli_y(trait, norm, table=None, tag="example"):
    return _cli(trait, norm, table, tag)["y"]


def _cli_zip(tmp, trait, norm):
    out = tmp / f"cli_{trait}_{norm}"
    cli.run_pipeline(cli._build_parser().parse_args(
        ["--vcf", str(_VCF), "--pheno", str(_PHENO), "--trait", trait, "--norm", norm,
         *_CLI_ARGS, "--output", str(out)]))
    zips = sorted(out.rglob("*.zip"))
    assert len(zips) == 1, zips
    return zips[0].read_bytes()


def _members(zip_bytes):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        return {n: zf.read(n) for n in zf.namelist()}


def _report_meta(html):
    (block,) = _META_BLOCK.findall(html)
    return json.loads(block.decode("utf-8"))


def _assert_matches_cli(gui_zip, cli_zip):
    """GUI ZIP == CLI ZIP: every member byte for byte, except the run's provenance."""
    gui, ref = _members(gui_zip), _members(cli_zip)
    assert sorted(gui) == sorted(ref)
    for name, data in gui.items():
        if name == "run_metadata.json":
            g, c = json.loads(data), json.loads(ref[name])
            assert g["Pipeline"] == "One-Click Full Analysis" and "Pipeline" not in c
            assert ({k: v for k, v in g.items() if k not in _PROVENANCE}
                    == {k: v for k, v in c.items() if k not in _PROVENANCE})
        elif name.endswith("report.html"):
            def _bare(html):
                return _META_BLOCK.sub(b"", _GENERATED.sub(b"", html))
            assert _bare(data) == _bare(ref[name])
            # the report's metadata block is the ZIP's run_metadata.json
            assert _report_meta(data) == json.loads(gui["run_metadata.json"])
        else:
            assert data == ref[name], name


def _normalization(zip_bytes):
    members = _members(zip_bytes)
    meta = json.loads(members["run_metadata.json"])
    assert _report_meta(members["report.html"])["Normalization"] == meta["Normalization"]
    return meta["Normalization"]


# ── session 1: the untouched panel (A), previews (B), then Apply (C, E, F) ─

@pytest.fixture(scope="module")
def cli_zips(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pheno_apply_cli")
    return {(t, n): _cli_zip(tmp, t, n)
            for t, n in (("Trait1", "none"), ("Trait1", "int"), ("Trait2", "none"))}


@pytest.fixture(scope="module")
def panel(cli_zips):
    rec, mp = _Recorder(), pytest.MonkeyPatch()
    try:
        at = _session(rec, mp)
        out = {"at": at, "rec": rec,
               "a": {"y": _y(at), "status": _status_lines(at), "applied": _applied(at),
                     "one_click": _one_click(at, rec, real=True)}}
        previews = []
        for trait, method in [("Trait2", TRANSFORM_YEOJOHNSON), ("Trait2", TRANSFORM_INT),
                              ("Trait1", TRANSFORM_YEOJOHNSON), ("Trait1", TRANSFORM_LOG10),
                              ("Trait1", TRANSFORM_INT)]:
            _preview(at, trait, method)
            previews.append({"preview": (trait, method), "y": _y(at),
                             "status": _status_lines(at), "applied": _applied(at)})
        _select(at, ["Trait1", "Trait2"])
        out["b"] = {"previews": previews, "stub": _one_click(at, rec, real=False)}
        yield out
    finally:
        mp.undo()
        st.cache_data.clear()


@pytest.fixture(scope="module")
def applied(panel):
    at, rec = panel["at"], panel["rec"]
    out = {"label": at.button(key=APPLY).label}       # the preview shows (Trait1, rank-INT)
    _click(at, APPLY)
    out.update(state=_applied(at), status=_status_lines(at), in_panel=_panel_status_lines(at),
               y=_y(at), remove_keys=[b.key for b in at.button
                                      if str(b.key).startswith("gwas_pheno_qc_remove__")])
    out["one_click"] = _one_click(at, rec, real=True)
    # F: preview another trait and method, without Apply; then Trait1 unselected
    _preview(at, "Trait2", TRANSFORM_YEOJOHNSON)
    out["switched"] = {"state": _applied(at), "status": _status_lines(at), "y": _y(at),
                       "stub": _one_click(at, rec, real=False)}
    _select(at, ["Trait2"])
    out["switched"].update(unselected_state=_applied(at), unselected_status=_status_lines(at),
                           y_trait2=_y(at))
    _select(at, ["Trait1"])
    out["switched"]["y_reselected"] = _y(at)
    return out


def test_untouched_panel_uses_raw_values_and_equals_cli_norm_none(panel, cli_zips):
    a = panel["a"]
    # the detailed view's phenotype (it used to be the CLI's --norm int y)
    assert np.array_equal(a["y"], _cli_y("Trait1", "none"))
    assert not np.array_equal(a["y"], _cli_y("Trait1", "int"))
    call = a["one_click"]
    pd.testing.assert_frame_equal(call["pheno"], _upload().set_index("SampleID"))
    assert _effective_norm(call["options"], "Trait1") == "none"
    assert not call["failed"]
    _assert_matches_cli(call["zips"]["Trait1"], cli_zips[("Trait1", "none")])
    assert _normalization(call["zips"]["Trait1"]) == "none"
    assert a["status"] == [RAW] and a["applied"] == {}


def test_preview_only_changes_nothing(panel):
    for snap in panel["b"]["previews"]:
        assert np.array_equal(snap["y"], _cli_y("Trait1", "none")), snap["preview"]
        assert snap["applied"] == {}, snap["preview"]
    call = panel["b"]["stub"]
    assert call["traits"] == ["Trait1", "Trait2"]
    pd.testing.assert_frame_equal(call["pheno"], _upload().set_index("SampleID"))
    assert [_effective_norm(call["options"], t) for t in call["traits"]] == ["none", "none"]
    assert all(snap["status"] == [RAW] for snap in panel["b"]["previews"])


def test_apply_records_the_method_and_names_it(applied):
    assert applied["label"] == "Apply rank-INT to Trait1"
    assert applied["state"] == {"Trait1": "int"}
    assert applied["status"] == ["Applied: Trait1 -> rank-INT - all other traits use raw values"]
    assert applied["in_panel"] == [], "the status line belongs outside the panel's expander"
    assert applied["remove_keys"] == ["gwas_pheno_qc_remove__Trait1"]
    assert np.array_equal(applied["y"], _cli_y("Trait1", "int"))


def test_applied_int_equals_cli_norm_int_and_other_traits_stay_raw(applied, cli_zips):
    call = applied["one_click"]
    assert call["traits"] == ["Trait1", "Trait2"] and not call["failed"]
    # the table stays the upload; the method reaches the pipeline as Trait1's --norm
    pd.testing.assert_frame_equal(call["pheno"], _upload().set_index("SampleID"))
    assert call["options"].norm_by_trait == {"Trait1": "int"} and call["options"].norm == "none"
    _assert_matches_cli(call["zips"]["Trait1"], cli_zips[("Trait1", "int")])
    _assert_matches_cli(call["zips"]["Trait2"], cli_zips[("Trait2", "none")])
    assert _normalization(call["zips"]["Trait1"]) == "int"
    assert _normalization(call["zips"]["Trait2"]) == "none"


def test_switching_the_preview_keeps_the_applied_trait(applied):
    sw = applied["switched"]
    line = ["Applied: Trait1 -> rank-INT - all other traits use raw values"]
    assert sw["state"] == {"Trait1": "int"} and sw["status"] == line
    assert np.array_equal(sw["y"], _cli_y("Trait1", "int"))
    assert sw["stub"]["options"].norm_by_trait == {"Trait1": "int"}
    assert [_effective_norm(sw["stub"]["options"], t) for t in ("Trait1", "Trait2")] == ["int", "none"]
    pd.testing.assert_frame_equal(sw["stub"]["pheno"], _upload().set_index("SampleID"))
    # an applied trait stays applied while it is not selected for GWAS
    assert sw["unselected_state"] == {"Trait1": "int"} and sw["unselected_status"] == line
    assert np.array_equal(sw["y_trait2"], _cli_y("Trait2", "none"))
    assert np.array_equal(sw["y_reselected"], _cli_y("Trait1", "int"))


# ── session 2: Remove, Reset all, refusals, an unselected trait, new uploads ─

@pytest.fixture(scope="module")
def listed():
    rec, mp = _Recorder(), pytest.MonkeyPatch()
    out = {}
    try:
        at = _session(rec, mp, traits=("Trait1", "Trait2"))
        _click(at, APPLY)                     # the preview opens on (Trait1, rank-INT)
        out["applied_y"] = _y(at)
        _click(at, "gwas_pheno_qc_remove__Trait1")
        out["removed"] = {"state": _applied(at), "status": _status_lines(at), "y": _y(at),
                          "remove_keys": [b.key for b in at.button
                                          if str(b.key).startswith("gwas_pheno_qc_remove__")]}
        _click(at, APPLY)
        _apply(at, "Trait2", TRANSFORM_YEOJOHNSON)
        out["two"] = {"state": _applied(at), "status": _status_lines(at)}
        _select(at, ["Trait2"])
        out["two"]["y_trait2"] = _y(at)
        _select(at, ["Trait1", "Trait2"])
        _click(at, RESET)
        out["reset"] = {"state": _applied(at), "status": _status_lines(at), "y": _y(at),
                        "reset_disabled": at.button(key=RESET).disabled,
                        "stub": _one_click(at, rec, real=False)}
        # a refused method, and None: Apply disabled with the reason shown
        _preview(at, "Trait1", TRANSFORM_LOG10)
        out["log_refused"] = {"disabled": at.button(key=APPLY).disabled,
                              "captions": [c.value for c in at.caption]}
        _click(at, APPLY)                     # AppTest fires a disabled button's callback
        out["log_refused"]["state"] = _applied(at)
        _preview(at, "Trait1", TRANSFORM_NONE)
        out["none"] = {"disabled": at.button(key=APPLY).disabled,
                       "label": at.button(key=APPLY).label,
                       "captions": [c.value for c in at.caption]}
        # Apply to a trait that is not selected for GWAS, then select it
        _select(at, ["Trait2"])
        _apply(at, "Trait1", TRANSFORM_INT)
        out["unselected"] = {"state": _applied(at), "y_trait2": _y(at)}
        _select(at, ["Trait1"])
        out["unselected"]["y_trait1"] = _y(at)
        # G: the same table again keeps the list; a new table clears it with a notice
        at.run()
        _ok(at)
        out["rerun"] = {"state": _applied(at), "infos": [i.value for i in at.info]}
        at.session_state["_persist_pheno"] = _extended()
        at.run()
        _ok(at)
        out["new"] = {"state": _applied(at), "status": _status_lines(at), "y": _y(at),
                      "infos": [i.value for i in at.info]}
        # the trait's own analysis samples: INT on an upload with non-genotyped rows
        _apply(at, "Trait1", TRANSFORM_INT)
        out["own_samples"] = {"y": _y(at), "stub": _one_click(at, rec, real=False)}
        # log10 applies to an all-positive trait (a new upload clears the list first)
        at.session_state["_persist_pheno"] = _positive()
        at.run()
        _ok(at)
        _preview(at, "Trait1", TRANSFORM_LOG10)
        out["log"] = {"label": at.button(key=APPLY).label}
        _click(at, APPLY)
        out["log"].update(state=_applied(at), status=_status_lines(at), y=_y(at))
    finally:
        mp.undo()
        st.cache_data.clear()
    return out


def test_remove_returns_the_raw_results(listed):
    assert np.array_equal(listed["applied_y"], _cli_y("Trait1", "int"))
    rm = listed["removed"]
    assert rm["state"] == {} and rm["status"] == [RAW] and rm["remove_keys"] == []
    assert np.array_equal(rm["y"], _cli_y("Trait1", "none"))


def test_two_traits_and_reset_all(listed):
    two = listed["two"]
    assert two["state"] == {"Trait1": "int", "Trait2": "yeojohnson"}
    assert two["status"] == ["Applied: Trait1 -> rank-INT; Trait2 -> Yeo-Johnson"]
    assert np.array_equal(two["y_trait2"], _cli_y("Trait2", "yeojohnson"))
    rs = listed["reset"]
    assert rs["state"] == {} and rs["status"] == [RAW] and rs["reset_disabled"]
    assert np.array_equal(rs["y"], _cli_y("Trait1", "none"))
    assert [_effective_norm(rs["stub"]["options"], t) for t in ("Trait1", "Trait2")] == ["none", "none"]


def test_apply_is_disabled_with_the_reason_when_refused(listed):
    log = listed["log_refused"]
    assert log["disabled"] and log["state"] == {}
    assert any(c.startswith("Apply is not available:") and "non-positive" in c
               for c in log["captions"])
    none = listed["none"]
    assert none["disabled"] and none["label"] == "Apply raw values to Trait1"
    assert any(c.startswith("Apply is not available:") for c in none["captions"])


def test_a_trait_applied_while_unselected_uses_it_when_selected(listed):
    un = listed["unselected"]
    assert un["state"] == {"Trait1": "int"}
    assert np.array_equal(un["y_trait2"], _cli_y("Trait2", "none"))
    assert np.array_equal(un["y_trait1"], _cli_y("Trait1", "int"))


def test_a_new_phenotype_file_clears_the_list_with_a_notice(listed):
    assert listed["rerun"]["state"] == {"Trait1": "int"}
    assert not any("cleared" in i for i in listed["rerun"]["infos"])
    new = listed["new"]
    assert new["state"] == {} and new["status"] == [RAW]
    assert any("new phenotype file" in i and "Trait1 -> rank-INT" in i for i in new["infos"])
    assert np.array_equal(new["y"], _cli_y("Trait1", "none", _extended(), tag="extended"))


def test_applied_int_uses_the_traits_analysis_samples(listed):
    own = listed["own_samples"]
    assert np.array_equal(own["y"], _cli_y("Trait1", "int", _extended(), tag="extended"))
    pd.testing.assert_frame_equal(own["stub"]["pheno"], _extended().set_index("SampleID"))
    assert _effective_norm(own["stub"]["options"], "Trait1") == "int"


def test_log10_applies_to_an_all_positive_trait(listed):
    log = listed["log"]
    assert log["label"] == "Apply log10 to Trait1"
    assert log["state"] == {"Trait1": "log"}
    assert log["status"] == ["Applied: Trait1 -> log10 - all other traits use raw values"]
    assert np.array_equal(log["y"], _cli_y("Trait1", "log", _positive(), tag="positive"))
    raw = _cli_y("Trait1", "none", _positive(), tag="positive")
    np.testing.assert_allclose(10.0 ** log["y"], raw, rtol=1e-12)   # base 10


# ── session 3: the single-trait view's records, results and subsampling (H) ─

_SUBSAMPLING = {"run_subsampling_gwas": True, "boot_n_reps": 10, "boot_n_jobs": 1}


def _run_view(at, reports_seen):
    n = len(reports_seen)
    [b for b in at.button if b.label == "Run GWAS"][0].click().run()
    _ok(at)
    assert len(reports_seen) == n + 1, "the run wrote no report"
    summ = at.session_state["gwas_run_summary"]
    members = _members(at.session_state[summ["zip_key"]]["data"])
    trait = summ["trait"]
    tables = {m: pd.read_csv(io.BytesIO(members[f"tables/{name}.csv"]))
              for m, name in (("MLM", f"GWAS_{trait}"), ("FarmCPU", f"GWAS_FarmCPU_{trait}"))}
    boot = [k for k in at.session_state.filtered_state
            if isinstance(k, str) and k.startswith(f"BOOT_GWAS::{trait}::")]
    return {"meta": json.loads(members["run_metadata.json"]), "report": reports_seen[-1],
            "tables": tables, "y": _y(at), "trait": trait,
            "boot_keys": boot,
            "boot_pvals": (np.array(at.session_state[boot[0]]["raw_pvals"], dtype=float, copy=True)
                           if len(boot) == 1 else None),
            "boot_summary": copy.deepcopy(at.session_state["subsampling_gwas_summary"])
            if "subsampling_gwas_summary" in at.session_state else None}


@pytest.fixture(scope="module")
def view_raw():
    rec, mp = _Recorder(), pytest.MonkeyPatch()
    reports_seen = []
    real_report = reports.generate_gwas_report

    def _report(*a, **k):
        html = real_report(*a, **k)
        reports_seen.append(html)
        return html

    try:
        mp.setattr(reports, "generate_gwas_report", _report)
        at = _session(rec, mp, extra=_SUBSAMPLING)   # sidebar default models: MLM + FarmCPU
        yield {"at": at, "reports": reports_seen, "raw": _run_view(at, reports_seen)}
    finally:
        mp.undo()
        st.cache_data.clear()


@pytest.fixture(scope="module")
def view(view_raw):
    at, seen = view_raw["at"], view_raw["reports"]
    out = {"raw": view_raw["raw"]}
    _click(at, APPLY)                           # the preview shows (Trait1, rank-INT)
    out["stale_notice"] = any("Current settings differ" in w.value for w in at.warning)
    out["int"] = _run_view(at, seen)
    _click(at, "gwas_pheno_qc_remove__Trait1")
    out["removed"] = _run_view(at, seen)
    # a second trait with its own method, run while the preview shows the first trait
    _select(at, ["Trait2"])
    _apply(at, "Trait2", TRANSFORM_YEOJOHNSON)
    _preview(at, "Trait1", TRANSFORM_INT)
    out["trait2"] = _run_view(at, seen)
    return out


def _records(run):
    m = run["meta"]
    return (m["Normalization method"], m["run_manifest"]["phenotype_model_normalization"],
            m["run_manifest"]["phenotype_preprocessing"]["transformations_section_2b"])


def test_untouched_view_records_raw_values(view_raw):
    raw = view_raw["raw"]
    assert _records(raw) == ("none", "none", {})
    assert '"Normalization method": "none"' in raw["report"]
    assert np.array_equal(raw["y"], _cli_y("Trait1", "none"))


def test_view_records_only_the_applied_transformation(view):
    assert view["stale_notice"], "Apply after a run must mark the shown results stale"
    assert _records(view["int"]) == ("int", "int", {"Trait1": "int"})
    assert '"Normalization method": "int"' in view["int"]["report"]
    assert _records(view["removed"]) == ("none", "none", {})
    assert view["trait2"]["trait"] == "Trait2"
    assert _records(view["trait2"]) == ("yeojohnson", "yeojohnson", {"Trait2": "yeojohnson"})


def test_view_results_follow_the_applied_transformation(view):
    """The view's own table differs in layout from the CLI's (no Significant_Meff or
    Model column), and its p-values agree to rounding, so they are compared to rtol
    1e-12 and the phenotype vector bitwise."""
    runs = ((view["raw"], "none"), (view["int"], "int"), (view["removed"], "none"),
            (view["trait2"], "yeojohnson"))
    for run, norm in runs:
        ref = _cli(run["trait"], norm)
        assert np.array_equal(run["y"], ref["y"]), (run["trait"], norm)
        pipe = ref["mlm"].set_index("SNP")["PValue"]
        got = run["tables"]["MLM"].set_index("SNP")["PValue"]
        assert sorted(got.index) == sorted(pipe.index)
        np.testing.assert_allclose(got.loc[pipe.index].to_numpy(), pipe.to_numpy(), rtol=1e-12)
    raw, applied_, removed = view["raw"], view["int"], view["removed"]
    # FarmCPU's cache is keyed by strings: it must not serve the raw run's results
    assert not np.allclose(applied_["tables"]["FarmCPU"]["PValue"], raw["tables"]["FarmCPU"]["PValue"])
    for model in ("MLM", "FarmCPU"):
        pd.testing.assert_frame_equal(removed["tables"][model], raw["tables"][model])


def test_view_subsampling_follows_the_applied_transformation(view):
    """Subsampling (10 reps) is recomputed for the applied method and again after Remove
    (one cached entry per trait), and each run's manifest carries its own summary."""
    raw, applied_, removed, trait2 = view["raw"], view["int"], view["removed"], view["trait2"]
    for run in (raw, applied_, removed, trait2):
        assert len(run["boot_keys"]) == 1 and run["boot_summary"] is not None
        assert run["meta"]["run_manifest"]["subsampling_gwas_resampling"] == run["boot_summary"]
    assert not np.allclose(applied_["boot_pvals"], raw["boot_pvals"])
    np.testing.assert_array_equal(removed["boot_pvals"], raw["boot_pvals"])
