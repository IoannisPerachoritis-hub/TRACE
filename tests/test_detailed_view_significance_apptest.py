"""The GWAS page's single-trait run gives MLMM and FarmCPU the pipeline's significance
columns and values (reviewer point 2-INT.6a, criterion D).

Real AppTest runs of the single-trait view with MLM, MLMM and FarmCPU on the planted
data (tests/_multitrait_data), under the M_eff rule and under a custom p-value, with raw
phenotype values (the phenotype-QC panel's preview no longer transforms the GWAS
phenotype; its preview is set to raw values all the same). For the same inputs each model's
table in the view's ZIP equals the pipeline's (gwas.pipeline.run_trait_pipeline, as the
CLI and One-Click run it): the same p-values and FDR, and the same Significant_* columns
and values. The view writes Significant_Meff only under the M_eff rule, for every model
alike, as it always did for MLM. Both rules run in one session, M_eff first: the second
run must rebuild the view's ZIP for the new rule (it used to serve the first run's). M_eff
is set to 5 (threshold 0.01) in the view and in the pipeline, because on the planted data
the real M_eff selects the same SNPs as Bonferroni. Under the custom rule, the Post-GWAS
page's Significant SNPs tab, given the view's MLMM table as the downstream table, applies
the tab's own custom p-value rather than the run's (the table now carries
Significant_Custom).
"""
import copy
import io
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import gwas.pipeline as pl  # noqa: E402
import gwas.plotting as gp  # noqa: E402
from gwas.io import load_vcf_cached  # noqa: E402
from tests import _multitrait_data as mt  # noqa: E402

# over about 30 s: run once per batch, not per task (-m 'not slow_gui')
pytestmark = pytest.mark.slow_gui

_PAGE = str(_ROOT / "pages" / "GWAS_analysis.py")
_POST = str(_ROOT / "pages" / "Post_GWAS_Analysis.py")
TRAIT = "T_chr1"
MODELS = ("MLM", "MLMM", "FarmCPU")
RUN_CUSTOM, TAB_CUSTOM = 5e-8, 1e-6
RULES = {
    "meff": {"label": "M_eff: Li & Ji (LD-aware Bonferroni)", "cli": "meff",
             "columns": ["FDR", "Significant_FDR", "Significant_Bonf", "Significant_Meff"]},
    "custom": {"label": "Custom p-value", "cli": RUN_CUSTOM,
               "columns": ["FDR", "Significant_FDR", "Significant_Bonf", "Significant_Custom"]},
}


def _sig_cols(df):
    return [c for c in df.columns if c == "FDR" or c.startswith("Significant_")]


def _collect(at):
    """The view's ZIP tables and session state after a run (one ZIP for the trait)."""
    # only AppTest's bare-mode page_link error (url_pathname) may occur, after the results
    assert all("url_pathname" in str(e.value) for e in at.exception), [e.value for e in at.exception]
    state = dict(at.session_state.filtered_state)
    # the next run in this session updates these in place: keep this run's copies
    for key in ("sig_rule", "custom_sig_thresh", "meff_val", "gwas_run_summary",
                "gwas_results_by_model", "gwas_results", "gwas_df", "downstream_model"):
        if key in state:
            state[key] = copy.deepcopy(state[key])
    assert not state.get("pheno_applied_transforms"), "a transformation was applied to the view's trait"
    zips = {k: v for k, v in state.items()
            if k.startswith("gwas_zip_") and isinstance(v, dict) and "data" in v}
    assert len(zips) == 1, sorted(zips)
    (zip_key, zip_entry), = zips.items()
    assert state["gwas_run_summary"]["zip_key"] == zip_key     # what the re-display offers
    with zipfile.ZipFile(io.BytesIO(zip_entry["data"])) as zf:
        gui = {m: pd.read_csv(io.BytesIO(zf.read(f"tables/{name}.csv")))
               for m, name in zip(MODELS, (f"GWAS_{TRAIT}", f"GWAS_MLMM_{TRAIT}", f"GWAS_FarmCPU_{TRAIT}"))}
    return gui, state


@pytest.fixture(scope="module")
def views():
    vcf_text, pheno, _info = mt.build()
    out = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gp, "compute_meff_li_ji", lambda geno: (5, np.array([])))
        st.cache_data.clear()
        at = AppTest.from_file(_PAGE, default_timeout=900)
        ss = at.session_state
        ss["_persist_vcf_bytes"] = vcf_text.encode()
        ss["_persist_is_gz"] = False
        ss["_persist_pheno"] = pheno
        ss["selected_traits_multiselect"] = [TRAIT]
        ss["gwas_pheno_qc_transform"] = "None (raw values)"   # the preview only; nothing is applied
        ss["sig_rule_select"] = RULES["meff"]["label"]
        at.run()
        models = [m for m in at.sidebar.multiselect if m.label == "Additional GWAS models"]
        models[0].set_value(["MLMM (iterative cofactors)", "FarmCPU (multi-locus)"])
        at.run()
        [b for b in at.button if b.label == "Run GWAS"][0].click().run()
        out["meff"] = _collect(at)
        # same session: switch the rule and run again
        at.selectbox(key="sig_rule_select").set_value(RULES["custom"]["label"]).run()
        at.number_input(key="custom_sig_thresh_input").set_value(RUN_CUSTOM).run()
        [b for b in at.button if b.label == "Run GWAS"][0].click().run()
        out["custom"] = _collect(at)
        st.cache_data.clear()
        parsed = pl.ParsedGenotypes.from_callset(load_vcf_cached(vcf_text.encode()))
        for key in ("meff", "custom"):
            res = pl.run_trait_pipeline(
                parsed, pheno.set_index("Accession"), TRAIT,
                pl.cli_default_options(vcf="planted.vcf", pheno="planted.csv", sig_thresh=RULES[key]["cli"],
                                       model=["mlm", "mlmm", "farmcpu"], no_plots=True))
            pipe = {m: res.tables[n] for m, n in zip(MODELS, (f"GWAS_{TRAIT}.csv", f"GWAS_MLMM_{TRAIT}.csv",
                                                              f"GWAS_FarmCPU_{TRAIT}.csv"))}
            gui, state = out[key]
            out[key] = {"rule": key, "gui": gui, "pipe": pipe, "summary": res.summary, "state": state}
    return out


@pytest.fixture(params=sorted(RULES))
def run(request, views):
    return views[request.param]


def test_the_second_run_rebuilt_the_zip_for_the_new_rule(views):
    """Same trait, models, SNP count and lambda, so the same ZIP key: the second run's ZIP
    must hold the custom rule's columns, not the M_eff run's (a cache keyed without the
    rule served the first run's tables)."""
    for model in MODELS:
        first, second = _sig_cols(views["meff"]["gui"][model]), _sig_cols(views["custom"]["gui"][model])
        assert "Significant_Meff" in first and "Significant_Custom" not in first, (model, first)
        assert "Significant_Custom" in second and "Significant_Meff" not in second, (model, second)


def test_each_model_has_the_rule_columns_and_the_pipelines_values(run):
    cols = RULES[run["rule"]]["columns"]
    for model in MODELS:
        g = run["gui"][model].set_index("SNP")
        p = run["pipe"][model].set_index("SNP")
        assert _sig_cols(g) == cols, model
        assert set(cols) <= set(p.columns), model     # the pipeline also writes Significant_Meff
        assert sorted(g.index) == sorted(p.index), model
        g = g.loc[p.index]
        np.testing.assert_allclose(g["PValue"], p["PValue"], rtol=1e-12, atol=0, err_msg=model)
        np.testing.assert_allclose(g["FDR"], p["FDR"], rtol=1e-12, atol=0, err_msg=model)
        for col in cols[1:]:
            assert g[col].astype(bool).equals(p[col].astype(bool)), (model, col)
        assert g[cols[-1]].any(), f"{model}: nothing passes {cols[-1]}, the check is vacuous"


def test_the_view_uses_mlms_thresholds_for_every_model(run):
    bonf = 0.05 / run["summary"]["n_snps"]
    assert len(run["gui"]["MLM"]) == run["summary"]["n_snps"]
    if run["rule"] == "meff":
        assert run["state"]["meff_val"] == run["summary"]["meff"]
    for model in MODELS:
        df = run["gui"][model]
        p = df["PValue"].to_numpy(dtype=float)
        assert np.array_equal(df["Significant_Bonf"].to_numpy(), p < bonf), model
        if run["rule"] == "meff":
            assert run["summary"]["meff"] == 5
            assert np.array_equal(df["Significant_Meff"].to_numpy(),
                                  p < 0.05 / run["summary"]["meff"]), model
            assert (df["Significant_Meff"] != df["Significant_Bonf"]).any(), model
        else:
            assert np.array_equal(df["Significant_Custom"].to_numpy(), p < RUN_CUSTOM), model
        reject, q, _, _ = multipletests(p, method="fdr_bh")
        assert np.array_equal(df["Significant_FDR"].to_numpy(), reject), model
        np.testing.assert_allclose(df["FDR"].to_numpy(), q, rtol=1e-12, atol=0)
    # the per-model tables the downstream selector picks from carry the same columns
    for model in ("MLMM", "FarmCPU"):
        assert _sig_cols(run["state"]["gwas_results_by_model"][TRAIT][model]) == RULES[run["rule"]]["columns"]


def test_post_gwas_significant_snps_tab_on_an_mlmm_downstream_table(run):
    """M_eff: the tab (on the page's rule) lists exactly the MLMM table's M_eff SNPs.
    Custom: the tab's own custom p-value decides, not the run's Significant_Custom."""
    state = dict(run["state"])
    mlmm = state["gwas_results_by_model"][TRAIT]["MLMM"].copy()   # what the downstream radio hands over
    assert RULES[run["rule"]]["columns"][-1] in mlmm.columns   # Significant_Meff / _Custom
    state["gwas_results"] = {TRAIT: mlmm}
    state["gwas_df"] = mlmm.copy()
    state["downstream_model"] = "MLMM"
    st.cache_data.clear()
    pg = AppTest.from_file(_POST, default_timeout=900)
    for k, v in state.items():
        try:
            pg.session_state[k] = v
        except Exception:  # noqa: BLE001 - widget-owned keys cannot be seeded
            pass
    pg.run()
    p = mlmm["PValue"].to_numpy(dtype=float)
    if run["rule"] == "custom":
        tab_input = [w for w in pg.number_input if w.key == "sigtab_custom_thresh"]
        assert tab_input, "the Significant SNPs tab did not render its custom p-value"
        tab_input[0].set_value(TAB_CUSTOM).run()
        assert (p < TAB_CUSTOM).sum() > (p < RUN_CUSTOM).sum(), "the two thresholds select the same SNPs"
        expected, threshold = p < TAB_CUSTOM, TAB_CUSTOM
    else:
        threshold = 0.05 / run["summary"]["meff"]
        expected = mlmm["Significant_Meff"].to_numpy(dtype=bool)
        assert np.array_equal(expected, p < threshold)
    assert not pg.exception, [e.value for e in pg.exception]
    tables = [d.value for d in pg.dataframe
              if hasattr(d.value, "columns") and "Sig_Threshold" in d.value.columns]
    assert tables, "no significant-SNP table on the Post-GWAS page"
    sig = tables[0]
    assert expected.any()
    assert sorted(sig["SNP"].astype(str)) == sorted(mlmm.loc[expected, "SNP"].astype(str))
    np.testing.assert_allclose(sorted(set(sig["Sig_Threshold"])), [threshold], rtol=1e-12)
    st.cache_data.clear()
