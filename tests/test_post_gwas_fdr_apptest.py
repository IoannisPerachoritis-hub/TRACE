"""Post-GWAS under the FDR rule with MLM as the downstream model (reviewer point
2-INT.11, bug 2).

The single-trait view hands MLM's table to the Post-GWAS page with SNP, Chr, ChrNum,
Pos and PValue only. Under the FDR rule there was no Significant_FDR column to judge,
so the Significant SNPs tab showed "Could not build the significant-SNP table" and the
lead-SNP gallery raised, stopping the page. gwas.significance.frame_for_rule now adds
FDR and Significant_FDR (Benjamini-Hochberg within the table's own p-values, as
add_significance_columns writes them).

A real AppTest run of the single-trait view on the planted data (tests/_multitrait_data,
raw phenotype, FDR rule, default models MLM + FarmCPU, so MLM is downstream), then the
Post-GWAS page seeded from that session: the page raises nothing, the tab lists exactly
the view's own MLM FDR SNPs, and the gallery offers them.
"""
import copy
import io
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

from tests import _gui_inputs as _gi  # noqa: E402
from tests import _multitrait_data as mt  # noqa: E402

# over about 30 s: run once per batch, not per task (-m 'not slow_gui')
pytestmark = pytest.mark.slow_gui

_PAGE = str(_ROOT / "pages" / "GWAS_analysis.py")
_POST = str(_ROOT / "pages" / "Post_GWAS_Analysis.py")
TRAIT = "T_chr1"
FDR_LABEL = "FDR (q < 0.05)"


@pytest.fixture(scope="module")
def session():
    vcf_text, pheno, _info = mt.build()
    st.cache_data.clear()
    at = AppTest.from_file(_PAGE, default_timeout=900)
    ss = at.session_state
    ss["_persist_vcf_path"] = _gi.vcf_file(vcf_text)
    ss["_persist_pheno"] = pheno
    ss["selected_traits_multiselect"] = [TRAIT]
    ss["gwas_pheno_qc_transform"] = "None (raw values)"
    ss["sig_rule_select"] = FDR_LABEL
    at.run()
    [b for b in at.button if b.label == "Run GWAS"][0].click().run()
    # only AppTest's bare-mode page_link error (url_pathname) may occur, after the results
    assert all("url_pathname" in str(e.value) for e in at.exception), [e.value for e in at.exception]
    state = dict(at.session_state.filtered_state)
    for key in ("gwas_results", "gwas_df", "gwas_results_by_model", "gwas_run_summary"):
        state[key] = copy.deepcopy(state[key])
    assert state["downstream_model"] == "MLM" and state["sig_rule"] == FDR_LABEL
    with zipfile.ZipFile(io.BytesIO(state[state["gwas_run_summary"]["zip_key"]]["data"])) as zf:
        mlm = pd.read_csv(io.BytesIO(zf.read(f"tables/GWAS_{TRAIT}.csv")))
    expected = set(mlm.loc[mlm["Significant_FDR"].astype(bool), "SNP"].astype(str))
    assert expected and expected != set(mlm.loc[mlm["Significant_Bonf"].astype(bool), "SNP"].astype(str))
    st.cache_data.clear()
    return {"state": state, "expected": expected}


@pytest.fixture(scope="module")
def post(session):
    st.cache_data.clear()
    pg = AppTest.from_file(_POST, default_timeout=900)
    for k, v in session["state"].items():
        try:
            pg.session_state[k] = v
        except Exception:  # noqa: BLE001 - widget-owned keys cannot be seeded
            pass
    pg.run()
    st.cache_data.clear()
    return pg


def test_the_hand_off_table_is_the_views_mlm_table_without_significance_columns(session):
    """The GWAS page's hand-off: the MLM's p-values and its effects (per ALT allele copy),
    without the significance columns."""
    assert list(session["state"]["gwas_df"].columns) == [
        "SNP", "Chr", "ChrNum", "Pos", "PValue", "Beta_MLM", "SE_MLM", "Beta_OLS", "SE_OLS",
        "Model"]


def test_the_post_gwas_page_raises_nothing(post):
    assert not post.exception, [e.value for e in post.exception]


def test_significant_snps_tab_lists_the_views_mlm_fdr_snps(session, post):
    assert not any("significant-SNP table" in e.value for e in post.error), [e.value for e in post.error]
    tables = [d.value for d in post.dataframe
              if hasattr(d.value, "columns") and "Sig_Threshold" in d.value.columns]
    assert tables, "no significant-SNP table on the Post-GWAS page"
    sig = tables[0]
    assert set(sig["SNP"].astype(str)) == session["expected"]
    assert set(sig["Sig_Rule"]) == {"fdr"} and sig["Sig_Threshold"].isna().all()


def test_lead_snp_gallery_offers_the_views_mlm_fdr_snps(session, post):
    pick = [s for s in post.selectbox if s.key == "lead_gallery_snp"]
    assert pick, "the lead-SNP gallery did not render"
    assert {str(o).split()[0] for o in pick[0].options} == session["expected"]
