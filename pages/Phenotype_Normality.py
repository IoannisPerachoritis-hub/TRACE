"""
Phenotype Normality & Distribution Check — standalone in-app QC tool.

Upload a phenotype table and test each trait for normality (Shapiro-Wilk,
D'Agostino K², Anderson-Darling + skew/kurtosis), inspect histogram and Q-Q
diagnostics, and preview/download the pipeline's normalising transforms
(log10, Yeo-Johnson, rank-INT). Fully standalone — no GWAS run required and
nothing is written back into the GWAS session.
"""

import streamlit as st
from utils.pub_theme import apply_matplotlib_theme, build_plotly_template

apply_matplotlib_theme()
build_plotly_template()

st.set_page_config(page_title="Phenotype Normality", layout="wide")
st.title("Phenotype Normality & Distribution Check")
st.markdown(
    "Upload a phenotype table to check whether each trait is normally "
    "distributed. Non-normal traits often benefit from a transform before "
    "GWAS (the mixed model assumes approximately normal residuals). This page "
    "reports normality tests, shows histogram and Q-Q diagnostics, and lets "
    "you preview and download normalising transforms — it does **not** change "
    "any GWAS session data."
)

from pages._pheno_qc import data_load, diagnostics, summary  # noqa: E402

# ── Phase 1: Load ─────────────────────────────────────────────
ctx = data_load.render()
if ctx is None:
    st.stop()

# ── Phase 2: Summary table ────────────────────────────────────
st.divider()
summary.render(ctx)

# ── Phase 3: Per-trait diagnostics & transforms ───────────────
st.divider()
diagnostics.render(ctx)
