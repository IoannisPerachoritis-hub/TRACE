"""Phase 1 — load a phenotype table and build the PhenoQCContext."""

import numpy as np
import streamlit as st

from . import PhenoQCContext, load_phenotype_file


def _build_context(pheno, id_col, source):
    """Validate numeric traits and wrap a loaded DataFrame in a context."""
    numeric_cols = pheno.select_dtypes(include=[np.number]).columns.tolist()
    n_total = pheno.shape[1]

    st.caption(f"ID column: **{id_col}**. First 10 IDs: {pheno.index[:10].tolist()}")

    if not numeric_cols:
        st.error(
            "No numeric trait columns detected. Check that your phenotype file "
            "contains numeric measurements (a header row, an ID column, and one "
            "or more numeric trait columns)."
        )
        return None
    if len(numeric_cols) < n_total:
        st.caption(
            f"{len(numeric_cols)} numeric trait column(s) detected; "
            f"{n_total - len(numeric_cols)} non-numeric column(s) will be skipped."
        )

    with st.expander("Data preview", expanded=True):
        st.dataframe(pheno.head())

    return PhenoQCContext(
        pheno_df=pheno,
        numeric_cols=numeric_cols,
        id_col=id_col,
        source=source,
    )


def render():
    """Return a PhenoQCContext from an upload or a session-loaded phenotype.

    Returns None (and shows a prompt) until a usable phenotype is available.
    """
    st.header("1 · Load phenotype data")
    st.markdown(
        "Upload a **phenotype table** (CSV / TSV / TXT). The separator is "
        "auto-detected. The first row must be a header; there must be an ID "
        "column (named e.g. `Accession`, `Sample`, `IID`, or the first column "
        "is used) and one or more **numeric** trait columns. Missing values "
        "may be empty cells or `NA` / `NaN`."
    )

    have_session = "pheno_raw" in st.session_state
    source_choice = "Upload a file"
    if have_session:
        source_choice = st.radio(
            "Phenotype source",
            ["Upload a file", "Use phenotype already loaded from a GWAS run"],
            horizontal=True,
        )

    if have_session and source_choice.startswith("Use phenotype"):
        pheno = st.session_state["pheno_raw"].copy()
        st.success("Using the raw phenotype loaded in this session.")
        return _build_context(pheno, id_col=pheno.index.name or "index", source="session")

    uploaded = st.file_uploader(
        "Upload phenotype file", type=["csv", "txt", "tsv"], key="pheno_qc_upload"
    )
    if uploaded is None:
        st.info("Upload a phenotype file to begin.")
        return None

    try:
        pheno, id_col, latin1_used = load_phenotype_file(uploaded)
    except ValueError as err:
        st.error(str(err))
        return None

    if latin1_used:
        st.warning(
            "Phenotype file is not UTF-8 (detected Latin-1/Windows-1252). "
            "Consider re-saving as UTF-8 for best compatibility."
        )

    return _build_context(pheno, id_col=id_col, source="upload")
