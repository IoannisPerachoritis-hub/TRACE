"""Tab 3 — Gene Annotation for LD Blocks."""

import logging
import pandas as pd
import streamlit as st

from gwas.ld import filter_contained_blocks
from . import LDContext


def render(ctx: LDContext):
    st.subheader("Automatic Gene Annotation for LD Blocks")

    if not ctx.has_annotation:
        st.error(
            "annotation.py module not found.\n"
            "Place it in your `ld/` folder or project root."
        )
        return

    from annotation import (
        load_gene_annotation,
        annotate_ld_blocks,
        format_annotation_summary,
    )

    st.markdown(
        "Annotate LD blocks with overlapping and flanking genes. "
        "Gene model and descriptions auto-load when a species is selected."
    )

    # Auto-load gene model + descriptions by species
    from utils.species_files import SPECIES_FILES as _SPECIES_FILES
    _ld_species_opts = list(_SPECIES_FILES.keys()) + ["Other (upload files)"]
    _ld_species = st.selectbox("Species", _ld_species_opts, key="ld_gene_species")
    _ld_auto = _SPECIES_FILES.get(_ld_species, {})

    _ld_has_build = "gene_model_SL3" in _ld_auto
    if _ld_has_build:
        _ld_build = st.selectbox(
            "Genome build", ["SL3", "SL4"], key="ld_genome_build",
            help="SL3 = SL3.1 gene models (matches Varitome / SL2.5 SNP coordinates). SL4 = ITAG4.0.",
        )
        _ld_gm_auto = _ld_auto.get(f"gene_model_{_ld_build}")
        _ld_desc_auto = _ld_auto.get(f"gene_desc_{_ld_build}")
    else:
        _ld_gm_auto = _ld_auto.get("gene_model")
        _ld_desc_auto = _ld_auto.get("gene_desc")

    gene_file = None
    desc_file = None

    if _ld_gm_auto and _ld_gm_auto.exists():
        st.write(f"Gene model: `{_ld_gm_auto.name}`")
        gene_file = _ld_gm_auto
    if _ld_desc_auto and _ld_desc_auto.exists():
        st.write(f"Gene descriptions: `{_ld_desc_auto.name}`")
        desc_file = _ld_desc_auto

    with st.expander("Override gene files"):
        _ld_gm_ov = st.file_uploader(
            "Gene coordinates (Sol_genes.csv)",
            type=["csv", "tsv", "txt"],
            key="gene_model_upload",
        )
        _ld_desc_ov = st.file_uploader(
            "Gene descriptions (SL3.1 or ITAG4.0)",
            type=["txt", "tsv"],
            key="gene_desc_upload",
        )
        if _ld_gm_ov is not None:
            gene_file = _ld_gm_ov
        if _ld_desc_ov is not None:
            desc_file = _ld_desc_ov

    if gene_file is not None:
        from pathlib import Path as _LdPath
        import tempfile, os

        # Handle both Path (auto-loaded) and UploadedFile
        if isinstance(gene_file, _LdPath):
            gene_tmp = str(gene_file)
        else:
            gene_tmp = os.path.join(tempfile.gettempdir(), "sol_genes_upload.csv")
            with open(gene_tmp, "wb") as f:
                f.write(gene_file.getbuffer())

        desc_tmp = None
        if desc_file is not None:
            if isinstance(desc_file, _LdPath):
                desc_tmp = str(desc_file)
            else:
                desc_tmp = os.path.join(tempfile.gettempdir(), "itag4_desc_upload.txt")
                with open(desc_tmp, "wb") as f:
                    f.write(desc_file.getbuffer())

        try:
            genes_df = load_gene_annotation(gene_tmp, desc_tmp)
            st.session_state["genes_df"] = genes_df
            st.write(
                f"Loaded {len(genes_df):,} genes across "
                f"{genes_df['Chr'].nunique()} chromosomes."
            )
        except Exception as e:
            logging.exception("Gene file loading failed")
            st.error(f"Error loading gene file: {e}")
            genes_df = None
    else:
        genes_df = st.session_state.get("genes_df", None)

    if genes_df is None:
        return

    st.markdown("#### Preview: gene model")
    st.dataframe(genes_df.head(10), use_container_width=True)

    # Which LD blocks to annotate?
    blocks_for_annot = None
    _use_hap_gwas_blocks = st.checkbox(
        "Use Haplotype GWAS blocks instead",
        value=False,
        key="annot_use_hap_gwas",
    )

    if _use_hap_gwas_blocks:
        blocks_for_annot = st.session_state.get("hap_gwas_df", None)
    else:
        blocks_for_annot, _ = filter_contained_blocks(
            ctx.haplo_df_auto,
            min_contained=int(st.session_state.get("mega_min_contained", 2)),
            size_ratio_threshold=float(st.session_state.get("mega_size_ratio", 3.0)),
            mode="remove",
        )

    if blocks_for_annot is None or (
            isinstance(blocks_for_annot, pd.DataFrame) and blocks_for_annot.empty):
        st.info("No LD blocks available from the selected source. Run detection first.")
        return

    n_flank = st.slider(
        "Flanking genes to report (per side)",
        min_value=1, max_value=5, value=2,
        help="For intergenic blocks: how many genes upstream/downstream to report.",
    )

    max_flank_kb = st.slider(
        "Max flanking distance (kb)",
        min_value=50, max_value=2000, value=500, step=50,
    )

    if st.button("Annotate LD blocks", key="btn_annotate"):
        with st.spinner("Annotating LD blocks with gene models..."):
            annotated = annotate_ld_blocks(
                blocks_for_annot,
                genes_df,
                n_flank=n_flank,
                max_flank_dist_bp=max_flank_kb * 1000,
            )

            st.session_state["annotated_ld_blocks"] = annotated

    annotated = st.session_state.get("annotated_ld_blocks", None)

    if annotated is None or annotated.empty:
        return

    # Summary table
    summary = format_annotation_summary(annotated)

    st.markdown("#### Annotation Summary (paper-ready)")
    st.dataframe(summary, use_container_width=True)

    # Status breakdown
    status_counts = annotated["annotation_status"].value_counts()
    st.markdown("**Block annotation breakdown:**")
    for status, count in status_counts.items():
        st.write(f"  {status}: {count} blocks")

    st.markdown("#### Full annotation table")
    st.dataframe(annotated, use_container_width=True)

    # Downloads
    st.download_button(
        "Download annotated LD blocks (CSV)",
        annotated.to_csv(index=False).encode("utf-8"),
        file_name=f"LD_blocks_annotated__{ctx.trait_col}.csv",
        mime="text/csv",
        key="dl_annotated_blocks",
    )

    st.download_button(
        "Download annotation summary (CSV)",
        summary.to_csv(index=False).encode("utf-8"),
        file_name=f"LD_annotation_summary__{ctx.trait_col}.csv",
        mime="text/csv",
        key="dl_annot_summary",
    )
