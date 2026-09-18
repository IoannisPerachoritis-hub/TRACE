"""Tab 3: Gene Annotation for LD Blocks."""

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
        summarize_gene_model,
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
            help="SL3 = SL3.1 gene models (default); SL4 = ITAG4.0. Varitome SNPs are in SL2.5, so "
                 "gene coordinates in either build are offset from the SNP positions (SL3 by ~0.5 Mb, "
                 "SL4 more); annotation is positional, not coordinate-exact.",
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

    with st.expander(
        "Upload a gene model (for a non-tomato species or a custom build)",
        expanded=(_ld_species == "Other (upload files)"),
    ):
        _ld_gm_ov = st.file_uploader(
            "Gene coordinates CSV (columns: Chr, Start, End, [Strand,] Gene_ID)",
            type=["csv", "tsv", "txt"],
            key="gene_model_upload",
            help=(
                "Overrides the bundled gene model if provided. Accepted column "
                "aliases: CHROM/Chr/chr, START/Start/start_pos, END/End/end_pos, "
                "GENE/Gene_ID/name/gene_name. Coordinates must be in the SAME "
                "assembly as your VCF (check the per-chromosome ranges shown "
                "after upload). For a new species, derive this from your GFF3 "
                "with a short pandas script. See "
                "[docs/gene_model_upload.md](https://github.com/IoannisPerachoritis-hub/TRACE/blob/main/docs/gene_model_upload.md)."
            ),
        )
        _ld_desc_ov = st.file_uploader(
            "Gene descriptions TSV (optional: gene_id \\t description)",
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
            _gm_summary = summarize_gene_model(genes_df)
            st.success(
                f"Loaded {len(genes_df):,} genes across "
                f"{len(_gm_summary)} chromosomes. Confirm the coordinate ranges "
                f"below are in the same assembly as your VCF."
            )
            st.dataframe(_gm_summary, use_container_width=True)
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
        # Containment is structurally impossible after the disjoint-block redesign
        # (WO4); invariant check only (never a silent drop) -- blocks pass through.
        _cf, _ = filter_contained_blocks(
            ctx.haplo_df_auto.copy(), min_contained=2,
            size_ratio_threshold=3.0, mode="flag")
        if "is_mega_block" in _cf.columns and bool(_cf["is_mega_block"].any()):
            logging.getLogger(__name__).warning(
                "LD containment detected (should be impossible post-occupancy): %s",
                [f"{r['Chr']}:{int(r['Start (bp)'])}-{int(r['End (bp)'])}"
                 for _, r in _cf[_cf["is_mega_block"]].iterrows()])
        blocks_for_annot = ctx.haplo_df_auto

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
