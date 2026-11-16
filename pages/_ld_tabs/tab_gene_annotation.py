"""Tab: Gene Annotation for LD blocks and unblocked significant SNPs, in two steps.

Step 1 loads the gene models: the bundled model for the species and build, or the user's
own gene file and, optionally, a descriptions file. The "?" of each uploader is
annotation.GENE_FILE_HELP / DESCRIPTIONS_FILE_HELP, the same texts as the GWAS page's
One-Click overrides. Step 2 lists every candidate interval of the run on screen (each LD
block and each run of unblocked significant SNPs, by the CLI's rule:
pages._ld_tabs._intervals) with the genes inside it, in one table and one CSV.
"""

import logging
import os
import tempfile
from pathlib import Path

import streamlit as st

from . import LDContext
from ._intervals import run_candidate_intervals

_BUILD_HELP = (
    "SL3 = SL3.1 gene models (default); SL4 = ITAG4.0. Varitome SNPs are in SL2.5, so "
    "gene coordinates in either build are offset from the SNP positions (SL3 by ~0.5 Mb, "
    "SL4 more); annotation is positional, not coordinate-exact."
)


@st.cache_data(show_spinner=False, max_entries=4)
def _load_genes(gene_bytes, desc_bytes):
    """The gene model (with descriptions) read from these file contents, by the loader the
    CLI uses (annotation.load_gene_annotation)."""
    from annotation import load_gene_annotation

    with tempfile.TemporaryDirectory(prefix="trace_genes_") as d:
        gene_path = os.path.join(d, "gene_model.csv")
        with open(gene_path, "wb") as fh:
            fh.write(gene_bytes)
        desc_path = None
        if desc_bytes is not None:
            desc_path = os.path.join(d, "gene_descriptions.txt")
            with open(desc_path, "wb") as fh:
                fh.write(desc_bytes)
        return load_gene_annotation(gene_path, desc_path)


def _bytes(source):
    """A bundled file's or an uploaded file's contents (None for none)."""
    if source is None:
        return None
    if isinstance(source, Path):
        return source.read_bytes()
    return source.getvalue()


_BUILDS = ["SL3", "SL4"]       # the "Genome build" choices of a species that has builds


def _species_options():
    from utils.species_files import SPECIES_FILES

    return list(SPECIES_FILES.keys()) + ["Other (upload files)"]


def _has_builds(species):
    from utils.species_files import SPECIES_FILES

    return "gene_model_SL3" in SPECIES_FILES.get(species, {})


def _builtin_files(species, build):
    """The species' built-in gene model and descriptions file (for ``build`` when the species
    has builds), each None when there is none on this computer."""
    from utils.species_files import SPECIES_FILES

    auto = SPECIES_FILES.get(species, {})
    if _has_builds(species):
        gm_auto, desc_auto = auto.get(f"gene_model_{build}"), auto.get(f"gene_desc_{build}")
    else:
        gm_auto, desc_auto = auto.get("gene_model"), auto.get("gene_desc")
    gm_auto = gm_auto if gm_auto is not None and gm_auto.exists() else None
    desc_auto = desc_auto if desc_auto is not None and desc_auto.exists() else None
    return gm_auto, desc_auto


def chosen_genes():
    """The gene model chosen in Step 1 now, for the Regional Plot, which runs before this tab:
    Step 1's widget values as the session holds them at the start of the run (their defaults
    before this tab first ran), loaded by Step 1's cached loader. None without a gene model,
    or when it does not load (Step 1 shows why)."""
    state = st.session_state
    options = _species_options()
    species = state.get("ld_gene_species", options[0])
    species = species if species in options else options[0]
    build = None
    if _has_builds(species):
        build = state.get("ld_genome_build", _BUILDS[0])
        build = build if build in _BUILDS else _BUILDS[0]
    gm_auto, desc_auto = _builtin_files(species, build)
    gm_up, desc_up = state.get("gene_model_upload"), state.get("gene_desc_upload")
    gene_file = gm_up if gm_up is not None else gm_auto
    desc_file = desc_up if desc_up is not None else desc_auto
    if gene_file is None:
        return None
    try:
        return _load_genes(_bytes(gene_file), _bytes(desc_file))
    except Exception:
        return None


def render(ctx: LDContext):
    st.subheader("Gene Annotation")

    if not ctx.has_annotation:
        st.error(
            "annotation.py module not found.\n"
            "Place it in your `ld/` folder or project root."
        )
        return

    from annotation import (
        ASSEMBLY_NOTE,
        DESCRIPTIONS_FILE_HELP,
        GENE_FILE_HELP,
        annotate_candidate_intervals,
        chromosomes_without_genes,
        summarize_gene_model,
    )
    st.markdown(
        "Every LD block and every unblocked significant SNP, with the genes inside its "
        "candidate interval: from the marker before the first SNP to the marker after the "
        "last, the span the Regional Plot shades."
    )

    # ── Step 1: the gene models ──
    # (the Regional Plot, which runs before this tab, reads the same choice: chosen_genes)
    st.markdown("#### Step 1: Gene models")
    _species_opts = _species_options()
    _species = st.selectbox("Species", _species_opts, key="ld_gene_species")
    _build = None
    if _has_builds(_species):
        _build = st.selectbox("Genome build", ["SL3", "SL4"], key="ld_genome_build",
                              help=_BUILD_HELP)
    gm_auto, desc_auto = _builtin_files(_species, _build)
    if gm_auto is not None:
        st.write(f"Built-in gene model: `{gm_auto.name}`"
                 + (f"; descriptions: `{desc_auto.name}`" if desc_auto is not None else ""))

    st.markdown("Upload your own gene file and, optionally, a descriptions file:")
    gm_up = st.file_uploader("Gene file", type=["csv", "tsv", "txt", "gff3", "gff", "gff3.gz", "gff.gz"],
                             key="gene_model_upload", help=GENE_FILE_HELP)
    desc_up = st.file_uploader("Descriptions file (optional)", type=["txt", "tsv"],
                               key="gene_desc_upload", help=DESCRIPTIONS_FILE_HELP)
    st.caption(ASSEMBLY_NOTE)

    gene_file = gm_up if gm_up is not None else gm_auto
    desc_file = desc_up if desc_up is not None else desc_auto
    genes_df = None
    if gene_file is not None:
        try:
            genes_df = _load_genes(_bytes(gene_file), _bytes(desc_file))
        except Exception as e:
            logging.exception("Gene file loading failed")
            st.error(f"Error loading gene file: {e}")
    if genes_df is None:
        if gene_file is None:
            st.info("Choose a species with a built-in gene model, or upload a gene file.")
        return

    summary = summarize_gene_model(genes_df)
    st.success(f"Loaded {len(genes_df):,} genes across {len(summary)} chromosomes.")
    missing = chromosomes_without_genes(ctx.chroms, genes_df) if ctx.chroms is not None else []
    if missing:
        st.warning(f"No genes for chromosome(s) {', '.join(missing)} of your genotype data; "
                   "check that the chromosome names match.")
    with st.expander("Genes per chromosome", expanded=False):
        st.caption("Check that the coordinate ranges cover the same span as your SNP positions.")
        st.dataframe(summary, use_container_width=True, hide_index=True)

    # ── Step 2: annotate ──
    st.markdown("#### Step 2: Annotate")
    intervals = run_candidate_intervals(ctx)
    if intervals is None or intervals.empty:
        st.info("No LD blocks or unblocked significant SNPs to annotate: run the GWAS, and "
                "the LD block detection on the LD Blocks & Haplotypes tab, first.")
        return
    table, line = annotate_candidate_intervals(intervals, genes_df)
    st.markdown(f"**{line}**")
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button(
        "Download gene annotation (CSV)", table.to_csv(index=False).encode("utf-8"),
        file_name=f"Gene_annotation_{ctx.trait_col}.csv", mime="text/csv",
        key="dl_gene_annotation",
    )
