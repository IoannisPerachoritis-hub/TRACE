"""Tab 4 — LD Decay by Chromosome."""

import numpy as np
import pandas as pd
import streamlit as st

from utils.pub_theme import export_matplotlib
from . import LDContext


def render(ctx: LDContext):
    st.subheader("LD Decay by Chromosome")

    if not ctx.has_annotation:
        st.error("annotation.py module not found.")
        return

    from annotation import compute_ld_decay_by_chromosome, plot_ld_decay_matplotlib

    st.markdown(
        "Computes LD decay curves per chromosome from your genotype data. "
        "Reports the distance at which r² drops below 0.2 and 0.1 — "
        "these values justify your LD block detection window sizes."
    )

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        max_snps_chr = st.number_input(
            "Max SNPs per chromosome (subsample)",
            min_value=200, max_value=5000, value=2000, step=200,
            help="Higher = more accurate but slower.",
        )
    with col_d2:
        max_dist_decay = st.number_input(
            "Max distance (kb)",
            min_value=100.0, max_value=10000.0, value=5000.0, step=500.0,
        )
    with col_d3:
        n_bins_decay = st.slider(
            "Distance bins", min_value=20, max_value=100, value=50,
        )

    if st.button("Compute LD decay", key="btn_ld_decay"):
        with st.spinner("Computing LD decay (this may take a minute)..."):
            decay_df, summary_df = compute_ld_decay_by_chromosome(
                chroms=ctx.chroms,
                positions=ctx.positions,
                geno_imputed=ctx.geno_ld,
                max_snps_per_chr=int(max_snps_chr),
                max_dist_kb=float(max_dist_decay),
                n_bins=int(n_bins_decay),
            )

            st.session_state["ld_decay_df"] = decay_df
            st.session_state["ld_decay_summary"] = summary_df

    decay_df = st.session_state.get("ld_decay_df", None)
    summary_df = st.session_state.get("ld_decay_summary", None)

    if decay_df is not None and not decay_df.empty:
        st.markdown("#### Per-chromosome decay summary")
        st.dataframe(summary_df, use_container_width=True)

        # Highlight genome-wide median
        if "decay_kb_r2_0.2" in summary_df.columns:
            median_decay = summary_df["decay_kb_r2_0.2"].median()
            if pd.notna(median_decay):
                st.info(f"Genome-wide median LD decay (r² ≤ 0.2): **{median_decay:.0f} kb**")

                if st.button("Use this as the LD decay estimate for block detection", key="btn_update_decay"):
                    st.session_state["ld_decay_kb"] = float(median_decay)
                    st.session_state["ld_decay_computed"] = True
                    st.success(
                        f"Updated LD decay estimate to {median_decay:.0f} kb. "
                        "This will be used for flank windows in LD block detection."
                    )

        # Plot
        fig_decay, ax_decay = plot_ld_decay_matplotlib(
            decay_df, summary_df,
            title=f"LD Decay — {ctx.trait_col}",
        )
        st.pyplot(fig_decay)

        # Export
        export_matplotlib(fig_decay, f"LD_decay__{ctx.trait_col}", label_prefix="Download LD decay")

        st.download_button(
            "Download LD decay data (CSV)",
            decay_df.to_csv(index=False).encode("utf-8"),
            file_name=f"LD_decay_data__{ctx.trait_col}.csv",
            mime="text/csv",
            key="dl_ld_decay_data",
        )
