"""Tab 4 — LD Decay by Chromosome."""

import numpy as np
import pandas as pd
import streamlit as st

from . import LDContext


def render(ctx: LDContext):
    st.subheader("LD Decay by Chromosome")

    if not ctx.has_annotation:
        st.error("annotation.py module not found.")
        return

    from annotation import compute_ld_decay_by_chromosome

    st.markdown(
        "Computes LD decay curves per chromosome from your genotype data. "
        "Reports the distance at which r² drops below 0.2 and 0.1 — "
        "these values justify your LD block detection window sizes."
    )
    st.caption(
        "The genome-wide median decay computed here also sets the **default window "
        "buffer (≈ decay × 2)** for the Regional Plot and Local LD tabs — so this "
        "estimate justifies the window sizes used across the page, not only block detection."
    )

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        max_snps_chr = st.number_input(
            "Max SNPs per chromosome (subsample)",
            min_value=200, max_value=5000, value=1500, step=200,
            help="Higher = more accurate but slower.",
        )
    with col_d2:
        max_dist_decay = st.number_input(
            "Max distance (kb)",
            min_value=100.0, max_value=10000.0, value=5000.0, step=500.0,
        )
    with col_d3:
        n_bins_decay = st.slider(
            "Distance bins", min_value=20, max_value=100, value=40,
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
        if "censored_r2_0.2" in summary_df.columns and bool(summary_df["censored_r2_0.2"].any()):
            _n_cens = int(summary_df["censored_r2_0.2"].sum())
            st.caption(
                f"`censored_r2_0.2` = grid-limited: for {_n_cens} chromosome(s) the r²≤0.2 "
                "crossing falls in the first distance bin, so the decay is **below the grid "
                "resolution** — read that value as **≤ X kb**, an upper bound, not an estimate."
            )

        # Highlight genome-wide median
        if "decay_kb_r2_0.2" in summary_df.columns:
            median_decay = summary_df["decay_kb_r2_0.2"].median()
            if pd.notna(median_decay):
                _cens_note = ""
                if "censored_r2_0.2" in summary_df.columns:
                    _nc = int(summary_df["censored_r2_0.2"].sum())
                    _nt = int(summary_df["decay_kb_r2_0.2"].notna().sum())
                    if _nc:
                        _cens_note = f"  ({_nc} of {_nt} chromosomes grid-limited — upper bound)"
                st.info(f"Genome-wide median LD decay (r² ≤ 0.2): **{median_decay:.0f} kb**{_cens_note}")

                if st.button("Use this as the LD decay estimate for block detection", key="btn_update_decay"):
                    st.session_state["ld_decay_kb"] = float(median_decay)
                    st.session_state["ld_decay_computed"] = True
                    st.success(
                        f"Updated LD decay estimate to {median_decay:.0f} kb. "
                        "This will be used for flank windows in LD block detection."
                    )

        st.download_button(
            "Download LD decay data (CSV)",
            decay_df.to_csv(index=False).encode("utf-8"),
            file_name=f"LD_decay_data__{ctx.trait_col}.csv",
            mime="text/csv",
            key="dl_ld_decay_data",
        )
