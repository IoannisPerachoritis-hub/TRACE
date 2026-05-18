"""Tab 2 — Local LD heatmap around a lead SNP."""

import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from streamlit.runtime.scriptrunner import StopException
from utils.pub_theme import LD_HEATMAP_CMAP, FIGSIZE, export_plotly

from gwas.ld import extract_block_geno_for_paper, _guess_lead_col
from . import LDContext


class _TabExit(Exception):
    pass


def render(ctx: LDContext, get_r2_cached):
    try:
        st.subheader("Local LD heatmap around a lead SNP / LD block")

        # --- Choose lead SNP from top hits ---
        top_snps = ctx.gwas_df.sort_values("PValue").head(200)
        if top_snps.empty:
            st.info("GWAS table seems empty. Run GWAS first.")
            raise _TabExit()

        lead_snp = st.selectbox(
            "Select lead SNP (top 200 by P-value):",
            options=top_snps["SNP"].tolist(),
            index=0,
        )

        # Detect LD blocks table + lead column if available
        lead_col = _guess_lead_col(ctx.haplo_df_auto)
        has_blocks = (
            isinstance(ctx.haplo_df_auto, pd.DataFrame)
            and not ctx.haplo_df_auto.empty
            and lead_col is not None
        )

        colA, colB = st.columns(2)
        with colA:
            use_block_if_available = st.checkbox(
                "Snap window to LD block if available",
                value=has_blocks,
                help="If LD blocks have been computed, use the LD block containing the lead SNP.",
            )
        with colB:
            buffer_kb_default = int(ctx.ld_decay_kb * 2)
            buffer_kb = st.slider(
                "Buffer around region (kb)",
                min_value=10,
                max_value=5000,
                value=buffer_kb_default,
                step=10,
                help="Extends the heatmap window by this many kb on each side of the LD block (or SNP if no block).",
            )

        # ---- Determine region [chr, start_bp, end_bp] ----
        snp_row = ctx.gwas_df.loc[ctx.gwas_df["SNP"] == lead_snp]
        if snp_row.empty:
            st.warning(f"SNP {lead_snp} not found in GWAS table.")
            raise _TabExit()

        chr_snp = str(snp_row.iloc[0]["Chr"])
        pos_snp = int(snp_row.iloc[0]["Pos"])

        use_block = False
        core_start = core_end = None

        # Try block-based window if available
        if use_block_if_available and has_blocks:
            lead_pat = re.escape(str(lead_snp))
            block_rows = ctx.haplo_df_auto[
                ctx.haplo_df_auto[lead_col].astype(str).str.contains(
                    rf"(^|[;,\s|]){lead_pat}($|[;,\s|])", regex=True
                )
            ]

            if not block_rows.empty:
                r = block_rows.iloc[0]

                core_start = int(r["Start (bp)"])
                core_end = int(r["End (bp)"])
                extra = buffer_kb * 1000
                start_bp = max(0, core_start - extra)
                end_bp = core_end + extra
                chr_sel = str(r["Chr"])
                label_source = (
                    f"LD block Chr{chr_sel}:{core_start:,}-{core_end:,} "
                    f"± {buffer_kb} kb buffer"
                )
                use_block = True

        # Fallback: simple SNP-centered window
        if not use_block:
            chr_sel = chr_snp
            start_bp = pos_snp - buffer_kb * 1000
            end_bp = pos_snp + buffer_kb * 1000
            label_source = f"SNP-centered window: Chr{chr_sel}:{pos_snp:,} ± {buffer_kb} kb"

        st.markdown(f"**Window used:** {label_source}")

        # ---- Extract region genotypes ----
        keep_mask = ctx.keep_mask

        region_geno, region_pos, region_sids = extract_block_geno_for_paper(
            geno_imputed=ctx.geno_ld,
            chroms=ctx.chroms,
            positions=ctx.positions,
            sid=ctx.sid,
            block_chr=chr_sel,
            start_bp=start_bp,
            end_bp=end_bp,
            sample_keep_mask=keep_mask,
            maf_threshold=st.session_state.get("maf_ld", 0.01)
        )

        if region_geno.size == 0:
            st.info("No SNPs in this region after QC.")
            raise _TabExit()

        # Remove monomorphic SNPs
        snp_var = np.nanvar(region_geno, axis=0)
        poly_mask = snp_var > 0
        region_geno = region_geno[:, poly_mask]
        region_pos = region_pos[poly_mask]
        region_sids = region_sids[poly_mask]

        # Keep SNPs ordered by genomic position
        order = np.argsort(region_pos)
        region_pos = region_pos[order]
        region_geno = region_geno[:, order]
        region_sids = region_sids[order]

        if region_geno.shape[1] < 2:
            st.info("Region contains <2 polymorphic SNPs — cannot compute LD.")
            return

        st.write(
            f"SNPs in LD window: **{region_geno.shape[1]}** "
            f"(Chr{chr_sel}:{int(region_pos.min()):,}–{int(region_pos.max()):,})"
        )

        # ---- Compute r² matrix ----
        r2 = get_r2_cached(
            region_geno, region_pos, min_pair_n=20,
            cache_key=f"r2::local::{chr_sel}:{start_bp}-{end_bp}::{lead_snp}::{region_geno.shape[1]}snps",
        )

        # ================================================================
        # OPTIONAL: Show raw correlation (r) instead of r²
        # ================================================================
        show_raw_r = st.checkbox("Show raw correlation (r) instead of r²", value=False)

        if show_raw_r:
            from gwas.ld import pairwise_r
            ld_matrix_to_plot = pairwise_r(region_geno)

            vmin_plot, vmax_plot = -1, 1
            colorbar_label = "Correlation (r)"
        else:
            ld_matrix_to_plot = r2
            vmin_plot, vmax_plot = 0, 1
            colorbar_label = "Linkage disequilibrium (r²)"

        # ---- Heatmap (lower triangle, publication-grade) ----
        show_labels = ctx.show_ld_labels

        fig_ld, ax = plt.subplots(figsize=FIGSIZE["heatmap"])
        tri_mask = np.triu(np.ones_like(ld_matrix_to_plot, dtype=bool))

        sns.heatmap(
            ld_matrix_to_plot,
            mask=tri_mask,
            cmap=LD_HEATMAP_CMAP,
            vmin=vmin_plot,
            vmax=vmax_plot,
            annot=False,
            square=True,
            linewidths=0.25,
            linecolor="white",
            cbar_kws={"shrink": 0.85},
            ax=ax,
        )

        # ---- Colorbar styling ----
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel(colorbar_label, fontsize=12)
        cbar.ax.tick_params(width=0)

        # ---- X-axis ticks: HARD CAP (≤ 8) + Mb labels ----
        max_ticks = 8
        tick_step = max(1, int(np.ceil(len(region_pos) / max_ticks)))
        tick_idx = np.arange(0, len(region_pos), tick_step)

        tick_labels = [f"{region_pos[i] / 1e6:.3f}" for i in tick_idx]

        ax.set_xticks(tick_idx + 0.5)
        ax.set_xticklabels(
            tick_labels,
            rotation=45,
            ha="right",
        )

        ax.set_yticks([])
        ax.set_xlabel("Genomic position (Mb)")

        ax.set_title(f"Local LD structure around {lead_snp} (Chr{chr_sel})")

        ax.tick_params(axis="both", which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        st.caption("LD computed from imputed dosages; haplotype labels use hard-called genotypes.")

        # Byte-cache the savefig output keyed on what affects the rendered image.
        local_cache_key = (
            "LOCAL_LD_BYTES",
            chr_sel, int(start_bp), int(end_bp),
            str(lead_snp),
            int(ld_matrix_to_plot.shape[0]),
            bool(show_raw_r),
            bool(show_labels),
        )
        local_cache = st.session_state.setdefault("_local_ld_cache", {})
        local_cached = local_cache.get(local_cache_key)

        if local_cached is None:
            buf_disp = BytesIO()
            fig_ld.savefig(buf_disp, format="png", dpi=120, bbox_inches="tight")
            buf_png = BytesIO()
            fig_ld.savefig(buf_png, format="png", dpi=600, bbox_inches="tight")
            buf_svg = BytesIO()
            fig_ld.savefig(buf_svg, format="svg", bbox_inches="tight")
            buf_pdf = BytesIO()
            fig_ld.savefig(buf_pdf, format="pdf", bbox_inches="tight")
            local_cached = {
                "display": buf_disp.getvalue(),
                "png": buf_png.getvalue(),
                "svg": buf_svg.getvalue(),
                "pdf": buf_pdf.getvalue(),
            }
            local_cache[local_cache_key] = local_cached
        plt.close(fig_ld)

        st.image(local_cached["display"])
        local_fname = f"LD_heatmap_{lead_snp}_Chr{chr_sel}_{start_bp}_{end_bp}"
        lcol_png, lcol_svg, lcol_pdf = st.columns(3)
        lcol_png.download_button(
            "📥 Download LD heatmap PNG", local_cached["png"],
            file_name=f"{local_fname}.png", mime="image/png",
            key=f"dl_localld_png_{local_fname}",
        )
        lcol_svg.download_button(
            "📥 Download LD heatmap SVG", local_cached["svg"],
            file_name=f"{local_fname}.svg", mime="image/svg+xml",
            key=f"dl_localld_svg_{local_fname}",
        )
        lcol_pdf.download_button(
            "📥 Download LD heatmap PDF", local_cached["pdf"],
            file_name=f"{local_fname}.pdf", mime="application/pdf",
            key=f"dl_localld_pdf_{local_fname}",
        )

    except StopException:
        pass
