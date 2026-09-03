"""Tab 2 — Local LD heatmap around a lead SNP."""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from streamlit.runtime.scriptrunner import StopException
from utils.pub_theme import LD_HEATMAP_CMAP, FIGSIZE, export_plotly  # noqa: F401 (export_plotly kept for parity)

from gwas.ld import extract_block_geno_for_paper
from gwas.plotting import ld_pairs_long
from . import LDContext


class _TabExit(Exception):
    pass


def render(ctx: LDContext, get_r2_cached, window):
    try:
        st.subheader("Local LD heatmap around a lead SNP / LD block")

        # The lead SNP / snap-to-block / buffer controls now live in the shared
        # selector above the tab bar (pages/_ld_tabs/_window.py); this tab and the
        # Regional Plot tab render one selection.
        if window is None:
            st.info("Select a lead SNP in the region selector above to view its local LD.")
            raise _TabExit()

        lead_snp = window.lead_snp
        chr_sel = window.chr
        start_bp = window.start_bp
        end_bp = window.end_bp

        st.markdown(f"**Window used:** {window.label}")
        _block_ids = getattr(window, "block_snp_ids", "") or None

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
            maf_threshold=st.session_state.get("maf_ld", 0.01),
            snp_ids=_block_ids,
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
            st.info("Region contains <2 polymorphic SNPs; cannot compute LD.")
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
        show_star = st.checkbox(
            "Show lead-SNP star", value=True, key="local_ld_show_star",
            help="Mark the lead SNP with a gold star on the heatmap.",
        )

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

        # Mark the lead SNP: a star at its row/column apex on the (masked) diagonal.
        # Tolerate the lead being absent from the plotted set (seed-not-member, or
        # dropped by the MAF / monomorphic filters) -- never index [0] unguarded.
        _lead_hit = np.where(np.asarray(region_sids).astype(str) == str(lead_snp))[0]
        if _lead_hit.size and show_star:
            _li = int(_lead_hit[0])
            ax.scatter(_li + 0.5, _li + 0.5, marker="*", s=220, color="#F0E442",
                       edgecolor="#333333", linewidth=0.6, zorder=6, clip_on=False)

        plt.tight_layout()
        if not show_star:
            _star_note = ""
        elif _lead_hit.size:
            _star_note = f" The gold star marks the lead SNP ({lead_snp})."
        else:
            _star_note = f" (Lead SNP {lead_snp} is not shown -- outside the window or filtered by QC.)"
        st.caption("LD computed from imputed dosages; haplotype labels use hard-called genotypes." + _star_note)

        # Byte-cache the savefig output keyed on what affects the rendered image.
        local_cache_key = (
            "LOCAL_LD_BYTES",
            chr_sel, int(start_bp), int(end_bp),
            str(lead_snp),
            int(ld_matrix_to_plot.shape[0]),
            bool(show_raw_r),
            bool(show_star),
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
            "Download LD heatmap PNG", local_cached["png"],
            file_name=f"{local_fname}.png", mime="image/png",
            key=f"dl_localld_png_{local_fname}",
        )
        lcol_svg.download_button(
            "Download LD heatmap SVG", local_cached["svg"],
            file_name=f"{local_fname}.svg", mime="image/svg+xml",
            key=f"dl_localld_svg_{local_fname}",
        )
        lcol_pdf.download_button(
            "Download LD heatmap PDF", local_cached["pdf"],
            file_name=f"{local_fname}.pdf", mime="application/pdf",
            key=f"dl_localld_pdf_{local_fname}",
        )

        # ---- Numeric export: the r² numbers behind the heatmap ----
        # Serialisation only — r² is already computed and cached. Two shapes so a
        # user can table TRACE's numbers instead of reading them off the figure.
        _n = int(r2.shape[0])
        _ld_stem = f"Chr{chr_sel}_{int(start_bp)}_{int(end_bp)}_{_n}snps"
        _long_csv = ld_pairs_long(r2, region_sids).to_csv(index=False).encode()
        _square_csv = pd.DataFrame(r2, index=region_sids, columns=region_sids).to_csv().encode()
        lcol_long, lcol_sq = st.columns(2)
        lcol_long.download_button(
            "Download r² (long: SNP_A, SNP_B, r2)", _long_csv,
            file_name=f"LD_r2_long_{_ld_stem}.csv", mime="text/csv",
            key=f"dl_localld_long_{_ld_stem}",
        )
        lcol_sq.download_button(
            "Download r² (square matrix)", _square_csv,
            file_name=f"LD_r2_matrix_{_ld_stem}.csv", mime="text/csv",
            key=f"dl_localld_sq_{_ld_stem}",
        )

    except StopException:
        pass
