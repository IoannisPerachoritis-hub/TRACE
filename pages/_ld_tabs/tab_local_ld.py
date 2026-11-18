"""Tab 2: Local LD heatmap around a lead SNP, or of a detected LD block."""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from streamlit.runtime.scriptrunner import StopException
from utils.pub_theme import LD_HEATMAP_CMAP, FIGSIZE, export_plotly, show_plot  # noqa: F401 (export_plotly kept for parity)

from gwas import guipost
from gwas.ld import extract_block_geno_for_paper
from gwas.plotting import (
    ld_pairs_long, lead_snp_position, plot_ld_block_heatmap, select_ld_heatmap_window,
)
from . import LDContext


class _TabExit(Exception):
    pass


def _block_heatmap(ctx, window):
    """Detected-block mode: the shared LD-block heatmap, the block and its
    neighbouring SNPs, drawn by gwas.plotting.plot_ld_block_heatmap as in the CLI
    and One-Click ZIPs (same imputed matrix, same window rule, same r²).

    Returns the display/PNG/SVG/PDF bytes, the lead SNP marked and a window summary.
    """
    label_choice = st.radio(
        "Heatmap tick labels", ["Genomic position (Mb)", "SNP IDs"], horizontal=True,
        key="local_ld_block_labels",
    )
    show_star = st.checkbox(
        "Show lead-SNP star", value=True, key="local_ld_show_star",
        help="Mark the lead SNP with a gold star on the heatmap.",
    )
    label_mode = "snp_id" if label_choice == "SNP IDs" else "position"
    lead = getattr(window, "block_lead_snp", "") or window.lead_snp
    lead_pos = lead_snp_position(lead, ctx.sid, ctx.chroms, ctx.positions, window.chr)
    geno = ctx.geno_ld_aligned if ctx.geno_ld_aligned is not None else ctx.geno_ld
    # Key the image cache on what is drawn: the table (its identity and rows) and the
    # window's SNPs, positions and IDs. The window's genotypes are read only to draw it.
    win = select_ld_heatmap_window(ctx.chroms, ctx.positions, window.chr,
                                   window.core_start, window.core_end, lead_pos=lead_pos)
    key = ("LOCAL_LD_BLOCK_BYTES", str(window.chr), int(window.core_start),
           int(window.core_end), str(lead), label_mode, bool(show_star),
           guipost.table_key(ctx), guipost.rows_key(ctx),
           guipost.digest(np.asarray(win["indices"]), np.asarray(win["positions"]),
                          np.asarray(ctx.sid)[win["indices"]].astype(str)))
    cache = guipost.bounded(st.session_state, "_local_ld_cache")
    hit = guipost.recall(cache, key)
    if hit is None:
        fig, info = plot_ld_block_heatmap(
            geno, ctx.chroms, ctx.positions, window.chr, window.core_start, window.core_end,
            lead_pos=lead_pos, label_mode=label_mode, sid=ctx.sid, lead_star=show_star,
        )
        hit = {"lead": lead, "info": {k: info[k] for k in (
            "n_shown", "n_block_shown", "n_window", "thinned", "exhausted",
            "block_share", "window_start", "window_end", "lead_index")}}
        for name, kw in (("display", {"format": "png", "dpi": 120}),
                         ("png", {"format": "png", "dpi": 600}),
                         ("svg", {"format": "svg"}),
                         ("pdf", {"format": "pdf"})):
            buf = BytesIO()
            fig.savefig(buf, bbox_inches="tight", **kw)
            hit[name] = buf.getvalue()
        plt.close(fig)
        guipost.remember(cache, key, hit, guipost.FIGURE_CACHE_MAX)
    return hit


def _lead_snp_heatmap(r2, region_pos, region_sids, chr_sel, start_bp, end_bp, lead_snp,
                      table=""):
    """Lead-SNP mode: the SNP-centred local LD heatmap. ``table`` names
    the genotype table and its rows in the image cache's key."""
    show_star = st.checkbox(
        "Show lead-SNP star", value=True, key="local_ld_show_star",
        help="Mark the lead SNP with a gold star on the heatmap.",
    )

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

    # Byte-cache the savefig output keyed on what affects the rendered image: the table and
    # its rows, the window's SNPs (in their order) and the display settings.
    local_cache_key = (
        "LOCAL_LD_BYTES",
        chr_sel, int(start_bp), int(end_bp),
        str(lead_snp),
        int(ld_matrix_to_plot.shape[0]),
        bool(show_star),
        str(table),
        guipost.digest(np.asarray(region_sids).astype(str), np.asarray(region_pos)),
    )
    local_cache = guipost.bounded(st.session_state, "_local_ld_cache")
    local_cached = guipost.recall(local_cache, local_cache_key)

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
        guipost.remember(local_cache, local_cache_key, local_cached, guipost.FIGURE_CACHE_MAX)
    plt.close(fig_ld)
    return local_cached


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
        block_mode = bool(getattr(window, "use_block", False)) and window.core_start is not None

        st.markdown(f"**Window used:** {window.label}")
        _block_ids = getattr(window, "block_snp_ids", "") or None

        # ---- Extract region genotypes ----
        # In detected-block mode these are the block's member SNPs: they back the r²
        # exports below; the heatmap itself is the shared block heatmap. Only the window
        # is read from the table (its phenotype-aligned rows).
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

        if not block_mode:
            st.write(
                f"SNPs in LD window: **{region_geno.shape[1]}** "
                f"(Chr{chr_sel}:{int(region_pos.min()):,}–{int(region_pos.max()):,})"
            )

        # ---- Compute r² matrix ----
        # keyed by the table's identity, its rows and the window's SNPs
        r2 = get_r2_cached(
            region_geno, region_pos, min_pair_n=20,
            cache_key=guipost.r2_key(ctx, "local", chr_sel, start_bp, end_bp, lead_snp,
                                     region_sids, min_pair_n=20),
        )

        if block_mode:
            local_cached = _block_heatmap(ctx, window)
            _i = local_cached["info"]
            st.write(
                f"Heatmap: **{_i['n_shown']}** SNPs "
                f"(Chr{chr_sel}:{_i['window_start']:,}–{_i['window_end']:,}), "
                f"**{_i['n_block_shown']}** of them between the block's start and end"
                + (f"; thinned evenly from {_i['n_window']}" if _i["thinned"] else "")
            )
            if _i["lead_index"] is not None:
                _lead_note = f" The red tick marks the block's lead SNP ({local_cached['lead']})."
            else:
                _lead_note = f" (Lead SNP {local_cached['lead']} is not in the window.)"
            if _i["exhausted"]:
                _share_note = (f"Chr{chr_sel} has no more SNPs, so all of them are shown "
                               f"and the block is {_i['block_share']:.0%} of them.")
            else:
                _share_note = ("At most half of the SNPs shown are in the block, and at "
                               "least 20 SNPs are shown.")
            st.caption(
                "The block's span (black outline) is shown with its neighbouring SNPs. "
                + _share_note + " The buffer applies to the Regional Plot only. LD "
                "computed from imputed dosages." + _lead_note
            )
        else:
            local_cached = _lead_snp_heatmap(
                r2, region_pos, region_sids, chr_sel, start_bp, end_bp, lead_snp,
                table=f"{guipost.table_key(ctx)}::{guipost.rows_key(ctx)}")

        show_plot(local_cached["display"], "medium")
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
        # Serialisation only: r² is already computed and cached. Two shapes so a
        # user can table TRACE's numbers instead of reading them off the figure.
        if block_mode:
            st.caption(f"The r² downloads cover the block's {int(r2.shape[0])} member SNPs.")
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
