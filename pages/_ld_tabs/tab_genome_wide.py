"""Tab 6: Peak-centric LD blocks & haplotype analysis."""

import logging
import re
import textwrap
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO
from streamlit.runtime.scriptrunner import StopException
from utils.pub_theme import (PALETTE_CYCLE, FIGSIZE, export_matplotlib, haplotype_color_map,
                             hap_sort_key, show_plot)
from utils.plot_data import pca_haplotype_labels, pca_points_table

from gwas import guipost
from gwas.ld import (
    find_ld_clusters_genomewide,
    filter_contained_blocks,
)
from . import LDContext
from ._decay import decay_run_label
from gwas.windows import ld_window_kb


def _hash_df(df: pd.DataFrame) -> str:
    import hashlib
    h = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.blake2b(h.tobytes(), digest_size=16).hexdigest()


# numeric sort for H1, H2, ..., H10, H11 (shared with the box plot: utils.pub_theme)
_hap_sort_key = hap_sort_key


def _render_fig_or_cached(cache_namespace, cache_key, fname_stem, label_prefix, build_fig_fn):
    """Display a matplotlib fig via session-state byte cache.

    On cache miss, calls build_fig_fn() to construct the figure, renders to
    PNG@120dpi (display) + PNG@600dpi + SVG + PDF byte buffers, caches them,
    closes the fig, then displays via st.image. Subsequent reruns with the
    same cache_key short-circuit straight to st.image with cached bytes. The cache keeps
    the most recent gwas.guipost.FIGURE_CACHE_MAX figures.
    """
    cache = guipost.bounded(st.session_state, cache_namespace)
    cached = guipost.recall(cache, cache_key)

    if cached is None:
        fig = build_fig_fn()
        if fig is None:
            return
        buf_disp = BytesIO()
        fig.savefig(buf_disp, format="png", dpi=120, bbox_inches="tight")
        buf_png = BytesIO()
        fig.savefig(buf_png, format="png", dpi=600, bbox_inches="tight")
        buf_svg = BytesIO()
        fig.savefig(buf_svg, format="svg", bbox_inches="tight")
        buf_pdf = BytesIO()
        fig.savefig(buf_pdf, format="pdf", bbox_inches="tight")
        plt.close(fig)
        cached = {
            "display": buf_disp.getvalue(),
            "png": buf_png.getvalue(),
            "svg": buf_svg.getvalue(),
            "pdf": buf_pdf.getvalue(),
        }
        guipost.remember(cache, cache_key, cached, guipost.FIGURE_CACHE_MAX)

    show_plot(cached["display"], "small")
    col_png, col_svg, col_pdf = st.columns(3)
    col_png.download_button(
        f"{label_prefix} PNG", cached["png"],
        file_name=f"{fname_stem}.png", mime="image/png",
        key=f"dl_{cache_namespace}_png_{fname_stem}",
    )
    col_svg.download_button(
        f"{label_prefix} SVG", cached["svg"],
        file_name=f"{fname_stem}.svg", mime="image/svg+xml",
        key=f"dl_{cache_namespace}_svg_{fname_stem}",
    )
    col_pdf.download_button(
        f"{label_prefix} PDF", cached["pdf"],
        file_name=f"{fname_stem}.pdf", mime="application/pdf",
        key=f"dl_{cache_namespace}_pdf_{fname_stem}",
    )


def _render_haplotype_pca(
    block_assignments, geno_sample_ids,
    block_chr, block_start, block_end,
    geno_view=None, table="", pcs_full=None,
):
    """PC1 vs PC2 inset, one point per accession, coloured by its haplotype in the block:
    the label of the block's sample-to-haplotype assignments (the CSV's Haplotype).
    Samples the haplotype test left out are hollow, "not tested"; samples without a
    haplotype keep the grey look, "no haplotype". ``pcs_full`` is the GWAS page's PCs of the
    table (the page's context: gwas.guipost.run_pcs). Without them it computes two from
    ``geno_view`` (the GWAS page's imputed table; the session's geno_imputed when not given),
    which needs the whole table: not done when the table is on disk. ``table`` names the
    table in the figure cache's key."""
    from gwas.genostore import is_lazy

    eigenvalues = st.session_state.get("pca_eigenvalues")

    if pcs_full is None or pcs_full.shape[1] < 2:
        from gwas.kinship import _standardize_geno_for_grm
        geno = geno_view if geno_view is not None else st.session_state.get("geno_imputed")
        if geno is None:
            st.info("PCA panel unavailable: no cached PCs and no imputed genotype.")
            return
        if is_lazy(geno):
            st.info("PCA panel unavailable: the GWAS run left no PCs, and computing them "
                    "here would read the whole genotype table, which is kept on disk.")
            return
        Z = _standardize_geno_for_grm(geno)
        U, S, _ = np.linalg.svd(Z, full_matrices=False)
        pcs_full = (U[:, :2] * S[:2]).astype(np.float32)
        n = Z.shape[0]
        eigenvalues = (S ** 2) / max(n - 1, 1)

    pcs_2 = np.asarray(pcs_full[:, :2], dtype=float)

    if eigenvalues is not None:
        eig = np.asarray(eigenvalues, dtype=float)
    else:
        eig = pcs_full.var(axis=0, ddof=1)
    if eig.sum() <= 0:
        pct = np.array([float("nan"), float("nan")])
    else:
        pct = eig[:2] / eig.sum() * 100.0

    geno_iids = pd.Index(np.asarray(geno_sample_ids, dtype=str))
    if pcs_2.shape[0] != len(geno_iids):
        st.warning(
            f"PCA panel: cached PC matrix has {pcs_2.shape[0]} rows but the loaded "
            f"genotype table has {len(geno_iids)} samples; alignment mismatch. "
            "Skipping PCA panel."
        )
        return

    # each sample's label is its Haplotype in the block's assignments (as in the CSV)
    hap_labels, in_test = pca_haplotype_labels(block_assignments, geno_iids)
    labels = pd.Series(hap_labels, name="Haplotype")

    if len(block_assignments) > 0 and (labels != "").sum() == 0:
        st.warning(
            "PCA panel: 0 accessions matched between the block's haplotype assignments "
            f"and the loaded genotype table ({len(block_assignments)} assignment rows vs "
            f"{len(geno_iids)} genotype rows). Sample-ID format mismatch; "
            "cannot colour by haplotype. Skipping PCA panel."
        )
        return

    # Display groups shared by both views: each label, split into tested and not tested
    # samples, plus the samples without a haplotype (grey, as "Other" looked before)
    present = [h for h in pd.unique(labels) if h != ""]
    hap_order = sorted([h for h in present if h != "Other"], key=_hap_sort_key)
    # one colour per label in every plot of the tab and in the box plots (Other grey)
    color_map = haplotype_color_map(present)
    groups = []   # (mask, label, colour, tested: True / False / None = no haplotype)
    _nohap = (labels == "").to_numpy()
    if _nohap.any():
        groups.append((_nohap, "no haplotype", color_map["Other"], None))
    for h in (["Other"] if "Other" in present else []) + hap_order:
        for tested in (True, False):
            m = (labels == h).to_numpy() & (in_test == tested)
            if m.any():
                groups.append((m, h, color_map[h], tested))

    def _legend(h, n_h, tested):
        if tested is None:
            return f"no haplotype (n={n_h})"
        return f"{h} (n={n_h})" if tested else f"{h} (n={n_h}, not tested)"

    def _build_static_pca_fig():
        fig, ax = plt.subplots(figsize=(5, 5))
        for m, h, col, tested in groups:
            n_h = int(m.sum())
            size = 14 if tested is None or h == "Other" else (18 if h == "H1" else 42)
            if tested is None:          # no haplotype: the grey filled dots
                style = dict(c=col, edgecolors="none", linewidths=0.4, alpha=0.55)
            elif tested:
                style = dict(c=col, edgecolors="white" if h != "Other" else "none",
                             linewidths=0.4, alpha=0.95 if h != "Other" else 0.55)
            else:                       # left out of the test: hollow, in its colour
                style = dict(facecolors="none", edgecolors=col, linewidths=1.0, alpha=0.95)
            ax.scatter(pcs_2[m, 0], pcs_2[m, 1], s=size, label=_legend(h, n_h, tested),
                       **style)

        if np.isfinite(pct).all():
            ax.set_xlabel(f"PC1 ({pct[0]:.1f}%)")
            ax.set_ylabel(f"PC2 ({pct[1]:.1f}%)")
        else:
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
        ax.legend(fontsize=8, loc="best", frameon=False)
        ax.set_title("Accessions in PC space")
        return fig

    def _build_interactive_pca_fig():
        fig = go.Figure()
        for m, h, col, tested in groups:
            n_h = int(m.sum())
            size = 6 if tested is None or h == "Other" else (8 if h == "H1" else 14)
            accessions = geno_iids[m].astype(str).to_numpy()
            if tested is None:          # no haplotype: the grey filled dots
                marker = dict(color=col, size=size, opacity=0.55,
                              line=dict(color="rgba(0,0,0,0.2)", width=0.7))
            elif tested:
                marker = dict(color=col, size=size, opacity=0.95 if h != "Other" else 0.55,
                              line=dict(color="white" if h != "Other" else "rgba(0,0,0,0.2)",
                                        width=0.7))
            else:                       # left out of the test: hollow, in its colour
                marker = dict(color=col, size=size, opacity=0.95, symbol="circle-open",
                              line=dict(color=col, width=1.5))
            fig.add_trace(go.Scatter(
                x=pcs_2[m, 0],
                y=pcs_2[m, 1],
                mode="markers",
                name=_legend(h, n_h, tested),
                marker=marker,
                customdata=np.column_stack([
                    accessions, np.full(n_h, h if tested is not None else "none"),
                    np.full(n_h, "in the test" if tested else "not tested")]),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Haplotype: %{customdata[1]} (%{customdata[2]})<br>"
                    "PC1: %{x:.3f}<br>"
                    "PC2: %{y:.3f}<extra></extra>"
                ),
                showlegend=True,
            ))

        x_label = f"PC1 ({pct[0]:.1f}%)" if np.isfinite(pct).all() else "PC1"
        y_label = f"PC2 ({pct[1]:.1f}%)" if np.isfinite(pct).all() else "PC2"

        # Font cascade matching matplotlib (no Arial fallback)
        pca_font_family = "DejaVu Sans, Verdana, Geneva, sans-serif"

        fig.update_layout(
            title=dict(
                text="<b>Accessions in PC space (interactive)</b>",
                x=0.5,
                xanchor="center",
                font=dict(size=16, color="black", family=pca_font_family),
            ),
            xaxis=dict(
                title=dict(
                    text=f"<b>{x_label}</b>",
                    font=dict(size=14, color="black", family=pca_font_family),
                ),
                showgrid=True,
                gridcolor="rgba(200,200,200,0.4)",
                zeroline=False,
                tickfont=dict(size=12, color="black", family=pca_font_family),
            ),
            yaxis=dict(
                title=dict(
                    text=f"<b>{y_label}</b>",
                    font=dict(size=14, color="black", family=pca_font_family),
                ),
                showgrid=True,
                gridcolor="rgba(200,200,200,0.4)",
                zeroline=False,
                tickfont=dict(size=12, color="black", family=pca_font_family),
                scaleanchor="x",
                scaleratio=1,
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=550,
            margin=dict(t=70, b=80, l=80, r=40),
            hovermode="closest",
            legend=dict(font=dict(size=12, color="black", family=pca_font_family)),
            font=dict(family=pca_font_family, color="black", size=12),
        )
        return fig

    # Static cache key (matplotlib only; Plotly rebuilds each fragment rerun): the block,
    # its labels, and the table whose PCs are drawn
    static_cache_key = (
        "_hap_pca_render",
        block_chr, int(block_start), int(block_end),
        hash((tuple(labels.values), tuple(in_test.tolist()))),
        pcs_2.shape[0],
        str(table),
    )
    cache = guipost.bounded(st.session_state, "_hap_pca_cache")

    pca_widget_key = (
        f"pca_viz_type_{block_chr}_{int(block_start)}_{int(block_end)}"
    )

    @st.fragment
    def _pca_toggle_fragment():
        with st.expander("Plot style", expanded=False):
            viz_type = st.radio(
                "Visualization",
                ("Interactive", "Static"),
                horizontal=True,
                key=pca_widget_key,
                help=(
                    "Interactive (Plotly) lets you hover over each dot to see "
                    "the accession ID, useful for identifying which carriers "
                    "of the rare haplotype lie at which PC coordinates. "
                    "Static (matplotlib) is publication-grade and downloadable "
                    "as PNG / SVG / PDF."
                ),
            )

        if viz_type == "Interactive":
            show_plot(
                _build_interactive_pca_fig(), "medium",
                key=(
                    f"hap_pca_plotly_{block_chr}_"
                    f"{int(block_start)}_{int(block_end)}"
                ),
                data=pca_points_table(geno_iids, pcs_2, labels.to_numpy(), in_test=in_test),
                data_name=f"haplotype_pca_{block_chr}_{int(block_start)}_{int(block_end)}",
            )
        else:
            cached = guipost.recall(cache, static_cache_key)
            if cached is None:
                fig = _build_static_pca_fig()
                buf_disp = BytesIO()
                fig.savefig(buf_disp, format="png", dpi=120, bbox_inches="tight")
                buf_png = BytesIO()
                fig.savefig(buf_png, format="png", dpi=600, bbox_inches="tight")
                buf_svg = BytesIO()
                fig.savefig(buf_svg, format="svg", bbox_inches="tight")
                buf_pdf = BytesIO()
                fig.savefig(buf_pdf, format="pdf", bbox_inches="tight")
                plt.close(fig)
                cached = {
                    "display": buf_disp.getvalue(),
                    "png": buf_png.getvalue(),
                    "svg": buf_svg.getvalue(),
                    "pdf": buf_pdf.getvalue(),
                }
                guipost.remember(cache, static_cache_key, cached, guipost.FIGURE_CACHE_MAX)

            show_plot(cached["display"], "medium",
                      data=pca_points_table(geno_iids, pcs_2, labels.to_numpy(),
                                            in_test=in_test),
                      data_name=f"haplotype_pca_{block_chr}_{int(block_start)}_{int(block_end)}")
            fname = f"haplotype_pca_{block_chr}_{block_start}_{block_end}"
            col_png, col_svg, col_pdf = st.columns(3)
            col_png.download_button(
                "Download PCA PNG", cached["png"],
                file_name=f"{fname}.png", mime="image/png",
                key=f"dl_pca_png_{fname}",
            )
            col_svg.download_button(
                "Download PCA SVG", cached["svg"],
                file_name=f"{fname}.svg", mime="image/svg+xml",
                key=f"dl_pca_svg_{fname}",
            )
            col_pdf.download_button(
                "Download PCA PDF", cached["pdf"],
                file_name=f"{fname}.pdf", mime="application/pdf",
                key=f"dl_pca_pdf_{fname}",
            )

    with st.expander("Population structure (PC1 vs PC2, coloured by haplotype)",
                     expanded=False):
        _pca_toggle_fragment()


def _rule_short_name(rule_label, custom=None):
    """'Bonferroni' for 'Bonferroni (α = 0.05)', 'M_eff' for 'M_eff: Li & Ji (...)';
    a custom rule names its p-value."""
    label = str(rule_label)
    if label.startswith("Custom"):
        return f"custom P < {float(custom):.1e}" if custom is not None else "custom"
    return label.split(":")[0].split(" (")[0].strip()


def _kb(v):
    """A distance in kb to one decimal, without a trailing .0."""
    return f"{round(float(v), 1):g}"


def ld_decay_line(ld_decay_kb, run):
    """The Settings panel's line on the LD decay behind the window's default, naming the
    run it comes from (decay_run_label): the window is int(2 x decay) by the CLI's rule
    (gwas.windows.ld_window_kb), or 300 kb without an estimate, when blocks are detected
    without a decay value, as in the CLI."""
    if ld_decay_kb is None:
        return (f"LD decay: no estimate from {run}, so the window is "
                f"{ld_window_kb(None)} kb, as in the CLI, and blocks are detected without "
                "a decay value.")
    return (f"LD decay (median across chromosomes, from {run}): {_kb(ld_decay_kb)} kb, "
            f"so the window is {ld_window_kb(ld_decay_kb)} kb.")


def _render_ld_decay_curve():
    """The GWAS page's LD-decay estimate behind the window's default: its per-chromosome
    summary and curve, shown, not recomputed."""
    decay_df = st.session_state.get("ld_decay_df")
    summary_df = st.session_state.get("ld_decay_summary")
    with st.expander("Show LD decay curve", expanded=False):
        if decay_df is None or getattr(decay_df, "empty", True):
            st.info("No LD decay curve in this session: the GWAS page estimates it when "
                    "the GWAS runs.")
            return
        if summary_df is not None and not getattr(summary_df, "empty", True):
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            if ("censored_r2_0.2" in summary_df.columns
                    and bool(summary_df["censored_r2_0.2"].any())):
                st.caption("`censored_r2_0.2`: for these chromosomes the r² ≤ 0.2 crossing "
                           "falls in the first distance bin, so read their value as an "
                           "upper bound.")
        from annotation import plot_ld_decay_matplotlib
        _decay_fig, _decay_ax = plot_ld_decay_matplotlib(decay_df, summary_df)
        show_plot(_decay_fig, "medium", data=decay_df, data_name="ld_decay_curve")
        plt.close(_decay_fig)


def settings_summary(seed_p, top_n, flank_kb, ld_r2, n_perm, min_hap_count, min_group_size,
                     gallery_rule, gallery_custom=None, hap_test="kinship"):
    """The Settings panel's one line: the values in use, e.g. "Seeds P < 1e-05 plus top 10
    · window 200 kb · r² ≥ 0.6 · kinship-aware test · 1,000 permutations · haplotypes ≥ 5
    samples · groups ≥ 3 · gallery: Bonferroni"."""
    seeds = f"Seeds P < {float(seed_p):.0e}"
    if int(top_n) > 0:
        seeds += f" plus top {int(top_n)}"
    return " · ".join([
        seeds,
        f"window {float(flank_kb):g} kb",
        f"r² ≥ {float(ld_r2):g}",
        "kinship-aware test" if hap_test == "kinship" else "PCs-only test",
        f"{int(n_perm):,} permutations",
        f"haplotypes ≥ {int(min_hap_count)} samples",
        f"groups ≥ {int(min_group_size)}",
        f"gallery: {_rule_short_name(gallery_rule, gallery_custom)}",
    ])


def _render_settings(ctx):
    """Every setting of the tab, in one collapsed "Settings" panel at the top, with a
    summary line of the values in use above it. Three groups: LD blocks (detection),
    Haplotype test, Lead-SNP gallery. Keys and defaults are those the widgets had in
    their former places (the CLI's defaults); returns their values."""
    from .tab_significant_snps import _SIG_LABELS

    summary_slot = st.empty()
    with st.expander("Settings", expanded=False):
        st.markdown("**LD blocks**")
        sig_thresh = st.number_input(
            "Seed threshold for LD blocks (P <)",
            min_value=1e-12,
            max_value=0.1,
            value=1e-5,
            format="%.1e",
            help=(
                "Markers below this p-value seed LD blocks. After the disjoint-block "
                "redesign each seed yields at most one block, so this threshold now "
                "determines the block count directly. This is the CLI's --ld-seed-p "
                "(default 1e-5), not the genome-wide significance rule."
            ),
        )
        top_n = st.number_input(
            "Also include top N SNPs by P-value",
            min_value=0,
            max_value=500,
            value=10,
            step=1,
        )
        # the CLI's rule (gwas.windows.ld_window_kb): int(2 x decay), or 300 kb without an
        # estimate; the input's range always holds it (2 x decay <= 10000: the decay
        # estimate reaches at most 5000 kb)
        flank_kb_default = ld_window_kb(ctx.ld_decay_kb)
        flank_kb_auto = st.number_input(
            "LD window around each lead SNP (kb)",
            min_value=0.0,
            max_value=max(10000.0, float(flank_kb_default)),
            value=float(flank_kb_default),
            step=50.0,
            help=(
                "Physical distance (in kb) to extend around each seed SNP. After the "
                "disjoint-block redesign it bounds how far one block can extend from "
                "its seed; clusters that do not contain the seed are discarded. The "
                "default is 2× the measured LD decay distance, as in the CLI "
                "(300 kb when the decay could not be estimated)."
            ),
        )
        # the LD decay behind the default, estimated by the run the page shows (not
        # recomputed here)
        st.caption(ld_decay_line(ctx.ld_decay_kb,
                                 decay_run_label(st.session_state.get("gwas_run_summary"))))
        _render_ld_decay_curve()
        ld_threshold_auto = st.number_input(
            "r² threshold for cluster definition",
            min_value=0.1,
            max_value=0.9,
            value=0.6,
            step=0.05,
            help=(
                "SNPs with pairwise r² above this threshold are grouped into the same LD block. "
                "0.6 is a common default for self-pollinating crops where LD "
                "decays slowly. Use 0.3-0.5 for outcrossing species."
            ),
        )

        st.markdown("**Haplotype test**")
        # the test (the CLI's --hap-test): the kinship-aware one by default
        hap_test = guipost.HAP_TEST_CHOICES[st.radio(
            "Test model", list(guipost.HAP_TEST_CHOICES), index=0, key="hap_test_choice",
            help=guipost.HAP_TEST_HELP,
        )]
        # tab-local: only this tab consumes n_perm_hap
        n_perm = st.number_input(
            "Haplotype permutations",
            min_value=100, max_value=10000, value=1000, step=100,
            key="n_perm_hap",
            help="Number of Freedman-Lane permutations for haplotype block analysis. "
                 "Higher = finer p-values but slower. Max -log10(p) ≈ log10(n_perm).",
        )
        min_hap_count = st.number_input(
            "Minimum total samples per haplotype (collapse rarer → Other)",
            min_value=2,
            max_value=50,
            value=5,
            step=1,
            help=(
                "Haplotypes observed in fewer samples than this threshold "
                "are collapsed into the 'Other' group."
            ),
        )
        min_group_size = st.number_input(
            "Minimum samples per haplotype group (for testing)",
            min_value=2,
            max_value=50,
            value=3,
            step=1,
            help=(
                "Only haplotype groups with at least this many samples "
                "are included in the ANOVA test."
            ),
        )

        st.markdown("**Lead-SNP gallery**")
        _default = getattr(ctx, "sig_rule_label", "") or ""
        _idx = _SIG_LABELS.index(_default) if _default in _SIG_LABELS else 1
        gallery_rule = st.selectbox(
            "Significance rule:", _SIG_LABELS, index=_idx, key="lead_gallery_rule",
            help="Which SNPs count as significant for this gallery (defaults to the GWAS-page rule).",
        )
        gallery_custom = None
        if gallery_rule.startswith("Custom"):
            gallery_custom = st.number_input(
                "Custom p-value", min_value=1e-300, max_value=1.0,
                value=float(ctx.custom_thresh) if ctx.custom_thresh else 5e-8,
                format="%.1e", key="lead_gallery_custom",
            )
    summary_slot.caption(settings_summary(sig_thresh, top_n, flank_kb_auto,
                                          ld_threshold_auto, n_perm,
                                          min_hap_count, min_group_size, gallery_rule,
                                          gallery_custom, hap_test=hap_test))
    return {"seed_p": sig_thresh, "top_n": top_n, "flank_kb": flank_kb_auto,
            "ld_r2": ld_threshold_auto, "hap_test": hap_test,
            "n_perm": int(n_perm), "min_hap_count": int(min_hap_count),
            "min_group_size": int(min_group_size), "gallery_rule": gallery_rule,
            "gallery_custom": gallery_custom}


def render(
    ctx: LDContext,
    get_r2_cached,
    cached_extract_block_geno,
    compute_block_qc_effects,
    run_haplotype_block_gwas_cached_fn,
):
    try:
        # every setting of the tab, in one collapsed panel at the top
        settings = _render_settings(ctx)
        sig_thresh = settings["seed_p"]
        flank_kb_auto = settings["flank_kb"]
        ld_threshold_auto = settings["ld_r2"]
        top_n = settings["top_n"]

        st.subheader("LD blocks")

        # Local aliases
        gwas_df = ctx.gwas_df
        chroms = ctx.chroms
        positions = ctx.positions
        geno_ld = ctx.geno_ld
        sid = ctx.sid
        trait_col = ctx.trait_col
        ld_decay_kb = ctx.ld_decay_kb
        haplo_df_auto = ctx.haplo_df_auto

        # LD-block detection auto-reruns on any parameter change (no run/refresh
        # checkbox): the parameter controls (in the Settings panel) render unconditionally,
        # and detection is cached by a session-state hash-guard (ld_key, mirroring the
        # hap_key guard below and the page-level needs_rebuild), so it recomputes only
        # when the data or a detection parameter changes.

        # Detection parameters the CLI has no flag for are fixed here so the GUI
        # exposes exactly what the CLI exposes (one control per flag), and a GUI
        # run is reproducible from the command line. adj_r2_min (0.2) is measured
        # inert -- the adaptive floor min(0.5, max(floor, 0.5*median adj r2)) never
        # binds on real blocks; gap_factor (10.0) has no CLI flag; min_snps (2) is
        # hardcoded in cli.py's detection call.
        st.session_state["adj_r2_min"] = 0.2   # consumed by Post_GWAS_Analysis.py

        # Cache key: the data version (reuse the page's gwas_hash, cheap; fall back to
        # hashing gwas_df) + every detection parameter, incl. the fixed literals so a
        # future change to them re-keys, + the table's identity. Detection recomputes only
        # on a key change; the session keeps the most recent detections only.
        _gwas_sig = st.session_state.get("haplo_df_auto_hash", {}).get(trait_col)
        if _gwas_sig is None:
            _gwas_sig = _hash_df(gwas_df) if isinstance(gwas_df, pd.DataFrame) else "nogwas"
        ld_key = guipost.detection_key(trait_col, ld_threshold_auto, flank_kb_auto, top_n,
                                       sig_thresh, ld_decay_kb, _gwas_sig, guipost.table_key(ctx))

        if ld_key not in st.session_state:
            with st.spinner("Finding LD blocks around significant SNPs…"):
                # the table is read window by window, each window the float64 copy the
                # whole-table conversion gave. At the defaults, with every sample aligned,
                # this is the page's automatic detection, whose result is served
                # (gwas.guipost.SharedDetection)
                haplo_df_auto = guipost.SharedDetection(
                    st.session_state, find_ld_clusters_genomewide, guipost.table_key(ctx),
                ).find_ld_clusters_genomewide(
                    gwas_df=gwas_df,
                    chroms=chroms,
                    positions=positions,
                    geno_imputed=geno_ld,
                    sid=sid,
                    ld_threshold=ld_threshold_auto,
                    flank_kb=flank_kb_auto,
                    min_snps=2,
                    top_n=top_n,
                    sig_thresh=sig_thresh,
                    max_dist_bp=None,
                    ld_decay_kb=ld_decay_kb,
                    adj_r2_min=0.2,
                    gap_factor=10.0,
                )

                # Mega-block containment is structurally impossible after the
                # disjoint-block redesign: _occupancy_select claims each accepted
                # span and rejects any candidate overlapping a claimed marker. This is
                # now an invariant CHECK on a copy, never a silent removal -- the block
                # set passes through unchanged.
                _cf, _ = filter_contained_blocks(
                    haplo_df_auto.copy(), min_contained=2,
                    size_ratio_threshold=3.0, mode="flag")
                if "is_mega_block" in _cf.columns and bool(_cf["is_mega_block"].any()):
                    _mb = _cf[_cf["is_mega_block"]]
                    logging.getLogger(__name__).warning(
                        "LD containment detected (should be impossible post-occupancy): %s",
                        [f"{r['Chr']}:{int(r['Start (bp)'])}-{int(r['End (bp)'])}"
                         for _, r in _mb.iterrows()])
                    st.caption(
                        "⚠ Nested LD blocks detected. This should not occur after the "
                        "disjoint-block redesign; the block set is left unchanged. "
                        "Please report (see _occupancy_select's span_idx.size==0 short-circuit).")

            st.session_state["ld_blocks"] = haplo_df_auto.copy()

            # --- LD metadata logging ---
            st.session_state["ld_block_metadata_auto"] = {
                "method": "peak-centric LD blocks",
                "ld_threshold_r2": float(ld_threshold_auto),
                "flank_kb": float(flank_kb_auto),
                "min_snps_per_block": 2,
                "top_n_snps": int(top_n),
                "sig_threshold": float(sig_thresh),
                "ld_decay_kb": None if ld_decay_kb is None else float(ld_decay_kb),
                "adjacent_r2_split": 0.2,
            }
            st.session_state[ld_key] = haplo_df_auto
            guipost.remember_family(st.session_state, "_lddetect_order", ld_key,
                                    guipost.RESULT_CACHE_MAX)
        else:
            haplo_df_auto = st.session_state[ld_key]
        # The blocks on screen, detected now or served from the tab's cache, are the trait's
        # blocks the other tabs use (haplo_df_auto[trait]: they read the page's context, built
        # from it before this tab, so from the next run on) and the final ones the GWAS page's
        # block stability aggregates. A hit writes them too, so neither is ever an earlier
        # detection's (other settings).
        if "haplo_df_auto" not in st.session_state:
            st.session_state["haplo_df_auto"] = {}
        guipost.keep_recent(st.session_state["haplo_df_auto"], trait_col, haplo_df_auto,
                            guipost.TRAIT_CACHE_MAX)
        st.session_state["ld_blocks_final"] = haplo_df_auto.copy()
        # which trait and table they are of (the GWAS page's block stability checks)
        st.session_state["ld_blocks_final_key"] = (trait_col, ctx.table_identity)

        # Show clusters if available
        if not isinstance(haplo_df_auto, pd.DataFrame) or haplo_df_auto.empty:
            st.info(
                "No LD blocks detected: no markers pass the seed threshold. Lower it "
                "in Settings, or check that the GWAS produced significant hits."
            )
        else:
            # every detected block is a row of "LD blocks and haplotype tests" below
            st.success(f"Found {haplo_df_auto.shape[0]} LD blocks.")

        # ============================================================
        # Stability of LD block discovery
        # --------------------------------------------------------
        # Haplotype / multi-locus genotype analysis
        # --------------------------------------------------------
        st.markdown("### Haplotype / multi-locus genotype analysis")

        # Haplotype / MLG analysis runs unconditionally (no checkbox): it is cached by
        # the hap_key session-state guard in _render_haplotype_gwas, so it computes once
        # per (dataset, block set, parameters) and is served from session state after.
        _render_haplotype_gwas(
            ctx=ctx,
            haplo_df_auto=haplo_df_auto,
            cached_extract_block_geno=cached_extract_block_geno,
            compute_block_qc_effects=compute_block_qc_effects,
            run_haplotype_block_gwas_cached_fn=run_haplotype_block_gwas_cached_fn,
            settings=settings,
        )

    except StopException:
        pass


# The "?" next to "LD blocks and haplotype tests": how to read the effect sizes and the
# LD-block coherence columns of the table under it.
ASSOCIATION_RESULTS_HELP = (
    "**Effect sizes (η²)**\n\n"
    "- **Variance explained (η²)** per LD block: the most interpretable measure "
    "of QTL importance for breeding:\n"
    "  - **> 10%** major QTL, strong marker-assisted-selection candidate\n"
    "  - **5-10%** moderate QTL, useful combined with other loci\n"
    "  - **< 5%** minor QTL, polygenic background\n"
    "- η² is the haplotype test's: with the kinship-aware test (the default) it comes from "
    "the test's F on the kinship-whitened model, and **h² (null model)** is that model's "
    "heritability; with the PCs-only test it is not adjusted for kinship and can be inflated "
    "in structured panels. **η² (PCs-only test)** gives that test's value for comparison.\n\n"
    "**LD-block coherence (r²)**\n\n"
    "- **Mean r² / Min r²**: mean / weakest member-pair r² in the block "
    "(how internally correlated it is).\n"
    "- **Frac r² ≥ 0.6 to lead**: fraction of members with r² ≥ 0.6 to the lead "
    "SNP. A low value on a lead's own block means the interval is a loose merge "
    "of sub-clusters, not one tight haplotype.\n\n"
    "Full guidance: **Help → Interpreting Results**."
)


def _block_intervals(ctx, blocks):
    """The blocks' candidate intervals over the run's post-QC markers, by the CLI's rule
    (gwas.isolated.block_candidate_intervals); None if they cannot be computed."""
    try:
        from gwas.isolated import block_candidate_intervals, edge_flank_bp_from_decay
        return block_candidate_intervals(blocks, ctx.chroms, ctx.positions, ctx.sid,
                                         edge_flank_bp=edge_flank_bp_from_decay(ctx.ld_decay_kb))
    except Exception:
        logging.exception("block candidate intervals for the block table")
        return None


def _render_haplotype_gwas(
    ctx,
    haplo_df_auto,
    cached_extract_block_geno,
    compute_block_qc_effects,
    run_haplotype_block_gwas_cached_fn,
    settings=None,
):
    """Sub-section: the haplotype analysis (its parameters are in the Settings panel),
    execution, and visualization."""
    # ------------------------------------------------------------------
    # Drift guard: do the genotypes and PCs on this page belong to the run on screen?
    # ------------------------------------------------------------------
    # `geno_row_ids` is written ONLY by a completed GWAS run (GWAS_analysis.py, the
    # `st.session_state["geno_row_ids"] = np.asarray(geno_df.index, dtype=str).tolist()`
    # in the export block), while the table this page reads (geno_table, whose own
    # samples are ctx.geno_df's index) is rebuilt on EVERY rerun of the GWAS page. A
    # disagreement therefore means the GWAS page rebuilt the table for another sample set
    # without completing a run, so the blocks on screen and the genotypes no longer come
    # from the same analysis, which cannot be repaired here.
    # np.array_equal, NOT a size test: same-length/different-ids is the silently-wrong
    # case -- extract_block_geno_for_paper() checks the mask SIZE only, so such a mask
    # passes its guard and mis-pairs rows positionally.
    _ids_last_run = np.asarray(st.session_state.get("geno_row_ids", []), dtype=str)
    _ids_now = np.asarray(ctx.geno_df.index, dtype=str)
    if _ids_last_run.size and not np.array_equal(_ids_last_run, _ids_now):
        _run_trait = st.session_state.get("gwas_run_summary", {}).get("trait") or ctx.ld_trait
        _sel = st.session_state.get("selected_traits_multiselect") or []
        _loaded_trait = str(_sel[0]) if _sel else None
        _loaded = (
            f"`{_loaded_trait}` ({_ids_now.size} samples)" if _loaded_trait
            else f"a different trait ({_ids_now.size} samples)"
        )
        st.error(
            f"The genotype data loaded on this page was rebuilt for {_loaded}, but the "
            f"results shown are from the GWAS on `{_run_trait}` ({_ids_last_run.size} "
            "samples). Haplotype statistics computed now would not match that analysis."
            f"\n\nRe-run the GWAS on `{_run_trait}`, then reopen Post-GWAS Analysis."
        )
        # st.stop() (NOT raise): StopException subclasses BaseException and is caught by
        # render()'s `except StopException`, so the tab ends cleanly instead of showing
        # a traceback.
        st.stop()
    # The PCs (the haplotype test, the window R2, the PCA panel) are the GWAS page's of this
    # table: the page takes them from GWAS_CTX, written with the table (gwas.guipost.run_pcs), not
    # from the session's "pcs" / "pcs_full", which a GWAS page run stopped before the table
    # leaves beside it. PCs of other samples would pair rows wrongly, or fail the test with
    # a traceback: the tab stops plainly instead, as above.
    if not guipost.pcs_of_samples(st.session_state, _ids_now, ctx.pcs,
                                  getattr(ctx, "pcs_full", None)):
        _run_trait = (st.session_state.get("gwas_run_summary") or {}).get("trait") or ctx.ld_trait
        st.error(
            "The principal components on this page are not those of the genotype table's "
            f"{_ids_now.size} samples, so haplotype statistics computed now would not match "
            f"the analysis.\n\nRe-run the GWAS on `{_run_trait}`, then reopen Post-GWAS "
            "Analysis."
        )
        st.stop()
    # The user covariates the haplotype test adjusts for (as the CLI's does) are the same
    # run's, written with the PCs (gwas.guipost.run_covariates), and checked the same way.
    _user_covar = getattr(ctx, "user_covar", None)
    if not guipost.pcs_of_samples(st.session_state, _ids_now, _user_covar):
        _run_trait = (st.session_state.get("gwas_run_summary") or {}).get("trait") or ctx.ld_trait
        st.error(
            "The covariates on this page are not those of the genotype table's "
            f"{_ids_now.size} samples, so haplotype statistics computed now would not match "
            f"the analysis.\n\nRe-run the GWAS on `{_run_trait}`, then reopen Post-GWAS "
            "Analysis."
        )
        st.stop()

    trait_col = ctx.trait_col
    ld_trait = ctx.ld_trait
    _gctx_norm = (st.session_state.get("GWAS_CTX") or {})
    if _gctx_norm.get("trait") == ld_trait and _gctx_norm.get("norm") not in (None, "none"):
        from pages._pheno_qc.stats import SLUG_NAMES as _SLUG_NAMES
        st.caption(
            f"The haplotype tests use the uploaded values of {ld_trait}; the GWAS scans "
            f"used {_SLUG_NAMES.get(_gctx_norm['norm'], _gctx_norm['norm'])} for it (as "
            "cli.py --norm does).")
    geno_ld = ctx.geno_ld
    geno_df = ctx.geno_df
    chroms = ctx.chroms
    positions = ctx.positions
    sid = ctx.sid
    pcs = ctx.pcs
    trait_col_resolved = ctx.ld_trait

    # the two minimum-sample settings (Settings panel; the CLI's defaults otherwise)
    settings = settings or {}
    min_hap_count = int(settings.get("min_hap_count", 5))
    min_group_size = int(settings.get("min_group_size", 3))

    if run_haplotype_block_gwas_cached_fn is None:
        st.error(
            "Haplotype analysis functions are not available. "
            "Check `GWAS_analysis.py` imports."
        )
        return

    haplo_blocks_to_use = haplo_df_auto

    if haplo_blocks_to_use is None or haplo_blocks_to_use.empty:
        st.warning("No LD blocks to test. Detection above found none.")
        return

    # ------------------------------------------------------------
    # Use GWAS-aligned phenotype for haplotype GWAS (MANDATORY)
    # ------------------------------------------------------------
    pheno_for_hap = None

    if "pheno_used_for_hap" in st.session_state:
        pheno_for_hap = st.session_state["pheno_used_for_hap"]
    elif "pheno_aligned" in st.session_state:
        pheno_for_hap = st.session_state["pheno_aligned"]
    elif "pheno" in st.session_state:
        pheno_for_hap = st.session_state["pheno"]

    if pheno_for_hap is None:
        st.error("No phenotype available for haplotype analysis.")
        st.stop()

    pheno_for_hap = pheno_for_hap.copy()
    pheno_label = "GWAS-aligned"

    # Store for downstream visualization
    st.session_state["pheno_used_for_hap"] = pheno_for_hap
    st.session_state["trait_col"] = trait_col
    st.session_state["trait_col_locked"] = trait_col
    st.session_state["pheno_used_for_hap_label"] = pheno_label

    st.session_state["haplo_blocks_to_use"] = haplo_blocks_to_use
    st.session_state["pcs_for_hap"] = pcs

    # ---------------------------------------------
    # Avoid recomputation unless inputs change
    # ---------------------------------------------
    blocks_hash = _hash_df(haplo_blocks_to_use.reset_index(drop=True))
    pheno_hash = _hash_df(pheno_for_hap[[trait_col_resolved]].copy())
    n_perm = int(st.session_state.get("n_perm_hap", 1000))

    # The test (Settings; the kinship-aware one by default, as in the CLI). It takes the
    # kernels of the GWAS page's Run GWAS on this table, recorded with its results
    # (gwas.guipost.run_kernels), never the session's current ones or another run's; without
    # them the tab stops plainly (a new Run, or the PCs-only test).
    hap_test = str(settings.get("hap_test", "kinship"))
    hap_kernels, hap_use_loco = None, None
    if hap_test == "kinship":
        _rk = guipost.run_kernels(st.session_state, guipost.table_key(ctx), _ids_now)
        if _rk is None:
            _run_trait = (st.session_state.get("gwas_run_summary") or {}).get("trait") or ld_trait
            st.error(guipost.NO_RUN_KERNELS_MESSAGE.format(trait=_run_trait))
            st.stop()
            return
        hap_kernels, hap_use_loco = (_rk[0], _rk[1]), _rk[2]

    # The result's session key names the genotype table by its identity (two tables of the
    # same shape are two keys), with the trait, the settings, the PCs, the blocks, the
    # phenotype, and the test with its kernel (LOCO or global).
    hap_key = guipost.haplotype_key(trait_col, min_hap_count, min_group_size, pcs,
                                    guipost.table_key(ctx), n_perm, blocks_hash, pheno_hash,
                                    hap_test=hap_test, use_loco=hap_use_loco)

    # a result is served from the session only with its own per-block tables (the
    # assignments and the block plots read them); without them it is computed again
    if hap_key not in st.session_state or f"{hap_key}::tables" not in st.session_state:

        with st.spinner("Running haplotype / MLG analysis on LD blocks…"):

            hap_gwas_df, hap_tables = run_haplotype_block_gwas_cached_fn(
                haplo_blocks_to_use,
                chroms,
                positions,
                geno_ld,
                sid,
                geno_df,
                pheno_for_hap,
                ld_trait,
                pcs,
                int(min_hap_count),
                int(min_group_size),
                int(n_perm),
                st.session_state.get("geno_encoding", "dosage012"),
                st.session_state.get("n_pcs_used", None),
                user_covar=_user_covar,
                hap_test=hap_test,
                kernels=hap_kernels,
            )
            st.session_state["hap_tables"] = hap_tables
            st.session_state["hap_gwas_df"] = hap_gwas_df
        st.session_state[hap_key] = hap_gwas_df
        # the per-block sample tables of THIS result, kept with it (the assignments
        # table, the downloads and the MLG plots read them), and its assignments built
        # anew. Every remembered result keeps its own: a result visited again (the test,
        # or the Run's LOCO choice, switched back) has its block plots and assignments
        st.session_state[f"{hap_key}::tables"] = hap_tables
        st.session_state.pop(f"{hap_key}::assign", None)
        # the session keeps the most recent results only, each with its tables and
        # assignments (dropped with it)
        guipost.remember_family(st.session_state, "_hapgwas_order", hap_key,
                                guipost.RESULT_CACHE_MAX, suffixes=("::tables", "::assign"))

    else:
        hap_gwas_df = st.session_state[hap_key]
    hap_tables = st.session_state.get(f"{hap_key}::tables", {})
    # why each untested block was left out (recorded by the test, in its result's attrs)
    _untested = dict(getattr(hap_gwas_df, "attrs", {}).get("untested", {}) or {})

    # LD-block coherence: the triage quality layer for every detected block (Mean r2 is
    # the same helper as the block's own, so one coherence number). Raw dosage is used
    # when present (higher fidelity than the imputed fallback).
    _ldq_full = None
    try:
        from gwas.ld import compute_block_ld_quality as _cblq
        _ldq_full = _cblq(haplo_df_auto, chroms, positions, sid, geno_ld, ctx.gwas_df,
                          geno_dosage_raw=ctx.geno_dosage_raw, r2_coherent=0.6)
    except Exception as _e:
        if guipost.table_released(st.session_state.get("geno_table")):
            # the table file was closed during this read: the page's plain message, not the
            # read error (which names the file)
            st.info(guipost.RELEASED_MESSAGE)
            st.stop()
        st.caption(f"LD-block coherence unavailable: {_e}")

    _seed_p = (settings or {}).get("seed_p")

    def _show_block_table(tested_df):
        """One row per detected block: the block, its haplotype test, its coherence."""
        from .block_table import WINDOW_R2_HELP, merge_block_table, screen_view
        _tbl = merge_block_table(haplo_df_auto, tested_df, _ldq_full, untested=_untested,
                                 seed_p=_seed_p, intervals=_block_intervals(ctx, haplo_df_auto))
        st.subheader("LD blocks and haplotype tests", help=ASSOCIATION_RESULTS_HELP)
        _all = st.toggle("Show all columns", value=False, key="block_table_all_columns")
        st.dataframe(screen_view(_tbl, _all), use_container_width=True)
        st.download_button(
            "Download block table (CSV)", _tbl.to_csv(index=False).encode("utf-8"),
            file_name=f"LD_blocks_and_haplotype_tests_{ld_trait}.csv", mime="text/csv",
            key="dl_block_table", help=WINDOW_R2_HELP,
        )

    # ------------------------------------------------------------
    # Block-level multiple testing correction (FDR)
    # ------------------------------------------------------------
    from statsmodels.stats.multitest import multipletests

    # checked before the FDR step: multipletests raises on an empty table
    if hap_gwas_df is None or hap_gwas_df.empty:
        _show_block_table(None)
        st.info(
            "No haplotype blocks had enough variation "
            "for association testing."
        )
        return

    # Sample-to-haplotype assignments of the tested blocks, from the test's own tables
    # and inputs: the same function and inputs as the CLI and One-Click ZIP member (the
    # test uses the uploaded values, so no Phenotype_raw). Built once per result.
    _assign_key = f"{hap_key}::assign"
    if _assign_key not in st.session_state:
        try:
            from gwas.haplotype import haplotype_assignments as _hap_assign
            _frame = _hap_assign(hap_gwas_df, hap_tables, geno_df, pheno_for_hap, ld_trait)
            st.session_state[_assign_key] = (
                _frame, _frame.to_csv(index=False).encode("utf-8"), None)
        except Exception as _assign_err:
            st.session_state[_assign_key] = (None, None, str(_assign_err))
    hap_assignments, hap_assign_csv, hap_assign_error = st.session_state[_assign_key]
    if hap_assign_error:
        st.caption(f"Sample-to-haplotype assignments unavailable: {hap_assign_error}")

    pvals = hap_gwas_df["PValue"].astype(float).values
    if hap_test == "pcs":
        _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
    else:
        # over the blocks with a p-value: a block the kinship-aware test could not test has
        # none (its kinship_model gives the reason), as the test's own FDR_BH
        from gwas.haplotype_kinship import bh_qvalues
        qvals = bh_qvalues(pvals)
    hap_gwas_df["FDR_qvalue"] = qvals

    # Enrich BEFORE display so Frac_other/Block_length_kb/MAF reach the
    # on-screen table and the window R² the CSV, not only the downloaded CSV/ZIP.
    with st.spinner("Computing block lengths, MAF and the window R²…"):
        hap_gwas_df = compute_block_qc_effects(
            hap_gwas_df=hap_gwas_df,
            chroms=chroms,
            positions=positions,
            geno_imputed=geno_ld,
            geno_df=geno_df,
            pheno_for_hap=pheno_for_hap,
            trait_col=ld_trait,
            min_hap_count=int(min_hap_count),
            min_group_size=int(min_group_size),
            compute_maf=True,
            geno_encoding=st.session_state.get("geno_encoding", "dosage012"),
        )
    st.session_state["hap_gwas_df_enriched"] = hap_gwas_df

    # the blocks the kinship-aware test could not test are rows without statistics (their
    # kinship_model says why): not among the blocks it ran on (the PCs-only test's "none")
    _n_failed = int(hap_gwas_df.get("kinship_model", pd.Series(dtype=object))
                    .astype(str).str.startswith("failed").sum())
    if _n_failed:
        st.success(
            f"Haplotype / MLG analysis ran on {hap_gwas_df.shape[0] - _n_failed} LD blocks; "
            f"the kinship-aware test could not test {_n_failed} more (the block table's "
            f"Tested column says why)."
        )
    else:
        st.success(
            f"Haplotype / MLG analysis ran on {hap_gwas_df.shape[0]} LD blocks."
        )
    # every detected block, its haplotype test and its coherence: one table, one CSV
    _show_block_table(hap_gwas_df)

    # Lead-SNP genotype-effect gallery: browse any significant SNP (threshold selector
    # + expander picker) above the per-block MLG visualization below. Module-level so it
    # runs even when _render_block_visualization is stubbed out in the AppTest.
    _render_lead_snp_gallery(ctx, hap_gwas_df, haplo_df_auto, hap_assignments=hap_assignments,
                             settings=settings)

    # ============================================================
    # BLOCK-LEVEL VISUALIZATION: MLG boxplots + Tukey HSD + Table
    # ============================================================
    _render_block_visualization(
        ctx=ctx,
        hap_gwas_df=hap_gwas_df,
        haplo_df_auto=haplo_df_auto,
        min_hap_count=int(min_hap_count),
        min_group_size=int(min_group_size),
        cached_extract_block_geno=cached_extract_block_geno,
        hap_assignments=hap_assignments,
        hap_assign_csv=hap_assign_csv,
        hap_assign_error=hap_assign_error,
    )


def _render_snp_effect(ctx, snp_id, chrom, pos, trait_col, geno_col, y, samp, geno_df, n_perm,
                       hap_kwargs=None, data_kwargs=None):
    """Render one SNP's genotype-class boxplot + one marginal rank-test caption.

    ``geno_col`` / ``y`` are already masked to the SNP's non-missing, phenotyped samples.
    The boxplot title (from ``render_snp_boxplot``) already carries the kinship-adjusted MLM
    effect size and the per-SNP variance explained; the caption below adds a single marginal
    genotype-class rank test -- Mann-Whitney U for two non-empty classes, Kruskal-Wallis for
    three -- disclosed as adjusting for neither population structure nor kinship. ``chrom`` /
    ``pos`` / ``trait_col`` / ``samp`` / ``geno_df`` / ``n_perm`` are retained in the
    signature but unused now that the per-SNP Freedman-Lane test and its PC covariates are gone.
    ``hap_kwargs`` goes to ``render_snp_boxplot``: the haplotype labels of the block that
    contains the SNP (or the note why there are none) and the phenotype axis label.
    ``data_kwargs`` goes to ``snp_boxplot_data`` (the samples' haplotypes, test flags,
    alleles and the phenotype's scale): the interactive plot and the CSV under it show
    exactly the samples and values the static plot draws.
    """
    from gwas.snpplots import (plotly_snp_boxplot, render_snp_boxplot, snp_boxplot_data,
                               snp_boxplot_title)
    import matplotlib.pyplot as _plt

    # Effect sizes READ from the GWAS results (guard each column -> None where absent); they
    # annotate the boxplot title (kinship-adjusted MLM effect + eta2). The renderer prints
    # its own "structure-unadjusted" note when they are missing.
    _bm = _se = _bo = None
    _gdf = ctx.gwas_df
    if isinstance(_gdf, pd.DataFrame) and "SNP" in _gdf.columns:
        _r = _gdf.loc[_gdf["SNP"].astype(str) == str(snp_id)]
        if not _r.empty:
            _r0 = _r.iloc[0]

            def _num(_col):
                _v = _r0.get(_col)
                try:
                    _f = float(_v)
                    return _f if np.isfinite(_f) else None
                except (TypeError, ValueError):
                    return None
            _bm, _se, _bo = _num("Beta_MLM"), _num("SE_MLM"), _num("Beta_OLS")

    with st.expander("Plot style", expanded=False):
        _style = st.radio(
            "Visualization", ("Interactive", "Static"), horizontal=True,
            key="lead_snp_viz_type",
            help=("Interactive (Plotly) shows each sample's ID, genotype, plotted value and "
                  "haplotype on hover. Static (matplotlib) is the report's figure."),
        )
    _hk = dict(hap_kwargs or {})
    _data = snp_boxplot_data(geno_col, y, samp, **(data_kwargs or {}))
    if _style == "Interactive":
        _pfig = plotly_snp_boxplot(
            _data, snp_boxplot_title(str(snp_id), geno_col, y, beta_mlm=_bm, se_mlm=_se,
                                     beta_ols=_bo),
            y_label=_hk.get("y_label"), hap_color_map=_hk.get("hap_color_map"))
        show_plot(_pfig, "small", key=f"lead_snp_plotly_{snp_id}")
        if _hk.get("hap_note"):
            st.caption(_hk["hap_note"])
    else:
        _fig = render_snp_boxplot(str(snp_id), geno_col, y, beta_mlm=_bm, se_mlm=_se,
                                  beta_ols=_bo, **_hk)
        show_plot(_fig, "small")
        _plt.close(_fig)
    st.download_button(
        "Download plotted data (CSV)", _data.to_csv(index=False).encode("utf-8"),
        file_name=f"lead_snp_{snp_id}_{trait_col}.csv", mime="text/csv",
        key=f"dl_lead_snp_{snp_id}",
    )

    # Marginal genotype-class rank test -- adjusts for NEITHER population structure NOR
    # kinship. Mann-Whitney U for two non-empty classes, Kruskal-Wallis for three; no minimum
    # class size, entering/excluded classes named, scipy failures surfaced not faked. Cheap
    # (~0.5 ms), so computed live (no cache).
    _g_int = np.rint(np.asarray(geno_col, dtype=float)).astype(int)
    _yv = np.asarray(y, dtype=float)
    _cls = {c: _yv[_g_int == c] for c in (0, 1, 2)}
    _present = [c for c in (0, 1, 2) if _cls[c].size > 0]
    _absent = [c for c in (0, 1, 2) if _cls[c].size == 0]
    _cls_txt = ",".join(str(c) for c in _present)
    _marg_stat, _marg_p = "", None
    if len(_present) >= 3:
        from scipy.stats import kruskal
        _marg_test = f"Kruskal-Wallis (classes {_cls_txt})"
        try:
            _stat, _marg_p = kruskal(*[_cls[c] for c in _present])
            _marg_stat = f"H = {float(_stat):.2f}"
        except Exception:
            _marg_stat, _marg_p = "test failed", None
    elif len(_present) == 2:
        from scipy.stats import mannwhitneyu
        _excl = f"; {','.join(str(c) for c in _absent)} empty" if _absent else ""
        _marg_test = f"Mann-Whitney U (classes {_cls_txt}{_excl})"
        try:
            _stat, _marg_p = mannwhitneyu(
                _cls[_present[0]], _cls[_present[1]], alternative="two-sided")
            _marg_stat = f"U = {float(_stat):.1f}"
        except Exception:
            _marg_stat, _marg_p = "test failed", None
    else:
        _marg_test = f"no test ({len(_present)} non-empty class)"

    # One caption -- middle-dot separated, noun phrases, no dashes; the trailing phrase is
    # the whole disclosure (the test adjusts for neither structure nor kinship).
    _parts = [_marg_test]
    if _marg_stat:
        _parts.append(_marg_stat)
    if _marg_p is not None:
        _parts.append(f"p = {_marg_p:.3g}")
    _parts.append("no population structure or kinship adjustment")
    st.caption(" · ".join(_parts))


def _render_lead_snp_gallery(ctx, hap_gwas_df, haplo_df_auto, hap_assignments=None,
                             settings=None):
    """Browsable per-SNP genotype-effect gallery over the significant SNPs.

    A threshold selector (M_eff / Bonferroni / FDR / Custom, via ``rule_from_streamlit``)
    picks the significant set from ``ctx.gwas_df``; an expander SNP-picker (sorted by p,
    tagged lead-of / in-block / unblocked) renders the ONE chosen SNP through the shared
    ``_render_snp_effect`` helper. Complementary to the per-block panel below (which
    auto-shows the selected block's lead); this browses ALL significant SNPs. The dots are
    coloured by the haplotypes (``hap_assignments``, the tab's haplotype test) of the LD
    block that contains the SNP, as in the CLI report's per-SNP plots.
    """
    gdf = ctx.gwas_df
    if not isinstance(gdf, pd.DataFrame) or gdf.empty or "SNP" not in gdf.columns:
        return
    # Genotype matrix, SNP axis and sample axis all come from ctx, so they cannot
    # disagree. The SNP axis must move with the matrix: `_ci` below is
    # a column index into it, so a stale sid/positions array would silently select the
    # wrong marker.
    geno_hard = ctx.geno_hard
    if geno_hard is None:
        return

    from gwas.significance import frame_for_rule, rule_from_streamlit
    from .tab_significant_snps import _SIG_LABELS

    st.markdown(
        "### Lead SNP genotype effect",
        help=("Browse the genotype effect of any significant SNP: pick a SNP below. Its "
              "box plot shows the phenotype the GWAS used per genotype class (0, 1 or 2 "
              "copies of the ALT allele), coloured by haplotype when the SNP lies in a "
              "tested LD block.\n\n"
              "**β_MLM ± SE** in the title is the GWAS's mixed-model effect, adjusted for "
              "kinship and any PCs the GWAS used: the change in the plotted phenotype per "
              "copy of the ALT allele. A positive β means the ALT allele raises the trait. "
              "The flag next to η² compares it with β_OLS, the effect without kinship: "
              "concordant, structure-sensitive (β_MLM under half of β_OLS) or "
              "sign-flip.\n\n"
              "**η²** in the title is the genotype classes' share of the plotted "
              "phenotype's variance, not adjusted for kinship. The caption under the plot "
              "gives one marginal rank test."),
    )

    # the rule is in the Settings panel (the GWAS page's rule unless changed there)
    if settings is not None:
        _rule_label, _custom = settings["gallery_rule"], settings.get("gallery_custom")
    else:
        _default = getattr(ctx, "sig_rule_label", "") or ""
        _rule_label = _default if _default in _SIG_LABELS else _SIG_LABELS[1]
        _custom = (float(ctx.custom_thresh) if ctx.custom_thresh else 5e-8) \
            if _rule_label.startswith("Custom") else None
    st.caption(f"Significant SNPs by the {_rule_short_name(_rule_label, _custom)} rule "
               "(change it in Settings above).")
    _rule = rule_from_streamlit(_rule_label, int(len(gdf)), ctx.meff_val, custom_thresh=_custom)
    # frame_for_rule: this gallery's own custom p-value, not a stored Significant_Custom
    _sig = gdf.loc[_rule.significant_mask(frame_for_rule(gdf, _rule))].copy()
    if _sig.empty:
        st.info("No SNPs pass the selected significance rule.")
        return

    # Tag each significant SNP by its LD-block relationship.
    _lead_map, _member_map = {}, {}
    if isinstance(haplo_df_auto, pd.DataFrame):
        for _bi, _br in haplo_df_auto.reset_index(drop=True).iterrows():
            _blk = f"block {_bi + 1}"
            _ld = str(_br.get("lead_snp", "") or "")
            if _ld:
                _lead_map.setdefault(_ld, _blk)
            for _m in str(_br.get("SNP_IDs", "") or "").split(","):
                _m = _m.strip()
                if _m:
                    _member_map.setdefault(_m, _blk)

    def _status(_snp):
        if _snp in _lead_map:
            return f"lead of {_lead_map[_snp]}"
        if _snp in _member_map:
            return f"in {_member_map[_snp]}"
        return "unblocked"

    _sig = _sig.sort_values("PValue", kind="stable")
    _snp_ids = _sig["SNP"].astype(str).tolist()
    _labels = [
        f"{s}  p={float(p):.1e}  [{_status(s)}]"
        for s, p in zip(_snp_ids, _sig["PValue"].tolist())
    ]

    with st.expander(f"Pick a significant SNP ({len(_snp_ids)} pass the rule)", expanded=False):
        _pick = st.selectbox("SNP:", _labels, key="lead_gallery_snp")
        _chosen = _snp_ids[_labels.index(_pick)]

        _sid_arr = np.asarray(ctx.sid).astype(str)
        _idx_arr = np.where(_sid_arr == _chosen)[0]
        if not _idx_arr.size:
            st.warning("Chosen SNP is not in the QC'd genotype matrix; cannot render.")
            return
        _ci = int(_idx_arr[0])
        # the one SNP's hard calls (a single column is read from the table)
        _dose = np.asarray(geno_hard[:, _ci]).astype(float)

        _ph_used = st.session_state.get("pheno_used_for_hap")
        # sample axis of `geno_hard` above -- same source, so the two always agree
        _geno_ids = np.asarray(ctx.geno_df.index, dtype=str)
        if _ph_used is None or _geno_ids.size == 0:
            st.warning("Phenotype/sample alignment unavailable; run haplotype analysis first.")
            return
        _ph_ser = pd.to_numeric(_ph_used[ctx.ld_trait], errors="coerce")
        _ph_ser.index = _ph_ser.index.astype(str)
        # When the GWAS page applied a transformation to this trait, plot the phenotype
        # the GWAS model analysed (its y), so the axis and eta2 are on the scale of the
        # beta shown, as in the CLI report's per-SNP plots; raw runs are unchanged.
        _gctx = st.session_state.get("GWAS_CTX") or {}
        _plotted_norm = None          # the normalisation of the values plotted (axis label)
        if (_gctx.get("trait") == ctx.ld_trait and _gctx.get("norm") not in (None, "none")
                and _gctx.get("y") is not None and _gctx.get("geno_index") is not None):
            _ph_ser = pd.Series(np.asarray(_gctx["y"], dtype=float).ravel(),
                                index=np.asarray(_gctx["geno_index"], dtype=str))
            _plotted_norm = _gctx["norm"]
        _y_full = _ph_ser.reindex(_geno_ids).values.astype(float)
        _ok = np.isfinite(_dose) & np.isfinite(_y_full)
        if int(_ok.sum()) < 5:
            st.warning("Too few non-missing genotype/phenotype pairs for this SNP.")
            return
        _g = _dose[_ok]
        _y = _y_full[_ok]
        _samp = pd.Index(_geno_ids[_ok]).astype(str)
        _pos = int(np.asarray(ctx.positions)[_ci])
        _chr = str(_sig.loc[_sig["SNP"].astype(str) == _chosen, "Chr"].iloc[0])
        _n_perm = int(st.session_state.get("n_perm_hap", 1000))

        # the haplotype labels of the LD block that contains the SNP (by position, as the
        # significant-SNP table's Block_ID), the same labels as the assignments table
        from gwas.haplotype import snp_block_assignments
        from gwas.snpplots import HAP_NOTE_NO_BLOCK, HAP_NOTE_NO_GROUPS, phenotype_axis_label
        _rows = None
        try:
            _rows, _hb, _why = snp_block_assignments(
                haplo_df_auto, hap_assignments, np.asarray(ctx.chroms)[_ci], _pos, _samp)
            if _rows is not None:     # a sample without a haplotype counts as "Other"
                _hk = {"hap_labels": np.asarray(
                    [h if h else "Other" for h in _rows["Haplotype"].tolist()], dtype=object)}
            else:
                _hk = {"hap_note": HAP_NOTE_NO_BLOCK if _why == "no_block"
                       else HAP_NOTE_NO_GROUPS}
        except Exception:
            _rows, _hk = None, {}
        # the axis names the values plotted (the GWAS's phenotype, transformed or raw)
        _hk["y_label"] = phenotype_axis_label(ctx.ld_trait, _plotted_norm)
        _allele_map = _gctx.get("allele_map") or {}
        _data_kw = {
            "haplotypes": None if _rows is None else _rows["Haplotype"].to_numpy(dtype=object),
            "in_test": None if _rows is None else _rows["in_haplotype_test"].to_numpy(dtype=bool),
            "alleles": _allele_map.get(_chosen), "norm": _plotted_norm,
        }

        st.markdown(f"#### {_chosen} genotype effect")
        _render_snp_effect(
            ctx, _chosen, _chr, _pos, ctx.trait_col, _g, _y, _samp, ctx.geno_df, _n_perm,
            hap_kwargs=_hk, data_kwargs=_data_kw)


def _render_block_visualization(
    ctx,
    hap_gwas_df,
    haplo_df_auto,
    min_hap_count,
    min_group_size,
    cached_extract_block_geno,
    hap_assignments=None,
    hap_assign_csv=None,
    hap_assign_error=None,
):
    """Block-level MLG boxplots, Tukey HSD, forest plot, and expression panel selection.

    The groups shown are the haplotype test's own: ``hap_assignments`` is
    gwas.haplotype.haplotype_assignments of the test on screen, and the selected block's
    rows with in_haplotype_test give the samples, labels, allele sequences and phenotype
    values (PC-adjusted when the GWAS used PCs, gwas.snpplots.hap_plot_values; Tukey HSD,
    the letters, the box plot and the forest plot all use them). Its two assignments
    downloads sit below the block selector; the plot offers its plotted data.
    ``cached_extract_block_geno`` is kept for callers; the panel no longer re-derives
    the groups from a coordinate window."""
    st.markdown("### Haplotype / MLG Effect Visualization")

    trait_col = ctx.trait_col
    ld_trait = ctx.ld_trait
    geno_df = ctx.geno_df

    # ================================
    # Build enhanced block labels
    # ================================
    labels = []
    id_map = []

    # Block MEMBERSHIP (len SNP_IDs) from the source LD-blocks table, keyed by lead SNP, so
    # the selector shows the block's SNP count AND the (call-rate/monomorphic/LD-thinned)
    # marker count that actually built the MLGs -- the two diverge when markers are dropped.
    _member_ct = {}
    if isinstance(haplo_df_auto, pd.DataFrame) and "lead_snp" in haplo_df_auto.columns:
        for _, _br in haplo_df_auto.iterrows():
            _ids = str(_br.get("SNP_IDs", "") or "")
            _member_ct[str(_br["lead_snp"])] = len([s for s in _ids.split(",") if s])
    if "n_snps" not in hap_gwas_df.columns:
        st.warning(
            "Haplotype results are missing the expected 'n_snps' column; the block "
            "selector cannot show the per-block MLG-marker count."
        )

    for idx, row in hap_gwas_df.iterrows():
        block_id = row["Block_ID"] if "Block_ID" in hap_gwas_df.columns else idx
        chr_ = str(row["Chr"])
        start = int(row["Start"])
        end = int(row["End"])

        n_mlg = int(row["n_snps"]) if "n_snps" in hap_gwas_df.columns else 0
        n_members = _member_ct.get(str(row.get("lead_snp", "")), n_mlg)
        _mlg_txt = "" if n_members == n_mlg else f" ({n_mlg} used for MLG)"

        pval = float(row["PValue"])
        logp = -np.log10(max(pval, 1e-300))

        label = (
            f"Block {block_id}, Chr{chr_}: "
            f"{start / 1e6:.2f}-{end / 1e6:.2f} Mb | "
            f"{n_members} SNPs{_mlg_txt} | -log10(p)={logp:.2f}"
        )

        labels.append(label)
        id_map.append(str(block_id))

    # -------------------------------
    # Selector with informative labels
    # -------------------------------
    selected_label = st.selectbox(
        "Select LD block for phenotype-haplotype visualization:",
        options=labels,
    )

    selected_block = id_map[labels.index(selected_label)]

    # Retrieve the block row
    if "Block_ID" in hap_gwas_df.columns:
        block_row = hap_gwas_df.loc[
            hap_gwas_df["Block_ID"].astype(str) == selected_block
        ].iloc[0]
    else:
        block_row = hap_gwas_df.loc[int(selected_block)]

    block_chr = str(block_row["Chr"])
    block_start = int(block_row["Start"])
    block_end = int(block_row["End"])

    st.info(f"Selected block: Chr{block_chr}:{block_start:,}-{block_end:,}")

    # --------------------------------------------------------
    # This block's sample-to-haplotype assignments: the haplotype test's own samples,
    # labels and allele sequences, so the plots below show exactly the groups it tested
    # --------------------------------------------------------
    _blk = None
    if isinstance(hap_assignments, pd.DataFrame) and not hap_assignments.empty:
        _blk = hap_assignments[
            (hap_assignments["Chr"].astype(str) == block_chr)
            & (hap_assignments["Block_start"].astype(int) == block_start)
            & (hap_assignments["Block_end"].astype(int) == block_end)
        ]
    if _blk is None or _blk.empty:
        # the plots need the test's groups; the results downloads below do not
        st.warning("The haplotype plots for this block are not shown: "
                   + (f"its sample-to-haplotype assignments could not be built "
                      f"({hap_assign_error})." if hap_assign_error else
                      "the haplotype test kept no per-sample table for it."))
        return

    # the block's sample-to-haplotype assignments and every block's sit here, under the
    # block selector; the plot below offers its own plotted data
    _block_csv = (
        _blk.to_csv(index=False).encode("utf-8"),
        f"haplotype_assignments_block_{block_chr}_{block_start}_{block_end}_{ld_trait}.csv",
        f"dl_hap_assign_{selected_block}",
    )
    _block_assignments_button(_block_csv)
    # the ZIP member's name for MLM; another downstream model's blocks are named so
    _model = str(st.session_state.get("downstream_model") or "MLM")
    st.download_button(
        "Download assignments for all blocks (CSV)",
        (hap_assign_csv if hap_assign_csv is not None
         else hap_assignments.to_csv(index=False).encode("utf-8")),
        file_name=(f"Haplotype_assignments_{ld_trait}.csv" if _model == "MLM"
                   else f"Haplotype_assignments_{ld_trait}_{_model}.csv"),
        mime="text/csv",
        key="dl_hap_assign_all",
    )

    _tested = _blk[_blk["in_haplotype_test"].astype(bool)]
    geno_sample_ids = np.asarray(ctx.geno_df.index, dtype=str)
    hap_strings = pd.Series(_tested["Allele_sequence"].astype(str).to_numpy(),
                            index=pd.Index(_tested["Sample"].astype(str)),
                            name="Allele_sequence")
    block_sids = np.asarray(str(_blk["SNP_order"].iloc[0]).split("|"), dtype=str)

    # --------------------------------------------------------
    # The values the panel plots and tests (Tukey HSD, the letters, the box plot, the
    # forest plot): PC-adjusted when the GWAS used PCs, as the haplotype test is, else
    # the uploaded values
    # --------------------------------------------------------
    from gwas.snpplots import hap_plot_values
    _vals = hap_plot_values(_blk, pcs=st.session_state.get("pcs_for_hap", None),
                            pc_samples=geno_df.index)
    _n_pcs = int(_vals.attrs.get("n_pcs", 0))
    sample_vis = pd.Index(_vals["Sample"])
    pheno_vis = _vals["Phenotype_plotted"].to_numpy(dtype=float)
    mlg_labels_vis = _vals["Haplotype"].to_numpy()
    trait_label = f"{trait_col} (PC-adjusted)" if _n_pcs else trait_col

    # --------------------------------------------------------
    # Build MLG dataframe: the tested groups, with the test's labels
    # --------------------------------------------------------
    unique_mlg_sorted = sorted(set(mlg_labels_vis.tolist()), key=_hap_sort_key)
    if len(unique_mlg_sorted) < 2:
        st.warning("Too few haplotypes remain after filtering.")
        st.stop()

    # the samples the test left out, by reason (from the assignments table)
    _why = _blk.loc[~_blk["in_haplotype_test"].astype(bool), "exclusion_reason"].value_counts()
    if len(_why):
        st.caption(
            f"{len(unique_mlg_sorted)} haplotype groups tested; excluded: "
            + "; ".join(f"{int(_n)} sample(s), {_r}" for _r, _n in _why.items()) + ".")

    mlg_df = pd.DataFrame({
        "Sample": sample_vis,
        "MLG": pd.Categorical(mlg_labels_vis, categories=unique_mlg_sorted),
        trait_col: pheno_vis,
    })
    mlg_df["MLG_label"] = mlg_df["MLG"].astype(str)

    # ---------------------------------------------------------
    # Save haplotype assignments
    # ---------------------------------------------------------
    st.session_state.setdefault("hap_tables", {})

    block_key = f"{block_chr}:{block_start}-{block_end}"
    store_df = mlg_df.merge(
        hap_strings.rename("Allele_sequence"),
        left_on="Sample",
        right_index=True,
        how="left"
    )

    store_df = store_df.rename(columns={"MLG_label": "Haplotype"}).copy()

    store_df["Trait_adj"] = pd.to_numeric(
        store_df[trait_col], errors="coerce"
    ).astype(float)

    # the shown blocks' groups, for the most recent blocks only
    guipost.keep_recent(st.session_state.setdefault("hap_tables_vis", {}), block_key,
                        store_df[["Sample", "Haplotype", "Allele_sequence", "Trait_adj"]].copy(),
                        guipost.RESULT_CACHE_MAX)

    hap_counts = mlg_df["MLG_label"].value_counts()

    # attach true haplotype strings
    mlg_df = mlg_df.merge(
        hap_strings.rename("Allele_sequence"),
        left_on="Sample",
        right_index=True,
        how="left"
    )

    mlg_df["Allele_sequence"] = mlg_df["Allele_sequence"].str.replace("|", "_", regex=False)

    # --------------------------------------------------------
    # Summary / MLG sequence table
    # --------------------------------------------------------
    mlg_summary = (
        mlg_df.groupby("MLG_label")
        .agg(
            Allele_sequence=(
                "Allele_sequence",
                lambda x: x.dropna().iloc[0] if x.dropna().size > 0 else ""
            ),
            n_samples=("Sample", "count"),
            phenotype_mean=(trait_col, "mean"),
        )
    )

    mlg_summary["SNPs_in_block"] = ";".join(block_sids.astype(str))

    # the MLG sequence table sits in the "Haplotype details" expander under the plot
    mlg_summary_out = mlg_summary.reset_index()
    _details_table = (
        mlg_summary, mlg_summary_out.to_csv(index=False).encode(),
        f"MLG_sequences_block_{block_chr}_{block_start}_{block_end}.csv",
        f"dl_mlg_summary_{selected_block}",
    )

    # ── Make the two views' differing sample sets VISIBLE, not implied ──
    # Show the haplotype view's denominator + how many
    # accessions it excluded; the lead-SNP view uses a different sample set.
    _n_tested = block_row.get("n_samples_tested")
    _n_block = block_row.get("n_samples_block")
    _n_other = block_row.get("mlg_n_other")
    if _n_tested is not None and pd.notna(_n_tested):
        _bits = [f"**Haplotype (MLG) view** tested **{int(_n_tested)}** samples"]
        if _n_block is not None and pd.notna(_n_block) and int(_n_block) > 0:
            _bits.append(f"of {int(_n_block)} in the block ({int(_n_tested) / int(_n_block):.0%} retained)")
        if _n_other is not None and pd.notna(_n_other) and int(_n_other) > 0:
            _bits.append(f"with {int(_n_other)} sample(s) in rare genotypes ('Other') excluded")
        st.caption(" ".join(_bits) + ".")
    # (the per-sample assignments, with every sample of the panel and the reason a
    # sample was left out, are the downloads below the block selector)

    # --------------------------------------------------------
    # Tukey HSD + CLD
    # --------------------------------------------------------
    _render_tukey_and_boxplot(
        mlg_df=mlg_df,
        trait_col=trait_col,
        trait_label=trait_label,
        mlg_order=list(unique_mlg_sorted),
        hap_counts=hap_counts,
        selected_block=selected_block,
        block_chr=block_chr,
        block_start=block_start,
        block_end=block_end,
        plot_values=_vals,
        title_lines=_hap_title(block_chr, block_start, block_end,
                               block_row.get("P_perm", block_row.get("PValue")),
                               block_row.get("eta2")),
        y_label=_hap_ylab(ctx.ld_trait, _n_pcs),
        details_table=_details_table,
        eta2_test=block_row.get("eta2"),
    )

    _render_haplotype_pca(
        block_assignments=_blk,   # the block's assignments: the CSV's labels
        geno_sample_ids=geno_sample_ids,
        block_chr=block_chr,
        block_start=block_start,
        block_end=block_end,
        geno_view=getattr(ctx, "geno_view", None),
        table=guipost.table_key(ctx),
        pcs_full=getattr(ctx, "pcs_full", None),
    )

    # Store for expression panel selection
    st.session_state["pheno_vis"] = pheno_vis
    st.session_state["sample_vis"] = sample_vis



def _block_assignments_button(block_csv):
    """The selected block's sample-to-haplotype assignments (every sample of the panel, and
    why one was left out), under the block selector."""
    if block_csv is not None:
        st.download_button(
            "Download sample-to-haplotype assignments (CSV)", block_csv[0],
            file_name=block_csv[1], mime="text/csv", key=block_csv[2],
        )


def _hap_title(chrom, start, end, p_perm, eta2):
    from gwas.snpplots import hap_boxplot_title
    return hap_boxplot_title(chrom, start, end, p_perm, eta2)


def _hap_ylab(trait, n_pcs=0):
    """The haplotype plot's axis: the trait, "(PC-adjusted)" when it shows PC-adjusted
    values (the GWAS used PCs)."""
    from gwas.snpplots import hap_axis_label
    return hap_axis_label(trait, n_pcs)


def _render_tukey_tables(tukey_df, cld_df):
    """The details expander's Tukey HSD table and compact letter display (CLD), from
    gwas.cld.tukey_cld (the letters on the plot come from the same call)."""
    st.markdown("#### Tukey HSD post-hoc test")
    st.dataframe(tukey_df, use_container_width=True)

    # ------------------------------------------------------------
    # Compact Letter Display: Piepho (2004)
    # ------------------------------------------------------------
    st.markdown(
        "#### Compact Letter Display (CLD)",
        help=("Two groups that share a letter are not significantly different "
              "(Tukey HSD). A group with more than one letter is statistically "
              "indistinguishable from groups that themselves differ (a chain of "
              "overlapping non-differences), so no single letter can separate every "
              "group. See **Help → Interpreting Results** for how to read this."),
    )
    # Maximal-clique CLD (gwas/cld.py): two groups share a letter iff they are NOT
    # significantly different
    st.dataframe(cld_df, use_container_width=True)


def _render_tukey_and_boxplot(
    mlg_df, trait_col, trait_label, mlg_order, hap_counts,
    selected_block, block_chr, block_start, block_end,
    plot_values=None, title_lines=None, y_label=None, details_table=None, eta2_test=None,
):
    """Tukey HSD, the block's haplotype box plot, one details expander, the forest plot.

    ``plot_values`` is ``gwas.snpplots.hap_plot_values`` of the block: the values Tukey
    HSD, the compact letters, the box plot and the forest plot (``mlg_df`` holds the same
    rows) all use, PC-adjusted when the GWAS used PCs. The plot is drawn as the lead-SNP
    plot (``plotly_hap_boxplot`` / ``render_hap_boxplot``) with each group's compact
    letters above its box, and "Download plotted data (CSV)" under it gives
    ``hap_plot_data``. The collapsed "Haplotype details" expander holds ``details_table``
    (the MLG sequence table: frame, CSV bytes, file name, key), Tukey HSD and the CLD.
    ``eta2_test`` is the haplotype test's eta2 for the block (the block table's "η²
    (haplotype test)"): the η² the forest plot shows, as the box plot's title does. The
    forest plot draws the groups' means of the plotted values; it computes no η² of its own.
    """
    # one colour per haplotype label, the same as the PCA panel and the lead-SNP box plot
    _hap_colors = haplotype_color_map(mlg_order)

    from gwas.cld import tukey_cld
    from gwas.snpplots import hap_plot_data, plotly_hap_boxplot, render_hap_boxplot

    # Tukey HSD first: its letters go on the plot. A failure's warning stays on the page;
    # the plot (then without letters), the allele sequences and the forest plot do not
    # need the test
    tukey_df = cld_df = None
    if plot_values is not None and len(plot_values):
        try:
            tukey_df, cld_df = tukey_cld(plot_values["Phenotype_plotted"],
                                         plot_values["Haplotype"])
        except Exception as e:
            logging.exception("Tukey HSD test failed")
            st.warning(
                f"Tukey HSD failed (likely constant phenotype after PC adjustment): {e}")
    _letters = dict(zip(cld_df["MLG_label"], cld_df["CLD"])) if cld_df is not None else {}
    _pdata = hap_plot_data(plot_values, _letters) if plot_values is not None else None
    _title = list(title_lines or [f"Chr{block_chr}:{int(block_start):,}-{int(block_end):,}"])
    box_cache_base = (
        "HAP_BOX",
        block_chr, int(block_start), int(block_end),
        (_hash_df(_pdata[["Haplotype", "Phenotype_plotted", "CLD_letters"]].copy())
         if _pdata is not None else "nodata"),
        tuple(_title),
        str(y_label),
    )

    viz_widget_key = (
        f"hap_viz_type_{block_chr}_{int(block_start)}_{int(block_end)}_{trait_col}"
    )

    @st.fragment
    def _viz_toggle_fragment():
        if _pdata is None or _pdata.empty:
            st.caption("No haplotype plot: the block has no samples in the test.")
            return
        with st.expander("Plot style", expanded=False):
            viz_type = st.radio(
                "Visualization",
                ("Interactive", "Static"),
                horizontal=True,
                key=viz_widget_key,
                help=(
                    "Interactive (Plotly) shows each sample's ID, haplotype and value on "
                    "hover. Static (matplotlib) is the same plot, downloadable as PNG / "
                    "SVG / PDF."
                ),
            )
        if viz_type == "Interactive":
            _hfig = plotly_hap_boxplot(_pdata, _title, y_label=y_label)
            show_plot(
                _hfig, "small",
                key=f"hap_plotly_{block_chr}_{int(block_start)}_{int(block_end)}_{trait_col}",
            )
        else:
            _render_fig_or_cached(
                cache_namespace="_hap_box_cache",
                cache_key=box_cache_base + ("static",),
                fname_stem=(
                    f"haplotype_boxplot_{trait_col}_{block_chr}_"
                    f"{block_start}_{block_end}"
                ),
                label_prefix="Download box plot",
                build_fig_fn=lambda: render_hap_boxplot(_pdata, _title, y_label=y_label),
            )
        # the plot's data: each sample's haplotype, allele sequence, uploaded and plotted
        # values, their scale and its letters, in the order Tukey HSD took them
        st.download_button(
            "Download plotted data (CSV)", _pdata.to_csv(index=False).encode("utf-8"),
            file_name=f"haplotype_plot_{block_chr}_{block_start}_{block_end}_{trait_col}.csv",
            mime="text/csv",
            key=f"dl_hap_plot_{block_chr}_{int(block_start)}_{int(block_end)}_{trait_col}",
        )

    _viz_toggle_fragment()

    # --------------------------------------------------------
    # The details: allele sequences, Tukey HSD, compact letters
    # --------------------------------------------------------
    with st.expander("Haplotype details: allele sequences, Tukey HSD, compact letters",
                     expanded=False):
        if details_table is not None:
            _mlg_tbl, _mlg_csv, _mlg_name, _mlg_key = details_table
            st.markdown("#### Haplotype (MLG) allele sequences")
            st.dataframe(_mlg_tbl, use_container_width=True)
            st.download_button(
                "Download MLG sequence table (CSV)", _mlg_csv, file_name=_mlg_name,
                mime="text/csv", key=_mlg_key,
            )
        if tukey_df is not None:
            _render_tukey_tables(tukey_df, cld_df)
        else:
            st.caption("No Tukey HSD or compact letters: the test failed (see the warning "
                       "above).")

    # --------------------------------------------------------
    # Forest plot
    # --------------------------------------------------------
    with st.expander("Haplotype effect sizes (forest plot)", expanded=False):

        import scipy.stats as _scipy_st_forest
        _forest_data = []
        for _h in mlg_order:
            _vals = (
                mlg_df.loc[mlg_df["MLG_label"] == _h, trait_col]
                .astype(float).dropna().values
            )
            if len(_vals) >= 2:
                _mean_h = float(np.mean(_vals))
                _se_h = float(_scipy_st_forest.sem(_vals))
                _forest_data.append({
                    "Haplotype": _h,
                    "n": len(_vals),
                    "Mean": _mean_h,
                    "SE": _se_h,
                    "CI95_lo": _mean_h - 1.96 * _se_h,
                    "CI95_hi": _mean_h + 1.96 * _se_h,
                })

        if len(_forest_data) >= 2:
            _forest_df = pd.DataFrame(_forest_data)

            _y_all_f = mlg_df[trait_col].astype(float).dropna().values
            _gm_f = float(np.mean(_y_all_f))
            # the haplotype test's eta2 for this block (its samples, groups and model), not
            # one computed here from the plotted values: one η² for the block on the tab
            try:
                _eta2_vis = float(eta2_test)
            except (TypeError, ValueError):
                _eta2_vis = float("nan")

            if np.isfinite(_eta2_vis):
                st.metric("Effect size (η²)", f"{_eta2_vis:.3f}",
                          help="The haplotype test's η² for this block, as in the block "
                               "table: the share of the trait's variance its haplotype "
                               "groups explain in the test's model.")

            _x_range = (
                _forest_df["CI95_hi"].max() - _forest_df["CI95_lo"].min()
            ) or 1.0
            _fig_forest, _ax_forest = plt.subplots(
                figsize=(FIGSIZE["forest"][0], max(3, len(_forest_data) * 0.75))
            )
            for _i, _r in _forest_df.iterrows():
                _col_f = _hap_colors[_r["Haplotype"]]
                _y_pos = len(_forest_data) - 1 - _i
                _ax_forest.plot(
                    [_r["CI95_lo"], _r["CI95_hi"]], [_y_pos, _y_pos],
                    color=_col_f, linewidth=2.5, solid_capstyle="round"
                )
                _ax_forest.plot(
                    _r["Mean"], _y_pos, "o",
                    color=_col_f, markersize=9, zorder=5
                )
                _ax_forest.text(
                    _r["CI95_hi"] + 0.03 * _x_range,
                    _y_pos,
                    f"n={_r['n']}",
                    va="center", fontsize=9, color="grey"
                )

            _ax_forest.axvline(
                _gm_f, color="grey", linestyle="--",
                linewidth=1, alpha=0.7, label="Grand mean"
            )
            _ax_forest.set_yticks(
                range(len(_forest_data) - 1, -1, -1)
            )
            _ax_forest.set_yticklabels(
                _forest_df["Haplotype"].tolist(),
            )
            _ax_forest.set_xlabel(trait_label)
            _title_f = (
                f"Haplotype Effect Sizes: \u03b7\u00b2 = {_eta2_vis:.3f}"
                if np.isfinite(_eta2_vis)
                else "Haplotype Effect Sizes"
            )
            _ax_forest.set_title(_title_f)
            _ax_forest.legend()
            for _sp in _ax_forest.spines.values():
                _sp.set_visible(False)
            _ax_forest.tick_params(axis="both", length=0)
            plt.tight_layout()
            show_plot(_fig_forest, "small")
            export_matplotlib(_fig_forest,
                              f"haplotype_forest_{block_chr}_{block_start}_{block_end}",
                              label_prefix="Download forest plot")
            plt.close(_fig_forest)

            st.download_button(
                "Download haplotype effects (CSV)",
                _forest_df.to_csv(index=False).encode("utf-8"),
                file_name=(
                    f"Haplotype_effects_{block_chr}"
                    f"_{block_start}_{block_end}.csv"
                ),
                mime="text/csv",
                key=f"dl_forest_{selected_block}",
            )
