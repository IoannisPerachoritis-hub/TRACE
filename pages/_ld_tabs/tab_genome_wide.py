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
from utils.plot_data import pca_points_table

from gwas.ld import (
    find_ld_clusters_genomewide,
    filter_contained_blocks,
)
from . import LDContext


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
    same cache_key short-circuit straight to st.image with cached bytes.
    """
    cache = st.session_state.setdefault(cache_namespace, {})
    cached = cache.get(cache_key)

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
        cache[cache_key] = cached

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
    mlg_df, geno_sample_ids,
    block_chr, block_start, block_end,
):
    """PC1 vs PC2 inset, one point per accession, coloured by current-block MLG_label."""
    pcs_full = st.session_state.get("pcs_full")
    eigenvalues = st.session_state.get("pca_eigenvalues")

    if pcs_full is None or pcs_full.shape[1] < 2:
        from gwas.kinship import _standardize_geno_for_grm
        geno = st.session_state.get("geno_imputed")
        if geno is None:
            st.info("PCA panel unavailable: no cached PCs and no imputed genotype.")
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

    mlg_by_iid = mlg_df.set_index(mlg_df["Sample"].astype(str))["MLG_label"].astype(str)
    labels = pd.Series(
        mlg_by_iid.reindex(geno_iids).fillna("Other").values,
        name="MLG_label",
    )

    if len(mlg_df) > 0 and (labels != "Other").sum() == 0:
        st.warning(
            "PCA panel: 0 accessions matched between haplotype assignments "
            f"and the loaded genotype table ({len(mlg_df)} haplotyped samples vs "
            f"{len(geno_iids)} genotype rows). Sample-ID format mismatch; "
            "cannot colour by haplotype. Skipping PCA panel."
        )
        return

    # Per-point arrays + colour map shared by both visualisation variants
    present = list(pd.unique(labels))
    hap_order = sorted([h for h in present if h != "Other"], key=_hap_sort_key)
    # one colour per label in every plot of the tab and in the box plots (Other grey)
    color_map = haplotype_color_map(present)
    plot_order = (["Other"] if "Other" in present else []) + hap_order

    def _build_static_pca_fig():
        fig, ax = plt.subplots(figsize=(5, 5))
        for h in plot_order:
            m = (labels == h).values
            if not m.any():
                continue
            size = 18 if h == "H1" else (14 if h == "Other" else 42)
            ax.scatter(
                pcs_2[m, 0], pcs_2[m, 1],
                s=size,
                c=color_map[h],
                edgecolors="white" if h != "Other" else "none",
                linewidths=0.4,
                label=f"{h} (n={int(m.sum())})",
                alpha=0.95 if h != "Other" else 0.55,
            )

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
        for h in plot_order:
            m = (labels == h).values
            n_h = int(m.sum())
            if n_h == 0:
                continue
            size = 8 if h == "H1" else (6 if h == "Other" else 14)
            accessions = geno_iids[m].astype(str).to_numpy()
            x_vals = pcs_2[m, 0]
            y_vals = pcs_2[m, 1]
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                name=f"{h} (n={n_h})",
                marker=dict(
                    color=color_map[h],
                    size=size,
                    line=dict(
                        color="white" if h != "Other" else "rgba(0,0,0,0.2)",
                        width=0.7,
                    ),
                    opacity=0.95 if h != "Other" else 0.55,
                ),
                customdata=np.column_stack([accessions, np.full(n_h, h)]),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Haplotype: %{customdata[1]}<br>"
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

    # Static cache key (matplotlib only; Plotly rebuilds each fragment rerun)
    static_cache_key = (
        "_hap_pca_render",
        block_chr, int(block_start), int(block_end),
        hash(tuple(labels.values)),
        pcs_2.shape[0],
    )
    cache = st.session_state.setdefault("_hap_pca_cache", {})

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
                data=pca_points_table(geno_iids, pcs_2, labels.to_numpy()),
                data_name=f"haplotype_pca_{block_chr}_{int(block_start)}_{int(block_end)}",
            )
        else:
            cached = cache.get(static_cache_key)
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
                cache[static_cache_key] = cached

            show_plot(cached["display"], "medium",
                      data=pca_points_table(geno_iids, pcs_2, labels.to_numpy()),
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


def settings_summary(seed_p, top_n, flank_kb, ld_r2, n_perm, min_hap_count, min_group_size,
                     gallery_rule, gallery_custom=None):
    """The Settings panel's one line: the values in use, e.g. "Seeds P < 1e-05 plus top 10
    · window 200 kb · r² ≥ 0.6 · 1,000 permutations · haplotypes ≥ 5 samples · groups ≥ 3
    · gallery: Bonferroni"."""
    seeds = f"Seeds P < {float(seed_p):.0e}"
    if int(top_n) > 0:
        seeds += f" plus top {int(top_n)}"
    return " · ".join([
        seeds,
        f"window {float(flank_kb):g} kb",
        f"r² ≥ {float(ld_r2):g}",
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
        flank_kb_default = float(np.clip(ctx.ld_decay_kb * 2, 200, 2000))
        flank_kb_auto = st.number_input(
            "LD window around each lead SNP (kb)",
            min_value=10.0,
            max_value=2000.0,
            value=float(flank_kb_default),
            step=50.0,
            help=(
                "Physical distance (in kb) to extend around each seed SNP. After the "
                "disjoint-block redesign it bounds how far one block can extend from "
                "its seed; clusters that do not contain the seed are discarded. The "
                "default is 2× the measured LD decay distance."
            ),
        )
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
                                          gallery_custom))
    return {"seed_p": sig_thresh, "top_n": top_n, "flank_kb": flank_kb_auto,
            "ld_r2": ld_threshold_auto,
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
        # future change to them re-keys. Detection recomputes only on a key change.
        _gwas_sig = st.session_state.get("haplo_df_auto_hash", {}).get(trait_col)
        if _gwas_sig is None:
            _gwas_sig = _hash_df(gwas_df) if isinstance(gwas_df, pd.DataFrame) else "nogwas"
        ld_key = (
            f"LDDETECT::{trait_col}"
            f"::r2{ld_threshold_auto}::flank{flank_kb_auto}::topn{int(top_n)}"
            f"::sig{sig_thresh}::decay{ld_decay_kb}::ms2::adj0.2::gap10"
            f"::G{_gwas_sig}"
        )

        if ld_key not in st.session_state:
            with st.spinner("Finding LD blocks around significant SNPs…"):
                haplo_df_auto = find_ld_clusters_genomewide(
                    gwas_df=gwas_df,
                    chroms=chroms,
                    positions=positions,
                    geno_imputed=np.asarray(geno_ld, float),
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
                # disjoint-block redesign (WO4): _occupancy_select claims each accepted
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

            if "haplo_df_auto" not in st.session_state:
                st.session_state["haplo_df_auto"] = {}

            st.session_state["haplo_df_auto"][trait_col] = haplo_df_auto
            st.session_state["ld_blocks_final"] = haplo_df_auto.copy()
            st.session_state["ld_blocks"] = haplo_df_auto.copy()

            # --- LD metadata logging ---
            st.session_state["ld_block_metadata_auto"] = {
                "method": "peak-centric LD blocks",
                "ld_threshold_r2": float(ld_threshold_auto),
                "flank_kb": float(flank_kb_auto),
                "min_snps_per_block": 2,
                "top_n_snps": int(top_n),
                "sig_threshold": float(sig_thresh),
                "ld_decay_kb": float(ld_decay_kb),
                "adjacent_r2_split": 0.2,
            }
            st.session_state[ld_key] = haplo_df_auto
        else:
            haplo_df_auto = st.session_state[ld_key]

        # Show clusters if available
        if not isinstance(haplo_df_auto, pd.DataFrame) or haplo_df_auto.empty:
            st.info(
                "No LD blocks detected: no markers pass the significance threshold. "
                "Lower it, or check that the GWAS produced significant hits."
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


# The "?" next to "Association results per block": how to read the effect sizes and the
# LD-block coherence columns of the table under it.
ASSOCIATION_RESULTS_HELP = (
    "**Effect sizes (η²)**\n\n"
    "- **Effect (β)**: trait change per minor-allele copy (sign = direction).\n"
    "- **Variance explained (η²)** per LD block: the most interpretable measure "
    "of QTL importance for breeding:\n"
    "  - **> 10%** major QTL, strong marker-assisted-selection candidate\n"
    "  - **5-10%** moderate QTL, useful combined with other loci\n"
    "  - **< 5%** minor QTL, polygenic background\n"
    "- η² is not adjusted for kinship; in structured panels it can be inflated.\n\n"
    "**LD-block coherence (r²)**\n\n"
    "- **Mean r2 / Min r2**: mean / weakest member-pair r² in the block "
    "(how internally correlated it is).\n"
    "- **Frac r2>=.6 to lead**: fraction of members with r² ≥ 0.6 to the lead "
    "SNP. A low value on a lead's own block means the interval is a loose merge "
    "of sub-clusters, not one tight haplotype.\n\n"
    "Full guidance: **Help → Interpreting Results**."
)


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
    # Drift guard: do the genotypes on this page belong to the run on screen?
    # ------------------------------------------------------------------
    # `geno_row_ids` is written ONLY by a completed GWAS run (GWAS_analysis.py, the
    # `st.session_state["geno_row_ids"] = np.asarray(geno_df.index, dtype=str).tolist()`
    # in the export block), while `geno_df` / `geno_imputed` are refreshed on EVERY
    # rerun that reaches the `for key in [` results loop -- and `geno_row_ids` is not
    # in that list. A disagreement therefore means the GWAS page rebuilt the genotype
    # objects for a different trait without completing a run, so the blocks on screen
    # and the genotypes no longer come from the same analysis. The previous run's
    # matrix is gone (overwritten by that loop), so this cannot be repaired here.
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

    pcs_sig = "nopcs"
    if pcs is not None:
        pcs_arr = np.asarray(pcs)
        pcs_sig = f"pcs{pcs_arr.shape[0]}x{pcs_arr.shape[1]}"

    geno_sig = f"{geno_ld.shape[0]}x{geno_ld.shape[1]}"
    n_perm = int(st.session_state.get("n_perm_hap", 1000))

    hap_key = (
        f"HAPGWAS::{trait_col}"
        f"::mhc{int(min_hap_count)}::mgs{int(min_group_size)}"
        f"::{pcs_sig}"
        f"::{geno_sig}"
        f"::nperm{n_perm}"
        f"::B{blocks_hash}::P{pheno_hash}"
    )

    if hap_key not in st.session_state:

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
            )
            st.session_state["hap_tables"] = hap_tables
            st.session_state["hap_gwas_df"] = hap_gwas_df
        st.session_state[hap_key] = hap_gwas_df
        # the per-block sample tables of THIS result, kept with it (the assignments
        # table, the downloads and the MLG plots read them); an earlier result's tables
        # and assignments are dropped
        for _old in [k for k in st.session_state.keys()
                     if isinstance(k, str) and k.startswith("HAPGWAS::")
                     and k.endswith(("::tables", "::assign"))]:
            del st.session_state[_old]
        st.session_state[f"{hap_key}::tables"] = hap_tables

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
        st.caption(f"LD-block coherence unavailable: {_e}")

    _seed_p = (settings or {}).get("seed_p")

    def _show_block_table(tested_df):
        """One row per detected block: the block, its haplotype test, its coherence."""
        from .block_table import compact_view, merge_block_table
        _tbl = merge_block_table(haplo_df_auto, tested_df, _ldq_full, untested=_untested,
                                 seed_p=_seed_p)
        st.subheader("LD blocks and haplotype tests", help=ASSOCIATION_RESULTS_HELP)
        _all = st.toggle("Show all columns", value=False, key="block_table_all_columns")
        st.dataframe(_tbl if _all else compact_view(_tbl), use_container_width=True)
        st.download_button(
            "Download block table (CSV)", _tbl.to_csv(index=False).encode("utf-8"),
            file_name=f"LD_blocks_and_haplotype_tests_{ld_trait}.csv", mime="text/csv",
            key="dl_block_table",
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
    _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
    hap_gwas_df["FDR_qvalue"] = qvals

    # T-34 (§2.8 fix): enrich BEFORE display so η²/Frac_other/Block_length_kb/MAF
    # reach the on-screen table, not only the downloaded CSV/ZIP.
    with st.spinner("Computing block effect sizes (η²) and MAF…"):
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
    # disagree (WO-HAPMASK-01). The SNP axis must move with the matrix: `_ci` below is
    # a column index into it, so a stale sid/positions array would silently select the
    # wrong marker.
    geno_hard = ctx.geno_hard
    if geno_hard is None:
        return

    from gwas.significance import frame_for_rule, rule_from_streamlit
    from .tab_significant_snps import _SIG_LABELS

    st.markdown("### Lead SNP genotype effect")
    st.caption(
        "Browse the genotype/phenotype effect for any significant SNP: pick a SNP. "
        "Each is rendered with the same two structure-aware tests as the per-block "
        "lead below."
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
        _dose = np.asarray(geno_hard)[:, _ci].astype(float)

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
    values (PC-adjusted for display). Its two downloads sit below the block selector.
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

    # the block's own file sits under its box/raincloud plot (the plot's data); the
    # all-blocks file stays here, under the block selector
    _block_csv = (
        _blk.to_csv(index=False).encode("utf-8"),
        f"haplotype_assignments_block_{block_chr}_{block_start}_{block_end}_{ld_trait}.csv",
        f"dl_hap_assign_{selected_block}",
    )
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
    sample_vis = pd.Index(_tested["Sample"].astype(str))
    pheno_vis = _tested["Phenotype_used"].to_numpy(dtype=float)
    mlg_labels_vis = _tested["Haplotype"].astype(str).to_numpy()
    hap_strings = pd.Series(_tested["Allele_sequence"].astype(str).to_numpy(),
                            index=sample_vis, name="Allele_sequence")
    block_sids = np.asarray(str(_blk["SNP_order"].iloc[0]).split("|"), dtype=str)

    # --------------------------------------------------------
    # PC-adjusted phenotype for visualization
    # --------------------------------------------------------
    pcs_vis = st.session_state.get("pcs_for_hap", None)
    trait_label = trait_col

    if pcs_vis is not None:
        try:
            pcs_df = pd.DataFrame(pcs_vis, index=geno_df.index.astype(str))
            pcs_block = pcs_df.loc[sample_vis]

            valid_pc = (~pcs_block.isna().any(axis=1)).to_numpy()
            pheno_vis = pheno_vis[valid_pc]
            sample_vis = sample_vis[valid_pc]
            mlg_labels_vis = mlg_labels_vis[valid_pc]
            pcs_block = pcs_block.loc[valid_pc]

            X = np.column_stack([np.ones((pcs_block.shape[0], 1)), pcs_block.values])
            beta, *_ = np.linalg.lstsq(X, pheno_vis.reshape(-1, 1), rcond=None)
            pheno_vis = (pheno_vis.reshape(-1, 1) - X @ beta).ravel()

            trait_label = f"{trait_col} (PC-adjusted)"

        except Exception:
            trait_label = trait_col

    # --------------------------------------------------------
    # Build MLG dataframe: the tested groups, with the test's labels
    # --------------------------------------------------------
    unique_mlg_sorted = sorted(set(mlg_labels_vis.tolist()), key=_hap_sort_key)
    if len(unique_mlg_sorted) < 2:
        st.warning("Too few haplotypes remain after filtering.")
        _block_assignments_button(_block_csv)
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

    st.session_state.setdefault("hap_tables_vis", {})
    st.session_state["hap_tables_vis"][block_key] = store_df[
        ["Sample", "Haplotype", "Allele_sequence", "Trait_adj"]].copy()

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

    # ── T-34: make the two views' differing sample sets VISIBLE, not implied ──
    # (the reviewer concern). Show the haplotype view's denominator + how many
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
        block_csv=_block_csv,
        plot_rows=_tested,
        title_lines=_hap_title(block_chr, block_start, block_end,
                               block_row.get("P_perm", block_row.get("PValue")),
                               block_row.get("eta2")),
        y_label=_hap_ylab(ctx.ld_trait),
        details_table=_details_table,
    )

    _render_haplotype_pca(
        mlg_df=mlg_df,
        geno_sample_ids=geno_sample_ids,
        block_chr=block_chr,
        block_start=block_start,
        block_end=block_end,
    )

    # Store for expression panel selection
    st.session_state["pheno_vis"] = pheno_vis
    st.session_state["sample_vis"] = sample_vis



def _block_assignments_button(block_csv):
    """The selected block's sample-to-haplotype assignments: the data of its box plot."""
    if block_csv is not None:
        st.download_button(
            "Download sample-to-haplotype assignments (CSV)", block_csv[0],
            file_name=block_csv[1], mime="text/csv", key=block_csv[2],
        )


def _hap_title(chrom, start, end, p_perm, eta2):
    from gwas.snpplots import hap_boxplot_title
    return hap_boxplot_title(chrom, start, end, p_perm, eta2)


def _hap_ylab(trait):
    """The haplotype plot's axis: the trait (the uploaded values the test compared)."""
    from gwas.snpplots import phenotype_axis_label
    return phenotype_axis_label(trait)


def _render_tukey_and_boxplot(
    mlg_df, trait_col, trait_label, mlg_order, hap_counts,
    selected_block, block_chr, block_start, block_end, block_csv=None,
    plot_rows=None, title_lines=None, y_label=None, details_table=None,
):
    """The block's haplotype box plot, one details expander, then the forest plot.

    The plot is drawn as the lead-SNP plot (``gwas.snpplots.plotly_hap_boxplot`` /
    ``render_hap_boxplot``) from ``plot_rows``, the block's in-test assignment rows: the
    samples, labels and uploaded values the haplotype test compared. ``block_csv`` is
    the block's assignments CSV (bytes, file name, key), offered under the plot as its
    data. The collapsed "Haplotype details" expander holds ``details_table`` (the MLG
    sequence table: frame, CSV bytes, file name, key), Tukey HSD and the compact letters.
    """
    # one colour per haplotype label, the same as the PCA panel and the lead-SNP box plot
    _hap_colors = haplotype_color_map(mlg_order)

    from gwas.snpplots import hap_boxplot_data, plotly_hap_boxplot, render_hap_boxplot
    _pdata = hap_boxplot_data(plot_rows) if plot_rows is not None else None
    _title = list(title_lines or [f"Chr{block_chr}:{int(block_start):,}-{int(block_end):,}"])
    box_cache_base = (
        "HAP_BOX",
        block_chr, int(block_start), int(block_end),
        (_hash_df(_pdata[["Haplotype", "Phenotype"]].copy())
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
            _block_assignments_button(block_csv)
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
        _block_assignments_button(block_csv)

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
        st.markdown("#### Tukey HSD post-hoc test")

        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        try:
            tukey = pairwise_tukeyhsd(
                endog=mlg_df[trait_col].astype(float),
                groups=mlg_df["MLG_label"].astype(str),
                alpha=0.05,
            )
        except Exception as e:
            logging.exception("Tukey HSD test failed")
            st.warning(
                f"Tukey HSD failed (likely constant phenotype after PC adjustment): {e}")
            st.stop()

        tukey_data = tukey.summary().data
        headers = tukey_data[0]
        rows = tukey_data[1:]

        tukey_df = pd.DataFrame(rows, columns=headers)
        tukey_df["reject"] = tukey_df["reject"].astype(bool)

        st.dataframe(tukey_df, use_container_width=True)

        # ------------------------------------------------------------
        # Compact Letter Display: Piepho (2004)
        # ------------------------------------------------------------
        st.markdown("#### Compact Letter Display (CLD)")

        groups = sorted(mlg_df["MLG_label"].unique())
        sig_matrix = pd.DataFrame(False, index=groups, columns=groups)
        for _, row in tukey_df.iterrows():
            g1 = str(row["group1"])
            g2 = str(row["group2"])
            reject = bool(row["reject"])
            sig_matrix.loc[g1, g2] = reject
            sig_matrix.loc[g2, g1] = reject

        # Maximal-clique CLD (gwas/cld.py): two groups share a letter iff they are NOT
        # significantly different. Replaces a greedy insert/absorb that could leave private
        # seed letters unmerged (3 mutually non-sig groups -> a, ab, ac instead of a, a, a).
        from gwas.cld import compact_letter_display

        _cld_means = mlg_df.groupby("MLG_label")[trait_col].mean().to_dict()
        cld_strings = compact_letter_display(groups, sig_matrix, means=_cld_means)

        cld_df = pd.DataFrame({
            "MLG_label": groups,
            "CLD": [cld_strings[g] for g in groups],
        })
        st.dataframe(cld_df, use_container_width=True)
        if any(len(_v) > 1 for _v in cld_strings.values()):
            st.caption(
                "Some groups carry more than one letter: they are statistically "
                "indistinguishable from groups that themselves differ (a chain of "
                "overlapping non-differences), so no single letter can separate every "
                "group. See **Help → Interpreting Results** for how to read this."
            )

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
            _ss_tot_f = float(np.sum((_y_all_f - _gm_f) ** 2))
            _ss_bet_f = float(sum(
                r["n"] * (r["Mean"] - _gm_f) ** 2
                for _, r in _forest_df.iterrows()
            ))
            _eta2_vis = _ss_bet_f / _ss_tot_f if _ss_tot_f > 0 else float("nan")

            if np.isfinite(_eta2_vis):
                st.metric("Effect size (η²)", f"{_eta2_vis:.3f}",
                          help="Proportion of phenotypic variance explained by haplotype group")

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
