"""Tab — Regional association plot (locus-zoom-style), T-08.

A display of the existing genome-wide scan around a lead SNP: −log₁₀(p) vs
position, points coloured by r² to the lead (computed from the user's own
genotypes), the active significance line, the detected LD block (or, for an
isolated SNP, its flanking-marker interval), and a gene track. This is NOT
fine-mapping — no credible sets, no candidate ranking, no p-value changes (R1.7
asks about fine-mapping separately; this does not answer it).

Thin renderer over the pure functions in ``gwas.plotting``; the window comes from
the shared selector (``pages/_ld_tabs/_window.py``) so this and Local LD render one
selection.
"""
import re
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from annotation import canon_chr
from gwas.plotting import (
    compute_r2_to_lead,
    plot_regional_association_static,
    plot_regional_association_interactive,
    _clamp_block_span,
    MAX_REGIONAL_SNPS,
)
from gwas.significance import rule_from_streamlit
from utils.pub_theme import export_matplotlib
from . import LDContext


def _split_ids(raw):
    return {s.strip() for s in re.split(r"[;,\s|]+", str(raw)) if s.strip()}


def _members_by_bounds(hd, core_start, core_end):
    """Member SNP_IDs of the block whose interval matches (core_start, core_end)."""
    if isinstance(hd, pd.DataFrame) and not hd.empty and "SNP_IDs" in hd.columns:
        rows = hd[
            (pd.to_numeric(hd["Start (bp)"], errors="coerce") == int(core_start))
            & (pd.to_numeric(hd["End (bp)"], errors="coerce") == int(core_end))
        ]
        if not rows.empty:
            return _split_ids(rows.iloc[0]["SNP_IDs"])
    return None


def _block_for_window(ctx, window):
    """(interval, members) for the shaded LD-block span, or (None, None).

    Detected-block mode carries explicit bounds (``window.core_start`` set) → use
    them. In lead-SNP mode, shade the detected block CONTAINING the lead (membership
    first, else positional containment) so a span appears with no checkbox. No
    containing block → (None, None) and the caller shades the flanking interval.
    """
    hd = ctx.haplo_df_auto
    if window.core_start is not None:
        return ((int(window.core_start), int(window.core_end)),
                _members_by_bounds(hd, window.core_start, window.core_end))
    if isinstance(hd, pd.DataFrame) and not hd.empty and {"Chr", "Start (bp)", "End (bp)"}.issubset(hd.columns):
        chr_c = canon_chr(window.chr)
        has_ids = "SNP_IDs" in hd.columns
        for _, r in hd.iterrows():
            if canon_chr(str(r["Chr"])) != chr_c:
                continue
            ids = _split_ids(r["SNP_IDs"]) if has_ids else set()
            s, e = int(r["Start (bp)"]), int(r["End (bp)"])
            if str(window.lead_snp) in ids or (s <= window.lead_pos <= e):
                return (s, e), (ids or None)
    return None, None


def _flank_interval(positions_in_window, lead_pos):
    """Nearest typed markers flanking the lead — the isolated-SNP fallback span."""
    pos = np.sort(np.asarray(positions_in_window, dtype=float))
    if pos.size == 0:
        return None
    left = pos[pos < lead_pos]
    right = pos[pos > lead_pos]
    lo = int(left.max()) if left.size else int(pos.min())
    hi = int(right.min()) if right.size else int(pos.max())
    return (lo, hi) if lo != hi else None


def _filter_genes(genes_df, chr_sel, start_bp, end_bp):
    if genes_df is None or getattr(genes_df, "empty", True):
        return None
    if not {"Chr", "Start", "End", "Gene_ID"}.issubset(genes_df.columns):
        return None
    chr_c = canon_chr(chr_sel)
    g = genes_df.copy()
    g_chr = g["Chr"].astype(str).map(canon_chr)
    sub = g[
        (g_chr == chr_c)
        & (pd.to_numeric(g["End"], errors="coerce") >= start_bp)
        & (pd.to_numeric(g["Start"], errors="coerce") <= end_bp)
    ]
    if sub.empty:
        return None
    if "Strand" not in sub.columns:
        sub = sub.assign(Strand=".")
    return sub


def _absent_gene_track_caption(genes_df):
    """Caption for the zero-gene case on the Regional Plot: distinguishes a missing
    gene model from a loaded model that has no annotated genes in this window (two
    different facts for interpreting a locus). None or empty -> no model is loaded."""
    if genes_df is None or getattr(genes_df, "empty", True):
        return ("No gene model loaded, so no gene track is shown. Load one "
                "in the Gene Annotation tab to add gene context here.")
    return "No annotated genes fall in this window."


def render(ctx: LDContext, window):
    st.subheader("Regional association plot")

    gwas_df = ctx.gwas_df
    if gwas_df is None or getattr(gwas_df, "empty", True):
        st.info("Run a GWAS first. This tab reads the GWAS results in session.")
        return
    if window is None:
        return  # the shared selector already explained why
    if ctx.geno_dosage_raw is None:
        st.info("Raw genotype dosages are not in session; re-run the GWAS to enable r² colouring.")
        return

    chroms = np.asarray(ctx.chroms).astype(str)
    positions = np.asarray(ctx.positions, dtype=float)
    chr_c = canon_chr(window.chr)
    chrom_canon = np.array([canon_chr(c) for c in chroms])
    mask = (chrom_canon == chr_c) & (positions >= window.start_bp) & (positions <= window.end_bp)
    n_win = int(mask.sum())
    if n_win == 0:
        st.info("No genotyped SNPs fall in this window.")
        return

    # Dense-window cap (matches the pairwise_r2 ceiling used elsewhere).
    capped = False
    if n_win > MAX_REGIONAL_SNPS:
        idx = np.where(mask)[0]
        near = idx[np.argsort(np.abs(positions[idx] - window.lead_pos))[:MAX_REGIONAL_SNPS]]
        mask = np.zeros_like(mask)
        mask[near] = True
        capped = True

    try:
        r2, _ = compute_r2_to_lead(ctx.geno_dosage_raw, ctx.sid, window.lead_snp, mask, min_pair_n=15)
    except ValueError:
        r2 = np.full(int(mask.sum()), np.nan)  # lead SNP not typed in the dosage matrix

    sid_arr = np.asarray(ctx.sid).astype(str)
    win_idx = np.where(mask)[0]
    wdf = pd.DataFrame({
        "SNP": sid_arr[win_idx],
        "Chr": chroms[win_idx],
        "Pos": positions[win_idx].astype(np.int64),
        "_r2": r2,
    })
    pv = gwas_df.drop_duplicates("SNP").set_index("SNP")["PValue"]
    wdf["PValue"] = wdf["SNP"].map(pv)
    wdf = wdf[wdf["PValue"].notna()].reset_index(drop=True)
    if wdf.empty:
        st.info("No scanned SNPs (with a p-value) fall in this window.")
        return
    r2_final = wdf.pop("_r2").to_numpy(dtype=float)

    rule = rule_from_streamlit(ctx.sig_rule_label, len(gwas_df), ctx.meff_val,
                               custom_thresh=ctx.custom_thresh)
    sig_threshold = rule.p_threshold  # None for FDR -> no horizontal line

    block_interval, block_members = _block_for_window(ctx, window)
    is_flank = False
    if block_interval is None:
        block_interval = _flank_interval(wdf["Pos"].to_numpy(), window.lead_pos)
        is_flank = block_interval is not None

    seed_threshold = float(st.session_state.get("ld_seed_p", 1e-5))
    genes = _filter_genes(st.session_state.get("genes_df"), window.chr,
                          window.start_bp, window.end_bp) if ctx.has_annotation else None

    # --- disclosures (the edge cases the design names) ---
    n_typed_r2 = int(np.isfinite(r2_final).sum())
    if capped:
        st.warning(f"Dense window: showing the {MAX_REGIONAL_SNPS} SNPs nearest the lead "
                   f"(of {n_win}). The rest are omitted from the plot only, not the analysis.")
    if window.lead_snp not in set(wdf["SNP"]):
        st.info(f"Lead SNP **{window.lead_snp}** is not inside the plotted window "
                "(a block's seed SNP can lie outside the block it labels).")
    # block-span shading mode: matches the plotter's clamp on the same x-range
    # (window_df["Pos"] min/max in Mb), so the caption states what actually renders.
    _mode = None
    if block_interval is not None:
        _xlo = float(wdf["Pos"].min()) / 1e6
        _xhi = float(wdf["Pos"].max()) / 1e6
        _bs, _be, _mode = _clamp_block_span(block_interval[0] / 1e6, block_interval[1] / 1e6,
                                            _xlo, _xhi)
    if is_flank:
        st.caption("No LD block here; the shaded span is the **flanking-marker interval** "
                   "around the SNP, not an LD block.")
    elif block_interval is not None:
        _b0, _b1 = block_interval
        if _mode == "edges":
            st.caption("The LD block spans essentially the whole window, so its edges are "
                       "marked instead of a shaded fill; widen the buffer to restore the shaded span.")
        elif _mode == "none":
            st.caption("The LD block clamps to an empty span within the view; nothing is shaded.")
        elif _b0 < int(wdf["Pos"].min()) or _b1 > int(wdf["Pos"].max()):
            st.caption(f"LD block Chr{window.chr}:{_b0:,}-{_b1:,} "
                       f"({(_b1 - _b0) / 1000:,.0f} kb) extends beyond the plotted window; "
                       "the shaded span is clipped to the view.")
    else:
        st.caption("No LD block or flanking interval to shade here "
                   "(the lead is the only typed marker in the window).")
    if n_typed_r2 == 0:
        st.caption("r² to the lead is not computable for these markers (e.g. a single typed "
                   "marker); points are shown in grey.")

    # --- interactive (plotly) — the GUI view, hover for SNP / p / r² / block ---
    _winsig = f"{window.chr}:{window.start_bp}-{window.end_bp}:{window.lead_snp}"
    fig_i = plot_regional_association_interactive(
        wdf, r2_final, window.lead_snp, sig_threshold,
        block_interval=block_interval, block_members=block_members,
        seed_threshold=seed_threshold, uirevision=_winsig)
    # uirevision (a per-window signature) makes Plotly reset the axes on a window
    # change and keep zoom within one. The key MUST be constant: a key that varies
    # with the window is a new widget that re-inits its value= default and forces
    # an extra rerun (that was the round-1 stale-render bug).
    st.plotly_chart(fig_i, use_container_width=True, key="regional_chart")

    st.caption(
        f"r² is computed from your own genotypes (not a reference panel). Threshold: "
        f"{rule.label}. A display of the existing scan; no p-value is changed."
    )

    # --- numeric export: the numbers behind the plot (put them in a table, not a figure) ---
    _members_set = set(block_members) if block_members else set()
    _export_df = wdf[["SNP", "Chr", "Pos", "PValue"]].copy()
    _export_df["r2_to_lead"] = r2_final
    _export_df["block_member"] = _export_df["SNP"].isin(_members_set)
    st.download_button(
        "Download regional data (CSV)",
        _export_df.to_csv(index=False).encode(),
        file_name=f"Regional_data_Chr{window.chr}_{window.start_bp}_{window.end_bp}_{window.lead_snp}.csv",
        mime="text/csv", key="dl_regional_csv",
    )

    # --- static (matplotlib) — the downloadable / report figure, with gene track ---
    MAX_GENE_LABELS = 12
    n_genes = 0 if genes is None else len(genes)
    with st.expander("Static figure + downloads (PNG / SVG / PDF)", expanded=False):
        if n_genes:
            gene_labels = st.checkbox(
                "Show gene labels", value=(n_genes <= MAX_GENE_LABELS),
                key="regional_gene_labels",
                help="Gene names on the track. Off by default in dense windows to avoid overprinting.")
            if not gene_labels:
                st.caption(f"{n_genes} genes in window; labels hidden. See the Gene Annotation tab "
                           "for the full list. Bars show position, extent and strand.")
        else:
            gene_labels = True
            st.caption(_absent_gene_track_caption(st.session_state.get("genes_df")))
        fig_s = plot_regional_association_static(
            wdf, r2_final, window.lead_snp, sig_threshold,
            block_interval=block_interval, block_members=block_members,
            genes=genes, seed_threshold=seed_threshold, gene_labels=gene_labels)
        buf = BytesIO()
        fig_s.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        st.image(buf.getvalue())
        stem = f"Regional_{window.lead_snp}_Chr{window.chr}_{window.start_bp}_{window.end_bp}"
        export_matplotlib(fig_s, stem, label_prefix="Download regional plot")
        plt.close(fig_s)
