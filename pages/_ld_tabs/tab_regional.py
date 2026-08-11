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
    MAX_REGIONAL_SNPS,
)
from gwas.significance import rule_from_streamlit
from utils.pub_theme import export_matplotlib
from . import LDContext


def _split_ids(raw):
    return {s.strip() for s in re.split(r"[;,\s|]+", str(raw)) if s.strip()}


def _block_for_window(ctx, window):
    """(interval, members) for the shaded LD-block span, or (None, None).

    Snapped-to-block windows carry the block start/end; look the block row back up
    by its interval to recover the member SNP_IDs. A non-snapped window returns
    ``(None, None)`` and the caller shades the flanking-marker interval instead.
    """
    if not window.use_block or window.core_start is None:
        return None, None
    hd = ctx.haplo_df_auto
    members = None
    if isinstance(hd, pd.DataFrame) and not hd.empty and "SNP_IDs" in hd.columns:
        rows = hd[
            (pd.to_numeric(hd["Start (bp)"], errors="coerce") == int(window.core_start))
            & (pd.to_numeric(hd["End (bp)"], errors="coerce") == int(window.core_end))
        ]
        if not rows.empty:
            members = _split_ids(rows.iloc[0]["SNP_IDs"])
    return (int(window.core_start), int(window.core_end)), members


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


def render(ctx: LDContext, window):
    st.subheader("Regional association plot")
    st.caption(
        "Association (−log₁₀ p) against position around the lead SNP, coloured by r² "
        "to the lead. A **display of the existing scan** — not fine-mapping."
    )

    gwas_df = ctx.gwas_df
    if gwas_df is None or getattr(gwas_df, "empty", True):
        st.info("Run a GWAS first — this tab reads the GWAS results in session.")
        return
    if window is None:
        return  # the shared selector already explained why
    if ctx.geno_dosage_raw is None:
        st.info("Raw genotype dosages are not in session — re-run the GWAS to enable r² colouring.")
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
    if is_flank:
        st.caption("No LD block here — the shaded span is the **flanking-marker interval** "
                   "around the SNP, not an LD block.")
    if n_typed_r2 == 0:
        st.caption("r² to the lead is not computable for these markers (e.g. a single typed "
                   "marker); points are shown in grey.")

    # --- interactive (plotly) — the GUI view, hover for SNP / p / r² / block ---
    fig_i = plot_regional_association_interactive(
        wdf, r2_final, window.lead_snp, sig_threshold,
        block_interval=block_interval, block_members=block_members,
        seed_threshold=seed_threshold)
    st.plotly_chart(fig_i, use_container_width=True)

    st.caption(
        f"r² is computed from your own genotypes (not a reference panel). Threshold: "
        f"{rule.label}. **Display of the existing scan** — no fine-mapping, credible sets, "
        f"or candidate ranking; no p-value is changed."
    )

    # --- static (matplotlib) — the downloadable / report figure, with gene track ---
    with st.expander("Static figure + downloads (PNG / SVG / PDF)", expanded=False):
        fig_s = plot_regional_association_static(
            wdf, r2_final, window.lead_snp, sig_threshold,
            block_interval=block_interval, block_members=block_members,
            genes=genes, seed_threshold=seed_threshold)
        buf = BytesIO()
        fig_s.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        st.image(buf.getvalue())
        stem = f"Regional_{window.lead_snp}_Chr{window.chr}_{window.start_bp}_{window.end_bp}"
        export_matplotlib(fig_s, stem, label_prefix="Download regional plot")
        plt.close(fig_s)
