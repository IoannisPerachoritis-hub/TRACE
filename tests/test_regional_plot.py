"""Unit tests for the regional (locus-zoom-style) association plot — T-08.

These exercise the PURE functions in gwas/plotting.py (no Streamlit): a matplotlib
Figure is built and inspected. Edge cases from the design doc (all-NaN r², no block,
no genes, below-seed marks, isolated flank interval) must render without error.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gwas.plotting import (  # noqa: E402
    plot_regional_association_static,
    plot_regional_association_interactive,
    ld_pairs_long,
    _r2_point_colors,
    _R2_BIN_COLORS,
    _R2_NA_COLOR,
)


def _window(n=30, seed=0, lead_idx=0):
    rng = np.random.default_rng(seed)
    pos = np.sort(rng.integers(1_000_000, 2_000_000, n))
    df = pd.DataFrame({
        "SNP": [f"s{i}" for i in range(n)],
        "Chr": ["2"] * n,
        "Pos": pos,
        "PValue": rng.uniform(1e-9, 1.0, n),
    })
    r2 = rng.uniform(0.0, 1.0, n)
    r2[lead_idx] = 1.0
    return df, r2


def test_r2_point_colors_bins_and_nan():
    r2 = np.array([0.05, 0.3, 0.5, 0.7, 0.9, np.nan])
    cols = _r2_point_colors(r2)
    assert cols[:5] == _R2_BIN_COLORS  # one colour per ascending bin
    assert cols[5] == _R2_NA_COLOR     # NaN -> grey


def test_static_returns_figure_with_threshold_line():
    df, r2 = _window()
    fig = plot_regional_association_static(df, r2, "s0", 5e-8)
    ax = fig.axes[0]
    thr = -np.log10(5e-8)
    dashed = [ln.get_ydata()[0] for ln in ax.get_lines()
              if ln.get_linestyle() in ("--", "dashed")]
    assert any(abs(y - thr) < 1e-6 for y in dashed), "no significance line at threshold"
    assert len(fig.axes) == 1  # no gene track requested


def test_static_no_threshold_line_when_none():
    """FDR path: sig_threshold=None -> no horizontal line drawn."""
    df, r2 = _window()
    fig = plot_regional_association_static(df, r2, "s0", None)
    ax = fig.axes[0]
    dashed = [ln for ln in ax.get_lines() if ln.get_linestyle() in ("--", "dashed")]
    assert not dashed


def test_static_with_block_genes_and_seed_marks():
    df, r2 = _window()
    genes = pd.DataFrame({
        "Chr": ["2", "2"],
        "Start": [1_100_000, 1_500_000],
        "End": [1_200_000, 1_600_000],
        "Strand": ["+", "-"],
        "Gene_ID": ["Solyc02g0001", "Solyc02g0002"],
    })
    fig = plot_regional_association_static(
        df, r2, "s0", 1e-3,
        block_interval=(1_200_000, 1_400_000),
        block_members={"s1", "s2"},
        genes=genes,
        seed_threshold=1e-5,
    )
    assert len(fig.axes) == 2  # association + gene track
    # gene track carries the two gene bars
    axg = fig.axes[1]
    assert len([ln for ln in axg.get_lines()]) >= 2


def test_static_all_nan_r2_renders_grey():
    df, r2 = _window()
    r2 = np.full(len(df), np.nan)  # single typed marker / no computable r²
    fig = plot_regional_association_static(df, r2, "s0", 5e-8)
    assert fig is not None and len(fig.axes) == 1


def test_static_isolated_flank_interval_no_members():
    """Isolated SNP: a flanking interval is shaded but no block members."""
    df, r2 = _window()
    fig = plot_regional_association_static(
        df, r2, "s0", 5e-8,
        block_interval=(900_000, 1_050_000),  # flanking-marker interval
        block_members=None,
    )
    assert fig is not None


def test_static_lead_outside_window_does_not_crash():
    df, r2 = _window()
    fig = plot_regional_association_static(df, r2, "not_in_window", 5e-8)
    assert fig is not None  # no diamond drawn, but renders


def test_interactive_returns_plotly_figure():
    df, r2 = _window()
    fig = plot_regional_association_interactive(
        df, r2, "s0", 5e-8,
        block_interval=(1_100_000, 1_300_000),
        block_members={"s1"},
        seed_threshold=1e-5,
    )
    assert fig is not None and len(fig.data) >= 1


# ---- T-08 follow-ups (interactive-check fixes) ----

def test_interactive_uirevision_is_set():
    """Fix 1: uirevision carries the window signature so Plotly resets axes on a
    window change but preserves zoom within one."""
    df, r2 = _window()
    fig = plot_regional_association_interactive(df, r2, "s0", 5e-8, uirevision="2:1-2:s0")
    assert fig.layout.uirevision == "2:1-2:s0"
    # default is None (backward-compatible)
    fig2 = plot_regional_association_interactive(df, r2, "s0", 5e-8)
    assert fig2.layout.uirevision is None


def test_static_gene_labels_toggle():
    """Fix 2: gene-name labels are drawn only when gene_labels=True; the gene axis
    is otherwise bars-only (strand arrowheads are Line2D, not text)."""
    df, r2 = _window()
    genes = pd.DataFrame({
        "Chr": ["2", "2"], "Start": [1_100_000, 1_500_000],
        "End": [1_200_000, 1_600_000], "Strand": ["+", "-"],
        "Gene_ID": ["Solyc02g0001", "Solyc02g0002"],
    })
    fig_on = plot_regional_association_static(df, r2, "s0", 1e-3, genes=genes, gene_labels=True)
    fig_off = plot_regional_association_static(df, r2, "s0", 1e-3, genes=genes, gene_labels=False)
    assert len(fig_on.axes[1].texts) == 2   # one name per gene
    assert len(fig_off.axes[1].texts) == 0  # bars only
    # strand arrowheads (Line2D) survive with labels off: gene bars + arrowheads
    assert len(fig_off.axes[1].lines) >= 4


def test_ld_pairs_long_shape_and_values():
    """Fix 3: the tidy long-format helper for the r² CSV export."""
    m = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.8], [0.2, 0.8, 1.0]])
    long = ld_pairs_long(m, ["a", "b", "c"])
    assert list(long.columns) == ["SNP_A", "SNP_B", "r2"]
    assert len(long) == 3  # n*(n-1)/2 upper-triangle pairs
    assert set(zip(long.SNP_A, long.SNP_B)) == {("a", "b"), ("a", "c"), ("b", "c")}
    bc = long[(long.SNP_A == "b") & (long.SNP_B == "c")].iloc[0]
    assert bc.r2 == 0.8


def test_clamp_block_span_modes():
    """Round-2 fix 2: clamp a block span to the plotted x-range + fill/edges/none."""
    from gwas.plotting import _clamp_block_span
    bs, be, mode = _clamp_block_span(1.2, 1.5, 1.0, 2.0)   # inside, small
    assert mode == "fill" and bs == 1.2 and be == 1.5
    bs, be, mode = _clamp_block_span(0.0, 5.0, 1.0, 2.0)   # far wider than the axis
    assert mode == "edges" and bs == 1.0 and be == 2.0     # clamped to [xlo, xhi]
    _, _, mode = _clamp_block_span(3.0, 4.0, 1.0, 2.0)     # entirely outside
    assert mode == "none"


def test_static_wide_block_does_not_blow_out_axis():
    """A block far wider than the data must not expand the panel (clamped + edges)."""
    df, r2 = _window()
    fig = plot_regional_association_static(df, r2, "s0", 5e-8, block_interval=(0, 50_000_000))
    x0, x1 = fig.axes[0].get_xlim()
    xhi = df["Pos"].max() / 1e6
    assert x1 < xhi + 1.0, "axis expanded to the unclamped block span"


def test_block_for_window_derives_from_lead_containing_block():
    """Round-2 fix 3: with no explicit block bounds, shade the block CONTAINING the lead."""
    import types
    from pages._ld_tabs.tab_regional import _block_for_window
    from pages._ld_tabs._window import RegionWindow
    hd = pd.DataFrame({
        "Chr": ["2"], "Start (bp)": [1_100_000], "End (bp)": [1_600_000],
        "SNP_IDs": ["s5,s6,s7"],
    })
    ctx = types.SimpleNamespace(haplo_df_auto=hd)
    win = RegionWindow(lead_snp="s6", chr="2", start_bp=1_000_000, end_bp=1_700_000,
                       core_start=None, core_end=None, use_block=False, label="",
                       lead_pos=1_300_000, buffer_kb=200)
    interval, members = _block_for_window(ctx, win)
    assert interval == (1_100_000, 1_600_000)
    assert members == {"s5", "s6", "s7"}
    # a lead NOT in any block -> no span (flank fallback happens in the caller)
    win2 = RegionWindow(lead_snp="zzz", chr="2", start_bp=5_000_000, end_bp=6_000_000,
                        core_start=None, core_end=None, use_block=False, label="",
                        lead_pos=5_500_000, buffer_kb=200)
    assert _block_for_window(ctx, win2) == (None, None)


def test_extract_by_snp_ids_is_member_aware():
    """Fix 4: snp_ids extraction returns block MEMBERS only (reproduces Block
    Heatmaps), while positional extraction returns the whole interval."""
    from gwas.ld import extract_block_geno_for_paper
    rng = np.random.default_rng(0)
    n = 30
    chroms = np.array(["2"] * 5)
    positions = np.array([1000, 1200, 1400, 1600, 1800])
    sid = np.array(["m1", "x1", "m2", "x2", "m3"])  # members m1/m2/m3; x1/x2 in-interval non-members
    geno = rng.binomial(2, 0.3, (n, 5)).astype(float)
    _, _, sids_mem = extract_block_geno_for_paper(
        geno, chroms, positions, sid, "2", 1000, 1800, snp_ids="m1,m2,m3")
    _, _, sids_pos = extract_block_geno_for_paper(
        geno, chroms, positions, sid, "2", 1000, 1800, snp_ids=None)
    assert set(sids_mem.astype(str)) == {"m1", "m2", "m3"}
    assert set(sids_pos.astype(str)) == {"m1", "x1", "m2", "x2", "m3"}


def test_absent_gene_track_caption_distinguishes_no_model_from_empty_window():
    """WO-GUI-01: the zero-gene case discloses WHICH fact applies. No model loaded
    (genes_df None or empty) points the user to load one; a loaded model with no
    genes in the window states that instead."""
    from pages._ld_tabs.tab_regional import _absent_gene_track_caption
    for gm in (None, pd.DataFrame()):
        msg = _absent_gene_track_caption(gm)
        assert "No gene model loaded" in msg and "Gene Annotation tab" in msg
    loaded = pd.DataFrame({"Chr": ["2"], "Start": [1000], "End": [2000], "Gene_ID": ["G1"]})
    assert _absent_gene_track_caption(loaded) == "No annotated genes fall in this window."
