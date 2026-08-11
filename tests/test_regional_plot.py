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
