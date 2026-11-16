"""The interactive Manhattan above 500,000 SNPs (gwas.plotting.plot_manhattan_interactive).

Above MANHATTAN_MAX_POINTS SNPs it draws every SNP with p <= 0.01 and every k-th of the others in genome
order, with k the smallest whole number that keeps the drawing at or below the limit, and the page shows
manhattan_caption's sentence under it. Up to the limit the figure is the one ff56e5c drew, byte for byte
(compared here with that commit's function, copied below). The limit is lowered to 1,000 for the figure
tests so they stay fast; the rule itself is checked at the real limit too.
"""
import numpy as np
import pandas as pd
import plotly.express as px

from gwas import plotting
from gwas.plotting import (MANHATTAN_KEEP_P, MANHATTAN_MAX_POINTS, compute_cumulative_positions,
                           manhattan_caption, manhattan_thinning, plot_manhattan_interactive)


def _ff56e5c_plot_manhattan_interactive(df, active_lod, active_label, title, secondary_lod=None,
                                        secondary_label=None):
    """gwas/plotting.py's plot_manhattan_interactive at ff56e5c, unchanged."""
    from annotation import canon_chr
    from utils.pub_theme import MANHATTAN_COLORS, PALETTE, SIG_LINE_COLOR

    def _chr_key(x):
        c = canon_chr(x)
        try:
            return (0, int(c))
        except ValueError:
            return (1, c)

    chroms_unique = sorted(df["Chr"].astype(str).unique(), key=_chr_key)
    chr_color_map = {ch: MANHATTAN_COLORS[i % 2] for i, ch in enumerate(chroms_unique)}
    df = df.copy()
    df["Chr"] = df["Chr"].astype(str)
    fig = px.scatter(
        df, x="CumPos", y="-log10p", color="Chr", color_discrete_map=chr_color_map,
        hover_data=[c for c in ["SNP", "Chr", "Pos", "PValue", "Beta_OLS", "SE_OLS", "Tstat_OLS"] if c in df.columns],
        title=title, render_mode="webgl")
    fig.update_traces(marker_size=4)
    fig.update_layout(showlegend=False)
    if secondary_lod is not None and secondary_label is not None and not str(secondary_label).startswith("FDR"):
        fig.add_hline(y=secondary_lod, line_dash="dot", annotation_text=secondary_label,
                      annotation_position="top right", line_color=PALETTE.get("grey", "#7F7F7F"))
    if active_lod is not None and not str(active_label).startswith("FDR"):
        fig.add_hline(y=active_lod, line_dash="dash", annotation_text=active_label,
                      annotation_position="top left", line_color=SIG_LINE_COLOR)
    return fig


def _table(m, seed=0, n_chr=3):
    """An MLM table in genome order with the page's hover columns; a few SNPs exactly at p = 0.01,
    some below, one NaN p-value."""
    rng = np.random.default_rng(seed)
    per = [m // n_chr + (1 if i < m % n_chr else 0) for i in range(n_chr)]
    chrom = np.concatenate([np.full(k, str(i + 1)) for i, k in enumerate(per)])
    pos = np.concatenate([np.sort(rng.choice(np.arange(1, 10_000_000), size=k, replace=False)) for k in per])
    p = rng.uniform(size=m)
    p[rng.choice(m, size=max(1, m // 50), replace=False)] = rng.uniform(1e-8, 0.01, size=max(1, m // 50))
    p[[3, 7]] = 0.01
    df = pd.DataFrame({"SNP": [f"s{i}" for i in range(m)], "Chr": chrom, "Pos": pos, "PValue": p,
                       "Beta_OLS": rng.normal(size=m), "SE_OLS": rng.uniform(0.1, 1, size=m),
                       "Tstat_OLS": rng.normal(size=m)})
    df.loc[11, "PValue"] = np.nan
    df["-log10p"] = -np.log10(df["PValue"])
    return compute_cumulative_positions(df)[0]


def test_up_to_the_limit_the_figure_is_ff56e5cs(monkeypatch):
    monkeypatch.setattr(plotting, "MANHATTAN_MAX_POINTS", 1000)
    for m in (300, 1000):                                    # below and at the (lowered) limit
        df = _table(m)
        new = plot_manhattan_interactive(df, 5.0, "Bonferroni", "t", secondary_lod=4.0, secondary_label="M_eff")
        old = _ff56e5c_plot_manhattan_interactive(df, 5.0, "Bonferroni", "t", secondary_lod=4.0,
                                                  secondary_label="M_eff")
        assert new.to_json() == old.to_json(), m
        assert manhattan_caption(df) is None


def _drawn_rows(df, fig):
    """The rows of ``df`` the figure draws, from each trace's points (SNP names in hover data)."""
    names = set()
    for tr in fig.data:
        names |= set(np.asarray(tr.customdata)[:, 0].astype(str)) if tr.customdata is not None else set()
    return np.flatnonzero(df["SNP"].astype(str).isin(names).to_numpy())


def test_above_the_limit_it_draws_the_snps_of_the_rule(monkeypatch):
    monkeypatch.setattr(plotting, "MANHATTAN_MAX_POINTS", 1000)
    df = _table(2600)
    p = df["PValue"].to_numpy()
    keep = np.flatnonzero(p <= MANHATTAN_KEEP_P)                 # includes the two at exactly 0.01
    assert {3, 7} <= set(keep.tolist())
    others = np.flatnonzero(~(p <= MANHATTAN_KEEP_P))            # the NaN p-value is among them
    others = others[np.argsort(df["CumPos"].to_numpy()[others], kind="stable")]
    k = 1
    while keep.size + -(-others.size // k) > 1000:               # the smallest k that fits
        k += 1
    expected = np.sort(np.concatenate([keep, others[::k]]))
    rows, k_got = manhattan_thinning(df)
    assert k_got == k and np.array_equal(rows, expected)
    assert rows.size <= 1000 < keep.size + -(-others.size // (k - 1))
    fig = plot_manhattan_interactive(df, 5.0, "Bonferroni", "t")
    assert np.array_equal(_drawn_rows(df, fig), expected)
    assert sum(len(tr.x) for tr in fig.data) == expected.size
    # the drawn rows are the table's rows themselves (same x, y and hover values)
    ref = _ff56e5c_plot_manhattan_interactive(df.iloc[expected], 5.0, "Bonferroni", "t")
    assert fig.to_json() == ref.to_json()
    assert manhattan_caption(df) == (
        f"{expected.size:,} of 2,600 SNPs drawn: all with p ≤ 0.01 and 1 in {k:,} of the others. "
        "The static Manhattan and the results table have every SNP.")


def test_the_rule_at_the_real_limit():
    assert MANHATTAN_MAX_POINTS == 500_000
    m = 500_000
    df = pd.DataFrame({"PValue": np.linspace(0.0, 1.0, m), "CumPos": np.arange(m)})
    assert manhattan_thinning(df) == (None, 1)                    # at the limit: every SNP
    df = pd.DataFrame({"PValue": np.linspace(0.0, 1.0, 1_070_795), "CumPos": np.arange(1_070_795)})
    rows, k = manhattan_thinning(df)
    n_keep = int((df["PValue"] <= 0.01).sum())
    assert k == 3 and rows.size == n_keep + -(-(1_070_795 - n_keep) // 3) <= 500_000


def test_when_the_small_p_values_alone_exceed_the_limit(monkeypatch):
    monkeypatch.setattr(plotting, "MANHATTAN_MAX_POINTS", 10)
    df = pd.DataFrame({"PValue": [0.001] * 12 + [0.5] * 5, "CumPos": np.arange(17)})
    rows, k = manhattan_thinning(df)
    assert k == 0 and np.array_equal(rows, np.arange(12))
    assert "none of the others" in manhattan_caption(df)


# ------------------------------------------------------- the cross-model overlay Manhattan
def _ff56e5c_overlay(all_res):
    """The GWAS page's "Overlay Manhattan by model" at ff56e5c (pages/GWAS_analysis.py), unchanged."""
    from gwas.io import _clean_chr_series
    from utils.pub_theme import PALETTE_CYCLE

    df_plot_multi = all_res.copy()
    chr_clean, chr_num_multi, order_labels_multi, order_map_multi = _clean_chr_series(df_plot_multi["Chr"])
    df_plot_multi["Chr"] = chr_clean
    df_plot_multi["ChrNum"] = chr_num_multi.astype(int)
    df_plot_multi["Pos"] = df_plot_multi["Pos"].astype(int)
    df_plot_multi = df_plot_multi.sort_values(["ChrNum", "Pos"]).reset_index(drop=True)
    gap = 1_000_000
    offset = 0
    tick_pos_multi, tick_lab_multi = [], []
    for chnum in np.sort(df_plot_multi["ChrNum"].unique()):
        mask = df_plot_multi["ChrNum"] == chnum
        pos_min = df_plot_multi.loc[mask, "Pos"].min()
        pos_max = df_plot_multi.loc[mask, "Pos"].max()
        df_plot_multi.loc[mask, "CumPos"] = df_plot_multi.loc[mask, "Pos"] + offset
        tick_pos_multi.append(offset + (pos_min + pos_max) / 2)
        tick_lab_multi.append(str(chnum))
        offset += (pos_max - pos_min) + gap
    _models_unique = df_plot_multi["Model"].unique().tolist()
    _model_color_map = {m: PALETTE_CYCLE[i % len(PALETTE_CYCLE)] for i, m in enumerate(_models_unique)}
    fig_overlay = px.scatter(
        df_plot_multi,
        x="CumPos", y="-log10p", color="Model",
        color_discrete_map=_model_color_map,
        hover_data=["SNP", "Chr", "Pos", "PValue"],
        title="Overlay Manhattan by model",
        render_mode="webgl"
    )
    fig_overlay.update_layout(
        xaxis_title="Chromosome (cumulative)",
        yaxis_title="-log10(p)",
        xaxis=dict(tickmode="array", tickvals=tick_pos_multi, ticktext=tick_lab_multi, showgrid=False),
        height=520,
    )
    return fig_overlay


def _all_res(m, models=("MLM", "MLMM", "FarmCPU"), seed=1):
    """The page's cross-model frame (all_res) of ``m`` SNPs per model."""
    base = _table(m, seed=seed)
    rng = np.random.default_rng(seed + 10)
    parts = []
    for i, model in enumerate(models):
        p = base["PValue"].to_numpy() if i == 0 else rng.uniform(size=m)
        parts.append(pd.DataFrame({"Model": model, "SNP": base["SNP"], "Chr": base["Chr"], "Pos": base["Pos"],
                                   "PValue": p, "-log10p": -np.log10(np.clip(p, 1e-300, 1.0))}))
    return pd.concat(parts, ignore_index=True)


def test_the_overlay_up_to_the_limit_is_ff56e5cs(monkeypatch):
    monkeypatch.setattr(plotting, "MANHATTAN_MAX_POINTS", 1000)
    for m in (100, 333):                                     # 300 and 999 points: below the limit
        res = _all_res(m)
        fig, table = plotting.plot_manhattan_overlay(res)
        assert fig.to_json() == _ff56e5c_overlay(res).to_json(), m
        assert manhattan_caption(table) is None and len(table) == 3 * m


def test_above_the_limit_the_overlay_draws_the_rule_over_every_models_points(monkeypatch):
    monkeypatch.setattr(plotting, "MANHATTAN_MAX_POINTS", 1000)
    res = _all_res(900)                                      # 2,700 points in all
    fig, table = plotting.plot_manhattan_overlay(res)
    rows, k = manhattan_thinning(table)
    assert k >= 2 and rows.size <= 1000
    assert sum(len(tr.x) for tr in fig.data) == rows.size
    def point(m, x, y):
        return m, float(x), repr(float(y))                   # a NaN p-value (one SNP) compares too

    drawn = {point(tr.name, x, y) for tr in fig.data for x, y in zip(tr.x, tr.y)}
    expected = table.iloc[rows]
    assert drawn == {point(m, x, y) for m, x, y in zip(expected["Model"], expected["CumPos"],
                                                        expected["-log10p"])}
    assert set(np.flatnonzero(table["PValue"].to_numpy() <= MANHATTAN_KEEP_P)) <= set(rows.tolist())
    full = _ff56e5c_overlay(res)                             # the colours and the axis of every point
    assert [(t.name, t.marker.color) for t in fig.data] == [(t.name, t.marker.color) for t in full.data]
    assert fig.layout.xaxis.tickvals == full.layout.xaxis.tickvals
    assert manhattan_caption(table).startswith(f"{rows.size:,} of 2,700 SNPs drawn: all with p ≤ 0.01")


def test_the_overlay_rule_counts_the_points_of_every_model():
    """At the real limit: 3 models x 200,000 SNPs is 600,000 points, so the overlay is thinned although
    each model alone is under the limit."""
    m = 200_000
    table = pd.DataFrame({"PValue": np.tile(np.linspace(0.0, 1.0, m), 3), "CumPos": np.tile(np.arange(m), 3)})
    assert manhattan_thinning(table.iloc[:m]) == (None, 1)
    rows, k = manhattan_thinning(table)
    assert k == 2 and rows.size <= MANHATTAN_MAX_POINTS
