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


def _per_model_rule(table, limit):
    """The overlay's rule written out: within each model, every point with p <= 0.01 and every k-th of
    its others in genome order, with the smallest k (the same for every model) that keeps the drawing
    at or below ``limit``."""
    p = pd.to_numeric(table["PValue"], errors="coerce").to_numpy()
    keep = np.flatnonzero(p <= MANHATTAN_KEEP_P)
    others = []
    for m in table["Model"].unique():
        o = np.flatnonzero(~(p <= MANHATTAN_KEEP_P) & (table["Model"] == m).to_numpy())
        others.append(o[np.argsort(table["CumPos"].to_numpy()[o], kind="stable")])
    k = 1
    while keep.size + sum(-(-o.size // k) for o in others) > limit:
        k += 1
    return np.sort(np.concatenate([keep] + [o[::k] for o in others])), k


def _drawn_points(fig):
    """(model, CumPos, p-value) of every point the overlay draws, in genome order (a SNP's models in the
    figure's order)."""
    pts = [(tr.name, float(x), float(cd[3]), i) for i, tr in enumerate(fig.data)
           for x, cd in zip(tr.x, tr.customdata)]
    return [(m, x, p) for m, x, p, _i in sorted(pts, key=lambda t: (t[1], t[3]))]


def _longest_run_of_one_model(points):
    """The longest run of one model among the drawn points with p > 0.01, in genome order."""
    models = [m for m, _x, p in points if not p <= MANHATTAN_KEEP_P]
    best = run = 0
    for i, m in enumerate(models):
        run = run + 1 if i and m == models[i - 1] else 1
        best = max(best, run)
    return best


def test_above_the_limit_the_overlay_draws_the_rule_within_every_model(monkeypatch):
    monkeypatch.setattr(plotting, "MANHATTAN_MAX_POINTS", 1000)
    res = _all_res(900)                                      # 2,700 points in all
    fig, table = plotting.plot_manhattan_overlay(res)
    rows, k = plotting.manhattan_overlay_thinning(table)
    expected_rows, expected_k = _per_model_rule(table, 1000)
    assert k == expected_k >= 2 and np.array_equal(rows, expected_rows) and rows.size <= 1000
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
    assert plotting.manhattan_overlay_caption(table) == (
        f"{rows.size:,} of 2,700 points drawn (900 SNPs x 3 models): all with p ≤ 0.01 and 1 in {k} of "
        "each model's others. Each model's static Manhattan and the cross-model results have every SNP.")


def test_the_overlay_rule_counts_the_points_of_every_model():
    """At the real limit: 3 models x 200,000 SNPs is 600,000 points, so the overlay is thinned although
    each model alone is under the limit; its caption counts points."""
    m = 200_000
    table = pd.DataFrame({"Model": np.repeat(["MLM", "MLMM", "FarmCPU"], m),
                          "PValue": np.tile(np.linspace(0.0, 1.0, m), 3), "CumPos": np.tile(np.arange(m), 3)})
    assert plotting.manhattan_overlay_thinning(table.iloc[:m]) == (None, 1)
    rows, k = plotting.manhattan_overlay_thinning(table)
    assert k == 2 and rows.size <= MANHATTAN_MAX_POINTS
    assert plotting.manhattan_overlay_caption(table).startswith(
        f"{rows.size:,} of 600,000 points drawn (200,000 SNPs x 3 models): all with p ≤ 0.01 and 1 in 2 of")


def _r9b_table(m, jitter, seed=1):
    """The round 1 re-check's r9b case, as plot_manhattan_overlay holds it: 2 models of ``m`` SNPs on 12
    chromosomes of 90 Mb, MLMM's p-values those of MLM with a small jitter (the models agree away from
    the peaks), the rows in genome order with a SNP's two rows together, MLM first."""
    rng = np.random.default_rng(seed)
    chrom = np.repeat(np.arange(1, 13), -(-m // 12))[:m]
    pos = np.concatenate([np.sort(rng.choice(90_000_000, size=int((chrom == c).sum()), replace=False)) + 1
                          for c in range(1, 13)])
    cum = pos + (chrom - 1) * 91_000_000                     # each chromosome 1 Mb after the last
    p_mlm = rng.uniform(size=m)
    p_mlmm = np.clip(p_mlm * np.exp(rng.normal(0, jitter, size=m)), 0, 1)
    return pd.DataFrame({"Model": np.tile(["MLM", "MLMM"], m), "CumPos": np.repeat(cum, 2),
                         "PValue": np.column_stack([p_mlm, p_mlmm]).ravel()})


def _windows_of_one_model(table, rows, width=1_000_000):
    """The share of the windows of ``width`` bp whose drawn points with p > 0.01 are 90% or more of one
    model."""
    d = table.iloc[rows]
    bg = d[~(d["PValue"].to_numpy() <= MANHATTAN_KEEP_P)]
    share = bg.groupby(bg["CumPos"] // width)["Model"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    return float((share >= 0.9).mean())


def test_every_region_draws_each_models_background_at_the_real_limit():
    """r9b's case (2 models x 300,000 SNPs, MLMM's p-values MLM's with a 5% jitter): k is 2, a multiple
    of the model count. One k over the models' points together drew one model's background only in
    91.5% of the 1-Mb windows (each SNP's two rows lie together, so every other point is the same
    model's until a SNP kept for one model only shifts the picks); within each model, every window
    draws both models alike."""
    table = _r9b_table(300_000, 0.05)
    rows, k = plotting.manhattan_overlay_thinning(table)
    assert k == 2 and rows.size <= MANHATTAN_MAX_POINTS
    assert _windows_of_one_model(table, rows) < 0.01
    bg = table.iloc[rows]
    bg = bg[~(bg["PValue"].to_numpy() <= MANHATTAN_KEEP_P)]
    drawn = bg["Model"].value_counts()
    others = (~(table["PValue"].to_numpy() <= MANHATTAN_KEEP_P))
    for model in ("MLM", "MLMM"):                            # each model: 1 in 2 of its others
        assert drawn[model] == -(-int((others & (table["Model"] == model).to_numpy()).sum()) // 2)
    combined, k_combined = manhattan_thinning(table)          # the one k over every point, as before
    assert k_combined == 2 and _windows_of_one_model(table, combined) > 0.9


def test_the_drawn_overlay_alternates_the_models_in_every_region(monkeypatch):
    """r9b's case scaled down (2 models x 3,000 SNPs, the limit lowered to 5,000, so k is 2): the figure's
    points with p > 0.01, in genome order, never run long on one model."""
    monkeypatch.setattr(plotting, "MANHATTAN_MAX_POINTS", 5_000)
    table = _r9b_table(3_000, 0.05)
    res = pd.DataFrame({"Model": table["Model"], "SNP": [f"s{i // 2}" for i in range(len(table))],
                        "Chr": (table["CumPos"] // 91_000_000 + 1).astype(str),
                        "Pos": table["CumPos"] % 91_000_000, "PValue": table["PValue"]})
    res["-log10p"] = -np.log10(np.clip(res["PValue"].astype(float), 1e-300, 1.0))
    res = pd.concat([res[res["Model"] == "MLM"], res[res["Model"] == "MLMM"]], ignore_index=True)
    fig, overlay = plotting.plot_manhattan_overlay(res)
    assert _longest_run_of_one_model(_drawn_points(fig)) <= 4
    rows, k = plotting.manhattan_overlay_thinning(overlay)
    assert k == 2 and sum(len(tr.x) for tr in fig.data) == rows.size
