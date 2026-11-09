"""The LD block's haplotype box plot, drawn as the lead-SNP plot (gwas/snpplots.py).

hap_plot_values takes a block's rows of the haplotype_assignments table and gives its
in-test samples with the values the panel plots and tests: PC-adjusted when the GWAS used
PCs (the residuals on an intercept plus the PCs), else the uploaded values. Tukey HSD and
its compact letters (gwas.cld.tukey_cld) take those values; hap_plot_data adds the scale
and the letters, and is the plot's "Download plotted data (CSV)". plotly_hap_boxplot and
render_hap_boxplot draw exactly those rows, coloured by haplotype_color_map, with each
group's letters above its box.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.stats.multicomp as mc  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

from gwas.cld import tukey_cld  # noqa: E402
from gwas.snpplots import (  # noqa: E402
    HAP_PLOT_DATA_COLUMNS,
    hap_axis_label,
    hap_boxplot_title,
    hap_plot_data,
    hap_plot_values,
    plotly_hap_boxplot,
    render_hap_boxplot,
)
from utils.pub_theme import haplotype_color_map  # noqa: E402

RNG = np.random.default_rng(2)
_N = 40
_SEQ = {"H1": "0|2|0", "H2": "2|0|2", "H3": "2|2|0", "Other": "0|0|2", "H4": "2|0|0"}
ROWS = pd.DataFrame({      # a block's assignments: H1-H3 tested, Other and H4 left out
    "Chr": "2", "Block_start": 100, "Block_end": 300,
    "Sample": [f"S{i:02d}" for i in range(_N)],
    "Haplotype": (["H2"] * 10 + ["H1"] * 14 + ["H3"] * 8 + ["Other"] * 5 + ["H4"] * 3),
    "Phenotype_used": RNG.normal(10, 2, _N).round(3),
})
ROWS.loc[ROWS["Haplotype"] == "H3", "Phenotype_used"] += 6.0      # H3 differs: its own letter
ROWS["Allele_sequence"] = ROWS["Haplotype"].map(_SEQ)
ROWS["in_haplotype_test"] = ROWS["Haplotype"].isin(["H1", "H2", "H3"])
TESTED = ROWS[ROWS["in_haplotype_test"]]

# the GWAS's PCs: another sample order, two samples not in the block, one tested sample
# (S03, H2) with a missing PC
_PC_IDS = [f"S{i:02d}" for i in range(_N)][::-1] + ["X1", "X2"]
_PCS = np.random.default_rng(5).normal(size=(len(_PC_IDS), 2))
_PCS[_PC_IDS.index("S03"), 1] = np.nan


def _plot_data(pcs=None):
    """The tab's calls: the panel's values, Tukey HSD on them, the plot's data."""
    v = hap_plot_values(ROWS, pcs=pcs, pc_samples=_PC_IDS)
    _tukey, cld = tukey_cld(v["Phenotype_plotted"], v["Haplotype"])
    return v, cld, hap_plot_data(v, dict(zip(cld["MLG_label"], cld["CLD"])))


def test_the_data_are_the_blocks_in_test_rows():
    d = hap_plot_data(hap_plot_values(ROWS))
    assert list(d.columns) == HAP_PLOT_DATA_COLUMNS
    assert d["Sample"].tolist() == TESTED["Sample"].tolist()          # in table order
    assert d["Haplotype"].tolist() == TESTED["Haplotype"].tolist()
    assert d["Allele_sequence"].tolist() == TESTED["Allele_sequence"].tolist()
    assert d["Phenotype_uploaded"].tolist() == TESTED["Phenotype_used"].tolist()
    assert d["Phenotype_plotted"].tolist() == TESTED["Phenotype_used"].tolist()
    assert set(d["Phenotype_scale"]) == {"uploaded"} and set(d["CLD_letters"]) == {""}
    assert bool(d["In_haplotype_test"].all())


def test_the_interactive_plot_shows_those_points():
    d = hap_plot_data(hap_plot_values(ROWS))
    title = hap_boxplot_title("2", 100, 300, 0.0123, 0.214)
    assert title == ["Chr2:100-300", "P (permutation) = 0.0123 · η² = 0.21"]
    fig = plotly_hap_boxplot(d, title, y_label="t")
    pts = [tr for tr in fig.data if tr.type == "scatter"]
    got = {cd[0]: (cd[1], float(yv), cd[2]) for tr in pts for yv, cd in zip(tr.y, tr.customdata)}
    assert got == {s: (h, float(v), "yes") for s, h, v in
                   zip(TESTED["Sample"], TESTED["Haplotype"], TESTED["Phenotype_used"])}
    cmap = haplotype_color_map(["H1", "H2", "H3"])
    assert {tr.name: to_hex(tr.marker.color) for tr in pts} == {
        h: to_hex(cmap[h]) for h in ("H1", "H2", "H3")}
    assert len([tr for tr in fig.data if tr.type == "box"]) == 3
    # H1, H2, ... from the left, whatever the rows' order (H2's rows come first)
    assert list(fig.layout.xaxis.ticktext) == ["H1<br>(n=14)", "H2<br>(n=10)", "H3<br>(n=8)"]
    assert fig.layout.title.text == "Chr2:100-300<br>P (permutation) = 0.0123 · η² = 0.21"
    assert fig.layout.yaxis.title.text == "t"
    assert not fig.layout.annotations                       # no letters given, none drawn


def test_the_static_plot_shows_the_same_points_at_the_same_places():
    d = hap_plot_data(hap_plot_values(ROWS))
    title = hap_boxplot_title("2", 100, 300, 0.0123, 0.214)
    fig = render_hap_boxplot(d, title, y_label="t")
    try:
        ax = fig.axes[0]
        dots = [(float(x), float(y), to_hex(ln.get_color())) for ln in ax.lines
                if ln.get_marker() == "o" for x, y in zip(ln.get_xdata(), ln.get_ydata())]
        assert [t.get_text() for t in ax.get_xticklabels()] == ["H1\n(n=14)", "H2\n(n=10)",
                                                                "H3\n(n=8)"]
        assert ax.get_title() == "\n".join(title) and ax.get_ylabel() == "t"
        assert not ax.texts
    finally:
        plt.close(fig)
    cmap = haplotype_color_map(["H1", "H2", "H3"])
    labels = ["H1", "H2", "H3"]
    want = sorted((float(v), to_hex(cmap[h])) for h, v in zip(TESTED["Haplotype"],
                                                            TESTED["Phenotype_used"]))
    assert sorted((y, c) for _x, y, c in dots) == want
    for x, _y, c in dots:                       # each dot sits over its own group
        assert to_hex(cmap[labels[int(round(x))]]) == c
    # the interactive points are at the static plot's places (same jitter)
    pfig = plotly_hap_boxplot(d, title)
    pxy = sorted((round(float(x), 12), float(y)) for tr in pfig.data if tr.type == "scatter"
                 for x, y in zip(tr.x, tr.y))
    assert pxy == sorted((round(x, 12), y) for x, y, _c in dots)


def test_with_pcs_the_plot_shows_the_values_tukey_uses(monkeypatch):
    seen = {}
    real = mc.pairwise_tukeyhsd

    def recorder(endog, groups, alpha=0.05):
        seen["endog"] = np.asarray(endog, dtype=float).tolist()
        seen["groups"] = [str(g) for g in groups]
        return real(endog=endog, groups=groups, alpha=alpha)

    monkeypatch.setattr(mc, "pairwise_tukeyhsd", recorder)
    v, _cld, d = _plot_data(pcs=_PCS)
    # Tukey HSD took exactly the plotted values: same samples, same order
    assert seen["endog"] == d["Phenotype_plotted"].tolist()
    assert seen["groups"] == d["Haplotype"].tolist()
    keep = TESTED[TESTED["Sample"] != "S03"]                   # S03's PC is missing
    assert d["Sample"].tolist() == keep["Sample"].tolist()
    assert d["Phenotype_uploaded"].tolist() == keep["Phenotype_used"].tolist()
    # they are the residuals on an intercept plus the two PCs, over those samples
    X = np.column_stack([np.ones(len(keep)),
                         pd.DataFrame(_PCS, index=_PC_IDS).loc[keep["Sample"]].to_numpy()])
    y = keep["Phenotype_used"].to_numpy(dtype=float)
    resid = d["Phenotype_plotted"].to_numpy(dtype=float)
    assert np.allclose(resid, y - X @ np.linalg.lstsq(X, y, rcond=None)[0], rtol=0, atol=1e-10)
    assert np.allclose(X.T @ resid, 0.0, atol=1e-8)
    assert not np.allclose(resid, y)
    assert v.attrs["n_pcs"] == 2 and set(d["Phenotype_scale"]) == {"PC-adjusted (2 PCs)"}
    assert hap_axis_label("t", 2) == "t (PC-adjusted)"
    # and both plots draw exactly those values
    fig = plotly_hap_boxplot(d, ["x"], y_label=hap_axis_label("t", v.attrs["n_pcs"]))
    got = {cd[0]: float(yv) for tr in fig.data if tr.type == "scatter"
           for yv, cd in zip(tr.y, tr.customdata)}
    assert got == dict(zip(d["Sample"], d["Phenotype_plotted"].astype(float)))
    assert fig.layout.yaxis.title.text == "t (PC-adjusted)"
    sfig = render_hap_boxplot(d, ["x"])
    try:
        ys = sorted(float(y) for ln in sfig.axes[0].lines if ln.get_marker() == "o"
                    for y in ln.get_ydata())
    finally:
        plt.close(sfig)
    assert ys == sorted(d["Phenotype_plotted"].astype(float))


def test_without_pcs_the_plotted_values_are_the_uploaded_ones():
    for pcs in (None, np.empty((len(_PC_IDS), 0))):
        v, _cld, d = _plot_data(pcs=pcs)
        assert v.attrs["n_pcs"] == 0
        assert d["Sample"].tolist() == TESTED["Sample"].tolist()
        assert d["Phenotype_plotted"].tolist() == TESTED["Phenotype_used"].tolist()
        assert set(d["Phenotype_scale"]) == {"uploaded"}
    assert hap_axis_label("t", 0) == "t"


def test_the_letters_on_the_plot_are_the_cld_tables():
    for pcs in (None, _PCS):
        _v, cld, d = _plot_data(pcs=pcs)
        table = dict(zip(cld["MLG_label"], cld["CLD"]))
        assert set(table) == {"H1", "H2", "H3"} and len(set(table.values())) >= 2
        assert d["CLD_letters"].tolist() == [table[h] for h in d["Haplotype"]]
        order = ["H1", "H2", "H3"]                              # the plots, from the left
        fig = render_hap_boxplot(d, ["x"])
        try:
            texts = {int(round(t.get_position()[0])): t.get_text() for t in fig.axes[0].texts}
            top = max(ln.get_ydata().max() for ln in fig.axes[0].lines if ln.get_marker() == "o")
            assert all(t.get_position()[1] > top for t in fig.axes[0].texts)   # above the dots
        finally:
            plt.close(fig)
        assert texts == {i: table[h] for i, h in enumerate(order)}
        pfig = plotly_hap_boxplot(d, ["x"])
        ann = {int(a.x): a.text.replace("<b>", "").replace("</b>", "")
               for a in pfig.layout.annotations}
        assert ann == {i: table[h] for i, h in enumerate(order)}
    # no letters (Tukey HSD failed): none drawn
    d0 = hap_plot_data(hap_plot_values(ROWS))
    fig = render_hap_boxplot(d0, ["x"])
    try:
        assert not fig.axes[0].texts
    finally:
        plt.close(fig)
    assert not plotly_hap_boxplot(d0, ["x"]).layout.annotations
