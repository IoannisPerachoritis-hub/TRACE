"""The LD block's haplotype box plot, drawn as the lead-SNP plot (gwas/snpplots.py).

hap_boxplot_data takes a block's in-test rows of the haplotype_assignments table;
plotly_hap_boxplot and render_hap_boxplot draw exactly those samples, labels and values
(the uploaded values the haplotype test compared), coloured by haplotype_color_map.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

from gwas.snpplots import (  # noqa: E402
    HAP_BOXPLOT_DATA_COLUMNS,
    hap_boxplot_data,
    hap_boxplot_title,
    plotly_hap_boxplot,
    render_hap_boxplot,
)
from utils.pub_theme import haplotype_color_map  # noqa: E402

RNG = np.random.default_rng(2)
_N = 40
ROWS = pd.DataFrame({      # a block's assignments: H1-H3 tested, Other and H4 left out
    "Chr": "2", "Block_start": 100, "Block_end": 300,
    "Sample": [f"S{i:02d}" for i in range(_N)],
    "Haplotype": (["H2"] * 10 + ["H1"] * 14 + ["H3"] * 8 + ["Other"] * 5 + ["H4"] * 3),
    "Phenotype_used": RNG.normal(10, 2, _N).round(3),
})
ROWS["in_haplotype_test"] = ROWS["Haplotype"].isin(["H1", "H2", "H3"])
TESTED = ROWS[ROWS["in_haplotype_test"]]


def test_the_data_are_the_blocks_in_test_rows():
    d = hap_boxplot_data(ROWS)
    assert list(d.columns) == HAP_BOXPLOT_DATA_COLUMNS
    assert sorted(d["Sample"]) == sorted(TESTED["Sample"])
    by_sample = TESTED.set_index("Sample")
    assert d["Haplotype"].tolist() == by_sample.loc[d["Sample"], "Haplotype"].tolist()
    assert d["Phenotype"].tolist() == by_sample.loc[d["Sample"], "Phenotype_used"].tolist()
    assert list(dict.fromkeys(d["Haplotype"])) == ["H1", "H2", "H3"]      # H1, H2, ... order
    assert bool(d["In_haplotype_test"].all())


def test_the_interactive_plot_shows_those_points():
    d = hap_boxplot_data(ROWS)
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
    assert list(fig.layout.xaxis.ticktext) == ["H1<br>(n=14)", "H2<br>(n=10)", "H3<br>(n=8)"]
    assert fig.layout.title.text == "Chr2:100-300<br>P (permutation) = 0.0123 · η² = 0.21"
    assert fig.layout.yaxis.title.text == "t"


def test_the_static_plot_shows_the_same_points_at_the_same_places():
    d = hap_boxplot_data(ROWS)
    title = hap_boxplot_title("2", 100, 300, 0.0123, 0.214)
    fig = render_hap_boxplot(d, title, y_label="t")
    try:
        ax = fig.axes[0]
        dots = [(float(x), float(y), to_hex(ln.get_color())) for ln in ax.lines
                if ln.get_marker() == "o" for x, y in zip(ln.get_xdata(), ln.get_ydata())]
        assert [t.get_text() for t in ax.get_xticklabels()] == ["H1\n(n=14)", "H2\n(n=10)",
                                                                "H3\n(n=8)"]
        assert ax.get_title() == "\n".join(title) and ax.get_ylabel() == "t"
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
