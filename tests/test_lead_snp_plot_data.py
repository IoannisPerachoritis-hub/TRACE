"""The lead-SNP box plot's data and interactive view (Post-GWAS "Lead SNP genotype effect").

gwas.snpplots.snp_boxplot_data gives one row per sample the box plot draws (the CSV under
the plot); plotly_snp_boxplot draws those rows as an interactive figure with the static
plot's jitter, colours and title. gwas.haplotype.snp_block_assignments gives the
haplotype labels and test flags of the SNP's tested block (the CC-13 table's rows).
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

from gwas.haplotype import snp_block_assignments  # noqa: E402
from gwas.snpplots import (  # noqa: E402
    SNP_BOXPLOT_DATA_COLUMNS,
    plotly_snp_boxplot,
    render_snp_boxplot,
    snp_boxplot_data,
    snp_boxplot_title,
)
from utils.pub_theme import haplotype_color_map  # noqa: E402

# 12 samples: one without a genotype, one without a phenotype, one with dosage 3 (not drawn)
IDS = np.array([f"S{i:02d}" for i in range(12)])
G = np.array([0, 0, 1, 1, 1, 2, 2, 0, np.nan, 1, 2, 3], dtype=float)
Y = np.array([1.0, 1.2, 2.0, 2.2, 2.4, 3.1, 3.3, 0.9, 5.0, np.nan, 3.0, 9.0])
DRAWN = [0, 1, 2, 3, 4, 5, 6, 7, 10]


def _assignments():
    """A CC-13 table for block 2:100-300 (S10 has no haplotype, S05 is not in the test)."""
    hap = {"S00": "H1", "S01": "H2", "S02": "H1", "S03": "H1", "S04": "Other", "S05": "H2",
           "S06": "H2", "S07": "H1", "S08": "H1", "S09": "H2", "S10": "", "S11": "H1"}
    return pd.DataFrame({
        "Chr": "2", "Block_start": 100, "Block_end": 300, "Sample": list(hap),
        "Haplotype": list(hap.values()),
        "in_haplotype_test": [h not in ("", "Other") and s != "S05" for s, h in hap.items()]})


_BLOCKS = pd.DataFrame({"Chr": ["2"], "Start (bp)": [100], "End (bp)": [300], "lead_snp": ["x"]})


def test_rows_are_the_samples_the_static_plot_draws():
    d = snp_boxplot_data(G, Y, IDS, alleles=("A", "G"), norm="int")
    assert list(d.columns) == SNP_BOXPLOT_DATA_COLUMNS
    assert d["Sample"].tolist() == IDS[DRAWN].tolist()
    assert d["Genotype_dosage"].tolist() == [0, 0, 1, 1, 1, 2, 2, 0, 2]
    assert d["Genotype_alleles"].tolist() == ["A/A", "A/A", "A/G", "A/G", "A/G", "G/G", "G/G",
                                              "A/A", "G/G"]
    np.testing.assert_array_equal(d["Phenotype"].to_numpy(), Y[DRAWN])   # the values plotted
    assert set(d["Phenotype_scale"]) == {"rank-INT"}
    assert set(snp_boxplot_data(G, Y, IDS)["Phenotype_scale"]) == {"raw"}
    # the static plot draws exactly these samples (one dot each)
    fig = render_snp_boxplot("snp", G, Y)
    try:
        dots = sorted(float(v) for ln in fig.axes[0].lines if ln.get_marker() == "o"
                      for v in ln.get_ydata())
    finally:
        plt.close(fig)
    assert dots == sorted(Y[DRAWN].tolist())
    with pytest.raises(ValueError):
        snp_boxplot_data(G, Y, IDS[:5])


def test_haplotypes_are_the_blocks_assignments():
    rows, key, why = snp_block_assignments(_BLOCKS, _assignments(), "chr2", 250, IDS)
    assert why is None and key == "2:100-300"
    a = _assignments().set_index("Sample")
    assert rows["Haplotype"].tolist() == a.loc[IDS, "Haplotype"].tolist()
    assert rows["in_haplotype_test"].tolist() == a.loc[IDS, "in_haplotype_test"].tolist()
    d = snp_boxplot_data(G, Y, IDS, haplotypes=rows["Haplotype"].to_numpy(),
                         in_test=rows["in_haplotype_test"].to_numpy())
    assert d["Haplotype"].tolist() == a.loc[IDS[DRAWN], "Haplotype"].tolist()   # S10 blank
    assert d["In_haplotype_test"].tolist() == a.loc[IDS[DRAWN], "in_haplotype_test"].tolist()
    assert snp_block_assignments(_BLOCKS, _assignments(), "2", 400, IDS)[2] == "no_block"
    assert snp_block_assignments(_BLOCKS, None, "2", 200, IDS)[2] == "no_groups"


def test_the_interactive_plot_shows_the_static_plots_points_colours_and_title():
    rows, _key, _why = snp_block_assignments(_BLOCKS, _assignments(), "2", 200, IDS)
    d = snp_boxplot_data(G, Y, IDS, haplotypes=rows["Haplotype"].to_numpy(),
                         in_test=rows["in_haplotype_test"].to_numpy(), alleles=("A", "G"))
    title = snp_boxplot_title("snp", G, Y, beta_mlm=0.5, se_mlm=0.1, beta_ols=0.6)
    fig = plotly_snp_boxplot(d, title, y_label="t (rank-INT)")
    pts = [tr for tr in fig.data if tr.type == "scatter"]
    # every drawn sample once, with its own value, ID and haplotype in the hover data
    got = {cd[0]: (float(yv), cd[2]) for tr in pts for yv, cd in zip(tr.y, tr.customdata)}
    want_hap = ["Other" if h == "" else h for h in d["Haplotype"]]
    assert got == {s: (float(v), h if h else "none") for s, v, h in
                   zip(d["Sample"], d["Phenotype"], d["Haplotype"])}
    cmap = haplotype_color_map(set(want_hap))
    for tr in pts:
        assert to_hex(tr.marker.color) == to_hex(cmap[tr.name])
    # the static plot's jitter: the same x for the same point
    static = render_snp_boxplot("snp", G, Y, beta_mlm=0.5, se_mlm=0.1, beta_ols=0.6,
                                hap_labels=np.asarray(["Other" if h == "" else h
                                                       for h in rows["Haplotype"]], object))
    try:
        sx = sorted((round(float(x), 12), float(yv)) for ln in static.axes[0].lines
                    if ln.get_marker() == "o" for x, yv in zip(ln.get_xdata(), ln.get_ydata()))
        assert static.axes[0].get_title() == "\n".join(title)
    finally:
        plt.close(static)
    px = sorted((round(float(x), 12), float(yv)) for tr in pts for x, yv in zip(tr.x, tr.y))
    assert px == sx
    assert fig.layout.title.text == "<br>".join(title)
    assert fig.layout.yaxis.title.text == "t (rank-INT)"
    assert list(fig.layout.xaxis.ticktext) == ["0 (A/A)<br>(n=3)", "1 (A/G)<br>(n=3)",
                                              "2 (G/G)<br>(n=3)"]


def test_without_haplotypes_one_colour_and_no_legend():
    d = snp_boxplot_data(G, Y, IDS)
    fig = plotly_snp_boxplot(d, snp_boxplot_title("snp", G, Y))
    pts = [tr for tr in fig.data if tr.type == "scatter"]
    assert len(pts) == 1 and to_hex(pts[0].marker.color) == "#0072b2"
    assert pts[0].showlegend is False
    assert list(fig.layout.xaxis.ticktext) == ["0<br>(n=3)", "1<br>(n=3)", "2<br>(n=3)"]
