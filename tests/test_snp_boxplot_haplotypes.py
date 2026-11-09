"""Lead-SNP box plot: dots coloured by haplotype (reviewer point 2-INT.9).

gwas.snpplots.render_snp_boxplot(..., hap_labels=, hap_color_map=, hap_note=, y_label=)
colours each dot by its sample's haplotype with utils.pub_theme.haplotype_color_map
(the block-level plots' colours: H<n> by its number, Other grey) and adds a legend of the
labels plotted, outside the axes. The jitter is drawn as without labels, so no dot
moves (data coordinates compared with the single-colour render). Without labels it
keeps the single colour and can write a note under the plot. gwas.haplotype
.snp_haplotype_labels finds the LD block that contains a SNP (by position) and its
labels in a haplotype_assignments table.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

from gwas.haplotype import snp_haplotype_labels  # noqa: E402
from gwas.snpplots import (  # noqa: E402
    HAP_NOTE_NO_BLOCK,
    HAP_NOTE_NO_GROUPS,
    phenotype_axis_label,
    render_snp_boxplot,
)
from utils.pub_theme import HAPLOTYPE_OTHER_COLOR, PALETTE_CYCLE, haplotype_color_map  # noqa: E402

# 18 samples: genotype class, phenotype, label; one with no genotype, one with no phenotype
G = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, np.nan, 2], dtype=float)
Y = np.linspace(1.0, 4.4, 18)
Y[-1] = np.nan
LABELS = np.array(["H1", "H1", "H2", "Other", "H1", "H3", "H2", "H2", "H7", "Other", "H1",
                   "H3", "H3", "H7", "H1", "H2", "H4", "H6"], dtype=object)


def _dots(fig):
    """Every plotted dot as (x, y, colour), in drawing order."""
    out = []
    for line in fig.axes[0].lines:
        if line.get_marker() != "o":
            continue
        c = to_hex(line.get_color())
        out += [(float(x), float(y), c) for x, y in zip(line.get_xdata(), line.get_ydata())]
    return out


def _boxes(fig):
    return [np.asarray(line.get_xydata()).tolist() for line in fig.axes[0].lines
            if line.get_marker() != "o"]


@pytest.fixture
def figs():
    plain = render_snp_boxplot("snp1", G, Y, beta_mlm=0.5, se_mlm=0.1, beta_ols=0.6)
    labelled = render_snp_boxplot("snp1", G, Y, beta_mlm=0.5, se_mlm=0.1, beta_ols=0.6,
                                  hap_labels=LABELS)
    yield plain, labelled
    plt.close(plain)
    plt.close(labelled)


def test_each_dot_has_its_samples_haplotype_colour(figs):
    plain, labelled = figs
    cmap = haplotype_color_map(LABELS)
    # the plain render draws the samples of each class in input order: recover who is where
    valid = np.isfinite(G) & np.isfinite(Y)
    order = [i for cls in (0, 1, 2) for i in np.where(valid & (np.rint(G) == cls))[0]]
    where = {(x, y): order[k] for k, (x, y, _c) in enumerate(_dots(plain))}
    got = _dots(labelled)
    assert len(got) == len(order) == 16
    for x, y, c in got:
        assert c == to_hex(cmap[LABELS[where[(x, y)]]]), (x, y)


def test_legend_lists_exactly_the_labels_plotted_outside_the_axes(figs):
    _plain, labelled = figs
    ax = labelled.axes[0]
    leg = ax.get_legend()
    # H4 sits only on the sample with no genotype, H6 only on the one with no phenotype
    assert [t.get_text() for t in leg.get_texts()] == ["H1", "H2", "H3", "H7", "Other"]
    cmap = haplotype_color_map(LABELS)
    assert [to_hex(h.get_color()) for h in leg.legend_handles] == [
        to_hex(cmap[h]) for h in ["H1", "H2", "H3", "H7", "Other"]]
    labelled.canvas.draw()
    r = labelled.canvas.get_renderer()
    assert leg.get_window_extent(r).x0 >= ax.get_window_extent(r).x1


def test_geometry_is_unchanged(figs):
    plain, labelled = figs
    assert sorted((x, y) for x, y, _c in _dots(plain)) == sorted((x, y) for x, y, _c in _dots(labelled))
    assert _boxes(plain) == _boxes(labelled)
    p, q = plain.axes[0], labelled.axes[0]
    assert p.get_xlim() == q.get_xlim() and p.get_ylim() == q.get_ylim()
    assert [t.get_text() for t in p.get_xticklabels()] == [t.get_text() for t in q.get_xticklabels()]
    assert p.get_title() == q.get_title()


def test_colours_follow_the_label_number_and_match_the_block_plots():
    cmap = haplotype_color_map(["H1", "H3", "H7"])
    assert cmap["H1"] == PALETTE_CYCLE[0] and cmap["H3"] == PALETTE_CYCLE[2]
    assert cmap["H7"] == PALETTE_CYCLE[0] and cmap["Other"] == HAPLOTYPE_OTHER_COLOR == "#bdbdbd"
    for k in range(1, 16):
        labels = [f"H{i}" for i in range(1, k + 1)]
        # the block-level plots colour their (contiguous) labels by rank
        legacy = {h: PALETTE_CYCLE[i % len(PALETTE_CYCLE)] for i, h in enumerate(labels)}
        legacy["Other"] = "#bdbdbd"
        assert haplotype_color_map(labels + ["Other"]) == legacy
        assert haplotype_color_map(labels) == legacy


def test_no_labels_keeps_the_single_colour_and_writes_the_note():
    fig = render_snp_boxplot("snp1", G, Y, hap_note=HAP_NOTE_NO_BLOCK)
    try:
        assert {c for _x, _y, c in _dots(fig)} == {"#0072b2"}
        assert fig.axes[0].get_legend() is None
        assert [t.get_text() for t in fig.texts] == [HAP_NOTE_NO_BLOCK]
    finally:
        plt.close(fig)
    plain = render_snp_boxplot("snp1", G, Y)
    try:     # the call without the new arguments is the old plot
        assert plain.get_size_inches().tolist() == [3.2, 3.0] and not plain.texts
        assert plain.axes[0].get_ylabel() == "Phenotype" and plain.axes[0].get_legend() is None
    finally:
        plt.close(plain)


def test_axis_label_names_the_values_plotted():
    assert phenotype_axis_label("locule_number") == "locule_number"
    assert phenotype_axis_label("locule_number", "none") == "locule_number"
    assert phenotype_axis_label("locule_number", "int") == "locule_number (rank-INT)"
    assert phenotype_axis_label("w", "log") == "w (log10)"
    fig = render_snp_boxplot("snp1", G, Y, y_label="locule_number (rank-INT)")
    try:
        assert fig.axes[0].get_ylabel() == "locule_number (rank-INT)"
    finally:
        plt.close(fig)


def test_misaligned_labels_are_refused():
    with pytest.raises(ValueError, match="labels"):
        render_snp_boxplot("snp1", G, Y, hap_labels=LABELS[:5])


def test_snp_haplotype_labels_finds_the_block_by_position():
    blocks = pd.DataFrame({"Chr": ["2", "2"], "Start (bp)": [100, 500], "End (bp)": [300, 900],
                           "lead_snp": ["a", "b"]})
    assign = pd.DataFrame({
        "Chr": ["2"] * 4, "Block_start": [100] * 3 + [500], "Block_end": [300] * 3 + [900],
        "Sample": ["s1", "s2", "s3", "s1"], "Haplotype": ["H1", "", "H2", "Other"],
        "in_haplotype_test": [True, False, True, False]})
    labels, key, why = snp_haplotype_labels(blocks, assign, "chr2", 300, ["s3", "s1", "s2", "s9"])
    assert why is None and key == "2:100-300"
    assert labels.tolist() == ["H2", "H1", "Other", "Other"]     # no haplotype -> Other
    assert snp_haplotype_labels(blocks, assign, "2", 400, ["s1"])[2] == "no_block"
    assert snp_haplotype_labels(blocks, assign, "3", 200, ["s1"])[2] == "no_block"
    # a block whose rows hold no H label (only Other) has no haplotype groups
    assert snp_haplotype_labels(blocks, assign, "2", 600, ["s1"])[2] == "no_groups"
    assert snp_haplotype_labels(blocks, None, "2", 200, ["s1"])[1:] == ("2:100-300", "no_groups")
    assert snp_haplotype_labels(None, assign, "2", 200, ["s1"])[2] == "no_block"
    assert HAP_NOTE_NO_GROUPS.startswith("LD block has no haplotype groups")
