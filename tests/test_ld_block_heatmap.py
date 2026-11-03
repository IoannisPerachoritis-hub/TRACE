"""LD-block heatmaps show neighbouring SNPs (reviewer point 2-INT.3).

gwas.plotting.plot_ld_block_heatmap is the one drawing behind every LD-block
heatmap (CLI ZIP, One-Click ZIP, Post-GWAS Local LD "Detected block"). These pin
its window rule (acceptance A), the fixed PNG size (C), and the r2 it draws.
"""
import io
import zipfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from gwas.ld import pairwise_r2  # noqa: E402
from gwas.plotting import (  # noqa: E402
    LD_HEATMAP_BLOCK_MAX_FRAC,
    LD_HEATMAP_MAX_DISPLAY,
    LD_HEATMAP_MAX_LABELS,
    LD_HEATMAP_MIN_WINDOW,
    _build_gwas_results_zip,
    _even_pick,
    lead_snp_position,
    plot_ld_block_heatmap,
    select_ld_heatmap_window,
)


def _layout(counts, spacing=1000):
    """Per-SNP chroms/positions: ``counts`` maps chromosome -> SNP count, SNPs at
    spacing, 2*spacing, ... on each chromosome, chromosomes concatenated."""
    chroms, pos = [], []
    for c, n in counts.items():
        chroms += [c] * n
        pos += [spacing * (i + 1) for i in range(n)]
    return np.array(chroms), np.array(pos, dtype=np.int64)


def _geno(n_snps, n_samples=60, seed=0):
    rng = np.random.default_rng(seed)
    return rng.binomial(2, 0.4, size=(n_samples, n_snps)).astype(float)


def _block(pos, offset, first, last):
    """Block coordinates covering SNPs first..last (0-based, per chromosome)."""
    return int(pos[offset + first]), int(pos[offset + last])


def test_constants_are_the_agreed_defaults():
    assert (LD_HEATMAP_BLOCK_MAX_FRAC, LD_HEATMAP_MIN_WINDOW, LD_HEATMAP_MAX_DISPLAY,
            LD_HEATMAP_MAX_LABELS) == (0.5, 20, 200, 10)


# ---- Acceptance A ---------------------------------------------------------------

def test_ten_snp_block_mid_chromosome_shows_five_neighbours_each_side():
    chroms, pos = _layout({"1": 30, "2": 100, "3": 30})
    s, e = _block(pos, 30, 40, 49)
    info = select_ld_heatmap_window(chroms, pos, "2", s, e)
    assert info["n_shown"] == 20 and info["n_block"] == 10
    assert (info["n_left"], info["n_right"]) == (5, 5)
    np.testing.assert_array_equal(info["indices"], np.arange(30 + 35, 30 + 55))
    np.testing.assert_array_equal(info["in_block"], [False] * 5 + [True] * 10 + [False] * 5)
    assert not info["thinned"] and not info["exhausted"]
    assert info["block_share"] == 0.5


def test_block_at_chromosome_start_takes_all_ten_flank_snps_on_the_right():
    chroms, pos = _layout({"1": 30, "2": 100})
    s, e = _block(pos, 30, 0, 9)
    info = select_ld_heatmap_window(chroms, pos, "2", s, e)
    assert info["n_shown"] == 20
    assert (info["n_left"], info["n_right"]) == (0, 10)
    # never borrows from the previous chromosome
    assert np.all(chroms[info["indices"]] == "2")
    np.testing.assert_array_equal(info["indices"], np.arange(30, 50))


def test_block_at_chromosome_end_takes_all_flank_snps_on_the_left():
    chroms, pos = _layout({"1": 100, "2": 30})
    s, e = _block(pos, 0, 90, 99)
    info = select_ld_heatmap_window(chroms, pos, "1", s, e)
    assert (info["n_left"], info["n_right"]) == (10, 0)
    assert np.all(chroms[info["indices"]] == "1")


def test_two_snp_block_shows_twenty():
    chroms, pos = _layout({"2": 100})
    s, e = _block(pos, 0, 50, 51)
    info = select_ld_heatmap_window(chroms, pos, "2", s, e)
    assert info["n_shown"] == 20 and info["n_block"] == 2
    assert (info["n_left"], info["n_right"]) == (9, 9)


def test_odd_flank_budget_puts_the_extra_snp_on_the_right():
    chroms, pos = _layout({"2": 100})
    s, e = _block(pos, 0, 40, 50)                          # 11 SNPs -> N = 22, F = 11
    info = select_ld_heatmap_window(chroms, pos, "2", s, e)
    assert (info["n_target"], info["n_left"], info["n_right"]) == (22, 5, 6)


def test_partly_short_side_gives_its_shortfall_to_the_other_side():
    chroms, pos = _layout({"2": 100})
    s, e = _block(pos, 0, 3, 12)                           # only 3 SNPs to the left
    info = select_ld_heatmap_window(chroms, pos, "2", s, e)
    assert (info["n_left"], info["n_right"], info["n_shown"]) == (3, 7, 20)


def _off_grid(n, k, start=1):
    """First index >= start that plain even thinning of n down to k would drop."""
    grid = set(_even_pick(n, k).tolist())
    return next(i for i in range(start, n - 1) if i not in grid)


def test_150_snp_block_is_thinned_to_200_with_edges_lead_and_nearest_flanks_kept():
    chroms, pos = _layout({"2": 600})
    first, last = 200, 349
    lead = first + _off_grid(150, 100)                     # dropped unless protected
    s, e = _block(pos, 0, first, last)
    info = select_ld_heatmap_window(chroms, pos, "2", s, e, lead_pos=int(pos[lead]))
    assert info["thinned"] and info["n_window"] == 300 and info["n_shown"] == 200
    assert info["block_share"] <= 0.5
    shown = set(info["indices"].tolist())
    assert {first, last, lead} <= shown                    # block edges and the lead
    assert {first - 1, last + 1} <= shown                  # nearest flank SNP each side
    assert {first - 75, last + 75} <= shown                # window extent kept
    assert info["lead_index"] == lead
    np.testing.assert_array_equal(info["indices"], np.sort(info["indices"]))
    # one thinning rate: block and flanks are each kept at about 2/3
    assert info["n_block_shown"] == 100


def test_lead_in_a_flank_is_kept_by_thinning():
    chroms, pos = _layout({"2": 600})
    s, e = _block(pos, 0, 200, 349)
    lead = 125 + _off_grid(75, 50)                         # left flank 125..199, off-grid
    info = select_ld_heatmap_window(chroms, pos, "2", s, e, lead_pos=int(pos[lead]))
    assert info["n_shown"] == 200 and lead in set(info["indices"].tolist())
    assert info["block_share"] <= 0.5 and {199, 350} <= set(info["indices"].tolist())


def test_thinning_remainder_goes_to_the_flanks_so_the_block_stays_at_half():
    chroms, pos = _layout({"2": 600})
    s, e = _block(pos, 0, 200, 350)                        # 151 SNPs -> window 302
    info = select_ld_heatmap_window(chroms, pos, "2", s, e)
    assert info["n_window"] == 302 and info["n_shown"] == 200
    assert info["block_share"] <= 0.5


def test_thinned_whole_chromosome_title_does_not_claim_every_snp_is_shown():
    chroms, pos = _layout({"2": 250})
    s, e = _block(pos, 0, 50, 199)                         # 150 SNPs, target 300 > 250
    fig, info = plot_ld_block_heatmap(_geno(250), chroms, pos, "2", s, e)
    plt.close(fig)
    assert info["exhausted"] and info["thinned"] and info["n_shown"] == 200
    assert "all of Chr2 (250 SNPs)" in info["title"]
    assert "thinned to 200 of 250" in info["title"]
    assert "block share" in info["title"] and "SNPs on Chr2 shown" not in info["title"]


def test_chromosome_with_fewer_snps_than_the_target_shows_all_and_states_the_share():
    chroms, pos = _layout({"1": 50, "3": 14})
    s, e = _block(pos, 50, 3, 12)                          # 10-SNP block, 4 others
    fig, info = plot_ld_block_heatmap(_geno(64), chroms, pos, "3", s, e)
    plt.close(fig)
    assert info["exhausted"] and info["n_shown"] == 14
    np.testing.assert_array_equal(info["indices"], np.arange(50, 64))
    assert "10 of 14 SNPs shown" in info["title"]
    assert "71%" in info["title"]


# ---- Drawing, r2, labels ----------------------------------------------------------

def test_r2_is_pairwise_r2_on_the_shown_columns():
    chroms, pos = _layout({"1": 40, "2": 80})
    G = _geno(120, seed=3)
    s, e = _block(pos, 40, 30, 39)
    fig, info = plot_ld_block_heatmap(G, chroms, pos, "2", s, e)
    plt.close(fig)
    np.testing.assert_array_equal(info["r2"], pairwise_r2(G[:, info["indices"]], min_pair_n=20))


def test_input_order_and_chromosome_spelling_do_not_change_the_window():
    chroms, pos = _layout({"1": 40, "2": 80})
    s, e = _block(pos, 40, 30, 39)
    ref = select_ld_heatmap_window(chroms, pos, "2", s, e)
    perm = np.random.default_rng(1).permutation(chroms.size)
    spelled = np.array(["chr0" + c for c in chroms[perm]])
    got = select_ld_heatmap_window(spelled, pos[perm], "Chr2", s, e)
    np.testing.assert_array_equal(perm[got["indices"]], ref["indices"])


def test_lead_snp_position_looks_up_the_lead_on_the_block_chromosome():
    chroms, pos = _layout({"1": 3, "2": 3})
    sid = np.array(["a", "b", "c", "d", "e", "f"])
    assert lead_snp_position("e", sid, chroms, pos, "2") == 2000
    assert lead_snp_position("b", sid, chroms, pos, "2") is None    # wrong chromosome
    assert lead_snp_position("", sid, chroms, pos, "2") is None
    assert lead_snp_position("zz", sid, chroms, pos, "2") is None


def test_title_ticks_and_lead_marking():
    chroms, pos = _layout({"2": 600})
    s, e = _block(pos, 0, 200, 349)
    fig, info = plot_ld_block_heatmap(_geno(600), chroms, pos, "2", s, e, lead_pos=int(pos[277]))
    ax = fig.axes[0]
    assert info["title"].startswith(f"LD heatmap: Chr2 block {s:,}-{e:,} (block = 100 of 200 SNPs shown)")
    assert "thinned to 200 of 300" in info["title"]
    assert f"window {int(pos[125]):,}-{int(pos[424]):,}" in info["title"]
    assert len(ax.get_xticks()) <= LD_HEATMAP_MAX_LABELS
    lead_col = int(np.flatnonzero(info["indices"] == 277)[0])
    assert lead_col in set(int(t) for t in ax.get_xticks())
    red = [t.get_text() for t in ax.get_xticklabels() if t.get_color() == "#C00000"]
    assert red == [f"{pos[277] / 1e6:.3f}"]
    plt.close(fig)


def test_snp_id_labels_on_request():
    chroms, pos = _layout({"2": 40})
    sid = np.array([f"snp{i}" for i in range(40)])
    s, e = _block(pos, 0, 15, 24)
    fig, info = plot_ld_block_heatmap(_geno(40), chroms, pos, "2", s, e,
                                      label_mode="snp_id", sid=sid)
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    plt.close(fig)
    assert labels[0] == "snp10" and labels[-1] == "snp29"


def test_dense_windows_get_distinct_mb_labels():
    chroms, pos = _layout({"2": 80}, spacing=100)          # SNPs 100 bp apart
    fig, _ = plot_ld_block_heatmap(_geno(80), chroms, pos, "2", *_block(pos, 0, 30, 39))
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    plt.close(fig)
    assert len(set(labels)) == len(labels)


def test_outline_encloses_exactly_the_within_block_pairs():
    from matplotlib.patches import Polygon
    from matplotlib.path import Path as MplPath
    chroms, pos = _layout({"2": 100})
    fig, info = plot_ld_block_heatmap(_geno(100), chroms, pos, "2", *_block(pos, 0, 40, 49))
    poly = [p for p in fig.axes[0].patches if isinstance(p, Polygon)][0]
    path = MplPath(poly.get_xy())
    blk = np.flatnonzero(info["in_block"])
    for i in range(info["n_shown"]):                     # cell centres (x = column j, y = row i)
        for j in range(i):
            inside = bool(path.contains_point((j, i)))
            assert inside == (blk[0] <= j and i <= blk[-1]), (i, j)
    plt.close(fig)


def test_all_text_stays_inside_the_canvas():
    chroms, pos = _layout({"2": 600})
    sid = np.array([f"SL4.0ch02_{p}_A_G_very_long_identifier" for p in pos])
    cases = [dict(), dict(label_mode="snp_id", sid=sid)]
    for kw in cases:
        fig, _ = plot_ld_block_heatmap(_geno(600), chroms, pos, "2", *_block(pos, 0, 200, 349),
                                       lead_pos=int(pos[277]), **kw)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        canvas = fig.bbox
        for t in fig.findobj(matplotlib.text.Text):
            if t.get_visible() and t.get_text():
                bb = t.get_window_extent(renderer)
                assert (bb.x0 >= canvas.x0 - 1 and bb.x1 <= canvas.x1 + 1
                        and bb.y0 >= canvas.y0 - 1 and bb.y1 <= canvas.y1 + 1), (kw, t.get_text())
        plt.close(fig)


def test_block_with_fewer_than_two_snps_raises():
    chroms, pos = _layout({"2": 40})
    with pytest.raises(ValueError):
        plot_ld_block_heatmap(_geno(40), chroms, pos, "2", 1_000_000, 2_000_000)


# ---- Acceptance C: bounded, constant PNG size -------------------------------------

def _zip_png_shape(fig, name):
    _zip_name, buf = _build_gwas_results_zip("T", pd.DataFrame({"SNP": ["x"], "PValue": [0.5]}),
                                             figures_dict={name: fig})
    data = buf.getvalue() if hasattr(buf, "getvalue") else buf
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return mpimg.imread(io.BytesIO(zf.read(f"figures/{name}")), format="png").shape[:2]


def test_png_pixel_size_is_the_same_for_small_and_large_windows():
    chroms, pos = _layout({"2": 600})
    G = _geno(600)
    small, _ = plot_ld_block_heatmap(G, chroms, pos, "2", *_block(pos, 0, 50, 51))
    large, _ = plot_ld_block_heatmap(G, chroms, pos, "2", *_block(pos, 0, 200, 349),
                                     lead_pos=int(pos[277]))
    sid = np.array([f"SL4.0ch02_{p}_A_G" for p in pos])
    ids, _ = plot_ld_block_heatmap(G, chroms, pos, "2", *_block(pos, 0, 300, 309),
                                   label_mode="snp_id", sid=sid)
    shapes = {_zip_png_shape(small, "LD_heatmap_Chr2_a.png"),
              _zip_png_shape(large, "LD_heatmap_Chr2_b.png"),
              _zip_png_shape(ids, "LD_heatmap_Chr2_c.png")}
    assert shapes == {(900, 1050)}           # FIGSIZE["heatmap"] (7 x 6 in) at 150 dpi
