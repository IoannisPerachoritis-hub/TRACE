"""T-21/T-84–T-86 — per-SNP boxplot rendering + plot-budget selection."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from gwas.snpplots import (
    SNP_PLOT_MODES,
    build_snp_view_index,
    render_snp_boxplot,
    select_snps_for_plotting,
)

_SIG = pd.DataFrame({"SNP": [f"s{i}" for i in range(5)], "Chr": ["1"] * 5,
                     "Pos": [10, 20, 30, 40, 50],
                     "PValue": [1e-9, 1e-3, 1e-8, 1e-5, 1e-7]})


# ---- T-84/T-86 budget ----
def test_select_modes_and_budget():
    assert select_snps_for_plotting(_SIG, mode="none") == []
    top2 = select_snps_for_plotting(_SIG, mode="capped", max_plots=2)
    assert top2 == ["s0", "s2"]                      # ranked by PValue asc
    allids = select_snps_for_plotting(_SIG, mode="all")
    assert set(allids) == set(_SIG["SNP"]) and len(allids) == 5


def test_select_rejects_bad_mode():
    with pytest.raises(ValueError):
        select_snps_for_plotting(_SIG, mode="everything")
    assert set(SNP_PLOT_MODES) == {"none", "capped", "all"}


def test_select_empty_table_is_empty():
    assert select_snps_for_plotting(pd.DataFrame(columns=["SNP", "PValue"]), mode="all") == []


# ---- T-21 rendering ----
def test_render_boxplot_returns_figure_with_effect():
    g = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, np.nan])
    y = np.arange(10.0)
    fig = render_snp_boxplot("s0", g, y, beta_mlm=0.4, se_mlm=0.1, beta_ols=0.6)
    assert fig is not None and len(fig.axes) == 1
    title = fig.axes[0].get_title()
    assert "s0" in title and "concordant" in title and "β_MLM" in title
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_render_boxplot_beta_absent_is_structure_unadjusted():
    g = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2], dtype=float)
    y = np.arange(15.0)
    fig = render_snp_boxplot("s1", g, y, beta_mlm=None, beta_ols=0.6)
    title = fig.axes[0].get_title()
    assert "structure-unadjusted" in title and "p-values unaffected" in title
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---- SNP_view_index ----
def test_build_snp_view_index_plotted_flag_and_collapse():
    plotted = ["s0", "s2"]
    idx = build_snp_view_index(_SIG, plotted)
    assert idx.loc[idx["SNP"] == "s0", "Plotted"].iloc[0]
    assert not idx.loc[idx["SNP"] == "s1", "Plotted"].iloc[0]
    assert (idx["Representative_SNP"] == idx["SNP"]).all()          # no collapse -> self

    collapse = pd.DataFrame({"SNP": ["s0", "s1"], "Representative_SNP": ["s0", "s0"],
                             "r2_to_representative": [1.0, 0.95],
                             "Representative_Of_Count": [2, 2]})
    idx2 = build_snp_view_index(_SIG, plotted, collapse_df=collapse)
    assert idx2.loc[idx2["SNP"] == "s1", "Representative_SNP"].iloc[0] == "s0"
    assert idx2.loc[idx2["SNP"] == "s1", "r2_to_representative"].iloc[0] == 0.95
    assert len(idx2) == len(_SIG)                                   # one row per sig SNP; complete
