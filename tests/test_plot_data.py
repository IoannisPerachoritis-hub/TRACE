"""The data behind the GUI's figures (utils.plot_data): what each "Download data (CSV)" holds.

Each builder must return exactly what its figure draws: the bars of a histogram (the
MAF spectrum, the QC strip's three panels), the components of the scree plot, the
points of a Q-Q plot, a PCA panel's samples.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from gwas.plotting import plot_maf_histogram, plot_pca_scree  # noqa: E402
from utils.plot_data import (  # noqa: E402
    HISTOGRAM_COLUMNS,
    csv_bytes,
    eigen_table,
    histogram_bins,
    pca_points_table,
    qq_table,
    values_table,
)
from utils.pub_theme import show_plot  # noqa: E402

RNG = np.random.default_rng(5)


def test_histogram_bins_are_the_bars_drawn():
    maf = RNG.uniform(0.05, 0.5, 400)
    fig = plot_maf_histogram(maf, maf_threshold=0.05)
    try:
        t = histogram_bins(fig)
    finally:
        plt.close(fig)
    assert list(t.columns) == HISTOGRAM_COLUMNS
    assert len(t) == 50 and int(t["Count"].sum()) == 400        # plot_maf_histogram: 50 bins
    np.testing.assert_allclose(t["Bin_start"].iloc[1:].to_numpy(), t["Bin_end"].iloc[:-1].to_numpy())
    counts, edges = np.histogram(maf, bins=50)
    assert t["Count"].tolist() == counts.tolist()
    np.testing.assert_allclose(t["Bin_start"].to_numpy(), edges[:-1])


def test_histogram_bins_of_a_strip_name_each_panel():
    fig, axes = plt.subplots(1, 3)
    for k, ax in enumerate(axes):
        ax.hist(RNG.normal(size=100), bins=40)
        ax.set_title(f"panel {k}")
    try:
        t = histogram_bins(fig)
    finally:
        plt.close(fig)
    assert t.groupby("Panel").size().to_dict() == {"panel 0": 40, "panel 1": 40, "panel 2": 40}
    assert t.groupby("Panel")["Count"].sum().tolist() == [100, 100, 100]


def test_eigen_table_is_the_scree_plot():
    ev = np.sort(RNG.uniform(0.1, 5.0, 30))[::-1]
    t = eigen_table(ev)
    assert list(t.columns) == ["PC", "Eigenvalue", "Variance_explained_pct", "Cumulative_pct"]
    assert t["PC"].tolist() == list(range(1, 31))
    assert t["Cumulative_pct"].iloc[-1] == pytest.approx(100.0)
    fig = plot_pca_scree(ev, n_pcs_used=3)
    try:
        bars = [p.get_height() for p in fig.axes[0].patches]
    finally:
        plt.close(fig)
    np.testing.assert_allclose(t.head(20)["Variance_explained_pct"].to_numpy(), bars)


def test_qq_table_is_the_qq_plots_points():
    from pages._pheno_qc.diagnostics import _qq_fig

    v = RNG.normal(3.0, 2.0, 80)
    t = qq_table(v)
    assert list(t.columns) == ["Theoretical_quantile", "Sample_quantile", "Fitted_line"]
    assert len(t) == 80
    fig = _qq_fig(v, "t")
    try:
        x, y = fig.axes[0].get_lines()[0].get_data()
    finally:
        plt.close(fig)
    np.testing.assert_allclose(t["Theoretical_quantile"].to_numpy(), x)
    np.testing.assert_allclose(t["Sample_quantile"].to_numpy(), y)


def test_values_and_pca_tables():
    v = values_table([1.0, 2.5, 4.0], "Trait1")
    assert list(v.columns) == ["Trait1"] and len(v) == 3
    pcs = RNG.normal(size=(5, 4))
    p = pca_points_table([f"s{i}" for i in range(5)], pcs, ["H1", "H2", "H1", "Other", "H2"])
    assert list(p.columns) == ["Sample", "PC1", "PC2", "Haplotype"] and len(p) == 5
    np.testing.assert_array_equal(p[["PC1", "PC2"]].to_numpy(), pcs[:, :2])


def test_csv_bytes_and_show_plot_needs_a_name():
    df = pd.DataFrame({"a": [1, 2]})
    assert csv_bytes(df) == df.to_csv(index=False).encode("utf-8")   # as every GUI download
    assert csv_bytes(b"x,y\n") == b"x,y\n" and csv_bytes("x\n") == b"x\n"
    fig = plt.figure()
    try:
        with pytest.raises(ValueError, match="data_name"):
            show_plot(fig, "small", data=df)
    finally:
        plt.close(fig)


def test_the_pca_inset_labels_are_the_assignments_csvs(tmp_path):
    """With the minimum group size raised so that H4 (4 samples) falls below it, the
    inset's label is the assignments CSV's Haplotype for every sample: H4 stays H4 (not
    "Other"), marked not tested; samples without a haplotype have none."""
    from gwas.haplotype import haplotype_assignments, run_haplotype_block_gwas
    from tests.test_haplotype_assignments import _synthetic
    from utils.plot_data import pca_haplotype_labels

    d = _synthetic()
    res, tables = run_haplotype_block_gwas(
        haplo_df=d["blocks"], chroms=d["chroms"], positions=d["positions"],
        geno_imputed=d["G"], sid=d["sid"], geno_df=d["geno_df"], pheno_df=d["pheno"],
        trait_col="t", pcs=None, min_hap_count=4, min_group_size=5, n_perm=50,
        hap_test="pcs")
    path = tmp_path / "assignments.csv"
    haplotype_assignments(res, tables, d["geno_df"], d["pheno"], "t").to_csv(path, index=False)
    csv = pd.read_csv(path, keep_default_na=False, dtype={"Haplotype": str})
    block = csv[csv["Block_start"] == 1000]
    samples = list(d["geno_df"].index)              # the PCA's rows: the genotype order
    labels, in_test = pca_haplotype_labels(block, samples)
    by_sample = block.set_index("Sample")
    assert list(labels) == [by_sample.loc[s, "Haplotype"] for s in samples]
    assert list(in_test) == [by_sample.loc[s, "in_haplotype_test"] in (True, "True")
                             for s in samples]
    h4 = [s for s in samples if by_sample.loc[s, "Haplotype"] == "H4"]
    assert len(h4) == 4 and not any(in_test[samples.index(s)] for s in h4)
    assert set(by_sample.loc[h4, "exclusion_reason"]) == {"group below minimum size"}
    nohap = [s for s in samples if by_sample.loc[s, "Haplotype"] == ""]
    assert len(nohap) == 4 and not any(in_test[samples.index(s)] for s in nohap)
    # a sample the table does not list: no haplotype, not tested
    extra, extra_test = pca_haplotype_labels(block, ["not_genotyped"])
    assert list(extra) == [""] and list(extra_test) == [False]
    # the inset's data CSV carries the same labels and the test flag
    pcs = np.zeros((len(samples), 2))
    pts = pca_points_table(samples, pcs, labels, in_test=in_test)
    assert pts["Haplotype"].tolist() == list(labels)
    assert pts["in_haplotype_test"].tolist() == list(in_test)

