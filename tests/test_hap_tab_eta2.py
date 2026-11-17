"""Every η² the LD Blocks & Haplotypes tab shows for a block is the haplotype test's eta2.

The block table's "η² (haplotype test)", the box plot's title and the forest plot's metric and
title show the test's eta2 for the block: its samples, groups and model (with the
kinship-aware test, the default, it is not a one-way ANOVA of the plotted values). The forest
plot used to compute its own η² from the plotted values. gwas.guipost.block_qc_effects' value
over every SNP of the block's coordinate window is another quantity: Window_R2 (it was named
EtaSq), "Window R² (all SNPs in the block window)" in the block table's CSV
(tests/test_block_table.py).

The tab's own function runs here outside a Streamlit page (bare mode), with the forest plot's
metric and figure recorded.
"""
import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _anova_r2(y, groups):
    """One-way ANOVA's SS_between / SS_total."""
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    m = y.mean()
    ssb = sum((groups == g).sum() * (y[groups == g].mean() - m) ** 2 for g in np.unique(groups))
    return float(ssb / ((y - m) ** 2).sum())


def _forest(monkeypatch, mlg_df, eta2_test):
    """The forest plot's metrics (label, value) and figure titles for one block."""
    import pages._ld_tabs.tab_genome_wide as tgw

    metrics, titles = [], []
    monkeypatch.setattr(tgw.st, "metric",
                        lambda label, value, *a, **k: metrics.append((label, value)))
    monkeypatch.setattr(tgw, "show_plot", lambda fig, *a, **k: titles.append(
        fig.axes[0].get_title() if getattr(fig, "axes", None) else None))
    monkeypatch.setattr(tgw, "export_matplotlib", lambda *a, **k: None)
    tgw._render_tukey_and_boxplot(
        mlg_df=mlg_df, trait_col="t", trait_label="t", mlg_order=["H1", "H2", "H3"],
        hap_counts=mlg_df["MLG_label"].value_counts(), selected_block="b0", block_chr="2",
        block_start=100, block_end=300, plot_values=None, eta2_test=eta2_test)
    return metrics, titles


def _block_rows(seed=4):
    rng = np.random.default_rng(seed)
    labels = np.repeat(["H1", "H2", "H3"], [12, 10, 8])
    vals = rng.normal(0.0, 1.0, labels.size) + np.select(
        [labels == "H2", labels == "H3"], [1.0, 2.5], 0.0)
    return pd.DataFrame({"Sample": [f"S{i:02d}" for i in range(labels.size)],
                         "MLG_label": labels, "t": vals})


def test_the_forest_plot_shows_the_tests_eta2(monkeypatch):
    rows = _block_rows()
    own = _anova_r2(rows["t"], rows["MLG_label"])      # what the forest plot used to show
    eta2_test = 0.1234
    assert f"{own:.3f}" != f"{eta2_test:.3f}"          # so the test tells the two apart
    metrics, titles = _forest(monkeypatch, rows, eta2_test)
    assert metrics == [("Effect size (η²)", "0.123")]
    assert titles == ["Haplotype Effect Sizes: η² = 0.123"]


def test_without_the_tests_eta2_the_forest_plot_shows_none(monkeypatch):
    # a block the kinship-aware test could not test has no eta2 (its kinship_model says why)
    rows = _block_rows()
    for missing in (float("nan"), None):
        metrics, titles = _forest(monkeypatch, rows, missing)
        assert metrics == [] and titles == ["Haplotype Effect Sizes"]


def test_the_block_effects_window_value_is_an_r2_named_so():
    from gwas import guipost as gp

    n, m = 60, 12
    ids = np.array([f"S{i:02d}" for i in range(n)])
    grp = np.arange(n) % 3
    rng = np.random.default_rng(8)
    G = rng.binomial(2, 0.3, (n, m)).astype(float)
    G[:, :5] = np.array([[0, 0, 1, 1, 2], [2, 1, 0, 2, 0], [1, 2, 2, 0, 1]], float)[grp]
    positions = 1_000_000 + np.arange(m) * 10_000          # the window holds the first five
    y = 10.0 + 2.0 * grp + rng.normal(0.0, 1.0, n)
    hap = pd.DataFrame({"Chr": ["2"], "Start": [1_000_000], "End": [1_040_000], "eta2": [0.5]})
    eff = gp.block_qc_effects(
        hap_gwas_df=hap, chroms=np.array(["2"] * m), positions=positions, geno_imputed=G,
        geno_df=pd.DataFrame(G, index=ids), pheno_for_hap=pd.DataFrame({"t": y}, index=ids),
        trait_col="t", min_hap_count=5, min_group_size=3, compute_maf=True,
        geno_encoding="dosage012", pcs=None, strict_selfing=False)
    assert "EtaSq" not in eff.columns
    assert eff["n_snps_total"].tolist() == [5] and eff["n_groups_tested_effect"].tolist() == [3]
    assert eff["n_samples_used_effect"].tolist() == [n]
    # the R2 of the window's three groups over every phenotyped sample
    assert np.isclose(eff["Window_R2"].iloc[0], _anova_r2(y, grp), rtol=0, atol=1e-12)
    assert eff["eta2"].tolist() == [0.5]                   # the test's eta2 is left as it was
