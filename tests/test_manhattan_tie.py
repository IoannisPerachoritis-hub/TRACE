"""The static Manhattan highlights exactly the SNPs the tables call significant.

The tables decide p < threshold. The figure enlarged every SNP with -log10(p) >=
-log10(threshold), so a p-value exactly on the threshold was highlighted, and counted
in the legend's "Significant (n)", without being significant in any table.
gwas.plotting.plot_manhattan_static now takes the rule's own column (``sig_mask``, as
the pipeline and the GWAS page pass it), and without one it is strict too. Checked
at an exact Bonferroni tie, and through the pipeline with a numeric threshold set
exactly to one SNP's p-value.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import gwas.pipeline as pl  # noqa: E402
import gwas.plotting as gp  # noqa: E402
from gwas.io import load_vcf_cached  # noqa: E402

_EX = Path(__file__).resolve().parent.parent / "examples"


def _highlighted(fig):
    """The enlarged points (size 18) and the legend's texts."""
    ax = fig.axes[0]
    big = [c for c in ax.collections if np.allclose(c.get_sizes(), 18)]
    pts = np.vstack([c.get_offsets() for c in big]) if big else np.empty((0, 2))
    return pts, [t.get_text() for t in ax.get_legend().get_texts()]


def test_a_p_value_exactly_on_the_threshold_is_not_highlighted():
    n = 200
    t = 0.05 / n                                       # Bonferroni
    p = np.full(n, 0.5)
    p[:3] = [t / 10, t / 2, t]                         # two below, one exactly on it
    df = pd.DataFrame({"Chr": np.repeat(["1", "2"], n // 2),
                       "Pos": np.tile(np.arange(n // 2) * 1000 + 1000, 2), "PValue": p})
    df["-log10p"] = -np.log10(df["PValue"])
    df, _tp, _tl = gp.compute_cumulative_positions(df)
    lod = -np.log10(t)
    assert int((df["-log10p"] == lod).sum()) == 1      # an exact tie in -log10 too
    for mask in (None, df["PValue"].to_numpy() < t):   # the fallback, and the rule's column
        fig = gp.plot_manhattan_static(df, lod, "Bonferroni", "tie", sig_mask=mask)
        try:
            pts, labels = _highlighted(fig)
            assert len(pts) == 2 and "Significant (2)" in labels
            assert not np.isclose(pts[:, 1], lod).any()
        finally:
            plt.close(fig)


def test_the_pipelines_figure_counts_the_tables_significant_snps(monkeypatch):
    vcf, pheno_csv = _EX / "example.vcf.gz", _EX / "example_pheno.csv"
    parsed = pl.ParsedGenotypes.from_callset(load_vcf_cached(vcf.read_bytes()))
    pheno = pd.read_csv(pheno_csv, index_col=0, encoding="utf-8-sig")
    first = pl.run_trait_pipeline(parsed, pheno, "Trait1", pl.cli_default_options(
        vcf=str(vcf), pheno=str(pheno_csv), no_plots=True))
    p = first.tables["GWAS_Trait1.csv"]["PValue"].to_numpy(dtype=float)
    t = float(np.sort(p)[2])           # a numeric threshold exactly on a SNP's p-value
    legends, real = [], gp.plot_manhattan_static

    def _rec(*a, **k):
        fig = real(*a, **k)
        legends.append(_highlighted(fig)[1])
        return fig

    monkeypatch.setattr(gp, "plot_manhattan_static", _rec)
    res = pl.run_trait_pipeline(parsed, pheno, "Trait1", pl.cli_default_options(
        vcf=str(vcf), pheno=str(pheno_csv), sig_thresh=t))
    table = res.tables["GWAS_Trait1.csv"]
    n_sig = int(table["Significant_Custom"].sum())
    assert (table["PValue"].to_numpy(dtype=float) == t).any()     # the tie is in the run
    assert n_sig == int((table["PValue"].to_numpy(dtype=float) < t).sum()) >= 1
    assert f"Significant ({n_sig})" in legends[0]                 # MLM's Manhattan
