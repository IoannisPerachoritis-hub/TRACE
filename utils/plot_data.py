"""The data behind the GUI's figures, as tidy tables for "Download data (CSV)".

Each builder returns exactly what its figure draws: a histogram's bars, a scree plot's
eigenvalues, a Q-Q plot's quantiles, a PCA panel's points. They are small (bins,
components or samples); genome-wide figures (Manhattan, QQ) point to the results table
instead, so no genome-wide copy is made.
"""
import numpy as np
import pandas as pd

HISTOGRAM_COLUMNS = ["Panel", "Bin_start", "Bin_end", "Count"]


def histogram_bins(fig):
    """The bars of every histogram in a matplotlib figure: one row per bar, with the
    panel's title (a 1 x 3 strip gives three panels)."""
    from matplotlib.patches import Rectangle

    rows = []
    for ax in fig.axes:
        for p in ax.patches:
            if isinstance(p, Rectangle) and p.get_width() > 0:
                h = float(p.get_height())
                rows.append({"Panel": ax.get_title(), "Bin_start": float(p.get_x()),
                             "Bin_end": float(p.get_x() + p.get_width()),
                             "Count": int(h) if float(h).is_integer() else h})
    return pd.DataFrame(rows, columns=HISTOGRAM_COLUMNS)


def eigen_table(eigenvalues):
    """One row per principal component: its eigenvalue, the % of variance it explains
    (of the eigenvalues given) and the cumulative %."""
    ev = np.asarray(eigenvalues, dtype=float).ravel()
    total = ev.sum()
    pct = ev / total * 100.0 if total > 0 else np.full(ev.shape, np.nan)
    return pd.DataFrame({"PC": np.arange(1, ev.size + 1), "Eigenvalue": ev,
                         "Variance_explained_pct": pct, "Cumulative_pct": np.cumsum(pct)})


def qq_table(values):
    """A normal Q-Q plot's points (scipy.stats.probplot, as the plot draws them) and its
    fitted line, one row per value."""
    from scipy import stats

    (osm, osr), (slope, intercept, _r) = stats.probplot(
        np.asarray(values, dtype=float), dist="norm")
    return pd.DataFrame({"Theoretical_quantile": osm, "Sample_quantile": osr,
                         "Fitted_line": slope * np.asarray(osm) + intercept})


def values_table(values, name="Value"):
    """A histogram's input values, one row each (the panel's phenotype values)."""
    return pd.DataFrame({name: np.asarray(values, dtype=float).ravel()})


def pca_points_table(sample_ids, pcs, labels=None, label_name="Haplotype"):
    """A PCA panel's points: Sample, PC1, PC2 and, when given, each sample's label."""
    pcs = np.asarray(pcs, dtype=float)
    out = pd.DataFrame({"Sample": np.asarray(sample_ids, dtype=str),
                        "PC1": pcs[:, 0], "PC2": pcs[:, 1]})
    if labels is not None:
        out[label_name] = np.asarray(labels, dtype=object)
    return out


def csv_bytes(data):
    """A table, or already encoded CSV, as the bytes a download button serves."""
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, str):
        return data.encode("utf-8")
    return data.to_csv(index=False).encode("utf-8")
