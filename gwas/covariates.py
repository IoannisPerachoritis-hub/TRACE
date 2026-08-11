"""User-supplied covariates for GWAS: loading, column selection, sample alignment.

The MLM / MLMM / FarmCPU scans already accept a fixed-effect covariate matrix (the
principal components). These helpers turn a user-supplied covariate file into a
matrix row-aligned to the analysis sample set, so it can be concatenated onto the
PC block. numpy + pandas only; no Streamlit / FaST-LMM dependency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# column names commonly used for the sample identifier (case-insensitive)
_ID_ALIASES = (
    "iid", "fid", "sample", "sample_id", "sampleid", "id", "taxa",
    "accession", "line", "genotype", "geno_id",
)


def load_covariate_frame(path, id_col=None):
    """Read a covariate file into a DataFrame indexed by sample ID (str).

    The file may be CSV / TSV / whitespace-delimited (auto-detected). The sample-ID
    column is ``id_col`` if given, otherwise the first column whose name matches a
    known alias (IID / FID / sample / accession / ...), otherwise the first column.
    Every remaining column is treated as a covariate.
    """
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    if df.shape[1] < 2:
        raise ValueError(
            "Covariate file needs a sample-ID column plus at least one covariate column."
        )
    if id_col is not None:
        if id_col not in df.columns:
            raise ValueError(
                f"Covariate ID column '{id_col}' not found (columns: {list(df.columns)})."
            )
    else:
        lower = {str(c).lower(): c for c in df.columns}
        id_col = next((lower[a] for a in _ID_ALIASES if a in lower), df.columns[0])
    df = df.copy()
    df[id_col] = df[id_col].astype(str).str.strip()
    df = df.set_index(id_col)
    df.index = df.index.astype(str)
    return df


def select_covariate_columns(covar_df, cols=None):
    """Return a numeric-only covariate DataFrame.

    ``cols`` (a list of column names) selects a subset; the default is every column.
    Columns are coerced to numeric; a column that is entirely non-numeric raises a
    clear error asking the user to encode categoricals as numeric indicators.
    """
    if cols:
        missing = [c for c in cols if c not in covar_df.columns]
        if missing:
            raise ValueError(
                f"Covariate column(s) not in file: {missing} "
                f"(available: {list(covar_df.columns)})."
            )
        sub = covar_df[list(cols)].copy()
    else:
        sub = covar_df.copy()
    if sub.shape[1] == 0:
        raise ValueError("No covariate columns selected.")
    out = sub.apply(pd.to_numeric, errors="coerce")
    non_numeric = [
        c for c in out.columns
        if out[c].isna().all() and not sub[c].isna().all()
    ]
    if non_numeric:
        raise ValueError(
            f"Covariate column(s) {non_numeric} are non-numeric. Encode categorical "
            f"covariates as numeric indicators (e.g. one-hot 0/1) before using them."
        )
    return out


def align_covariates(covar_df, sample_ids):
    """Align a numeric covariate DataFrame to ``sample_ids`` (the analysis sample order).

    Returns
    -------
    matrix : (n, k) float ndarray, rows in ``sample_ids`` order.
    names  : list of covariate column names.
    complete_mask : (n,) bool ndarray, True where the sample has every covariate
        value present (samples absent from ``covar_df``, or with any NaN, are False).
    """
    ids = [str(s) for s in np.asarray(sample_ids).ravel()]
    sub = covar_df.reindex(ids)
    matrix = sub.to_numpy(dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    complete = np.isfinite(matrix).all(axis=1)
    return matrix, list(covar_df.columns), complete
