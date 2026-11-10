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
    clear error asking the user to encode categoricals as numeric indicators. A
    constant, duplicate, or exactly-collinear column is also rejected (it would reach
    FaST-LMM unguarded and silently alter the p-values).
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
    # A constant / duplicate / exactly-collinear covariate passes numeric validation but
    # reaches FaST-LMM's single_snp unguarded, where it silently alters the p-values (a
    # constant column is collinear with the model intercept). Reject with the offending
    # names, in the same style as the non-numeric error above.
    _chk = out.dropna(axis=0, how="any")
    if _chk.shape[0] >= 2:
        _std = _chk.std(axis=0, ddof=0)
        _const = [c for c in out.columns if float(_std.get(c, 0.0)) <= 1e-12]
        if _const:
            raise ValueError(
                f"Covariate column(s) {_const} are constant (zero variance); a constant "
                f"covariate is collinear with the model intercept. Remove them."
            )
        _X = np.column_stack([np.ones(_chk.shape[0]), _chk.to_numpy(dtype=float)])
        if _chk.shape[0] >= _X.shape[1] and np.linalg.matrix_rank(_X) < _X.shape[1]:
            _keep = [0]
            for _j in range(1, _X.shape[1]):
                if np.linalg.matrix_rank(_X[:, _keep + [_j]]) > len(_keep):
                    _keep.append(_j)
            _cols = list(out.columns)
            _bad = [_cols[_j - 1] for _j in range(1, _X.shape[1]) if _j not in _keep]
            raise ValueError(
                f"Covariate column(s) {_bad} are collinear (linearly dependent on the "
                f"other covariates or the intercept). Remove or combine them."
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


class CovariateDropError(ValueError):
    """No analysis sample has every covariate value (often: the covariate file's sample
    IDs do not match the VCF's)."""


def drop_covariate_missing(covar_df, geno_df, pheno, y):
    """Keep the samples that have every covariate value, at the stage the CLI drops the
    others: after phenotype QC, before SNP QC, imputation, kinship and the PCs. The CLI,
    One-Click and the GWAS page all call it there, so every array built later holds the
    same samples.

    ``geno_df`` and ``pheno`` are row-aligned by sample (``geno_df.index``) and ``y`` is
    the trait vector in that order. Returns (geno_df, pheno, y, dropped sample IDs); the
    inputs come back unchanged when no sample is dropped.
    """
    _, _, ok = align_covariates(covar_df, geno_df.index)
    dropped = [str(s) for s, keep in zip(geno_df.index, ok) if not keep]
    if dropped:
        geno_df, pheno, y = geno_df.iloc[ok], pheno.iloc[ok], y[ok]
    return geno_df, pheno, y, dropped


def covariate_drop_message(dropped, n_total):
    """The warning for the samples dropped for a missing covariate value (the CLI's log,
    the GWAS page and One-Click): the count, the first 10 IDs and how many more."""
    dropped = [str(s) for s in dropped]
    n = len(dropped)
    shown = ", ".join(dropped[:10]) + (f", and {n - 10} more" if n > 10 else "")
    return (f"{n} of {int(n_total)} analysis samples have no covariate value and were "
            f"dropped: {shown}.")
