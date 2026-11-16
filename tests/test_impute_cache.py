"""The raw-export contract (P3) of the GWAS page's genotype matrices.

Until batch 13 this file also tested the imputation's st.cache_data entry (persist="disk").
That cache is gone: the page keeps one current preprocessing per session, keyed by every
setting (gwas.guigwas.preprocess), and the mean imputation's equality with SimpleImputer is
checked in tests/test_impute.py. What is left here is the raw export: its missing calls stay
NaN, and the imputed table has none.
"""
import numpy as np
import pandas as pd

from gwas.qc import _pipeline_build_geno_matrices


def _df(n, m, seed=0, missing=0.05):
    rng = np.random.default_rng(seed)
    G = rng.integers(0, 3, (n, m)).astype(float)
    G[rng.random((n, m)) < missing] = np.nan
    return pd.DataFrame(G, index=[f"S{i}" for i in range(n)])


def test_raw_export_preserves_nan_count():
    df = _df(20, 15, seed=3, missing=0.10)
    raw, rate, imputed, iid = _pipeline_build_geno_matrices(df, method="mean")
    export = pd.DataFrame(raw, index=list(df.index)).reset_index()
    nan_in_export = int(export.drop(columns=export.columns[0]).isna().sum().sum())
    assert nan_in_export == int(np.isnan(raw).sum())
    assert nan_in_export > 0                            # this matrix has missing calls
    assert not np.isnan(imputed).any()                  # the imputed export has none
