"""Persistent imputation cache (P1) + raw-export contract (P3).

The cache key is the RETAINED SAMPLE SET, not the trait -- imputation depends on
which samples survived phenotype QC, so two traits with the same retained sample
set share the cache and two with different sets must not.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from gwas.utils import hash_bytes
from gwas.qc import _impute_cached, _pipeline_build_geno_matrices


def _sshash(df):
    return hash_bytes(repr(tuple(sorted(df.index.astype(str)))).encode())


def _df(n, m, seed=0, missing=0.05):
    rng = np.random.default_rng(seed)
    G = rng.integers(0, 3, (n, m)).astype(float)
    G[rng.random((n, m)) < missing] = np.nan
    return pd.DataFrame(G, index=[f"S{i}" for i in range(n)])


def _call(df, method="mean"):
    return _impute_cached(df, "vcfhash", _sshash(df), 0.05, 0.10, 5, 0.20, 0.0,
                          False, "none", None, method, 5, 20)


def _eq(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f" else np.array_equal(a, b)


# ── sample-set hash ─────────────────────────────────────────────────────────
def test_sample_set_hash_same_and_differ():
    df = _df(20, 10)
    assert _sshash(df) == _sshash(df.copy())
    assert _sshash(df) == _sshash(df.iloc[::-1])       # order-independent (sorted)
    assert _sshash(df) != _sshash(df.iloc[:-1])        # one dropped sample -> differs


# ── cache correctness ───────────────────────────────────────────────────────
def test_cache_output_equals_direct_build():
    _impute_cached.clear()
    df = _df(24, 14, seed=1)
    cached = _call(df)
    direct = _pipeline_build_geno_matrices(df, method="mean")
    assert all(_eq(a, b) for a, b in zip(cached, direct))


def test_cached_mean_bit_identical_to_simpleimputer():
    _impute_cached.clear()
    df = _df(24, 14, seed=4, missing=0.10)
    _raw, _rate, cached_imp, _iid = _call(df, "mean")
    direct = SimpleImputer(strategy="mean").fit_transform(df).astype("float32")
    assert np.array_equal(cached_imp, direct) and cached_imp.dtype == np.float32


# ── the failure mode that matters: no cross-contamination ───────────────────
def test_different_sample_set_is_not_served_the_wrong_matrix():
    _impute_cached.clear()
    df166 = _df(166, 40, seed=2)
    df157 = df166.iloc[:157]                            # a genuinely different sample set
    assert _sshash(df166) != _sshash(df157)
    r166 = _call(df166)
    r157 = _call(df157)
    assert r166[2].shape[0] == 166
    assert r157[2].shape[0] == 157                      # NOT the cached 166-row matrix
    # and the reverse order (157 already cached) still returns 166 for the 166 key
    assert _call(df166)[2].shape[0] == 166


# ── raw export (P3) ─────────────────────────────────────────────────────────
def test_raw_export_preserves_nan_count():
    df = _df(20, 15, seed=3, missing=0.10)
    raw, rate, imputed, iid = _pipeline_build_geno_matrices(df, method="mean")
    export = pd.DataFrame(raw, index=list(df.index)).reset_index()
    nan_in_export = int(export.drop(columns=export.columns[0]).isna().sum().sum())
    assert nan_in_export == int(np.isnan(raw).sum())
    assert nan_in_export > 0                            # this matrix has missing calls
    assert not np.isnan(imputed).any()                  # the imputed export has none
