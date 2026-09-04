"""User covariates: gwas/covariates.py helpers + MLM integration in _run_gwas_impl."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from gwas.covariates import (
    load_covariate_frame, select_covariate_columns, align_covariates,
)


@pytest.fixture(autouse=True)
def _mock_streamlit(monkeypatch):
    """Passthrough st.cache_* so gwas.models imports without a Streamlit server."""
    import streamlit as st
    _session = {}
    monkeypatch.setattr(st, "session_state", _session)

    def _passthrough(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda fn: fn
    monkeypatch.setattr(st, "cache_data", _passthrough)
    monkeypatch.setattr(st, "cache_resource", _passthrough)
    monkeypatch.setattr(st, "empty", lambda: MagicMock())
    monkeypatch.setattr(st, "progress", lambda *a, **k: MagicMock())
    yield _session


# ── loader / selection / alignment (pure) ────────────────────

def test_load_covariate_frame_picks_id_column(tmp_path):
    p = tmp_path / "cov.csv"
    p.write_text("Accession,age,batch\nS1,3.0,1\nS2,5.0,2\nS3,7.0,1\n")
    df = load_covariate_frame(p)
    assert list(df.index) == ["S1", "S2", "S3"]
    assert list(df.columns) == ["age", "batch"]


def test_load_covariate_frame_tsv_explicit_id(tmp_path):
    p = tmp_path / "cov.tsv"
    p.write_text("x\tsample\tv\n1\tA\t9\n2\tB\t8\n")
    df = load_covariate_frame(p, id_col="sample")
    assert list(df.index) == ["A", "B"]
    assert "x" in df.columns and "v" in df.columns


def test_select_columns_subset_and_nonnumeric():
    df = pd.DataFrame({"age": [1.0, 2.0], "site": ["north", "south"]}, index=["A", "B"])
    assert list(select_covariate_columns(df, ["age"]).columns) == ["age"]
    with pytest.raises(ValueError, match="non-numeric"):
        select_covariate_columns(df, ["site"])


def test_select_columns_rejects_constant():
    # a constant column is collinear with the intercept; it reaches FaST-LMM unguarded.
    df = pd.DataFrame({"age": [1.0, 2.0, 3.0], "flat": [5.0, 5.0, 5.0]},
                      index=["A", "B", "C"])
    with pytest.raises(ValueError, match="constant"):
        select_covariate_columns(df)


def test_select_columns_rejects_duplicate():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]},
                      index=["A", "B", "C", "D"])
    with pytest.raises(ValueError, match="collinear"):
        select_covariate_columns(df)


def test_select_columns_rejects_collinear():
    df = pd.DataFrame({"a": [1.0, 2.0, 0.0, 3.0, 1.0], "b": [0.0, 1.0, 2.0, 1.0, 3.0]},
                      index=["A", "B", "C", "D", "E"])
    df["c"] = df["a"] + df["b"]           # exact linear combination
    with pytest.raises(ValueError, match="collinear"):
        select_covariate_columns(df)


def test_select_columns_accepts_full_rank():
    df = pd.DataFrame({"a": [1.0, 2.0, 0.0, 3.0, 1.0], "b": [0.0, 1.0, 2.0, 1.0, 3.0]},
                      index=["A", "B", "C", "D", "E"])
    out = select_covariate_columns(df)     # independent + varying -> passes
    assert list(out.columns) == ["a", "b"]


def test_align_covariates_order_and_completeness():
    df = pd.DataFrame({"c1": [10.0, 20.0, 30.0]}, index=["A", "B", "C"])
    M, names, complete = align_covariates(df, ["C", "A", "Z"])
    assert names == ["c1"]
    assert M[0, 0] == 30.0 and M[1, 0] == 10.0        # reordered to sample order
    assert list(complete) == [True, True, False]       # Z absent -> incomplete


# ── MLM integration (acceptance) ─────────────────────────────

def test_user_covar_none_is_byte_identical(
    gwas_geno, gwas_phenotype, gwas_pcs, gwas_snp_metadata, gwas_iid,
    gwas_K0, gwas_K_by_chr, gwas_pheno_reader,
):
    """The additive param must not change the result when no covariate is given."""
    from gwas.models import _run_gwas_impl
    meta = gwas_snp_metadata
    args = (gwas_geno, gwas_phenotype, gwas_pcs, 3, meta["sid"], meta["positions"],
            meta["chroms"], meta["chroms_num"], gwas_iid, gwas_K0, gwas_K_by_chr,
            gwas_pheno_reader, "trait")
    a = _run_gwas_impl(*args)
    b = _run_gwas_impl(*args, user_covar=None)
    pd.testing.assert_frame_equal(a, b)


def test_covariate_correlated_with_phenotype_removes_signal(
    gwas_geno, gwas_pcs, gwas_snp_metadata, gwas_iid, gwas_K0, gwas_K_by_chr,
):
    """A covariate (near-)collinear with the phenotype absorbs the planted signal."""
    from gwas.models import _run_gwas_impl
    from gwas.utils import PhenoData
    meta = gwas_snp_metadata
    rng = np.random.default_rng(2024)
    n = gwas_geno.shape[0]
    y = (2.0 * gwas_geno[:, 0].astype(float) + rng.normal(0, 1.0, n)).astype(np.float32)
    pheno = PhenoData(iid=gwas_iid, val=y)
    sid0 = str(meta["sid"][0])

    def scan(uc):
        return _run_gwas_impl(
            gwas_geno, y, gwas_pcs, 3, meta["sid"], meta["positions"],
            meta["chroms"], meta["chroms_num"], gwas_iid, gwas_K0, gwas_K_by_chr,
            pheno, "trait", user_covar=uc,
        ).set_index("SNP")

    p_base = float(scan(None).loc[sid0, "PValue"])
    covar = (y.astype(float) + rng.normal(0, 0.05 * float(np.std(y)), n)).reshape(-1, 1)
    wc = scan(covar)
    p_cov = float(wc.loc[sid0, "PValue"]) if sid0 in wc.index else 1.0

    assert p_base < 1e-2, f"planted signal should be detected (p={p_base:.2e})"
    assert p_cov > 100 * p_base, f"covariate should absorb the signal (base={p_base:.2e}, cov={p_cov:.2e})"
    assert p_cov > 0.05
