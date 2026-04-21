"""Tests for gwas/qc.py pipeline sub-functions.

These tests import _pipeline_* functions directly — they work WITHOUT
the _mock_streamlit fixture, validating the Streamlit decoupling.
"""
import numpy as np
import pandas as pd
import pytest

from gwas.qc import (
    _pipeline_phenotype_qc,
    _pipeline_snp_qc,
    _pipeline_build_geno_matrices,
)


# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_geno_df(rng):
    """10 samples, 20 SNPs with some NaN."""
    n, m = 10, 20
    G = rng.choice([0.0, 1.0, 2.0], size=(n, m)).astype(float)
    # Inject ~10% missingness
    mask = rng.random((n, m)) < 0.1
    G[mask] = np.nan
    samples = [f"S{i}" for i in range(n)]
    snps = [f"SNP_{i}" for i in range(m)]
    return pd.DataFrame(G, index=samples, columns=snps)


@pytest.fixture
def small_pheno(small_geno_df, rng):
    """Phenotype DataFrame aligned to small_geno_df."""
    n = small_geno_df.shape[0]
    return pd.DataFrame(
        {"trait": rng.normal(5.0, 1.0, n)},
        index=small_geno_df.index,
    )


# ── _pipeline_phenotype_qc ─────────────────────────────────

class TestPipelinePhenotypeQc:
    def test_returns_aligned_shapes(self, small_geno_df, small_pheno):
        geno, pheno, y = _pipeline_phenotype_qc(
            small_geno_df, small_pheno, "trait", "None (raw values)", 0.5,
        )
        assert geno.shape[0] == pheno.shape[0] == y.shape[0]
        assert geno.index.equals(pheno.index)

    def test_drops_nan_phenotype(self, small_geno_df, small_pheno):
        # Set first sample phenotype to NaN
        small_pheno.iloc[0, 0] = np.nan
        geno, pheno, y = _pipeline_phenotype_qc(
            small_geno_df, small_pheno, "trait", "None (raw values)", 0.5,
        )
        assert geno.shape[0] == small_geno_df.shape[0] - 1
        assert not np.isnan(y).any()

    def test_zscore_normalization(self, small_geno_df, small_pheno):
        _, _, y = _pipeline_phenotype_qc(
            small_geno_df, small_pheno, "trait", "Z-score (mean=0, sd=1)", 0.5,
        )
        assert abs(np.mean(y)) < 0.1
        assert abs(np.std(y) - 1.0) < 0.2

    def test_rank_int_normalization(self, small_geno_df, small_pheno):
        _, _, y = _pipeline_phenotype_qc(
            small_geno_df, small_pheno, "trait",
            "Rank-based inverse normal (INT)", 0.5,
        )
        # INT output should have mean ~0 and sd ~1
        assert abs(np.mean(y)) < 0.1
        assert abs(np.std(y) - 1.0) < 0.2

    def test_near_zero_variance_raises(self, small_geno_df):
        pheno = pd.DataFrame(
            {"trait": [5.0] * small_geno_df.shape[0]},
            index=small_geno_df.index,
        )
        with pytest.raises(ValueError, match="near-zero variance"):
            _pipeline_phenotype_qc(
                small_geno_df, pheno, "trait", "None (raw values)", 0.5,
            )


# ── _pipeline_snp_qc ───────────────────────────────────────

class TestPipelineSnpQc:
    def test_qc_counts(self, small_geno_df):
        chroms = np.array(["1"] * small_geno_df.shape[1])
        positions = np.arange(1, small_geno_df.shape[1] + 1) * 1000
        sid = np.array(small_geno_df.columns)

        geno, chroms_out, chroms_num, pos, sid_out, n_raw, qc, _ = _pipeline_snp_qc(
            small_geno_df, chroms, positions, sid,
            maf_thresh=0.05, miss_thresh=0.5, mac_thresh=1, drop_alt=False,
        )
        assert qc["Total SNPs"] == small_geno_df.shape[1]
        assert qc["Pass ALL"] == geno.shape[1]
        assert qc["Fail ANY"] == n_raw - geno.shape[1]

    def test_strict_maf_removes_monomorphic(self, rng):
        """A monomorphic SNP (all 0s) should be removed by MAF filter."""
        n = 20
        G = np.column_stack([
            np.zeros(n),  # monomorphic
            rng.choice([0.0, 1.0, 2.0], size=n),  # polymorphic
        ])
        geno_df = pd.DataFrame(G, index=[f"S{i}" for i in range(n)],
                               columns=["mono", "poly"])
        chroms = np.array(["1", "1"])
        positions = np.array([1000, 2000])
        sid = np.array(["mono", "poly"])

        geno, *_ = _pipeline_snp_qc(
            geno_df, chroms, positions, sid,
            maf_thresh=0.01, miss_thresh=0.5, mac_thresh=1, drop_alt=False,
        )
        assert "mono" not in geno.columns

    def test_sorted_by_position(self, small_geno_df):
        chroms = np.array(["1"] * small_geno_df.shape[1])
        # Reverse positions so they start unsorted
        positions = np.arange(small_geno_df.shape[1], 0, -1) * 1000
        sid = np.array(small_geno_df.columns)

        geno, _, _, pos_out, _, _, _, _ = _pipeline_snp_qc(
            small_geno_df, chroms, positions, sid,
            maf_thresh=0.0, miss_thresh=1.0, mac_thresh=0, drop_alt=False,
        )
        assert np.all(pos_out[:-1] <= pos_out[1:])


# ── _pipeline_build_geno_matrices ───────────────────────────

class TestPipelineBuildGenoMatrices:
    def test_shapes(self, small_geno_df):
        raw, imp_rate, imputed, iid = _pipeline_build_geno_matrices(small_geno_df)
        n, m = small_geno_df.shape
        assert raw.shape == (n, m)
        assert imputed.shape == (n, m)
        assert imp_rate.shape == (m,)
        assert iid.shape == (n, 2)

    def test_imputed_has_no_nan(self, small_geno_df):
        _, _, imputed, _ = _pipeline_build_geno_matrices(small_geno_df)
        assert not np.isnan(imputed).any()

    def test_raw_preserves_nan(self, small_geno_df):
        raw, _, _, _ = _pipeline_build_geno_matrices(small_geno_df)
        # small_geno_df has ~10% NaN injected
        assert np.isnan(raw).any()

    def test_imputation_rate(self, small_geno_df):
        _, imp_rate, _, _ = _pipeline_build_geno_matrices(small_geno_df)
        # Imputation rate = fraction of NaN per SNP
        expected = small_geno_df.isna().mean(axis=0).values.astype(np.float32)
        np.testing.assert_array_almost_equal(imp_rate, expected)
