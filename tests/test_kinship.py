"""Tests for gwas/kinship.py — GRM computation, genotype standardization, LD pruning."""
import numpy as np
import pytest
from gwas.kinship import (
    _build_grm_from_Z,
    _standardize_geno_for_grm,
    _ld_prune_for_grm_by_chr_bp,
)


# ── _build_grm_from_Z ───────────────────────────────────────

class TestBuildGRM:
    def test_diagonal_near_one(self):
        rng = np.random.default_rng(42)
        Z = rng.normal(0, 1, (20, 100)).astype(np.float32)
        K = _build_grm_from_Z(Z)
        assert np.mean(np.diag(K)) == pytest.approx(1.0, abs=0.05)

    def test_symmetric(self):
        rng = np.random.default_rng(42)
        Z = rng.normal(0, 1, (15, 50)).astype(np.float32)
        K = _build_grm_from_Z(Z)
        np.testing.assert_allclose(K, K.T, atol=1e-6)

    def test_shape(self):
        Z = np.random.default_rng(1).normal(0, 1, (10, 30)).astype(np.float32)
        K = _build_grm_from_Z(Z)
        assert K.shape == (10, 10)

    def test_few_snps_returns_identity(self):
        Z = np.random.default_rng(1).normal(0, 1, (5, 5)).astype(np.float32)
        K = _build_grm_from_Z(Z)
        np.testing.assert_array_equal(K, np.eye(5, dtype=np.float32))

    def test_related_samples(self):
        rng = np.random.default_rng(42)
        Z = rng.normal(0, 1, (10, 100)).astype(np.float32)
        # Duplicate row 0 → samples 0 and 10 are "identical twins"
        Z_dup = np.vstack([Z, Z[0:1, :]])
        K = _build_grm_from_Z(Z_dup)
        # K[0, 10] should be close to K[0, 0] ≈ 1
        assert K[0, 10] == pytest.approx(K[0, 0], abs=0.05)


# ── _standardize_geno_for_grm ───────────────────────────────

class TestStandardizeGenoForGRM:
    def test_output_shape(self):
        G = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 0.0]], dtype=np.float32)
        Z = _standardize_geno_for_grm(G)
        assert Z.shape == G.shape

    def test_no_nan_output(self):
        G = np.array([[0.0, np.nan], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        Z = _standardize_geno_for_grm(G)
        assert not np.any(np.isnan(Z))

    def test_approximately_zero_mean_columns(self):
        rng = np.random.default_rng(42)
        G = rng.choice([0.0, 1.0, 2.0], size=(50, 20)).astype(np.float32)
        Z = _standardize_geno_for_grm(G)
        col_means = Z.mean(axis=0)
        np.testing.assert_allclose(col_means, 0.0, atol=0.3)

    def test_dtype_float32(self):
        G = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)
        Z = _standardize_geno_for_grm(G)
        assert Z.dtype == np.float32


# ── _ld_prune_for_grm_by_chr_bp ──────────────────────────────

class TestLdPruneForGrm:
    def test_reduces_snp_count(self):
        """LD pruning should remove some correlated SNPs."""
        rng = np.random.default_rng(42)
        n, m = 50, 200
        Z = rng.normal(0, 1, (n, m))
        # Create LD: copy column 0 with noise to columns 1-4
        for j in range(1, 5):
            Z[:, j] = Z[:, 0] + rng.normal(0, 0.1, n)
        chroms = np.array(["1"] * m)
        positions = np.arange(m) * 1000
        Zp = _ld_prune_for_grm_by_chr_bp(chroms, positions, Z, r2_thresh=0.2)
        assert Zp.shape[1] < m

    def test_single_chromosome(self):
        """Should work with a single chromosome."""
        rng = np.random.default_rng(42)
        Z = rng.normal(0, 1, (20, 30))
        chroms = np.array(["1"] * 30)
        positions = np.arange(30) * 10000
        Zp = _ld_prune_for_grm_by_chr_bp(chroms, positions, Z)
        assert Zp.shape[0] == 20
        assert Zp.shape[1] <= 30

    def test_few_snps_kept(self):
        """Chromosomes with < 2 SNPs should be kept as-is."""
        rng = np.random.default_rng(42)
        Z = rng.normal(0, 1, (10, 1))
        chroms = np.array(["1"])
        positions = np.array([100000])
        Zp = _ld_prune_for_grm_by_chr_bp(chroms, positions, Z)
        assert Zp.shape[1] == 1

    def test_return_mask(self):
        """return_mask=True should return (Z_pruned, bool_mask)."""
        rng = np.random.default_rng(42)
        Z = rng.normal(0, 1, (20, 50))
        chroms = np.array(["1"] * 25 + ["2"] * 25)
        positions = np.concatenate([np.arange(25) * 10000, np.arange(25) * 10000])
        Zp, mask = _ld_prune_for_grm_by_chr_bp(
            chroms, positions, Z, return_mask=True,
        )
        assert mask.dtype == bool
        assert mask.shape == (50,)
        assert Zp.shape[1] == mask.sum()
        np.testing.assert_array_equal(Zp, Z[:, mask])

    def test_alt_chromosome_excluded(self):
        """SNPs on chromosome 'ALT' should be excluded."""
        rng = np.random.default_rng(42)
        # Use enough independent SNPs so pruning keeps ≥10 (avoids fallback)
        n_chr1 = 40
        n_alt = 5
        Z = rng.normal(0, 1, (20, n_chr1 + n_alt))
        chroms = np.array(["1"] * n_chr1 + ["ALT"] * n_alt)
        positions = np.arange(n_chr1 + n_alt) * 10000
        Zp, mask = _ld_prune_for_grm_by_chr_bp(
            chroms, positions, Z, return_mask=True,
        )
        # ALT SNPs should never be kept
        assert not mask[n_chr1:].any()

    def test_multi_chromosome(self):
        """Pruning operates per chromosome independently."""
        rng = np.random.default_rng(42)
        Z = rng.normal(0, 1, (30, 60))
        chroms = np.array(["1"] * 30 + ["2"] * 30)
        positions = np.concatenate([np.arange(30) * 10000, np.arange(30) * 10000])
        Zp = _ld_prune_for_grm_by_chr_bp(chroms, positions, Z)
        assert Zp.shape[0] == 30
        assert Zp.shape[1] <= 60
