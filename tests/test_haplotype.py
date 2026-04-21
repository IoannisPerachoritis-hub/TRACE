"""Tests for gwas/haplotype.py — haplotype block tests and permutation."""
import numpy as np
import pytest
from gwas.haplotype import (
    block_test_lm_with_pcs,
    freedman_lane_perm_pvalue,
    compute_haplotype_effects,
)


# ── block_test_lm_with_pcs ───────────────────────────────────

class TestBlockTestLM:
    def test_strong_effect(self):
        # y perfectly determined by group → large F, small p
        y = np.array([1.0]*10 + [5.0]*10 + [9.0]*10)
        groups = np.array(["A"]*10 + ["B"]*10 + ["C"]*10)
        F, p, df1, df2 = block_test_lm_with_pcs(y, groups)
        assert F > 50
        assert p < 0.001

    def test_no_effect(self):
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 30)
        groups = np.array(["A"]*10 + ["B"]*10 + ["C"]*10)
        F, p, df1, df2 = block_test_lm_with_pcs(y, groups)
        # Should usually not be significant with random data
        assert np.isfinite(F)
        assert 0 <= p <= 1

    def test_single_group_returns_nan(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        groups = np.array(["A", "A", "A", "A"])
        F, p, df1, df2 = block_test_lm_with_pcs(y, groups)
        assert np.isnan(F)
        assert df1 == 0

    def test_with_pcs(self):
        rng = np.random.default_rng(42)
        n = 30
        y = np.array([1.0]*10 + [5.0]*10 + [9.0]*10)
        groups = np.array(["A"]*10 + ["B"]*10 + ["C"]*10)
        pcs = rng.normal(0, 1, (n, 2))
        F, p, df1, df2 = block_test_lm_with_pcs(y, groups, pcs=pcs)
        assert np.isfinite(F)
        assert 0 <= p <= 1


# ── freedman_lane_perm_pvalue ────────────────────────────────

class TestFreedmanLane:
    def test_deterministic_with_seed(self):
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 20)
        groups = np.array(["A"]*7 + ["B"]*7 + ["C"]*6)
        F1, p1 = freedman_lane_perm_pvalue(y, groups, n_perm=100, seed=99)
        F2, p2 = freedman_lane_perm_pvalue(y, groups, n_perm=100, seed=99)
        assert F1 == pytest.approx(F2)
        assert p1 == pytest.approx(p2)

    def test_strong_signal_low_p(self):
        # y is perfectly separated by group → p should be very small
        y = np.array([0.0]*10 + [10.0]*10 + [20.0]*10)
        groups = np.array(["A"]*10 + ["B"]*10 + ["C"]*10)
        F, p = freedman_lane_perm_pvalue(y, groups, n_perm=200, seed=0)
        assert p < 0.05

    def test_null_signal_high_p(self):
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 30)
        groups = np.array(["A"]*10 + ["B"]*10 + ["C"]*10)
        F, p = freedman_lane_perm_pvalue(y, groups, n_perm=200, seed=0)
        # With random data, p should generally not be tiny
        assert p > 0.01

    def test_with_covariates(self):
        rng = np.random.default_rng(42)
        n = 30
        y = np.array([0.0]*10 + [10.0]*10 + [20.0]*10)
        groups = np.array(["A"]*10 + ["B"]*10 + ["C"]*10)
        pcs = rng.normal(0, 1, (n, 2))
        F, p = freedman_lane_perm_pvalue(y, groups, pcs=pcs, n_perm=100, seed=0)
        assert np.isfinite(F)
        assert 0 <= p <= 1

    def test_single_group_nan(self):
        y = np.array([1.0, 2.0, 3.0])
        groups = np.array(["A", "A", "A"])
        F, p = freedman_lane_perm_pvalue(y, groups, n_perm=50, seed=0)
        assert np.isnan(F)


# ── compute_haplotype_effects ────────────────────────────────

class TestComputeHaplotypeEffects:
    def test_perfect_separation(self):
        y = np.array([1.0, 1.0, 5.0, 5.0, 9.0, 9.0])
        g = np.array(["A", "A", "B", "B", "C", "C"])
        result = compute_haplotype_effects(y, g)
        assert result["eta2"] == pytest.approx(1.0, abs=1e-10)

    def test_no_effect(self):
        y = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        g = np.array(["A", "A", "B", "B", "C", "C"])
        result = compute_haplotype_effects(y, g)
        # All values identical → ss_total = 0 → eta2 = 0/0 = NaN or 0
        assert result["eta2"] == 0.0 or np.isnan(result["eta2"])

    def test_eta2_range(self):
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 30)
        g = np.array(["A"]*10 + ["B"]*10 + ["C"]*10)
        result = compute_haplotype_effects(y, g)
        assert 0.0 <= result["eta2"] <= 1.0

    def test_hap_stats_structure(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        g = np.array(["A", "A", "A", "B", "B", "B"])
        result = compute_haplotype_effects(y, g)
        assert "eta2" in result
        assert "hap_stats" in result
        for hap, stats in result["hap_stats"].items():
            assert "n" in stats
            assert "mean" in stats
            assert "se" in stats

    def test_few_samples_nan(self):
        y = np.array([1.0, 2.0])
        g = np.array(["A", "B"])
        result = compute_haplotype_effects(y, g)
        assert np.isnan(result["eta2"])
