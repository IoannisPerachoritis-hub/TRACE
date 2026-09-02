"""Tests for gwas/models.py — OLS, F-tests, auto PC selection."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from scipy import stats as sp_stats
from gwas.models import (
    _ols_fit,
    _nested_f_test,
    _one_hot_drop_first,
)


# ── _ols_fit ─────────────────────────────────────────────────

class TestOLSFit:
    def test_perfect_fit(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x + 3.0
        X = np.column_stack([np.ones_like(x), x])
        beta, resid, sse, df = _ols_fit(y, X)
        np.testing.assert_allclose(beta, [3.0, 2.0], atol=1e-10)
        assert sse < 1e-20

    def test_intercept_only(self):
        y = np.array([2.0, 4.0, 6.0])
        X = np.ones((3, 1))
        beta, resid, sse, df = _ols_fit(y, X)
        assert beta[0] == pytest.approx(4.0)
        np.testing.assert_allclose(resid, [-2.0, 0.0, 2.0], atol=1e-10)

    def test_residuals_orthogonal_to_X(self):
        rng = np.random.default_rng(10)
        n = 50
        x = rng.normal(0, 1, n)
        y = 3.0 * x + rng.normal(0, 0.1, n)
        X = np.column_stack([np.ones(n), x])
        beta, resid, sse, df = _ols_fit(y, X)
        # Normal equations: X.T @ resid ≈ 0
        np.testing.assert_allclose(X.T @ resid, 0.0, atol=1e-10)

    def test_df_resid(self):
        n, p = 20, 3
        X = np.random.default_rng(1).normal(0, 1, (n, p))
        y = np.random.default_rng(2).normal(0, 1, n)
        _, _, _, df = _ols_fit(y, X)
        assert df == n - p


# ── _nested_f_test ───────────────────────────────────────────

class TestNestedFTest:
    def test_identical_models_nan(self):
        # Same SSE and DF → df_num = 0 → NaN
        F, p = _nested_f_test(100.0, 100.0, 18, 18)
        assert np.isnan(F)

    def test_known_values(self):
        # Manually: SSE reduced=120, SSE full=80, df0=18, df1=16
        # F = ((120-80)/2) / (80/16) = 20/5 = 4.0
        F, p = _nested_f_test(120.0, 80.0, 18, 16)
        assert F == pytest.approx(4.0)
        assert 0 < p < 1

    def test_significant_reduction(self):
        # Large SSE reduction → small p
        F, p = _nested_f_test(1000.0, 10.0, 100, 98)
        assert F > 10
        assert p < 0.001

    def test_sse1_zero_nan(self):
        F, p = _nested_f_test(100.0, 0.0, 18, 16)
        assert np.isnan(F)

    def test_against_scipy_f_oneway(self):
        # Simple one-way ANOVA with 3 groups
        rng = np.random.default_rng(42)
        g1 = rng.normal(0, 1, 20)
        g2 = rng.normal(3, 1, 20)
        g3 = rng.normal(6, 1, 20)
        y = np.concatenate([g1, g2, g3])
        groups = np.array(["A"] * 20 + ["B"] * 20 + ["C"] * 20)

        # Build design matrices
        n = len(y)
        X0 = np.ones((n, 1))  # intercept only
        H = np.zeros((n, 2))
        H[20:40, 0] = 1.0  # group B
        H[40:60, 1] = 1.0  # group C
        X1 = np.column_stack([X0, H])

        _, _, sse0, df0 = _ols_fit(y, X0)
        _, _, sse1, df1 = _ols_fit(y, X1)
        F_ours, p_ours = _nested_f_test(sse0, sse1, df0, df1)

        F_scipy, p_scipy = sp_stats.f_oneway(g1, g2, g3)
        assert F_ours == pytest.approx(F_scipy, rel=1e-6)
        assert p_ours == pytest.approx(p_scipy, rel=1e-4)


# ── _one_hot_drop_first ──────────────────────────────────────

class TestOneHotDropFirst:
    def test_three_groups(self):
        g = np.array(["A", "B", "C", "A", "B", "C"])
        H, keep, all_lev = _one_hot_drop_first(g)
        assert H.shape == (6, 2)
        assert len(keep) == 2
        assert len(all_lev) == 3

    def test_single_group_returns_none(self):
        g = np.array(["A", "A", "A"])
        H, keep, all_lev = _one_hot_drop_first(g)
        assert H is None
        assert keep == []

    def test_two_groups(self):
        g = np.array(["X", "Y", "X", "Y"])
        H, keep, all_lev = _one_hot_drop_first(g)
        assert H.shape == (4, 1)
        assert len(keep) == 1

    def test_column_sums(self):
        g = np.array(["A", "A", "B", "B", "C"])
        H, keep, all_lev = _one_hot_drop_first(g)
        # keep = ["B", "C"]; B has 2, C has 1
        assert H[:, 0].sum() == 2  # group B count
        assert H[:, 1].sum() == 1  # group C count



# ── auto_select_pcs ──────────────────────────────────────────

def _make_fake_gwas_df(n_snps=200, seed=0):
    """Create a synthetic GWAS result DataFrame for mocking."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "SNP": [f"snp_{i}" for i in range(n_snps)],
        "Chr": ["1"] * n_snps,
        "Pos": np.arange(n_snps) * 1000,
        "PValue": rng.uniform(0.001, 1.0, n_snps),
    })
