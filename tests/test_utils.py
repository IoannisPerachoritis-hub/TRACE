"""Tests for gwas/utils.py — utility functions."""
import numpy as np
import pytest
from gwas.utils import _mean_impute_cols, _rank_int_1d, stable_seed


# ── _mean_impute_cols ────────────────────────────────────────

class TestMeanImputeCols:
    def test_no_nan_unchanged(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = _mean_impute_cols(X)
        np.testing.assert_array_equal(out, X)

    def test_replaces_nan_with_column_mean(self):
        X = np.array([[1.0, 2.0], [np.nan, 4.0], [3.0, 6.0]])
        out = _mean_impute_cols(X)
        # Column 0 mean = (1+3)/2 = 2.0
        assert out[1, 0] == pytest.approx(2.0)
        # Column 1 unchanged
        assert out[1, 1] == 4.0

    def test_all_nan_column_filled_with_zero(self):
        X = np.array([[np.nan], [np.nan], [np.nan]])
        out = _mean_impute_cols(X)
        np.testing.assert_array_equal(out, [[0.0], [0.0], [0.0]])

    def test_shape_preserved(self):
        X = np.array([[1.0, np.nan], [np.nan, 3.0]])
        out = _mean_impute_cols(X)
        assert out.shape == X.shape

    def test_does_not_modify_input(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        X_orig = X.copy()
        _mean_impute_cols(X)
        np.testing.assert_array_equal(X, X_orig)


# ── _rank_int_1d ─────────────────────────────────────────────

class TestRankINT:
    def test_preserves_order(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = _rank_int_1d(x)
        # Output should be monotonically increasing
        assert np.all(np.diff(out) > 0)

    def test_nan_preserved(self):
        x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        out = _rank_int_1d(x)
        assert np.isnan(out[1])
        assert np.isfinite(out[0])

    def test_output_approx_standard_normal(self):
        rng = np.random.default_rng(7)
        x = rng.normal(50, 10, size=1000)
        out = _rank_int_1d(x)
        assert abs(np.mean(out)) < 0.1
        assert abs(np.std(out) - 1.0) < 0.1

    def test_few_values_returns_input(self):
        x = np.array([5.0, np.nan])
        out = _rank_int_1d(x)
        # < 3 finite values, should return input unchanged
        np.testing.assert_array_equal(out[0], x[0])


# ── stable_seed ──────────────────────────────────────────────

class TestStableSeed:
    def test_deterministic(self):
        s1 = stable_seed("trait", 42, "gwas")
        s2 = stable_seed("trait", 42, "gwas")
        assert s1 == s2

    def test_different_inputs_different_seeds(self):
        s1 = stable_seed("trait_A", 1)
        s2 = stable_seed("trait_B", 2)
        assert s1 != s2
