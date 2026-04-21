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
    auto_select_pcs,
    select_best_pc_from_lambdas,
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


class TestAutoSelectPCs:
    """Tests for auto_select_pcs using mocked GWAS runs."""

    def _mock_run_gwas(self, lambda_values):
        """Return a side_effect function that returns DFs with controlled p-values."""
        from scipy.stats import chi2
        call_count = [0]

        def _side_effect(**kwargs):
            k = call_count[0]
            call_count[0] += 1
            lam = lambda_values[k] if k < len(lambda_values) else 1.0
            rng = np.random.default_rng(k)
            n_snps = 500
            # Generate chi2 values with desired lambda, convert back to p-values
            chi2_vals = rng.exponential(scale=lam * 0.4549, size=n_snps)
            pvals = 1.0 - chi2.cdf(chi2_vals, df=1)
            pvals = np.clip(pvals, 1e-300, 1.0)
            return pd.DataFrame({
                "SNP": [f"snp_{i}" for i in range(n_snps)],
                "Chr": ["1"] * n_snps,
                "Pos": np.arange(n_snps) * 1000,
                "PValue": pvals,
            })
        return _side_effect

    @patch("gwas.kinship._build_loco_kernels_impl")
    @patch("gwas.models._run_gwas_impl")
    def test_returns_correct_columns(self, mock_gwas, mock_loco):
        """Result DataFrame has expected columns."""
        rng = np.random.default_rng(0)
        n, p = 30, 100
        iid = np.array([f"S{i}" for i in range(n)])

        mock_loco.return_value = (None, {}, {})
        mock_gwas.side_effect = self._mock_run_gwas([1.5, 1.2, 1.0, 0.95])

        df = auto_select_pcs(
            geno_imputed=rng.normal(0, 1, (n, p)),
            y=rng.normal(0, 1, n),
            sid=np.array([f"snp_{i}" for i in range(p)]),
            chroms=np.array(["1"] * p),
            chroms_num=np.ones(p, dtype=int),
            positions=np.arange(p) * 1000,
            iid=iid,
            Z_grm=rng.normal(0, 1, (n, p)),
            chroms_grm=np.array(["1"] * p),
            K_base=np.eye(n),
            pcs_full=rng.normal(0, 1, (n, 3)),
            max_pcs=3,
        )
        assert set(df.columns) == {"n_pcs", "lambda_gc", "delta_from_1", "recommended"}
        assert len(df) == 4  # 0, 1, 2, 3 PCs

    @patch("gwas.kinship._build_loco_kernels_impl")
    @patch("gwas.models._run_gwas_impl")
    def test_recommends_closest_to_one(self, mock_gwas, mock_loco):
        """closest_to_1 strategy picks the PC count with smallest delta_from_1."""
        rng = np.random.default_rng(1)
        n, p = 30, 100
        iid = np.array([f"S{i}" for i in range(n)])

        mock_loco.return_value = (None, {}, {})
        mock_gwas.side_effect = self._mock_run_gwas([1.5, 1.2, 1.01])

        df = auto_select_pcs(
            geno_imputed=rng.normal(0, 1, (n, p)),
            y=rng.normal(0, 1, n),
            sid=np.array([f"snp_{i}" for i in range(p)]),
            chroms=np.array(["1"] * p),
            chroms_num=np.ones(p, dtype=int),
            positions=np.arange(p) * 1000,
            iid=iid,
            Z_grm=rng.normal(0, 1, (n, p)),
            chroms_grm=np.array(["1"] * p),
            K_base=np.eye(n),
            pcs_full=rng.normal(0, 1, (n, 2)),
            max_pcs=2,
            strategy="closest_to_1",
        )
        best = df.loc[df["recommended"] == "★"]
        assert len(best) == 1
        # The recommended row must have the smallest delta_from_1
        best_delta = best.iloc[0]["delta_from_1"]
        assert best_delta == df["delta_from_1"].min()

    @patch("gwas.kinship._build_loco_kernels_impl")
    @patch("gwas.models._run_gwas_impl")
    def test_progress_callback_called(self, mock_gwas, mock_loco):
        """Progress callback is invoked for each PC count."""
        rng = np.random.default_rng(2)
        n, p = 30, 100
        iid = np.array([f"S{i}" for i in range(n)])

        mock_loco.return_value = (None, {}, {})
        mock_gwas.side_effect = self._mock_run_gwas([1.0, 1.0, 1.0])

        calls = []
        def _cb(k, total):
            calls.append((k, total))

        auto_select_pcs(
            geno_imputed=rng.normal(0, 1, (n, p)),
            y=rng.normal(0, 1, n),
            sid=np.array([f"snp_{i}" for i in range(p)]),
            chroms=np.array(["1"] * p),
            chroms_num=np.ones(p, dtype=int),
            positions=np.arange(p) * 1000,
            iid=iid,
            Z_grm=rng.normal(0, 1, (n, p)),
            chroms_grm=np.array(["1"] * p),
            K_base=np.eye(n),
            pcs_full=rng.normal(0, 1, (n, 2)),
            max_pcs=2,
            progress_callback=_cb,
        )
        assert len(calls) == 3  # k=0, 1, 2
        assert calls[0] == (0, 3)
        assert calls[2] == (2, 3)

    @patch("gwas.kinship._build_loco_kernels_impl")
    @patch("gwas.models._run_gwas_impl")
    def test_handles_gwas_failure(self, mock_gwas, mock_loco):
        """If GWAS raises for some PCs, those rows get NaN lambda."""
        rng = np.random.default_rng(3)
        n, p = 30, 100
        iid = np.array([f"S{i}" for i in range(n)])

        mock_loco.return_value = (None, {}, {})
        call_count = [0]
        def _fail_on_1(**kwargs):
            k = call_count[0]
            call_count[0] += 1
            if k == 1:
                raise RuntimeError("Simulated failure")
            return _make_fake_gwas_df(n_snps=500, seed=k)
        mock_gwas.side_effect = _fail_on_1

        df = auto_select_pcs(
            geno_imputed=rng.normal(0, 1, (n, p)),
            y=rng.normal(0, 1, n),
            sid=np.array([f"snp_{i}" for i in range(p)]),
            chroms=np.array(["1"] * p),
            chroms_num=np.ones(p, dtype=int),
            positions=np.arange(p) * 1000,
            iid=iid,
            Z_grm=rng.normal(0, 1, (n, p)),
            chroms_grm=np.array(["1"] * p),
            K_base=np.eye(n),
            pcs_full=rng.normal(0, 1, (n, 2)),
            max_pcs=2,
        )
        assert np.isnan(df.loc[1, "lambda_gc"])
        assert df.loc[0, "lambda_gc"] > 0  # k=0 succeeded

    @patch("gwas.plotting.compute_lambda_gc")
    @patch("gwas.kinship._build_loco_kernels_impl")
    @patch("gwas.models._run_gwas_impl")
    def test_closest_to_1_strategy(self, mock_gwas, mock_loco, mock_lambda):
        """closest_to_1 strategy picks absolute minimum delta regardless of trend."""
        rng = np.random.default_rng(11)
        n, p = 30, 100
        iid = np.array([f"S{i}" for i in range(n)])

        mock_loco.return_value = (None, {}, {})
        mock_gwas.return_value = _make_fake_gwas_df(n_snps=500)
        # λGC: 1.3, 1.05, 1.02, 0.995 — closest_to_1 should pick k=3 (0.995)
        mock_lambda.side_effect = [1.3, 1.05, 1.02, 0.995]

        df = auto_select_pcs(
            geno_imputed=rng.normal(0, 1, (n, p)),
            y=rng.normal(0, 1, n),
            sid=np.array([f"snp_{i}" for i in range(p)]),
            chroms=np.array(["1"] * p),
            chroms_num=np.ones(p, dtype=int),
            positions=np.arange(p) * 1000,
            iid=iid,
            Z_grm=rng.normal(0, 1, (n, p)),
            chroms_grm=np.array(["1"] * p),
            K_base=np.eye(n),
            pcs_full=rng.normal(0, 1, (n, 3)),
            max_pcs=3,
            strategy="closest_to_1",
        )
        best = df.loc[df["recommended"] == "★"]
        assert len(best) == 1
        assert best.iloc[0]["n_pcs"] == 3

    # ── Band strategy tests ─────────────────────────────────────

    def _band_helper(self, mock_gwas, mock_loco, mock_lambda, lambda_values,
                     max_pcs=None, **kwargs):
        """Run auto_select_pcs with band strategy and mocked lambda values."""
        rng = np.random.default_rng(42)
        n, p = 30, 100
        iid = np.array([f"S{i}" for i in range(n)])
        if max_pcs is None:
            max_pcs = len(lambda_values) - 1

        mock_loco.return_value = (None, {}, {})
        mock_gwas.return_value = _make_fake_gwas_df(n_snps=500)
        mock_lambda.side_effect = list(lambda_values)

        return auto_select_pcs(
            geno_imputed=rng.normal(0, 1, (n, p)),
            y=rng.normal(0, 1, n),
            sid=np.array([f"snp_{i}" for i in range(p)]),
            chroms=np.array(["1"] * p),
            chroms_num=np.ones(p, dtype=int),
            positions=np.arange(p) * 1000,
            iid=iid,
            Z_grm=rng.normal(0, 1, (n, p)),
            chroms_grm=np.array(["1"] * p),
            K_base=np.eye(n),
            pcs_full=rng.normal(0, 1, (n, max_pcs)),
            max_pcs=max_pcs,
            strategy="band",
            **kwargs,
        )

    @patch("gwas.plotting.compute_lambda_gc")
    @patch("gwas.kinship._build_loco_kernels_impl")
    @patch("gwas.models._run_gwas_impl")
    def test_band_picks_smallest_in_band(self, mock_gwas, mock_loco, mock_lambda):
        """Inflation case: λGC drops into [0.95, 1.05] at k=2."""
        # λGC: 1.25, 1.10, 1.03, 0.99 → in-band: k=2, k=3 → smallest k=2
        df = self._band_helper(mock_gwas, mock_loco, mock_lambda,
                               [1.25, 1.10, 1.03, 0.99])
        best = df.loc[df["recommended"] == "★"]
        assert len(best) == 1
        assert best.iloc[0]["n_pcs"] == 2

    @patch("gwas.plotting.compute_lambda_gc")
    @patch("gwas.kinship._build_loco_kernels_impl")
    @patch("gwas.models._run_gwas_impl")
    def test_band_handles_deflation(self, mock_gwas, mock_loco, mock_lambda):
        """Deflation case: λGC starts below 0.95, rises into band.
        Guard caps at trough (k=0) → picks k=0."""
        # λGC: 0.80, 0.91, 0.99, 1.02 → in-band: k=2, k=3
        # But lam0=0.80 < 0.95 → guard caps at k=0 (trough)
        df = self._band_helper(mock_gwas, mock_loco, mock_lambda,
                               [0.80, 0.91, 0.99, 1.02])
        best = df.loc[df["recommended"] == "★"]
        assert len(best) == 1
        assert best.iloc[0]["n_pcs"] == 0

    @patch("gwas.plotting.compute_lambda_gc")
    @patch("gwas.kinship._build_loco_kernels_impl")
    @patch("gwas.models._run_gwas_impl")
    def test_band_already_calibrated(self, mock_gwas, mock_loco, mock_lambda):
        """All k values in band, picks k=0 (most parsimonious)."""
        # λGC: 1.02, 1.01, 0.99, 0.98 → all in band → smallest k=0
        df = self._band_helper(mock_gwas, mock_loco, mock_lambda,
                               [1.02, 1.01, 0.99, 0.98])
        best = df.loc[df["recommended"] == "★"]
        assert len(best) == 1
        assert best.iloc[0]["n_pcs"] == 0

    @patch("gwas.plotting.compute_lambda_gc")
    @patch("gwas.kinship._build_loco_kernels_impl")
    @patch("gwas.models._run_gwas_impl")
    def test_band_fallback_strong_signal(self, mock_gwas, mock_loco, mock_lambda):
        """No k in band (strong QTL deflation), fallback picks k=0."""
        # λGC: 0.72, 0.73, 0.73, 0.74, 0.75
        # deltas: 0.28, 0.27, 0.27, 0.26, 0.25
        # best_delta=0.25, tol=max(0.02, 0.25*0.15)=0.0375
        # zone: delta <= 0.2875 → all qualify → smallest k=0
        df = self._band_helper(mock_gwas, mock_loco, mock_lambda,
                               [0.72, 0.73, 0.73, 0.74, 0.75])
        best = df.loc[df["recommended"] == "★"]
        assert len(best) == 1
        assert best.iloc[0]["n_pcs"] == 0

    @patch("gwas.plotting.compute_lambda_gc")
    @patch("gwas.kinship._build_loco_kernels_impl")
    @patch("gwas.models._run_gwas_impl")
    def test_band_fallback_all_inflated(self, mock_gwas, mock_loco, mock_lambda):
        """All λGC > 1.05, fallback picks closest with parsimony."""
        # λGC: 1.50, 1.30, 1.10, 1.08
        # deltas: 0.50, 0.30, 0.10, 0.08
        # best_delta=0.08, tol=max(0.02, 0.08*0.15)=0.02
        # zone: delta <= 0.10 → k=2 (0.10), k=3 (0.08) → smallest k=2
        df = self._band_helper(mock_gwas, mock_loco, mock_lambda,
                               [1.50, 1.30, 1.10, 1.08])
        best = df.loc[df["recommended"] == "★"]
        assert len(best) == 1
        assert best.iloc[0]["n_pcs"] == 2

    @patch("gwas.plotting.compute_lambda_gc")
    @patch("gwas.kinship._build_loco_kernels_impl")
    @patch("gwas.models._run_gwas_impl")
    def test_band_with_nan(self, mock_gwas, mock_loco, mock_lambda):
        """GWAS failure at k=0, band strategy skips NaN and picks from band."""
        rng = np.random.default_rng(42)
        n, p = 30, 100
        iid = np.array([f"S{i}" for i in range(n)])

        mock_loco.return_value = (None, {}, {})
        # k=0 fails, k=1,2,3 succeed
        call_count = [0]
        def _fail_on_0(**kwargs):
            k = call_count[0]
            call_count[0] += 1
            if k == 0:
                raise RuntimeError("Simulated failure")
            return _make_fake_gwas_df(n_snps=500, seed=k)
        mock_gwas.side_effect = _fail_on_0
        # Only 3 lambda calls (k=0 fails before compute_lambda_gc)
        mock_lambda.side_effect = [1.10, 1.02, 0.99]

        df = auto_select_pcs(
            geno_imputed=rng.normal(0, 1, (n, p)),
            y=rng.normal(0, 1, n),
            sid=np.array([f"snp_{i}" for i in range(p)]),
            chroms=np.array(["1"] * p),
            chroms_num=np.ones(p, dtype=int),
            positions=np.arange(p) * 1000,
            iid=iid,
            Z_grm=rng.normal(0, 1, (n, p)),
            chroms_grm=np.array(["1"] * p),
            K_base=np.eye(n),
            pcs_full=rng.normal(0, 1, (n, 3)),
            max_pcs=3,
            strategy="band",
        )
        assert np.isnan(df.loc[0, "lambda_gc"])
        best = df.loc[df["recommended"] == "★"]
        assert len(best) == 1
        # In-band: k=2 (1.02), k=3 (0.99) → smallest k=2
        assert best.iloc[0]["n_pcs"] == 2

    # ── Deflation guard tests ─────────────────────────────────

    @patch("gwas.plotting.compute_lambda_gc")
    @patch("gwas.kinship._build_loco_kernels_impl")
    @patch("gwas.models._run_gwas_impl")
    def test_band_deflation_guard_triggers(self, mock_gwas, mock_loco,
                                           mock_lambda):
        """Deflated baseline, PCs push λ up into band → guard caps at k=0."""
        # λGC: 0.75, 0.80, 0.85, 0.90, 0.96, 0.98
        # Without guard: in-band k=4 (0.96), k=5 (0.98) → picks k=4
        # Guard: lam0=0.75 < 0.95, trough=k=0, rec k=4 > 0 → re-select
        # within k≤0 → only k=0 (0.75) → fallback picks k=0
        df = self._band_helper(mock_gwas, mock_loco, mock_lambda,
                               [0.75, 0.80, 0.85, 0.90, 0.96, 0.98])
        best = df.loc[df["recommended"] == "★"]
        assert len(best) == 1
        assert best.iloc[0]["n_pcs"] == 0

    @patch("gwas.plotting.compute_lambda_gc")
    @patch("gwas.kinship._build_loco_kernels_impl")
    @patch("gwas.models._run_gwas_impl")
    def test_band_deflation_guard_fallback(self, mock_gwas, mock_loco,
                                           mock_lambda):
        """Deflated baseline, PCs push λ up but never reach band → guard
        overrides fallback from k=5 to k=0."""
        # λGC: 0.75, 0.78, 0.82, 0.85, 0.88, 0.90
        # Without guard: no in-band, fallback closest-to-1 picks k=5 (0.90)
        # Guard: lam0=0.75 < 0.95, trough=k=0, rec k=5 > 0 → re-select
        # within k≤0 → k=0 (0.75) → picks k=0
        df = self._band_helper(mock_gwas, mock_loco, mock_lambda,
                               [0.75, 0.78, 0.82, 0.85, 0.88, 0.90])
        best = df.loc[df["recommended"] == "★"]
        assert len(best) == 1
        assert best.iloc[0]["n_pcs"] == 0

    @patch("gwas.plotting.compute_lambda_gc")
    @patch("gwas.kinship._build_loco_kernels_impl")
    @patch("gwas.models._run_gwas_impl")
    def test_band_no_guard_when_pcs_decrease_lambda(self, mock_gwas,
                                                     mock_loco, mock_lambda):
        """Inflated baseline, PCs bring λ down → guard does NOT trigger."""
        # λGC: 1.20, 1.10, 1.02, 0.98, 0.96, 0.94
        # lam0=1.20 >= 0.95 → guard never fires
        # In-band: k=2 (1.02), k=3 (0.98), k=4 (0.96) → picks k=2
        df = self._band_helper(mock_gwas, mock_loco, mock_lambda,
                               [1.20, 1.10, 1.02, 0.98, 0.96, 0.94])
        best = df.loc[df["recommended"] == "★"]
        assert len(best) == 1
        assert best.iloc[0]["n_pcs"] == 2

    @patch("gwas.plotting.compute_lambda_gc")
    @patch("gwas.kinship._build_loco_kernels_impl")
    @patch("gwas.models._run_gwas_impl")
    def test_band_no_guard_baseline_in_band(self, mock_gwas, mock_loco,
                                             mock_lambda):
        """Baseline already in band → guard does NOT trigger."""
        # λGC: 0.97, 0.98, 0.99, 1.00, 1.01, 1.02
        # lam0=0.97 >= 0.95 → guard never fires
        # All in band → picks k=0 (smallest)
        df = self._band_helper(mock_gwas, mock_loco, mock_lambda,
                               [0.97, 0.98, 0.99, 1.00, 1.01, 1.02])
        best = df.loc[df["recommended"] == "★"]
        assert len(best) == 1
        assert best.iloc[0]["n_pcs"] == 0

    @patch("gwas.plotting.compute_lambda_gc")
    @patch("gwas.kinship._build_loco_kernels_impl")
    @patch("gwas.models._run_gwas_impl")
    def test_band_deflation_guard_oscillating_lambda(
        self, mock_gwas, mock_loco, mock_lambda,
    ):
        """Deflated baseline + oscillating λ → guard must still force k=0.

        Regression for the FarmCPU auto-PC scan reported by the user:
        λ = [0.8954, 0.9940, 0.8845, 0.9746, 0.8981]. Without the fix,
        the band strategy picks k=1 (0.9940 is in-band) and the old
        trough-based guard aborts because trough=k=2 and the check
        `rec(1) > trough(2)` is False. The recovery at k=1 is noise,
        not a stable fix — λ dips back below band at k=2 and k=4.
        The guard must force k=0.
        """
        df = self._band_helper(
            mock_gwas, mock_loco, mock_lambda,
            [0.8954, 0.9940, 0.8845, 0.9746, 0.8981],
        )
        best = df.loc[df["recommended"] == "★"]
        assert len(best) == 1
        assert best.iloc[0]["n_pcs"] == 0

    # ── Per-model scan (MLMM, FarmCPU) ────────────────────────

    def test_invalid_model_raises(self):
        """Passing an unknown ``model`` string must raise ValueError."""
        rng = np.random.default_rng(0)
        n, p = 20, 40
        with pytest.raises(ValueError, match="model must be one of"):
            auto_select_pcs(
                geno_imputed=rng.normal(0, 1, (n, p)),
                y=rng.normal(0, 1, n),
                sid=np.array([f"snp_{i}" for i in range(p)]),
                chroms=np.array(["1"] * p),
                chroms_num=np.ones(p, dtype=int),
                positions=np.arange(p) * 1000,
                iid=np.array([f"S{i}" for i in range(n)]),
                Z_grm=rng.normal(0, 1, (n, p)),
                chroms_grm=np.array(["1"] * p),
                K_base=np.eye(n),
                pcs_full=rng.normal(0, 1, (n, 3)),
                max_pcs=3,
                model="bogus",
            )

    @patch("gwas.plotting.compute_lambda_gc")
    @patch("gwas.kinship._build_loco_kernels_impl")
    @patch("gwas.models.run_mlmm_research_grade_fast")
    def test_mlmm_happy_path(self, mock_mlmm, mock_loco, mock_lambda):
        """model='mlmm' uses MLMM runner and returns expected columns."""
        rng = np.random.default_rng(7)
        n, p = 30, 80
        iid = np.array([f"S{i}" for i in range(n)])

        mock_loco.return_value = (None, {}, {})
        # Fake MLMM result: (mlmm_df, cof_table) tuple
        mock_mlmm.return_value = (
            _make_fake_gwas_df(n_snps=300), pd.DataFrame(),
        )
        # λGC: 1.20, 1.05, 0.99, 1.00 → in-band k=1 (1.05), k=2 (0.99),
        # k=3 (1.00). Smallest in-band → picks k=1.
        mock_lambda.side_effect = [1.20, 1.05, 0.99, 1.00]

        df = auto_select_pcs(
            geno_imputed=rng.normal(0, 1, (n, p)),
            y=rng.normal(0, 1, n),
            sid=np.array([f"snp_{i}" for i in range(p)]),
            chroms=np.array(["1"] * p),
            chroms_num=np.ones(p, dtype=int),
            positions=np.arange(p) * 1000,
            iid=iid,
            Z_grm=rng.normal(0, 1, (n, p)),
            chroms_grm=np.array(["1"] * p),
            K_base=np.eye(n),
            pcs_full=rng.normal(0, 1, (n, 3)),
            max_pcs=3,
            strategy="band",
            model="mlmm",
        )
        # Expected DataFrame shape and columns
        assert set(df.columns) == {
            "n_pcs", "lambda_gc", "delta_from_1", "recommended"
        }
        assert len(df) == 4
        # Exactly one recommended row
        best = df.loc[df["recommended"] == "★"]
        assert len(best) == 1
        assert best.iloc[0]["n_pcs"] == 1
        # MLMM runner was called once per k (4 calls for k=0..3)
        assert mock_mlmm.call_count == 4

    @patch("gwas.plotting.compute_lambda_gc")
    @patch("gwas.kinship._build_loco_kernels_impl")
    @patch("gwas.models.run_farmcpu")
    def test_farmcpu_happy_path(self, mock_fc, mock_loco, mock_lambda):
        """model='farmcpu' uses FarmCPU runner and returns expected columns."""
        rng = np.random.default_rng(8)
        n, p = 30, 80
        iid = np.array([f"S{i}" for i in range(n)])

        mock_loco.return_value = (None, {}, {})
        # FarmCPU returns (farmcpu_df, pseudo_qtn_table, convergence_info)
        mock_fc.return_value = (
            _make_fake_gwas_df(n_snps=300), pd.DataFrame(), {},
        )
        # λGC: 1.30, 1.12, 1.02, 0.99 → in-band: k=2 (1.02), k=3 (0.99)
        # Smallest in-band → picks k=2.
        mock_lambda.side_effect = [1.30, 1.12, 1.02, 0.99]

        df = auto_select_pcs(
            geno_imputed=rng.normal(0, 1, (n, p)),
            y=rng.normal(0, 1, n),
            sid=np.array([f"snp_{i}" for i in range(p)]),
            chroms=np.array(["1"] * p),
            chroms_num=np.ones(p, dtype=int),
            positions=np.arange(p) * 1000,
            iid=iid,
            Z_grm=rng.normal(0, 1, (n, p)),
            chroms_grm=np.array(["1"] * p),
            K_base=np.eye(n),
            pcs_full=rng.normal(0, 1, (n, 3)),
            max_pcs=3,
            strategy="band",
            model="farmcpu",
            farmcpu_p_threshold=0.01,
            farmcpu_max_iterations=5,
        )
        assert set(df.columns) == {
            "n_pcs", "lambda_gc", "delta_from_1", "recommended"
        }
        best = df.loc[df["recommended"] == "★"]
        assert len(best) == 1
        assert best.iloc[0]["n_pcs"] == 2
        assert mock_fc.call_count == 4

    @patch("gwas.plotting.compute_lambda_gc")
    @patch("gwas.kinship._build_loco_kernels_impl")
    @patch("gwas.models.run_farmcpu")
    def test_farmcpu_deflation_guard_applies(self, mock_fc, mock_loco,
                                              mock_lambda):
        """FarmCPU path now gets the same deflation guard as MLM.

        Previously the inline FarmCPU scan in GWAS_analysis.py only used
        :func:`select_best_pc_from_lambdas` (which has its own guard), but
        the unified :func:`auto_select_pcs` must apply the guard for all
        models including FarmCPU.
        """
        rng = np.random.default_rng(9)
        n, p = 30, 80
        iid = np.array([f"S{i}" for i in range(n)])

        mock_loco.return_value = (None, {}, {})
        mock_fc.return_value = (
            _make_fake_gwas_df(n_snps=300), pd.DataFrame(), {},
        )
        # Deflated baseline: λGC starts below 0.95 and rises into band.
        # Without guard: in-band k=3 (0.96), k=4 (0.99) → would pick k=3.
        # Guard: lam0=0.78 < 0.95, trough=k=0 → cap candidates at k≤0 →
        # forces selection to k=0.
        mock_lambda.side_effect = [0.78, 0.85, 0.92, 0.96, 0.99]

        df = auto_select_pcs(
            geno_imputed=rng.normal(0, 1, (n, p)),
            y=rng.normal(0, 1, n),
            sid=np.array([f"snp_{i}" for i in range(p)]),
            chroms=np.array(["1"] * p),
            chroms_num=np.ones(p, dtype=int),
            positions=np.arange(p) * 1000,
            iid=iid,
            Z_grm=rng.normal(0, 1, (n, p)),
            chroms_grm=np.array(["1"] * p),
            K_base=np.eye(n),
            pcs_full=rng.normal(0, 1, (n, 4)),
            max_pcs=4,
            strategy="band",
            model="farmcpu",
        )
        best = df.loc[df["recommended"] == "★"]
        assert len(best) == 1
        assert best.iloc[0]["n_pcs"] == 0


# ── select_best_pc_from_lambdas ──────────────────────────────

class TestSelectBestPcFromLambdas:
    # NOTE: inputs use inflation-side lambdas (lam0 > 0.95) so these
    # tests exercise the band / fallback / parsimony logic *without*
    # triggering the directional deflation guard. Guard-specific
    # scenarios live in TestSelectBestPcFromLambdasGuard below.

    def test_band_picks_smallest_in_band(self):
        # k=0: 1.10 (out), k=1: 0.97 (in), k=2: 1.02 (in), k=3: 1.00 (in)
        assert select_best_pc_from_lambdas(
            [1.10, 0.97, 1.02, 1.00], strategy="band") == 1

    def test_band_fallback_closest(self):
        # Nothing in [0.95, 1.05] → fallback to closest with parsimony
        # deltas = [0.20, 0.15, 0.10, 0.08]; best_delta=0.08, tol=0.02
        # Only k=3 (delta 0.08) is strictly within best_delta + tol
        # (k=2 at delta 0.10 is excluded by floating-point rounding).
        assert select_best_pc_from_lambdas(
            [1.20, 1.15, 1.10, 1.08], strategy="band") == 3

    def test_closest_to_1(self):
        assert select_best_pc_from_lambdas(
            [0.80, 1.10, 0.99, 1.05], strategy="closest_to_1") == 2

    def test_all_nan_returns_zero(self):
        assert select_best_pc_from_lambdas(
            [float("nan"), float("nan")], strategy="band") == 0

    def test_band_parsimony_prefers_fewer_pcs(self):
        # k=0: 1.15, k=1: 1.11, k=2: 1.10 — all out of band (inflated)
        # deltas = [0.15, 0.11, 0.10]; best_delta=0.10, tol=0.02
        # near = indices with delta <= 0.12 → [1, 2] → picks k=1
        assert select_best_pc_from_lambdas(
            [1.15, 1.11, 1.10], strategy="band") == 1


class TestSelectBestPcFromLambdasGuard:
    """Directional deflation guard on the FarmCPU-facing helper.

    Mirrors the MLM guard tests in TestAutoSelectPcsBand. These are
    plain list-in / int-out tests — no mocking required.
    """

    def test_guard_triggers_and_pins_trough(self):
        # User-reported scenario: lam0 = 0.80 (< 0.95), first in-band
        # is k=2 (0.95). Guard must cap at k=0 (the trough).
        lambdas = [0.80, 0.88, 0.95, 1.00, 1.02, 1.03]
        assert select_best_pc_from_lambdas(
            lambdas, strategy="band") == 0

    def test_guard_honours_trough_below_first_in_band(self):
        # lam0 = 0.90 < 0.95 → guard fires; trough is at k=1 (0.82).
        # Without guard, first in-band would be k=3 (0.96). Guard caps
        # candidates to k in [0, 1] and the fallback band-then-parsimony
        # inside the cap picks k=0 (smallest k with minimal delta).
        lambdas = [0.90, 0.82, 0.88, 0.96, 1.00]
        assert select_best_pc_from_lambdas(
            lambdas, strategy="band") == 0

    def test_no_guard_when_baseline_in_band(self):
        # lam0 = 0.97 is already in band → guard must NOT fire.
        lambdas = [0.97, 0.99, 1.00, 1.01]
        assert select_best_pc_from_lambdas(
            lambdas, strategy="band") == 0

    def test_no_guard_when_baseline_inflated(self):
        # lam0 = 1.20 > band_hi → guard must NOT fire (inflation side;
        # PCs genuinely help bring lambda into band).
        lambdas = [1.20, 1.10, 1.02, 0.98]
        assert select_best_pc_from_lambdas(
            lambdas, strategy="band") == 2

    def test_closest_to_1_ignores_guard(self):
        # closest_to_1 strategy does not apply the guard, matching
        # auto_select_pcs behaviour for the non-band strategy.
        lambdas = [0.80, 0.88, 0.95, 1.00]
        assert select_best_pc_from_lambdas(
            lambdas, strategy="closest_to_1") == 3

    def test_guard_trough_already_selected_no_change(self):
        # Fallback already picks the trough (k=0) by parsimony → guard
        # sees best == trough_k and leaves selection untouched.
        lambdas = [0.72, 0.74, 0.76, 0.78]
        assert select_best_pc_from_lambdas(
            lambdas, strategy="band") == 0

    def test_guard_oscillating_lambda(self):
        """Regression for user-reported FarmCPU oscillation.

        λ = [0.8954, 0.9940, 0.8845, 0.9746, 0.8981]. The first
        in-band value is k=1, but λ dips back below 0.95 at k=2
        and k=4 — not a stable recovery. The guard must force k=0.
        """
        lambdas = [0.8954, 0.9940, 0.8845, 0.9746, 0.8981]
        assert select_best_pc_from_lambdas(
            lambdas, strategy="band") == 0
