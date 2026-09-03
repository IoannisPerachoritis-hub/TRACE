"""Tests for gwas.pc_diagnostics -- REPORT the genotype-PCA eigenvalue spectrum ONLY; never select, never recommend.

R1.4: the conventional-criteria table (Kaiser / Marchenko-Pastur / broken-stick / parallel-analysis / Tracy-Widom)
was removed; ``compute_pc_diagnostics`` returns only {"spectrum", "meta"}. These pin that the module reports a
spectrum, carries no selector/criteria machinery, and emits NO PC count.
"""
import numpy as np

from gwas import pc_diagnostics as pcd


def _known_spectrum():
    return np.array([8.0, 4.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1])


class TestNeverSelects:
    def test_module_has_no_selector_functions(self):
        assert not hasattr(pcd, "auto_select_pcs")
        assert not hasattr(pcd, "select_best_pc_from_lambdas")

    def test_module_has_no_criteria_functions(self):
        # R1.4: the criteria table + its helpers (Kaiser/MP/broken-stick/parallel-analysis) were removed.
        assert not hasattr(pcd, "pc_criteria_table")
        assert not hasattr(pcd, "parallel_analysis_k")
        assert not hasattr(pcd, "_broken_stick_k")

    def test_compute_returns_spectrum_and_meta_only(self):
        Z = np.random.default_rng(0).standard_normal((40, 200)).astype(np.float32)
        out = pcd.compute_pc_diagnostics(Z, n=40, m=200)
        assert set(out) == {"spectrum", "meta"}
        assert out["meta"]["selects"] is False
        assert "criteria" not in out

    def test_result_contains_no_pc_count(self):
        # Structural (§5): nothing in the returned object is (or names) a chosen PC count.
        Z = np.random.default_rng(1).standard_normal((40, 200)).astype(np.float32)
        out = pcd.compute_pc_diagnostics(Z, n=40, m=200)
        forbidden = {"n_pcs", "k", "k_implied", "recommended", "best", "optimal", "suggested", "selected"}
        assert forbidden.isdisjoint(out.keys())
        assert forbidden.isdisjoint(out["meta"].keys())
        # the spectrum reports description only; no integer decision column
        assert list(out["spectrum"].columns) == ["rank", "eigenvalue", "pct_of_trace", "cumulative_pct"]
        assert out["meta"].get("selects") is False


class TestSpectrum:
    def test_spectrum_table_pct_and_cumulative(self):
        ev = _known_spectrum()
        sp = pcd.pc_spectrum_table(ev, depth=5)
        assert len(sp) == 5
        assert abs(sp["pct_of_trace"].iloc[0] - 100 * ev[0] / ev.sum()) < 1e-4
        assert sp["cumulative_pct"].is_monotonic_increasing

    def test_spectrum_depth_caps_rows(self):
        ev = _known_spectrum()
        assert len(pcd.pc_spectrum_table(ev, depth=20)) == len(ev)  # depth caps at ev.size
        assert len(pcd.pc_spectrum_table(ev, depth=3)) == 3
