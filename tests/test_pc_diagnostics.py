"""Tests for gwas.pc_diagnostics -- the R1.4 PC-selection diagnostics REPORT, never select.

The hard acceptance criterion (D-R1.4-FINAL / §2e): no criterion is labelled recommended/best/optimal/
suggested, and no code path sets a PC count from any criterion. These tests pin that, plus the arithmetic
(correlation-normalised Kaiser/MP; proportion-based cumulative-variance/broken-stick; MP labelled as an
asymptotic bound; Tracy-Widom a documented, unimplemented gap; parallel analysis opt-in).
"""
import numpy as np
import pytest

from gwas import pc_diagnostics as pcd


def _known_spectrum():
    return np.array([8.0, 4.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1])


class TestNeverSelects:
    def test_no_recommendation_label_anywhere(self):
        ct = pcd.pc_criteria_table(_known_spectrum(), n=50, m=1000)
        blob = " ".join(str(x).lower() for x in ct.to_numpy().ravel())
        blob += " " + " ".join(c.lower() for c in ct.columns)
        for bad in pcd._FORBIDDEN_LABELS:
            assert bad not in blob, f"criteria table must not use '{bad}'"

    def test_criteria_columns_are_informational_only(self):
        ct = pcd.pc_criteria_table(_known_spectrum(), n=50, m=1000)
        assert list(ct.columns) == ["criterion", "k_implied", "note"]
        assert not any(("recommend" in c) or ("select" in c) or ("best" in c) for c in ct.columns)

    def test_module_has_no_selector_functions(self):
        assert not hasattr(pcd, "auto_select_pcs")
        assert not hasattr(pcd, "select_best_pc_from_lambdas")

    def test_compute_returns_only_report(self):
        Z = np.random.default_rng(0).standard_normal((40, 200)).astype(np.float32)
        out = pcd.compute_pc_diagnostics(Z, n=40, m=200)
        assert set(out) == {"spectrum", "criteria", "meta"}
        assert out["meta"]["selects"] is False
        assert "n_pcs" not in out
        assert "recommended" not in list(out["criteria"].columns)


class TestArithmetic:
    def test_kaiser_uses_correlation_normalised_eigenvalues(self):
        m = 100
        ev = np.array([3.0] * 40 + [1.5] * 20 + [0.05] * 40)   # trace ~= 152 -> trace/m ~= 1.52
        ct = pcd.pc_criteria_table(ev, n=101, m=m).set_index("criterion")
        ev_c = ev * m / ev.sum()
        assert int((ev_c > 1.0).sum()) == int(ct.loc["Kaiser (eigenvalue > 1)", "k_implied"])
        # correlation-normalised (40) differs from raw ev>1 (60) -> the normalisation is applied
        assert int(ct.loc["Kaiser (eigenvalue > 1)", "k_implied"]) != int((ev > 1.0).sum())

    def test_cumvar_and_broken_stick_are_scale_invariant(self):
        ev = _known_spectrum()
        a = pcd.pc_criteria_table(ev, n=50, m=1000).set_index("criterion")
        b = pcd.pc_criteria_table(ev * 7.3, n=50, m=1000).set_index("criterion")
        for crit in ("Cumulative variance >= 20%", "Cumulative variance >= 50%", "Broken-stick"):
            assert a.loc[crit, "k_implied"] == b.loc[crit, "k_implied"]

    def test_mp_edge_labelled_not_a_significance_test(self):
        ct = pcd.pc_criteria_table(_known_spectrum(), n=50, m=1000).set_index("criterion")
        assert "not a significance test" in str(ct.loc["Marchenko-Pastur edge", "note"]).lower()

    def test_tracy_widom_is_unimplemented_gap(self):
        ct = pcd.pc_criteria_table(_known_spectrum(), n=50, m=1000).set_index("criterion")
        assert ct.loc["Tracy-Widom", "k_implied"] is None
        assert "not implemented" in str(ct.loc["Tracy-Widom", "note"]).lower()

    def test_spectrum_table_pct_and_cumulative(self):
        ev = _known_spectrum()
        sp = pcd.pc_spectrum_table(ev, depth=5)
        assert len(sp) == 5
        assert abs(sp["pct_of_trace"].iloc[0] - 100 * ev[0] / ev.sum()) < 1e-4
        assert sp["cumulative_pct"].is_monotonic_increasing

    def test_parallel_analysis_is_opt_in(self):
        Z = np.random.default_rng(1).standard_normal((30, 100)).astype(np.float32)
        off = pcd.compute_pc_diagnostics(Z, n=30, m=100, run_parallel_analysis=False)
        assert off["criteria"].set_index("criterion").loc["Parallel analysis (Horn)", "k_implied"] is None
        on = pcd.compute_pc_diagnostics(Z, n=30, m=100, run_parallel_analysis=True, pa_B=20, pa_seed=0)
        assert on["criteria"].set_index("criterion").loc["Parallel analysis (Horn)", "k_implied"] is not None
        assert on["meta"]["parallel_analysis"] == {"B": 20, "seed": 0, "quantile": 95.0}


def test_reproduces_part_a_correlation_normalised_numbers():
    """The correlation-normalised criteria must match the R1.4 Part A eliminated-rules table."""
    import json
    from pathlib import Path
    jp = Path(__file__).resolve().parents[1] / "docs/revision/measurements/r14_pc_selection.json"
    if not jp.exists():
        pytest.skip("r14_pc_selection.json not present")
    d = json.loads(jp.read_text())
    expect = {"tomato/full/FULL": {"Kaiser (eigenvalue > 1)": 158, "Marchenko-Pastur edge": 155,
                                   "Cumulative variance >= 50%": 9, "Broken-stick": 11},
              "pepper/full/FULL": {"Kaiser (eigenvalue > 1)": 311, "Marchenko-Pastur edge": 268,
                                   "Cumulative variance >= 50%": 44, "Broken-stick": 13}}
    for e in d["matrices"]:
        if e["tag"] in expect:
            ct = pcd.pc_criteria_table(np.array(e["ev_full"]), n=e["n"], m=e["m_var"]).set_index("criterion")
            for crit, k in expect[e["tag"]].items():
                assert int(ct.loc[crit, "k_implied"]) == k, (e["tag"], crit, ct.loc[crit, "k_implied"], k)
