"""Tests for the published FarmCPU Step 5 pilot (``step5_reml_bins``).

Step 5 replaces TRACE's fixed 3-bin candidate selection with REML-optimised
``(bin_size x top_N)`` selection: for each grid point the min-p marker per
occupied bin is taken (no significance gate), ranked, the top_N kept, and a
random-effect model ``y = X + K + e`` (K = ZZ'/m) fitted by REML; the grid
point with the minimum REML deviance is selected.

The two tests here cover the NEW code directly:
  * the REML routine (fastlmm ``LMM.findH2``) MUST recover a true h2 of ~0.5 and
    return ~0 (median, the diagnostic) at true 0 -- the estimator is validated
    BEFORE it is used to select anything (a naive REML on the panel's duplicate
    genotypes once returned a spurious 0.89);
  * ``_step5_reml_bin_select`` returns a p-ranked pool bounded by ``top_N`` and
    records the chosen grid point.

Step5-off byte-identity is covered by the existing FarmCPU suite (the default
path is unchanged); end-to-end ``run_farmcpu(step5_reml_bins=True)`` is exercised
by the pilot measurement.
"""
import numpy as np
import pytest

from gwas import models


def test_reml_findh2_recovers_h2():
    """fastlmm LMM.findH2 (REML) recovers h2~0.5 and gives median~0 at true 0."""
    from fastlmm.inference.lmm import LMM

    rng = np.random.default_rng(42)
    n, m = 120, 40

    def _h2(y, Z):
        lmm = LMM()
        lmm.setG(G0=Z / np.sqrt(Z.shape[1]))
        lmm.setX(np.ones((n, 1)))
        lmm.sety(y)
        return float(lmm.findH2(REML=True)["h2"])

    hi, h0 = [], []
    for _ in range(15):
        Z = rng.standard_normal((n, m))
        Z = (Z - Z.mean(0)) / Z.std(0)
        g = Z @ rng.standard_normal(m)
        g = (g - g.mean()) / g.std()
        e = rng.standard_normal(n)
        hi.append(_h2(np.sqrt(0.5) * g + np.sqrt(0.5) * e, Z))
        h0.append(_h2(rng.standard_normal(n), Z))       # pure noise, true h2 = 0

    # true h2 = 0.5: the mean recovers it (n=120 -> some spread).
    assert 0.30 < np.mean(hi) < 0.70, f"h2=0.5 not recovered: {np.mean(hi):.3f}"
    # true h2 = 0: a bounded REML component cannot average 0; the MEDIAN is ~0.
    assert np.median(h0) < 0.10, f"h2=0 median not ~0: {np.median(h0):.3f}"


def test_step5_bin_select_bounded_and_ranked():
    """_step5_reml_bin_select returns a non-empty, top_N-bounded, p-ranked pool
    and records the chosen (bin_size, top_N) in diag."""
    rng = np.random.default_rng(7)
    n, m = 80, 300
    geno_std = rng.standard_normal((n, m))
    geno_std = (geno_std - geno_std.mean(0)) / geno_std.std(0)
    # markers on 3 chromosomes, 100 each, spaced 100 kb apart
    chroms = np.repeat(["1", "2", "3"], 100).astype(str)
    positions = np.tile(np.arange(100) * 100_000, 3).astype(int)
    # a handful of "signal" markers get tiny p-values; the rest ~ uniform
    pvals = rng.uniform(0.02, 1.0, size=m)
    signal = np.array([5, 40, 120, 205, 260])
    pvals[signal] = np.array([1e-8, 1e-7, 1e-6, 1e-6, 1e-5])
    # y correlated with the signal markers (so REML has something to fit)
    y = geno_std[:, signal] @ np.array([1.5, 1.2, 1.0, 0.8, 0.6]) + rng.standard_normal(n)
    X_fixed = np.ones((n, 1))

    diag = {}
    reps = models._step5_reml_bin_select(
        pvals, geno_std, chroms, positions, X_fixed, y,
        bin_sizes=(500_000, 5_000_000, 50_000_000),
        topn_grid=(10, 20, 30), exclude_idxs=None, diag=diag,
    )
    assert reps, "step5 returned an empty pool despite signal"
    assert len(reps) <= 30, "pool exceeds max(top_N)"
    assert len(reps) == diag["step5_top_n"], "pool size != recorded top_N"
    assert diag["step5_bin_size"] in (500_000, 5_000_000, 50_000_000)
    assert np.isfinite(diag["step5_nll"])
    # reps are p-ranked (ascending), and the strongest signal marker leads.
    assert list(reps) == sorted(reps, key=lambda j: pvals[j])
    assert reps[0] == signal[np.argmin(pvals[signal])]

    # exclude_idxs are honoured (the strongest marker is dropped from reps).
    diag2 = {}
    reps2 = models._step5_reml_bin_select(
        pvals, geno_std, chroms, positions, X_fixed, y,
        bin_sizes=(5_000_000,), topn_grid=(20,),
        exclude_idxs=[int(signal[0])], diag=diag2,
    )
    assert int(signal[0]) not in reps2


def test_step5_accept_bound_has_the_radical():
    """The Step-5 pseudo-QTN ceiling is sqrt(n/log10(n)), not n/log10(n)."""
    assert models._step5_accept_bound(165) == 9
    assert models._step5_accept_bound(166) == 9
    assert models._step5_accept_bound(350) == 12
    assert models._step5_accept_bound(100) == 7
    assert models._step5_accept_bound(1000) == 18
    assert models._step5_accept_bound(5000) == 37
    # it is the SQUARE ROOT of the old (buggy) un-rooted bound, not that bound.
    assert models._step5_accept_bound(165) != int(round(165 / np.log10(165)))  # != 74


def test_step5_clamp_grid_min_maps_never_empties():
    """The grid clamp uses MIN-MAPPING, not filtering: at a bound below the grid's
    minimum it collapses to (bound,), never to an empty grid (the silent-failure
    trap a later refactor would 'simplify' back in)."""
    grid = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    assert models._step5_clamp_grid(grid, 9) == (9,)       # filter -> ()
    assert models._step5_clamp_grid(grid, 7) == (7,)       # filter -> ()
    assert models._step5_clamp_grid(grid, 12) == (10, 12)
    assert models._step5_clamp_grid(grid, 18) == (10, 18)
    assert models._step5_clamp_grid(grid, 37) == (10, 20, 30, 37)
    for bound in (7, 9):                                    # the min-vs-filter proof
        assert len(models._step5_clamp_grid(grid, bound)) >= 1
        assert [s for s in grid if s <= bound] == []       # filter WOULD be empty


def test_step5_bound_param_clamps_and_holds_invariant():
    """_step5_reml_bin_select with an UNCLAMPED grid + bound=9 selects top_N<=9
    (never empty) and the invariant assertion holds -- guards the silent empty."""
    rng = np.random.default_rng(11)
    n, m = 165, 400
    geno_std = rng.standard_normal((n, m))
    geno_std = (geno_std - geno_std.mean(0)) / geno_std.std(0)
    chroms = np.repeat([str(c) for c in range(1, 5)], 100).astype(str)
    positions = np.tile(np.arange(100) * 100_000, 4).astype(int)
    pvals = rng.uniform(0.02, 1.0, size=m)
    signal = np.array([5, 40, 120, 205, 260, 310, 350])
    pvals[signal] = np.logspace(-9, -4, len(signal))
    y = geno_std[:, signal] @ rng.uniform(0.6, 1.5, len(signal)) + rng.standard_normal(n)
    X_fixed = np.ones((n, 1))

    diag = {}
    reps = models._step5_reml_bin_select(
        pvals, geno_std, chroms, positions, X_fixed, y,
        bin_sizes=(500_000, 5_000_000, 50_000_000),
        topn_grid=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100),   # UNclamped
        exclude_idxs=None, diag=diag, bound=9,                  # bound enforces it
    )
    assert reps, "clamped grid returned an EMPTY pool -- the silent failure mode"
    assert len(reps) <= 9, f"selected {len(reps)} > bound 9"
    assert diag["step5_top_n"] <= 9, f"step5_top_n {diag['step5_top_n']} > 9"
    assert diag["step5_bin_size"] in (500_000, 5_000_000, 50_000_000)


def test_shipped_farmcpu_is_e_f_nogate2_d101():
    """D-101: the shipped FarmCPU model IS E_F_nogate2 -- the faithful published
    Step 5 at the corrected pseudo-QTN bound sqrt(n/log10 n). Pin the six flipped
    run_farmcpu defaults so an accidental revert to the old A_F arm is caught."""
    import inspect
    d = {k: v.default for k, v in inspect.signature(models.run_farmcpu).parameters.items()}
    # the six flipped by D-101:
    assert d["pool_cap"] == 100, "pool_cap must ship at 100"
    assert d["step5_reml_bins"] is True, "shipped FarmCPU must run published Step 5"
    assert d["step5_substitution"] is True, "shipped FarmCPU must do Step 3 substitution"
    assert d["step5_skip_validation"] is True, "shipped FarmCPU must be gate-free (canonical)"
    assert d["step5_prune_in_loop"] is True, "shipped FarmCPU must prune in-loop (Step 6 inside)"
    assert d["step5_reselect"] is True, "shipped FarmCPU must re-select (canonical Step 7)"
    # D-102: final_scan now ships as the classical OLS fixed-effect scan
    # (the published FarmCPU end-to-end); MLM is the opt-in alternative.
    assert d["final_scan"] == "ols"
    # unchanged shipped defaults:
    assert d["selection_kinship"] == "global"
    assert d["carry_validated_set"] is False
