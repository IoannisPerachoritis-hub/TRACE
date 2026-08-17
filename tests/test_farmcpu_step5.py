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
