"""LD-kNNi imputation (Money et al. 2015) -- gwas/impute.py.

Covers the algorithm's load-bearing properties (distance normalisation by n, the
constant c, inverse-distance-weighted voting, N excluding target-missing samples,
discrete output, map-independence) and the type-(b) guarantee that the default
mean path is byte-identical.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from gwas.impute import (ld_knni, imputation_selfcheck, _ld_distances, _vote,
                         _ld_rank_topl, _mode_or_zero, empty_topl_fraction)
from gwas.qc import _pipeline_build_geno_matrices


# ── distance (Eq. 3) ────────────────────────────────────────────────────────
def test_distance_normalised_by_n_not_l():
    # two candidates over l=4 predictors; pair A shares 4 SNPs, pair B shares 2.
    a = np.array([[0.0, 0.0, 0.0, 0.0]])                 # the query
    b = np.array([[2.0, 2.0, 2.0, 2.0],                  # cand 0: 4 co-obs, Σ|.|=8
                  [2.0, 2.0, np.nan, np.nan]])           # cand 1: 2 co-obs, Σ|.|=4
    d = _ld_distances(a, b, c=1.0)[0]
    # normalise by each pair's own n (4 and 2), NOT by l=4:
    assert d[0] == pytest.approx(1.0 + 8.0 / 4.0)        # 3.0
    assert d[1] == pytest.approx(1.0 + 4.0 / 2.0)        # 3.0  (would be 2.0 if /l)
    # if it wrongly normalised by l, cand 1 would be 1 + 4/4 = 2.0 -- distinct
    assert d[1] != pytest.approx(1.0 + 4.0 / 4.0)


def test_constant_c_no_zero_distance():
    # genotypically identical samples over the predictors -> Σ|.|=0, but d=c=1, never 0
    a = np.array([[1.0, 2.0, 0.0]])
    d = _ld_distances(a, a, c=1.0)
    assert d[0, 0] == pytest.approx(1.0) and d.min() > 0.0


def test_no_co_observed_is_infinite():
    a = np.array([[0.0, np.nan]])
    b = np.array([[np.nan, 1.0]])                        # no shared observed column
    assert np.isinf(_ld_distances(a, b)[0, 0])


# ── voting (Eq. 2) ──────────────────────────────────────────────────────────
def test_voting_is_inverse_distance_weighted():
    # one near neighbour (d=1, class 2) outvotes two distant ones (d=10, class 0):
    #   weight(2) = 1/1 = 1.0  >  weight(0) = 2 * 1/10 = 0.2
    votes = np.array([0.2, 0.0, 1.0])
    assert _vote(votes, np.array([10., 10., 1.]), np.array([0, 0, 2])) == 2.0
    # unweighted majority would pick class 0 (2 votes vs 1) -- the weighting flips it


def test_tiebreak_smallest_summed_distance_then_lowest_class():
    # classes 0 and 2 tie on weight; class 2's neighbour is nearer -> smaller summed d
    votes = np.array([0.5, 0.0, 0.5])
    assert _vote(votes, np.array([4.0, 2.0]), np.array([0, 2])) == 2.0
    # exact tie on both -> lowest class index
    assert _vote(np.array([0.5, 0.0, 0.5]), np.array([2.0, 2.0]), np.array([0, 2])) == 0.0


# ── ld_knni end-to-end properties ───────────────────────────────────────────
def _panel(seed=0, missing=0.10, n=40, blocks=6, per=5):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 3, (n, blocks)).astype(float)
    G = np.repeat(base, per, axis=1)                     # LD blocks (identical columns)
    Gm = G.copy()
    Gm[rng.random(G.shape) < missing] = np.nan
    return G, Gm


def test_output_is_discrete_and_nan_free():
    _, Gm = _panel()
    out = ld_knni(Gm)
    assert not np.isnan(out).any()
    assert set(np.unique(out)).issubset({0.0, 1.0, 2.0})
    assert out.dtype == np.float32


def test_N_excludes_samples_missing_at_target():
    # target SNP col 0: only sample 0 is observed (class 2); everyone else missing there.
    # a strong LD predictor (col 1) makes sample 5 the nearest -- but sample 5 is
    # missing at the target, so it cannot vote; the only voter is sample 0 -> class 2.
    n = 10
    G = np.zeros((n, 2))
    G[:, 1] = np.arange(n) % 3                            # a predictor
    G[:, 0] = 2.0                                         # target truth (all class 2)
    Gm = G.copy()
    Gm[1:, 0] = np.nan                                    # only sample 0 observed at target
    out = ld_knni(Gm, k=5, l=1)
    assert np.all(out[:, 0] == 2.0)                       # every fill comes from the lone voter


def test_map_independence_permuting_snps_permutes_only_columns():
    _, Gm = _panel()
    rng = np.random.default_rng(3)
    perm = rng.permutation(Gm.shape[1])
    o1 = ld_knni(Gm)
    o2 = ld_knni(Gm[:, perm])
    assert np.array_equal(o1[:, perm], o2)               # no position/chromosome used


def test_ld_knni_signature_takes_no_position_or_chrom():
    import inspect
    params = set(inspect.signature(ld_knni).parameters)
    assert not (params & {"positions", "pos", "chroms", "chrom", "map", "chromosome"})


def test_complete_input_returned_unchanged():
    G, _ = _panel()
    out = ld_knni(G.astype(float))
    assert np.array_equal(out, G.astype(np.float32))


def test_all_missing_snp_handled_without_raising():
    G, Gm = _panel()
    Gm[:, 0] = np.nan                                    # a fully-missing SNP
    out = ld_knni(Gm)                                    # must not raise
    assert not np.isnan(out[:, 0]).any()
    assert set(np.unique(out[:, 0])).issubset({0.0})     # documented fallback: 0


def test_observed_entries_are_copied_unchanged():
    G, Gm = _panel()
    obs = ~np.isnan(Gm)
    out = ld_knni(Gm)
    assert np.array_equal(out[obs], Gm[obs].astype(np.float32))


# ── type-(b) guarantee: default mean path byte-identical ────────────────────
def test_default_mean_path_bit_identical():
    rng = np.random.default_rng(7)
    G = rng.integers(0, 3, (30, 40)).astype(float)
    G[rng.random(G.shape) < 0.1] = np.nan
    df = pd.DataFrame(G, index=[f"S{i}" for i in range(30)])
    _, _, imp_new, _ = _pipeline_build_geno_matrices(df, method="mean")
    imp_old = SimpleImputer(strategy="mean").fit_transform(df).astype("float32")
    assert np.array_equal(imp_new, imp_old) and imp_new.dtype == np.float32


def test_ldknni_path_preserves_raw_nan():
    rng = np.random.default_rng(9)
    G = rng.integers(0, 3, (30, 40)).astype(float)
    G[rng.random(G.shape) < 0.1] = np.nan
    df = pd.DataFrame(G, index=[f"S{i}" for i in range(30)])
    raw, rate, imputed, _ = _pipeline_build_geno_matrices(df, method="ldknni")
    assert np.isnan(raw).any()                           # geno_dosage_raw untouched
    assert not np.isnan(imputed).any()                   # geno_imputed fully filled
    expected_rate = df.isna().mean(axis=0).values.astype(np.float32)
    np.testing.assert_array_almost_equal(rate, expected_rate)  # rate = missingness, not method


# ── self-check ──────────────────────────────────────────────────────────────
def test_selfcheck_reports_both_methods():
    _, Gm = _panel(missing=0.08)
    out = imputation_selfcheck(Gm, n_mask=50, chroms=np.repeat(np.arange(6), 5))
    for m in ("mean", "ldknni"):
        assert 0.0 <= out[m]["discrete_concordance"] <= 1.0
    assert out["winner_concordance"] in ("mean", "ldknni")
    assert "rounded" in out["note"]                      # states the mean-rounding convention


def test_empty_topl_fraction_100pct_below_min_pair_n():
    # n < min_pair_n=20 -> no pair reaches the co-observed floor -> every SNP empty.
    rng = np.random.default_rng(3)
    G = rng.integers(0, 3, (18, 120)).astype(float)
    G[rng.random(G.shape) < 0.05] = np.nan
    frac, scanned = empty_topl_fraction(G, min_pair_n=20)
    assert frac == 1.0 and scanned == 120


def test_empty_topl_fraction_target_idx_and_subsample():
    rng = np.random.default_rng(4)
    G = rng.integers(0, 3, (40, 500)).astype(float)          # n >= min_pair_n
    G[rng.random(G.shape) < 0.05] = np.nan
    frac, scanned = empty_topl_fraction(G, target_idx=np.arange(300), max_targets=100)
    assert 0.0 <= frac <= 1.0 and scanned == 100             # subsample cap honoured
    assert empty_topl_fraction(G, target_idx=np.array([], dtype=int)) == (0.0, 0)
