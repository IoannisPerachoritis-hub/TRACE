"""The kinship-aware haplotype test (gwas.haplotype_kinship), on synthetic data.

- The whitened F is the generalised least squares F (computed here with V^-1 itself).
- With K = I and h2 forced, the test is the PCs-only test: F to 1e-10, P exactly.
- At h2 = 0 (forced, or where the REML search lands) the block runs the PCs-only test's own
  functions: F and both p-values bit for bit.
- The permutations are the PCs-only test's (its generator and index sequence), so P is
  reproducible from the block's seed.
- The REML h2 is FaST-LMM's: LMM.findH2(REML=True) and single_snp's Nullh2, to 1e-6.
- Flipping the signs of the kernel's eigenvectors changes nothing.
- The kernel is never written to; it is restricted to the tested samples by ID.
"""
import numpy as np
import pytest
from scipy.stats import f as fdist

from gwas.haplotype import (_eta2_confint, _lead_snp_partial_r2, _omega2_from_f,
                            block_test_lm_with_pcs, freedman_lane_perm_pvalue)
from gwas.haplotype_kinship import (H2_MAX, KinshipTestError, kernel_for_samples,
                                    kinship_block_test, null_h2, permutation_indices, reml_h2,
                                    _orth)
from gwas.models import _one_hot_drop_first


class _Kernel:
    """A stand-in for pysnptools' KernelData: iid (n x 2) and val."""

    def __init__(self, ids, val):
        self.iid = np.c_[ids, ids]
        self.val = val


def _data(seed=11, n=200, m=400, h2=0.8, n_groups=3):
    """Genotypes, their kernel (mean diagonal 1, as TRACE builds it), two covariates,
    haplotype groups, and y = 1 + C b + g + e with g ~ N(0, h2 K). The defaults give an
    interior REML optimum (h2 about 0.78), so the whitened path is the one tested."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.1, 0.5, m)
    G = rng.binomial(2, p, size=(n, m)).astype(float)
    Z = (G - 2 * p) / np.sqrt(2 * p * (1 - p))
    K = Z @ Z.T / m
    K /= np.mean(np.diag(K))
    g = Z @ rng.normal(0.0, np.sqrt(h2 / m), m)
    C = rng.normal(size=(n, 2))
    groups = np.array([f"H{k + 1}" for k in rng.integers(0, n_groups, n)])
    y = 1.0 + C @ np.array([0.5, -0.3]) + g + rng.normal(0.0, np.sqrt(1 - h2), n)
    ids = np.array([f"S{i:03d}" for i in range(n)])
    Kr, _ = kernel_for_samples(_Kernel(ids, K.astype(np.float32)), ids)
    return dict(G=G, K=K, Kr=Kr, C=C, groups=groups, y=y, ids=ids, dose=G[:, 0])


def _same(a, b):
    """Two results equal value for value, exactly (NaN equal to NaN)."""
    assert a.keys() == b.keys()
    for k in a:
        assert a[k] == b[k] or (np.isnan(a[k]) and np.isnan(b[k])), (k, a[k], b[k])


def _gls_f(y, X0, H, V):
    """The nested GLS F of X0 against [X0, H] with covariance V, from V^-1 itself."""
    Vi = np.linalg.inv(V)

    def q(X):
        b = np.linalg.solve(X.T @ Vi @ X, X.T @ Vi @ y)
        r = y - X @ b
        return float(r @ Vi @ r)

    n = len(y)
    df0, df1 = n - X0.shape[1], n - X0.shape[1] - H.shape[1]
    return ((q(X0) - q(np.column_stack([X0, H]))) / (df0 - df1)) / (q(np.column_stack([X0, H])) / df1)


# ------------------------------------------------------------------------- the whitened F
@pytest.mark.parametrize("h2", [0.3, 0.8])
def test_the_whitened_f_is_the_gls_f(h2):
    d = _data()
    n = len(d["y"])
    res = kinship_block_test(d["y"], d["groups"], d["C"], d["Kr"], n_perm=9, seed=1, h2=h2)
    H, _, _ = _one_hot_drop_first(d["groups"])
    X0 = np.column_stack([np.ones(n), d["C"]])
    F = _gls_f(d["y"], X0, H, h2 * d["Kr"] + (1 - h2) * np.eye(n))
    assert res["h2"] == h2
    assert res["F_param"] == pytest.approx(F, rel=1e-9)
    assert res["F_perm"] == res["F_param"]
    assert (res["df1"], res["df2"]) == (H.shape[1], n - X0.shape[1] - H.shape[1])
    assert res["PValue_param"] == pytest.approx(fdist.sf(F, res["df1"], res["df2"]), rel=1e-7)
    # the effect sizes are those of the same F
    df1, df2 = res["df1"], res["df2"]
    assert res["eta2"] == pytest.approx(df1 * F / (df1 * F + df2), rel=1e-9)
    lo, hi = _eta2_confint(res["F_param"], df1, df2)
    assert (res["eta2_ci_low"], res["eta2_ci_high"]) == (lo, hi)
    assert res["omega2"] == _omega2_from_f(res["F_param"], df1, df2)


def test_the_lead_snp_r2_is_the_whitened_partial_r2():
    d = _data()
    n, h2 = len(d["y"]), 0.6
    res = kinship_block_test(d["y"], d["groups"], d["C"], d["Kr"], n_perm=9, seed=1, h2=h2,
                             lead_dose=d["dose"])
    S, U = np.linalg.eigh(h2 * d["Kr"] + (1 - h2) * np.eye(n))
    W = U @ np.diag(S ** -0.5) @ U.T
    Xs = W @ np.column_stack([np.ones(n), d["C"]])
    P = np.eye(n) - Xs @ np.linalg.pinv(Xs)
    e, g = P @ (W @ d["y"]), P @ (W @ d["dose"])
    assert res["lead_snp_r2"] == pytest.approx((e @ g) ** 2 / ((e @ e) * (g @ g)), rel=1e-9)
    # monomorphic on the samples, or missing: no value
    flat = kinship_block_test(d["y"], d["groups"], d["C"], d["Kr"], n_perm=9, seed=1, h2=h2,
                              lead_dose=np.ones(n))
    assert np.isnan(flat["lead_snp_r2"])
    assert np.isnan(kinship_block_test(d["y"], d["groups"], d["C"], d["Kr"], n_perm=9, seed=1,
                                       h2=h2)["lead_snp_r2"])


# ---------------------------------------------------------- K = I, and h2 = 0: the PCs test
@pytest.mark.parametrize("with_covariates", [False, True])
def test_with_k_identity_and_h2_forced_it_is_the_pcs_test(with_covariates):
    """The REML likelihood is flat when K = I (every h2 gives V = I), so h2 is forced: the
    whitening is then the identity and the test is the PCs-only test."""
    d = _data()
    C = d["C"] if with_covariates else None
    n = len(d["y"])
    res = kinship_block_test(d["y"], d["groups"], C, np.eye(n), n_perm=199, seed=5, h2=0.5)
    F, p, df1, df2 = block_test_lm_with_pcs(d["y"], d["groups"], pcs=C)
    F_obs, p_perm = freedman_lane_perm_pvalue(d["y"], d["groups"], pcs=C, n_perm=199, seed=5)
    assert res["F_param"] == pytest.approx(F, rel=1e-10, abs=0)
    assert res["F_perm"] == pytest.approx(F_obs, rel=1e-10, abs=0)
    assert res["PValue_param"] == pytest.approx(p, rel=1e-8, abs=0)
    assert res["P_perm"] == p_perm
    assert (res["df1"], res["df2"]) == (df1, df2)


@pytest.mark.parametrize("with_covariates", [False, True])
def test_at_h2_zero_the_pcs_tests_own_functions_run(with_covariates):
    d = _data()
    C = d["C"] if with_covariates else None
    res = kinship_block_test(d["y"], d["groups"], C, d["Kr"], n_perm=199, seed=5, h2=0.0)
    F, p, df1, df2 = block_test_lm_with_pcs(d["y"], d["groups"], pcs=C)
    F_obs, p_perm = freedman_lane_perm_pvalue(d["y"], d["groups"], pcs=C, n_perm=199, seed=5)
    assert res["h2"] == 0.0
    assert (res["F_param"], res["PValue_param"], res["df1"], res["df2"]) == (F, p, df1, df2)
    assert (res["F_perm"], res["P_perm"]) == (F_obs, p_perm)


def test_the_reml_search_reaches_h2_zero_exactly():
    """A phenotype whose variance lies along the kernel's smallest eigenvalues has its REML
    optimum at the bound: the search lands on 0 itself (a grid point; the best point
    evaluated is kept), and the block is tested by the PCs-only test's functions."""
    d = _data()
    n = len(d["y"])
    S, U = np.linalg.eigh(d["Kr"])
    rng = np.random.default_rng(2)
    y = 2.0 + U[:, :60] @ rng.normal(0.0, 3.0, 60) + rng.normal(0.0, 0.1, n)
    Q = _orth(np.column_stack([np.ones(n), d["C"]]))
    assert reml_h2(S, U, y, Q)[0] == 0.0
    res = kinship_block_test(y, d["groups"], d["C"], d["Kr"], n_perm=99, seed=3)
    F, p, _, _ = block_test_lm_with_pcs(y, d["groups"], pcs=d["C"])
    F_obs, p_perm = freedman_lane_perm_pvalue(y, d["groups"], pcs=d["C"], n_perm=99, seed=3)
    assert res["h2"] == 0.0
    assert (res["F_param"], res["PValue_param"], res["F_perm"], res["P_perm"]) == \
        (F, p, F_obs, p_perm)
    # the lead SNP's r2 is then today's partial r2 (the same quantity, computed apart)
    res = kinship_block_test(y, d["groups"], d["C"], d["Kr"], n_perm=9, seed=3,
                             lead_dose=d["dose"])
    X = np.column_stack([np.ones(n), d["C"]])
    resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    today = _lead_snp_partial_r2("s0", np.array(["s0"]), d["dose"].reshape(-1, 1),
                                 d["ids"], d["ids"], resid, X)
    assert res["lead_snp_r2"] == pytest.approx(today, rel=1e-12)


# ------------------------------------------------------------------------ the permutations
def test_the_permutations_are_the_pcs_tests_and_follow_the_seed():
    idx = permutation_indices(30, 50, seed=123)
    rng, running = np.random.default_rng(123), np.arange(30)
    for b in range(50):
        rng.shuffle(running)
        assert np.array_equal(idx[b], running), b
    d = _data()
    a = kinship_block_test(d["y"], d["groups"], d["C"], d["Kr"], n_perm=199, seed=77)
    b = kinship_block_test(d["y"], d["groups"], d["C"], d["Kr"], n_perm=199, seed=77)
    assert 0 < a["h2"] < H2_MAX                     # the whitened path
    _same(a, b)
    c = kinship_block_test(d["y"], d["groups"], d["C"], d["Kr"], n_perm=199, seed=78)
    assert c["F_param"] == a["F_param"] and c["h2"] == a["h2"]


# ---------------------------------------------------------------------------- the REML h2
def test_the_reml_h2_is_fastlmms():
    """h2 equals FaST-LMM's REML h2 on the same y, X and K: LMM.findH2(REML=True), and the
    Nullh2 single_snp reports (an independent code path, lmm_cov), to 1e-6."""
    from fastlmm.association import single_snp
    from fastlmm.inference.lmm import LMM
    from pysnptools.kernelreader import KernelData
    from pysnptools.snpreader import SnpData

    from gwas.utils import CovarData, PhenoData

    d = _data(seed=3, n=150, h2=0.6)
    n = len(d["y"])
    ours = null_h2(d["y"], d["C"], d["Kr"])
    assert 0.05 < ours < H2_MAX - 0.05, ours           # an interior optimum
    lmm = LMM()
    lmm.setK(d["Kr"].copy())
    lmm.setX(np.column_stack([np.ones(n), d["C"]]))
    lmm.sety(d["y"].copy())
    h_lmm = float(lmm.findH2(REML=True)["h2"])
    iid = np.c_[d["ids"], d["ids"]]
    snps = SnpData(iid=iid, sid=np.array(["a", "b"]), val=d["G"][:, :2].copy(),
                   pos=np.array([[1, 0, 1], [1, 0, 2]], dtype=float))
    res = single_snp(test_snps=snps, pheno=PhenoData(d["ids"], d["y"]),
                     K0=KernelData(iid=iid, val=d["Kr"].copy()),
                     covar=CovarData(d["ids"], d["C"]), leave_out_one_chrom=False)
    h_ss = float(res["Nullh2"].iloc[0])
    assert abs(ours - h_lmm) < 1e-6, (ours, h_lmm)
    assert abs(ours - h_ss) < 1e-6, (ours, h_ss)
    # and the test's own h2 is it
    assert kinship_block_test(d["y"], d["groups"], d["C"], d["Kr"], n_perm=9, seed=1)["h2"] == ours


# ----------------------------------------------------------------------------- invariance
def test_flipping_the_eigenvectors_signs_changes_nothing():
    d = _data()
    S, U = np.linalg.eigh(d["Kr"])
    flip = np.where(np.random.default_rng(9).random(S.size) < 0.5, -1.0, 1.0)
    a = kinship_block_test(d["y"], d["groups"], d["C"], eig=(S, U), n_perm=199, seed=4,
                           lead_dose=d["dose"])
    b = kinship_block_test(d["y"], d["groups"], d["C"], eig=(S, U * flip), n_perm=199, seed=4,
                           lead_dose=d["dose"])
    assert a["h2"] > 0
    _same(a, b)


# ------------------------------------------------------------------------------- the kernel
def test_the_kernel_is_restricted_by_id_and_never_written():
    d = _data()
    n = len(d["ids"])
    val = d["K"].astype(np.float32)
    val.setflags(write=False)                       # a write would raise
    before = val.copy()
    kern = _Kernel(d["ids"], val)
    pick = d["ids"][::-1][:70]                       # another order, a subset
    Kr, pos = kernel_for_samples(kern, pick)
    assert Kr.dtype == np.float64 and Kr.shape == (70, 70)
    assert np.array_equal(pos, np.arange(n - 1, n - 71, -1))
    sub = val[np.ix_(pos, pos)].astype(np.float64)
    assert np.allclose(Kr, sub * (70 / np.trace(sub)), rtol=1e-15, atol=0)
    assert np.trace(Kr) == pytest.approx(70.0, rel=1e-14)
    res = kinship_block_test(d["y"][pos], d["groups"][pos], d["C"][pos], Kr, n_perm=19, seed=2)
    assert np.isfinite(res["F_param"])
    assert np.array_equal(val, before)
    Kc = Kr.copy()
    kinship_block_test(d["y"][pos], d["groups"][pos], None, Kr, n_perm=19, seed=2)
    assert np.array_equal(Kr, Kc)


@pytest.mark.parametrize("case, reason", [
    ("missing", "not in the kernel"), ("twice", "appears twice"),
    ("kernel ids repeat", "not unique"), ("not finite", "not finite")])
def test_a_kernel_that_does_not_fit_the_samples_is_refused(case, reason):
    d = _data(n=40, m=100)
    ids, val, pick = d["ids"].copy(), d["K"].copy(), d["ids"][:20].copy()
    if case == "missing":
        pick[3] = "nobody"
    elif case == "twice":
        pick[3] = pick[4]
    elif case == "kernel ids repeat":
        ids[30] = ids[31]
    else:
        val[2, 5] = np.nan
    with pytest.raises(KinshipTestError, match=reason):
        kernel_for_samples(_Kernel(ids, val), pick)


def test_a_block_it_cannot_test_is_refused():
    d = _data(n=40, m=100)
    with pytest.raises(KinshipTestError, match="fewer than 2 haplotype groups"):
        kinship_block_test(d["y"], np.array(["H1"] * 40), None, d["Kr"], n_perm=9)
    with pytest.raises(KinshipTestError, match="cannot fit"):
        kinship_block_test(d["y"][:3], np.array(["H1", "H2", "H1"]), d["C"][:3],
                           d["Kr"][:3, :3], n_perm=9)
