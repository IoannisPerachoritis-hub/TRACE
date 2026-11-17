"""The kinship-aware haplotype test (gwas.haplotype_kinship), on synthetic data.

- The whitened F is the generalised least squares F (computed here with V^-1 itself).
- With K = I and h2 forced, the test is the PCs-only test: F to 1e-10, P exactly.
- At h2 = 0 (forced, or where the REML search lands) the block runs the PCs-only test's own
  functions: F and both p-values bit for bit.
- The permutations are the PCs-only test's (its generator and index sequence), so P is
  reproducible from the block's seed.
- The REML h2 is FaST-LMM's: LMM.findH2(REML=True) and single_snp's Nullh2, to 1e-6; it and
  the test do not move when a constant is added to the trait.
- Flipping the signs of the kernel's eigenvectors changes nothing.
- The kernel is never written to; it is restricted to the tested samples by ID.
- Haplotype groups that the intercept, PCs and covariates span on the tested samples are
  refused with the reason; with part of them spanned, the F is the GLS F of the rest.
- A block whose F is not defined (df1 = 0, or a full model that fits the samples exactly) is
  refused with the reason.

And in run_haplotype_block_gwas:
- the kinship-aware test is the default and needs the scan's kernels;
- each block takes its own chromosome's kernel (LOCO, or "global kernel" when it is K0),
  its row is the engine's result for the block's tested samples and seed, and eta2_pcs_only
  is the PCs-only test's eta2;
- a block without a kernel, or with a tested sample the kernel lacks, is kept with
  kinship_model "failed: <reason>" and NaN statistics, and FDR_BH is over the other blocks;
- the PCs-only test adds kinship_model "none", an empty h2_null and eta2_pcs_only = eta2.

And in the pipeline (the CLI and One-Click), on the bundled example: --hap-test kinship (the
default) gives the test the scan's kernels, "Hap_test (--hap-test)" is in run_metadata.json in
both modes, --no-loco gives the global kernel, a wrong value stops the run, and neither
LD_blocks_annotated_* nor LD_triage_* gains a column.
"""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import f as fdist

from gwas.haplotype import (_eta2_confint, _lead_snp_partial_r2, _omega2_from_f,
                            block_test_lm_with_pcs, freedman_lane_perm_pvalue,
                            run_haplotype_block_gwas)
from gwas.haplotype_kinship import (H2_MAX, PERMUTATION_TYPE, KinshipTestError, bh_qvalues,
                                    kernel_for_samples, kinship_block_test, null_h2,
                                    permutation_indices, reml_h2, _orth)
from gwas.models import _one_hot_drop_first
from gwas.utils import stable_seed


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


def test_the_reml_h2_on_a_reordered_subset_is_single_snps():
    """single_snp, given the full kernel as built (mean diagonal 1 over all its samples) and
    the phenotype and covariates of a reordered subset, intersects the IDs and rescales the
    kernel itself (DiagKtoN); its Nullh2 equals null_h2 on kernel_for_samples(full kernel,
    subset), so the ID restriction and the rescale are FaST-LMM's. Also with a mean of 1,000
    added to the trait (single_snp's null model does not move with the mean)."""
    from fastlmm.association import single_snp
    from pysnptools.kernelreader import KernelData
    from pysnptools.snpreader import SnpData

    from gwas.utils import CovarData, PhenoData

    d = _data(seed=3, n=150, h2=0.6)
    iid = np.c_[d["ids"], d["ids"]]
    pick = np.random.default_rng(6).permutation(len(d["ids"]))[:110]
    sub = d["ids"][pick]
    K32 = d["K"].astype(np.float32)                 # as TRACE keeps it
    Kr, _ = kernel_for_samples(_Kernel(d["ids"], K32), sub)
    assert abs(np.trace(K32[np.ix_(pick, pick)].astype(float)) / 110 - 1) > 1e-3   # a rescale
    snps = SnpData(iid=iid, sid=np.array(["a", "b"]), val=d["G"][:, :2].copy(),
                   pos=np.array([[1, 0, 1], [1, 0, 2]], dtype=float))
    for shift in (0.0, 1e3):
        y = d["y"][pick] + shift
        res = single_snp(test_snps=snps, pheno=PhenoData(sub, y),
                         K0=KernelData(iid=iid, val=K32.astype(np.float64)),
                         covar=CovarData(sub, d["C"][pick]), leave_out_one_chrom=False)
        h_ss = float(res["Nullh2"].iloc[0])
        ours = null_h2(y, d["C"][pick], Kr)
        assert 0.05 < ours < H2_MAX - 0.05, ours
        assert abs(ours - h_ss) < 1e-6, (shift, ours, h_ss)


def test_the_reml_h2_and_the_test_do_not_move_with_the_traits_mean():
    """The REML likelihood depends on y only through its residual on X, which reml_h2
    rotates (as single_snp's null model does), so a constant added to the trait (a mean of
    10,000 against a spread of about 1) moves neither h2 nor the test. Rotating the raw y
    moved h2 by about 3e-4 on these data, and F by about 1.5e-4 relative."""
    d = _data(seed=3, n=150, h2=0.6)
    a = kinship_block_test(d["y"], d["groups"], d["C"], d["Kr"], n_perm=199, seed=7,
                           lead_dose=d["dose"])
    b = kinship_block_test(d["y"] + 1e4, d["groups"], d["C"], d["Kr"], n_perm=199, seed=7,
                           lead_dose=d["dose"])
    assert 0.05 < a["h2"] < H2_MAX - 0.05, a["h2"]     # an interior optimum
    assert abs(b["h2"] - a["h2"]) < 1e-6, (a["h2"], b["h2"])
    for k in ("F_param", "PValue_param", "eta2", "eta2_ci_low", "eta2_ci_high", "omega2",
              "lead_snp_r2"):
        assert b[k] == pytest.approx(a[k], rel=1e-6, abs=0), k
    assert b["P_perm"] == a["P_perm"]
    assert abs(null_h2(d["y"] + 1e4, d["C"], d["Kr"]) - a["h2"]) < 1e-6


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


def _gls_f_lstsq(y, X0, H, V):
    """The nested GLS F of X0 against [X0, H] with covariance V, by least squares on the data
    whitened by V^-1/2 (so a full design that loses rank is allowed), with the PCs-only
    test's degrees of freedom (columns)."""
    S, U = np.linalg.eigh(V)
    W = U @ np.diag(S ** -0.5) @ U.T
    ys = W @ y

    def sse(X):
        Xs = W @ X
        r = ys - Xs @ np.linalg.lstsq(Xs, ys, rcond=None)[0]
        return float(r @ r)

    n = len(y)
    df0, df1 = n - X0.shape[1], n - X0.shape[1] - H.shape[1]
    s0, s1 = sse(X0), sse(np.column_stack([X0, H]))
    return ((s0 - s1) / (df0 - df1)) / (s1 / df1)


@pytest.mark.parametrize("h2", [None, 0.0, 0.3, 0.8])
def test_groups_the_covariates_span_are_not_tested(h2):
    """A covariate that defines the block's tested groups (here the group indicator itself, as
    the lead SNP's 0/2 dosage does in a two-group block of a conditional analysis) leaves no
    haplotype direction to test: the block is refused with the reason, at the REML h2 and at
    any forced h2, instead of testing what rounding leaves of the indicator."""
    d = _data(n_groups=2)
    H, _, _ = _one_hot_drop_first(d["groups"])
    Cc = np.column_stack([d["C"], 2.0 * H[:, 0]])
    with pytest.raises(KinshipTestError,
                       match="lie in the span of the intercept, PCs and covariates"):
        kinship_block_test(d["y"], d["groups"], Cc, d["Kr"], n_perm=99, seed=3, h2=h2)


@pytest.mark.parametrize("h2", [None, 0.0, 0.5])
def test_a_block_whose_f_is_not_defined_is_refused(h2):
    """Seven tested samples (groups of 4 and 3) with five PCs: the null model keeps one
    residual degree of freedom, so the F has none (df1 = 0, by the PCs-only test's rule); and
    eight samples in three groups with five PCs: the full model fits them exactly. Each block
    is refused with the reason (before any REML fit), not reported with NaN statistics beside
    a REML h2 as if tested. The PCs-only test (unchanged) gives NaN statistics at df1 = 0."""
    d = _data(n=40, m=100)
    rng = np.random.default_rng(1)
    for k, groups, why in [
            (7, np.array(["H1"] * 4 + ["H2"] * 3),
             r"the F test has no degrees of freedom \(df1 = 0\): 7 tested samples, 6 fixed "
             r"effects, 2 haplotype groups"),
            (8, np.array(["H1"] * 3 + ["H2"] * 3 + ["H3"] * 2),
             r"the full model fits the tested samples exactly, leaving no residual variance: "
             r"8 tested samples, 6 fixed effects, 3 haplotype groups")]:
        C = rng.normal(size=(k, 5))
        Kr, _ = kernel_for_samples(_Kernel(d["ids"], d["K"]), d["ids"][:k])
        with pytest.raises(KinshipTestError, match=why):
            kinship_block_test(d["y"][:k], groups, C, Kr, n_perm=9, seed=1, h2=h2)
    F, _p, df1, df2 = block_test_lm_with_pcs(d["y"][:7], np.array(["H1"] * 4 + ["H2"] * 3),
                                             pcs=rng.normal(size=(7, 5)))
    assert np.isnan(F) and (df1, df2) == (0, 1)


@pytest.mark.parametrize("h2", [0.3, 0.8])
def test_a_group_the_covariates_span_leaves_the_others_tested(h2):
    """With one of three groups' indicators among the covariates, the F is the GLS F of the
    direction left (least squares on the whitened data)."""
    d = _data()
    n = len(d["y"])
    H, _, _ = _one_hot_drop_first(d["groups"])
    Cc = np.column_stack([d["C"], H[:, 0]])
    res = kinship_block_test(d["y"], d["groups"], Cc, d["Kr"], n_perm=9, seed=1, h2=h2)
    F = _gls_f_lstsq(d["y"], np.column_stack([np.ones(n), Cc]), H,
                     h2 * d["Kr"] + (1 - h2) * np.eye(n))
    assert res["F_param"] == pytest.approx(F, rel=1e-8)


# ------------------------------------------------- run_haplotype_block_gwas, kinship mode
def _two_chromosome_panel(seed=21, n=150, m_bg=300):
    """Families of related samples; one block of three SNPs (three haplotype groups) at the
    start of each of two chromosomes, then background SNPs; the scan's kernels built from the
    background SNPs by the pipeline's LOCO builder; a phenotype with a family effect."""
    from gwas.kinship import _build_loco_kernels_impl, _standardize_geno_for_grm

    rng = np.random.default_rng(seed)
    ids = np.array([f"A{i:03d}" for i in range(n)])
    fam = rng.integers(0, 15, n)
    base = rng.binomial(2, 0.3, size=(15, 2 * m_bg))
    G_bg = np.where(rng.random((n, 2 * m_bg)) < 0.8, base[fam],
                    rng.binomial(2, 0.3, size=(n, 2 * m_bg))).astype(float)
    pat = np.array([[0, 0, 2], [2, 2, 0], [0, 2, 2]], float)
    g1, g2 = rng.integers(0, 3, n), rng.integers(0, 3, n)
    G = np.column_stack([pat[g1], G_bg[:, :m_bg], pat[g2], G_bg[:, m_bg:]])
    chroms = np.array(["1"] * (3 + m_bg) + ["2"] * (3 + m_bg))
    pos = np.tile((np.arange(3 + m_bg) + 1) * 1000.0, 2)
    sid = np.array([f"c{c}s{j}" for c, j in zip(chroms, np.tile(np.arange(3 + m_bg), 2))])
    blocks = pd.DataFrame({
        "Chr": ["1", "2"], "Start (bp)": [1000, 1000], "End (bp)": [3000, 3000],
        "lead_snp": ["c1s1", "c2s0"], "SNP_IDs": ["c1s0,c1s1,c1s2", "c2s0,c2s1,c2s2"]})
    fam_eff = rng.normal(0.0, 1.0, 15)
    y = 0.4 * g1 + 0.3 * g2 + fam_eff[fam] + rng.normal(0.0, 0.8, n)
    Z = _standardize_geno_for_grm(G_bg)
    K0, K_by_chr, _ = _build_loco_kernels_impl(np.c_[ids, ids], Z,
                                               np.array(["1"] * m_bg + ["2"] * m_bg))
    return dict(blocks=blocks, chroms=chroms, pos=pos, G=G, sid=sid, ids=ids,
                geno_df=pd.DataFrame(index=ids), pheno=pd.DataFrame({"t": y}, index=ids),
                K0=K0, K_by_chr=K_by_chr)


def _run(p, **kw):
    return run_haplotype_block_gwas(
        haplo_df=p["blocks"], chroms=p["chroms"], positions=p["pos"], geno_imputed=p["G"],
        sid=p["sid"], geno_df=p["geno_df"], pheno_df=p["pheno"], trait_col="t", pcs=None,
        n_perm=199, **kw)


def test_the_kinship_test_is_the_default_and_needs_the_kernels():
    p = _two_chromosome_panel()
    for kw in ({}, {"hap_test": "kinship"}):
        with pytest.raises(ValueError, match="needs the scan's kernels"):
            _run(p, **kw)
    with pytest.raises(ValueError, match="hap_test must be one of"):
        _run(p, hap_test="lmm", kernels=(p["K0"], p["K_by_chr"]))


def test_each_block_is_tested_with_its_own_chromosomes_kernel():
    p = _two_chromosome_panel()
    res, tables = _run(p, hap_test="kinship", kernels=(p["K0"], p["K_by_chr"]))
    pcs, _ = _run(p, hap_test="pcs")
    assert len(res) == 2 and list(res.columns) == list(pcs.columns)
    for i, r in res.iterrows():
        tab = tables[f"{r['Chr']}:{int(r['Start'])}-{int(r['End'])}"]
        tab = tab[tab["in_haplotype_test"]]
        samples = tab["Sample"].astype(str).to_numpy()
        K, _ = kernel_for_samples(p["K_by_chr"][str(r["Chr"])], samples)
        lead = np.flatnonzero(p["sid"] == r["lead_snp"])[0]
        want = kinship_block_test(
            p["pheno"]["t"].reindex(samples).to_numpy(float),
            tab["Haplotype"].astype(str).to_numpy(), None, K, n_perm=199,
            seed=stable_seed(str(r["Chr"]), int(r["Start"]), int(r["End"]), "t", "FL"),
            lead_dose=p["G"][:, lead][pd.Index(p["ids"]).get_indexer(samples)])
        assert want["h2"] > 0, "the family effect should give the null model some h2"
        for c in ("F_param", "PValue_param", "df1", "df2", "F_perm", "P_perm", "eta2",
                  "eta2_ci_low", "eta2_ci_high", "omega2", "lead_snp_r2"):
            assert r[c] == want[c], c
        assert r["PValue"] == want["P_perm"] and r["h2_null"] == want["h2"]
        assert r["kinship_model"] == "LOCO kernel"
        assert r["permutation_type"] == PERMUTATION_TYPE
        # the PCs-only test's eta2, for the same block and samples
        assert r["eta2_pcs_only"] == pcs.loc[i, "eta2"]
    assert res["FDR_BH"].tolist() == bh_qvalues(res["PValue"].to_numpy()).tolist()
    # with K0 for every chromosome (--no-loco): the global kernel; chromosome labels are
    # matched as canonical names ("chr1" is chromosome 1)
    glob, _ = _run(p, kernels=(p["K0"], {"chr1": p["K0"], "chr2": p["K0"]}))
    assert glob["kinship_model"].tolist() == ["global kernel", "global kernel"]
    assert not np.allclose(glob["F_param"], res["F_param"])


def test_a_block_the_kinship_test_cannot_test_is_kept_with_the_reason():
    p = _two_chromosome_panel()
    only1 = {"1": p["K_by_chr"]["1"]}
    res, tables = _run(p, kernels=(p["K0"], only1))
    full, _ = _run(p, kernels=(p["K0"], p["K_by_chr"]))
    pcs, _ = _run(p, hap_test="pcs")
    assert res["kinship_model"].tolist() == ["LOCO kernel", "failed: no kernel for chromosome 2"]
    bad = res.iloc[1]
    for c in ("F_param", "PValue_param", "F_perm", "P_perm", "PValue", "eta2", "eta2_ci_low",
              "eta2_ci_high", "omega2", "lead_snp_r2", "h2_null", "FDR_BH", "-log10p"):
        assert np.isnan(bad[c]), c
    # the design's degrees of freedom and the PCs-only eta2 are still given
    assert (bad["df1"], bad["df2"]) == (pcs.iloc[1]["df1"], pcs.iloc[1]["df2"])
    assert bad["eta2_pcs_only"] == pcs.iloc[1]["eta2"]
    assert res.dtypes["df1"] == full.dtypes["df1"]
    # the other block is as before, and FDR_BH is over the blocks with a p-value
    assert res.iloc[0]["P_perm"] == full.iloc[0]["P_perm"]
    assert res.iloc[0]["FDR_BH"] == res.iloc[0]["PValue"]
    # a tested sample the kernel lacks fails the block too
    k1 = p["K_by_chr"]["1"]
    keep = np.arange(1, len(p["ids"]))
    short = type(k1)(iid=k1.iid[keep], val=np.asarray(k1.val)[np.ix_(keep, keep)])
    res2, _ = _run(p, kernels=(p["K0"], {"1": short, "2": p["K_by_chr"]["2"]}))
    assert res2["kinship_model"].iloc[0] == "failed: 1 tested sample(s) are not in the kernel"
    assert res2["kinship_model"].iloc[1] == "LOCO kernel"


def test_a_block_whose_groups_the_covariates_span_is_kept_with_the_reason():
    """User covariates that together define block 1's three groups (its first two SNPs):
    block 1 is reported as failed with the reason, block 2 is tested."""
    p = _two_chromosome_panel()
    cov = p["G"][:, [0, 1]]
    res, _ = _run(p, kernels=(p["K0"], p["K_by_chr"]), user_covar=cov)
    pcs, _ = _run(p, hap_test="pcs", user_covar=cov)
    assert res["kinship_model"].iloc[0] == (
        "failed: the haplotype groups lie in the span of the intercept, PCs and covariates on "
        "the tested samples: no haplotype effect is left to test")
    for c in ("F_param", "PValue_param", "P_perm", "eta2", "h2_null"):
        assert np.isnan(res[c].iloc[0]), c
    assert res["eta2_pcs_only"].iloc[0] == pcs["eta2"].iloc[0]
    assert res["kinship_model"].iloc[1] == "LOCO kernel"
    assert np.isfinite(res["F_param"].iloc[1]) and np.isfinite(res["P_perm"].iloc[1])


def test_a_block_whose_f_has_no_degrees_of_freedom_is_kept_with_the_reason():
    """Block 1 tested on seven samples (groups of 4 and 3) with five PCs: df1 = 0. Its row says
    why, with no h2; the design's degrees of freedom and the PCs-only eta2 are kept. The
    PCs-only test's row is today's: NaN F and p-values, a finite eta2."""
    p = _two_chromosome_panel()
    G = p["G"]
    grp = np.select([(G[:, 0] == 0) & (G[:, 1] == 0), G[:, 0] == 2], [0, 1], 2)
    keep = np.r_[np.flatnonzero(grp == 0)[:4], np.flatnonzero(grp == 1)[:3]]
    pheno = p["pheno"].copy()
    pheno.loc[~pheno.index.isin(p["ids"][keep]), "t"] = np.nan
    kw = dict(haplo_df=p["blocks"], chroms=p["chroms"], positions=p["pos"], geno_imputed=G,
              sid=p["sid"], geno_df=p["geno_df"], pheno_df=pheno, trait_col="t",
              pcs=np.random.default_rng(4).normal(size=(len(p["ids"]), 5)),
              min_hap_count=3, min_group_size=3, n_perm=99)
    res, _ = run_haplotype_block_gwas(**kw, kernels=(p["K0"], p["K_by_chr"]))
    pcs, _ = run_haplotype_block_gwas(**kw, hap_test="pcs")
    r = res[res["Chr"].astype(str) == "1"].iloc[0]
    q = pcs[pcs["Chr"].astype(str) == "1"].iloc[0]
    assert r["n_samples_tested"] == 7
    assert r["kinship_model"] == ("failed: the F test has no degrees of freedom (df1 = 0): "
                                  "7 tested samples, 6 fixed effects, 2 haplotype groups")
    for c in ("h2_null", "F_param", "PValue_param", "P_perm", "eta2"):
        assert np.isnan(r[c]), c
    assert (r["df1"], r["df2"]) == (q["df1"], q["df2"]) == (0, 1)
    assert np.isnan(q["F_param"]) and np.isnan(q["P_perm"]) and np.isfinite(q["eta2"])
    assert r["eta2_pcs_only"] == q["eta2"]


def test_the_pcs_test_adds_only_the_three_columns():
    p = _two_chromosome_panel()
    res, _ = _run(p, hap_test="pcs")
    assert list(res.columns[-6:]) == ["kinship_model", "h2_null", "eta2_pcs_only", "MidPos",
                                      "-log10p", "FDR_BH"]
    assert res["kinship_model"].tolist() == ["none", "none"]
    assert res["h2_null"].isna().all()
    assert res["eta2_pcs_only"].tolist() == res["eta2"].tolist()
    assert (res["permutation_type"] == "Freedman-Lane_residual_permutation_with_PCs").all()
    # an empty result has the same columns
    empty, _ = run_haplotype_block_gwas(
        haplo_df=p["blocks"].iloc[:0], chroms=p["chroms"], positions=p["pos"],
        geno_imputed=p["G"], sid=p["sid"], geno_df=p["geno_df"], pheno_df=p["pheno"],
        trait_col="t", hap_test="pcs")
    assert {"kinship_model", "h2_null", "eta2_pcs_only"} <= set(empty.columns)


def test_the_kinship_test_never_writes_the_kernels():
    p = _two_chromosome_panel()
    kernels = [p["K0"], *p["K_by_chr"].values()]
    before = [np.asarray(k.val).copy() for k in kernels]
    for k in kernels:
        np.asarray(k.val).setflags(write=False)
    _run(p, kernels=(p["K0"], p["K_by_chr"]))
    for k, b in zip(kernels, before):
        assert np.array_equal(np.asarray(k.val), b)


# ------------------------------------------------------------------- the pipeline (CLI)
_EX = Path(__file__).resolve().parent / "data" / "example_v1"


def _example_run(monkeypatch, **opts):
    """The pipeline on the first example (Trait2, the b4 cases' LD settings, no plots); the
    haplotype test's keyword arguments and result frames of every model, and the ZIP."""
    import gwas.haplotype as gh
    import gwas.pipeline as pl
    from gwas.io import load_vcf_cached

    seen = []
    real = gh.run_haplotype_block_gwas

    def _rec(*a, **k):
        out = real(*a, **k)
        seen.append((k, out[0]))
        return out

    monkeypatch.setattr(gh, "run_haplotype_block_gwas", _rec)
    vcf, pheno_csv = _EX / "example.vcf.gz", _EX / "example_pheno.csv"
    parsed = pl.ParsedGenotypes.from_callset(load_vcf_cached(vcf.read_bytes()))
    pheno = pd.read_csv(pheno_csv, index_col=0, encoding="utf-8-sig")
    res = pl.run_trait_pipeline(parsed, pheno, "Trait2", pl.cli_default_options(
        vcf=str(vcf), pheno=str(pheno_csv), ld_r2=0.1, ld_flank_kb=200, hap_perms=100,
        no_plots=True, **opts))
    with zipfile.ZipFile(io.BytesIO(res.zip_bytes)) as zf:
        members = {n: zf.read(n) for n in zf.namelist()}
    return res, seen, members


def test_the_pipeline_gives_the_kinship_test_the_scans_kernels(monkeypatch):
    res, seen, members = _example_run(monkeypatch)
    pcs_res, pcs_seen, pcs_members = _example_run(monkeypatch, hap_test="pcs")
    meta = json.loads(members["run_metadata.json"])
    pcs_meta = json.loads(pcs_members["run_metadata.json"])
    assert meta["Hap_test (--hap-test)"] == "kinship"
    assert pcs_meta["Hap_test (--hap-test)"] == "pcs"
    # the two runs' metadata differ only by that key (and its line in the report)
    assert {k: v for k, v in meta.items() if k != "Hap_test (--hap-test)"} == \
        {k: v for k, v in pcs_meta.items() if k != "Hap_test (--hap-test)"}
    k, frame = seen[0]
    assert k["hap_test"] == "kinship"
    K0, K_by_chr = k["kernels"]
    assert set(map(str, K_by_chr)) and all(hasattr(v, "val") for v in K_by_chr.values())
    assert not frame.empty
    assert set(frame["kinship_model"]) <= {"LOCO kernel", "global kernel"}
    assert frame["h2_null"].notna().all()
    assert pcs_seen[0][0]["hap_test"] == "pcs" and pcs_seen[0][0]["kernels"] is None
    assert (pcs_seen[0][1]["kinship_model"] == "none").all()
    # no new column reaches the LD block or triage tables (named column lists)
    for name in ("LD_blocks_annotated_MLM.csv", "LD_triage_MLM.csv"):
        assert list(res.tables[name].columns) == list(pcs_res.tables[name].columns), name
        assert not {"kinship_model", "h2_null", "eta2_pcs_only"} & set(res.tables[name].columns)
    # the report's haplotype table shows the columns
    assert b"<th>kinship_model</th>" in members["report.html"]
    assert b"<th>kinship_model</th>" in pcs_members["report.html"]


def test_without_loco_every_block_has_the_global_kernel(monkeypatch):
    _res, seen, _m = _example_run(monkeypatch, no_loco=True)
    frame = seen[0][1]
    assert not frame.empty and (frame["kinship_model"] == "global kernel").all()


def test_a_wrong_haplotype_test_stops_the_run(monkeypatch):
    from gwas.pipeline import TraitPipelineError

    with pytest.raises(TraitPipelineError, match="--hap-test must be one of"):
        _example_run(monkeypatch, hap_test="lmm")


def test_the_shared_kernel_helper_builds_the_pipelines_kernels(monkeypatch):
    """gwas.kinship.scan_kernels, from the inputs the pipeline's kinship step gets, gives the
    kernels the pipeline hands the haplotype test, value for value (LOCO, and --no-loco):
    the routes outside the pipeline (capture_golden, the null calibration) use it."""
    import gwas.qc as qc
    from gwas.kinship import scan_kernels

    for no_loco in (False, True):
        got = {}
        real = qc._pipeline_build_kinship

        def _rec(geno_df, geno_imputed, chroms, positions, _real=real, _got=got):
            _got["args"] = (geno_df.copy(), np.array(geno_imputed, copy=True),
                            np.array(chroms, copy=True), np.array(positions, copy=True))
            return _real(geno_df, geno_imputed, chroms, positions)

        monkeypatch.setattr(qc, "_pipeline_build_kinship", _rec)
        _res, seen, _m = _example_run(monkeypatch, no_loco=no_loco)
        monkeypatch.setattr(qc, "_pipeline_build_kinship", real)
        K0, K_by_chr = seen[0][0]["kernels"]
        H0, H_by_chr = scan_kernels(*got["args"], np.asarray(K0.iid), no_loco=no_loco)
        assert np.array_equal(np.asarray(H0.val), np.asarray(K0.val))
        assert np.array_equal(np.asarray(H0.iid), np.asarray(K0.iid))
        assert sorted(map(str, H_by_chr)) == sorted(map(str, K_by_chr))
        for c, K in K_by_chr.items():
            assert np.array_equal(np.asarray(H_by_chr[c].val), np.asarray(K.val)), c
            assert (H_by_chr[c] is H0) == (K is K0), c
