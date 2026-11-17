"""The kinship-aware Freedman-Lane haplotype test: the default haplotype test (--hap-test kinship).

For one LD block, on the block's tested samples, the model is

    y = X b + H c + u + e,    u ~ N(0, s2_g K),    e ~ N(0, s2_e I)

X: an intercept, the PCs and the user covariates (the PCs-only test's design);
H: the tested haplotype groups, one-hot with the first group dropped (as in the PCs-only test);
K: the scan's own kernel for the block's chromosome (the LOCO kernel, or the global one),
   restricted to the tested samples by ID and rescaled so its diagonal sums to their number
   (kernel_for_samples; FaST-LMM's DiagKtoN does the same).

1. The null model y ~ X is fitted by REML: h2 = s2_g / (s2_g + s2_e) minimises FaST-LMM's REML
   negative log-likelihood (reml_nll, fastlmm.inference.lmm.LMM.nLLeval for a full-rank kernel)
   by FaST-LMM's own search, fastlmm.util.mingrid.minimize1D over [0, 0.99999] (an 11-point
   grid, then bounded and Brent refinements), keeping the best point evaluated, as
   LMM.findH2 does (reml_h2). The likelihood is evaluated on y's residual on X, which it
   depends on alone, as single_snp's null model (fastlmm.inference.lmm_cov) does: h2 then does
   not move with the trait's mean, as it does when the raw y is rotated.
2. V = h2 K + (1 - h2) I = U diag(d) U' is whitened by the symmetric W = U diag(d^-1/2) U'. It
   depends only on K and h2: the test does not change with the signs LAPACK gives the
   eigenvectors, or with the basis it picks inside a repeated eigenvalue. W is applied as
   U (d^-1/2 * (U' v)) and never formed.
3. The parametric test is the nested F of [WX] against [WX, WH] (least squares on the whitened
   data is generalised least squares), with the PCs-only test's degrees of freedom
   (max(1, n - columns) for each model).
4. Freedman-Lane on the whitened scale: the residuals of Wy on WX are permuted with the PCs-only
   test's generator and index sequence (numpy's default_rng(seed) and one in-place shuffle of a
   running index per permutation: permutation_indices), and each permutation's F is computed
   from the permuted residuals (the fitted values lie in both models' column spaces and drop
   out). A permutation F that is not finite counts as 0, as in the PCs-only test, and
   P = (1 + #{F_b >= F_obs}) / (1 + n_perm). F_obs is the parametric F itself.
5. The effect sizes are those of the same whitened model: eta2 = df1 F / (df1 F + df2), its
   noncentral-F interval and omega2 from the same F, and lead_snp_r2, the r2 of the whitened
   phenotype and the whitened lead SNP after WX is projected out of both.

At h2 = 0, which the search reaches exactly (0 is a grid point and the best point evaluated is
kept), the whitening is the identity, and the block is tested by the PCs-only test's own
functions (gwas.haplotype.block_test_lm_with_pcs and freedman_lane_perm_pvalue): its F and
p-values are then those of the PCs-only test bit for bit.

A block the test cannot test raises KinshipTestError with the reason; the caller reports it,
never falling back to another test.
"""
import numpy as np
import pandas as pd

# the haplotype result's permutation_type for this test (the PCs-only test's is
# "Freedman-Lane_residual_permutation_with_PCs")
PERMUTATION_TYPE = "Freedman-Lane_residual_permutation_kinship_whitened"
H2_MAX = 0.99999     # FaST-LMM's upper bound for h2 (LMM.findH2, lmm_cov.LMM.findH2)
N_GRID = 10          # FaST-LMM's nGridH2: 11 grid points on [0, H2_MAX]


class KinshipTestError(Exception):
    """A block the kinship-aware test cannot test. The message is the reason."""


# ------------------------------------------------------------------------------ the kernel
def kernel_for_samples(kernel, samples):
    """(K, positions): ``kernel`` restricted to ``samples`` (IDs, in that order) as a new
    float64 array, rescaled so that its diagonal sums to the number of samples (FaST-LMM's
    DiagKtoN), and the samples' row positions in the kernel. ``kernel`` is a KernelData: the
    second column of its ``iid`` (or ``iid`` itself when one-dimensional) names the rows of
    ``val``. Nothing is written to it. Raises KinshipTestError when a sample is not in the
    kernel or appears twice, when the kernel's IDs repeat, or when its values on the samples
    are not finite."""
    iid = np.asarray(kernel.iid)
    ids = pd.Index((iid[:, 1] if iid.ndim == 2 else iid).astype(str))
    if not ids.is_unique:
        raise KinshipTestError("the kernel's sample IDs are not unique")
    want = pd.Index(np.asarray(samples).astype(str))
    if not want.is_unique:
        raise KinshipTestError("a tested sample appears twice")
    pos = ids.get_indexer(want)
    if (pos < 0).any():
        raise KinshipTestError(f"{int((pos < 0).sum())} tested sample(s) are not in the kernel")
    K = np.array(np.asarray(kernel.val)[np.ix_(pos, pos)], dtype=np.float64)
    if not np.isfinite(K).all():
        raise KinshipTestError("the kernel is not finite on the tested samples")
    trace = float(np.trace(K))
    if not (np.isfinite(trace) and trace > 0):
        raise KinshipTestError("the kernel's diagonal does not sum to a positive value on the "
                               "tested samples")
    factor = float(K.shape[0]) / trace
    if abs(factor - 1.0) > 1e-15:          # DiagKtoN's rule
        K *= factor
    return K, pos


def _orth(A):
    """An orthonormal basis of the column space of ``A``: its left singular vectors above
    numpy's rank tolerance (numpy.linalg.matrix_rank's)."""
    A = np.asarray(A, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.shape[1] == 0:
        return np.zeros((A.shape[0], 0))
    u, s, _ = np.linalg.svd(A, full_matrices=False)
    tol = (s.max() if s.size else 0.0) * max(A.shape) * np.finfo(float).eps
    return u[:, : int((s > tol).sum())]


# -------------------------------------------------------------------------------- the REML
def reml_nll(h2, S, UX, Uy, n, d):
    """FaST-LMM's REML negative log-likelihood of y ~ X with covariance h2 K + (1 - h2) I,
    up to its scale: fastlmm.inference.lmm.LMM.nLLeval (REML, a full-rank kernel; no
    Student-t, penalty or excluded SNPs), for K = U diag(S) U'. ``UX`` is U'Q for an
    orthonormal basis Q of X's column space (so log|X'X| = 0; the REML likelihood depends on
    X only through that space), ``Uy`` is U'r for y's residual r on X (it depends on y only
    through r; reml_h2), ``n`` the samples and ``d`` the rank of X.
    Out of [0, 1] it is 3e20, as in FaST-LMM; where the covariance is not positive definite
    it is NaN (FaST-LMM's logarithm gives NaN there), which the search never keeps."""
    if (h2 < 0.0) or (h2 > 1.0):
        return 3e20
    Sd = h2 * S + (1.0 - h2)
    if not np.all(Sd > 0):
        return float("nan")
    UXS = UX / Sd[:, None]
    UyS = Uy / Sd
    XKX = UXS.T.dot(UX)
    XKy = UXS.T.dot(Uy)
    yKy = UyS.T.dot(Uy)
    logdetK = np.log(Sd).sum()
    SxKx, UxKx = np.linalg.eigh(XKX)
    i_pos = SxKx > 1e-10
    beta = UxKx[:, i_pos].dot(UxKx[:, i_pos].T.dot(XKy) / SxKx[i_pos])
    r2 = yKy - XKy.dot(beta)
    with np.errstate(divide="ignore", invalid="ignore"):
        logdetXKX = np.log(SxKx).sum()
        sigma2 = r2 / (n - d)
        nll = 0.5 * (logdetK + logdetXKX + (n - d) * (np.log(2.0 * np.pi * sigma2) + 1))
    return float(nll)


def reml_h2(S, U, y, Q):
    """(h2, nLL): the REML estimate of h2 for y ~ X with the kernel U diag(S) U' (``Q`` an
    orthonormal basis of X's column space), found as FaST-LMM's LMM.findH2 finds it: the
    best point evaluated by fastlmm.util.mingrid.minimize1D over [0, H2_MAX] with N_GRID
    grid steps (a point replaces the best only when strictly lower, so a tie keeps the first
    evaluated, and 0, the first grid point, is reached exactly).

    The likelihood is evaluated on y's residual on X, r = y - QQ'y: the REML likelihood
    depends on y only through its contrasts orthogonal to X, so this is exact, and it is what
    single_snp's null model does (fastlmm.inference.lmm_cov regresses X out before it
    rotates). Rotating the raw y, as LMM.nLLeval does, makes yKy - XKy.beta cancel more as
    the trait's mean grows against its spread, and h2 then moves with the mean."""
    from fastlmm.util.mingrid import minimize1D

    y = np.asarray(y, dtype=float).reshape(-1)
    n, d = y.size, Q.shape[1]
    r = y - Q.dot(Q.T.dot(y))
    UX, Uy = U.T.dot(Q), U.T.dot(r)
    best = {"h2": None, "nll": None}

    def f(x):
        v = reml_nll(x, S, UX, Uy, n, d)
        if best["nll"] is None or v < best["nll"]:
            best["h2"], best["nll"] = float(x), v
        return v

    minimize1D(f=f, nGrid=N_GRID, minval=0.0, maxval=H2_MAX)
    return best["h2"], best["nll"]


def null_h2(y, covariates, K):
    """The REML h2 of the null model y ~ [1, covariates] with kernel ``K`` (an n x n array
    already restricted and rescaled, as kernel_for_samples gives it): reml_h2 on K's
    eigendecomposition."""
    y = np.asarray(y, dtype=float).reshape(-1)
    X = _design(covariates, y.size)
    S, U = np.linalg.eigh(np.asarray(K, dtype=float))
    return reml_h2(S, U, y, _orth(X))[0]


# --------------------------------------------------------------------------- the whitening
def _whitener(S, U, h2):
    """v -> W v for W = U diag((h2 S + 1 - h2)^-1/2) U' (the symmetric whitening)."""
    Sd = h2 * S + (1.0 - h2)
    if not np.all(Sd > 0):
        raise KinshipTestError(f"the null model's covariance is not positive definite at "
                               f"h2 = {h2:.6g}")
    s = 1.0 / np.sqrt(Sd)

    def W(A):
        A = np.asarray(A, dtype=float)
        if A.ndim == 1:
            return U.dot(s * U.T.dot(A))
        return U.dot(s[:, None] * U.T.dot(A))

    return W


def permutation_indices(n, n_perm, seed):
    """The PCs-only test's permutations (gwas.haplotype.freedman_lane_perm_pvalue), one row
    per permutation: numpy's default_rng(seed), and one in-place shuffle of a running index
    per permutation, so permutation b is the shuffle of permutation b - 1."""
    rng = np.random.default_rng(int(seed))
    idx = np.arange(int(n))
    out = np.empty((int(n_perm), int(n)), dtype=idx.dtype)
    for b in range(int(n_perm)):
        rng.shuffle(idx)
        out[b] = idx
    return out


def _f_of_residuals(R, Q0, QH, rr, df_num, df_den):
    """The nested F of each column of ``R`` (residual vectors of norm^2 ``rr``, possibly
    permuted) for the reduced model's basis ``Q0`` against the full model's, ``QH`` spanning
    the full model's columns outside Q0; NaN where gwas.models._nested_f_test gives NaN (no
    degrees of freedom, a non-positive residual sum of squares, an F that is not finite or
    negative)."""
    T0 = Q0.T.dot(R)
    TH = QH.T.dot(R)
    sse0 = rr - (T0 * T0).sum(axis=0)
    ssb = (TH * TH).sum(axis=0)
    sse1 = sse0 - ssb
    with np.errstate(divide="ignore", invalid="ignore"):
        F = (ssb / df_num) / (sse1 / df_den)
    ok = (df_num > 0) & (df_den > 0) & np.isfinite(sse0) & np.isfinite(sse1) & (sse1 > 0) \
        & np.isfinite(F) & (F >= 0)
    return np.where(ok, F, np.nan)


def _design(covariates, n):
    """[1, covariates]: the intercept and the covariate columns (none when ``covariates`` is
    None), as the PCs-only test builds its reduced model."""
    from gwas.utils import _ensure_2d

    C = np.zeros((n, 0), float) if covariates is None else _ensure_2d(covariates)
    if C.shape[0] != n:
        raise ValueError(f"covariate rows ({C.shape[0]}) != y length ({n}).")
    return np.column_stack([np.ones(n), C])


def _lead_r2(dose, e, Q0, W):
    """The r2 of the whitened phenotype's residual ``e`` and the whitened lead SNP ``dose``
    after Q0 (the whitened design) is projected out; NaN when the dose is missing, not
    finite, monomorphic on the samples, or nothing of it (or of the phenotype) is left."""
    if dose is None:
        return float("nan")
    dose = np.asarray(dose, dtype=float).reshape(-1)
    n = e.size
    if dose.size != n or n < 4 or not np.isfinite(dose).all() or np.ptp(dose) == 0:
        return float("nan")
    ds = W(dose)
    g = ds - Q0.dot(Q0.T.dot(ds))
    ee, gg = float(e.dot(e)), float(g.dot(g))
    if not (ee > n * 1e-24 and gg > n * 1e-24):
        return float("nan")
    eg = float(e.dot(g))
    return eg * eg / (ee * gg)


def bh_qvalues(p):
    """Benjamini-Hochberg q-values over the finite p-values of ``p``, NaN elsewhere: a block
    the kinship-aware test could not test has no p-value and is not one of the tests. With
    every p-value finite this is statsmodels' fdr_bh over all of them."""
    from statsmodels.stats.multitest import multipletests

    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q


# ---------------------------------------------------------------------------------- the test
def kinship_block_test(y, groups, covariates, K=None, *, n_perm=1000, seed=0, lead_dose=None,
                       h2=None, eig=None):
    """The kinship-aware test of one block (the module's docstring).

    ``y`` and ``groups`` are the tested samples' phenotype and haplotype labels;
    ``covariates`` the PCs and user covariates (n x c, or None), without the intercept;
    ``K`` the kernel on the same samples in the same order (kernel_for_samples), or ``eig``
    its eigendecomposition (S, U) when already computed; ``seed`` the block's permutation
    seed and ``n_perm`` the permutations; ``lead_dose`` the lead SNP's dosage on the samples
    (or None). ``h2`` forces the null model's h2 instead of its REML estimate.

    Returns a dict: h2, F_param, PValue_param, df1, df2, F_perm (the parametric F),
    P_perm, eta2, eta2_ci_low, eta2_ci_high, omega2, lead_snp_r2. Raises KinshipTestError
    when the block cannot be tested.
    """
    from scipy.stats import f as fdist

    from gwas.haplotype import (_eta2_confint, _omega2_from_f, block_test_lm_with_pcs,
                                freedman_lane_perm_pvalue)
    from gwas.models import _one_hot_drop_first

    y = np.asarray(y, dtype=float).reshape(-1)
    n = y.size
    X0 = _design(covariates, n)
    H, _, _ = _one_hot_drop_first(np.asarray(groups, object))
    if H is None or H.shape[1] < 1:
        raise KinshipTestError("fewer than 2 haplotype groups")
    Q = _orth(X0)
    if n - Q.shape[1] < 1:
        raise KinshipTestError(f"{n} tested samples cannot fit the null model's "
                               f"{Q.shape[1]} fixed effects")
    if eig is None:
        Kv = np.asarray(K, dtype=float)
        if Kv.shape != (n, n):
            raise ValueError(f"kernel shape {Kv.shape} != ({n}, {n}).")
        if not np.isfinite(Kv).all():
            raise KinshipTestError("the kernel is not finite on the tested samples")
        try:
            S, U = np.linalg.eigh(Kv)
        except np.linalg.LinAlgError:
            raise KinshipTestError("the kernel's eigendecomposition did not converge") from None
    else:
        S, U = eig
    if h2 is None:
        h2, nll = reml_h2(S, U, y, Q)
        if h2 is None or not np.isfinite(nll):
            raise KinshipTestError("the REML fit of the null model failed")
    h2 = float(h2)

    if h2 == 0.0:
        # the whitening is the identity: the PCs-only test's own functions
        F, p_param, df_num, df_den = block_test_lm_with_pcs(y, groups, pcs=covariates)
        F_perm, p_perm = freedman_lane_perm_pvalue(y, groups, pcs=covariates,
                                                   n_perm=int(n_perm), seed=seed)

        def W(A):
            return np.asarray(A, dtype=float)

        Q0 = Q
        e = y - Q0.dot(Q0.T.dot(y))
    else:
        W = _whitener(S, U, h2)
        ys, Xs, Hs = W(y), W(X0), W(H)
        df0 = int(max(1, n - X0.shape[1]))
        df_den = int(max(1, n - X0.shape[1] - H.shape[1]))
        df_num = int(df0 - df_den)
        Q0 = _orth(Xs)
        QH = _orth(Hs - Q0.dot(Q0.T.dot(Hs)))
        e = ys - Q0.dot(Q0.T.dot(ys))              # the whitened residuals of the null model
        rr = float(e.dot(e))
        F = float(_f_of_residuals(e[:, None], Q0, QH, rr, df_num, df_den)[0])
        p_param = float(fdist.sf(F, df_num, df_den)) if np.isfinite(F) else float("nan")
        if np.isfinite(F):
            idx = permutation_indices(n, n_perm, seed)
            Fb = _f_of_residuals(e[idx.T], Q0, QH, rr, df_num, df_den)
            Fb = np.where(np.isfinite(Fb), Fb, 0.0)
            p_perm = (1.0 + float(np.sum(Fb >= F))) / (1.0 + float(int(n_perm)))
        else:
            p_perm = float("nan")
        F_perm = F

    F = float(F) if np.isfinite(F) else float("nan")
    ok = np.isfinite(F) and df_num >= 1 and df_den >= 1
    eta2 = float(df_num * F / (df_num * F + df_den)) if ok and (df_num * F + df_den) > 0 \
        else float("nan")
    lo, hi = _eta2_confint(F, df_num, df_den)
    return {
        "h2": h2,
        "F_param": F,
        "PValue_param": float(p_param) if np.isfinite(p_param) else float("nan"),
        "df1": int(df_num),
        "df2": int(df_den),
        "F_perm": float(F_perm) if np.isfinite(F_perm) else float("nan"),
        "P_perm": float(p_perm) if np.isfinite(p_perm) else float("nan"),
        "eta2": eta2,
        "eta2_ci_low": lo,
        "eta2_ci_high": hi,
        "omega2": _omega2_from_f(F, df_num, df_den),
        "lead_snp_r2": _lead_r2(lead_dose, e, Q0, W),
    }
