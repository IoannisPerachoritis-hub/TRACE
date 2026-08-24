"""LD-kNNi genotype imputation (Money et al. 2015).

Faithful implementation of LinkImpute's LD-kNNi, an OPT-IN alternative to the
default mean imputation used to build the GWAS mixed-model genotype matrix.

Reference
---------
Money D, Migicovsky Z, Gardner K, Myles S (2015) "LinkImpute: fast and accurate
genotype imputation for nonmodel organisms." G3 5:2383-2390.
doi:10.1534/g3.115.021667  (PMC4632058)

The published equations (cross-checked verbatim against the PMC full text):

  Distance, LD-restricted (Eq. 3):
      d_l(s1, s2) = c + (1/n) * SUM_{p in L(p_j), both observed} |g(s1,p) - g(s2,p)|
    * L(p_j) = the ``l`` SNPs in strongest LD (squared correlation r^2) with the
      SNP being imputed.  "LD" means correlation across the panel, NOT physical
      linkage -- no genetic/physical map, chromosome or window is used.  Ranking
      by r^2 over the whole panel is the method's central premise.
    * n = the number of SNPs actually summed: a SNP is skipped for a pair when
      EITHER sample is missing there, so n varies per pair.  Normalise by n (not
      by l) -- this is the missing-data correction, not optional.
    * c = 1.  A constant added so the distance is never zero (Eq. 2 would be
      undefined otherwise); the paper notes the value has little effect (Table S1)
      and sets it to one.

  Imputation, inverse-distance-weighted modal vote (Eq. 2):
      g_hat(s_i, p_j) = argmax_{a in {0,1,2}} SUM_{s in N} (1 / d_l(s_i, s)) * I(g(s,p_j) = a)
    * N = the ``k`` nearest samples to s_i that have a KNOWN genotype at p_j.
      Samples missing at the target SNP cannot vote.
    * Votes are weighted by 1/distance (nearer samples count more) -- a weighted
      modal average, not a mean and not an unweighted majority.
    * The output is a discrete class in {0,1,2}, not a fractional dosage.

Genotype-coding convention
--------------------------
TRACE codes dosage as the ALTERNATE-allele count (see gwas/qc.py), so class 0 is
homozygous-REFERENCE, not homozygous-major as the paper's wording implies.  This
is harmless for LD-kNNi: both equations use only |g1 - g2| and an equality
indicator, and both are invariant to swapping the 0/2 labels.  We deliberately do
NOT add a minor-allele flip -- it would change nothing here and would diverge from
the rest of the pipeline.  Input is guaranteed in [0, 2] with NaN for missing.

Deviations from the paper (documented)
--------------------------------------
* Tie-break (paper is silent): among alleles with equal summed weight, choose the
  one with the smallest summed neighbour distance; ties there break to the lowest
  class index.  Deterministic.
* Un-votable entry (no candidate shares an observed SNP in L, so every distance is
  undefined): fall back to the SNP's observed MODE (kept discrete).  An all-missing
  SNP (no observed genotype at all) is filled with 0.  Both guarantee a NaN-free,
  discrete output.  Such SNPs are QC-removed upstream (>10% missing fails), so this
  is defensive.
* Missing-data handling in the LD (r^2) ranking: the paper does not specify it.  We
  use exact PAIRWISE-COMPLETE correlation (each pair's r^2 over the samples observed
  at BOTH SNPs), which introduces no bias.  A mean-fill-then-correlate shortcut
  would bias the ranking toward high-missingness markers; we avoid it.
"""
import numpy as np


def _mode_or_zero(col):
    """Discrete fallback: the mode of a SNP's observed genotypes, else 0.

    Ties break to the lowest class index (np.unique returns sorted values)."""
    obs = col[np.isfinite(col)]
    if obs.size == 0:
        return 0.0
    vals, counts = np.unique(np.rint(obs).astype(np.int64), return_counts=True)
    return float(vals[int(np.argmax(counts))])


def _ld_rank_topl(G, l=20, r2_cap=None, block_size=512, min_pair_n=20):
    """For each SNP, the indices of its top-``l`` SNPs by pairwise-complete r^2.

    Blocked to avoid the full m x m matrix (7.7 GB at m=44k).  For a block of
    target SNPs, every pairwise quantity is a (block, m) product of the
    NaN-zeroed data ``X0`` and the observed mask ``M``:

        n   = M_B  @ M^T          (co-observed sample counts)
        Sx  = X0_B @ M^T          Sy  = M_B  @ X0^T
        Sxx = X2_B @ M^T          Syy = M_B  @ X2^T
        Sxy = X0_B @ X0^T
        r^2 = (n*Sxy - Sx*Sy)^2 / [(n*Sxx - Sx^2)(n*Syy - Sy^2)]

    Pairs with fewer than ``min_pair_n`` co-observed samples get r^2 = 0 (as in
    ``gwas.ld.pairwise_r2``'s contract).  ``r2_cap`` (if set) drops near-duplicate
    proxies (r^2 >= cap).  Self-correlation is excluded.  Returns ``(topl, topl_r2)``:
    an (m, l) int64 index array (-1 where no valid partner) and the parallel (m, l)
    float32 r^2 of each selected partner (NaN where the index is -1).
    """
    G = np.asarray(G, dtype=np.float32)
    n_samples, m = G.shape
    Gt = np.ascontiguousarray(G.T)                       # (m, n_samples)
    M = np.isfinite(Gt).astype(np.float32)
    X0 = np.where(np.isfinite(Gt), Gt, 0.0).astype(np.float32)
    X2 = (X0 * X0).astype(np.float32)
    topl = np.full((m, l), -1, dtype=np.int64)
    topl_r2 = np.full((m, l), np.nan, dtype=np.float32)
    kth = min(l, m - 1)
    for start in range(0, m, block_size):
        end = min(start + block_size, m)
        Xb, Mb, X2b = X0[start:end], M[start:end], X2[start:end]
        n = Mb @ M.T
        Sx, Sy = Xb @ M.T, Mb @ X0.T
        Sxx, Syy = X2b @ M.T, Mb @ X2.T
        Sxy = Xb @ X0.T
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = n * Sxy - Sx * Sy
            vx = n * Sxx - Sx * Sx
            vy = n * Syy - Sy * Sy
            denom = vx * vy
            r2 = np.where(denom > 0, (cov * cov) / denom, 0.0).astype(np.float32)
        r2[n < min_pair_n] = 0.0
        r2[np.arange(end - start), np.arange(start, end)] = -1.0   # exclude self
        if r2_cap is not None:
            r2[r2 >= r2_cap] = -1.0
        part = np.argpartition(-r2, kth, axis=1)[:, :l]            # top-l (unordered)
        r2_sel = np.take_along_axis(r2, part, axis=1)
        valid = r2_sel > 0.0
        topl[start:end] = np.where(valid, part, -1)               # drop non-partners
        topl_r2[start:end] = np.where(valid, r2_sel, np.nan)
        del n, Sx, Sy, Sxx, Syy, Sxy, r2, part, r2_sel, valid
    return topl, topl_r2


def _ld_distances(P_a, P_b, c=1.0):
    """Pairwise LD-restricted taxicab distance (Eq. 3) between the rows of ``P_a`` and
    ``P_b`` over their shared predictor columns.  NaN = missing; a column is skipped
    for a pair when either side is missing there; ``d = c + (1/n) * Σ|g_a - g_b|`` with
    ``n`` = the number of co-observed columns (normalise by n, NOT by the column count).
    Returns a (len(P_a), len(P_b)) array; a pair with no co-observed column gets inf."""
    Ma, Mb = np.isfinite(P_a), np.isfinite(P_b)
    Pa0, Pb0 = np.where(Ma, P_a, 0.0), np.where(Mb, P_b, 0.0)
    both = Ma[:, None, :] & Mb[None, :, :]
    n_ab = both.sum(axis=2)
    absdiff = np.where(both, np.abs(Pa0[:, None, :] - Pb0[None, :, :]), 0.0).sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_ab > 0, c + absdiff / n_ab, np.inf)


def _vote(votes, d_nn, geno_nn):
    """argmax with the documented deterministic tie-break."""
    mx = votes.max()
    tied = np.where(votes >= mx - 1e-12)[0]
    if tied.size == 1:
        return float(tied[0])
    sums = np.array([d_nn[geno_nn == a].sum() if np.any(geno_nn == a) else np.inf
                     for a in tied])
    return float(tied[int(np.argmin(sums))])       # sorted tied -> lowest class on a sum tie


def ld_knni(G, k=5, l=20, c=1.0, r2_cap=None, random_state=0, _topl=None):
    """LD-kNNi imputation of a genotype dosage matrix.

    Parameters
    ----------
    G : ndarray (n_samples, n_snps), float, NaN for missing (i.e. geno_dosage_raw).
    k : neighbours voting per missing call (default 5, paper Fig S1).
    l : SNPs (by r^2) restricting the distance (default 20, paper Fig S1).
    c : distance constant (default 1.0, paper).
    r2_cap : optional; drop LD predictors with r^2 >= this (near-duplicate proxies).
    random_state : reserved; the algorithm is deterministic (ties break by rule).

    Returns
    -------
    ndarray (n_samples, n_snps) float32: observed entries copied unchanged, missing
    entries filled with a discrete class in {0,1,2}.
    """
    G = np.asarray(G, dtype=np.float64)
    n_samples, m = G.shape
    miss = np.isnan(G)
    if not miss.any():
        return G.astype(np.float32)                # complete input returned unchanged
    out = G.copy()
    obs = ~miss
    # _topl: an optional precomputed (m, l) ranking (e.g. from the complete matrix,
    # reused across masking patterns in the power pilot -- top-l LD is robust to a few
    # percent masking).  Production callers leave it None -> ranked on the actual matrix.
    topl = _topl if _topl is not None else _ld_rank_topl(G, l=l, r2_cap=r2_cap)[0]
    for j in range(m):
        rows_miss = np.where(miss[:, j])[0]
        if rows_miss.size == 0:
            continue
        cand = np.where(obs[:, j])[0]
        L = topl[j]
        L = L[L >= 0]
        if cand.size == 0 or L.size == 0:
            out[rows_miss, j] = _mode_or_zero(G[:, j])
            continue
        Pm, Pc = G[np.ix_(rows_miss, L)], G[np.ix_(cand, L)]        # (n_miss, |L|), (n_cand, |L|)
        d = _ld_distances(Pm, Pc, c=c)                             # d_l; n=0 pair -> inf
        cand_geno = np.rint(G[cand, j]).astype(np.int64)
        col_mode = None
        for ii in range(rows_miss.size):
            di = d[ii]
            finite = np.where(np.isfinite(di))[0]
            if finite.size == 0:
                if col_mode is None:
                    col_mode = _mode_or_zero(G[:, j])
                out[rows_miss[ii], j] = col_mode
                continue
            kk = min(k, finite.size)
            nn = finite[np.argpartition(di[finite], kk - 1)[:kk]]  # k nearest usable candidates
            w = 1.0 / di[nn]
            g_nn = cand_geno[nn]
            votes = np.array([w[g_nn == a].sum() for a in (0, 1, 2)])
            out[rows_miss[ii], j] = _vote(votes, di[nn], g_nn)
    return out.astype(np.float32)


def imputation_selfcheck(G, n_mask=2000, k=5, l=20, chroms=None, random_state=0):
    """Mask known genotypes, impute both ways, and report which wins on THIS panel.

    No published benchmark covers selfing crops, so this measures mean vs LD-kNNi
    directly.  Mean imputation is ROUNDED to the nearest class to score discrete
    concordance (stated here and in the returned dict).

    Returns a dict with, per method, discrete concordance, dosage r^2 vs truth, and
    the allele-frequency (MAF) deviation from truth; plus the LD-availability
    diagnostics: median top-l r^2 and (if ``chroms`` given) the fraction of top-l
    predictors on the same chromosome.
    """
    rng = np.random.default_rng(random_state)
    G = np.asarray(G, dtype=np.float64)
    known = np.argwhere(np.isfinite(G))
    n_mask = int(min(n_mask, known.shape[0]))
    pick = known[rng.choice(known.shape[0], n_mask, replace=False)]
    ri, ci = pick[:, 0], pick[:, 1]
    truth = np.rint(G[ri, ci]).astype(np.int64)

    Gm = G.copy()
    Gm[ri, ci] = np.nan

    col_mean = np.nanmean(Gm, axis=0)
    mean_pred = col_mean[ci]
    mean_disc = np.rint(mean_pred).astype(np.int64)
    Gmean = Gm.copy()
    _nan = np.isnan(Gmean)
    Gmean[_nan] = np.take(col_mean, np.where(_nan)[1])

    Gk = ld_knni(Gm, k=k, l=l)
    ldk_pred = np.rint(Gk[ri, ci]).astype(np.int64)

    def _r2(pred, tr):
        if np.std(pred) == 0 or np.std(tr) == 0:
            return float("nan")
        r = np.corrcoef(pred.astype(float), tr.astype(float))[0, 1]
        return float(r * r)

    def _maf(mat):
        af = np.nanmean(mat, axis=0) / 2.0
        return np.minimum(af, 1.0 - af)

    maf_true = _maf(G)
    maf_dev_mean = float(np.nanmean(np.abs(_maf(Gmean) - maf_true)))
    maf_dev_ldk = float(np.nanmean(np.abs(_maf(Gk) - maf_true)))
    maf_max_mean = float(np.nanmax(np.abs(_maf(Gmean) - maf_true)))
    maf_max_ldk = float(np.nanmax(np.abs(_maf(Gk) - maf_true)))

    topl, topl_r2 = _ld_rank_topl(Gm, l=l)
    valid = topl >= 0
    r2_med = float(np.nanmedian(topl_r2[valid])) if valid.any() else float("nan")
    same_chr = float("nan")
    if valid.any() and chroms is not None:
        chroms = np.asarray(chroms)
        # same-chromosome fraction of the realised top-l predictors (diagnostic only;
        # ld_knni itself never uses chromosome)
        tgt = np.repeat(chroms[:, None], topl.shape[1], axis=1)[valid]
        prt = chroms[topl[valid]]
        same_chr = float((tgt == prt).mean())

    return {
        "n_masked": n_mask,
        "note": "mean imputation is rounded to the nearest class to score discrete concordance",
        "mean": {
            "discrete_concordance": float((mean_disc == truth).mean()),
            "dosage_r2": _r2(mean_pred, truth),
            "maf_dev_mean": maf_dev_mean,
            "maf_dev_max": maf_max_mean,
        },
        "ldknni": {
            "discrete_concordance": float((ldk_pred == truth).mean()),
            "dosage_r2": _r2(ldk_pred, truth),
            "maf_dev_mean": maf_dev_ldk,
            "maf_dev_max": maf_max_ldk,
        },
        "ld_availability": {
            "median_topl_r2": r2_med,          # filled by the caller/benchmark when needed
            "frac_topl_same_chr": same_chr,
        },
        "winner_concordance": "ldknni" if (ldk_pred == truth).mean() > (mean_disc == truth).mean() else "mean",
    }
