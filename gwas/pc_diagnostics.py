"""PC-selection diagnostics -- REPORT the eigenvalue spectrum + conventional criteria; NEVER select.

R1.4 (D-R1.4-FINAL-1/-2): the lambda-GC auto-PC selector was removed. TRACE ships one documented fixed default
(``--n-pcs``) plus these diagnostics, which REPORT what conventional PC-selection rules would imply and never set the
PC count. A diagnostic that ranks or recommends is the selector under another name, so this module deliberately:

  * labels NOTHING "recommended", "best", "optimal", or "suggested";
  * NEVER returns or mutates ``n_pcs`` -- the "k implied" column is informational only (TRACE uses ``--n-pcs``);
  * runs ZERO model fits (it operates on the genotype PCA spectrum, not any GWAS scan).

Normalisation (load-bearing). The per-SNP VanRaden divisor assumes HWE variance ``2p(1-p)``; inbred dosage variance is
nearer ``4p(1-p)``, so ``trace/m`` is ~1.9 (not 1) on these panels. Threshold criteria calibrated to unit trace
(Kaiser, Marchenko-Pastur edge) MUST operate on CORRELATION-NORMALISED eigenvalues ``ev_c = ev * m / trace`` (average
per-marker eigenvalue = 1). Proportion-based criteria (cumulative variance, broken-stick) are scale-invariant and use
raw proportions. The applied normalisation and the ``trace/m`` ratio are reported in the returned meta.

The criteria require the FULL spectrum (all ``min(n-1, m)`` non-zero eigenvalues), which is more than the top-20 the
pipeline computes for the PC scores -- so this module recomputes the full spectrum once (one eigendecomposition via
TRACE's own ``_compute_pcs_full_impl``; NOT a model fit).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Labels forbidden by the R1.4 acceptance criterion -- the diagnostics must not select.
_FORBIDDEN_LABELS = ("recommended", "best", "optimal", "suggested")


def full_spectrum(Z_for_pca: np.ndarray, n: int, m: int) -> np.ndarray:
    """All non-zero PCA eigenvalues (descending) of the pruned genotype matrix, via TRACE's own PCA.

    Returns the ``min(n-1, m)`` non-zero eigenvalues (``explained_variance_``). One eigendecomposition; no model fit.
    """
    from gwas.kinship import _compute_pcs_full_impl

    k_full = int(min(int(n) - 1, int(m)))
    if k_full <= 0:
        return np.zeros(0, dtype=float)
    _, ev = _compute_pcs_full_impl(np.asarray(Z_for_pca, dtype=np.float32), max_pcs=k_full)
    if ev is None:
        return np.zeros(0, dtype=float)
    ev = np.asarray(ev, dtype=float)
    ev = ev[np.isfinite(ev)]
    return np.sort(ev)[::-1]


def pc_spectrum_table(eigenvalues: np.ndarray, depth: int = 20) -> pd.DataFrame:
    """Eigenvalue spectrum table: rank, eigenvalue, % of trace, cumulative %. Top ``depth`` ranks."""
    ev = np.asarray(eigenvalues, dtype=float)
    trace = float(ev.sum())
    if trace <= 0 or ev.size == 0:
        return pd.DataFrame(columns=["rank", "eigenvalue", "pct_of_trace", "cumulative_pct"])
    pct = 100.0 * ev / trace
    cum = np.cumsum(pct)
    d = int(min(depth, ev.size))
    return pd.DataFrame({
        "rank": np.arange(1, d + 1),
        "eigenvalue": np.round(ev[:d], 6),
        "pct_of_trace": np.round(pct[:d], 4),
        "cumulative_pct": np.round(cum[:d], 4),
    })


def _broken_stick_k(prop: np.ndarray) -> int:
    """Sequential broken-stick: keep component i while its proportion exceeds the broken-stick expectation."""
    L = len(prop)
    if L == 0:
        return 0
    bs_exp = np.array([np.sum(1.0 / np.arange(i + 1, L + 1)) for i in range(L)]) / L
    k = 0
    for i in range(L):
        if prop[i] > bs_exp[i]:
            k += 1
        else:
            break
    return int(k)


def parallel_analysis_k(Z_for_pca: np.ndarray, n: int, m: int,
                        B: int = 200, seed: int = 0, quantile: float = 95.0) -> int:
    """Horn's parallel analysis (OPT-IN; not run by default). B column-permutations of the marker matrix,
    each recomputing the spectrum; keep components whose observed eigenvalue exceeds the permutation quantile.
    Adds B eigendecompositions (NOT model fits). Reports the implied k only."""
    Z = np.asarray(Z_for_pca, dtype=np.float32)
    ev_obs = full_spectrum(Z, n, m)
    L = ev_obs.size
    if L == 0:
        return 0
    rng = np.random.default_rng(int(seed))
    perm_ev = np.empty((int(B), L), dtype=float)
    for b in range(int(B)):
        order = rng.random((Z.shape[0], Z.shape[1])).argsort(axis=0)
        Zp = np.take_along_axis(Z, order, axis=0)   # permute each column independently -> destroys structure
        evp = full_spectrum(Zp, n, m)
        perm_ev[b, :] = evp[:L] if evp.size >= L else np.pad(evp, (0, L - evp.size))
    thr = np.percentile(perm_ev, quantile, axis=0)
    k = 0
    for i in range(L):
        if ev_obs[i] > thr[i]:
            k += 1
        else:
            break
    return int(k)


def pc_criteria_table(eigenvalues: np.ndarray, n: int, m: int,
                      parallel_analysis_result: int | None = None,
                      pa_params: dict | None = None) -> pd.DataFrame:
    """Conventional-criteria table: the k each rule IMPLIES (informational; TRACE uses --n-pcs). Never selects.

    Kaiser + Marchenko-Pastur run on correlation-normalised eigenvalues (ev * m/trace); cumulative-variance +
    broken-stick are proportion-based (scale-invariant). Tracy-Widom is a documented, unimplemented gap.
    """
    ev = np.asarray(eigenvalues, dtype=float)
    trace = float(ev.sum())
    rows = []
    if trace <= 0 or ev.size == 0:
        return pd.DataFrame(columns=["criterion", "k_implied", "note"])
    ev_c = ev * (m / trace)                     # correlation scale: average per-marker eigenvalue = 1
    prop = ev / trace
    cum = np.cumsum(prop)

    kaiser = int((ev_c > 1.0).sum())
    rows.append(("Kaiser (eigenvalue > 1)", kaiser, "correlation-normalised eigenvalues (x m/trace)"))

    cv20 = int(np.searchsorted(cum, 0.20) + 1)
    cv50 = int(np.searchsorted(cum, 0.50) + 1)
    rows.append(("Cumulative variance >= 20%", cv20, "smallest k reaching 20% of trace"))
    rows.append(("Cumulative variance >= 50%", cv50, "smallest k reaching 50% of trace"))

    mp_edge = (1.0 + np.sqrt(float(n) / float(m))) ** 2
    mp = int((ev_c > mp_edge).sum())
    rows.append(("Marchenko-Pastur edge", mp,
                 f"eigenvalue (corr. scale) > {mp_edge:.3f} -- ASYMPTOTIC BOUND, NOT a significance test"))

    rows.append(("Broken-stick", _broken_stick_k(prop), "proportion-based (scale-invariant)"))

    if parallel_analysis_result is not None:
        p = pa_params or {}
        rows.append(("Parallel analysis (Horn)", int(parallel_analysis_result),
                     f"OPT-IN; B={p.get('B')}, seed={p.get('seed')}, quantile={p.get('quantile')}"))
    else:
        rows.append(("Parallel analysis (Horn)", None,
                     "OPT-IN only (permutation cost) -- enable to compute"))

    rows.append(("Tracy-Widom", None,
                 "NOT IMPLEMENTED -- documented gap (Patterson-Price-Reich 2006 normalisation, "
                 "no EIGENSOFT cross-check available); an unverified TW is worse than none"))

    # k_implied is object-typed so "not computed" (parallel analysis off) and "not
    # implemented" (Tracy-Widom) stay None rather than coercing to NaN -- keeps the
    # informational-only filtering (``is not None``) exact for consumers.
    return pd.DataFrame({
        "criterion": [r[0] for r in rows],
        "k_implied": pd.array([r[1] for r in rows], dtype=object),
        "note": [r[2] for r in rows],
    })


def compute_pc_diagnostics(Z_for_pca: np.ndarray, n: int, m: int, prune_params: dict | None = None,
                           spectrum_depth: int = 20, run_parallel_analysis: bool = False,
                           pa_B: int = 200, pa_seed: int = 0, pa_quantile: float = 95.0) -> dict:
    """Compute both diagnostic tables + meta. REPORTS only -- returns no PC count and mutates nothing.

    Returns dict(spectrum, criteria, meta) where spectrum/criteria are DataFrames and meta records n, m,
    trace, trace/m, the normalisation applied, the prune params, and (if run) the parallel-analysis settings.
    """
    ev = full_spectrum(Z_for_pca, n, m)
    trace = float(ev.sum())
    pa_result = None
    pa_params = None
    if run_parallel_analysis and ev.size:
        pa_params = {"B": int(pa_B), "seed": int(pa_seed), "quantile": float(pa_quantile)}
        pa_result = parallel_analysis_k(Z_for_pca, n, m, B=pa_B, seed=pa_seed, quantile=pa_quantile)
    spectrum = pc_spectrum_table(ev, depth=spectrum_depth)
    criteria = pc_criteria_table(ev, n, m, parallel_analysis_result=pa_result, pa_params=pa_params)
    meta = {
        "n_samples": int(n),
        "m_markers_pruned": int(m),
        "n_eigenvalues": int(ev.size),
        "trace": round(trace, 4),
        "trace_over_m": round(trace / m, 4) if m else None,
        "normalisation": "Kaiser & MP on correlation-normalised eigenvalues (x m/trace); "
                         "cumulative-variance & broken-stick proportion-based",
        "ld_prune_params": prune_params or {},
        "selects": False,  # invariant: diagnostics never set n_pcs
    }
    if pa_params is not None:
        meta["parallel_analysis"] = pa_params
    return {"spectrum": spectrum, "criteria": criteria, "meta": meta}
