"""PC-selection diagnostics -- REPORT the genotype-PCA eigenvalue spectrum; NEVER select.

(D-FINAL): the lambda-GC auto-PC selector was removed, and the conventional-criteria table
(Kaiser / Marchenko-Pastur edge / cumulative variance / broken-stick / Horn's parallel analysis /
Tracy-Widom) was removed as well -- on genotype data those counts are not usable as covariate counts and
read as decision support they are not. TRACE ships one documented fixed default (``--n-pcs``) plus this
spectrum diagnostic, which REPORTS the eigenvalue spectrum (variance explained per PC + cumulative) and
NEVER sets the PC count.

``compute_pc_diagnostics`` returns only ``{"spectrum", "meta"}`` -- no criterion, no ``k``, no recommended
count. It runs ONE eigendecomposition on the LD-pruned genotype-PCA matrix (via TRACE's own
``_compute_pcs_full_impl``; zero model fits).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


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


def compute_pc_diagnostics(Z_for_pca: np.ndarray, n: int, m: int, prune_params: dict | None = None,
                           spectrum_depth: int = 20) -> dict:
    """Compute the eigenvalue-spectrum diagnostic + meta. REPORTS only -- returns no PC count and mutates nothing.

    Returns ``dict(spectrum, meta)`` where ``spectrum`` is a DataFrame (rank / eigenvalue / % of trace /
    cumulative %) and ``meta`` records n, m, trace, trace/m and the LD-prune params. NO criterion, NO ``k``,
    NO recommended count -- TRACE uses the fixed ``--n-pcs``; this reports the genotype-PCA spectrum only.
    """
    ev = full_spectrum(Z_for_pca, n, m)
    trace = float(ev.sum())
    spectrum = pc_spectrum_table(ev, depth=spectrum_depth)
    meta = {
        "n_samples": int(n),
        "m_markers_pruned": int(m),
        "n_eigenvalues": int(ev.size),
        "trace": round(trace, 4),
        "trace_over_m": round(trace / m, 4) if m else None,
        "normalisation": "spectrum reported as proportion of trace (raw eigenvalues; no normalisation)",
        "ld_prune_params": prune_params or {},
        "selects": False,  # invariant: diagnostics never set n_pcs
    }
    return {"spectrum": spectrum, "meta": meta}
