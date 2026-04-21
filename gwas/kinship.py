import logging
from pysnptools.kernelreader import KernelData as PSKernelData
from sklearn.decomposition import PCA
import numpy as np

_log = logging.getLogger(__name__)
try:
    import streamlit as st
except ImportError:
    st = None
from gwas.utils import _mean_impute_cols

def _ld_prune_for_grm_by_chr_bp(
    chroms, positions, Z,
    r2_thresh=0.2,
    window_bp=500_000,
    step_bp=100_000,
    return_mask: bool = False
):
    """
    Chromosome-aware LD pruning using PHYSICAL windows (bp).
    If return_mask=True, returns (Z_pruned, keep_mask_over_original_snps).
    """
    chroms = np.asarray(chroms).astype(str)
    positions = np.asarray(positions).astype(int)
    Z = np.asarray(Z, float)

    keep_global = np.zeros(Z.shape[1], dtype=bool)

    for ch in np.unique(chroms):
        if str(ch) == "ALT":
            continue

        idx_chr = np.where(chroms == ch)[0]
        if idx_chr.size < 2:
            keep_global[idx_chr] = True
            continue

        idx_chr = idx_chr[np.argsort(positions[idx_chr])]
        pos_chr = positions[idx_chr]
        Zc = Z[:, idx_chr]
        m = Zc.shape[1]
        keep = np.ones(m, dtype=bool)

        start_pos = int(pos_chr.min())
        end_pos = int(pos_chr.max())

        for w_start in range(start_pos, end_pos + 1, int(step_bp)):
            w_end = w_start + int(window_bp)
            loc = np.where((pos_chr >= w_start) & (pos_chr <= w_end) & keep)[0]
            if loc.size < 2:
                continue

            # -------------------------------------------------
            # SAFETY GUARD: cap SNPs per LD window to avoid O(m^2) RAM blowups
            # -------------------------------------------------
            MAX_LOC_SNPS = 800
            if loc.size > MAX_LOC_SNPS:
                step = max(1, loc.size // MAX_LOC_SNPS)
                loc = loc[::step]

            Zw = Zc[:, loc]
            Zw = _mean_impute_cols(Zw)
            v = np.var(Zw, axis=0)
            okv = v > 1e-12
            if okv.sum() < 2:
                continue

            Zw = Zw[:, okv]
            loc2 = loc[okv]

            # Faster correlation matrix
            Zw = Zw - Zw.mean(axis=0, keepdims=True)
            sd = Zw.std(axis=0, keepdims=True)
            sd[sd == 0] = 1.0
            Zw = Zw / sd

            C = (Zw.T @ Zw) / (Zw.shape[0] - 1)
            C = np.clip(C, -1.0, 1.0)

            if not np.isfinite(C).all():
                continue
            R2 = C * C

            for i in range(R2.shape[0]):
                if not keep[loc2[i]]:
                    continue
                high = np.where(R2[i, :] > r2_thresh)[0]
                high = high[high > i]
                keep[loc2[high]] = False

        keep_global[idx_chr[keep]] = True

    if keep_global.sum() < 10:
        if return_mask:
            return Z, np.ones(Z.shape[1], dtype=bool)
        return Z

    Zp = Z[:, keep_global]
    if return_mask:
        return Zp, keep_global
    return Zp

def _build_loco_kernels_impl(iid, Z_grm, chroms_grm, K_base=None):
    """Pure computation for LOCO kernel construction. No Streamlit dependency."""
    iid = np.asarray(iid).astype(str)
    chroms_grm = np.asarray(chroms_grm).astype(str)

    # ----------------------------
    # CRITICAL: basis alignment
    # ----------------------------
    if Z_grm is None:
        raise ValueError("Z_grm is None: LOCO requires GRM-basis genotype matrix.")
    Z_grm = np.asarray(Z_grm, dtype=np.float32, order="C")

    if chroms_grm.shape[0] != Z_grm.shape[1]:
        raise ValueError(
            f"chroms_grm length ({chroms_grm.shape[0]}) != Z_grm n_snps ({Z_grm.shape[1]}). "
            "This indicates a cache mismatch or pruning metadata drift."
        )

    # ----------------------------
    # Base kernel
    # ----------------------------
    if K_base is not None:
        K_base = np.asarray(K_base, dtype=np.float32, order="C")
        if K_base.shape[0] != iid.shape[0] or K_base.shape[1] != iid.shape[0]:
            raise ValueError("K_base shape does not match iid sample count.")
    else:
        K_base = _build_grm_from_Z(Z_grm)

    K0 = PSKernelData(iid=iid, val=K_base)

    K_by_chr = {}
    diagnostics_by_chr = {}

    m_total = int(Z_grm.shape[1])

    for ch in [c for c in np.unique(chroms_grm) if str(c) != "ALT"]:

        on_chr = (chroms_grm == ch)
        off_chr = ~on_chr

        m_on = int(on_chr.sum())
        m_off = int(off_chr.sum())

        diag = {
            "m_grm_total": m_total,
            "m_grm_on_chr": m_on,
            "m_grm_off_chr": m_off,
            "status": "ok",
        }

        # If off-chr SNP count is too small, LOCO becomes unstable → fall back to K0
        if m_off < 10:
            K_by_chr[str(ch)] = K0
            diag["status"] = "fallback_too_few_off_chr_snps"
            diagnostics_by_chr[str(ch)] = diag
            continue

        Z_off = Z_grm[:, off_chr]
        K_chr = _build_grm_from_Z(Z_off)

        # Finite guard
        if not np.isfinite(K_chr).all():
            K_by_chr[str(ch)] = K0
            diag["status"] = "fallback_nonfinite_kernel"
        else:
            K_by_chr[str(ch)] = PSKernelData(iid=iid, val=K_chr.astype(np.float32, copy=False))

        diagnostics_by_chr[str(ch)] = diag

    return K0, K_by_chr, diagnostics_by_chr


def build_loco_kernels_cached(iid, Z_grm, chroms_grm, K_base=None):
    """Cached wrapper for _build_loco_kernels_impl."""
    return _build_loco_kernels_impl(iid, Z_grm, chroms_grm, K_base)

if st is not None:
    build_loco_kernels_cached = st.cache_resource(
        show_spinner="Building LOCO kernels (cached)…"
    )(build_loco_kernels_cached)


def _compute_pcs_full_impl(Z_for_pca: np.ndarray, max_pcs: int = 20):
    """Pure PCA computation. No Streamlit dependency.

    Returns
    -------
    tuple of (pcs, eigenvalues) where pcs is (n, max_pcs) float32
    and eigenvalues is (max_pcs,) float64 (explained variance per PC).
    Returns (None, None) if max_pcs <= 0.
    """
    Z = np.array(Z_for_pca, dtype=np.float32, order="C", copy=True)

    max_pcs = int(max_pcs)
    if max_pcs <= 0:
        return None, None

    pca = PCA(n_components=max_pcs, svd_solver="auto", random_state=0)
    pcs_full = pca.fit_transform(Z)

    return pcs_full.astype(np.float32, copy=False), pca.explained_variance_


def compute_pcs_full_cached(Z_for_pca: np.ndarray, max_pcs: int = 20):
    """Cached wrapper for _compute_pcs_full_impl."""
    return _compute_pcs_full_impl(Z_for_pca, max_pcs)

if st is not None:
    compute_pcs_full_cached = st.cache_resource(
        show_spinner="Computing PCA (cached)…"
    )(compute_pcs_full_cached)


def _build_grm_from_Z(Z: np.ndarray) -> np.ndarray:
    """
    Compute normalized GRM from standardized genotype matrix Z.
    Returns K (n×n), normalized so mean(diag) ≈ 1.
    """
    # Use float64 for matrix multiply to avoid accumulation error in large panels,
    # then downcast result to float32 for memory efficiency.
    Z = np.asarray(Z, dtype=np.float64)
    m = Z.shape[1]
    if m < 10:
        _log.warning("_build_grm_from_Z: only %d SNPs — returning identity matrix", m)
        return np.eye(Z.shape[0], dtype=np.float32)

    K = (Z @ Z.T) / float(m)
    dmean = float(np.mean(np.diag(K)))
    if dmean > 0 and np.isfinite(dmean):
        K /= dmean
    return K.astype(np.float32, copy=False)

def _standardize_geno_for_grm(G: np.ndarray) -> np.ndarray:
    """
    Standardize imputed genotype matrix for GRM:
    center by 2p, scale by sqrt(2p(1-p)), NaN→0.
    """
    G = np.asarray(G, dtype=np.float32)
    p = np.nanmean(G, axis=0) / 2.0
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    denom = np.sqrt(2.0 * p * (1.0 - p))
    denom[~np.isfinite(denom) | (denom == 0)] = 1.0
    Z = (G - 2.0 * p) / denom
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return Z.astype(np.float32, copy=False)