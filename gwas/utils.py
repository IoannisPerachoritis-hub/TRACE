
import numpy as np
import pandas as pd
try:
    import streamlit as st
except ImportError:
    st = None
import hashlib
from scipy import stats
from pandas.util import hash_pandas_object

class PhenoData:
    def __init__(self, iid, val):
        iid = np.asarray(iid)
        if iid.ndim == 1:
            iid = np.c_[iid, iid]
        self.iid = iid.astype(str)
        self.val = np.asarray(val, float).reshape(-1, 1)
        self.sid = np.array(["trait"], dtype=str)
        self.sid_count = 1
        self.iid_count = self.iid.shape[0]

    @property
    def shape(self):
        return self.val.shape

    def read(self, **kwargs):
        return self

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index = index[0]
        iid_sub = self.iid[index, :] if isinstance(index, slice) else self.iid[np.asarray(index)]
        val_sub = np.asarray(self.val[index, :], float).reshape(-1, self.val.shape[1])
        return PhenoData(iid_sub, val_sub)


class CovarData:
    def __init__(self, iid, val, names=None):
        iid = np.asarray(iid)
        if iid.ndim == 1:
            iid = np.c_[iid, iid]
        self.iid = iid.astype(str)
        V = np.asarray(val, float)
        if V.ndim == 1:
            V = V.reshape(-1, 1)
        self.val = V
        self.sid = np.asarray(names or [f"cov{i}" for i in range(V.shape[1])], dtype=str)
        self.sid_count = V.shape[1]
        self.iid_count = self.iid.shape[0]

    def read(self, **kwargs):
        return self

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index = index[0]
        iid_sub = self.iid[index, :] if isinstance(index, slice) else self.iid[np.asarray(index)]
        val_sub = np.asarray(self.val[index, :], float).reshape(-1, self.val.shape[1])
        return CovarData(iid_sub, val_sub, names=list(self.sid))

def _ensure_2d(X):
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X

def hash_bytes(b: bytes, digest_size=32) -> str:
    return hashlib.blake2b(b, digest_size=digest_size).hexdigest()

def hash_df(df: pd.DataFrame) -> str:
    """
    Stable FULL DataFrame hash (no row sampling).
    Safe for phenotype caching (avoids stale caches when only some rows change).
    """
    if df is None:
        return "none"

    d = df.copy()
    d.index = d.index.astype(str)
    d.columns = d.columns.astype(str)

    # Hash per-cell content + index, platform-independent
    row_hash = hash_pandas_object(d, index=True).values

    payload = (
        str(d.shape).encode()
        + "|".join(d.columns).encode()
        + "|".join(d.index.astype(str)).encode()
        + row_hash.tobytes()
    )
    return hashlib.blake2b(payload, digest_size=32).hexdigest()

def hash_numpy(arr, n_sample=50000, _small_threshold_bytes=50 * 1024 * 1024):
    """
    Robust hash for numpy-like arrays.

    For arrays under 50 MB, hashes the full byte content to avoid collisions.
    For larger arrays, uses a deterministic stride sample plus the first/last
    1000 elements as anchors to reduce false-collision probability.
    """
    a = np.asarray(arr)
    flat = a.ravel()

    if a.nbytes <= _small_threshold_bytes:
        # Small array: hash full content — no collision risk
        sample = flat
    elif flat.size > n_sample:
        stride_idx = np.linspace(0, flat.size - 1, n_sample, dtype=int)
        anchor_head = np.arange(min(1000, flat.size))
        anchor_tail = np.arange(max(0, flat.size - 1000), flat.size)
        idx = np.unique(np.concatenate([anchor_head, stride_idx, anchor_tail]))
        sample = flat[idx]
    else:
        sample = flat

    if np.issubdtype(sample.dtype, np.number):
        # guard against all-NaN sample
        if np.isfinite(sample).any():
            mn = np.nanmin(sample)
            mx = np.nanmax(sample)
            m1 = np.nanmean(sample)
            s1 = np.nanstd(sample)
        else:
            mn = mx = m1 = s1 = np.nan

        payload = (
            np.asarray(sample, dtype=a.dtype).tobytes()
            + str(a.shape).encode()
            + str(a.dtype).encode()
            + str(m1).encode()
            + str(s1).encode()
            + str(mn).encode()
            + str(mx).encode()
        )
    else:
        sample_str = np.asarray(sample).astype(str)
        payload = (
            "|".join(sample_str).encode("utf-8")
            + str(a.shape).encode()
            + str(a.dtype).encode()
        )

    return hashlib.blake2b(payload, digest_size=16).hexdigest()
def hash_kerneldata(kd, n_sample=50000):
    """
    Hash a PSKernelData object (iid + numeric content sample).
    """
    if kd is None:
        return ("none",)
    try:
        val = kd.val
        iid = kd.iid
    except (ValueError, TypeError, AttributeError):
        return ("invalid_kernel",)

    iid_sample = tuple(
        map(tuple, np.asarray(iid).astype(str)[: min(20, len(iid))])
    )

    return ("KernelData", iid_sample, hash_numpy(val, n_sample=n_sample))

def hash_k_by_chr(K_by_chr: dict, n_sample=20000):
    """
    Stable hash summary for LOCO kernels dict.
    """
    if not K_by_chr:
        return ("none",)
    keys = sorted([str(k) for k in K_by_chr.keys()])
    parts = ["K_by_chr", tuple(keys)]
    for k in keys:
        parts.append((k, hash_kerneldata(K_by_chr[k], n_sample=n_sample)))
    return tuple(parts)

def put_array_in_session(arr: np.ndarray, prefix: str, *parts) -> str:
    """
    Store a heavy numpy array in session_state under a deterministic key.
    Returns a lightweight string key you can pass into cached functions.
    """
    key = "ARRKEY::" + prefix + "::" + "|".join(map(str, parts))
    st.session_state[key] = np.asarray(arr)  # store canonical array
    return key

# Store non-hashable kernel objects (e.g. FastLMM KernelData)
def put_kernel_in_session(K, prefix, *parts):
    key = "KERNELKEY::" + prefix + "::" + "|".join(map(str, parts))
    st.session_state[key] = K
    return key

def put_object_in_session(obj, prefix, *parts):
    key = "OBJKEY::" + prefix + "::" + "|".join(map(str, parts))
    st.session_state[key] = obj
    return key

def stable_seed(*parts, mod=(2**32 - 1)):
    """Deterministic seed independent of Python session."""
    s = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.blake2b(s, digest_size=8).hexdigest(), 16) % mod

def bump_data_version():
    """Increment the global data version counter in session_state.

    Call after phenotype load, GWAS completion, or any operation that
    invalidates downstream caches (LD, etc.).
    """
    st.session_state["_data_version"] = st.session_state.get("_data_version", 0) + 1


def check_data_version(page_key: str) -> bool:
    """Check if page's data is stale compared to the global version.

    Returns True if data is current, False if stale (shows a warning).
    The page's last-seen version is updated automatically on each call.
    """
    current = st.session_state.get("_data_version", 0)
    page_ver_key = f"_page_version_{page_key}"
    last_seen = st.session_state.get(page_ver_key, None)

    if last_seen is not None and last_seen < current:
        st.warning(
            "Upstream data has changed since this page last ran. "
            "Results below may be stale — re-run the analysis to refresh."
        )
        st.session_state[page_ver_key] = current
        return False

    st.session_state[page_ver_key] = current
    return True


def trait_was_transformed(trait: str) -> bool:
    """
    Returns True if the current trait was transformed in Section 2b.
    Uses session_state['pheno_transformations'] (columns_transformed list).
    """
    info = st.session_state.get("pheno_transformations", {}) or {}
    cols = set(map(str, info.get("columns_transformed", []) or []))
    return str(trait) in cols

def _mean_impute_cols(Z: np.ndarray) -> np.ndarray:
    """
    Column-wise mean impute for NaNs. Used ONLY for PCA/GRM pruning steps
    (not for association testing).
    """
    Z = np.asarray(Z, float)
    if not np.isnan(Z).any():
        return Z

    col_means = np.nanmean(Z, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)

    idx = np.where(np.isnan(Z))
    Z2 = Z.copy()
    Z2[idx] = col_means[idx[1]]
    return Z2

def canonicalize_chr(x):
    """
    Canonicalize chromosome labels.
    Works on scalars AND array-like (Series/Index/list/ndarray).
    """
    from annotation import canon_chr as canon_chr_scalar
    if isinstance(x, (pd.Series, pd.Index, list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=object)
        return np.asarray([canon_chr_scalar(v) for v in arr], dtype=object)
    return canon_chr_scalar(x)

def align_pheno_to_geno(pheno_df, geno_df, trait_col):
    ph = pheno_df.reindex(geno_df.index).copy()
    if trait_col not in ph.columns:
        raise KeyError(f"{trait_col} not in phenotype columns")
    y = pd.to_numeric(ph[trait_col], errors="coerce").astype(float)
    keep = np.isfinite(y.to_numpy())
    ph = ph.iloc[np.where(keep)[0]].copy()
    y = y.to_numpy()[keep]
    return ph, y, keep

def resolve_trait_column(trait_name: str, df: pd.DataFrame) -> str:
    """
    Return the actual column name in df that corresponds to trait_name.
    Handles whitespace + case mismatches safely.
    Raises a clear error if not found.
    """
    import difflib

    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Phenotype table is missing or not a DataFrame.")

    # normalize df columns once
    df = df.copy()
    df.columns = df.columns.astype(str)

    t = str(trait_name).strip()

    # 1) exact match
    if t in df.columns:
        return t

    # 2) match after stripping df columns
    stripped_map = {c.strip(): c for c in df.columns}
    if t in stripped_map:
        return stripped_map[t]

    # 3) case-insensitive match on stripped names
    lower_map = {c.strip().lower(): c for c in df.columns}
    if t.lower() in lower_map:
        return lower_map[t.lower()]

    # 4) helpful suggestions
    candidates = list(df.columns)
    close = difflib.get_close_matches(t, candidates, n=8, cutoff=0.6)

    raise KeyError(
        f"Trait '{trait_name}' not found in phenotype columns.\n"
        f"First 30 columns: {candidates[:30]}\n"
        f"Close matches: {close}"
    )

def _rank_int_1d(x):
    """
    Rank-based inverse normal transform (INT) for a 1D array-like.
    Keeps NaNs.
    """
    x = np.asarray(x, float).reshape(-1)
    out = np.full_like(x, np.nan, dtype=float)
    mask = np.isfinite(x)
    if mask.sum() < 3:
        return x
    ranks = stats.rankdata(x[mask], method="average")
    u = (ranks - 0.5) / mask.sum()
    out[mask] = stats.norm.ppf(u)
    return out
