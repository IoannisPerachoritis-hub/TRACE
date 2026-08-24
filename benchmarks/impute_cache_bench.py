"""Measurement harness for the persistent imputation cache (P1) + the
persist="disk" round-trip (P2) + the raw-export NaN contract (P3).

Not a pytest -- a standalone measurement script run once, reporting numbers.
Headless (no Streamlit runtime), `st.cache_data(persist="disk")` degrades to an
in-memory cache within the process, so P1 (a cache HIT within one run) is
measurable directly.  persist="disk" itself is a headless no-op through the
decorator, so P2 drives `LocalDiskCacheStorage` directly with raw bytes.

Run:  .venv/Scripts/python.exe benchmarks/impute_cache_bench.py
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gwas.utils import hash_bytes
from gwas.qc import _impute_cached, _pipeline_build_geno_matrices

TOMATO = Path("benchmarks/qc_data/tomato_locule_number/QC_genotype_matrix.csv")
MISS_RATE = 0.05
SEED = 42


def _sshash(df: pd.DataFrame) -> str:
    return hash_bytes(repr(tuple(sorted(df.index.astype(str)))).encode())


def _inject_missing(df: pd.DataFrame, rate: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    G = df.to_numpy(dtype=float).copy()
    mask = rng.random(G.shape) < rate
    G[mask] = np.nan
    out = pd.DataFrame(G, index=df.index, columns=df.columns)
    return out, int(mask.sum())


def _call(df: pd.DataFrame, method: str):
    """Same 14-arg call the pipeline makes."""
    return _impute_cached(
        df, "vcfhash", _sshash(df), 0.05, 0.10, 5, 0.20, 0.0,
        False, "none", None, method, 5, 20)


def _eq_tuple(a, b) -> bool:
    for x, y in zip(a, b):
        x, y = np.asarray(x), np.asarray(y)
        if x.dtype.kind == "f":
            if not np.array_equal(x, y, equal_nan=True):
                return False
        elif not np.array_equal(x, y):
            return False
    return True


def p1_timing(df_miss: pd.DataFrame) -> None:
    print("\n=== P1 -- cache timing (ldknni, tomato, 5% injected missingness) ===")
    _impute_cached.clear()

    t0 = time.perf_counter()
    cold = _call(df_miss, "ldknni")
    t_cold = time.perf_counter() - t0
    print(f"  cold impute (miss)        : {t_cold:8.2f} s   "
          f"geno_imputed {cold[2].shape} dtype {cold[2].dtype}")

    t0 = time.perf_counter()
    hit = _call(df_miss, "ldknni")           # SAME key
    t_hit = time.perf_counter() - t0
    print(f"  cache HIT (same key)      : {t_hit:8.4f} s   "
          f"speedup {t_cold - t_hit:8.2f} s faster")

    # a genuinely different sample set -> different sample_set_hash -> re-impute
    df_dropped = df_miss.iloc[:-8]
    assert _sshash(df_dropped) != _sshash(df_miss)
    t0 = time.perf_counter()
    reimp = _call(df_dropped, "ldknni")
    t_reimp = time.perf_counter() - t0
    print(f"  different sample set      : {t_reimp:8.2f} s   "
          f"geno_imputed {reimp[2].shape}  (n dropped by 8)")

    # correctness: cache output == direct build
    direct = _pipeline_build_geno_matrices(df_miss, method="ldknni",
                                           impute_k=5, impute_l=20)
    print(f"  cached == direct build    : {_eq_tuple(hit, direct)}")
    print(f"  ACCEPTANCE  hit >100s faster: {(t_cold - t_hit) > 100.0}  "
          f"(cold {t_cold:.1f}s, hit {t_hit:.4f}s)")


def p2_disk_roundtrip(tuple_29mb) -> None:
    print("\n=== P2 -- persist=\"disk\" round-trip via LocalDiskCacheStorage ===")
    from streamlit.runtime.caching.storage.local_disk_cache_storage import (
        LocalDiskCacheStorage,
    )
    from streamlit.runtime.caching.storage.cache_storage_protocol import (
        CacheStorageContext,
    )

    ctx = CacheStorageContext(
        function_key="impute_cache_bench",
        function_display_name="impute_cache_bench",
        ttl_seconds=None,
        max_entries=4,
        persist="disk",
    )
    store = LocalDiskCacheStorage(ctx)
    store.clear()

    blob = pickle.dumps(tuple_29mb, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  pickled payload           : {len(blob) / 1e6:8.2f} MB")

    t0 = time.perf_counter()
    store.set("entry_0", blob)
    t_write = time.perf_counter() - t0
    print(f"  disk write                : {t_write:8.3f} s")

    t0 = time.perf_counter()
    got = store.get("entry_0")
    t_read = time.perf_counter() - t0
    t0 = time.perf_counter()
    restored = pickle.loads(got)
    t_deser = time.perf_counter() - t0
    print(f"  disk read                 : {t_read:8.3f} s")
    print(f"  deserialise               : {t_deser:8.3f} s")
    print(f"  byte round-trip identical : {got == blob}")
    print(f"  array round-trip identical: {_eq_tuple(restored, tuple_29mb)}")

    # footprint after 4 distinct entries
    for i in range(1, 4):
        store.set(f"entry_{i}", blob)
    cache_dir = Path.home() / ".streamlit" / "cache"
    files = list(cache_dir.glob("impute_cache_bench-*.memo")) if cache_dir.exists() else []
    total = sum(f.stat().st_size for f in files)
    print(f"  on-disk .memo files       : {len(files)}  total {total / 1e6:8.2f} MB "
          f"({cache_dir})")
    print(f"  ACCEPTANCE  deser << 109s : {t_deser < 109.0}  (deser {t_deser:.3f}s)")
    store.clear()


def p3_raw_nan(df_miss: pd.DataFrame, injected: int) -> None:
    print("\n=== P3 -- raw genotype export preserves NaN ===")
    raw, rate, imputed, iid = _pipeline_build_geno_matrices(
        df_miss, method="mean", impute_k=5, impute_l=20)
    n_raw = int(np.isnan(raw).sum())
    print(f"  injected missing cells    : {injected:,}")
    print(f"  NaN in geno_dosage_raw    : {n_raw:,}")
    print(f"  NaN in geno_imputed       : {int(np.isnan(imputed).sum()):,}")
    print(f"  ACCEPTANCE raw==injected  : {n_raw == injected}")


def main() -> None:
    print(f"Loading {TOMATO} ...")
    df = pd.read_csv(TOMATO, index_col=0)
    print(f"  panel {df.shape} (samples x SNPs), raw NaN {int(df.isna().sum().sum())}")
    df_miss, injected = _inject_missing(df, MISS_RATE, SEED)
    print(f"  injected {injected:,} missing cells at {MISS_RATE:.0%}")

    p1_timing(df_miss)
    p3_raw_nan(df_miss, injected)
    # build a ~29 MB tuple for P2 from a mean build (fast, complete)
    tup = _pipeline_build_geno_matrices(df, method="mean")
    p2_disk_roundtrip(tup)


if __name__ == "__main__":
    main()
