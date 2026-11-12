"""Digests of the pipeline's intermediate objects (a diagnostic for tests; off by default).

With TRACE_TAPS=1 in the environment, gwas.pipeline._run_trait records at named stages a digest
of each intermediate object (dtype, shape, layout flags, the first 32 hex digits of a SHA-256 of
the bytes) in <output>/../taps_<trait>.json. Two runs (the table in memory and on disk, or two
piece sizes) are compared stage by stage: equal digests mean the stage handed on bit-identical
objects, including objects that never reach the results ZIP (the kept SNP and sample sets, the
tables, the kinship matrix, the PCs, the LOCO kernels, M_eff, the standardised table). Without
the variable every call returns at once and nothing is written.

A genotype table (ndarray or gwas.genostore.LazyColumns) is digested by content: its values
column by column (F order), read in blocks, so the in-memory and on-disk tables give the same
digest exactly when they hold the same values.
"""
import hashlib
import json
import os

import numpy as np

_STATE = {"path": None, "rows": []}


def on():
    return os.environ.get("TRACE_TAPS", "") == "1"


def start(path):
    if on():
        _STATE["path"] = str(path)
        _STATE["rows"] = []


def _sha(b):
    return hashlib.sha256(b).hexdigest()[:32]


def table(x, block=4096):
    """Content digest of a samples x SNPs table (ndarray or LazyColumns)."""
    from gwas.genostore import is_lazy

    h = hashlib.sha256()
    n, m = x.shape
    for k in range(0, m, block):
        j = min(m, k + block)
        if is_lazy(x):
            b = x.block(np.arange(k, j), layout="F")
        else:
            b = np.asfortranarray(x[:, k:j])
        h.update(b.tobytes(order="F"))
    return {"kind": "table", "dtype": str(np.dtype(x.dtype)), "shape": [int(n), int(m)], "sha": h.hexdigest()[:32]}


def _one(x):
    import pandas as pd

    if x is None:
        return {"kind": "none"}
    if isinstance(x, dict) and x.get("kind") == "table":
        return x
    if isinstance(x, np.ndarray):
        if x.dtype == object or x.dtype.kind in "US":
            return {"kind": "strarray", "shape": list(x.shape),
                    "sha": _sha("\x1f".join(map(str, x.ravel().tolist())).encode("utf-8"))}
        return {"kind": "ndarray", "dtype": str(x.dtype), "shape": list(x.shape),
                "C": bool(x.flags.c_contiguous), "F": bool(x.flags.f_contiguous),
                "sha": _sha(np.ascontiguousarray(x).tobytes())}
    if isinstance(x, pd.Index):
        return {"kind": "index", "n": int(len(x)), "sha": _sha("\x1f".join(map(str, x.tolist())).encode("utf-8"))}
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return {"kind": "frame", "shape": list(x.shape), "sha": _sha(x.to_csv().encode("utf-8"))}
    if isinstance(x, dict):
        return {"kind": "dict", "sha": _sha(json.dumps(x, sort_keys=True, default=repr).encode("utf-8"))}
    if isinstance(x, (list, tuple)):
        return {"kind": "seq", "n": len(x), "sha": _sha(repr(x).encode("utf-8"))}
    return {"kind": type(x).__name__, "repr": repr(x)[:200]}


def tap(stage, **objects):
    """Record the digests of ``objects`` under ``stage`` (no-op unless TRACE_TAPS=1)."""
    if not on() or _STATE["path"] is None:
        return
    for name, obj in objects.items():
        _STATE["rows"].append({"stage": stage, "name": name, **_one(obj)})
    with open(_STATE["path"], "w", encoding="utf-8") as fh:
        json.dump(_STATE["rows"], fh, indent=0)
