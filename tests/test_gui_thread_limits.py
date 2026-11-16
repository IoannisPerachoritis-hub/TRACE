"""The BLAS thread limits the subsampling sets (gwas.provenance.limited_threads).

threadpoolctl's limits hold for the whole process, and the GUI's sessions are threads of one server
process. Two overlapping sections (A sets, B sets, A restores, B restores) would leave the process at a
worker's thread count for good. Every section holds one process-wide lock, so a second section waits
for the first, and the process keeps its own count afterwards.
"""
import ast
import threading
from pathlib import Path

import numpy as np  # noqa: F401  (numpy's BLAS loaded)
from threadpoolctl import threadpool_info

from gwas import provenance

REPO = Path(__file__).resolve().parents[1]


def _counts():
    return {lib["filepath"]: lib["num_threads"] for lib in threadpool_info()
            if lib.get("internal_api") in ("openblas", "mkl", "openmp", "blis")}


def _other_limits():
    """Limits that differ from the process's current counts, keyed by prefix (as
    worker_threadpool_limits gives them)."""
    out = {}
    for lib in threadpool_info():
        if lib.get("internal_api") in ("openblas", "mkl", "openmp", "blis"):
            out[lib["prefix"]] = 1 if lib["num_threads"] > 1 else 2
    return out


def test_two_sessions_limits_never_overlap_and_the_count_is_restored():
    start = _counts()
    limits = _other_limits()
    a_in, b_in, a_out = threading.Event(), threading.Event(), threading.Event()
    seen = {}

    def session_a():
        with provenance.limited_threads(limits):
            a_in.set()
            # B tries to enter now; with the lock it cannot until A leaves
            seen["b_entered_while_a_inside"] = b_in.wait(0.5)
        a_out.set()

    def session_b():
        a_in.wait(10)
        with provenance.limited_threads(limits):
            b_in.set()
            seen["a_left_first"] = a_out.is_set()

    ta, tb = threading.Thread(target=session_a), threading.Thread(target=session_b)
    ta.start()
    tb.start()
    ta.join(30)
    tb.join(30)
    assert seen == {"b_entered_while_a_inside": False, "a_left_first": True}
    assert _counts() == start


def test_none_sets_no_limit_and_takes_no_lock():
    start = _counts()
    provenance.THREAD_LIMITS_LOCK.acquire()
    try:
        done = []

        def other_session():
            with provenance.limited_threads(None):
                done.append(True)

        t = threading.Thread(target=other_session)
        t.start()
        t.join(5)
        assert done == [True]                    # did not wait for the lock held here
    finally:
        provenance.THREAD_LIMITS_LOCK.release()
    assert _counts() == start


def test_every_thread_limit_in_the_code_goes_through_the_lock():
    """No module but gwas.provenance calls threadpoolctl's threadpool_limits: the GUI's subsampling
    fallback (gwas.guigwas.subsampling) and the replicates on disk (gwas.subsampling) use
    limited_threads."""
    callers = []
    for f in [*REPO.glob("gwas/*.py"), *REPO.glob("pages/**/*.py"), REPO / "cli.py", REPO / "app.py",
              *REPO.glob("utils/*.py"), REPO / "annotation.py"]:
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
                if name == "threadpool_limits":
                    callers.append(f"{f.relative_to(REPO)}:{node.lineno}")
    assert callers and all(c.startswith(str(Path("gwas") / "provenance.py")) for c in callers), callers
    for f in ("gwas/guigwas.py", "gwas/subsampling.py"):
        assert "limited_threads(" in (REPO / f).read_text(encoding="utf-8"), f
