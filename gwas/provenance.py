"""The machine facts that decide the last bits of a run's numbers, for its provenance.

The linear algebra libraries (OpenBLAS or MKL; numpy and scipy each bring their own) split
matrix products over their threads, and the order of the partial sums changes the last bits of
the results: between 1 and 12 threads the
p-values of the same run differ by up to about 5 parts in 10 million.
With --boot-jobs above 1, each subsampling worker gets the thread count set in the
environment (OPENBLAS_NUM_THREADS and the like) or, when none is set, max(cores / workers, 1),
so subsampling can also depend on the number of cores. A run's numbers are byte-identical to
another's at the same library, thread count and (for parallel subsampling) worker thread count;
these are recorded in the report's provenance table and in run_metadata.json.
"""
import os
import threading
from contextlib import contextmanager

# threadpoolctl's limits hold for the whole process, and the GUI's sessions are threads of one
# server process. Two sections that set limits must not overlap: each restores, when it ends,
# the counts it found when it began, so overlapping sections would leave the process at a
# worker's thread count for good. Every section holds this lock (limited_threads).
THREAD_LIMITS_LOCK = threading.RLock()


@contextmanager
def limited_threads(limits):
    """``threadpoolctl.threadpool_limits(limits=limits)`` (a worker_threadpool_limits dict) under
    THREAD_LIMITS_LOCK, so no other section changes the limits meanwhile. ``None`` sets nothing
    and takes no lock. While a section runs, other threads of the process use its counts too."""
    if limits is None:
        yield
        return
    from threadpoolctl import threadpool_limits

    with THREAD_LIMITS_LOCK, threadpool_limits(limits=limits):
        yield


# joblib starts each process worker with these variables set to the parent's value when the
# parent's environment has it, and to max(cpu_count // workers, 1) otherwise
# (joblib/_parallel_backends.py, _prepare_worker_env). Each library reads its own variable at
# start and uses at most the processors it sees.
WORKER_THREAD_VARS = {"openblas": "OPENBLAS_NUM_THREADS", "mkl": "MKL_NUM_THREADS",
                      "blis": "BLIS_NUM_THREADS", "openmp": "OMP_NUM_THREADS"}


def _owner(lib):
    """The package a BLAS library file belongs to ("numpy", "scipy"), from the folder it sits in
    (numpy.libs/ or numpy/.libs/ in the wheels); "" when it is neither."""
    from pathlib import Path

    parents = Path(lib.get("filepath") or "").parents
    for p in list(parents)[:2]:
        name = p.name.lower()
        for pkg in ("numpy", "scipy"):
            if name == pkg or name.startswith(pkg + "."):
                return pkg
    return ""


def _blas_libs():
    """threadpoolctl's BLAS entries in a fixed order: numpy's first, then scipy's, then the rest
    by file path. numpy and scipy each bring their own OpenBLAS, and threadpoolctl lists them in
    an order that changes from one process to the next (seen on Linux)."""
    try:
        import numpy  # noqa: F401  -- loads numpy's BLAS, which threadpoolctl lists
        from threadpoolctl import threadpool_info

        libs = [lib for lib in threadpool_info() if lib.get("user_api") == "blas"]
    except Exception:
        return []
    rank = {"numpy": 0, "scipy": 1}
    return sorted(libs, key=lambda lib: (rank.get(_owner(lib), 2), str(lib.get("filepath") or "")))


def _numpy_blas():
    """threadpoolctl's entry for the BLAS numpy uses (the first of _blas_libs), or None."""
    libs = _blas_libs()
    return libs[0] if libs else None


def blas_info():
    """(the BLAS libraries with their versions, numpy's first, for example "openblas 0.3.30
    (numpy); openblas 0.3.31 (scipy)"; the threads numpy's uses), from threadpoolctl;
    ("unknown", None) when they cannot be read. Both libraries take part in the analysis
    (FaST-LMM works through scipy), so both are named."""
    libs = _blas_libs()
    if not libs:
        return "unknown", None
    names = []
    for lib in libs:
        name = f"{lib.get('internal_api', '?')} {lib.get('version') or ''}".strip()
        owner = _owner(lib)
        names.append(f"{name} ({owner})" if owner else name)
    return "; ".join(dict.fromkeys(names)), libs[0].get("num_threads")


def _processors():
    """The processors this process may run on, which caps a library's thread count at start."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def worker_thread_limits(boot_jobs):
    """{library: threads} that each joblib worker of a ``boot_jobs`` pool starts with, keyed by
    threadpoolctl's internal_api ("openblas", "mkl", "blis", "openmp"); None when the replicates
    run in this process (one worker, or joblib cannot start more). A variable that is not a whole
    number above zero is taken as unset (joblib's default)."""
    if boot_jobs is None or int(boot_jobs) == 1:
        return None
    from joblib import cpu_count, effective_n_jobs

    jobs = effective_n_jobs(int(boot_jobs))
    if jobs == 1:
        return None
    default = max(cpu_count() // jobs, 1)
    cap = _processors()
    limits = {}
    for api, var in WORKER_THREAD_VARS.items():
        try:
            n = int(os.environ.get(var, default))
        except ValueError:
            n = default
        limits[api] = min(n if n > 0 else default, cap)
    return limits


def worker_threadpool_limits(boot_jobs):
    """worker_thread_limits for the libraries loaded now, keyed by their file prefix as
    threadpoolctl's ``threadpool_limits`` takes them (it matches a prefix or a user_api, not the
    library's name); None as there. With the table on disk the subsampling replicates run one
    after another under these limits, so they give the numbers of the parallel run."""
    limits = worker_thread_limits(boot_jobs)
    if limits is None:
        return None
    from threadpoolctl import threadpool_info

    return {lib["prefix"]: limits[lib["internal_api"]] for lib in threadpool_info()
            if lib.get("internal_api") in limits}


def subsampling_threads(boot_jobs):
    """The BLAS threads each subsampling worker gets (worker_thread_limits for numpy's BLAS);
    None for one worker (the run's own thread count applies)."""
    limits = worker_thread_limits(boot_jobs)
    if limits is None:
        return None
    lib = _numpy_blas()
    return limits.get((lib or {}).get("internal_api"), limits["openblas"])


def thread_provenance(args):
    """The provenance rows: BLAS library, its threads, the cores, --boot-jobs."""
    lib, threads = blas_info()
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
    except Exception:
        physical = None
    sub = bool(getattr(args, "subsampling", False))
    jobs = getattr(args, "boot_jobs", 1)
    per_worker = subsampling_threads(jobs) if sub else None
    return {
        "BLAS library": lib,
        "BLAS threads": threads if threads is not None else "unknown",
        "CPU cores (logical)": os.cpu_count() or "unknown",
        "CPU cores (physical)": physical if physical is not None else "unknown",
        "Subsampling workers (--boot-jobs)": jobs if sub else "N/A",
        "Subsampling BLAS threads per worker": (per_worker if per_worker is not None
                                                else ("as the run" if sub else "N/A")),
    }
