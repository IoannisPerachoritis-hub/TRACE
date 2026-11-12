"""The machine facts that decide the last bits of a run's numbers, for its provenance.

The linear algebra library (OpenBLAS or MKL) splits matrix products over its threads, and the
order of the partial sums changes the last bits of the results: between 1 and 12 threads the
p-values of the same run differ by up to about 5 parts in 10 million (scaling audit, section
8.4). With --boot-jobs above 1, each subsampling worker gets max(cores / workers, 1) threads,
so subsampling also depends on the number of cores. A run's numbers are byte-identical to
another's at the same library, thread count and (for parallel subsampling) core count; these
are recorded in the report's provenance table and in run_metadata.json.
"""
import os


def blas_info():
    """(library and version, threads in effect) of the BLAS numpy uses, from threadpoolctl;
    ("unknown", None) when it cannot be read."""
    try:
        import numpy  # noqa: F401  -- loads numpy's BLAS, which threadpoolctl lists
        from threadpoolctl import threadpool_info

        for lib in threadpool_info():
            if lib.get("user_api") == "blas":
                name = f"{lib.get('internal_api', '?')} {lib.get('version') or ''}".strip()
                return name, lib.get("num_threads")
    except Exception:
        pass
    return "unknown", None


def subsampling_threads(boot_jobs):
    """The BLAS threads each subsampling worker gets: joblib's max(cores // workers, 1) for more
    than one worker (the table on disk runs the replicates one after another with this same
    limit); None for one worker (the run's own thread count applies)."""
    if boot_jobs is None or int(boot_jobs) == 1:
        return None
    from joblib import cpu_count, effective_n_jobs

    return max(cpu_count() // effective_n_jobs(int(boot_jobs)), 1)


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
