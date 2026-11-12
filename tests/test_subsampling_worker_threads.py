"""With the table on disk, subsampling runs its replicates one after another, each under the
thread counts a joblib worker of the parallel run starts with: the parent's OPENBLAS_NUM_THREADS
(and the like) when the environment sets one, max(cores // workers, 1) otherwise. The disk path
used to ignore the environment, so with the threads pinned there (as on most clusters) and
--boot-jobs above 1 its subsampling tables differed in their last digits from the memory run
(tomato, 4 workers, 12 threads pinned: 3 of 27 members). The provenance row of the threads per
worker uses the same rule."""
from types import SimpleNamespace

import pytest
from joblib import Parallel, cpu_count, delayed

from gwas.provenance import (WORKER_THREAD_VARS, _processors, subsampling_threads,
                             thread_provenance, worker_thread_limits)

_VARS = list(WORKER_THREAD_VARS.values()) + ["GOTO_NUM_THREADS"]


@pytest.fixture
def env(monkeypatch):
    for v in _VARS:
        monkeypatch.delenv(v, raising=False)
    return monkeypatch


def _worker_threads():
    import numpy  # noqa: F401
    import scipy.linalg  # noqa: F401
    import sklearn.decomposition  # noqa: F401  -- loads the OpenMP runtime, where there is one
    from threadpoolctl import threadpool_info

    return [(lib["internal_api"], lib["num_threads"]) for lib in threadpool_info()]


@pytest.mark.parametrize("pinned", [{}, {"OPENBLAS_NUM_THREADS": "1"}, {"OMP_NUM_THREADS": "1"},
                                    {"OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OMP_NUM_THREADS": "2"}])
def test_limits_are_those_real_workers_start_with(env, pinned):
    for k, v in pinned.items():
        env.setenv(k, v)
    want = worker_thread_limits(2)
    seen = Parallel(n_jobs=2, prefer="processes")(delayed(_worker_threads)() for _ in range(4))
    checked = 0
    for libs in seen:
        for api, n in libs:
            if api in want:
                assert n == want[api], (pinned, libs, want)
                checked += 1
    assert checked >= 4                       # numpy's BLAS in each of the 4 calls at least


def test_rule_matches_joblibs_worker_environment(env):
    """joblib's own rule (a private method, so this test notices if joblib changes it)."""
    lb = pytest.importorskip("joblib._parallel_backends")
    for pinned in ({}, {"OPENBLAS_NUM_THREADS": "3"}, {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "4"}):
        for v in _VARS:
            env.delenv(v, raising=False)
        for k, v in pinned.items():
            env.setenv(k, v)
        for jobs in (2, 3, 7):
            worker_env = lb.LokyBackend()._prepare_worker_env(n_jobs=jobs)
            limits = worker_thread_limits(jobs)
            for api, var in WORKER_THREAD_VARS.items():
                assert limits[api] == min(int(worker_env[var]), _processors()), (pinned, jobs, api)


def test_rule_details(env):
    assert worker_thread_limits(1) is None and subsampling_threads(1) is None
    assert worker_thread_limits(None) is None
    assert worker_thread_limits(4)["openblas"] == max(cpu_count() // 4, 1)
    assert worker_thread_limits(-1)["openblas"] == 1            # all cores: one thread each
    env.setenv("OPENBLAS_NUM_THREADS", "100000")
    assert worker_thread_limits(2)["openblas"] == _processors()  # a library uses at most these
    for bad in ("0", "-2", "four", ""):
        env.setenv("OPENBLAS_NUM_THREADS", bad)
        assert worker_thread_limits(2)["openblas"] == max(cpu_count() // 2, 1)
    env.setenv("OPENBLAS_NUM_THREADS", "1")
    env.setenv("MKL_NUM_THREADS", "1")
    assert subsampling_threads(4) == 1
    p = thread_provenance(SimpleNamespace(subsampling=True, boot_jobs=4))
    assert p["Subsampling BLAS threads per worker"] == 1


@pytest.mark.parametrize("pinned", [{}, {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"}])
def test_disk_replicates_run_with_the_threads_of_a_worker(tmp_path, env, pinned):
    """The thread counts in effect inside each replicate on disk equal those inside a real
    worker of a 4-worker pool, library by library (the BLAS ones decide the last bits)."""
    import gwas.subsampling as gs
    from tests.test_store_models import _model_inputs

    for k, v in pinned.items():
        env.setenv(k, v)
    real_rep = gs._subsample_one_rep
    seen = []

    def spy(*a, **k):
        seen.append(sorted(_worker_threads()))
        return real_rep(*a, **k)

    env.setattr(gs, "_subsample_one_rep", spy)
    mem, sto, st, parsed, im, is_ = _model_inputs(tmp_path, 0)
    try:
        K, Z, chroms_grm, _p, _m = st.build_kinship(sto["S08"]["geno_df"], sto["S09"]["imp"],
                                                    sto["S08"]["chroms"], sto["S08"]["positions"])
        gs.subsample_gwas_resampling(
            geno_imputed=sto["S09"]["imp"], y=is_["y"], sid=is_["sid"], chroms=is_["chroms"],
            chroms_num=is_["chroms_num"], positions=is_["positions"], iid=is_["iid"], Z_for_grm=Z,
            n_pcs=0, n_reps=2, sample_frac=0.8, discovery_thresh=1e-4, seed=42, n_jobs=4,
            use_loco=True, chroms_grm=chroms_grm)
    finally:
        parsed.store.close()
    workers = Parallel(n_jobs=4, prefer="processes")(delayed(_worker_threads)() for _ in range(4))
    want = worker_thread_limits(4)
    assert len(seen) == 2
    for rep in seen:
        assert {n for api, n in rep if api == "openblas"} == {want["openblas"]}, (rep, want)
        for api, n in rep:
            assert n == want[api], (rep, want)
        for w in workers:
            assert {n for api, n in w if api == "openblas"} == {n for api, n in rep if api == "openblas"}, (w, rep)
