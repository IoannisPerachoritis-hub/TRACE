"""The table on disk shared by several users of one process (the GUI's sessions share a table
through st.cache_resource): reads from several threads at once return what one reader gets, the
per-SNP mean/SD cache is bounded and emptied on close, a closed table refuses reads, and the
folder helpers the CLI and the GUI share make and remove the table's folder."""
import os
import threading

import numpy as np
import pytest

from gwas import genostore as gs
from gwas.genostore import LazyColumns, write_store

N, M = 37, 2500


def _table(seed=1):
    rng = np.random.default_rng(seed)
    G = rng.integers(0, 3, size=(N, M)).astype(np.float32)
    G[rng.random((N, M)) < 0.05] = np.nan
    return G


@pytest.fixture()
def store(tmp_path):
    G = _table()
    st = write_store(tmp_path / "s", G)
    yield st, G
    st.close()


def test_reads_from_many_threads_equal_one_reader(store):
    st, G = store
    rng = np.random.default_rng(7)
    picks = [np.sort(rng.choice(M, size=int(rng.integers(1, 400)), replace=False)) for _ in range(64)]
    want = [st.gather(c) for c in picks]
    got = [None] * len(picks)

    def worker(i):
        for _ in range(5):                       # interleave as much as possible
            got[i] = st.gather(picks[i])

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(picks))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for w, g in zip(want, got):
        assert g.tobytes() == w.tobytes()


def test_standardized_cache_is_bounded_and_cleared_on_close(store):
    st, G = store
    tables = [LazyColumns(st, np.arange(k, M, 3)) for k in range(st.STD_CACHE_MAX + 3)]
    first = tables[0].standardized().block(np.arange(5))
    for t in tables[1:]:
        t.standardized()
        assert len(st._std_cache) <= st.STD_CACHE_MAX
    assert len(st._std_cache) == st.STD_CACHE_MAX
    # an evicted set is computed again, to the same values
    again = tables[0].standardized().block(np.arange(5))
    assert again.tobytes() == first.tobytes()
    st.close()
    assert st._std_cache == {}


def test_closed_table_refuses_reads(store):
    st, _ = store
    st.close()
    with pytest.raises(ValueError, match="closed"):
        st.gather([0, 1])
    st.close()                                   # closing twice is harmless


def test_new_store_dir_and_remove_store(tmp_path):
    base = tmp_path / "work" / "deeper"          # made when missing
    a = gs.new_store_dir(base)
    b = gs.new_store_dir(base, prefix="_trace_geno_store_gui")
    assert a.parent == base and b.parent == base and a != b
    assert a.name.startswith(f"_trace_geno_store_{os.getpid()}_")
    assert b.name.startswith(f"_trace_geno_store_gui_{os.getpid()}_")
    assert not any(a.iterdir()) and not any(b.iterdir())
    G = _table()
    st = write_store(a, G)

    class Parsed:
        store = st

    gs.remove_store(a, Parsed())
    assert not a.exists()
    with pytest.raises(ValueError, match="closed"):   # closed before the folder went
        st.gather([0])
    gs.remove_store(b)                           # no table: the folder alone
    assert not b.exists()
    gs.remove_store(b)                           # already gone: no error


def test_new_store_dir_raises_where_it_cannot_write(tmp_path):
    blocker = tmp_path / "a_file"
    blocker.write_text("x")
    with pytest.raises(OSError):
        gs.new_store_dir(blocker / "sub")        # a file stands where the folder should go
