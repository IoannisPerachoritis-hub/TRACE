"""The on-disk genotype table (gwas/genostore.py): every read returns today's values,
number format and layout.

The in-memory reference is the array today's pipeline holds at the same point: the raw table
(float32, NaN for a missing call) or the mean-imputed table (float32, the per-SNP fill in
place of NaN), F- or C-ordered. Each indexing form the pipeline uses is applied to both, and
the results must agree bit for bit, with the same dtype and the same contiguity: BLAS takes
another code path for another layout, and the last bits change (scaling audit, section 8.3).
"""
import warnings

import numpy as np
import pytest

from gwas import genostore as gs
from gwas.genostore import LazyColumns, as_columns, encode_calls, is_lazy, write_store

N, M = 40, 300


def _table(seed=0, n=N, m=M, miss=0.08):
    rng = np.random.default_rng(seed)
    G = rng.integers(0, 3, size=(n, m)).astype(np.float32)
    G[rng.random((n, m)) < miss] = np.nan
    G[:, 4] = np.nan                       # a column with no call
    G[:, 6] = 1.0                          # a monomorphic column
    return G


def _fill(G):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)         # the column with no call
        f = np.nanmean(G.astype(np.float64), axis=0).astype(np.float32)
    return np.where(np.isfinite(f), f, np.float32(0.0)).astype(np.float32)


def _imputed(G, fill):
    return np.where(np.isnan(G), fill[None, :], G).astype(np.float32)


@pytest.fixture()
def store(tmp_path):
    G = _table()
    st = write_store(tmp_path / "s", G)
    yield st, G
    st.close()


def _same(a, b):
    """Bit-identical values, dtype, shape and contiguity."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype
    assert a.shape == b.shape
    assert np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()
    assert a.flags["C_CONTIGUOUS"] == b.flags["C_CONTIGUOUS"]
    assert a.flags["F_CONTIGUOUS"] == b.flags["F_CONTIGUOUS"]
    if a.ndim == 2 and not (a.flags["C_CONTIGUOUS"] or a.flags["F_CONTIGUOUS"]):
        # a strided view: the same axis must be the contiguous one
        assert (a.strides[1] == a.itemsize) == (b.strides[1] == b.itemsize)
        assert (a.strides[0] == a.itemsize) == (b.strides[0] == b.itemsize)


# ------------------------------------------------------------------------------ the store
def test_round_trip_and_gather(store):
    st, G = store
    g8 = encode_calls(G)
    assert np.array_equal(st.gather(np.arange(M)), g8)
    cols = np.array([299, 5, 6, 7, 8, 120, 3, 3])          # unsorted, repeated, runs
    rows = np.array([7, 0, 39, 12])
    assert np.array_equal(st.gather(cols, rows), g8[np.ix_(rows, cols)])
    assert st.gather(cols, rows).flags["F_CONTIGUOUS"]
    for p in (1, 2, 4096):
        assert np.array_equal(st.gather(cols, rows, piece=p), g8[np.ix_(rows, cols)])


@pytest.mark.parametrize("p", [1, 7, 8192])
def test_count_pass_is_exact(store, p, monkeypatch):
    st, G = store
    monkeypatch.setenv("TRACE_STORE_PIECE", str(p))
    rows = np.array([3, 1, 4, 15, 9, 26, 5])
    cols = np.array([10, 2, 299, 4, 6, 150])
    c = st.count_pass(cols, rows)
    sub = G[np.ix_(rows, cols)]
    assert np.array_equal(c["n1"], (sub == 1).sum(axis=0))
    assert np.array_equal(c["n2"], (sub == 2).sum(axis=0))
    assert np.array_equal(c["nmiss"], np.isnan(sub).sum(axis=0))
    assert np.array_equal(c["row_nmiss"], np.isnan(sub).sum(axis=1))
    assert c["gmax"] == int(np.nanmax(sub)) and c["n"] == rows.size


def test_short_read_raises(tmp_path):
    G = _table(m=10)
    st = write_store(tmp_path / "s", G)
    st.close()
    bad = gs.GenotypeStore(tmp_path / "s", N, 11)            # one variant more than the file holds
    with pytest.raises(IOError):
        bad.gather([10])
    bad.close()


# -------------------------------------------------------------------- values and layouts
@pytest.mark.parametrize("imputed", [False, True])
@pytest.mark.parametrize("layout", ["F", "C"])
def test_whole_table_values(store, imputed, layout):
    st, G = store
    fill = _fill(G) if imputed else None
    ref = _imputed(G, fill) if imputed else G
    ref = np.asfortranarray(ref) if layout == "F" else np.ascontiguousarray(ref)
    L = LazyColumns(st, np.arange(M), fill=fill, layout=layout)
    _same(np.asarray(L), ref)
    assert L.shape == ref.shape and len(L) == N and L.size == ref.size and L.ndim == 2


RIDX = np.array([3, 0, 17, 9])
RMASK = np.zeros(N, dtype=bool)
RMASK[[2, 5, 11, 30]] = True
IDX = np.array([5, 1, 77, 200, 13])
MASK = np.zeros(M, dtype=bool)
MASK[[0, 4, 6, 90, 299]] = True

# The indexing forms of the scaling audit (work/proto/layout_rules.txt) that the pipeline
# applies to a genotype table. A form that yields a LazyColumns is compared through
# np.asarray, the array the code would hold.
FORMS = {
    "A[:, 10:50]": lambda A: A[:, 10:50],
    "A[:, idx]": lambda A: A[:, IDX],
    "A[:, mask]": lambda A: A[:, MASK],
    "A[:, 7]": lambda A: A[:, 7],
    "A[:, [7]]": lambda A: A[:, [7]],
    "A[ridx]": lambda A: A[RIDX],
    "A[rmask]": lambda A: A[RMASK],
    "A[ridx, :]": lambda A: A[RIDX, :],
    "A[ridx][:, 10:50]": lambda A: A[RIDX][:, 10:50],
    "A[ridx][:, idx]": lambda A: A[RIDX][:, IDX],
    "A[:, idx][ridx]": lambda A: A[:, IDX][RIDX],
    "A.astype(float)": lambda A: A.astype(float),
    "np.asarray(A, float)": lambda A: np.asarray(A, float),
    "np.asarray(A, np.float32, order='C')": lambda A: np.asarray(A, np.float32, order="C"),
    "np.asarray(A[:, 10:50], np.float64)": lambda A: np.asarray(A[:, 10:50], np.float64),
    "np.asarray(A[:, idx], np.float64)": lambda A: np.asarray(A[:, IDX], np.float64),
    "A[:, 10:50].astype(float)": lambda A: A[:, 10:50].astype(float),
    "np.asfortranarray(A, dtype=float)": lambda A: np.asfortranarray(A, dtype=float),
    "A[:, 10:50] - 1.0": lambda A: A[:, 10:50] - 1.0,
    "A[:, idx] - 1.0": lambda A: A[:, IDX] - 1.0,
    "np.rint(A)": lambda A: np.rint(A),
    "A[:, mask][:, [0, 2]]": lambda A: A[:, MASK][:, [0, 2]],
    "np.array(A[:, 10:50])": lambda A: np.array(A[:, 10:50]),
    "np.ascontiguousarray(A[:, idx])": lambda A: np.ascontiguousarray(A[:, IDX]),
    "A[ridx].astype(float)": lambda A: A[RIDX].astype(float),
    "A[rmask, :][:, mask]": lambda A: A[RMASK, :][:, MASK],
    "as_columns(A, float)": lambda A: as_columns(A, float),
    "as_columns(A, np.float32)[:, 3:5]": lambda A: as_columns(A, np.float32)[:, 3:5],
}


@pytest.mark.parametrize("form", list(FORMS))
@pytest.mark.parametrize("imputed", [False, True])
@pytest.mark.parametrize("layout", ["F", "C"])
def test_indexing_forms_match_numpy(store, form, imputed, layout):
    st, G = store
    fill = _fill(G) if imputed else None
    ref = _imputed(G, fill) if imputed else G
    A = np.asfortranarray(ref) if layout == "F" else np.ascontiguousarray(ref)
    L = LazyColumns(st, np.arange(M), fill=fill, layout=layout)
    got = FORMS[form](L)
    if is_lazy(got):
        got = np.asarray(got)
    _same(got, FORMS[form](A))


@pytest.mark.parametrize("layout", ["F", "C"])
@pytest.mark.parametrize("width", [1, 2, 3])
def test_narrow_column_slices_keep_numpy_layout(store, layout, width):
    """One- to three-column operands: the case where BLAS changed path (FarmCPU's last chunk)."""
    st, G = store
    fill = _fill(G)
    ref = _imputed(G, fill)
    A = np.asfortranarray(ref) if layout == "F" else np.ascontiguousarray(ref)
    L = LazyColumns(st, np.arange(M), fill=fill, layout=layout)
    for a in (0, 137, M - width):
        _same(L[:, a:a + width], A[:, a:a + width])
        _same(L[:, list(range(a, a + width))], A[:, list(range(a, a + width))])


def test_unsupported_forms_raise(store):
    st, G = store
    L = LazyColumns(st, np.arange(M))
    with pytest.raises(NotImplementedError):
        L[np.ix_(RIDX, IDX)]
    with pytest.raises(NotImplementedError):
        L[3]
    with pytest.raises(NotImplementedError):
        L[RIDX, IDX]
    with pytest.raises(IndexError):
        L[:, np.zeros(M - 1, dtype=bool)]


def test_lazy_subsets_compose(store):
    st, G = store
    fill = _fill(G)
    ref = np.asfortranarray(_imputed(G, fill))
    L = LazyColumns(st, np.arange(M), fill=fill)
    sub = L.lazy_cols(IDX).lazy_rows(RIDX)
    _same(np.asarray(sub), np.ascontiguousarray(ref[:, IDX][RIDX]))
    sub2 = L.lazy_rows(slice(5, 30)).lazy_cols(MASK)
    _same(np.asarray(sub2), ref[5:30][:, MASK])
    assert is_lazy(sub) and is_lazy(L.astype(np.float64)) and not is_lazy(ref)


def test_as_columns_on_arrays_is_np_asarray():
    a = np.asfortranarray(_table(m=20))
    assert as_columns(a) is np.asarray(a)
    _same(as_columns(a, float), np.asarray(a, float))
    _same(as_columns(a, np.float32), np.asarray(a, np.float32))
    lst = [[0.0, 1.0], [2.0, np.nan]]
    _same(as_columns(lst, float), np.asarray(lst, float))


def test_as_columns_keeps_a_lazy_table_on_disk(store):
    st, G = store
    L = LazyColumns(st, np.arange(M))
    L64 = as_columns(L, np.float64)
    assert is_lazy(L64) and L64.dtype == np.float64 and as_columns(L) is L
    _same(np.asarray(L64), np.asarray(np.asfortranarray(G), np.float64))


# --------------------------------------------------------------------------- FaST-LMM reader
def _snpdata(ref):
    from pysnptools.snpreader import SnpData

    iid = np.array([[f"s{i}", f"s{i}"] for i in range(ref.shape[0])], dtype=str)
    sid = np.array([f"rs{j}" for j in range(ref.shape[1])], dtype=str)
    pos = np.c_[np.ones(ref.shape[1]), np.arange(ref.shape[1]) * 100.0, np.arange(ref.shape[1]) * 100.0]
    return SnpData(iid=iid, sid=sid, val=ref, pos=pos), iid, sid, pos


@pytest.mark.parametrize("order", ["F", "C", "A"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_reader_blocks_equal_snpdata(store, order, dtype):
    st, G = store
    fill = _fill(G)
    ref = np.asfortranarray(_imputed(G, fill))
    sd, iid, sid, pos = _snpdata(ref)
    rd = LazyColumns(st, np.arange(M), fill=fill).snp_reader(iid, sid, pos)
    assert rd.iid_count == N and rd.sid_count == M
    assert np.array_equal(rd.sid, sd.sid) and np.array_equal(rd.pos, sd.pos)
    for sel in (np.s_[:, :], np.s_[:, 10:50], np.s_[:, IDX], np.s_[RIDX, :], np.s_[RIDX, 280:]):
        a = rd[sel].read(order=order, dtype=dtype).val
        b = sd[sel].read(order=order, dtype=dtype).val
        _same(a, b)


# ------------------------------------------------------------------------------ the read log
def test_read_log_and_forbid(store, monkeypatch):
    st, G = store
    L = LazyColumns(st, np.arange(M))
    gs.reset_read_log()
    np.asarray(L)
    log = gs.read_log()
    assert len(log["materialized"]) == 1
    assert log["materialized"][0]["shape"] == [N, M]
    assert "test_genostore.py" in log["materialized"][0]["site"]
    assert log["file_bytes"] >= N * M

    monkeypatch.setenv("TRACE_STORE_FORBID_MATERIALIZE", "1")
    with pytest.raises(MemoryError):
        np.asarray(L)
    L[:, 3:9]                                               # pieces are still allowed
    monkeypatch.delenv("TRACE_STORE_FORBID_MATERIALIZE")

    monkeypatch.setenv("TRACE_STORE_MAX_BLOCK_CELLS", str(N * 10))
    L[:, :10]
    with pytest.raises(MemoryError):
        L[:, :11]
    np.asarray(L)                                           # a whole read by design is logged, not limited


@pytest.mark.parametrize("width", [1, 2, 7])
@pytest.mark.parametrize("layout", ["F", "C"])
def test_reader_order_any_follows_the_table_layout(store, layout, width):
    """read(order='A'): SnpData answers in the order of the array it holds (F if that array is
    F-contiguous, else C); the reader over a table on disk answers as SnpData over today's array
    of the same layout."""
    st, G = store
    fill = _fill(G)
    ref = _imputed(G, fill)[:, :width]
    ref = np.asfortranarray(ref) if layout == "F" else np.ascontiguousarray(ref)
    sd, iid, sid, pos = _snpdata(ref)
    rd = LazyColumns(st, np.arange(width), fill=fill, layout=layout).snp_reader(iid, sid, pos)
    for sel in (np.s_[:, :], np.s_[RIDX, :]):
        for dtype in (np.float32, np.float64):
            _same(rd[sel].read(order="A", dtype=dtype).val, sd[sel].read(order="A", dtype=dtype).val)


def test_standardized_table_blocks_and_subsets(store):
    """The standardised table keeps its (mean, SD) through column subsets."""
    from gwas.models import _standardize_genotypes_impl

    st, G = store
    fill = _fill(G)
    Z = _standardize_genotypes_impl(_imputed(G, fill))
    S = LazyColumns(st, np.arange(M), fill=fill).standardized()
    assert S.layout == "C" and S.std is not None
    _same(S[:, IDX], Z[:, IDX])
    _same(np.asarray(S.lazy_cols(MASK)), np.ascontiguousarray(Z[:, MASK]))


class _WatchedCache(dict):
    """The store's standardisation cache, noting every look-up and change made without the store's
    lock held (the GUI's sessions share one store, so its cache is shared by threads)."""

    def __init__(self, lock):
        super().__init__()
        self.lock, self.unlocked = lock, []

    def _note(self, what):
        if not self.lock.locked():
            self.unlocked.append(what)

    def get(self, *a):
        self._note("get")
        return super().get(*a)

    def pop(self, *a):
        self._note("pop")
        return super().pop(*a)

    def __setitem__(self, k, v):
        self._note("set")
        super().__setitem__(k, v)


def test_standardisation_cache_is_used_under_the_stores_lock(store):
    """Every look-up, eviction and insertion of the shared cache happens under the store's lock, so
    two sessions standardising one table cannot evict the same entry twice (KeyError) or change
    the dict while another iterates it."""
    st, _G = store
    st._std_cache = _WatchedCache(st._lock)
    views = [LazyColumns(st, np.arange(M)) for _ in range(st.STD_CACHE_MAX + 2)]   # misses that evict
    for v in views:
        v.standardized()
    views[-1].standardized()                                                      # a hit
    assert len(st._std_cache) == st.STD_CACHE_MAX
    assert st._std_cache.unlocked == []


def test_standardisation_from_several_threads(store):
    """Threads standardising fresh views of one store, with frequent thread switches: no error, and
    every (mean, SD) is the one-thread value."""
    import sys
    import threading

    st, _G = store
    ref = LazyColumns(st, np.arange(M)).standardized().std
    st._std_cache.clear()
    errors, wrong = [], []

    def work():
        for _ in range(60):
            try:
                mu, sd = LazyColumns(st, np.arange(M)).standardized().std
                if mu.tobytes() != ref[0].tobytes() or sd.tobytes() != ref[1].tobytes():
                    wrong.append(1)
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        ts = [threading.Thread(target=work) for _ in range(6)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    finally:
        sys.setswitchinterval(old)
    assert errors == [] and wrong == []
