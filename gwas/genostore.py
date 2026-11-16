"""The genotype table kept on disk and read in pieces (``--geno-store disk``).

GenotypeStore   one file, ``calls.i8``: one byte per call (the ALT-allele count, -1 for a
                missing call), variant-major (n_variants x n_samples), so the bytes of a run of
                variants are one contiguous read. Reads are explicit (seek + read), never
                memory-mapped: a mapped file counts as resident memory on Windows and in
                containers, which misleads memory guards (scaling audit, section 9.5).
LazyColumns     a samples x SNPs table that stays on disk: a selection of store rows and
                columns, an optional per-SNP fill value (the mean-imputed table) and the layout
                (C or F) today's in-memory array has at the point where it stands in. Every
                read returns an ndarray with today's values, number format and layout, so the
                code that receives it does today's arithmetic. A conversion of the whole table
                (np.asarray) still works, and is logged with its call site.
as_columns      ``np.asarray(x, dtype)`` for an array; the same table, still on disk, for a
                LazyColumns. Code shared by both modes uses it where it used ``np.asarray``.
ingest_vcf      the VCF read in record chunks (scikit-allel's iter_vcf_chunks); each chunk is
                parsed by today's gwas.qc._pipeline_parse_vcf_biallelic and written to the
                store, so the stored calls are today's parsed values. Returns a StoreParsed,
                which stands in for gwas.pipeline.ParsedGenotypes.

Nothing here is used unless the table is kept on disk; the in-memory path is unchanged.

Diagnostics (environment variables, read when used; tests and measurements only):
  TRACE_STORE_FORBID_MATERIALIZE=1  a whole-table conversion raises MemoryError instead of
                                    reading (proves a run never reads the table whole).
  TRACE_STORE_MAX_BLOCK_CELLS=N     a piecewise read of more than N cells raises MemoryError
                                    (one-piece reads by design are logged, not limited).
  TRACE_STORE_PIECE=N               every streaming pass reads N SNP columns at a time instead
                                    of its default (results must not depend on it).
  TRACE_STORE_CHUNK_RECORDS=N       the ingest reads N VCF records per chunk instead of its
                                    default (results must not depend on it).
"""
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- read accounting
_LOG = {"file_reads": 0, "file_bytes": 0, "blocks": 0, "max_block_cells": 0, "max_block_shape": None,
        "max_block_site": None, "materialized": [], "by_site": {}}
_HERE = os.path.normcase(os.path.abspath(__file__))


def _forbid_materialize():
    return os.environ.get("TRACE_STORE_FORBID_MATERIALIZE", "") == "1"


def _max_block_cells():
    return int(os.environ.get("TRACE_STORE_MAX_BLOCK_CELLS", "0") or 0)    # 0 = no limit


def _call_site():
    """file:line:function of the first frame outside this module (who asked for the block)."""
    f = sys._getframe(1)
    while f is not None and os.path.normcase(os.path.abspath(f.f_code.co_filename)) == _HERE:
        f = f.f_back
    if f is None:
        return "?"
    return f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}:{f.f_code.co_name}"


def _note_block(shape, whole=False, table_cols=None):
    """Log one block read (every read of the table passes here once, with the caller's site).
    ``whole``: a whole-table read by design (one-piece rule, export). ``table_cols``: the width of
    the table the block is cut from; a block that covers all of it is counted as covers_all."""
    cells = int(shape[0]) * int(shape[1])
    site = _call_site()
    _LOG["blocks"] += 1
    s = _LOG["by_site"].setdefault(site, {"blocks": 0, "max_cells": 0, "max_cols": 0, "whole": 0,
                                          "covers_all": 0, "max_frac": 0.0})
    s["blocks"] += 1
    s["max_cells"] = max(s["max_cells"], cells)
    s["max_cols"] = max(s["max_cols"], int(shape[1]))
    s["whole"] += int(bool(whole))
    if table_cols:
        frac = int(shape[1]) / float(table_cols)
        s["max_frac"] = max(s["max_frac"], round(frac, 6))
        s["covers_all"] += int(int(shape[1]) >= int(table_cols))
        _LOG["table_cells"] = max(_LOG.get("table_cells", 0), int(shape[0]) * int(table_cols))
    if cells > _LOG["max_block_cells"]:
        _LOG["max_block_cells"] = cells
        _LOG["max_block_shape"] = tuple(int(x) for x in shape)
        _LOG["max_block_site"] = site
    if whole:
        _LOG["materialized"].append({"site": site, "shape": [int(x) for x in shape]})
    limit = _max_block_cells()
    if limit and cells > limit and not whole:
        raise MemoryError(f"genotype block of {shape[0]} x {shape[1]} cells exceeds "
                          f"TRACE_STORE_MAX_BLOCK_CELLS ({limit}) at {site}")


def read_log():
    """A copy of the read log: file reads and bytes, blocks per call site, whole-table reads."""
    return json.loads(json.dumps(_LOG))


def reset_read_log():
    _LOG.clear()
    _LOG.update(file_reads=0, file_bytes=0, blocks=0, max_block_cells=0, max_block_shape=None,
                max_block_site=None, materialized=[], by_site={})


def piece(default):
    """The number of SNP columns a streaming pass reads at a time (TRACE_STORE_PIECE overrides)."""
    v = os.environ.get("TRACE_STORE_PIECE", "")
    return int(v) if v else int(default)


def is_lazy(x):
    return isinstance(x, LazyColumns)


def as_columns(x, dtype=None):
    """``np.asarray(x, dtype=dtype)`` for an array (the expression the pipeline has always used);
    for a LazyColumns the same table, still on disk, with that dtype."""
    if isinstance(x, LazyColumns):
        return x if dtype is None else x.astype(dtype)
    return np.asarray(x) if dtype is None else np.asarray(x, dtype=dtype)


# ------------------------------------------------------------------------------- the store
def encode_calls(G):
    """The store's bytes for a samples x SNPs table of calls (NaN = missing): int8 (n x m),
    -1 for missing. Calls are whole numbers (ALT-allele counts) in the int8 range."""
    G = np.asarray(G)
    return np.where(np.isnan(G), -1, G).astype(np.int8)


def write_store(store_dir, G):
    """A new store in the folder ``store_dir`` holding the table ``G`` (samples x SNPs, NaN =
    missing). Used by tests; the VCF ingest writes the same file chunk by chunk."""
    G = np.asarray(G)
    os.makedirs(store_dir, exist_ok=True)
    with open(os.path.join(store_dir, "calls.i8"), "wb") as out:
        out.write(np.ascontiguousarray(encode_calls(G).T).tobytes())
    return GenotypeStore(store_dir, G.shape[0], G.shape[1])


class GenotypeStore:
    """The genotype table on disk: ``calls.i8`` (n_variants x n_samples int8, -1 missing).

    One file handle, read with seek + read under a lock, so threads that share one store (the
    GUI's sessions share a table through st.cache_resource) cannot interleave their reads."""

    STD_CACHE_MAX = 4      # per-SNP mean/SD sets kept (each holds a few arrays of the SNP count)

    def __init__(self, store_dir, n_samples, n_variants):
        self.dir = Path(store_dir)
        self.N, self.M = int(n_samples), int(n_variants)
        self.path = self.dir / "calls.i8"
        self._lock = threading.Lock()
        self._fh = open(self.path, "rb", buffering=0)
        self._std_cache = {}

    def close(self):
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            self._std_cache.clear()

    @property
    def closed(self):
        """True once the table file is closed (read only): every read then raises ValueError."""
        return self._fh is None

    def _read_run(self, a, b):
        """Variants a..b-1 as an int8 (b - a, N) C array (one file read)."""
        a, b = int(a), int(b)
        with self._lock:
            if self._fh is None:
                raise ValueError(f"the genotype table {self.path} is closed")
            self._fh.seek(a * self.N)
            buf = np.fromfile(self._fh, dtype=np.int8, count=(b - a) * self.N)
            if buf.size != (b - a) * self.N:
                raise IOError(f"short read from {self.path}: variants {a}..{b}")
            _LOG["file_reads"] += 1
            _LOG["file_bytes"] += int(buf.size)
        return buf.reshape(b - a, self.N)

    def gather(self, cols, rows=None, piece=4096, _log=True, table_cols=None):
        """int8 (n x len(cols)), F-ordered: the calls of store variants ``cols`` (any order) for
        the samples ``rows`` (store row indices; None = all, in store order). Ascending runs of
        variants within ``piece`` of each other are read with one file read. The read is logged
        (``_log=False`` for LazyColumns.block, which logs it itself)."""
        cols = np.asarray(cols, dtype=np.int64)
        n = self.N if rows is None else int(len(rows))
        if _log:
            _note_block((n, cols.size), table_cols=table_cols)
        out = np.empty((n, cols.size), dtype=np.int8, order="F")
        k = 0
        while k < cols.size:
            lo, j = int(cols[k]), k
            while j + 1 < cols.size and cols[j + 1] > cols[j] and int(cols[j + 1]) - lo < piece:
                j += 1
            run = self._read_run(lo, int(cols[j]) + 1)                 # (w, N)
            if int(cols[j]) - lo != j - k:                             # not a contiguous run
                run = run[cols[k:j + 1] - lo]
            if rows is not None:
                run = np.take(run, rows, axis=1)                       # (k', n)
            out[:, k:j + 1] = run.T
            k = j + 1
        return out

    def count_pass(self, cols, rows=None, piece=8192):
        """Exact call counts over ``rows`` for the variants ``cols``: per variant n1, n2, nmiss
        (int64); per row the number of missing calls (int64); the largest value (gmax)."""
        cols = np.asarray(cols, dtype=np.int64)
        n = self.N if rows is None else int(len(rows))
        n1 = np.empty(cols.size, dtype=np.int64)
        n2 = np.empty(cols.size, dtype=np.int64)
        nm = np.empty(cols.size, dtype=np.int64)
        row_nm = np.zeros(n, dtype=np.int64)
        gmax = -1
        step = globals()["piece"](piece)
        for k in range(0, cols.size, step):
            g = self.gather(cols[k:k + step], rows, table_cols=cols.size)
            miss = g < 0
            n1[k:k + step] = (g == 1).sum(axis=0, dtype=np.int64)
            n2[k:k + step] = (g == 2).sum(axis=0, dtype=np.int64)
            nm[k:k + step] = miss.sum(axis=0, dtype=np.int64)
            row_nm += miss.sum(axis=1, dtype=np.int64)
            if g.size:
                gmax = max(gmax, int(g.max()))
        return {"n1": n1, "n2": n2, "nmiss": nm, "row_nmiss": row_nm, "gmax": gmax, "n": n}


def new_store_dir(base, prefix="_trace_geno_store"):
    """A new, empty folder for a genotype table inside ``base`` (created if missing), named
    ``<prefix>_<pid>_<random>``. The exclusive create of a uniquely named folder proves that
    ``base`` can be written and keeps runs that share it apart. Raises OSError otherwise (the
    caller words the message: the CLI names --geno-store-dir, the GUI its work folder)."""
    import uuid

    base = Path(base)
    store_dir = base / f"{prefix}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    base.mkdir(parents=True, exist_ok=True)
    store_dir.mkdir(exist_ok=False)
    return store_dir


def remove_store(store_dir, parsed=None):
    """Close the table ``parsed`` holds (if any) and delete ``store_dir`` with everything in it.
    The file is closed first: on Windows an open file cannot be deleted."""
    import shutil

    store = getattr(parsed, "store", None) if parsed is not None else None
    if store is not None:
        store.close()
    shutil.rmtree(store_dir, ignore_errors=True)


# ----------------------------------------------------------------------- the lazy table
class LazyColumns:
    """A samples x SNPs float table that stays on disk (see the module docstring).

    store   the GenotypeStore
    cols    store variant indices, in the pipeline's SNP order
    rows    store row indices (None = every sample, in store order)
    fill    float32 per column: the value of a missing call (the mean-imputed table); None for
            the raw table, whose missing calls are NaN
    dtype   the number format blocks come back in
    layout  "F" or "C": the layout today's array has where this object stands in
    std     None, or the per-column (mean, SD) float32 of the standardised table (MLMM, FarmCPU)
    """

    ndim = 2

    def __init__(self, store, cols, rows=None, fill=None, dtype=np.float32, layout="F"):
        self.store = store
        self.cols = np.asarray(cols, dtype=np.int64)
        self.rows = None if rows is None else np.asarray(rows, dtype=np.int64)
        self.fill = fill
        self.dtype = np.dtype(dtype)
        self.layout = layout
        self.std = None

    def standardized(self, block=2000):
        """gwas.models._standardize_genotypes_impl without the whole table. The per-SNP mean and
        SD come from C-ordered float32 blocks of all rows: the layout and number format that
        function reduces over, so each column's reduction runs in the same order. A block is
        never one column wide unless the table is, because a one-column block is also
        F-contiguous and numpy would reduce it in another order. Then every block read is
        (G - mean) / SD with NaN and infinities set to 0, in float32."""
        base = self._derive(dtype=np.dtype(np.float32), std=None)
        m = base.shape[1]
        # one pass per (rows, columns, fill): MLMM and FarmCPU standardise the same table
        key = (id(self.cols), id(self.fill), id(self.rows))
        hit = self.store._std_cache.get(key)
        if hit is not None and hit[0] is self.cols and hit[1] is self.fill and hit[2] is self.rows:
            mu, sd = hit[3], hit[4]
        else:
            mu = np.empty(m, dtype=np.float32)
            sd = np.empty(m, dtype=np.float32)
            step = piece(block)
            for k in range(0, m, int(step)):
                j = min(m, k + int(step))
                a = k - 1 if (j - k == 1 and k > 0) else k
                G = base.block(np.arange(a, j), layout="C")
                mu[a:j] = np.nanmean(G, axis=0)
                sd[a:j] = np.nanstd(G, axis=0)
            sd[~np.isfinite(sd) | (sd == 0)] = 1.0
            cache = self.store._std_cache
            while len(cache) >= self.store.STD_CACHE_MAX:        # bounded: drop the oldest set
                cache.pop(next(iter(cache)))
            cache[key] = (self.cols, self.fill, self.rows, mu, sd)
        return base._derive(std=(mu, sd), layout="C")

    # -- ndarray-like surface
    @property
    def shape(self):
        return (self.store.N if self.rows is None else int(self.rows.size), int(self.cols.size))

    @property
    def size(self):
        return self.shape[0] * self.shape[1]

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return (f"LazyColumns(shape={self.shape}, dtype={self.dtype}, layout={self.layout}, "
                f"{'imputed' if self.fill is not None else 'raw'})")

    def _derive(self, **kw):
        new = LazyColumns.__new__(LazyColumns)
        new.__dict__.update(self.__dict__)
        new.__dict__.update(kw)
        return new

    def astype(self, dtype, copy=True):
        return self._derive(dtype=np.dtype(dtype))

    def as_layout(self, layout):
        return self._derive(layout=layout)

    def lazy_cols(self, sel):
        """A lazy column subset (``sel``: index array, slice or boolean mask over this object's
        columns)."""
        sel = self._col_index(sel)
        return self._derive(cols=self.cols[sel], fill=None if self.fill is None else self.fill[sel],
                            std=None if self.std is None else (self.std[0][sel], self.std[1][sel]))

    def lazy_rows(self, sel):
        """A lazy row subset; like numpy's ``A[rows]``, its blocks are C-ordered (a slice keeps
        the layout, as a numpy row slice is a view)."""
        n = self.shape[0]
        base = np.arange(self.store.N, dtype=np.int64) if self.rows is None else self.rows
        if isinstance(sel, slice):
            return self._derive(rows=base[sel])
        sel = np.asarray(sel)
        if sel.ndim != 1:
            raise NotImplementedError("LazyColumns: row selections must be one-dimensional")
        if sel.dtype == bool:
            if sel.shape != (n,):
                raise IndexError("boolean row mask of the wrong length")
            sel = np.flatnonzero(sel)
        return self._derive(rows=base[sel], layout="C")

    def _col_index(self, sel):
        m = self.shape[1]
        if isinstance(sel, slice):
            return np.arange(*sel.indices(m), dtype=np.int64)
        sel = np.asarray(sel)
        if sel.dtype == bool:
            if sel.shape != (m,):
                raise IndexError("boolean column mask of the wrong length")
            return np.flatnonzero(sel)
        sel = sel.astype(np.int64, copy=False).ravel()
        return np.where(sel < 0, sel + m, sel)

    def block(self, sel, layout=None, _whole=False, _strided=False):
        """ndarray (n x k): today's values of the local columns ``sel`` in this object's dtype;
        ``layout`` "F" or "C" (default: this object's layout). ``_strided`` (C layout only): the
        block comes back as a view of a buffer one column wider, so its rows are NOT contiguous,
        as a column slice ``G[:, a:b]`` of today's whole C-ordered table is not. BLAS takes
        another path for a contiguous one- to three-column operand, and the last bits change
        (scaling audit, section 8.3, family C)."""
        sel = self._col_index(sel)
        g8 = self.store.gather(self.cols[sel], self.rows, _log=False)
        out = np.empty(g8.shape, dtype=np.float32, order="F")
        out[...] = g8
        miss = g8 < 0
        if self.fill is None:
            out[miss] = np.nan
        elif miss.any():
            out[miss] = np.broadcast_to(self.fill[sel], out.shape)[miss]
        del g8, miss
        if self.std is not None:
            # (G - mean) / SD in float32, NaN and infinities -> 0 (_standardize_genotypes_impl)
            out = (out - self.std[0][sel]) / self.std[1][sel]
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        if self.dtype != np.float32:
            out = out.astype(self.dtype)                    # keeps the F layout
        if (layout or self.layout) == "C":
            if _strided:
                buf = np.empty((out.shape[0], out.shape[1] + 1), dtype=out.dtype, order="C")
                buf[:, :out.shape[1]] = out
                out = buf[:, :out.shape[1]]
            else:
                out = np.ascontiguousarray(out)
        _note_block(out.shape, whole=_whole, table_cols=self.shape[1])
        return out

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key, slice(None))
        if len(key) != 2:
            raise IndexError("LazyColumns takes [rows, cols]")
        rk, ck = key
        full_r = isinstance(rk, slice) and rk == slice(None)
        full_c = isinstance(ck, slice) and ck == slice(None)
        if not full_r:
            if not isinstance(rk, slice) and np.ndim(rk) == 0:
                raise NotImplementedError("LazyColumns: single-row reads are not supported")
            sub = self.lazy_rows(rk)
            if full_c:
                return sub
            if not isinstance(ck, (slice, int, np.integer)):
                raise NotImplementedError("LazyColumns: [index rows, index cols] is point-wise in numpy")
            return sub[:, ck]
        if full_c:
            return self
        if isinstance(ck, (int, np.integer)):
            # a column of a C-ordered table is a strided 1-D view in numpy: keep it strided
            return self.block([int(ck)], _strided=(self.layout == "C"))[:, 0]
        if isinstance(ck, slice):
            idx = self._col_index(ck)
            if ck.step in (None, 1):
                # a slice keeps the source layout; of a C-ordered table it is a strided view
                return self.block(idx, _strided=(self.layout == "C"))
            return self.block(idx, layout="F")
        return self.block(ck, layout="F")                    # numpy: A[:, index array] is F-ordered

    def materialize(self):
        """The whole table as today's array (logged with its call site; refusable)."""
        if _forbid_materialize():
            raise MemoryError("whole-table read refused (TRACE_STORE_FORBID_MATERIALIZE=1) "
                              f"at {_call_site()}")
        return self.block(np.arange(self.shape[1]), _whole=True)

    def __array__(self, dtype=None, copy=None):
        out = self.materialize()
        if dtype is not None and np.dtype(dtype) != out.dtype:
            out = out.astype(dtype)                          # keeps the layout, as np.asarray(x, dtype)
        return out

    # -- FaST-LMM
    def snp_reader(self, iid, sid, pos):
        """A pysnptools SnpReader over this table: every block FaST-LMM asks for is today's
        float32 value cast to the number format and order it requests."""
        return _make_reader(self, iid, sid, pos)


def _make_reader(lazy, iid, sid, pos):
    from pysnptools.snpreader import SnpReader

    class LazySnpReader(SnpReader):
        def __init__(self):
            super(LazySnpReader, self).__init__()
            self._row, self._col, self._col_property = iid, sid, pos

        def __repr__(self):
            return f"LazySnpReader({len(self._col)} SNPs)"

        @property
        def row(self):
            return self._row

        @property
        def col(self):
            return self._col

        @property
        def col_property(self):
            return self._col_property

        def copyinputs(self, copier):
            pass

        def _read(self, row_index_or_none, col_index_or_none, order, dtype, force_python_only, view_ok,
                  num_threads):
            dtype = np.dtype(dtype)
            if order == "A":
                # as SnpData (pysnptools.util.sub_matrix): the order of the array it holds, F if
                # that array is F-contiguous (an F table, or a single row or column), else C
                n_, m_ = lazy.shape
                order = "F" if (lazy.layout == "F" or n_ == 1 or m_ == 1) else "C"
            m = len(self._col)
            if col_index_or_none is None:
                ci = np.arange(m)
            elif isinstance(col_index_or_none, slice):
                ci = np.arange(m)[col_index_or_none]
            else:
                ci = np.asarray(col_index_or_none)
            blk = lazy.astype(np.float32).block(ci, layout="F")
            if row_index_or_none is not None:
                ri = (np.arange(blk.shape[0])[row_index_or_none] if isinstance(row_index_or_none, slice)
                      else np.asarray(row_index_or_none))
                blk = blk[ri, :]
            val = np.empty(blk.shape, dtype=dtype, order=order)
            val[...] = blk
            return val

    return LazySnpReader()


# ---------------------------------------------------------------------------------- ingest
@dataclass
class StoreParsed:
    """Stands in for gwas.pipeline.ParsedGenotypes: the same metadata, the table on disk.
    ``stats`` holds exact call counts over all samples and all variants (n1, n2, nmiss and the
    largest value per variant; missing calls per sample; the largest value overall)."""

    store: GenotypeStore
    samples: np.ndarray
    chroms: np.ndarray
    positions: np.ndarray
    vid: object
    ref: object
    alt: object
    info_scores: object
    info_field: object
    stats: dict = field(default_factory=dict)
    ingest: dict = field(default_factory=dict)
    is_store: bool = True

    @property
    def genotypes(self):
        return LazyColumns(self.store, np.arange(self.store.M))


def _info_keys():
    from gwas.io import _INFO_QUALITY_FIELDS

    return tuple(f"variants/{f}" for f in _INFO_QUALITY_FIELDS)


def _sanitized_plain_vcf(vcf_path, store_dir):
    """The cleaned copy of a plain-text VCF (gwas.io.sanitize_plain_vcf_stream) in ``store_dir``.
    Raises load_vcf_cached's error when the '#CHROM' line is missing."""
    from gwas.io import sanitize_plain_vcf_stream

    out = Path(store_dir) / "sanitized.vcf"
    if not sanitize_plain_vcf_stream(vcf_path, out):
        try:
            os.remove(out)
        except OSError:
            pass
        raise RuntimeError(
            "VCF file is missing the mandatory '#CHROM' header line. "
            "The file may be truncated or corrupted."
        )
    return out


class DeferredParseError(Exception):
    """A stop that the in-memory path raises when it PARSES the callset, that is after the
    phenotype file and the trait names were checked. The chunked ingest meets it earlier; the
    CLI raises ``original`` at the place the in-memory path would."""

    def __init__(self, original):
        super().__init__(str(original))
        self.original = original


_NO_RECORDS = ("VCF file contains no variant records. "
               "Check that the file has data lines below the #CHROM header.")


def ingest_vcf(vcf_path, store_dir, chunk_bytes=64 * 2**20, log=None, chunk_records=None):
    """Read the VCF in record chunks into a new store in ``store_dir``; return a StoreParsed.

    The fields are those of gwas.io.load_vcf_cached. A plain-text VCF is first cleaned as
    load_vcf_cached cleans it (into the store folder, deleted after the read). Each chunk is
    parsed by gwas.qc._pipeline_parse_vcf_biallelic, so the stored calls decode to today's
    parsed values and the metadata arrays are today's, concatenated. The imputation-quality
    field is chosen over all records, as gwas.io._extract_info_scores does. The chunk length
    comes from a byte budget (a GT chunk holds records x samples x 2 bytes), at most 65,536
    records (scikit-allel's default); ``chunk_records`` (or TRACE_STORE_CHUNK_RECORDS) sets it.
    """
    import gzip

    import allel
    import pandas as pd

    from gwas.io import info_one_value_per_record
    from gwas.qc import _pipeline_parse_vcf_biallelic

    t0 = time.perf_counter()
    vcf_path = Path(vcf_path)
    store_dir = Path(store_dir)
    if store_dir.exists() and any(store_dir.iterdir()):   # a new folder, or one just made empty
        raise FileExistsError(f"the folder for the genotype table is not empty: {store_dir}")
    store_dir.mkdir(parents=True, exist_ok=True)
    with open(vcf_path, "rb") as fh:
        gz = fh.read(2) == b"\x1f\x8b"
    tmp_plain = None
    if gz:
        stream = gzip.open(vcf_path, mode="rb")
    else:
        tmp_plain = _sanitized_plain_vcf(vcf_path, store_dir)
        stream = open(tmp_plain, "rb")

    info_keys = _info_keys()
    fields = ["calldata/GT", "variants/CHROM", "variants/POS", "variants/ALT", "variants/REF",
              "variants/ID"] + list(info_keys)
    acc = {k: [] for k in ("chroms", "positions", "vid", "ref", "alt", "n1", "n2", "nmiss", "vmax")}
    info_acc = {k: [] for k in info_keys}
    info_seen = {k: False for k in info_keys}
    info_finite = {k: False for k in info_keys}
    n_all = 0
    have = {"vid": False, "ref": False, "alt": False}
    samples = None
    row_nmiss = None
    M = n_chunks = 0
    gmax = -1
    calls_path = store_dir / "calls.i8"
    try:
        _f, samples_raw, _h, it = allel.iter_vcf_chunks(stream, fields=list(fields), chunk_length=65536)
        n_samp = max(1, len(samples_raw))
        L = int(min(65536, max(1024, chunk_bytes // (2 * n_samp))))
        if chunk_records is None and os.environ.get("TRACE_STORE_CHUNK_RECORDS", ""):
            chunk_records = int(os.environ["TRACE_STORE_CHUNK_RECORDS"])
        if chunk_records is not None:
            L = int(chunk_records)
        if L != 65536:
            stream.close()
            stream = gzip.open(vcf_path, mode="rb") if gz else open(tmp_plain, "rb")
            _f, samples_raw, _h, it = allel.iter_vcf_chunks(stream, fields=list(fields), chunk_length=L)
        with open(calls_path, "wb") as out:
            for chunk, _len, _c, _p in it:
                chunk = dict(chunk)
                chunk["samples"] = samples_raw
                if chunk.get("calldata/GT") is None or chunk["calldata/GT"].shape[0] == 0:
                    raise DeferredParseError(ValueError(_NO_RECORDS))   # as _pipeline_parse_vcf_biallelic
                n_rec = int(chunk["calldata/GT"].shape[0])
                # Which records of this chunk today's parse keeps: a record counter rides
                # through it as the first imputation-quality field; the real fields are set
                # aside and handled below over all records.
                real_info = {k: chunk.get(k) for k in info_keys}
                chunk[info_keys[0]] = np.arange(n_rec, dtype=np.float64)
                for k in info_keys[1:]:
                    chunk.pop(k, None)
                try:
                    G, smp, ch, ps, v_id, ref, alt, kept, _fld = _pipeline_parse_vcf_biallelic(chunk)
                except Exception as exc:          # in memory this stop comes after the phenotype checks
                    raise DeferredParseError(exc) from exc
                kept = np.asarray(kept).astype(np.int64) if kept is not None else np.arange(G.shape[1])
                if samples is None:
                    samples = smp
                    row_nmiss = np.zeros(len(samples), dtype=np.int64)
                g8 = encode_calls(G)                                    # (N, m) F view of (m, N) C
                out.write(np.ascontiguousarray(g8.T).tobytes())
                miss = g8 < 0
                acc["n1"].append((g8 == 1).sum(axis=0, dtype=np.int64))
                acc["n2"].append((g8 == 2).sum(axis=0, dtype=np.int64))
                acc["nmiss"].append(miss.sum(axis=0, dtype=np.int64))
                acc["vmax"].append(g8.max(axis=0) if g8.shape[0] else np.full(g8.shape[1], -1, np.int8))
                row_nmiss += miss.sum(axis=1, dtype=np.int64)
                if g8.size:
                    gmax = max(gmax, int(g8.max()))
                acc["chroms"].append(ch)
                acc["positions"].append(ps)
                for name, arr in (("vid", v_id), ("ref", ref), ("alt", alt)):
                    if arr is not None:
                        have[name] = True
                        acc[name].append(arr)
                for k in info_keys:
                    if real_info[k] is not None:
                        # gwas.io._extract_info_scores, per chunk: the same conversion; whether
                        # the field has any finite value is decided over ALL records, as today
                        info_seen[k] = True
                        raw = info_one_value_per_record(real_info[k], chunk.get("variants/ALT", None))
                        sc = pd.to_numeric(raw, errors="coerce").astype(np.float32)
                        if sc.size and np.any(np.isfinite(sc)):
                            info_finite[k] = True
                        info_acc[k].append(sc[kept])
                n_all += n_rec
                M += int(g8.shape[1])
                n_chunks += 1
                del G, g8, miss, chunk
    finally:
        stream.close()
        if tmp_plain is not None:
            try:
                os.remove(tmp_plain)
            except OSError:
                pass
    if n_chunks == 0 or samples is None:
        raise DeferredParseError(ValueError(_NO_RECORDS))
    cat = np.concatenate
    info_scores = info_field = None
    for k in info_keys:              # the first field with any finite value over all records
        if info_seen[k] and info_finite[k]:
            info_scores, info_field = cat(info_acc[k]), k.split("/", 1)[1]
            break
    stats = {"n1": cat(acc["n1"]), "n2": cat(acc["n2"]), "nmiss": cat(acc["nmiss"]),
             "row_nmiss": row_nmiss, "gmax": gmax, "vmax": cat(acc["vmax"]), "n": int(len(samples))}
    store = GenotypeStore(store_dir, len(samples), M)
    parsed = StoreParsed(
        store=store, samples=samples, chroms=cat(acc["chroms"]), positions=cat(acc["positions"]),
        vid=cat(acc["vid"]) if have["vid"] else None, ref=cat(acc["ref"]) if have["ref"] else None,
        alt=cat(acc["alt"]) if have["alt"] else None, info_scores=info_scores, info_field=info_field,
        stats=stats,
        ingest={"seconds": round(time.perf_counter() - t0, 3), "chunks": n_chunks, "chunk_length": L,
                "calls_bytes": int(calls_path.stat().st_size), "gz": bool(gz)})
    for p in (parsed.samples, parsed.chroms, parsed.positions, parsed.vid, parsed.ref, parsed.alt,
              parsed.info_scores):
        if isinstance(p, np.ndarray):
            p.setflags(write=False)          # a trait must never modify the shared parse
    (store_dir / "meta.json").write_text(json.dumps(
        {"format": "int8, variant-major (variants x samples), -1 = missing call",
         "n_samples": int(len(samples)), "n_variants": int(M), "vcf": str(vcf_path), **parsed.ingest},
        indent=1))
    if log is not None:
        log.info("  genotype table on disk: %s (%d samples x %d variants, %.1f MB, %d chunks, %.1f s)",
                 store_dir, len(samples), M, parsed.ingest["calls_bytes"] / 1e6, n_chunks,
                 parsed.ingest["seconds"])
    return parsed
