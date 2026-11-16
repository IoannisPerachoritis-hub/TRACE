"""The GUI's genotype file and genotype table (the GWAS page, One-Click and the Post-GWAS page).

Where the VCF comes from
    check_vcf_location  the location box: a .vcf or .vcf.gz file the server can read
    save_upload         an upload written once, in pieces, into a folder of this server in the work
                        folder; the page then keeps only its path, never the bytes
    file_identity       the content hash every cache key names: blake2b-256 of the file read in
                        pieces, the same value gwas.utils.hash_bytes gives the whole file (so the
                        keys and seeds built on it keep their values), computed once per
                        (path, size, modification time)

Where the table is kept
    default_choice, default_work_dir, path_input_disabled
                        the environment's defaults (TRACE_GENO_STORE, TRACE_GENO_STORE_DIR,
                        TRACE_DISABLE_PATH_INPUT), so a server or a Docker image can set them
    page_message, page_stop
                        gwas.storeplan's message and stops in the page's words (no command-line
                        flags)
    TableRegistry       at most two tables per server, keyed by (file identity, mode): the parsed
                        genotypes in memory (read-only arrays) or a table file on disk in its own
                        folder, _trace_geno_store_gui_<pid>_<random>. A folder is removed when its
                        entry is evicted, by Clear and when the server stops normally
    get_registry        the server's registry, held under st.cache_resource
    SessionTable        the small object the GWAS page leaves in the session for One-Click and the
                        Post-GWAS page

"Clear session data"
    clear_all           closes and deletes this server's tables and saved uploads, empties every cached
                        result, removes TRACE's .memo files from Streamlit's cache folder and drops the
                        session's keys (the GWAS page's and the Post-GWAS page's)

Shared arrays are read-only (``flags.writeable`` False), so an in-place change fails loudly
instead of altering another session's data.
"""
import atexit
import hashlib
import os
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VCF_SUFFIXES = (".vcf", ".vcf.gz")
HASH_PIECE = 8 * 2**20                     # bytes read at a time when hashing or saving a file
GUI_STORE_PREFIX = "_trace_geno_store_gui"
UPLOAD_PREFIX = "_trace_upload_gui"
CHOICES = ("auto", "memory", "disk")
CHOICE_LABELS = {"auto": "Automatic", "memory": "In memory", "disk": "On disk"}


# ----------------------------------------------------------------- the environment's defaults
def path_input_disabled(environ=None):
    """TRACE_DISABLE_PATH_INPUT=1 hides the location box (a shared server, whose users should
    not open the server's files)."""
    env = os.environ if environ is None else environ
    return str(env.get("TRACE_DISABLE_PATH_INPUT", "") or "").strip() == "1"


def default_choice(environ=None):
    """Where to keep the table unless the page says otherwise: TRACE_GENO_STORE (auto, memory or
    disk), else auto. Any other value is ignored."""
    env = os.environ if environ is None else environ
    v = str(env.get("TRACE_GENO_STORE", "") or "").strip().lower()
    return v if v in CHOICES else "auto"


def default_work_dir(environ=None):
    """The folder for saved uploads and table files: TRACE_GENO_STORE_DIR, else a folder
    ``trace_gui`` in the system's temporary folder."""
    env = os.environ if environ is None else environ
    v = str(env.get("TRACE_GENO_STORE_DIR", "") or "").strip()
    return Path(v) if v else Path(tempfile.gettempdir()) / "trace_gui"


def size_text(n_bytes):
    n = float(n_bytes)
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{int(n)} bytes"


# ------------------------------------------------------------------------ the VCF's location
@dataclass
class LocationCheck:
    ok: bool
    path: object = None     # the resolved Path when ok
    size: int = 0           # bytes
    message: str = ""


def check_vcf_location(text):
    """The location box's check: a .vcf or .vcf.gz file that exists and can be read. Quotes
    around the text (Windows' "Copy as path") and a leading ~ are accepted."""
    raw = str(text or "").strip().strip('"').strip("'").strip()
    if not raw:
        return LocationCheck(False, message="Type the full location of a .vcf or .vcf.gz file.")
    p = Path(os.path.expanduser(raw))
    if not p.name.lower().endswith(VCF_SUFFIXES):
        return LocationCheck(False, message=f"TRACE reads .vcf and .vcf.gz files; {p.name} is neither.")
    if not p.exists():
        return LocationCheck(False, message=f"No file at {p}.")
    if not p.is_file():
        return LocationCheck(False, message=f"{p} is a folder, not a file.")
    try:
        with open(p, "rb") as fh:
            fh.read(1)
    except OSError as exc:
        return LocationCheck(False, message=f"TRACE cannot read {p} ({exc.strerror or exc}).")
    size = p.stat().st_size
    return LocationCheck(True, path=p.resolve(), size=size, message=f"{p.name}: {size_text(size)}")


# ------------------------------------------------------------------------- the file identity
_ID_CACHE = OrderedDict()                  # (path, size, mtime_ns) -> identity
_ID_MAX = 64
_ID_LOCK = threading.Lock()


def _id_key(path):
    st = os.stat(path)
    return (os.path.normcase(os.path.abspath(path)), int(st.st_size), int(st.st_mtime_ns))


def _remember(key, identity):
    with _ID_LOCK:
        _ID_CACHE[key] = identity
        _ID_CACHE.move_to_end(key)
        while len(_ID_CACHE) > _ID_MAX:
            _ID_CACHE.popitem(last=False)


def file_identity(path):
    """blake2b-256 of the file's content, read in pieces: the value gwas.utils.hash_bytes gives
    the whole file. Computed once per (path, size, modification time)."""
    key = _id_key(path)
    with _ID_LOCK:
        if key in _ID_CACHE:
            _ID_CACHE.move_to_end(key)
            return _ID_CACHE[key]
    h = hashlib.blake2b(digest_size=32)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(HASH_PIECE), b""):
            h.update(chunk)
    identity = h.hexdigest()
    _remember(key, identity)
    return identity


def forget_file_identities():
    """Empty the file identities this server remembers (Clear): a file rewritten in place with
    the same size and modification time is then hashed again when next used."""
    with _ID_LOCK:
        _ID_CACHE.clear()


# ---------------------------------------------------------------------------- saved uploads
_SAVED_UPLOADS = set()                     # folders of uploads this server saved
_SAVED_LOCK = threading.Lock()


def save_upload(uploaded, work_dir):
    """Write the upload (a Streamlit UploadedFile, or any file-like object with ``name`` and
    ``read``) once, in pieces, into a new folder of this server in ``work_dir``, under its own
    name. The identity is computed while writing. Returns (path, identity)."""
    from gwas.genostore import new_store_dir

    name = Path(str(getattr(uploaded, "name", "") or "upload.vcf")).name
    folder = new_store_dir(work_dir, prefix=UPLOAD_PREFIX)
    with _SAVED_LOCK:
        _SAVED_UPLOADS.add(folder)
    out = folder / name
    h = hashlib.blake2b(digest_size=32)
    if hasattr(uploaded, "seek"):
        uploaded.seek(0)
    with open(out, "wb") as fh:
        for chunk in iter(lambda: uploaded.read(HASH_PIECE), b""):
            fh.write(chunk)
            h.update(chunk)
    identity = h.hexdigest()
    _remember(_id_key(out), identity)
    return out, identity


def saved_upload_folders():
    with _SAVED_LOCK:
        return sorted(_SAVED_UPLOADS)


def remove_saved_uploads():
    """Delete every upload this server saved (Clear, and when the server stops)."""
    import shutil

    with _SAVED_LOCK:
        folders = list(_SAVED_UPLOADS)
        _SAVED_UPLOADS.clear()
    for f in folders:
        shutil.rmtree(f, ignore_errors=True)


atexit.register(remove_saved_uploads)


# ------------------------------------------------------------- storeplan in the page's words
def page_message(plan):
    """storeplan's log line in the page's words (no command-line flags)."""
    return (plan.message
            .replace(" (--geno-store memory)", " (as chosen under Large datasets)")
            .replace(" (--geno-store disk)", " (as chosen under Large datasets)"))


def page_stop(plan, work_dir):
    """storeplan's stop reworded for the page: what to change on the page, not which flag."""
    from gwas import storeplan as sp

    e_gb = plan.estimate_mib * sp.MIB / sp.GB
    a_gb = plan.available_bytes / sp.GB
    calls = sp._calls_text(plan.size)
    kind = getattr(plan, "stop_kind", "")
    if kind == "memory":
        return (f"An in-memory run of this dataset ({calls} genotype calls) would need about "
                f"{e_gb:.1f} GB of memory, and {a_gb:.1f} GB are free. Choose On disk under Large "
                "datasets (the same results with far less memory), or Automatic.")
    if kind == "disk":
        need = sp.disk_need_bytes(plan.size) / sp.GB
        free = sp.free_disk_bytes(work_dir) / sp.GB
        return (f"The genotype table on disk needs about {need:.1f} GB in {work_dir}, and "
                f"{free:.1f} GB are free there. Free some space, or choose a work folder on "
                "another drive under Large datasets.")
    if kind == "ldknni":
        if plan.choice == "auto":
            return (f"LD-kNNi imputation ranks every SNP against all others and needs the whole "
                    f"genotype table in memory: about {e_gb:.1f} GB for this dataset, and "
                    f"{a_gb:.1f} GB are free. Choose mean imputation, which works with the table "
                    "on disk, or free some memory.")
        return ("LD-kNNi imputation ranks every SNP against all others and needs the whole "
                "genotype table in memory, so it cannot run with the table on disk. Choose mean "
                "imputation, or keep the table in memory under Large datasets.")
    if kind == "export_qc":
        return (f"Exporting the QC'd genotype table writes it whole as text ({calls} calls), which "
                "is not available for a dataset this large with the table on disk.")
    return plan.stop or ""


# --------------------------------------------------------------------------- the registry
def _read_only(parsed):
    """Mark every array of a parsed VCF read-only (the in-memory parse is shared by sessions)."""
    for v in vars(parsed).values() if hasattr(parsed, "__dict__") else ():
        if isinstance(v, np.ndarray):
            v.setflags(write=False)
    return parsed


class TableRegistry:
    """The tables of this server: at most ``max_entries``, keyed by (file identity, mode).

    memory  the parsed genotypes (gwas.pipeline.ParsedGenotypes), read once from the file
            without Streamlit's byte-keyed cache; their arrays are read-only
    disk    a table file built by gwas.genostore.ingest_vcf in its own new folder in the work
            folder (``_trace_geno_store_gui_<pid>_<random>``)

    A new table is built first; only once it is open is the least recently used entry evicted
    (a disk entry's table is closed and its folder deleted), so a VCF that cannot be read
    closes nothing. A table is built outside the registry's lock, one build per key at a time:
    other sessions' look-ups, other builds and the folder listing do not wait for it.
    ``close_all`` (Clear, and the server's normal stop) drops everything.

    The limit counts the registry's own entries. An evicted in-memory table is not closed (it
    cannot be): it stays in memory while a session still holds it, until that session's next
    run takes the registry's table. Nothing here is ever pickled."""

    MAX_ENTRIES = 2

    def __init__(self, max_entries=MAX_ENTRIES):
        self.max_entries = int(max_entries)
        self._lock = threading.RLock()       # the entries; never held while a table is built
        self._entries = OrderedDict()
        self._builds = {}                    # key -> the lock its build holds
        self._building_dirs = set()          # folders of disk tables being built
        atexit.register(self.close_all)

    def _hit(self, key):
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
        return entry

    def get(self, identity, mode, vcf_path, work_dir=None, log=None):
        """The table of ``vcf_path`` in ``mode`` ("memory" or "disk"), built if needed."""
        if mode not in ("memory", "disk"):
            raise ValueError(f"unknown table mode {mode!r}")
        key = (str(identity), mode)
        with self._lock:
            entry = self._hit(key)
            if entry is not None:
                return entry["parsed"]
            build = self._builds.setdefault(key, threading.Lock())
        evicted = []
        with build:                                   # one build per key
            try:
                with self._lock:
                    entry = self._hit(key)            # built meanwhile by another session
                    if entry is not None:
                        return entry["parsed"]
                entry = self._open(mode, vcf_path, work_dir, log)      # a failure evicts nothing
                with self._lock:
                    self._entries[key] = entry
                    self._entries.move_to_end(key)
                    self._building_dirs.discard(entry.get("dir"))
                    while len(self._entries) > self.max_entries:
                        evicted.append(self._entries.popitem(last=False)[1])
            finally:
                with self._lock:
                    if self._builds.get(key) is build:
                        del self._builds[key]
        for old in evicted:
            self._release(old)
        return entry["parsed"]

    def _open(self, mode, vcf_path, work_dir, log):
        if mode == "memory":
            from gwas.io import load_vcf_cached
            from gwas.pipeline import ParsedGenotypes

            load = getattr(load_vcf_cached, "__wrapped__", load_vcf_cached)    # not the cache
            parsed = ParsedGenotypes.from_callset(load(Path(vcf_path).read_bytes()))
            return {"parsed": _read_only(parsed), "dir": None}
        from gwas import genostore

        store_dir = genostore.new_store_dir(work_dir if work_dir is not None else default_work_dir(),
                                            prefix=GUI_STORE_PREFIX)
        with self._lock:
            self._building_dirs.add(store_dir)        # this server's folder while it is built
        try:
            parsed = genostore.ingest_vcf(vcf_path, store_dir, log=log)
        except BaseException:
            genostore.remove_store(store_dir)
            with self._lock:
                self._building_dirs.discard(store_dir)
            raise
        return {"parsed": parsed, "dir": store_dir}

    @staticmethod
    def _release(entry):
        if entry.get("dir") is not None:
            from gwas import genostore

            genostore.remove_store(entry["dir"], entry["parsed"])

    def keys(self):
        with self._lock:
            return list(self._entries)

    def folders(self):
        """The folders of this registry's disk tables, those being built included."""
        with self._lock:
            return ([e["dir"] for e in self._entries.values() if e.get("dir") is not None]
                    + sorted(self._building_dirs))

    def drop(self, identity=None, mode=None):
        """Drop (and close) the entries of ``identity`` and/or ``mode``; all of them by default."""
        with self._lock:
            gone = [self._entries.pop(k) for k in list(self._entries)
                    if (identity is None or k[0] == str(identity)) and (mode is None or k[1] == mode)]
        for entry in gone:
            self._release(entry)

    def close_all(self):
        self.drop()


def _new_registry():
    return TableRegistry()


try:                                        # the GUI: one registry per server process
    import streamlit as _st

    get_registry = _st.cache_resource(show_spinner=False)(_new_registry)
except ImportError:                         # pragma: no cover - streamlit is a TRACE dependency
    import functools

    get_registry = functools.lru_cache(maxsize=1)(_new_registry)


def own_folders():
    """The folders this server made (table files and saved uploads): never listed as stale."""
    return set(get_registry().folders()) | set(saved_upload_folders())


# --------------------------------------------------------------------- Clear session data
# TRACE's .memo files in Streamlit's cache folder (~/.streamlit/cache, written by st.cache_data
# with persist="disk"): named <function key>-<value key>.memo. TRACE's one persisted cache was
# gwas.qc._impute_cached until batch 13; these are the function keys of all its versions, from
# Streamlit 1.50's _make_function_key over the git history of gwas/qc.py.
TRACE_MEMO_FUNCTION_KEYS = (
    "b3e084e866e81f46fac0e0291c06e77b",     # c359acf .. a585645 (the source did not change)
    "1802ca04726892ccb1ead35e3a04d72e",     # 7c312e0, the first version
)

# The session keys the Post-GWAS page and its tabs write at ff56e5c (pages/Post_GWAS_Analysis.py,
# pages/_ld_tabs/*): Clear drops them, together with gwas.guipost's lists (the page's keys now;
# the two overlap, and the old list also holds keys the page no longer writes).
POST_GWAS_KEYS_FF56E5C = (
    "r2_cache", "ld_geno_hard", "ld_chroms", "ld_positions", "ld_sid", "geno_hard", "chroms", "positions",
    "sid", "_ld_prev_trait", "haplo_df_auto", "haplo_df_auto_hash", "ld_blocks_final", "ld_blocks",
    "ld_window_buffer_kb", "genes_df", "hap_tables", "hap_tables_vis", "pheno_vis", "sample_vis",
    "_hap_pca_cache", "_hap_box_cache", "adj_r2_min", "ld_block_metadata_auto", "geno_row_ids",
    "pheno_used_for_hap", "trait_col", "trait_col_locked", "pheno_used_for_hap_label",
    "haplo_blocks_to_use", "pcs_for_hap", "hap_gwas_df", "hap_gwas_df_enriched", "_local_ld_cache",
)
POST_GWAS_PREFIXES_FF56E5C = ("LDDETECT::", "HAPGWAS::")


def streamlit_cache_folder(environ=None):
    """Where Streamlit 1.50 writes the files of the caches it keeps on disk: ~/.streamlit/cache
    (TRACE_STREAMLIT_CACHE_DIR names another folder: the tests' own, never the home folder)."""
    env = os.environ if environ is None else environ
    if str(env.get("TRACE_STREAMLIT_CACHE_DIR", "") or "").strip():
        return Path(env["TRACE_STREAMLIT_CACHE_DIR"])
    from streamlit.runtime.caching.storage.local_disk_cache_storage import get_cache_folder_path

    return Path(get_cache_folder_path())


def trace_memo_files(folder=None):
    """TRACE's .memo files in Streamlit's cache folder (other apps' files are not TRACE's)."""
    folder = streamlit_cache_folder() if folder is None else Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.name.endswith(".memo")
                  and p.name.split("-", 1)[0] in TRACE_MEMO_FUNCTION_KEYS)


def remove_trace_memo_files(folder=None):
    """Delete TRACE's .memo files; returns (files, bytes) removed."""
    n = size = 0
    for p in trace_memo_files(folder):
        try:
            b = p.stat().st_size
            p.unlink()
        except OSError:
            continue
        n, size = n + 1, size + b
    return n, size


def clear_cached_results():
    """Empty every st.cache_data and st.cache_resource cache of this server (all of them are
    TRACE's), each function's cache on its own: st.cache_data.clear() would, on a running server,
    delete Streamlit's whole cache folder, other apps' files too. The server's table registry
    (get_registry) is kept: clear_all empties it, and every session goes on using the one
    registry, so a table built while Clear runs is kept in it and no old registry is left."""
    from streamlit.runtime.caching import cache_data_api, cache_resource_api

    keep = {getattr(get_registry, "_function_key", None)} - {None}
    for caches in (cache_data_api._data_caches, cache_resource_api._resource_caches):
        with caches._caches_lock:
            function_caches = [c for k, c in caches._function_caches.items() if k not in keep]
        for cache in function_caches:
            cache.clear()


def session_keys_to_clear(keys=(), prefixes=()):
    """(keys, prefixes) Clear drops from the session: the caller's, the Post-GWAS page's at
    ff56e5c, and the Post-GWAS page's now (gwas.guipost's lists)."""
    from gwas import guipost                # here: the GWAS page needs it only when Clear runs

    keys = set(keys) | set(POST_GWAS_KEYS_FF56E5C) | set(guipost.POST_GWAS_SESSION_KEYS)
    prefixes = (tuple(prefixes) + POST_GWAS_PREFIXES_FF56E5C
                + tuple(guipost.POST_GWAS_SESSION_PREFIXES))
    return keys, prefixes


def clear_all(state=None, keys=(), prefixes=(), memo_folder=None):
    """The GWAS page's "Clear session data", in this order:
    1. this server's tables (the registry: in memory, and the table files with their folders on
       disk) are closed and deleted, and the uploads it saved are deleted; the registry itself
       stays, empty (a table another session builds meanwhile is kept in it);
    2. every cached result (st.cache_data and st.cache_resource) is emptied, and the file
       identities this server remembers (a file is hashed again when next used);
    3. TRACE's .memo files are removed from Streamlit's cache folder;
    4. the session's keys (``keys``, ``prefixes`` and the Post-GWAS page's) are dropped from
       ``state``.
    Returns a summary dict (for the page's message and the tests)."""
    registry = get_registry()
    folders = [str(f) for f in registry.folders()] + [str(f) for f in saved_upload_folders()]
    registry.close_all()
    remove_saved_uploads()
    clear_cached_results()
    forget_file_identities()
    n_memo, memo_bytes = remove_trace_memo_files(memo_folder)
    dropped = []
    if state is not None:
        keys, prefixes = session_keys_to_clear(keys, prefixes)
        for k in list(state.keys()):
            if isinstance(k, str) and (k in keys or k.startswith(prefixes)):
                state.pop(k, None)
                dropped.append(k)
    return {"folders": folders, "memo_files": n_memo, "memo_bytes": memo_bytes, "session_keys": dropped}


# ------------------------------------------------------------------------ the session's table
def table_identity(file_identity, **settings):
    """The identity of one trait's genotype table: the file's identity and every setting that
    changes the table (QC and imputation settings, the trait and its samples, covariates, ...),
    as one short string for cache and session keys."""
    payload = repr((str(file_identity), sorted((str(k), repr(v)) for k, v in settings.items())))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


@dataclass
class SessionTable:
    """What the GWAS page leaves in the session for One-Click and the Post-GWAS page.

    identity        the trait table's identity: the file's identity and every setting that
                    changes the table (QC, imputation, samples, covariates, chromosome options)
    file_identity   the VCF's identity
    mode            "memory" or "disk"
    parsed          the registry's parsed genotypes or table file (shared, read-only)
    view            the trait's mean-imputed table: the in-memory float32 array, or a
                    gwas.genostore.LazyColumns that stays on disk (with its fill)
    raw             the trait's raw dosage table, missing calls NaN (the Post-GWAS page's MAF,
                    significant SNPs and regional r2 read it): the array, or a LazyColumns
                    without fill
    rows            the trait's rows in ``parsed`` (None: the view's own rows)
    fill            the per-SNP fill of the mean-imputed table on disk (None in memory)
    sid, chroms, positions, iid
                    the SNP and sample labels of the view
    """

    identity: str
    file_identity: str
    mode: str
    parsed: object = None
    view: object = None
    raw: object = None
    rows: object = None
    fill: object = None
    sid: object = None
    chroms: object = None
    positions: object = None
    iid: object = None

    @property
    def shape(self):
        return tuple(int(x) for x in self.view.shape) if self.view is not None else (0, 0)


def new_upload_token(uploaded):
    """What identifies one upload in the session (so a rerun does not save it again)."""
    return (str(getattr(uploaded, "name", "")), int(getattr(uploaded, "size", 0) or 0),
            str(getattr(uploaded, "file_id", "") or ""))
