""""Clear session data" frees everything (gwas.guitable.clear_all, the GWAS page's button).

After it, none of this server's tables (in memory, or table files with their folders on disk), saved
uploads or cached results (st.cache_data and st.cache_resource) remains; TRACE's .memo files are gone from
Streamlit's cache folder (another app's file stays: Streamlit's own clear would delete the whole folder);
and the session's keys are dropped, the GWAS page's and the Post-GWAS page's (ff56e5c's list, and
gwas.guipost's), while the widget settings stay.
"""
import io
import logging
import warnings
from pathlib import Path

import numpy as np
import pytest

from gwas import guigwas, guitable, kinship

REPO = Path(__file__).resolve().parents[1]
EX1 = REPO / "tests" / "data" / "example_v1" / "example.vcf.gz"


class _Upload(io.BytesIO):
    def __init__(self, data, name):
        super().__init__(data)
        self.name, self.size, self.file_id = name, len(data), "f1"


def _data_cache_entries():
    """Every entry of every st.cache_data cache of this process (Streamlit's cache stats: the
    entries are pickled bytes, so sizing them is safe)."""
    from streamlit.runtime.caching import cache_data_api

    return cache_data_api.get_data_cache_stats_provider().get_stats()


def _n_cache_entries():
    """How many entries the st.cache_data and st.cache_resource caches of this process hold,
    leaving out the server's table registry (gwas.guitable.get_registry), which Clear empties and
    keeps.

    The st.cache_resource entries are counted, not sized: Streamlit's stats size every live value
    with pympler, which raises on a NumPy array that does not own its data (the read-only PCs, for
    one) when that is the first array it sizes in the process, and which array comes first depends
    on the order in which earlier tests first called the cached functions. st.cache_data entries
    are pickled bytes, so their stats are safe."""
    from streamlit.runtime.caching import cache_resource_api

    registry = getattr(guitable.get_registry, "_function_key", None)
    caches = cache_resource_api._resource_caches
    with caches._caches_lock:
        function_caches = [c for k, c in caches._function_caches.items() if k != registry]
    return len(_data_cache_entries()) + sum(len(cache._mem_cache) for cache in function_caches)


@pytest.fixture(autouse=True)
def _quiet():
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_clear_frees_tables_uploads_cached_results_memo_files_and_session_keys(tmp_path, monkeypatch):
    memo = tmp_path / "streamlit_cache"
    memo.mkdir()
    monkeypatch.setenv("TRACE_STREAMLIT_CACHE_DIR", str(memo))
    trace_files = [memo / f"{key}-{'0' * 32}.memo" for key in guitable.TRACE_MEMO_FUNCTION_KEYS]
    for f in trace_files:
        f.write_bytes(b"x" * 1000)
    other_app = memo / f"{'a' * 32}-{'1' * 32}.memo"
    other_app.write_bytes(b"y")
    not_memo = memo / f"{guitable.TRACE_MEMO_FUNCTION_KEYS[0]}-notes.txt"
    not_memo.write_bytes(b"z")

    # this server's tables (in memory and on disk) and a saved upload
    reg = guitable.get_registry()
    work = tmp_path / "work"
    disk = reg.get("clear-test-disk", "disk", EX1, work)
    reg.get("clear-test-memory", "memory", EX1)
    folders = reg.folders()
    assert folders and all(f.exists() for f in folders)
    upload, _ = guitable.save_upload(_Upload(EX1.read_bytes(), "panel.vcf.gz"), work)
    assert upload.exists()
    # cached results: a st.cache_data entry and st.cache_resource entries
    rng = np.random.default_rng(0)
    G = rng.integers(0, 3, size=(30, 60)).astype(np.float32)
    guigwas.ld_decay("clear-test", G, np.repeat(np.array(["1", "2"]), 30), np.tile(np.arange(30) * 1000, 2))
    kinship.compute_pcs_full_cached("clear-test", G, max_pcs=2)
    assert _n_cache_entries() >= 3
    # the session: the GWAS page's keys, the Post-GWAS page's, and widget settings that stay
    page_keys = {"_persist_vcf_path": "x", "geno_imputed": 1, "geno_table": 1, guigwas.PREP_KEY: 1,
                 guigwas.MEFF_KEY: 1, "gwas_zip_pheno__T__MLM__n1": {}, "BOOT_GWAS::T::id": {},
                 "ARRKEY::Y_VEC::id": 1, "GWAS_CTX": {}}
    post_keys = {"haplo_df_auto": {}, "HAPGWAS::T::x::tables": {}, "LDDETECT::T::x": 1, "_local_ld_cache": {},
                 "r2_cache": {}, "ld_blocks": 1, "geno_hard": 1, "hap_gwas_df": 1}
    widgets = {"sig_rule_select": "Bonferroni (α = 0.05)", "selected_traits_multiselect": ["T"],
               "geno_store_choice": "auto"}
    state = {**page_keys, **post_keys, **widgets}

    out = guitable.clear_all(state, keys=guigwas.SESSION_KEYS, prefixes=guigwas.SESSION_PREFIXES)

    assert _n_cache_entries() == 0 and _data_cache_entries() == []    # no cached result is left
    assert not any(f.exists() for f in folders)                       # the table files and folders
    with pytest.raises(ValueError, match="closed"):                   # ... closed first
        disk.store.gather([0])
    assert not upload.exists() and not upload.parent.exists() and guitable.saved_upload_folders() == []
    assert not any(f.exists() for f in trace_files) and other_app.exists() and not_memo.exists()
    assert out["memo_files"] == len(trace_files) and out["memo_bytes"] == 1000 * len(trace_files)
    assert state == widgets
    assert set(out["session_keys"]) == set(page_keys) | set(post_keys)
    assert guitable.get_registry() is reg and reg.keys() == []        # the same registry, empty


def test_clear_keeps_the_one_registry_and_forgets_file_identities(tmp_path, monkeypatch):
    """Clear empties the server's registry but keeps the object every session uses: a table a
    session builds while Clear runs lands in it (it is this server's, not a stale folder, and the
    session's next run finds it), and repeated Clears add no registry and no exit hook. The file
    identities are forgotten, so a file rewritten in place is hashed again."""
    import atexit
    import threading

    from gwas import genostore

    monkeypatch.setenv("TRACE_STREAMLIT_CACHE_DIR", str(tmp_path / "memo"))
    reg = guitable.get_registry()
    work = tmp_path / "work"
    reg.get("clear-race-old", "disk", EX1, work)
    guitable.file_identity(EX1)
    assert len(guitable._ID_CACHE) >= 1
    started, release = threading.Event(), threading.Event()
    real_ingest = genostore.ingest_vcf

    def slow_ingest(*a, **k):
        started.set()
        release.wait(30)
        return real_ingest(*a, **k)

    monkeypatch.setattr(genostore, "ingest_vcf", slow_ingest)
    got = {}
    t = threading.Thread(target=lambda: got.setdefault("b", guigwas.table("clear-race-b", "disk", EX1, work)))
    t.start()
    assert started.wait(30)
    n_hooks = atexit._ncallbacks()
    for _ in range(3):
        guitable.clear_all({})                                        # Clear while B builds
    assert atexit._ncallbacks() == n_hooks                            # no new registry, no new hook
    assert len(guitable._ID_CACHE) == 0
    release.set()
    t.join(60)
    try:
        assert guitable.get_registry() is reg
        assert reg.keys() == [("clear-race-b", "disk")]               # B's table landed in it
        (folder,) = reg.folders()
        assert folder in guitable.own_folders()                       # never listed as stale
        assert guigwas.table("clear-race-b", "disk", EX1, work) is got["b"]   # no second copy
    finally:
        guitable.clear_all({})
    assert reg.keys() == [] and not folder.exists()


def test_no_trace_cache_persists_to_disk_any_more():
    """TRACE's .memo files are those of gwas.qc._impute_cached, its only cache with persist="disk"
    until batch 13 (TRACE_MEMO_FUNCTION_KEYS: Streamlit 1.50's function keys of its versions). No
    TRACE cache persists to disk now, so Clear only ever removes leftovers."""
    import re

    from gwas import qc

    assert not hasattr(qc, "gwas_pipeline") and not hasattr(qc, "_impute_cached")
    sources = [*REPO.glob("gwas/*.py"), *REPO.glob("pages/**/*.py"), *REPO.glob("utils/*.py"),
               REPO / "annotation.py", REPO / "app.py"]
    call = re.compile(r"cache_data\([^)]*persist", re.S)
    assert not [f.name for f in sources if call.search(f.read_text(encoding="utf-8"))]
    assert all(len(k) == 32 and int(k, 16) >= 0 for k in guitable.TRACE_MEMO_FUNCTION_KEYS)
