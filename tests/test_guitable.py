"""The GUI's genotype file and table (gwas/guitable.py), at function level: the location box's
check, the saved upload, the file identity, the environment's defaults, storeplan's messages in
the page's words, the registry of table files (naming, at most two, removal) and read-only
shared arrays. Also: nothing in the pages or the shared code keeps VCF bytes in the session."""
import builtins
import io
import os
import re
from pathlib import Path

import numpy as np
import pytest

from gwas import guitable as gt
from gwas import storeplan as sp
from gwas.utils import hash_bytes

REPO = Path(__file__).resolve().parents[1]
EX1 = REPO / "tests" / "data" / "example_v1" / "example.vcf.gz"
GB = 1e9


# ------------------------------------------------------------------- the location box
def test_location_check_accepts_a_readable_vcf(tmp_path):
    p = tmp_path / "panel.vcf.gz"
    p.write_bytes(b"\x1f\x8b" + b"x" * 1000)
    ok = gt.check_vcf_location(f'  "{p}"  ')               # Windows' "Copy as path" quotes
    assert ok.ok and ok.path == p.resolve() and ok.size == 1002
    assert "panel.vcf.gz" in ok.message
    plain = tmp_path / "panel.VCF"
    plain.write_text("##fileformat=VCFv4.2\n")
    assert gt.check_vcf_location(str(plain)).ok              # the extension in any case


@pytest.mark.parametrize("name", ["panel.csv", "panel.vcf.zip", "panel.bcf", "panel"])
def test_location_check_refuses_another_extension(tmp_path, name):
    p = tmp_path / name
    p.write_text("x")
    r = gt.check_vcf_location(str(p))
    assert not r.ok and "TRACE reads .vcf and .vcf.gz files" in r.message


def test_location_check_refuses_a_missing_file_a_folder_and_an_empty_box(tmp_path):
    missing = gt.check_vcf_location(str(tmp_path / "nope.vcf"))
    assert not missing.ok and missing.message.startswith("No file at")
    folder = tmp_path / "dir.vcf"
    folder.mkdir()
    assert "is a folder" in gt.check_vcf_location(str(folder)).message
    assert not gt.check_vcf_location("   ").ok


def test_location_check_refuses_an_unreadable_file(tmp_path, monkeypatch):
    p = tmp_path / "locked.vcf"
    p.write_text("x")
    real_open = builtins.open

    def deny(path, *a, **k):
        if Path(path) == p:
            raise PermissionError(13, "Permission denied")
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", deny)
    r = gt.check_vcf_location(str(p))
    assert not r.ok and "cannot read" in r.message and "Permission denied" in r.message


# --------------------------------------------------------- the saved upload and identity
class _Upload(io.BytesIO):
    def __init__(self, data, name):
        super().__init__(data)
        self.name, self.size, self.file_id = name, len(data), "f1"


def test_saved_upload_equals_the_upload_and_its_identity_is_hash_bytes(tmp_path):
    data = os.urandom(3 * gt.HASH_PIECE + 12345)             # several pieces
    up = _Upload(data, "my panel.vcf.gz")
    up.read(10)                                              # a read position left anywhere
    path, ident = gt.save_upload(up, tmp_path / "work")
    try:
        assert path.read_bytes() == data
        assert path.name == "my panel.vcf.gz"
        assert path.parent.name.startswith(f"{gt.UPLOAD_PREFIX}_{os.getpid()}_")
        assert ident == hash_bytes(data) == gt.file_identity(path)
        assert path.parent in gt.saved_upload_folders()
    finally:
        gt.remove_saved_uploads()
    assert not path.exists() and gt.saved_upload_folders() == []


def test_identity_is_stable_and_one_changed_byte_gives_a_new_one(tmp_path):
    p = tmp_path / "v.vcf"
    data = bytearray(os.urandom(100_000))
    p.write_bytes(bytes(data))
    a = gt.file_identity(p)
    assert gt.file_identity(p) == a == hash_bytes(bytes(data))
    data[54_321] ^= 0x01
    p.write_bytes(bytes(data))
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))   # a new modification time
    b = gt.file_identity(p)
    assert b != a and b == hash_bytes(bytes(data))


def test_table_identity_names_every_setting():
    base = gt.table_identity("f", maf=0.05, impute="mean", trait="T")
    assert gt.table_identity("f", trait="T", impute="mean", maf=0.05) == base     # order-free
    assert gt.table_identity("g", maf=0.05, impute="mean", trait="T") != base
    assert gt.table_identity("f", maf=0.05, impute="ldknni", trait="T") != base
    assert gt.table_identity("f", maf=0.05, impute="mean", trait="T", covar="c1") != base
    st = gt.SessionTable(identity=base, file_identity="f", mode="memory", view=np.zeros((3, 4)))
    assert st.shape == (3, 4) and st.rows is None and st.fill is None and st.raw is None


def test_upload_token_names_the_upload():
    up = _Upload(b"abc", "x.vcf")
    assert gt.new_upload_token(up) == ("x.vcf", 3, "f1")


# ---------------------------------------------------------------- the environment
def test_environment_defaults(tmp_path):
    assert gt.default_choice({}) == "auto"
    assert gt.default_choice({"TRACE_GENO_STORE": "DISK"}) == "disk"
    assert gt.default_choice({"TRACE_GENO_STORE": "memory"}) == "memory"
    assert gt.default_choice({"TRACE_GENO_STORE": "somewhere"}) == "auto"
    assert gt.default_work_dir({"TRACE_GENO_STORE_DIR": str(tmp_path)}) == tmp_path
    assert gt.default_work_dir({}).name == "trace_gui"
    assert gt.path_input_disabled({"TRACE_DISABLE_PATH_INPUT": "1"})
    assert not gt.path_input_disabled({"TRACE_DISABLE_PATH_INPUT": "0"})
    assert not gt.path_input_disabled({})


# ------------------------------------------------------- the check before reading
def _gt_vcf(tmp_path, n=50, m=400):
    rng = np.random.default_rng(0)
    head = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + \
        "\t".join(f"S{i}" for i in range(n)) + "\n"
    rows = [f"1\t{j + 1}\t.\tA\tT\t.\tPASS\t.\tGT\t" + "\t".join(rng.choice(["0/0", "0/1", "1/1"], n))
            for j in range(m)]
    p = tmp_path / "v.vcf"
    p.write_text(head + "\n".join(rows) + "\n")
    return p


def test_the_page_choice_goes_to_disk_when_free_memory_is_lowered(tmp_path):
    v = _gt_vcf(tmp_path)
    choice = gt.default_choice({"TRACE_GENO_STORE": "auto"})
    est = sp.estimate_memory_mib(50, 400, models=("mlm", "mlmm", "farmcpu")) * sp.MIB
    roomy = sp.plan(v, choice, tmp_path, models=("mlm", "mlmm", "farmcpu"),
                    available=4 * est, free_disk=100 * GB)
    tight = sp.plan(v, choice, tmp_path, models=("mlm", "mlmm", "farmcpu"),
                    available=est / sp.AUTO_MEMORY_FRACTION - 1, free_disk=100 * GB)
    assert roomy.mode == "memory" and tight.mode == "disk"
    for p in (roomy, tight):
        assert "--" not in gt.page_message(p)


def test_page_messages_and_stops_name_no_command_line_flag(tmp_path):
    v = _gt_vcf(tmp_path)
    est = sp.estimate_memory_mib(50, 400) * sp.MIB
    forced = [sp.plan(v, "memory", tmp_path, available=16 * GB, free_disk=100 * GB),
              sp.plan(v, "disk", tmp_path, available=16 * GB, free_disk=100 * GB)]
    for p in forced:
        msg = gt.page_message(p)
        assert "--geno-store" not in msg and "as chosen under Large datasets" in msg
    need = sp.disk_need_bytes(sp.vcf_size(v, exact=True))
    stops = {
        "memory": sp.plan(v, "memory", tmp_path, available=est - 1, free_disk=100 * GB),
        "disk": sp.plan(v, "disk", tmp_path, available=1, free_disk=need - 1),
        "ldknni": sp.plan(v, "auto", tmp_path, impute="ldknni", available=1, free_disk=100 * GB),
        "ldknni_forced": sp.plan(v, "disk", tmp_path, impute="ldknni", available=1, free_disk=100 * GB),
    }
    for name, p in stops.items():
        assert p.stop, name
        text = gt.page_stop(p, tmp_path)
        assert text and "--" not in text, (name, text)
    assert "Choose On disk under Large datasets" in gt.page_stop(stops["memory"], tmp_path)
    assert "work folder on another drive" in gt.page_stop(stops["disk"], tmp_path)
    assert "Choose mean imputation" in gt.page_stop(stops["ldknni"], tmp_path)


# ---------------------------------------------------------------- the table registry
def test_registry_names_keeps_two_and_removes_folders(tmp_path):
    reg = gt.TableRegistry()
    work = tmp_path / "work"
    a = reg.get("id-a", "disk", EX1, work)
    assert reg.get("id-a", "disk", EX1, work) is a           # the same entry, not a new table
    (fa,) = reg.folders()
    assert fa.parent == work and re.match(rf"{gt.GUI_STORE_PREFIX}_{os.getpid()}_[0-9a-f]{{8}}$", fa.name)
    assert (fa / "calls.i8").exists()
    m = reg.get("id-a", "memory", EX1)
    assert reg.keys() == [("id-a", "disk"), ("id-a", "memory")]
    assert m.genotypes.shape == (len(a.samples), a.store.M)
    b = reg.get("id-b", "disk", EX1, work)                    # a third: the oldest goes
    assert reg.keys() == [("id-a", "memory"), ("id-b", "disk")]
    assert not fa.exists()                                   # its folder is deleted ...
    with pytest.raises(ValueError, match="closed"):          # ... after its table was closed
        a.store.gather([0])
    (fb,) = reg.folders()
    reg.close_all()
    assert reg.keys() == [] and not fb.exists()
    assert b.store._fh is None


def test_registry_memory_entry_is_read_only(tmp_path):
    reg = gt.TableRegistry()
    m = reg.get("id-m", "memory", EX1)
    for name in ("genotypes", "positions", "chroms", "samples"):
        arr = getattr(m, name)
        assert not arr.flags.writeable, name
    with pytest.raises(ValueError):
        m.genotypes[0, 0] = 1.0
    reg.close_all()


def test_registry_cleans_up_a_failed_build(tmp_path):
    bad = tmp_path / "bad.vcf"
    bad.write_text("not a vcf at all\n")
    reg = gt.TableRegistry()
    with pytest.raises(Exception):
        reg.get("id-bad", "disk", bad, tmp_path / "work")
    assert reg.keys() == []
    assert not list((tmp_path / "work").glob(f"{gt.GUI_STORE_PREFIX}_*"))


def test_get_registry_is_one_per_process():
    assert gt.get_registry() is gt.get_registry()


@pytest.mark.parametrize("mode", ["disk", "memory"])
def test_a_vcf_that_cannot_be_opened_closes_nothing(tmp_path, mode):
    """The new table is built first and the oldest entry evicted only once it is open: a file
    the reader refuses (no #CHROM line; storeplan.plan lets it through) leaves both tables open."""
    bad = tmp_path / "bad.vcf"
    bad.write_text("##fileformat=VCFv4.2\n1\t100\t.\tA\tT\t.\tPASS\t.\tGT\t0/0\n")
    reg = gt.TableRegistry()
    work = tmp_path / "work"
    try:
        a = reg.get("id-a", "disk", EX1, work)
        b = reg.get("id-b", "disk", EX1, work)
        with pytest.raises(Exception):
            reg.get("id-bad", mode, bad, work)
        assert reg.keys() == [("id-a", "disk"), ("id-b", "disk")]
        assert not a.store.closed and not b.store.closed
        a.store.gather([0])
        assert len(reg.folders()) == 2 and all(f.exists() for f in reg.folders())
    finally:
        reg.close_all()


def test_a_build_holds_neither_the_registry_nor_another_key(tmp_path, monkeypatch):
    """While one session builds a table, another session's look-up of a table already there and
    the folder listing return at once, and the folder being built is listed as this server's."""
    import threading
    import time

    from gwas import genostore

    reg = gt.TableRegistry()
    work = tmp_path / "work"
    ready = reg.get("id-held", "memory", EX1)
    started, release = threading.Event(), threading.Event()
    real_ingest = genostore.ingest_vcf

    def slow_ingest(*a, **k):
        started.set()
        release.wait(30)
        return real_ingest(*a, **k)

    monkeypatch.setattr(genostore, "ingest_vcf", slow_ingest)
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("built", reg.get("id-new", "disk", EX1, work)))
    t.start()
    try:
        assert started.wait(30)
        t0 = time.perf_counter()
        assert reg.get("id-held", "memory", EX1) is ready              # a look-up: no wait
        building = reg.folders()                                       # the listing: no wait
        assert time.perf_counter() - t0 < 1.0
        assert len(building) == 1 and building[0].parent == work and building[0].exists()
    finally:
        release.set()
        t.join(60)
    assert reg.keys() == [("id-held", "memory"), ("id-new", "disk")]
    assert reg.folders() == building
    reg.close_all()


def test_one_build_per_table_when_sessions_ask_together(tmp_path, monkeypatch):
    import threading

    from gwas import genostore

    reg = gt.TableRegistry()
    calls = []
    real_ingest = genostore.ingest_vcf
    gate = threading.Event()

    def counted_ingest(*a, **k):
        calls.append(1)
        gate.wait(30)
        return real_ingest(*a, **k)

    monkeypatch.setattr(genostore, "ingest_vcf", counted_ingest)
    got = []
    ts = [threading.Thread(target=lambda: got.append(reg.get("id-x", "disk", EX1, tmp_path / "w")))
          for _ in range(3)]
    for t in ts:
        t.start()
    gate.set()
    for t in ts:
        t.join(60)
    try:
        assert len(calls) == 1 and len(got) == 3 and got[0] is got[1] is got[2]
        assert len(reg.folders()) == 1
    finally:
        reg.close_all()


# ----------------------------------------------------------- shared kinship arrays
def test_shared_kinship_and_pcs_are_read_only():
    from gwas import kinship

    rng = np.random.default_rng(3)
    n, m = 30, 120
    Z = rng.standard_normal((n, m)).astype(np.float32)
    iid = np.array([[f"S{i}", f"S{i}"] for i in range(n)])
    chroms = np.repeat(np.array(["1", "2", "3"]), m // 3)
    K0, K_by_chr, _ = kinship.build_loco_kernels_cached("shared-kinship-test", _iid=iid, _Z_grm=Z,
                                                        _chroms_grm=chroms)
    for kd in [K0, *K_by_chr.values()]:
        assert not kd.val.flags.writeable
    with pytest.raises(ValueError):
        K0.val[0, 0] = 0.0
    pcs, eig = kinship.compute_pcs_full_cached("shared-kinship-test", _Z_for_pca=Z, max_pcs=3)
    assert not pcs.flags.writeable and not eig.flags.writeable
    with pytest.raises(ValueError):
        pcs[0, 0] = 1.0


# -------------------------------------------------- no VCF bytes in the session
def test_no_vcf_bytes_are_kept_in_the_session():
    """CC-50: the page keeps only the VCF's path; no VCF_BYTES:: key, no _persist_vcf_bytes,
    and nothing reads VCF bytes from st.session_state."""
    sources = list((REPO / "pages").rglob("*.py")) + list((REPO / "gwas").rglob("*.py")) + \
        [REPO / "app.py"] + list((REPO / "utils").rglob("*.py"))
    for f in sources:
        text = f.read_text(encoding="utf-8")
        assert "VCF_BYTES::" not in text, f
        assert "_persist_vcf_bytes" not in text, f
        assert not re.search(r"session_state[^\n]*vcf_bytes", text), f


def test_the_page_plan_counts_a_files_records_once(tmp_path, monkeypatch):
    """The page plans on every rerun. A gzip VCF that is not BGZF has its records counted by reading
    the whole file; with the sizes kept per file identity (vcf_size_of) that happens once, until
    Clear, and the plan is the same."""
    import gzip

    gz = tmp_path / "v.vcf.gz"
    gz.write_bytes(gzip.compress(_gt_vcf(tmp_path).read_bytes()))     # plain gzip: counted
    counted = []
    real = sp.count_records
    monkeypatch.setattr(sp, "count_records", lambda p: counted.append(1) or real(p))
    args = dict(choice="auto", store_base=tmp_path, models=("mlm", "mlmm", "farmcpu"), free_disk=100 * GB)
    for available in (16 * GB, 1):                                     # fits / too big: refined
        counted.clear()
        plain = [sp.plan(gz, available=available, **args) for _ in range(2)]
        assert len(counted) == 2                                       # once per plan
        counted.clear()
        ident = gt.file_identity(gz)
        kept = [sp.plan(gz, available=available, size_of=gt.vcf_size_of(ident), **args) for _ in range(3)]
        assert len(counted) == 1                                       # once per file identity
        for p in kept:
            assert (p.mode, p.message, p.estimate_mib, p.size, p.reason) == \
                (plain[0].mode, plain[0].message, plain[0].estimate_mib, plain[0].size, plain[0].reason)
        gt.forget_file_identities()                                    # Clear: counted again
    plain_vcf = _gt_vcf(tmp_path)
    counted.clear()
    for _ in range(3):                                                 # a bound, then an exact count
        sp.plan(plain_vcf, available=1, size_of=gt.vcf_size_of(gt.file_identity(plain_vcf)), **args)
    assert len(counted) == 1


# --------------------------------------------------------------- the GUI's memory estimate
# The exclusive measurements of 16 Nov (batch 13, track A evidence gui_peak_X_*.json; the lead's table):
# the GUI's in-memory peak as the job runner's process-tree working set, 8 settings, every cache full,
# MLM + MLMM + FarmCPU, subsampling with one worker, SVG exports through kaleido's Chrome.
# (name, samples, SNPs read, CLI estimate MiB, measured peak MiB, GUI estimate MiB)
GUI_PEAKS = [
    ("example_v1", 50, 510, 1201.5, 1493, 2554),
    ("example_v2", 100, 600, 1204.1, 1503, 2561),
    ("tomato", 166, 44_317, 1554.2, 2633, 3078),
    ("1000 x 20,000", 1000, 20_000, 2275.7, 2788, 4314),
    ("1000 x 50,000", 1000, 50_000, 3706.3, 4445, 6043),
]


@pytest.mark.parametrize("name,n,m,cli,peak,gui", GUI_PEAKS)
def test_the_gui_estimate_lies_above_every_measured_gui_peak(name, n, m, cli, peak, gui):
    opts = dict(models=("mlm", "mlmm", "farmcpu"), subsampling=True, boot_jobs=1)
    assert round(sp.estimate_memory_mib(n, m, **opts), 1) == cli, name
    est = gt.gui_estimate_mib(n, m, **opts)
    assert abs(est - gui) < 1, name                       # the table rounds to whole MiB
    assert peak <= est / 1.14, name                       # batch 12's margin
    # the rule's parts: the CLI's estimate, 600 MiB, 8 B per call, 512 B per sample pair, 750 MiB
    # and 2.4 KiB per SNP up to 500,000 (SVG export)
    assert est == pytest.approx(cli + 600 + (8 * n * m + 512 * n * n + 2.4 * 1024 * min(m, 500_000)) / 2**20
                                + 750, abs=0.06)


def test_the_page_chooses_by_the_gui_estimate_and_records_the_clis(tmp_path):
    """Automatic keeps the table in memory when the GUI's estimate fits: with free memory between the
    two estimates' thresholds the page's plan goes to disk where the CLI's would not, and the plan's
    metadata (run_metadata.json) still records the CLI's estimate."""
    v = _gt_vcf(tmp_path)
    opts = dict(models=("mlm", "mlmm", "farmcpu"))
    size = sp.vcf_size(v, exact=True)
    cli = sp.estimate_memory_mib(size.n_samples, size.n_records, **opts) * sp.MIB
    gui = gt.gui_estimate_mib(size.n_samples, size.n_records, **opts) * sp.MIB
    assert gui > cli
    between = (cli + gui) / 2 / sp.AUTO_MEMORY_FRACTION
    p_cli = sp.plan(v, "auto", tmp_path, available=between, free_disk=100 * GB, **opts)
    p_gui = sp.plan(v, "auto", tmp_path, available=between, free_disk=100 * GB, estimate=gt.gui_estimate_mib,
                    **opts)
    assert p_cli.mode == "memory" and p_gui.mode == "disk"
    assert p_gui.estimate_mib * sp.MIB == pytest.approx(gui)          # the page's message shows it
    assert p_gui.metadata(50, 400) == {**p_gui.metadata(50, 400),
                                       "Memory estimate, in-memory run (GB)": round(cli / sp.GB, 2)}
    page = (REPO / "pages" / "GWAS_analysis.py").read_text(encoding="utf-8")
    assert page.count("estimate=_gt.gui_estimate_mib") == 2           # the page's plan and One-Click's
