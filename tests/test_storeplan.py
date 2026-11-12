"""The check before a run reads the VCF (gwas/storeplan.py): the memory estimate covers every
measured in-memory peak, the VCF's records are bounded or counted correctly, and the free
memory and disk space are read as documented.
"""
import gzip
import io
import struct
import zlib

import numpy as np
import pytest

from gwas import storeplan as sp

# Measured peaks of in-memory runs (MiB): scaling audit, section 7.2, and its per-stage files
# audit11/evidence/big/*_memory_*_stages.csv (OS peak working set of the pipeline process).
# (samples, SNPs, models, peak, kind): "done" is a completed run's peak; "stopped" is a lower
# bound (the run was stopped by the memory guard; the true peak is higher).
MEASURED = [
    (164, 43_749, ("mlm",), 995.64, "done"),                       # tomato, default
    (164, 43_749, ("mlm", "mlmm", "farmcpu"), 1356.91, "done"),    # tomato, three models
    (500, 50_000, ("mlm",), 1689.25, "done"),                      # simulated 500 x 50,000
    (500, 50_000, ("mlm", "mlmm", "farmcpu"), 1947.00, "done"),
    (1_000, 75_000, ("mlm",), 3783.57, "done"),                    # largest size completed today
    (1_000, 100_000, ("mlm",), 4646.0, "stopped"),                 # stopped in kinship
    (1_000, 100_000, ("mlm", "mlmm", "farmcpu"), 3988.0, "stopped"),
    (1_000, 150_000, ("mlm",), 5292.08, "stopped"),                # SNP QC, exact, then stopped
]
# Batch 12 calibration runs (batch12/evidence/calib/*_stagelog.csv and the runner's process-tree
# peaks in batch12/logs/runlog.jsonl): (samples, SNPs, options, peak MiB, kind)
CALIBRATION = [
    (164, 43_749, dict(), 937.0, "done"),
    (164, 43_749, dict(models=("mlm", "mlmm", "farmcpu"), subsampling=True, boot_jobs=1), 1275.0, "done"),
    (164, 43_749, dict(models=("mlm", "mlmm", "farmcpu"), subsampling=True, boot_jobs=4), 2194.0, "done"),
    (164, 43_749, dict(impute="ldknni"), 1786.0, "done"),
    (164, 43_749, dict(export_qc=True), 1140.0, "done"),
    (1_000, 50_000, dict(), 2785.0, "done"),
    (1_000, 50_000, dict(export_qc=True), 3697.0, "done"),
    (1_000, 50_000, dict(subsampling=True, boot_jobs=1), 2772.0, "done"),
    (1_000, 50_000, dict(subsampling=True, boot_jobs=4), 4568.0, "stopped"),
    (1_000, 50_000, dict(models=("mlm", "mlmm", "farmcpu")), 3003.0, "done"),
]


@pytest.mark.parametrize("n,m,opts,peak,kind", CALIBRATION)
def test_estimate_covers_the_calibration_runs(n, m, opts, peak, kind):
    est = sp.estimate_memory_mib(n, m, **opts)
    assert est >= 1.1 * peak, (est, peak)


@pytest.mark.parametrize("n,m,models,peak,kind", MEASURED)
def test_estimate_covers_every_measured_in_memory_peak(n, m, models, peak, kind):
    est = sp.estimate_memory_mib(n, m, models=models)
    assert est >= 1.1 * peak, (est, peak)


def test_estimate_grows_with_the_options():
    base = sp.estimate_memory_mib(1000, 100_000)
    assert sp.estimate_memory_mib(1000, 100_000, models=("mlm", "farmcpu")) > base
    assert sp.estimate_memory_mib(1000, 100_000, subsampling=True, boot_jobs=4) > base
    assert sp.estimate_memory_mib(1000, 100_000, subsampling=True, boot_jobs=1) == base
    assert sp.estimate_memory_mib(1000, 100_000, export_qc=True) > base
    assert sp.estimate_memory_mib(1000, 100_000, impute="ldknni") > base
    # the reviewer's panel size (996 x 1,070,795): about 50 GB in memory
    assert sp.estimate_memory_mib(996, 1_070_795) * sp.MIB / 1e9 > 45


# ----------------------------------------------------------------------------- the VCF's size
def _vcf_text(n, m, fmt="GT", newline_at_end=True):
    head = ["##fileformat=VCFv4.2", '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">',
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(f"S{i}" for i in range(n))]
    rng = np.random.default_rng(n + m)
    rows = []
    for j in range(m):
        calls = ["0/0", "0/1", "1/1"]
        g = [calls[k] + (":12:99" if fmt != "GT" else "") for k in rng.integers(0, 3, n)]
        rows.append(f"1\t{100 + j}\t.\tA\tT\t.\tPASS\t.\t{fmt}\t" + "\t".join(g))
    text = "\n".join(head + rows)
    return text + ("\n" if newline_at_end else "")


def _bgzf(data, block=4000):
    """BGZF: gzip members of at most ``block`` input bytes with the BC extra field, then the
    empty end-of-file block."""
    out = io.BytesIO()
    pieces = [data[i:i + block] for i in range(0, len(data), block)] + [b""]
    for p in pieces:
        c = zlib.compressobj(6, zlib.DEFLATED, -15)
        body = c.compress(p) + c.flush()
        bsize = len(body) + 25
        out.write(b"\x1f\x8b\x08\x04" + b"\x00" * 4 + b"\x00\xff" + struct.pack("<H", 6)
                  + b"BC" + struct.pack("<HH", 2, bsize))
        out.write(body)
        out.write(struct.pack("<II", zlib.crc32(p) & 0xFFFFFFFF, len(p)))
    return out.getvalue()


@pytest.mark.parametrize("kind", ["plain", "gzip", "bgzf"])
@pytest.mark.parametrize("fmt", ["GT", "GT:DP:GQ"])
@pytest.mark.parametrize("n,m", [(1, 1), (7, 40), (50, 300), (0, 12)])
def test_vcf_size_counts_or_bounds_the_records(tmp_path, kind, fmt, n, m):
    text = _vcf_text(n, m, fmt=fmt).encode()
    path = tmp_path / ("v.vcf" if kind == "plain" else "v.vcf.gz")
    path.write_bytes(text if kind == "plain" else (gzip.compress(text) if kind == "gzip" else _bgzf(text)))
    quick = sp.vcf_size(path)
    exact = sp.vcf_size(path, exact=True)
    assert quick.n_samples == exact.n_samples == n
    assert exact.exact and exact.n_records == m
    assert quick.n_records >= m                         # never below the true count
    if kind == "gzip":
        assert quick.exact and quick.n_records == m     # gzip that is not BGZF: counted
    else:
        assert not quick.exact
        assert quick.how == ("bound from the file size" if kind == "plain" else "bound from the BGZF blocks")
        if fmt == "GT" and n >= 7:
            assert quick.n_records <= 2.2 * m + 1      # about twice the count for GT only
    assert quick.plain_bytes == (len(text) if kind == "plain" else 0)


def test_count_records_without_a_final_newline(tmp_path):
    p = tmp_path / "v.vcf"
    p.write_text(_vcf_text(3, 5, newline_at_end=False))
    assert sp.count_records(p) == 5


def test_bgzf_size_reads_the_block_trailers(tmp_path):
    data = _vcf_text(20, 200).encode()
    p = tmp_path / "v.vcf.gz"
    p.write_bytes(_bgzf(data))
    assert sp._bgzf_uncompressed_size(p) == len(data)
    q = tmp_path / "plain.gz"
    q.write_bytes(gzip.compress(data))
    assert sp._bgzf_uncompressed_size(q) is None


# ----------------------------------------------------------------------- memory and disk
def test_cgroup_v2_available_memory(tmp_path, monkeypatch):
    root = tmp_path / "cg"
    root.mkdir()
    (root / "memory.max").write_text("2147483648\n")
    (root / "memory.current").write_text("1073741824\n")
    (root / "memory.stat").write_text("anon 500000000\ninactive_file 268435456\nactive_file 1\n")
    real = sp.Path

    def fake(p, *a):
        p = str(p)
        return real(p.replace("/sys/fs/cgroup", str(root))) if p.startswith("/sys/fs/cgroup") else real(p, *a)
    monkeypatch.setattr(sp, "Path", fake)
    assert sp._cgroup_available_bytes() == 2147483648 - (1073741824 - 268435456)
    (root / "memory.max").write_text("max\n")
    assert sp._cgroup_available_bytes() is None


def test_available_memory_is_the_smaller_of_os_and_cgroup(monkeypatch):
    import psutil

    class VM:
        available = 8 * 2 ** 30
    monkeypatch.setattr(psutil, "virtual_memory", lambda: VM())
    monkeypatch.setattr(sp, "_cgroup_available_bytes", lambda: 2 ** 30)
    assert sp.available_memory_bytes() == 2 ** 30
    monkeypatch.setattr(sp, "_cgroup_available_bytes", lambda: None)
    assert sp.available_memory_bytes() == 8 * 2 ** 30


def test_disk_need_and_free_space(tmp_path):
    size = sp.VcfSize(n_samples=1000, n_records=100_000, exact=True, how="counted", plain_bytes=500_000_000)
    assert sp.disk_need_bytes(size) == int((1000 * 100_000 + 500_000_000) * 1.1) + 100 * 10 ** 6
    assert sp.free_disk_bytes(tmp_path / "not" / "yet") > 0


# ----------------------------------------------------------------------------- the decision
def _gt_vcf(tmp_path, n=50, m=400, name="v.vcf"):
    p = tmp_path / name
    p.write_text(_vcf_text(n, m))
    return p


GB = 1e9


def test_auto_keeps_a_small_table_in_memory(tmp_path):
    p = sp.plan(_gt_vcf(tmp_path), "auto", tmp_path, available=16 * GB, free_disk=100 * GB)
    assert p.mode == "memory" and p.stop is None and not p.size.exact      # no counting pass needed
    assert "keeps the genotype table in memory" in p.message
    assert "16.0 GB are free" in p.message                                 # the log line names it
    meta = p.metadata(50, 400)
    assert list(meta) == list(sp.MODE_KEYS)
    assert meta["Genotype table"] == "memory" and meta["Genotype table choice"] == "auto"
    assert meta["Genotype table reason"] == "auto: the in-memory estimate fits in the free memory"
    assert meta["Genotype calls"] == "20,000 (50 samples x 400 SNPs read)"
    assert meta["Memory estimate, in-memory run (GB)"] == round(sp.estimate_memory_mib(50, 400) * sp.MIB / GB, 2)


def test_metadata_has_no_number_that_changes_between_runs(tmp_path):
    """The free memory is not recorded, and the estimate and the call count come from the
    genotypes as read: a run that had to count the records exactly (little free memory) and one
    that went by the bound from the file size (more free memory) record the same."""
    v = _gt_vcf(tmp_path)
    bound = sp.vcf_size(v)
    assert not bound.exact and bound.n_records > 400
    e_bound = sp.estimate_memory_mib(50, bound.n_records) * sp.MIB
    e_exact = sp.estimate_memory_mib(50, 400) * sp.MIB
    counted = sp.plan(v, "auto", tmp_path, available=(e_bound + e_exact) / 2 / sp.AUTO_MEMORY_FRACTION,
                      free_disk=100 * GB)
    by_bound = sp.plan(v, "auto", tmp_path, available=4 * e_bound, free_disk=100 * GB)
    assert counted.mode == by_bound.mode == "memory"
    assert counted.size.exact and not by_bound.size.exact
    assert counted.message != by_bound.message                             # the log line differs
    assert counted.metadata(50, 400) == by_bound.metadata(50, 400)
    for p in (counted, by_bound):
        assert "GB are free" in p.message
        assert not any(ch.isdigit() for ch in p.metadata(50, 400)["Genotype table reason"])


def test_auto_chooses_disk_when_the_estimate_does_not_fit(tmp_path):
    v = _gt_vcf(tmp_path)
    est = sp.estimate_memory_mib(50, 400) * sp.MIB
    p = sp.plan(v, "auto", tmp_path, available=est / sp.AUTO_MEMORY_FRACTION - 1, free_disk=100 * GB)
    assert p.mode == "disk" and p.stop is None
    assert p.size.exact and p.size.n_records == 400                       # counted before choosing disk
    assert "so TRACE keeps the genotype table on disk. Results are the same either way." in p.message
    assert "about 20,000 genotype calls" in p.message
    assert p.metadata(50, 400)["Genotype table reason"] == "auto: the in-memory estimate did not fit in the free memory"
    # at the margin (80% of the free memory) it stays in memory
    p2 = sp.plan(v, "auto", tmp_path, available=est / sp.AUTO_MEMORY_FRACTION + 1, free_disk=100 * GB)
    assert p2.mode == "memory"


def test_forced_memory_stops_only_when_it_will_not_fit(tmp_path):
    v = _gt_vcf(tmp_path)
    est = sp.estimate_memory_mib(50, 400) * sp.MIB
    ok = sp.plan(v, "memory", tmp_path, available=est + 1, free_disk=0)
    assert ok.mode == "memory" and ok.stop is None                         # no margin, no disk check
    assert ok.metadata(50, 400)["Genotype table reason"] == "--geno-store memory: kept in memory as asked"
    bad = sp.plan(v, "memory", tmp_path, available=est - 1, free_disk=100 * GB)
    assert bad.stop and "would need about" in bad.stop and "--geno-store disk" in bad.stop


def test_forced_disk_and_its_stops(tmp_path):
    v = _gt_vcf(tmp_path)
    ok = sp.plan(v, "disk", tmp_path, available=1, free_disk=100 * GB)
    assert ok.mode == "disk" and ok.stop is None and "--geno-store disk" in ok.message
    assert ok.metadata(50, 400)["Genotype table reason"] == "--geno-store disk: kept on disk as asked"
    knn = sp.plan(v, "disk", tmp_path, impute="ldknni", available=1, free_disk=100 * GB)
    assert "not available with --geno-store disk" in knn.stop
    knn_auto = sp.plan(v, "auto", tmp_path, impute="ldknni", available=1, free_disk=100 * GB)
    assert knn_auto.mode == "disk" and "needs the whole genotype table in memory: about" in knn_auto.stop
    exp = sp.plan(v, "disk", tmp_path, export_qc=True, available=1, free_disk=100 * GB, one_piece_cells=19_999)
    assert "--export-qc" in exp.stop
    exp_ok = sp.plan(v, "disk", tmp_path, export_qc=True, available=1, free_disk=100 * GB, one_piece_cells=20_000)
    assert exp_ok.stop is None
    need = sp.disk_need_bytes(sp.vcf_size(v, exact=True))
    short = sp.plan(v, "disk", tmp_path, available=1, free_disk=need - 1)
    assert short.stop and "--geno-store-dir" in short.stop and short.size.exact
    assert sp.plan(v, "disk", tmp_path, available=1, free_disk=need).stop is None


def test_unreadable_vcf_leaves_the_message_to_the_reader(tmp_path):
    bad = tmp_path / "bad.vcf.gz"
    bad.write_bytes(b"\x1f\x8b not really gzip")
    p = sp.plan(bad, "auto", tmp_path, available=16 * GB, free_disk=100 * GB)
    assert p.mode == "memory" and p.stop is None
    assert p.reason == "auto: the size of the VCF could not be read before the run"


def test_stale_store_folders_are_named_not_deleted(tmp_path):
    old = tmp_path / "_trace_geno_store_1_deadbeef"
    old.mkdir()
    (old / "calls.i8").write_bytes(b"x" * 1000)
    own = tmp_path / "_trace_geno_store_2_cafe0000"
    own.mkdir()
    assert sp.stale_store_folders(tmp_path, own=own) == [(old, 1000)]
    assert old.exists()


def test_compare_script_ignores_exactly_the_keys_that_name_the_mode():
    """The three keys that differ between the modes; the estimate and the call count, which are
    the same in both, are compared."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "compare_result_zips", Path(__file__).resolve().parents[1] / "scripts" / "compare_result_zips.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert tuple(mod.WHERE_KEYS) == sp.WHERE_KEYS == sp.MODE_KEYS[:3]
