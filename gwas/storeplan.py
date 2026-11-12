"""Where the genotype table is kept: the estimate and the check before a run reads the VCF.

``--geno-store auto`` (the default) keeps the table in memory when an in-memory run fits in the
free memory with a margin, and otherwise on disk (gwas/genostore.py); results are the same
either way. ``memory`` and ``disk`` force a mode. Before anything is read, this module estimates
the peak memory of an in-memory run from the number of samples, the number of SNPs and the
chosen options, compares it with the free memory, and checks the free disk space for the table.

The estimate (estimate_memory_mib) is a linear envelope of measured peaks of in-memory runs
(the process tree's working set; scaling audit, section 7.2, and batch 12, CC-42):

  peak (MiB) = fixed part + bytes per genotype call x samples x SNPs + option terms + n x n

* bytes per call: 50. The table is copied several times in memory: SNP QC reaches 35.5 bytes
  per call (3,599 MiB at 1,000 x 100,000 and 5,292 MiB at 1,000 x 150,000, exact OS peaks);
  kinship 49.8 bytes per call above a 213 MiB process baseline at 1,000 x 75,000 (3,773 MiB);
  the results ZIP with its figures 43.9 bytes per call between 500 x 50,000 and 1,000 x 75,000.
* fixed part: 800 MiB with MLM alone (the figures rendered into the ZIP set the peak of small
  runs: 996 MiB on tomato, 164 x 43,749), 1,200 MiB with MLMM or FarmCPU (1,357 MiB on tomato).
* --boot-jobs N > 1 with --subsampling: 300 MiB and 12 bytes per call for each worker after the
  first (each is a Python process with its own libraries and its share of the table): tomato
  with three models, 1,275 MiB with one worker and 2,194 MiB with four; 1,000 x 50,000, 2,772
  MiB with one worker and more than 4,568 MiB with four (stopped by the guard).
* --export-qc: 24 bytes per call (the whole table twice as text inside the ZIP): 1,137 against
  933 MiB on tomato, 3,697 against 2,781 MiB at 1,000 x 50,000.
* --impute ldknni: 16 bytes per call and 24 KiB per SNP (it ranks the SNPs 512 at a time against
  all others, with about ten 512 x SNPs float32 arrays): 1,782 against 933 MiB on tomato.
* n x n: the kinship matrix, one LOCO kernel per chromosome and FaST-LMM's working matrices,
  float64 (16 matrices assumed).

Every measured peak is at most the estimate divided by 1.1 (a margin of at least 10%); the test
tests/test_storeplan.py holds the measured values.
"""
import gzip
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

MIB = 2 ** 20
GB = 1e9

BYTES_PER_CALL = 50.0
FIXED_MIB_MLM = 800.0
FIXED_MIB_MULTI = 1200.0
NN_MATRICES = 16
# measured options (batch 12, CC-42): extra bytes per call
BYTES_PER_CALL_PER_EXTRA_WORKER = 12.0    # --subsampling with --boot-jobs N > 1, per worker after the first
FIXED_MIB_PER_EXTRA_WORKER = 300.0        # each extra worker is a Python process with its own libraries
BYTES_PER_CALL_EXPORT_QC = 24.0           # --export-qc: two tables as CSV text inside the ZIP
BYTES_PER_CALL_LDKNNI = 16.0              # --impute ldknni: float32 copies of the table
BYTES_PER_SNP_LDKNNI = 24 * 1024          # --impute ldknni: its blocks of 512 SNPs against all SNPs

AUTO_MEMORY_FRACTION = 0.8     # auto keeps the table in memory up to 80% of the free memory
DISK_MARGIN = 1.1              # the table file plus 10%, plus DISK_EXTRA_BYTES
DISK_EXTRA_BYTES = 100 * 10 ** 6
ONE_RECORD_MIN_BYTES = 18      # a VCF data line: 9 fields of at least 1 byte, 9 tabs
ONE_SAMPLE_MIN_BYTES = 2       # a sample field of at least 1 byte and its tab or newline


# ----------------------------------------------------------------------------- the VCF's size
@dataclass
class VcfSize:
    n_samples: int
    n_records: int          # data lines (records), at least the number of biallelic SNPs
    exact: bool             # True: counted; False: an upper bound from the file size
    how: str                # "counted", "bound from the file size", "bound from the BGZF blocks"
    plain_bytes: int        # bytes of a plain-text VCF (its cleaned copy needs as much), else 0


def _open_text(path):
    with open(path, "rb") as fh:
        gz = fh.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rb") if gz else open(path, "rb"), gz


def _header(path):
    """(number of samples, uncompressed bytes up to the end of the #CHROM line, gz)."""
    fh, gz = _open_text(path)
    with fh:
        n_bytes = 0
        for line in fh:
            n_bytes += len(line)
            if line.startswith(b"#CHROM"):
                fields = line.rstrip(b"\r\n").split(b"\t")
                return sum(1 for f in fields[9:] if f.strip()), n_bytes, gz
            if not line.startswith(b"#"):
                break
    return 0, n_bytes, gz


def _bgzf_uncompressed_size(path):
    """The uncompressed size of a BGZF file, from its blocks' trailers (no decompression);
    None if the file is not BGZF."""
    total = 0
    with open(path, "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        pos = 0
        while pos < size:
            fh.seek(pos)
            head = fh.read(18)
            if len(head) < 18 or head[:4] != b"\x1f\x8b\x08\x04" or head[12:14] != b"BC":
                return None
            bsize = struct.unpack("<H", head[16:18])[0] + 1
            fh.seek(pos + bsize - 4)
            total += struct.unpack("<I", fh.read(4))[0]
            pos += bsize
    return total


def count_records(path):
    """The number of data lines of a VCF, counted in one streaming pass."""
    fh, _gz = _open_text(path)
    lines = header = 0
    last = b"\n"
    in_header = True
    with fh:
        while True:
            chunk = fh.read(8 * MIB)
            if not chunk:
                break
            if in_header:
                # header lines start with '#'; count them while they last
                for line in chunk.split(b"\n")[:-1]:
                    if line.startswith(b"#"):
                        header += 1
                    else:
                        in_header = False
                        break
            lines += chunk.count(b"\n")
            last = chunk[-1:]
    if last != b"\n":
        lines += 1                     # a last line without its newline
    return max(0, lines - header)


def vcf_size(path, exact=False):
    """The VCF's samples and records, without reading the genotypes. The record count is an
    upper bound from the file size unless ``exact`` (or the file is gzip but not BGZF), when it
    is counted. A record is at least 18 bytes plus 2 per sample, so the bound is about twice the
    true count for files with GT only and more for files with other FORMAT fields."""
    path = Path(path)
    n, head_bytes, gz = _header(path)
    plain_bytes = 0 if gz else path.stat().st_size
    if exact:
        return VcfSize(n, count_records(path), True, "counted", plain_bytes)
    per_record = ONE_RECORD_MIN_BYTES + ONE_SAMPLE_MIN_BYTES * n
    if not gz:
        data = max(0, path.stat().st_size - head_bytes)
        return VcfSize(n, data // per_record + 1, False, "bound from the file size", plain_bytes)
    total = _bgzf_uncompressed_size(path)
    if total is not None:
        data = max(0, total - head_bytes)
        return VcfSize(n, data // per_record + 1, False, "bound from the BGZF blocks", plain_bytes)
    return VcfSize(n, count_records(path), True, "counted", plain_bytes)


# ----------------------------------------------------------------------------- the estimate
def estimate_memory_mib(n_samples, n_snps, models=("mlm",), subsampling=False, boot_jobs=1,
                        export_qc=False, impute="mean"):
    """Peak memory (MiB) of an in-memory run (see the module docstring for the basis)."""
    calls = float(n_samples) * float(n_snps)
    multi = any(m in ("mlmm", "farmcpu") for m in (models or ()))
    fixed = FIXED_MIB_MULTI if multi else FIXED_MIB_MLM
    per_call = BYTES_PER_CALL
    if subsampling and boot_jobs is not None:
        workers = (os.cpu_count() or 1) if int(boot_jobs) < 0 else int(boot_jobs)
        per_call += BYTES_PER_CALL_PER_EXTRA_WORKER * max(0, workers - 1)
        fixed += FIXED_MIB_PER_EXTRA_WORKER * max(0, workers - 1)
    if export_qc:
        per_call += BYTES_PER_CALL_EXPORT_QC
    per_snp = 0.0
    if impute == "ldknni":
        per_call += BYTES_PER_CALL_LDKNNI
        per_snp += BYTES_PER_SNP_LDKNNI
    nn = NN_MATRICES * 8.0 * float(n_samples) ** 2
    return fixed + (per_call * calls + per_snp * float(n_snps) + nn) / MIB


# ----------------------------------------------------------------------------- free memory
def _read_int(path):
    try:
        v = Path(path).read_text().strip()
    except OSError:
        return None
    if v in ("", "max"):
        return None
    try:
        return int(v)
    except ValueError:
        return None


def _cgroup_available_bytes():
    """In a Linux container: the cgroup's limit minus what it holds that cannot be reclaimed
    (usage minus inactive file pages, which the kernel can drop). None outside a limit."""
    base = Path("/sys/fs/cgroup")
    limit = _read_int(base / "memory.max")                                   # cgroup v2
    if limit is not None:
        usage = _read_int(base / "memory.current") or 0
        stat = base / "memory.stat"
    else:
        limit = _read_int(base / "memory" / "memory.limit_in_bytes")          # cgroup v1
        if limit is None or limit >= 2 ** 60:
            return None
        usage = _read_int(base / "memory" / "memory.usage_in_bytes") or 0
        stat = base / "memory" / "memory.stat"
    inactive = 0
    try:
        for line in stat.read_text().splitlines():
            k, _, v = line.partition(" ")
            if k in ("inactive_file", "total_inactive_file"):
                inactive = int(v)
                break
    except (OSError, ValueError):
        pass
    return max(0, limit - max(0, usage - inactive))


def available_memory_bytes():
    """Memory free for a new run: the operating system's available memory, and inside a
    container with a memory limit the smaller of that and what the limit leaves."""
    import psutil

    avail = int(psutil.virtual_memory().available)
    cg = _cgroup_available_bytes()
    return min(avail, cg) if cg is not None else avail


def free_disk_bytes(folder):
    """Free space on the drive of ``folder`` (or of its nearest existing parent)."""
    p = Path(folder).resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    return int(shutil.disk_usage(str(p)).free)


def disk_need_bytes(size: VcfSize):
    """Bytes the table on disk needs: one per call, plus the cleaned copy of a plain-text VCF
    (the ingest writes it next to the table and removes it after reading), with a margin."""
    return int((size.n_samples * size.n_records + size.plain_bytes) * DISK_MARGIN) + DISK_EXTRA_BYTES


# ----------------------------------------------------------------------------- the decision
MODE_KEYS = ("Genotype table", "Genotype table choice", "Genotype table reason",
             "Memory estimate, in-memory run (GB)", "Genotype calls")
# where the table was kept and why: the only keys that differ between the two modes
WHERE_KEYS = MODE_KEYS[:3]

REASON_AUTO_MEMORY = "auto: the in-memory estimate fits in the free memory"
REASON_AUTO_DISK = "auto: the in-memory estimate did not fit in the free memory"
REASON_AUTO_NOT_READ = "auto: the size of the VCF could not be read before the run"


@dataclass
class Plan:
    mode: str               # "memory" or "disk"
    choice: str             # "auto", "memory" or "disk" (what --geno-store asked for)
    message: str            # one plain line for the log (it names the free memory)
    stop: object            # None, or the message of a stop before anything is read
    estimate_mib: float     # the estimate the choice was made on (from the file, before reading)
    available_bytes: int
    size: VcfSize
    reason: str = ""        # why, with no number in it (run_metadata.json)
    options: dict = None    # the options of the estimate (estimate_memory_mib's keywords)

    def metadata(self, n_samples, n_snps):
        """The keys run_metadata.json gets (and only it: the report is the same either way).

        None of them holds a number that changes between two runs of the same analysis, so two
        such runs give identical ZIPs: the free memory is in the log line only, and the estimate
        and the call count are those of the genotypes as read (``n_samples`` x ``n_snps``), not
        the plan's, which is an exact count in one run and a bound from the file size in another
        depending on the free memory."""
        n, m = int(n_samples), int(n_snps)
        est = estimate_memory_mib(n, m, **(self.options or {}))
        return {
            "Genotype table": self.mode,
            "Genotype table choice": "auto" if self.choice == "auto" else f"--geno-store {self.choice}",
            "Genotype table reason": self.reason,
            "Memory estimate, in-memory run (GB)": round(est * MIB / GB, 2),
            "Genotype calls": f"{n * m:,} ({n:,} samples x {m:,} SNPs read)",
        }


def _calls(size):
    return int(size.n_samples) * int(size.n_records)


def _calls_text(size):
    # a record count includes multi-allelic records, so it is "about"; a bound is "up to"
    return f"{'about' if size.exact else 'up to'} {_calls(size):,}"


def plan(vcf_path, choice="auto", store_base=".", models=("mlm",), subsampling=False, boot_jobs=1,
         export_qc=False, impute="mean", available=None, free_disk=None, one_piece_cells=None):
    """Where the table is kept for this run, decided before the VCF is read.

    auto: in memory when the estimate is at most AUTO_MEMORY_FRACTION of the free memory, else
    on disk. memory: in memory; the run stops if the estimate is above the free memory. disk: on
    disk. With the table on disk the run stops for --impute ldknni, for --export-qc above the
    one-piece size, and when the drive of the store folder has too little free space. When the
    record count is only a bound, it is counted exactly before any choice of disk or any stop.
    ``available`` and ``free_disk`` (bytes) replace the measured values (tests)."""
    if one_piece_cells is None:
        from gwas.storepipe import one_piece_cells as _opc
        one_piece_cells = _opc()
    try:
        size = vcf_size(vcf_path)
    except Exception:          # unreadable here: the reader then gives the usual message
        size = VcfSize(0, 0, True, "not read", 0)
    avail = available_memory_bytes() if available is None else int(available)
    options = dict(models=tuple(models or ()), subsampling=subsampling, boot_jobs=boot_jobs,
                   export_qc=export_qc, impute=impute)

    def est(sz):
        return estimate_memory_mib(sz.n_samples, sz.n_records, **options)

    def refine():
        nonlocal size
        if not size.exact:
            try:
                size = vcf_size(vcf_path, exact=True)
            except Exception:
                pass
        return est(size)

    e = est(size)
    stop = None
    if choice == "auto":
        if e * MIB > AUTO_MEMORY_FRACTION * avail:
            e = refine()
        mode = "memory" if e * MIB <= AUTO_MEMORY_FRACTION * avail else "disk"
    elif choice == "memory":
        mode = "memory"
        if e * MIB > avail:
            e = refine()
        if e * MIB > avail:
            stop = (f"An in-memory run of this dataset ({_calls_text(size)} genotype calls) would need "
                    f"about {e * MIB / GB:.1f} GB of memory, and {avail / GB:.1f} GB are free. "
                    "Use --geno-store disk (the same results with far less memory) or --geno-store auto.")
    elif choice == "disk":
        mode = "disk"
    else:
        raise ValueError(f"unknown --geno-store {choice!r}")

    e_gb, a_gb = e * MIB / GB, avail / GB
    if choice == "auto":
        if mode == "disk":
            message = (f"This dataset has {_calls_text(size)} genotype calls; an in-memory run would need "
                       f"about {e_gb:.1f} GB, and {a_gb:.1f} GB are free, so TRACE keeps the genotype "
                       "table on disk. Results are the same either way.")
        else:
            message = (f"This dataset has {_calls_text(size)} genotype calls; an in-memory run needs "
                       f"about {e_gb:.1f} GB, and {a_gb:.1f} GB are free, so TRACE keeps the genotype "
                       "table in memory.")
    elif choice == "memory":
        message = (f"The genotype table is kept in memory (--geno-store memory); an in-memory run needs "
                   f"about {e_gb:.1f} GB, and {a_gb:.1f} GB are free.")
    else:
        message = ("The genotype table is kept on disk (--geno-store disk); results are the same as "
                   "in memory.")
    if choice != "auto":
        reason = f"--geno-store {choice}: kept {'in memory' if mode == 'memory' else 'on disk'} as asked"
    elif size.how == "not read":
        reason = REASON_AUTO_NOT_READ
    else:
        reason = REASON_AUTO_MEMORY if mode == "memory" else REASON_AUTO_DISK

    if mode == "disk" and stop is None:
        if impute == "ldknni":
            if choice == "auto":
                stop = (f"--impute ldknni ranks every SNP against all others and needs the whole genotype "
                        f"table in memory: about {e_gb:.1f} GB for this dataset, and {a_gb:.1f} GB are "
                        "free. Free some memory, or use --impute mean (the default), which works with "
                        "the table on disk.")
            else:
                stop = ("--impute ldknni ranks every SNP against all others and needs the whole genotype "
                        "table in memory; it is not available with --geno-store disk.")
        elif export_qc and _calls(size) > one_piece_cells:
            refine()
            if _calls(size) > one_piece_cells:
                stop = (f"--export-qc writes the whole genotype table as text ({_calls_text(size)} calls) "
                        f"and is not available above {one_piece_cells:,} calls with the table on disk; "
                        "this dataset does not fit in memory for it.")
        if stop is None:
            free = free_disk_bytes(store_base) if free_disk is None else int(free_disk)
            if disk_need_bytes(size) > free:
                refine()
            need = disk_need_bytes(size)
            if need > free:
                stop = (f"The genotype table on disk needs about {need / GB:.1f} GB in {store_base}, and "
                        f"{free / GB:.1f} GB are free there. Free some space, or choose a folder on "
                        "another drive with --geno-store-dir.")
    return Plan(mode=mode, choice=choice, message=message, stop=stop, estimate_mib=e,
                available_bytes=avail, size=size, reason=reason, options=options)


def stale_store_folders(base, own=None):
    """Folders of the table on disk left in ``base`` by earlier runs that did not finish (killed),
    with their sizes in bytes. They are named, never deleted: another run may still be using one."""
    out = []
    for d in sorted(Path(base).glob("_trace_geno_store_*")):
        if own is not None and Path(d).resolve() == Path(own).resolve():
            continue
        if d.is_dir():
            out.append((d, sum(f.stat().st_size for f in d.rglob("*") if f.is_file())))
    return out
