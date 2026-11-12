"""Slow tests: the genotype table in memory and on disk give the same results.

Marker ``slow``: left out of the everyday runs (-m "not slow"); CI runs them once, in a job of
their own. Each run is a fresh ``cli.py`` process with the OPENBLAS, MKL and OMP threads pinned
to TRACE_SLOW_THREADS (default 1): results are byte-identical only at a fixed thread count.

1. test_memory_equals_disk: the 16 cases of the scaling audit's comparison (tests/_store_cases.py;
   the 6 on the planted panel, tomato and pepper need TRACE_B4_DATA), --geno-store memory against
   --geno-store disk: every ZIP member identical apart from the three run_metadata.json keys that
   say where the table was kept and why, every stage digest (gwas/taps.py) identical, and no
   table folder left behind.
2. test_stored_baseline: the memory run's member digests, run_metadata.json whole, against the
   baseline stored for this machine (tests/data/slow_baseline/<fingerprint>.json, made at 1
   thread). The fingerprint names
   the OS, CPU, BLAS libraries, threads and numeric packages; anywhere else the test is skipped
   with a message. TRACE_SLOW_WRITE_BASELINE=1 writes this machine's baseline instead.
3. test_piece_sizes: the digest suite: the same cases on disk read in pieces of 1, 64 and 997
   SNPs (VCF chunks of 97, 997 and 5,000 records): ZIP members and stage digests equal to memory.
4. test_farmcpu_last_chunk: the 7 FarmCPU cases with 1 to 3 columns in the last OLS chunk.
5. test_boot_jobs: subsampling with 7 and 4 workers: in memory the replicates run in parallel
   workers, on disk one after another under a worker's thread counts.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path

import pytest

from tests import _store_cases as sc

BASELINE_DIR = Path(__file__).resolve().parent / "data" / "slow_baseline"
PIECES = {"disk_p1": ("1", "97"), "disk_p64": ("64", "997"), "disk_p997": ("997", "5000")}
CASES = dict(sc.cases())
_RUNS = {}


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    return tmp_path_factory.mktemp("slow_store")


def _run(work, config, name, argv):
    """One run of a case in one configuration ('mem', 'disk' or a piece size), once per session."""
    key = (config, name)
    if key not in _RUNS:
        folder = work / config / name
        folder.mkdir(parents=True)
        env = {"TRACE_TAPS": "1"}
        if config in PIECES:
            env["TRACE_STORE_PIECE"], env["TRACE_STORE_CHUNK_RECORDS"] = PIECES[config]
        mode = "memory" if config == "mem" else "disk"
        p = sc.run_cli([*argv, "--geno-store", mode], folder / "results", env)
        assert p.returncode == 0, f"{name} {config}: exit {p.returncode}\n{p.stderr[-3000:]}"
        _RUNS[key] = folder
    return _RUNS[key]


def _argv(name):
    argv = CASES[name]
    if argv is None:
        pytest.skip("needs TRACE_B4_DATA (the planted panel, tomato and pepper inputs)")
    return argv


def _folders(work):
    return [sc.REPO, sc.DATA, work]


def _same(work, a, b, label):
    ma, mb = sc.zip_members(a / "results", _folders(work)), sc.zip_members(b / "results", _folders(work))
    assert ma.keys() == mb.keys(), (label, sorted(set(ma) ^ set(mb)))
    differ = [k for k in ma if ma[k] != mb[k]]
    assert differ == [], (label, differ)
    ta, tb = sc.stage_digests(a), sc.stage_digests(b)
    assert ta, (label, "no stage digests written")
    assert ta.keys() == tb.keys(), (label, sorted(set(ta) ^ set(tb))[:10])
    assert [k for k in ta if ta[k] != tb[k]] == [], label
    return ma


# --------------------------------------------------------------------- 1 and 2: 16 cases
@pytest.mark.slow
@pytest.mark.parametrize("name", list(CASES))
def test_memory_equals_disk(work, name):
    argv = _argv(name)
    mem, disk = _run(work, "mem", name, argv), _run(work, "disk", name, argv)
    _same(work, mem, disk, name)
    assert not list((disk / "results").glob("_trace_geno_store_*"))


def _baseline_path():
    raw, fp = sc.fingerprint()
    return raw, BASELINE_DIR / f"{fp}.json"


@pytest.mark.slow
@pytest.mark.parametrize("name", list(CASES))
def test_stored_baseline(work, name):
    raw, path = _baseline_path()
    writing = os.environ.get("TRACE_SLOW_WRITE_BASELINE") == "1"
    if not writing and not path.exists():
        pytest.skip(f"no stored baseline for this machine ({path.name}: {raw})")
    argv = _argv(name)
    # the same mode as the stored runs, so nothing in run_metadata.json is left out
    got = sc.digests(sc.zip_members(_run(work, "mem", name, argv) / "results", _folders(work), drop_where=False))
    if writing:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "fingerprint": raw, "threads": sc.threads(), "cases": {}}
        data["cases"][name] = got
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return
    want = json.loads(path.read_text(encoding="utf-8"))["cases"].get(name)
    if want is None:
        pytest.skip(f"{name} is not in the stored baseline {path.name}")
    assert got.keys() == want.keys(), sorted(set(got) ^ set(want))
    assert [k for k in want if got[k] != want[k]] == []


# --------------------------------------------------------------------- 3: the digest suite
@pytest.mark.slow
@pytest.mark.parametrize("config", list(PIECES))
@pytest.mark.parametrize("name", list(CASES))
def test_piece_sizes(work, name, config):
    argv = _argv(name)
    _same(work, _run(work, "mem", name, argv), _run(work, config, name, argv), f"{name} {config}")


# ------------------------------------------------------------ 4 and 5: FarmCPU, --boot-jobs
# content digests of the scaling audit's input files (the VCF uncompressed, the phenotype file
# with LF line ends): the generator ported into tests/_store_cases.py writes the same inputs
AUDIT_INPUTS = {
    "h2001": ("ddd8e4c99fc8f90fbf7d", "dc5e150c79e6b1f3cfaf"),
    "h2001n40": ("16f705164b5f396dbd8d", "1b8e073a986761ffffd8"),
    "h4001": ("34f9d0bc1bfb72f13b5e", "dc5e150c79e6b1f3cfaf"),
    "h2003": ("ce01a213a213c2867976", "dc5e150c79e6b1f3cfaf"),
    "k2001": ("59aed181c71089bad484", "28d150faa3b9878eb8e4"),
    "k2001d": ("c6f13273ce9eb07f7e34", "28d150faa3b9878eb8e4"),
}


def test_farmcpu_inputs_are_the_audits(tmp_path):
    seen = set()
    for name in sc.FARMCPU_CASES:
        argv = sc.farmcpu_case(name, tmp_path)
        vcf, pheno = Path(argv[1]), Path(argv[3])
        stem = vcf.name[: -len(".vcf.gz")]
        got = (hashlib.sha256(gzip.decompress(vcf.read_bytes())).hexdigest()[:20],
               hashlib.sha256(pheno.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:20])
        assert got == AUDIT_INPUTS[stem], stem
        seen.add(stem)
    assert seen == set(AUDIT_INPUTS)


@pytest.mark.slow
@pytest.mark.parametrize("name", list(sc.FARMCPU_CASES))
def test_farmcpu_last_chunk(work, name):
    inputs = work / "inputs"
    inputs.mkdir(exist_ok=True)
    argv = sc.farmcpu_case(name, inputs)
    _same(work, _run(work, "mem", name, argv), _run(work, "disk", name, argv), name)


@pytest.mark.slow
@pytest.mark.parametrize("name", list(sc.BOOT_JOBS_CASES))
def test_boot_jobs(work, name):
    argv = sc.BOOT_JOBS_CASES[name]()
    if argv is None:
        pytest.skip("needs TRACE_B4_DATA (the tomato inputs)")
    _same(work, _run(work, "mem", name, argv), _run(work, "disk", name, argv), name)
