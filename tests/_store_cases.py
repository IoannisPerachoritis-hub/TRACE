"""Inputs and helpers of the slow two-mode tests (tests/test_slow_store_equivalence.py).

* cases(): the comparison set of the scaling audit, 16 cases: 8 on the old example
  (tests/data/example_v1), 2 on the new one (examples/), the planted three-trait panel, 3 on
  tomato and 2 on pepper. The planted, tomato and pepper inputs are not in the repository: they
  are read from the folder named by TRACE_B4_DATA (subfolders planted/, tomato/, pepper/), and
  the cases are skipped without it.
* FARMCPU_CASES: the 7 inputs on which the scaling audit found FarmCPU's one-column last chunk
  (a SNP count that leaves 1 to 3 columns in the last OLS chunk), regenerated exactly from their
  seeds (the audit's generator; the files are byte-identical to the audit's).
* BOOT_JOBS_CASES: subsampling with several workers (the new example with 7, tomato with 4).
* zip_members(): every ZIP member of a run, after removing what differs between two runs of
  the same analysis: the keys that say where the genotype table was kept, the report's time
  and the input folders.
"""
import gzip
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

from gwas.storeplan import MODE_KEYS

REPO = Path(__file__).resolve().parents[1]
EX1 = REPO / "tests" / "data" / "example_v1"
EX2 = REPO / "examples"
DATA = Path(os.environ["TRACE_B4_DATA"]) if os.environ.get("TRACE_B4_DATA") else None

_LD = ["--ld-r2", "0.1", "--ld-flank-kb", "200"]
_ALL = ["--model", "mlm", "mlmm", "farmcpu"]
_TOMATO = ("tomato", "varitome_filtered.vcf", "varitome_phenotypes.csv")
_PEPPER = ("pepper", "G2P-SOL_Pepper_CC_Geno_data_2023.vcf", "pepper_phenotypes_averaged.csv")
_PLANTED = ("planted", "planted.vcf", "planted_pheno.csv")


def _ex1(*a):
    return ["--vcf", str(EX1 / "example.vcf.gz"), "--pheno", str(EX1 / "example_pheno.csv"), *a]


def _ex2(*a):
    return ["--vcf", str(EX2 / "example.vcf.gz"), "--pheno", str(EX2 / "example_pheno.csv"), *a]


def _data(sub, vcf, pheno, *a):
    if DATA is None:
        return None
    return ["--vcf", str(DATA / sub / vcf), "--pheno", str(DATA / sub / pheno), *a]


def cases():
    """[(name, argv or None when its data are not available)]."""
    return [
        ("ex_default", _ex1("--trait", "Trait1")),
        ("ex_t2", _ex1("--trait", "Trait2")),
        ("ex_ld", _ex1("--trait", "Trait2", *_LD)),
        ("ex_ld_t1", _ex1("--trait", "Trait1", *_LD)),
        ("ex_full", _ex1("--trait", "Trait1", *_ALL, *_LD, "--subsampling", "--boot-reps", "5", "--export-qc")),
        ("ex_fdr3", _ex1("--trait", "Trait1", *_ALL, *_LD, "--sig-thresh", "fdr")),
        ("ex_custom_noplots", _ex1("--trait", "Trait1", *_LD, "--sig-thresh", "1e-4", "--no-plots")),
        ("ex_norm_int", _ex1("--trait", "Trait1", "--norm", "int")),
        ("planted3", _data(*_PLANTED, "--trait", "T_chr1", "T_chr2", "T_chr3", *_ALL)),
        ("tomato_default", _data(*_TOMATO, "--trait", "locule_number")),
        ("tomato_meff3", _data(*_TOMATO, "--trait", "locule_number", *_ALL, "--sig-thresh", "meff")),
        ("tomato_fdr3", _data(*_TOMATO, "--trait", "locule_number", *_ALL, "--sig-thresh", "fdr")),
        ("pepper_bx", _data(*_PEPPER, "--trait", "BX", "--no-annotation")),
        ("pepper_fwe3", _data(*_PEPPER, "--trait", "FWe", "--no-annotation", *_ALL)),
        ("ex2_default", _ex2("--trait", "Trait1")),
        ("ex2_full", _ex2("--trait", "Trait1", "Trait2", *_ALL, "--subsampling", "--boot-reps", "5", "--export-qc")),
    ]


BOOT_JOBS_CASES = {
    "j7_ex2_jobs7": lambda: _ex2(*_LD, "--trait", "Trait1", "--subsampling", "--boot-reps", "7", "--boot-jobs", "7"),
    "t1_tomato_jobs4": lambda: _data(*_TOMATO, "--trait", "locule_number", "--subsampling", "--boot-reps", "6",
                                     "--boot-jobs", "4"),
    "t1b_tomato_jobs4_pcs": lambda: _data(*_TOMATO, "--trait", "locule_number", "--subsampling", "--boot-reps",
                                          "6", "--boot-jobs", "4", "--n-pcs", "3"),
}


def threads():
    """The BLAS thread count of the runs (TRACE_SLOW_THREADS, default 1)."""
    return os.environ.get("TRACE_SLOW_THREADS", "1")


def run_cli(argv, out, extra_env=None):
    """cli.py in a fresh process with the BLAS threads pinned; returns the completed process."""
    t = threads()
    env = dict(os.environ, OPENBLAS_NUM_THREADS=t, MKL_NUM_THREADS=t, OMP_NUM_THREADS=t,
               PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    env.pop("PYTHONPATH", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run([sys.executable, str(REPO / "cli.py"), *argv, "--output", str(out)],
                          cwd=str(REPO), env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


# ------------------------------------------------------------------------ comparing ZIPs
_TEXT = (".csv", ".json", ".html", ".txt")
_GENERATED = re.compile(r"<strong>Generated:</strong>[^\n]*")


def _normalise(name, data, folders):
    """A member without what differs between two runs of the same analysis."""
    if not name.endswith(_TEXT):
        return data
    text = data.decode("utf-8", "replace")
    for f in sorted({str(f) for f in folders if f}, key=len, reverse=True):
        for form in (f.replace("\\", "\\\\"), f, f.replace("\\", "/")):     # JSON, plain, slashes
            text = text.replace(form, "<input>")
    if name.endswith("run_metadata.json"):
        meta = json.loads(text)
        for k in MODE_KEYS:
            meta.pop(k, None)
        text = json.dumps(meta, indent=2, sort_keys=True)
    if name.endswith("report.html"):
        text = _GENERATED.sub("", text)
    return text.encode("utf-8")


def zip_members(folder, folders=()):
    """{(trait or 'combined', member): normalised bytes} of the results ZIPs in ``folder`` (the
    file name holds the run's time, so the trait is the key)."""
    out = {}
    zips = sorted(Path(folder).glob("*.zip"))
    assert zips, f"no results ZIP in {folder}"
    for z in zips:
        key = "combined" if "MultiTrait_" in z.name else z.name.split("__")[1]
        with zipfile.ZipFile(z) as f:
            for n in f.namelist():
                out[(key, n)] = _normalise(n, f.read(n), folders)
    return out


def digests(members):
    return {f"{k}::{n}": hashlib.sha256(v).hexdigest()[:20] for (k, n), v in sorted(members.items())}


def stage_digests(folder):
    """{(file, stage, object, occurrence): digest} from the taps_<trait>.json files a run with
    TRACE_TAPS=1 writes next to its output folder (gwas/taps.py)."""
    out = {}
    for p in sorted(Path(folder).glob("taps_*.json")):
        seen = {}
        for r in json.loads(p.read_text(encoding="utf-8")):
            k = (r["stage"], r["name"])
            seen[k] = seen.get(k, -1) + 1
            out[(p.name, r["stage"], r["name"], seen[k])] = {a: b for a, b in r.items() if a not in ("stage", "name")}
    return out


def fingerprint():
    """What decides the last bits of a run's numbers: the machine, the BLAS libraries (version
    and the kernels they chose for this CPU), the thread count and the numeric packages."""
    from importlib.metadata import version

    from threadpoolctl import threadpool_info

    blas = sorted({f"{l.get('internal_api')} {l.get('version')} {l.get('architecture')}"
                   for l in threadpool_info() if l.get("user_api") == "blas"})
    parts = [platform.system(), platform.machine(), platform.processor() or "?", f"cores {os.cpu_count()}",
             f"python {sys.version_info[0]}.{sys.version_info[1]}", *blas, f"threads {threads()}"]
    for pkg in ("numpy", "scipy", "pandas", "scikit-learn", "fastlmm", "pysnptools", "scikit-allel", "matplotlib"):
        try:
            parts.append(f"{pkg} {version(pkg)}")
        except Exception:
            parts.append(f"{pkg} ?")
    raw = " | ".join(parts)
    return raw, hashlib.sha256(raw.encode()).hexdigest()[:12]


# ------------------------------------------------------------------- the FarmCPU inputs
_GT = np.array(["./.", "0/0", "0/1", "1/1"])


def _struct_geno(rng, n, m, n_groups=8, lo=0.25, hi=0.75, drift=0.10):
    grp = np.arange(n) % n_groups
    p = rng.uniform(lo, hi, size=m)
    pg = np.clip(p[None, :] + rng.normal(0, drift, size=(n_groups, m)), 0.08, 0.92)
    G = rng.binomial(2, pg[grp, :]).astype(np.int8)
    for j in range(m):
        while True:
            af = G[:, j].mean() / 2.0
            if 0.15 <= af <= 0.85:
                break
            G[:, j] = rng.binomial(2, 0.5, size=n)
    return G, grp


def _layout(sizes, step=10_000):
    chroms, pos = [], []
    for c, k in sizes:
        chroms += [str(c)] * k
        pos += list(10_000 + step * np.arange(k))
    return np.array(chroms), np.array(pos, dtype=np.int64)


def _stride_sensitive(col):
    """The intercept-only projection of a column differs between a strided and a compact copy."""
    n = col.size
    C = np.ones((n, 1), float)
    C_pinv = np.linalg.inv(C.T @ C) @ C.T
    wide = np.zeros((n, 3), float)
    wide[:, 2] = col
    return bool((C_pinv @ wide[:, 2:3])[0, 0] != (C_pinv @ np.ascontiguousarray(wide[:, 2:3]))[0, 0])


def _write(folder, name, chroms, pos, G, samples, traits, fmt):
    lines = ["##fileformat=VCFv4.2", '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
             "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples)]
    codes = _GT[(G + 1).T]
    for j in range(G.shape[1]):
        lines.append(f"{chroms[j]}\t{int(pos[j])}\tv{j}\tA\tT\t.\tPASS\t.\tGT\t" + "\t".join(codes[j]))
    vcf = Path(folder) / f"{name}.vcf.gz"
    with gzip.GzipFile(vcf, "wb", mtime=0) as fh:
        fh.write(("\n".join(lines) + "\n").encode("utf-8"))
    rows = ["SampleID," + ",".join(traits)]
    for i, s in enumerate(samples):
        rows.append(f"{s}," + ",".join(fmt.format(float(traits[t][i])) for t in traits))
    pheno = Path(folder) / f"{name}_pheno.csv"
    pheno.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return vcf, pheno


def _gen_h(folder, m, n=60, seed=1106):
    rng = np.random.default_rng(seed)
    G, grp = _struct_geno(rng, n, 4001)
    geff = rng.normal(0, 0.3, size=8)[grp]
    noise = rng.normal(0, 0.6, size=n)
    tn1, tn2 = rng.normal(0, 1, size=n), rng.normal(0, 1, size=n)
    G = G[:, :m].copy()
    tries = 0
    while not _stride_sensitive(G[:, m - 1].astype(float)):
        tries += 1
        G[:, m - 1] = _struct_geno(rng, n, 1)[0][:, 0]
        if tries > 500:
            raise RuntimeError("no stride-sensitive column found")
    per = [m // 4, m // 4, m // 4, m - 3 * (m // 4)]
    chroms, pos = _layout([(c + 1, k) for c, k in enumerate(per)])
    samples = [f"H{i:03d}" for i in range(n)]
    y = sum(b * G[:, j].astype(float) for j, b in [(300, 1.3), (900, 1.1), (1500, 1.0)]) + geff + noise
    name = f"h{m}" + ("" if n == 60 else f"n{n}")
    return _write(folder, name, chroms, pos, G, samples, {"Tq": y, "Tnull": tn1, "Tnull2": tn2}, "{:.5f}")


def _gen_k(folder, duplicate, seed=0):
    n, m = 60, 2001
    G, _grp = _struct_geno(np.random.default_rng(1106), n, 4001)
    G = G[:, :m].copy()
    r = np.random.default_rng(5000 + seed)          # the seed the audit's search found
    col = _struct_geno(r, n, 1)[0][:, 0]
    y = 1.5 * col + r.normal(0, 0.7, size=n)
    G[:, m - 1] = col
    if duplicate:
        G[:, m - 2] = col
    per = [m // 4, m // 4, m // 4, m - 3 * (m // 4)]
    chroms, pos = _layout([(c + 1, k) for c, k in enumerate(per)])
    samples = [f"K{i:03d}" for i in range(n)]
    return _write(folder, "k2001d" if duplicate else "k2001", chroms, pos, G, samples, {"Tlast": y}, "{!r}")


_NULL = ["--trait", "Tnull", "--model", "mlm", "farmcpu"]
FARMCPU_CASES = {
    "h2001_null": (lambda f: _gen_h(f, 2001), _NULL),
    "h2001n40_null": (lambda f: _gen_h(f, 2001, n=40), _NULL),
    "h4001_null": (lambda f: _gen_h(f, 4001), _NULL),
    "h2003_null": (lambda f: _gen_h(f, 2003), _NULL),
    "h2001_null2": (lambda f: _gen_h(f, 2001), ["--trait", "Tnull2", "--model", "mlm", "mlmm", "farmcpu"]),
    "k2001_last": (lambda f: _gen_k(f, False), ["--trait", "Tlast", "--model", "mlm", "farmcpu"]),
    "k2001d_last": (lambda f: _gen_k(f, True), ["--trait", "Tlast", "--model", "mlm", "farmcpu"]),
}


def farmcpu_case(name, folder):
    """The argv of one of the 7 FarmCPU cases, its inputs written into ``folder``."""
    gen, args = FARMCPU_CASES[name]
    vcf, pheno = gen(folder)
    return ["--vcf", str(vcf), "--pheno", str(pheno), *args]
