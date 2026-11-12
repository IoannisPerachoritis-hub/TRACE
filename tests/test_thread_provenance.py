"""The thread and core counts behind a run's last bits are recorded (gwas/provenance.py): in
run_metadata.json and in the report's provenance table."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from gwas.provenance import blas_info, subsampling_threads, thread_provenance

REPO = Path(__file__).resolve().parents[1]
KEYS = ["BLAS library", "BLAS threads", "CPU cores (logical)", "CPU cores (physical)",
        "Subsampling workers (--boot-jobs)", "Subsampling BLAS threads per worker"]


def test_keys_and_values():
    p = thread_provenance(SimpleNamespace(subsampling=False, boot_jobs=1))
    assert list(p) == KEYS
    assert p["CPU cores (logical)"] == os.cpu_count()
    assert p["Subsampling workers (--boot-jobs)"] == "N/A"
    q = thread_provenance(SimpleNamespace(subsampling=True, boot_jobs=4))
    assert q["Subsampling workers (--boot-jobs)"] == 4
    assert q["Subsampling BLAS threads per worker"] == max((os.cpu_count() or 1) // 4, 1)
    assert thread_provenance(SimpleNamespace(subsampling=True, boot_jobs=1))[
        "Subsampling BLAS threads per worker"] == "as the run"
    assert subsampling_threads(1) is None


def test_blas_thread_count_follows_the_environment():
    code = "from gwas.provenance import blas_info; print(blas_info()[1])"
    for n in ("1", "3"):
        env = dict(os.environ, OPENBLAS_NUM_THREADS=n, MKL_NUM_THREADS=n, OMP_NUM_THREADS=n,
                   PYTHONPATH=str(REPO))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=REPO)
        assert out.stdout.strip() == n, out.stderr
    assert blas_info()[0] != "unknown"


def test_keys_reach_the_metadata_and_the_report(tmp_path, monkeypatch):
    import zipfile

    import numpy as np
    import pandas as pd

    import cli

    rng = np.random.default_rng(3)
    n, m = 30, 150
    samples = [f"S{i}" for i in range(n)]
    lines = ["##fileformat=VCFv4.2", '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">',
             "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples)]
    for j in range(m):
        g = "\t".join(["0/0", "0/1", "1/1"][k] for k in rng.integers(0, 3, n))
        lines.append(f"chr{1 + j // 60}\t{1000 + 3000 * j}\t.\tA\tT\t.\tPASS\t.\tGT\t{g}")
    (tmp_path / "v.vcf").write_text("\n".join(lines) + "\n")
    pd.DataFrame({"T": rng.normal(size=n)}, index=pd.Index(samples, name="id")).to_csv(tmp_path / "p.csv")
    monkeypatch.setattr(sys, "argv", ["cli.py", "--vcf", str(tmp_path / "v.vcf"), "--pheno", str(tmp_path / "p.csv"),
                                      "--trait", "T", "--output", str(tmp_path / "o"), "--no-plots"])
    cli.main()
    z = sorted((tmp_path / "o").glob("*.zip"))[0]
    with zipfile.ZipFile(z) as f:
        meta = json.loads(f.read("run_metadata.json"))
        html = f.read("report.html").decode()
    for k in KEYS:
        assert k in meta
        assert k in html
    assert meta["BLAS threads"] == blas_info()[1]
