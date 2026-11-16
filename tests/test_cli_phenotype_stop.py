"""A phenotype-QC stop in a single-trait CLI run prints one line, "Error: <message>", and exits
with code 1, without a traceback, with the table in memory and on disk. The stop is a ValueError
raised in gwas.qc._pipeline_phenotype_qc (memory) or gwas.storepipe.TraitStore.phenotype_qc
(disk); gwas.pipeline.is_phenotype_qc_stop recognises both, and nothing else."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gwas.pipeline import is_phenotype_qc_stop

REPO = Path(__file__).resolve().parents[1]


def _inputs(tmp_path, n=30, m=120):
    rng = np.random.default_rng(5)
    samples = [f"S{i}" for i in range(n)]
    lines = ["##fileformat=VCFv4.2", '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">',
             "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples)]
    for j in range(m):
        g = "\t".join(["0/0", "0/1", "1/1"][k] for k in rng.integers(0, 3, n))
        lines.append(f"chr{1 + j // 40}\t{1000 + 3000 * j}\t.\tA\tT\t.\tPASS\t.\tGT\t{g}")
    (tmp_path / "v.vcf").write_text("\n".join(lines) + "\n")
    pd.DataFrame({"T": rng.normal(size=n), "C": np.full(n, 3.0)},
                 index=pd.Index(samples, name="id")).to_csv(tmp_path / "p.csv")
    return tmp_path / "v.vcf", tmp_path / "p.csv"


@pytest.mark.parametrize("mode", ["memory", "disk"])
def test_constant_trait_stops_with_one_error_line(tmp_path, mode):
    vcf, pheno = _inputs(tmp_path)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    out = subprocess.run([sys.executable, str(REPO / "cli.py"), "--vcf", str(vcf), "--pheno", str(pheno),
                          "--trait", "C", "--output", str(tmp_path / "o"), "--no-plots",
                          "--geno-store", mode],
                         cwd=str(REPO), env=env, capture_output=True, text=True, encoding="utf-8",
                         errors="replace")
    assert out.returncode == 1, out.stderr
    assert "Traceback" not in out.stderr and "Traceback" not in out.stdout
    errors = [ln for ln in out.stderr.splitlines() if ln.startswith("Error: ")]
    assert len(errors) == 1, out.stderr
    assert not list((tmp_path / "o").glob("_trace_geno_store_*"))     # the table was removed


def test_the_predicate_names_only_phenotype_qc():
    from gwas.qc import _pipeline_phenotype_qc

    geno_df = pd.DataFrame({"a": [0.0, 1.0, 2.0]}, index=["x", "y", "z"])
    pheno = pd.DataFrame({"C": [1.0, 1.0, 1.0]}, index=["x", "y", "z"])
    with pytest.raises(ValueError) as caught:
        _pipeline_phenotype_qc(geno_df, pheno, "C", "none", 0.2)
    assert is_phenotype_qc_stop(caught.value)
    try:
        raise ValueError("something else")
    except ValueError as other:
        assert not is_phenotype_qc_stop(other)
    assert not is_phenotype_qc_stop(RuntimeError("x"))
