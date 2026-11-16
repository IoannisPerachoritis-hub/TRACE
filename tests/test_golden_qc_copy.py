"""The three Table S8 tests (test_varitome_locule_blocks_match_table_s8, test_ld_blocks_bitwise_stable,
test_haplotype_blocks_bitwise_stable) skip, with the reason, on a copy of the tomato QC data that
is not the one the goldens were made from, instead of failing; on the corrected copy they run.
The other copy here is a temporary folder named by TRACE_VARITOME_DIR: the real benchmarks/qc_data
folder is never changed."""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests import _golden_qc as gq

REPO = Path(__file__).resolve().parents[1]
THREE = ["tests/test_golden_blocks.py::test_varitome_locule_blocks_match_table_s8",
         "tests/test_golden_published.py::test_ld_blocks_bitwise_stable",
         "tests/test_golden_published.py::test_haplotype_blocks_bitwise_stable"]


def _other_copy(tmp_path):
    d = tmp_path / "old_qc"
    d.mkdir()
    (d / "QC_genotype_matrix.csv").write_text("x\n")
    (d / "platform_GWAS_locule_number.csv").write_text("SNP,PValue\nA,0.5\n")
    return d


def test_problem_names_an_absent_and_another_copy(tmp_path):
    assert "absent" in gq.qc_copy_problem(tmp_path / "nothing_here")
    other = gq.qc_copy_problem(_other_copy(tmp_path))
    assert "another copy" in other and "REGENERATION_LOG" in other


def test_corrected_copy_has_no_problem():
    real = REPO / "benchmarks" / "qc_data" / "tomato_locule_number"
    if not (real / "platform_GWAS_locule_number.csv").exists():
        pytest.skip("no local tomato QC copy (gitignored)")
    problem = gq.qc_copy_problem(real)
    if problem and "another copy" in problem:
        pytest.skip("this machine holds an older QC copy: " + problem)
    assert problem is None


@pytest.mark.golden
def test_the_three_tests_skip_on_another_copy(tmp_path):
    env = dict(os.environ, TRACE_VARITOME_DIR=str(_other_copy(tmp_path)), PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-m", "pytest", *THREE, "-rs", "-q", "-p", "no:cacheprovider",
                          f"--basetemp={tmp_path / 'pt'}"],
                         cwd=str(REPO), env=env, capture_output=True, text=True, encoding="utf-8",
                         errors="replace")
    assert "3 skipped" in out.stdout, out.stdout[-3000:]
    # pytest's summary merges identical reasons into one "SKIPPED [n] file:line: reason" line
    n = sum(int(m.group(1)) for m in re.finditer(r"SKIPPED \[(\d+)\][^\n]*another copy of the tomato QC data",
                                                 out.stdout))
    assert n == 3, out.stdout[-3000:]
