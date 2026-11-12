"""An imputation-quality field declared Number=A or Number=R (Beagle 5 declares DR2 as
Number=A) is read as one value per biallelic record, the ALT allele's; it used to stop the
parse with an IndexError (the records x k array was flattened, then masked per record).
Both modes, the table in memory and on disk, give the same values.
"""
from pathlib import Path

import numpy as np
import pytest

import gwas.io as gio
from gwas.genostore import ingest_vcf
from gwas.pipeline import ParsedGenotypes

_load = getattr(gio.load_vcf_cached, "__wrapped__", gio.load_vcf_cached)


def _vcf(path, number, n=12, m=20, seed=4):
    rng = np.random.default_rng(seed)
    lines = ["##fileformat=VCFv4.2",
             f'##INFO=<ID=DR2,Number={number},Type=Float,Description="Dosage R-squared">',
             '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
             "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(f"S{i}" for i in range(n))]
    expected = []
    for j in range(m):
        multi = j % 7 == 3                       # a multi-allelic record: dropped by the parse
        alt = "T,G" if multi else "T"
        dr2 = [round(float(rng.uniform(0.3, 1.0)), 3) for _ in range(2 if (multi and number != "1") else 1)]
        if number in ("A", "1"):
            info = "DR2=" + ",".join(map(str, dr2))
        else:                                    # Number=R: REF's value first
            info = "DR2=" + ",".join(map(str, [1.0] + dr2))
        if not multi:
            expected.append(np.float32(dr2[0]))
        gt = "\t".join(["0/0", "0/1", "1/1"][k] for k in rng.integers(0, 3, n))
        lines.append(f"1\t{100 + 10 * j}\t.\tA\t{alt}\t.\tPASS\t{info}\tGT\t{gt}")
    path.write_text("\n".join(lines) + "\n")
    return np.array(expected, dtype=np.float32)


@pytest.mark.parametrize("number", ["A", "R"])
def test_one_value_per_record_in_both_modes(tmp_path, number):
    vcf = tmp_path / f"q_{number}.vcf"
    expected = _vcf(vcf, number)
    mem = ParsedGenotypes.from_callset(_load(Path(vcf).read_bytes()))
    assert mem.info_field == "DR2"
    assert mem.info_scores.dtype == np.float32
    assert np.array_equal(mem.info_scores, expected)
    disk = ingest_vcf(vcf, tmp_path / "s")
    try:
        assert disk.info_field == "DR2"
        assert disk.info_scores.tobytes() == mem.info_scores.tobytes()
    finally:
        disk.store.close()


def test_number_one_fields_are_unchanged(tmp_path):
    """A quality field with one value per record (Number=1) is read as before."""
    vcf = tmp_path / "q1.vcf"
    expected = _vcf(vcf, "1")
    mem = ParsedGenotypes.from_callset(_load(Path(vcf).read_bytes()))
    assert np.array_equal(mem.info_scores, expected)
    disk = ingest_vcf(vcf, tmp_path / "s")
    try:
        assert disk.info_scores.tobytes() == mem.info_scores.tobytes()
    finally:
        disk.store.close()
