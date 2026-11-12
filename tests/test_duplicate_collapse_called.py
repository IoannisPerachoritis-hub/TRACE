"""The collapse of records at the same chr:pos keeps a copy with called genotypes. It took
np.argmax over the copies' minor allele frequencies, and an all-missing copy has a NaN
frequency, which np.argmax returns first; SNP QC then removed that copy and the position was
lost although a called copy existed. Among copies with calls, the rule is unchanged (highest
MAF, the first on a tie). Both modes."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gwas.io as gio
from gwas import qc as gqc
from gwas.genostore import ingest_vcf
from gwas.pipeline import ParsedGenotypes
from gwas.storepipe import TraitStore

_load = getattr(gio.load_vcf_cached, "__wrapped__", gio.load_vcf_cached)


def _files(tmp_path, n=24, seed=9):
    rng = np.random.default_rng(seed)
    samples = [f"S{i:02d}" for i in range(n)]
    lines = ["##fileformat=VCFv4.2", '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">',
             "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples)]

    def gts(p=0.3):
        return "\t".join(["0/0", "0/1", "1/1"][int(rng.random() < p) + int(rng.random() < p)] for _ in range(n))
    for j in range(30):
        lines.append(f"1\t{1000 + 100 * j}\t.\tA\tT\t.\tPASS\t.\tGT\t{gts()}")
    # at 1,050: an all-missing copy FIRST, then a called copy
    lines.append("1\t1050\tdup_missing\tA\tG\t.\tPASS\t.\tGT\t" + "\t".join(["./."] * n))
    lines.append(f"1\t1050\tdup_called\tA\tT\t.\tPASS\t.\tGT\t{gts(0.4)}")
    # at 1,150: two called copies with different MAF (today's rule: the higher MAF)
    lines.append(f"1\t1150\tdup_low\tA\tG\t.\tPASS\t.\tGT\t{gts(0.1)}")
    lines.append(f"1\t1150\tdup_high\tA\tT\t.\tPASS\t.\tGT\t{gts(0.45)}")
    # at 1,250: only all-missing copies (the first is kept, as before)
    lines.append("1\t1250\tdup_m1\tA\tG\t.\tPASS\t.\tGT\t" + "\t".join(["./."] * n))
    lines.append("1\t1250\tdup_m2\tA\tT\t.\tPASS\t.\tGT\t" + "\t".join(["./."] * n))
    body = sorted(lines[3:], key=lambda l: int(l.split("\t")[1]))
    (tmp_path / "d.vcf").write_text("\n".join(lines[:3] + body) + "\n")
    pd.DataFrame({"T": rng.normal(size=n)}, index=pd.Index(samples, name="id")).to_csv(tmp_path / "d_pheno.csv")
    return tmp_path / "d.vcf", tmp_path / "d_pheno.csv"


@pytest.mark.parametrize("mode", ["memory", "disk"])
def test_called_copy_is_kept(tmp_path, mode):
    vcf, pheno_path = _files(tmp_path)
    pheno = pd.read_csv(pheno_path, index_col=0)
    if mode == "memory":
        parsed = ParsedGenotypes.from_callset(_load(Path(vcf).read_bytes()))
        harmonize = gqc._pipeline_harmonize_ids
    else:
        parsed = ingest_vcf(vcf, tmp_path / "s")
        harmonize = TraitStore(parsed).harmonize_ids
    try:
        _g, _p, chroms, positions, sid, *_ = harmonize(parsed.genotypes, parsed.samples, parsed.chroms,
                                                       parsed.positions, parsed.vid, parsed.ref, parsed.alt,
                                                       pheno, parsed.info_scores)
        kept = set(map(str, sid))
        assert "dup_called" in kept and "dup_missing" not in kept
        assert "dup_high" in kept and "dup_low" not in kept
        assert "dup_m1" in kept and "dup_m2" not in kept
    finally:
        if mode == "disk":
            parsed.store.close()
