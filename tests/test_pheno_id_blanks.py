"""Phenotype sample IDs with blanks around them match the VCF's samples. The blanks were
removed for the duplicate check only, and the matching used the raw IDs, so every padded row
was silently left out (25 of 50 samples analysed in the scaling audit's test file). Both modes.
"""
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


def _files(tmp_path, n=50, m=60, seed=8):
    rng = np.random.default_rng(seed)
    samples = [f"SAMPLE_{i:03d}" for i in range(n)]
    lines = ["##fileformat=VCFv4.2", '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">',
             "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples)]
    for j in range(m):
        g = "\t".join(["0/0", "0/1", "1/1"][k] for k in rng.integers(0, 3, n))
        lines.append(f"1\t{100 + 50 * j}\t.\tA\tT\t.\tPASS\t.\tGT\t{g}")
    vcf = tmp_path / "ids.vcf"
    vcf.write_text("\n".join(lines) + "\n")
    # every second phenotype ID padded with blanks, as in the audit's file
    rows = ["SampleID,Trait1"] + [f"{(' ' + s + '  ') if i % 2 else s},{rng.normal():.4f}"
                                  for i, s in enumerate(samples)]
    pheno = tmp_path / "ids_pheno.csv"
    pheno.write_text("\n".join(rows) + "\n")
    return vcf, pheno


def _capture_warnings(monkeypatch):
    """The pipeline's warnings (gwas.qc._log_or_warn, also imported by gwas.storepipe)."""
    import gwas.storepipe as sp

    seen = []
    monkeypatch.setattr(gqc, "_log_or_warn", seen.append)
    monkeypatch.setattr(sp, "_log_or_warn", seen.append)
    return seen


@pytest.mark.parametrize("mode", ["memory", "disk"])
def test_padded_phenotype_ids_all_match(tmp_path, mode, monkeypatch):
    seen = _capture_warnings(monkeypatch)
    vcf, pheno_path = _files(tmp_path)
    pheno = pd.read_csv(pheno_path, index_col=0, encoding="utf-8-sig")
    if mode == "memory":
        parsed = ParsedGenotypes.from_callset(_load(Path(vcf).read_bytes()))
        harmonize = gqc._pipeline_harmonize_ids
    else:
        parsed = ingest_vcf(vcf, tmp_path / "s")
        harmonize = TraitStore(parsed).harmonize_ids
    try:
        geno_df, ph, *_ = harmonize(parsed.genotypes, parsed.samples, parsed.chroms, parsed.positions,
                                    parsed.vid, parsed.ref, parsed.alt, pheno, parsed.info_scores)
        assert len(geno_df) == 50 and len(ph) == 50
        assert list(ph.index) == list(geno_df.index)
        assert all(s == s.strip() for s in ph.index)
        assert any(m.startswith("25 phenotype sample IDs had blanks") for m in seen), seen
    finally:
        if mode == "disk":
            parsed.store.close()


def test_ids_without_blanks_are_unchanged(tmp_path, monkeypatch):
    seen = _capture_warnings(monkeypatch)
    vcf, pheno_path = _files(tmp_path)
    pheno = pd.read_csv(pheno_path, index_col=0, encoding="utf-8-sig")
    pheno.index = [s.strip() for s in pheno.index]
    parsed = ParsedGenotypes.from_callset(_load(Path(vcf).read_bytes()))
    geno_df, ph, *_ = gqc._pipeline_harmonize_ids(parsed.genotypes, parsed.samples, parsed.chroms,
                                                  parsed.positions, parsed.vid, parsed.ref, parsed.alt,
                                                  pheno, parsed.info_scores)
    assert len(geno_df) == 50
    assert not any("blank" in m for m in seen)
