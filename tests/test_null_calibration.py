"""End-to-end null calibration test.

Simulates a phenotype that is independent of every SNP and asserts:

1. Genomic-control inflation (lambda_GC) lies in [0.85, 1.15]. The
   acceptable window is wider than the auto-PC band [0.95, 1.05] because
   simulation noise at small n produces larger sampling variance in the
   median-chi-squared estimator.
2. No SNP passes a Bonferroni 5% threshold.

The test runs the LOCO MLM kernel (the same code path the manuscript
benchmarks exercise) and is fast enough for CI (~10 s on Linux runners).
A failure here means a calibration regression has slipped in — the
single most common way for a new GWAS implementation to silently break.
"""
from __future__ import annotations

import numpy as np
import pytest

from gwas.kinship import (
    _build_grm_from_Z,
    _build_loco_kernels_impl,
    _standardize_geno_for_grm,
)
from gwas.models import _run_gwas_impl
from gwas.plotting import compute_lambda_gc
from gwas.utils import PhenoData


@pytest.fixture(scope="module")
def null_dataset():
    """Simulate 200 samples x 1500 SNPs across 5 chromosomes, null trait."""
    rng = np.random.default_rng(20260420)
    n_samples = 200
    n_chroms = 5
    snps_per_chrom = 300
    n_snps = n_chroms * snps_per_chrom

    # Biallelic genotypes under HWE with MAF in [0.10, 0.45]
    mafs = rng.uniform(0.10, 0.45, size=n_snps)
    geno = np.zeros((n_samples, n_snps), dtype=np.float32)
    for j, maf in enumerate(mafs):
        geno[:, j] = rng.binomial(2, maf, size=n_samples).astype(np.float32)

    chroms_num = np.repeat(np.arange(1, n_chroms + 1), snps_per_chrom).astype(int)
    chroms = np.array([f"{c}" for c in chroms_num])
    positions = np.tile(np.arange(1, snps_per_chrom + 1) * 10000, n_chroms).astype(int)
    sid = np.array([f"chr{c}_{p}" for c, p in zip(chroms, positions)])
    iid = np.array([[f"S{i:03d}", f"S{i:03d}"] for i in range(n_samples)])

    # Null phenotype: independent normal, no SNP effects
    y = rng.normal(0.0, 1.0, size=n_samples).astype(np.float32)

    return {
        "geno": geno, "y": y, "sid": sid,
        "chroms": chroms, "chroms_num": chroms_num,
        "positions": positions, "iid": iid,
    }


def test_null_lambda_gc_is_calibrated(null_dataset):
    """Under the null, lambda_GC should be within [0.85, 1.15]."""
    d = null_dataset
    Z_grm = _standardize_geno_for_grm(d["geno"])
    K_base = _build_grm_from_Z(Z_grm).astype(np.float32)
    K0, K_by_chr, _ = _build_loco_kernels_impl(d["iid"], Z_grm, d["chroms"], K_base=K_base)
    pheno_reader = PhenoData(iid=d["iid"], val=d["y"])

    gwas_df = _run_gwas_impl(
        d["geno"], d["y"], pcs_full=None, n_pcs=0,
        sid=d["sid"], positions=d["positions"],
        chroms=d["chroms"], chroms_num=d["chroms_num"],
        iid=d["iid"], _K0=K0, _K_by_chr=K_by_chr,
        pheno_reader=pheno_reader, trait_name="NullTrait",
    )

    pvals = gwas_df["PValue"].astype(float).values
    lam = compute_lambda_gc(pvals, trim=False)

    assert np.isfinite(lam), f"lambda_GC is non-finite ({lam})"
    assert 0.85 <= lam <= 1.15, (
        f"Null calibration broken: lambda_GC = {lam:.3f}, expected [0.85, 1.15]. "
        "A regression here usually means kinship/PC handling has shifted."
    )


def test_null_has_no_genome_wide_hits(null_dataset):
    """Under the null, no SNP should pass Bonferroni 5%."""
    d = null_dataset
    Z_grm = _standardize_geno_for_grm(d["geno"])
    K_base = _build_grm_from_Z(Z_grm).astype(np.float32)
    K0, K_by_chr, _ = _build_loco_kernels_impl(d["iid"], Z_grm, d["chroms"], K_base=K_base)
    pheno_reader = PhenoData(iid=d["iid"], val=d["y"])

    gwas_df = _run_gwas_impl(
        d["geno"], d["y"], pcs_full=None, n_pcs=0,
        sid=d["sid"], positions=d["positions"],
        chroms=d["chroms"], chroms_num=d["chroms_num"],
        iid=d["iid"], _K0=K0, _K_by_chr=K_by_chr,
        pheno_reader=pheno_reader, trait_name="NullTrait",
    )

    pvals = gwas_df["PValue"].astype(float).values
    bonf = 0.05 / len(pvals)
    n_hits = int((pvals < bonf).sum())
    assert n_hits == 0, (
        f"Null produced {n_hits} Bonferroni-significant SNPs (threshold {bonf:.2e}). "
        "This indicates the test statistic is mis-calibrated under H0."
    )
