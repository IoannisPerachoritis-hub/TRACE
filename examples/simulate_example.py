#!/usr/bin/env python3
"""
Generate a small synthetic dataset for TRACE quick-start demonstration.

Creates:
  - example.vcf.gz  (~50 samples, ~500 biallelic SNPs across 3 chromosomes)
  - example_pheno.csv  (two traits: Trait1 with one embedded QTL, Trait2 as noise)

The QTL is placed near the middle of chromosome 2 with moderate effect
(h² ~ 0.4). This dataset runs through the full TRACE pipeline in <30 s
and is intended for reviewers and new users to verify installation.

Usage:
    python simulate_example.py
"""
import gzip
import os
import sys

import numpy as np
import pandas as pd

# ── Parameters ──────────────────────────────────────────────────────
N_SAMPLES = 50
N_CHROMS = 3
SNPS_PER_CHROM = 170  # ~510 total
QTL_CHROM = 2
QTL_IDX = 85  # middle of chrom 2
QTL_BETA = 1.5  # effect size (standardised genotype units)
H2_TARGET = 0.4
MAF_RANGE = (0.10, 0.45)
SEED = 42

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _make_genotypes(rng):
    """Simulate biallelic genotypes (0/1/2) under HWE."""
    geno = np.empty((N_SAMPLES, N_CHROMS * SNPS_PER_CHROM), dtype=np.int8)
    mafs = rng.uniform(*MAF_RANGE, size=N_CHROMS * SNPS_PER_CHROM)
    for j, maf in enumerate(mafs):
        geno[:, j] = rng.binomial(2, maf, size=N_SAMPLES)
    return geno, mafs


def _make_phenotypes(rng, geno):
    """Trait1 = QTL signal + noise (h²~0.4); Trait2 = pure noise."""
    qtl_col = (QTL_CHROM - 1) * SNPS_PER_CHROM + QTL_IDX
    g = geno[:, qtl_col].astype(float)
    g = (g - g.mean()) / (g.std() + 1e-12)
    signal = QTL_BETA * g
    var_signal = np.var(signal)
    var_noise = var_signal * (1 - H2_TARGET) / H2_TARGET
    noise = rng.normal(0, np.sqrt(var_noise), size=N_SAMPLES)
    trait1 = signal + noise

    trait2 = rng.normal(0, 1, size=N_SAMPLES)

    return trait1, trait2


def _write_vcf(geno, mafs, path):
    """Write a minimal VCF 4.1 file (bgzip-compatible gzip)."""
    n_snps = geno.shape[1]

    with gzip.open(path, "wt") as fh:
        # Header
        fh.write("##fileformat=VCFv4.1\n")
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        for c in range(1, N_CHROMS + 1):
            fh.write(f"##contig=<ID=chr{c:02d},length={SNPS_PER_CHROM * 10000}>\n")
        samples = "\t".join(f"SAMPLE_{i:03d}" for i in range(N_SAMPLES))
        fh.write(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{samples}\n")

        # Genotype encoding
        gt_map = {0: "0/0", 1: "0/1", 2: "1/1"}
        for j in range(n_snps):
            chrom_idx = j // SNPS_PER_CHROM
            pos_in_chrom = (j % SNPS_PER_CHROM) + 1
            chrom = f"chr{chrom_idx + 1:02d}"
            pos = pos_in_chrom * 10000
            snp_id = f"{chrom}_{pos}"
            gts = "\t".join(gt_map[g] for g in geno[:, j])
            fh.write(f"{chrom}\t{pos}\t{snp_id}\tA\tT\t.\tPASS\t.\tGT\t{gts}\n")

    return path


def _write_phenotypes(trait1, trait2, path):
    """Write phenotype CSV with sample IDs matching the VCF."""
    df = pd.DataFrame({
        "SampleID": [f"SAMPLE_{i:03d}" for i in range(N_SAMPLES)],
        "Trait1": np.round(trait1, 4),
        "Trait2": np.round(trait2, 4),
    })
    df.to_csv(path, index=False)
    return path


def main():
    rng = np.random.default_rng(SEED)
    print("Generating synthetic genotypes...")
    geno, mafs = _make_genotypes(rng)
    print(f"  {geno.shape[1]} SNPs x {geno.shape[0]} samples across {N_CHROMS} chromosomes")

    print("Simulating phenotypes...")
    trait1, trait2 = _make_phenotypes(rng, geno)
    qtl_snp = f"chr{QTL_CHROM:02d}_{(QTL_IDX + 1) * 10000}"
    print(f"  QTL at {qtl_snp} (h2 ~ {H2_TARGET}, beta = {QTL_BETA})")

    vcf_path = os.path.join(OUT_DIR, "example.vcf.gz")
    pheno_path = os.path.join(OUT_DIR, "example_pheno.csv")

    print(f"Writing {vcf_path} ...")
    _write_vcf(geno, mafs, vcf_path)
    print(f"Writing {pheno_path} ...")
    _write_phenotypes(trait1, trait2, pheno_path)

    print("\nDone! Run TRACE with:")
    print(f"  trace-gwas --vcf {vcf_path} --pheno {pheno_path} --trait Trait1 --output examples/output/")


if __name__ == "__main__":
    main()
