#!/usr/bin/env python3
"""
Generate a small synthetic dataset for TRACE quick-start demonstration.

Creates:
  - example.vcf.gz     (100 inbred lines, 600 biallelic SNPs on 3 chromosomes)
  - example_pheno.csv  (two traits: Trait1 with one QTL on chromosome 2, Trait2 noise)

The lines are mosaics of four founder haplotypes: along each chromosome a line copies
one founder until a recombination breakpoint, then another. Neighbouring SNPs often
share the founder pattern (which founders carry the ALT allele), so SNPs close together
are in strong linkage disequilibrium (LD) and LD decays with distance. At TRACE's
defaults the QTL forms an LD block with its neighbours, whose haplotypes are then
tested, so every Post-GWAS step has something to show.

Usage:
    python simulate_example.py [OUT_DIR]
"""
import gzip
import io
import os
import sys

import numpy as np
import pandas as pd

# ── Parameters ──────────────────────────────────────────────────────
N_SAMPLES = 100
N_CHROMS = 3
SNPS_PER_CHROM = 200          # one SNP every 10 kb: 2 Mb per chromosome
SPACING_BP = 10_000
N_FOUNDERS = 4
FOUNDER_FREQ = (0.4, 0.3, 0.2, 0.1)
BREAKS_PER_MB = 4             # recombination breakpoints per line per Mb
RUN_MEAN_SNPS = 6             # mean length of a run of SNPs sharing a founder pattern
QTL_CHROM = 2
QTL_IDX = 100                 # the QTL is the 101st SNP of chromosome 2 (1,010,000 bp)
QTL_RUN = 9                   # the QTL's run: SNPs QTL_IDX-4 .. QTL_IDX+4 share its pattern
QTL_PATTERN = (1, 1, 0, 0)    # founders 1 and 2 carry the ALT allele at the QTL's run
H2_TARGET = 0.3
SEED = 42

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# every split of the four founders into ALT carriers and the rest, minor side first
_PATTERNS = [p for p in np.ndindex(*(2,) * N_FOUNDERS) if 0 < sum(p) <= N_FOUNDERS // 2]


def _founder_patterns(rng):
    """For each SNP, which founders carry its ALT allele: runs of SNPs share a pattern."""
    n = SNPS_PER_CHROM
    out = np.empty((N_CHROMS, n, N_FOUNDERS), dtype=np.int8)
    for c in range(N_CHROMS):
        j = 0
        while j < n:
            run = int(rng.geometric(1.0 / RUN_MEAN_SNPS))
            out[c, j:j + run] = _PATTERNS[rng.integers(len(_PATTERNS))]
            j += run
    qc = QTL_CHROM - 1
    half = QTL_RUN // 2
    out[qc, QTL_IDX - half:QTL_IDX + half + 1] = QTL_PATTERN
    return out


def _founder_mosaics(rng):
    """For each line and chromosome, the founder copied at each SNP."""
    length_mb = SNPS_PER_CHROM * SPACING_BP / 1e6
    pos = (np.arange(SNPS_PER_CHROM) + 1) * SPACING_BP
    out = np.empty((N_SAMPLES, N_CHROMS, SNPS_PER_CHROM), dtype=np.int8)
    for i in range(N_SAMPLES):
        for c in range(N_CHROMS):
            n_breaks = rng.poisson(BREAKS_PER_MB * length_mb)
            breaks = np.sort(rng.uniform(0, SNPS_PER_CHROM * SPACING_BP, n_breaks))
            seg = np.searchsorted(breaks, pos)
            founders = rng.choice(N_FOUNDERS, size=n_breaks + 1, p=FOUNDER_FREQ)
            out[i, c] = founders[seg]
    return out


def _make_genotypes(rng):
    """Homozygous genotypes (0 or 2 ALT copies): the founder's allele at each SNP."""
    patterns = _founder_patterns(rng)
    mosaics = _founder_mosaics(rng)
    geno = np.empty((N_SAMPLES, N_CHROMS * SNPS_PER_CHROM), dtype=np.int8)
    snp = np.arange(SNPS_PER_CHROM)[None, :]
    for c in range(N_CHROMS):
        # line i at SNP j carries the allele of the founder it copies there
        alt = patterns[c][snp, mosaics[:, c]]
        geno[:, c * SNPS_PER_CHROM:(c + 1) * SNPS_PER_CHROM] = 2 * alt
    return geno


def _make_phenotypes(rng, geno):
    """Trait1 = QTL signal + noise (h2 ~ 0.5); Trait2 = pure noise."""
    qtl_col = (QTL_CHROM - 1) * SNPS_PER_CHROM + QTL_IDX
    g = geno[:, qtl_col].astype(float)
    g = (g - g.mean()) / (g.std() + 1e-12)
    var_noise = (1 - H2_TARGET) / H2_TARGET
    trait1 = g + rng.normal(0, np.sqrt(var_noise), size=N_SAMPLES)
    trait2 = rng.normal(0, 1, size=N_SAMPLES)
    return trait1, trait2


def _write_vcf(geno, path):
    """Write a minimal VCF 4.1 file (gzip, byte-reproducible: mtime 0)."""
    with open(path, "wb") as _raw, \
            gzip.GzipFile(filename="", mode="wb", fileobj=_raw, mtime=0) as _gz, \
            io.TextIOWrapper(_gz, encoding="utf-8", newline="\n") as fh:
        fh.write("##fileformat=VCFv4.1\n")
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        for c in range(1, N_CHROMS + 1):
            fh.write(f"##contig=<ID=chr{c:02d},length={SNPS_PER_CHROM * SPACING_BP}>\n")
        samples = "\t".join(f"SAMPLE_{i:03d}" for i in range(N_SAMPLES))
        fh.write(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{samples}\n")
        gt_map = {0: "0/0", 1: "0/1", 2: "1/1"}
        for j in range(geno.shape[1]):
            chrom = f"chr{j // SNPS_PER_CHROM + 1:02d}"
            pos = (j % SNPS_PER_CHROM + 1) * SPACING_BP
            gts = "\t".join(gt_map[int(g)] for g in geno[:, j])
            fh.write(f"{chrom}\t{pos}\t{chrom}_{pos}\tA\tT\t.\tPASS\t.\tGT\t{gts}\n")
    return path


def _write_phenotypes(trait1, trait2, path):
    """Write the phenotype CSV with sample IDs matching the VCF."""
    pd.DataFrame({
        "SampleID": [f"SAMPLE_{i:03d}" for i in range(N_SAMPLES)],
        "Trait1": np.round(trait1, 4),
        "Trait2": np.round(trait2, 4),
    }).to_csv(path, index=False, lineterminator="\n")   # LF on every OS
    return path


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else OUT_DIR
    rng = np.random.default_rng(SEED)
    print("Generating synthetic genotypes...")
    geno = _make_genotypes(rng)
    print(f"  {geno.shape[1]} SNPs x {geno.shape[0]} lines across {N_CHROMS} chromosomes")
    print("Simulating phenotypes...")
    trait1, trait2 = _make_phenotypes(rng, geno)
    qtl_snp = f"chr{QTL_CHROM:02d}_{(QTL_IDX + 1) * SPACING_BP}"
    print(f"  QTL at {qtl_snp} (h2 ~ {H2_TARGET})")
    vcf_path = os.path.join(out_dir, "example.vcf.gz")
    pheno_path = os.path.join(out_dir, "example_pheno.csv")
    print(f"Writing {vcf_path} ...")
    _write_vcf(geno, vcf_path)
    print(f"Writing {pheno_path} ...")
    _write_phenotypes(trait1, trait2, pheno_path)
    print("\nDone! Run TRACE with:")
    print(f"  trace-gwas --vcf {vcf_path} --pheno {pheno_path} --trait Trait1 --output examples/output/")


if __name__ == "__main__":
    main()
