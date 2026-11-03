"""Planted-causal multi-trait data for the multi-trait tests (CLI and One-Click).

Rebuilds the case commit 822610f describes ("three traits, each with a causal SNP
on a different chromosome"); that data was never committed. Each causal SNP sits in
a short run of SNPs in strong LD, so every trait forms an LD block, runs the
haplotype test and gets a heatmap. The traits lack different samples (multi-trait
criterion B), and a constant trait fails phenotype QC (criterion C).
"""
import numpy as np
import pandas as pd

CHROMS = ("1", "2", "3")
N_SAMPLES = 120
SNPS_PER_CHR = 60
SPACING = 1000
# trait -> (chromosome, causal SNP index on that chromosome)
CAUSAL = {"T_chr1": ("1", 20), "T_chr2": ("2", 30), "T_chr3": ("3", 40)}
CONSTANT_TRAIT = "T_constant"
MISSING = {"T_chr2": slice(0, 10), "T_chr3": slice(-20, None)}  # samples without a value


def _genotypes(rng):
    """Samples x SNPs dosages (0/1/2): around each causal SNP a run of 6 SNPs in
    strong LD (copies of one column with ~3% re-drawn), elsewhere independent."""
    n, m = N_SAMPLES, SNPS_PER_CHR * len(CHROMS)
    G = rng.binomial(2, 0.35, size=(n, m)).astype(np.int8)
    for c, j in CAUSAL.values():
        base = CHROMS.index(c) * SNPS_PER_CHR + j
        src = G[:, base].copy()
        for k in range(-2, 4):
            col = src.copy()
            flip = rng.random(n) < 0.03
            col[flip] = rng.integers(0, 3, flip.sum())
            G[:, base + k] = col
        G[:, base] = src
    return G


def build(seed=11):
    """Return ``(vcf_text, pheno_df, info)``. ``pheno_df`` has an ``Accession``
    column and one column per trait in CAUSAL plus CONSTANT_TRAIT; ``info`` maps
    each planted trait to its chromosome, causal SNP ID and number of samples."""
    rng = np.random.default_rng(seed)
    G = _genotypes(rng)
    samples = [f"S{i:03d}" for i in range(N_SAMPLES)]
    lines = [
        "##fileformat=VCFv4.2",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples),
    ]
    gt = {0: "0/0", 1: "0/1", 2: "1/1"}
    for ci, c in enumerate(CHROMS):
        for j in range(SNPS_PER_CHR):
            col = G[:, ci * SNPS_PER_CHR + j]
            lines.append(f"{c}\t{(j + 1) * SPACING}\tchr{c}_{j}\tA\tG\t.\tPASS\t.\tGT\t"
                         + "\t".join(gt[int(v)] for v in col))
    pheno = pd.DataFrame({"Accession": samples})
    for trait, (c, j) in CAUSAL.items():
        g = G[:, CHROMS.index(c) * SNPS_PER_CHR + j].astype(float)
        pheno[trait] = 1.2 * (g - g.mean()) + rng.normal(0.0, 1.0, N_SAMPLES)
    for trait, rows in MISSING.items():
        pheno.loc[pheno.index[rows], trait] = np.nan
    pheno[CONSTANT_TRAIT] = 5.0
    info = {t: {"chrom": c, "snp": f"chr{c}_{j}", "n": int(pheno[t].notna().sum())}
            for t, (c, j) in CAUSAL.items()}
    return "\n".join(lines) + "\n", pheno, info


def write(directory, seed=11):
    """Write ``planted.vcf`` and ``planted_pheno.csv`` into ``directory``; return
    ``(vcf_path, pheno_path, info)``."""
    vcf_text, pheno, info = build(seed)
    vcf = directory / "planted.vcf"
    vcf.write_text(vcf_text)
    csv = directory / "planted_pheno.csv"
    pheno.to_csv(csv, index=False)
    return vcf, csv, info
