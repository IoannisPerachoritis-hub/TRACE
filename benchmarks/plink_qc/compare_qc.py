"""Compare QC output between TRACE and PLINK2.

Runs PLINK2 on the same VCF files with matched QC parameters,
then compares retained SNP/sample counts.

Prerequisites: PLINK2 must be installed and on PATH.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

DATASETS = {
    "tomato_locule_number": {
        "vcf": ROOT / "data" / "tomato" / "varitome_filtered.vcf",
        "platform_qc": ROOT / "benchmarks" / "qc_data" / "tomato_locule_number",
    },
    "pepper_FWe": {
        "vcf": ROOT / "data" / "pepper" / "G2P-SOL_Pepper_CC_Geno_data_2023.vcf",
        "platform_qc": ROOT / "benchmarks" / "qc_data" / "pepper_FWe",
    },
}


def check_plink2():
    """Check if PLINK2 is available."""
    try:
        result = subprocess.run(
            ["plink2", "--version"], capture_output=True, text=True, timeout=10,
        )
        print(f"PLINK2 found: {result.stdout.strip().split(chr(10))[0]}")
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("WARNING: PLINK2 not found on PATH.")
        print("Download from: https://www.cog-genomics.org/plink/2.0/")
        return False


def run_plink_qc(vcf_path: Path, output_prefix: str, out_dir: Path):
    """Run PLINK2 QC with matched parameters."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / output_prefix

    cmd = [
        "plink2",
        "--vcf", str(vcf_path),
        "--maf", "0.05",
        "--geno", "0.10",
        "--mac", "5",
        "--mind", "0.20",
        "--snps-only", "just-acgt",
        "--max-alleles", "2",
        "--allow-extra-chr",
        "--set-all-var-ids", "@:#:\\$r:\\$a",
        "--make-bed",
        "--out", str(prefix),
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        print(f"  PLINK2 error:\n{result.stderr}")
        return None

    # Parse PLINK2 log for counts
    log_text = result.stdout + "\n" + result.stderr
    n_snps = None
    n_samples = None

    for line in (result.stdout + result.stderr).split("\n"):
        if "variants remaining" in line.lower() or "variants loaded" in line.lower():
            # Try to extract number
            parts = line.strip().split()
            for p in parts:
                if p.isdigit() and int(p) > 100:
                    n_snps = int(p)
                    break
        if "samples" in line.lower() and ("remaining" in line.lower() or "loaded" in line.lower()):
            parts = line.strip().split()
            for p in parts:
                if p.isdigit():
                    n_samples = int(p)
                    break

    # Read BIM/FAM for exact counts
    bim_path = Path(str(prefix) + ".bim")
    fam_path = Path(str(prefix) + ".fam")

    if bim_path.exists():
        bim = pd.read_csv(bim_path, sep="\t", header=None,
                          names=["Chr", "SNP", "cM", "Pos", "A1", "A2"])
        n_snps = len(bim)
    if fam_path.exists():
        fam = pd.read_csv(fam_path, sep=r"\s+", header=None,
                          names=["FID", "IID", "Father", "Mother", "Sex", "Pheno"])
        n_samples = len(fam)

    return {"n_snps": n_snps, "n_samples": n_samples}


def _canon_chr(raw: str) -> str:
    """Canonicalize chromosome name to a bare number/letter."""
    import re
    s = str(raw).strip()
    # Solanaceae patterns: SL2.50ch01, SL4.0ch01, Ca5, chr03, etc.
    m = re.search(r"ch0*(\d+)", s, re.IGNORECASE)
    if m:
        return m.group(1)
    # Already a bare number
    if s.isdigit():
        return s
    return s


def _compute_intersection(platform_dir: Path, bim_path: Path):
    """Compute actual SNP intersection between TRACE QC and PLINK2 BIM.

    Matches on canonicalized (chr, pos) pairs.
    Returns (n_shared, plink_in_trace_pct, trace_in_plink_pct).
    """
    # TRACE SNPs: (Chr, Pos) from QC_snp_map.csv
    snp_map = pd.read_csv(platform_dir / "QC_snp_map.csv")
    trace_loci = set(
        zip(snp_map["Chr"].astype(str), snp_map["Pos"].astype(int))
    )

    # PLINK2 SNPs: (Chr, Pos) from BIM file
    bim = pd.read_csv(
        bim_path, sep="\t", header=None,
        names=["Chr", "SNP", "cM", "Pos", "A1", "A2"],
    )
    plink_loci = set(
        zip(bim["Chr"].map(_canon_chr), bim["Pos"].astype(int))
    )

    shared = trace_loci & plink_loci
    n_shared = len(shared)
    plink_in_trace = round(100 * n_shared / len(plink_loci), 1) if plink_loci else 0
    trace_in_plink = round(100 * n_shared / len(trace_loci), 1) if trace_loci else 0
    return n_shared, plink_in_trace, trace_in_plink


def compare_all():
    """Run PLINK2 on all datasets and compare with platform QC."""
    if not check_plink2():
        print("\nSkipping PLINK QC comparison (PLINK2 not available).")
        print("To install: download from https://www.cog-genomics.org/plink/2.0/")
        return

    out_dir = ROOT / "benchmarks" / "plink_qc" / "results"
    rows = []

    for ds_name, ds_info in DATASETS.items():
        print(f"\n{'='*50}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*50}")

        vcf_path = ds_info["vcf"]
        if not vcf_path.exists():
            print(f"  VCF not found: {vcf_path}")
            continue

        # TRACE counts
        platform_dir = ds_info["platform_qc"]
        platform_geno = pd.read_csv(platform_dir / "QC_genotype_matrix.csv", nrows=1)
        platform_n_snps = len(platform_geno.columns) - 1  # Minus SampleID
        platform_n_samples = len(
            pd.read_csv(platform_dir / "QC_phenotype.csv")
        )

        # PLINK counts
        plink_result = run_plink_qc(vcf_path, ds_name, out_dir)

        if plink_result:
            row = {
                "dataset": ds_name,
                "platform_snps": platform_n_snps,
                "plink_snps": plink_result["n_snps"],
                "platform_samples": platform_n_samples,
                "plink_samples": plink_result["n_samples"],
            }

            # Compute actual intersection on (chr, pos)
            bim_path = out_dir / (ds_name + ".bim")
            if bim_path.exists():
                n_shared, plink_in_trace, trace_in_plink = _compute_intersection(
                    platform_dir, bim_path,
                )
                row["shared_snps"] = n_shared
                row["plink_in_trace_pct"] = plink_in_trace
                row["trace_in_plink_pct"] = trace_in_plink

            rows.append(row)
            print(f"\n  TRACE: {row['platform_snps']:,} SNPs, "
                  f"{row['platform_samples']} samples")
            print(f"  PLINK2:   {row['plink_snps']:,} SNPs, "
                  f"{row['plink_samples']} samples")
            if "shared_snps" in row:
                print(f"  Shared:   {row['shared_snps']:,} SNPs")
                print(f"  PLINK2 in TRACE: {row['plink_in_trace_pct']}%")
                print(f"  TRACE in PLINK2: {row['trace_in_plink_pct']}%")

    if rows:
        df = pd.DataFrame(rows)
        out_path = ROOT / "benchmarks" / "plink_qc" / "qc_concordance.csv"
        df.to_csv(out_path, index=False)
        print(f"\n\nQC concordance saved to {out_path}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    compare_all()
