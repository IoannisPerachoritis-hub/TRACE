"""Extract QC'd data from platform ZIP files for competitor tool benchmarking.

Reads QC_genotype_matrix.csv, QC_snp_map.csv, QC_phenotype.csv from each
platform result ZIP and writes them to benchmarks/qc_data/<run_name>/.
"""
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

RUNS = {
    "pepper_FWe": "results/pepper_FWe",
    "pepper_BX": "results/pepper_BX",
    "tomato_weight_g": "results/tomato_weight_g",
    "tomato_locule_number": "results/tomato_locule_number",
}

QC_FILES = ["QC_genotype_matrix", "QC_snp_map", "QC_phenotype"]

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "benchmarks" / "qc_data"


def extract_run(run_name: str, result_dir: str):
    rdir = ROOT / result_dir
    zips = list(rdir.glob("*.zip"))
    if not zips:
        print(f"  SKIP {run_name}: no ZIP found in {rdir}")
        return
    zpath = max(zips, key=lambda p: p.stat().st_mtime)
    out_dir = OUT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        for qc_prefix in QC_FILES:
            matches = [n for n in names if qc_prefix in n and n.endswith(".csv")]
            if not matches:
                print(f"  WARNING: {qc_prefix} not found in {zpath.name}")
                continue
            fname = matches[0]
            with zf.open(fname) as f:
                df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"))
            out_path = out_dir / f"{qc_prefix}.csv"
            df.to_csv(out_path, index=False)
            print(f"  {run_name}/{qc_prefix}.csv: {df.shape}")

    # Also extract the GWAS results CSV for comparison
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        gwas_csvs = [n for n in names if "GWAS_" in n and n.endswith(".csv")
                     and "QC_" not in n]
        for gc in gwas_csvs:
            with zf.open(gc) as f:
                df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"))
            basename = Path(gc).name
            out_path = out_dir / f"platform_{basename}"
            df.to_csv(out_path, index=False)
            print(f"  {run_name}/platform_{basename}: {df.shape}")


def main():
    print("Extracting QC data from platform ZIPs...")
    for run_name, result_dir in RUNS.items():
        print(f"\n{run_name}:")
        extract_run(run_name, result_dir)
    print("\nDone. QC data written to benchmarks/qc_data/")


if __name__ == "__main__":
    main()
