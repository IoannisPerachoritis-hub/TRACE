"""Table S8: the haplotype tests of the four LD blocks of tomato locule number.

Runs benchmarks/capture_golden.py with the settings of the golden manifest
(tests/golden/tomato_locule/run_manifest.json: flank, LD decay, permutations, gene model) and the
kinship-aware haplotype test, the default, on the tomato QC data, and writes one row per block:
its interval, lead SNP, SNPs in the block and tested, haplotypes observed and tested, the tested N
(df1 + df2 + 1), the kernel, the null model's h2, F, P (permutation), P (parametric), eta2, the
PCs-only test's eta2 and the block FDR q.

    python benchmarks/table_s8.py [--work DIR]

Input: benchmarks/qc_data/tomato_locule_number/ (QC_genotype_matrix.csv, QC_snp_map.csv,
QC_phenotype.csv and platform_GWAS_locule_number.csv; make_qc_data.sh). Output:
benchmarks/table_s8.csv (full precision) and the table in the paper's layout on the screen;
capture_golden's own files go to --work (default: a temporary folder). The kinship-aware test's
floats move in their last digits with the BLAS thread count: the committed table was made with
12 threads (OPENBLAS_NUM_THREADS=12).
"""
import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def capture(work):
    """capture_golden with the golden manifest's settings and the kinship-aware test; its haplotype
    frame (the call to run_haplotype_block_gwas recorded) and its ld_blocks.csv."""
    manifest = json.loads((REPO / "tests" / "golden" / "tomato_locule" / "run_manifest.json")
                          .read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("capture_golden", REPO / "benchmarks" / "capture_golden.py")
    cap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cap)
    seen = []
    real = cap.run_haplotype_block_gwas

    def rec(*a, **k):
        res = real(*a, **k)
        seen.append(res[0])
        return res

    cap.run_haplotype_block_gwas = rec
    qc = REPO / "benchmarks" / "qc_data" / "tomato_locule_number"
    genes = manifest.get("genes_csv")
    cap.capture_golden(
        qc, qc / "platform_GWAS_locule_number.csv", Path(work),
        flank_kb=manifest["flank_kb"], ld_decay_kb=manifest.get("ld_decay_kb"),
        n_perm=manifest.get("n_perm", 1000), hap_test="kinship",
        genes_csv=(REPO / genes) if genes else None, trait_col=manifest.get("trait_col"))
    if len(seen) != 1:
        raise SystemExit(f"capture_golden ran the haplotype test {len(seen)} times, expected once")
    return seen[0], pd.read_csv(Path(work) / "ld_blocks.csv"), manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None, help="capture_golden's folder (default: a temporary one)")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(a.work or tmp)
        work.mkdir(parents=True, exist_ok=True)
        frame, ld, manifest = capture(work)
    keys = sorted((str(r["Chr"]), int(r["Start"]), int(r["End"])) for _, r in frame.iterrows())
    by_key = {(str(r["Chr"]), int(r["Start"]), int(r["End"])): r for _, r in frame.iterrows()}
    rows = []
    for k in keys:
        r = by_key[k]
        b = ld[(ld["Start (bp)"] == k[1]) & (ld["End (bp)"] == k[2])].iloc[0]
        rows.append({
            "Block": f"Chr{k[0]}:{k[1]:,}-{k[2]:,}", "Start (bp)": k[1], "End (bp)": k[2],
            "Lead SNP": r["lead_snp"],
            "SNPs in block": len([s for s in str(b["SNP_IDs"]).split(",") if s.strip()]),
            "SNPs tested": int(r["n_snps"]), "Haplotypes observed": int(r["n_haplotypes"]),
            "Haplotypes tested": int(r["n_tested_haplotypes"]),
            "Tested N (df1 + df2 + 1)": int(r["df1"]) + int(r["df2"]) + 1,
            "Kernel": r["kinship_model"], "h2_null": float(r["h2_null"]),
            "F": float(r["F_param"]), "P_perm": float(r["P_perm"]),
            "PValue_param": float(r["PValue_param"]), "eta2": float(r["eta2"]),
            "eta2_pcs_only": float(r["eta2_pcs_only"]), "FDR_BH": float(r["FDR_BH"])})
    tab = pd.DataFrame(rows)
    out = REPO / "benchmarks" / "table_s8.csv"
    tab.to_csv(out, index=False)
    md = ["| Block | Lead SNP | SNPs (tested) | Haplotypes observed / tested | N | h² (null) | F | "
          "P (permutation) | P (parametric) | η² | η² (PCs-only test) | FDR q |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        md.append(f"| {r['Block']} | {r['Lead SNP']} | {r['SNPs in block']} ({r['SNPs tested']}) | "
                  f"{r['Haplotypes observed']} / {r['Haplotypes tested']} | "
                  f"{r['Tested N (df1 + df2 + 1)']} | {r['h2_null']:.3f} | {r['F']:.2f} | "
                  f"{r['P_perm']:.4g} | {r['PValue_param']:.2e} | {r['eta2']:.3f} | "
                  f"{r['eta2_pcs_only']:.3f} | {r['FDR_BH']:.4g} |")
    n_perm = int(manifest.get("n_perm", 1000))
    md += ["", f"Blocks: flank {manifest['flank_kb']} kb (2 x the median LD decay), min_snps "
               f"{manifest['min_snps']}; the kinship-aware haplotype test, each block with its "
               f"chromosome's LOCO kernel; {n_perm} permutations, so the smallest P (permutation) is "
               f"1/{n_perm + 1} = {1 / (n_perm + 1):.6f}."]
    print("\n".join(md))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
