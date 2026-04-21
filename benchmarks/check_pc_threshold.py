"""Quick check: does band PC strategy select different PCs than elbow?

Loads QC data for all 4 benchmark datasets, runs auto_select_pcs() with
strategy="band", and prints the lambda profile + selected k.
Compare to old elbow-selected values:
  - Pepper FWe: k=3, lam=1.034
  - Pepper BX: k=0, lam=1.113
  - Tomato weight_g: k=0, lam=0.915
  - Tomato locule_number: k=2, lam=0.727
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gwas.utils import PhenoData, _mean_impute_cols
from gwas.kinship import (
    _build_grm_from_Z,
    _build_loco_kernels_impl,
    _compute_pcs_full_impl,
    _ld_prune_for_grm_by_chr_bp,
    _standardize_geno_for_grm,
)
from gwas.models import auto_select_pcs
from annotation import canon_chr

QC_DIR = ROOT / "benchmarks" / "qc_data"

RUNS = {
    "pepper_FWe": {"trait": "FWe"},
    "pepper_BX": {"trait": "BX"},
    "tomato_weight_g": {"trait": "weight_g"},
    "tomato_locule_number": {"trait": "locule_number"},
}

OLD_RESULTS = {
    "pepper_FWe": {"k": 3, "lambda": 1.034},
    "pepper_BX": {"k": 0, "lambda": 1.113},
    "tomato_weight_g": {"k": 0, "lambda": 0.915},
    "tomato_locule_number": {"k": 2, "lambda": 0.727},
}


def check_single(run_name: str, trait: str):
    """Load QC data and run auto_select_pcs with band strategy."""
    qc = QC_DIR / run_name
    geno_df = pd.read_csv(qc / "QC_genotype_matrix.csv", index_col=0)
    snp_map = pd.read_csv(qc / "QC_snp_map.csv")
    pheno_df = pd.read_csv(qc / "QC_phenotype.csv")

    # Align samples
    common = sorted(set(geno_df.index) & set(pheno_df["SampleID"]))
    geno_df = geno_df.loc[common]
    pheno_df = pheno_df.set_index("SampleID").loc[common].reset_index()

    geno_imputed = _mean_impute_cols(geno_df.values.astype(np.float32))
    sid = np.array(geno_df.columns, dtype=str)
    n_samples, n_snps = geno_imputed.shape

    # SNP metadata
    snp_order = {s: i for i, s in enumerate(snp_map["SNP_ID"].values)}
    idx_order = [snp_order[s] for s in sid]
    chroms_raw = snp_map["Chr"].values[idx_order].astype(str)
    positions = snp_map["Pos"].values[idx_order].astype(int)
    chroms_canon = np.array([canon_chr(c) for c in chroms_raw], dtype=str)

    # Filter non-numeric chromosomes
    numeric_mask = np.array([c.isdigit() for c in chroms_canon])
    if not numeric_mask.all():
        geno_imputed = geno_imputed[:, numeric_mask]
        sid = sid[numeric_mask]
        chroms_raw = chroms_raw[numeric_mask]
        positions = positions[numeric_mask]
        chroms_canon = chroms_canon[numeric_mask]

    chroms = chroms_raw
    chroms_num = chroms_canon.astype(int)
    iid = np.array([[s, s] for s in common], dtype=str)
    y = pheno_df[trait].values.astype(float)

    # GRM standardization + LD pruning
    Z_std = _standardize_geno_for_grm(geno_imputed)
    Z_pruned, keep = _ld_prune_for_grm_by_chr_bp(
        chroms_canon, positions, Z_std,
        r2_thresh=0.2, window_bp=500_000, step_bp=100_000,
        return_mask=True,
    )
    var_ok = np.nanvar(Z_pruned, axis=0) > 0
    Z_grm = Z_pruned[:, var_ok].astype(np.float32)
    chroms_grm = chroms_canon[keep][var_ok]

    K_base = _build_grm_from_Z(Z_grm)
    pcs_full, _eigvals = _compute_pcs_full_impl(Z_grm, max_pcs=10)

    # Run auto_select_pcs with threshold
    df = auto_select_pcs(
        geno_imputed=geno_imputed, y=y, sid=sid,
        chroms=chroms, chroms_num=chroms_num, positions=positions,
        iid=iid, Z_grm=Z_grm, chroms_grm=chroms_grm,
        K_base=K_base, pcs_full=pcs_full,
        max_pcs=10, strategy="band",
    )
    return df


def main():
    print("=" * 70)
    print("  PC Band Check: Does band select different PCs?")
    print("=" * 70)

    any_changed = False
    for run_name, info in RUNS.items():
        old = OLD_RESULTS[run_name]
        print(f"\n  {run_name} (trait={info['trait']})")
        print(f"  Old elbow: k={old['k']}, lam={old['lambda']:.4f}")

        df = check_single(run_name, info["trait"])

        # Print lambda profile
        for _, row in df.iterrows():
            marker = "*" if row["recommended"] == "\u2605" else row["recommended"]
            lam = row["lambda_gc"]
            k = int(row["n_pcs"])
            lam_str = f"{lam:.4f}" if np.isfinite(lam) else "NaN"
            le_105 = "<=1.05" if np.isfinite(lam) and lam <= 1.05 else ""
            print(f"    k={k}: lam={lam_str} {le_105} {marker}")

        rec = df[df["recommended"] == "\u2605"]
        if len(rec) > 0:
            new_k = int(rec.iloc[0]["n_pcs"])
            new_lam = rec.iloc[0]["lambda_gc"]
            changed = "CHANGED" if new_k != old["k"] else "same"
            if new_k != old["k"]:
                any_changed = True
            print(f"  -> Band: k={new_k}, lam={new_lam:.4f} [{changed}]")

    print(f"\n{'=' * 70}")
    if any_changed:
        print("  RESULT: PC counts CHANGED -> full re-run needed (Phases 2-4)")
    else:
        print("  RESULT: PC counts unchanged -> skip to Phase 5 (figures only)")
    print("=" * 70)


if __name__ == "__main__":
    main()
