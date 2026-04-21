"""Re-run TRACE MLM GWAS with global kinship (--no-loco) for tomato datasets.

Generates GWAS result ZIPs in benchmarks/loco_sensitivity/<run>/ using
global kinship (no LOCO) with k=0 PCs, matching the deflation guard
selection. These are used by compare_loco.py for the LOCO ablation table.
"""
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.rerun_trace_mlm import load_qc_dataset
from gwas.kinship import (
    _standardize_geno_for_grm,
    _build_grm_from_Z,
    _ld_prune_for_grm_by_chr_bp,
)
from gwas.models import _run_gwas_impl
from gwas.utils import PhenoData
from pysnptools.kernelreader import KernelData as PSKernelData

LOCO_DIR = ROOT / "benchmarks" / "loco_sensitivity"

# Only tomato datasets need regeneration (pepper k unchanged)
RUNS = {
    "tomato_weight_g": "weight_g",
    "tomato_locule_number": "locule_number",
}

N_PCS = 0  # Matching deflation guard selection


def run_global_mlm(run_name, trait_name):
    """Run TRACE MLM with global kinship (no LOCO), k=0 PCs."""
    print(f"\n{'='*60}")
    print(f"  {run_name} ({trait_name}) -- GLOBAL kinship, k={N_PCS}")
    print(f"{'='*60}")

    geno, y, sid, chroms_str, chroms_num, positions, iid, trait_col = load_qc_dataset(run_name)
    n, m = geno.shape
    print(f"  Loaded: {n} samples, {m} SNPs")

    # Build GRM (global — no LOCO)
    Z_pruned, keep_mask = _ld_prune_for_grm_by_chr_bp(
        chroms_str, positions, geno, return_mask=True,
    )
    Z_grm = _standardize_geno_for_grm(Z_pruned)
    K_base = _build_grm_from_Z(Z_grm)

    eigvals, eigvecs = eigh(K_base)
    idx_sort = np.argsort(eigvals)[::-1]
    pcs_full = eigvecs[:, idx_sort[:10]]

    pheno_reader = PhenoData(iid=iid, val=y)

    # Wrap K_base in PSKernelData (required by FaST-LMM single_snp)
    K0 = PSKernelData(iid=iid, val=K_base)

    # Run global MLM (K0=global GRM, K_by_chr={} → no LOCO)
    gwas_df = _run_gwas_impl(
        geno_imputed=geno,
        y=y,
        pcs_full=pcs_full,
        n_pcs=N_PCS,
        sid=sid,
        positions=positions,
        chroms=chroms_str,
        chroms_num=chroms_num,
        iid=iid,
        _K0=K0,
        _K_by_chr={},
        pheno_reader=pheno_reader,
    )

    print(f"  Results: {len(gwas_df)} SNPs, min p = {gwas_df['PValue'].min():.2e}")

    # Save as ZIP in loco_sensitivity directory (format expected by compare_loco.py)
    out_dir = LOCO_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    zip_name = f"GWAS_varitome_phenotypes__{trait_name}__{timestamp}.zip"
    zip_path = out_dir / zip_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        csv_content = gwas_df.to_csv(index=False)
        zf.writestr(f"tables/GWAS_{trait_name}.csv", csv_content)

    print(f"  Saved: {zip_path}")
    return gwas_df


if __name__ == "__main__":
    for run_name, trait_name in RUNS.items():
        run_global_mlm(run_name, trait_name)
    print("\nGlobal GWAS complete.")
