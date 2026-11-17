# -*- coding: utf-8 -*-
"""GRM LD-pruning ablation: the Platform-LOCO-unpruned and Platform-Global-unpruned rows of Table S11b.

GAPIT3 and rMVP build their GRM on all markers; TRACE LD-prunes it (r2_thresh 0.2, the shipped
default). This reruns TRACE MLM with the pruning bypassed (r2_thresh 1.0; ALT chromosomes stay
excluded, as in the published runs), in both conditions (LOCO, Global), at the published PC
counts (published_pc_counts.resolve_k), and compares each with GAPIT3 and rMVP: Spearman rho on
-log10 p and lambda_GC, with the formulas and loaders of loco_sensitivity/compare_loco.py.

    python benchmarks/prune_ablation.py [--data-root DIR] [--k-set band] [--rmvp-pcs-twice]
    python benchmarks/prune_ablation.py --write ...    also write the rows

First it checks that the pruned run reproduces the published Platform-LOCO table (rho >= 0.999).
Input: benchmarks/qc_data/<run>/, results/gapit/<run>/, results/rmvp/<run>/. Output with --write:
benchmarks/loco_sensitivity/table_s11_unpruned.csv (table_s11_unpruned_pcs_twice.csv), written whole, with
the unrounded rho and lambda_GC beside the rounded ones. The rho of the published run (made on
the earlier QC data) are printed beside the new ones.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmarks._options import options, runs_in_root  # noqa: E402
OPT = options()
ROOT = OPT.root   # the data root: this repository unless --data-root is given

from benchmarks.rerun_trace_mlm import load_qc_dataset
from benchmarks.published_pc_counts import resolve_k
from benchmarks.loco_sensitivity.compare_loco import (
    lambda_gc,
    load_gapit_results,
    load_rmvp_results,
    load_platform_loco_results,
)
from gwas.kinship import (
    _standardize_geno_for_grm,
    _build_grm_from_Z,
    _ld_prune_for_grm_by_chr_bp,
    _build_loco_kernels_impl,
)
from gwas.models import _run_gwas_impl
from gwas.utils import PhenoData
from pysnptools.kernelreader import KernelData as PSKernelData

CSV_PATH = ROOT / "benchmarks" / "loco_sensitivity" / f"table_s11_unpruned{OPT.suffix}.csv"

# panel run-name -> trait column (CSV dataset order preserved for grouping the appended rows)
PANELS = {
    "tomato_locule_number": "locule_number",
    "tomato_weight_g": "weight_g",
    "pepper_FWe": "FWe",
    "pepper_BX": "BX",
}
PANELS = runs_in_root(PANELS)   # the panels whose QC folder is in the data root

# The 16 rho of the published run (on the earlier QC data), printed beside the new ones.
PUBLISHED_RHO = {
    ("tomato_locule_number", "Platform-LOCO-unpruned", "GAPIT3"): 0.6622,
    ("tomato_locule_number", "Platform-LOCO-unpruned", "rMVP"): 0.7213,
    ("tomato_locule_number", "Platform-Global-unpruned", "GAPIT3"): 0.8501,
    ("tomato_locule_number", "Platform-Global-unpruned", "rMVP"): 0.9532,
    ("tomato_weight_g", "Platform-LOCO-unpruned", "GAPIT3"): 0.6229,
    ("tomato_weight_g", "Platform-LOCO-unpruned", "rMVP"): 0.6545,
    ("tomato_weight_g", "Platform-Global-unpruned", "GAPIT3"): 0.8152,
    ("tomato_weight_g", "Platform-Global-unpruned", "rMVP"): 0.9462,
    ("pepper_FWe", "Platform-LOCO-unpruned", "GAPIT3"): 0.9156,
    ("pepper_FWe", "Platform-LOCO-unpruned", "rMVP"): 0.8599,
    ("pepper_FWe", "Platform-Global-unpruned", "GAPIT3"): 0.9626,
    ("pepper_FWe", "Platform-Global-unpruned", "rMVP"): 0.8730,
    ("pepper_BX", "Platform-LOCO-unpruned", "GAPIT3"): 0.8937,
    ("pepper_BX", "Platform-LOCO-unpruned", "rMVP"): 0.8955,
    ("pepper_BX", "Platform-Global-unpruned", "GAPIT3"): 0.9621,
    ("pepper_BX", "Platform-Global-unpruned", "rMVP"): 0.9642,
}


def _sp(df):
    """Standardize a results frame to SNP/PValue columns.

    The standardized competitor CSVs and TRACE gwas_df already use these names.
    """
    if "SNP" in df.columns and "PValue" in df.columns:
        return df[["SNP", "PValue"]].copy()
    scol = next((c for c in df.columns if "snp" in c.lower()), df.columns[0])
    pcol = next((c for c in df.columns if "pval" in c.lower() or c in ("P", "P.value")), None)
    return df[[scol, pcol]].rename(columns={scol: "SNP", pcol: "PValue"})


def _rho(df_a, df_b, unrounded=False):  # unrounded=True: also the unrounded rho
    """Spearman rho on -log10(p) -- replicates compare_loco.compare_tools exactly."""
    m = df_a.merge(df_b, on="SNP", suffixes=("_1", "_2")).dropna()
    if len(m) < 100:
        return (float("nan"), len(m)) + ((float("nan"),) if unrounded else ())
    l1 = -np.log10(m["PValue_1"].clip(lower=1e-300))
    l2 = -np.log10(m["PValue_2"].clip(lower=1e-300))
    rho, _ = stats.spearmanr(l1, l2)
    return (round(float(rho), 4), len(m)) + ((float(rho),) if unrounded else ())


def run_condition(panel, r2, loco, n_pcs):
    """Replicated TRACE MLM run; returns (SNP/PValue frame, n_markers_in_GRM)."""
    geno, y, sid, chroms_str, chroms_num, positions, iid, _trait = load_qc_dataset(panel)
    Zp, keep = _ld_prune_for_grm_by_chr_bp(chroms_str, positions, geno, r2_thresh=r2, return_mask=True)
    chroms_pruned = chroms_str[keep]
    Zg = _standardize_geno_for_grm(Zp)
    Kb = _build_grm_from_Z(Zg)
    eigvals, eigvecs = eigh(Kb)
    pcs_full = eigvecs[:, np.argsort(eigvals)[::-1][:10]]
    if loco:
        K0, K_by_chr, _ = _build_loco_kernels_impl(
            iid=iid, Z_grm=Zg, chroms_grm=chroms_pruned, K_base=Kb
        )
    else:
        K0, K_by_chr = PSKernelData(iid=iid, val=Kb), {}
    gwas_df = _run_gwas_impl(
        geno_imputed=geno, y=y, pcs_full=pcs_full, n_pcs=n_pcs, sid=sid,
        positions=positions, chroms=chroms_str, chroms_num=chroms_num, iid=iid,
        _K0=K0, _K_by_chr=K_by_chr, pheno_reader=PhenoData(iid=iid, val=y),
    )
    return _sp(gwas_df), int(keep.sum())


def validate_pruned_replication(panels=tuple(p for p in ("tomato_locule_number", "pepper_FWe")
                                             if p in PANELS)):
    """Prove the replicated harness reproduces the published PRUNED LOCO runs (rho -> 1.000).

    A drift here means run_condition / resolve_k no longer reproduce the published p-values,
    so the ablation numbers cannot be trusted. Fail loud.
    """
    print("=== VALIDATION: replicated PRUNED LOCO reproduces on-disk published (expect rho~1.000) ===")
    for panel in panels:
        k = resolve_k(panel, OPT.k_set)
        mine, nk = run_condition(panel, r2=0.2, loco=True, n_pcs=k)
        pub = load_platform_loco_results(panel)
        if pub is None:
            raise SystemExit(f"published platform_GWAS for {panel} not found (qc_data absent)")
        rep, n = _rho(mine, _sp(pub))
        ok = rep is not None and rep >= 0.999
        print(f"  {panel:<22} k={k} grm_markers={nk}  vs published rho={rep} (n={n})  {'OK' if ok else 'DRIFT'}")
        if not ok:
            raise SystemExit(f"pruned-replication drift on {panel}: rho={rep} < 0.999 -- STOP.")


def compute_ablation_rows():
    """Run the 16 unpruned conditions; return a list of CSV row dicts + a drift report."""
    competitors = [("GAPIT3", load_gapit_results), ("rMVP", load_rmvp_results)]
    conditions = [("Platform-LOCO-unpruned", True), ("Platform-Global-unpruned", False)]
    rows, drift = [], []
    print("\n=== ABLATION: 16 unpruned rho (r2_thresh=1.0, published per-panel k) ===")
    print(f"{'panel':<22}{'condition':<26}{'competitor':<8}{'rho':>9}{'old':>9}{'lam1':>8}")
    for panel in PANELS:
        k = resolve_k(panel, OPT.k_set)
        comp_frames = {name: _sp(load(panel)) for name, load in competitors}
        comp_lams_raw = {name: lambda_gc(comp_frames[name]["PValue"].dropna().values)
                         for name, _ in competitors}   # unrounded; rounded below as before
        comp_lams = {name: round(comp_lams_raw[name], 4) for name, _ in competitors}
        for cond_name, is_loco in conditions:
            trace, nk = run_condition(panel, r2=1.0, loco=is_loco, n_pcs=k)
            lam1_raw = lambda_gc(trace["PValue"].dropna().values)   # unrounded
            lam1 = round(lam1_raw, 4)
            for comp_name, _ in competitors:
                rho, n, rho_raw = _rho(trace, comp_frames[comp_name], unrounded=True)
                exp = PUBLISHED_RHO[(panel, cond_name, comp_name)]
                if rho is None or abs(rho - exp) > 1e-4:
                    drift.append((panel, cond_name, comp_name, rho, exp))
                rows.append({
                    "dataset": panel,
                    "tool_1": cond_name,
                    "tool_2": comp_name,
                    "spearman_rho": rho,
                    "n_snps": n,
                    "lambda_gc_1": lam1,
                    "lambda_gc_2": comp_lams[comp_name],
                    # the unrounded values beside the rounded ones (the tables round once)
                    "spearman_rho_unrounded": rho_raw,
                    "lambda_gc_1_unrounded": lam1_raw,
                    "lambda_gc_2_unrounded": comp_lams_raw[comp_name],
                })
                print(f"{panel:<22}{cond_name:<26}{comp_name:<8}{rho:>9}{exp:>9}{lam1:>8}"
                      f"  grm={nk}")
    return rows, drift


def append_rows(rows):
    """Write the rows to their own file (CSV_PATH), whole."""
    cols = ["dataset", "tool_1", "tool_2", "spearman_rho", "n_snps", "lambda_gc_1", "lambda_gc_2",
            "spearman_rho_unrounded", "lambda_gc_1_unrounded", "lambda_gc_2_unrounded"]  # as compare_loco.py's columns
    # the rows go to their own file, written whole (a rerun cannot keep old rows)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=cols).to_csv(CSV_PATH, index=False)
    print(f"\nWrote {len(rows)} rows to {CSV_PATH}.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="write the rows to table_s11_unpruned.csv (default: dry-run)")
    args = ap.parse_args()

    validate_pruned_replication()
    rows, drift = compute_ablation_rows()

    # the published rho came from the earlier QC data: a difference is printed, not a stop
    if drift:
        print("\nOld (published run, earlier QC data) against new, rows differing by > 1e-4:")
        for panel, cond, comp, got, exp in drift:
            print(f"    {panel} {cond} vs {comp}: new {got}, old {exp}")
    else:
        print("\nAll regenerated rho equal the published run's within 1e-4.")

    if args.write:
        append_rows(rows)
    else:
        print(f"\n(dry-run; pass --write to write {CSV_PATH.name})")


if __name__ == "__main__":
    main()
