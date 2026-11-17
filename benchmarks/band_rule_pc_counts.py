"""The PC count the strict band rule behind the paper's PC counts chooses on the corrected QC data.

The rule (smallest k with 0.95 <= lambda_GC <= 1.05; commit 03095b7; _pc_band_select.select_k_strict_band)
is run on the full lambda_GC scan, k = 0..10: LOCO MLM at each k (_pc_band_select.auto_select_pcs,
max_pcs=10, strategy "band", early_stop=False; untrimmed lambda_GC rounded to 4 decimals), with the
LD-pruned GRM and the top 10 PCs built as rerun_trace_mlm.py builds them. The later, edge-tolerant
variant's pick (auto_select_pcs with early stopping) is given beside it. On the corrected data the
rule gives FWe 3, BX 3, weight 0, locule 0 (published_pc_counts.BAND_RULE_K): the k of the side
rows of Tables S4 and S11.

    python benchmarks/band_rule_pc_counts.py [--data-root DIR]

Input: benchmarks/qc_data/<run>/ (make_qc_data.sh). Output: benchmarks/band_rule_pc_counts.csv in
the data root, one row per run and k: lambda_gc, delta_from_1 = |lambda_gc - 1|, the scan's
status, and whether each rule picks that k (strict_pick, edge_tolerant_pick).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks._options import options  # noqa: E402

OPT = options()
ROOT = OPT.root   # the data root: this repository unless --data-root is given

from benchmarks._pc_band_select import auto_select_pcs, select_k_strict_band  # noqa: E402
from benchmarks.rerun_trace_mlm import RUNS, load_qc_dataset  # noqa: E402
from gwas.kinship import (  # noqa: E402
    _build_grm_from_Z,
    _ld_prune_for_grm_by_chr_bp,
    _standardize_geno_for_grm,
)

STAR = "★"


def starred_k(df):
    rec = df[df["recommended"] == STAR]
    return int(rec["n_pcs"].values[0]) if len(rec) > 0 else 0


def one_run(run_name):
    geno, y, sid, chroms_str, chroms_num, positions, iid, trait_col = load_qc_dataset(run_name)
    n, m = geno.shape
    print(f"\n=== {run_name}: {n} samples, {m} SNPs", flush=True)
    # the GRM and PCs as rerun_trace_mlm.py builds them
    Z_pruned, keep_mask = _ld_prune_for_grm_by_chr_bp(chroms_str, positions, geno, return_mask=True)
    chroms_pruned = chroms_str[keep_mask]
    Z_grm = _standardize_geno_for_grm(Z_pruned)
    K_base = _build_grm_from_Z(Z_grm)
    eigvals, eigvecs = eigh(K_base)
    idx_sort = np.argsort(eigvals)[::-1]
    pcs_full = eigvecs[:, idx_sort[:10]]
    common = dict(geno_imputed=geno, y=y, sid=sid, chroms=chroms_str, chroms_num=chroms_num,
                  positions=positions, iid=iid, Z_grm=Z_grm, chroms_grm=chroms_pruned,
                  K_base=K_base, pcs_full=pcs_full, max_pcs=10, strategy="band")
    t0 = time.perf_counter()
    early = auto_select_pcs(**common)                       # the edge-tolerant variant
    full = auto_select_pcs(**common, early_stop=False)      # the full scan, k = 0..10
    k_edge, k_edge_full = starred_k(early), starred_k(full)
    if k_edge != k_edge_full:
        raise SystemExit(f"{run_name}: the edge-tolerant calls pick different k ({k_edge}, {k_edge_full})")
    k_strict, how = select_k_strict_band(full["lambda_gc"].tolist())
    print(full[["n_pcs", "lambda_gc", "delta_from_1", "status"]].to_string(index=False))
    print(f"  {run_name}: strict band rule k = {k_strict} ({how}); edge-tolerant k = {k_edge}; "
          f"{time.perf_counter() - t0:.1f} s", flush=True)
    return full, k_strict, k_edge


def main():
    rows, picks = [], {}
    for run in RUNS:
        full, k_strict, k_edge = one_run(run)
        picks[run] = (k_strict, k_edge)
        rows += [dict(run=run, n_pcs=r["n_pcs"], lambda_gc=r["lambda_gc"], delta_from_1=r["delta_from_1"],
                      status=r["status"], strict_pick=(r["n_pcs"] == k_strict),
                      edge_tolerant_pick=(r["n_pcs"] == k_edge))
                 for r in full.drop(columns=["recommended"]).to_dict(orient="records")]
    out = ROOT / "benchmarks" / "band_rule_pc_counts.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("\nstrict band rule k:", {r: p[0] for r, p in picks.items()},
          "| edge-tolerant k:", {r: p[1] for r, p in picks.items()})
    print(f"written: {out}")


if __name__ == "__main__":
    main()
