"""TRACE-FarmCPU on the four real datasets: the TRACE side of Table S10 and Figure S6.

Runs TRACE's FarmCPU (the E_F_nogate2_ols arm: the OLS final scan with Step 5, the published
Table S10 run's settings) on each run's QC data, at the PC counts of the published Table S10
run, FWe 4, BX 5, weight 10, locule 10 (PUBLISHED_S10_K; fixed, not chosen by a picker). The PCs
are the first k of a 10-component PCA computed with the exact SVD (svd_solver="full"), so a
rerun gives the same table. The other gate-free arms of the original study stay selectable.

    python benchmarks/rerun_trace_farmcpu.py [--data-root DIR] [--arm ARM ...] [--runs RUN ...]

Input: benchmarks/qc_data/<run>/ (make_qc_data.sh). Output in
results/farmcpu_gatefree_concordance/<run>/: platform_GWAS_FarmCPU_<arm>_<trait>.csv and
pseudo_qtns_<arm>_<trait>.csv, with farmcpu_gatefree_diagnostics.csv and .json (k, its source,
retention, iterations, final lambda_GC).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmarks._options import options, runs_in_root  # noqa: E402
OPT = options()
ROOT = OPT.root   # the data root: this repository unless --data-root is given

from gwas.utils import PhenoData, CovarData, _mean_impute_cols
from gwas.kinship import _build_grm_from_Z, _standardize_geno_for_grm
from gwas.models import run_farmcpu
# R1.4: select_best_pc_from_lambdas was removed from the shipped tool; recovered for reproducibility.
from benchmarks._pc_band_select import select_best_pc_from_lambdas
from gwas.plotting import compute_lambda_gc
from annotation import canon_chr

QC_DIR = ROOT / "benchmarks" / "qc_data"
DEFAULT_OUT = ROOT / "benchmarks" / "results" / "farmcpu_gatefree_concordance"

RUNS = {
    "pepper_FWe": {"trait": "FWe"},
    "pepper_BX": {"trait": "BX"},
    "tomato_weight_g": {"trait": "weight_g"},
    "tomato_locule_number": {"trait": "locule_number"},
}

# Gate-free arms (mirror measure_farmcpu_2x2.CONFIGS). final_scan is PER ARM and
# is threaded into BOTH the PC-scan and the final run (a mismatch would select
# PCs under one model and report p-values under another). NOTE: arm flags
# thread into the library run_farmcpu, whose defaults are now the shipped OLS
# Step-5 model -- so A_F pins its step5_* OFF explicitly to remain the historical
# bare-FarmCPU MLM arm (it was silently inheriting the new Step-5 defaults). The
# E_F_* arms below still partially inherit those defaults; flagged, out of scope.
ARMS = {
    "A_F": dict(final_scan="mlm", step5_reml_bins=False, step5_substitution=False,
                step5_skip_validation=False, step5_prune_in_loop=False,
                step5_reselect=False, pool_cap=None),
    "E_F_nogate": dict(step5_reml_bins=True, step5_substitution=True,
                       step5_skip_validation=True, pool_cap=100, final_scan="mlm"),
    "E_F_nogate2": dict(step5_reml_bins=True, step5_substitution=True,
                        step5_skip_validation=True, pool_cap=100,
                        step5_reselect=True, step5_prune_in_loop=True, final_scan="mlm"),
    # CANONICAL config: OLS (fixed-effect) final scan, NO kinship -- matches
    # GAPIT3's algorithm end-to-end. With the retained pseudo-QTNs (bounded by
    # sqrt(n/log10 n) = 9-12) already absorbing structure, an MLM final scan (GRM
    # on top) is double-correcting.
    "E_F_nogate2_ols": dict(step5_reml_bins=True, step5_substitution=True,
                            step5_skip_validation=True, pool_cap=100,
                            step5_reselect=True, step5_prune_in_loop=True,
                            final_scan="ols"),
}

MAX_PCS_SCAN = 10
BAND_LO = 0.95

# TRACE-FarmCPU's PC counts, fixed at those of the published Table S10 run: the E_F_nogate2_ols
# selected_k of the 19 Aug 2026 run, read from its diagnostics file (farmcpu_gatefree_diagnostics.json,
# sha256 390a76b2c6d2f1315b09f9b6bb76a73a8bc54f210611a4e251c19f429067bb37). The picker
# (select_best_pc_from_lambdas) is not used.
PUBLISHED_S10_K = {"pepper_FWe": 4, "pepper_BX": 5, "tomato_weight_g": 10, "tomato_locule_number": 10}


def load_qc_data(run_name):
    qc = QC_DIR / run_name
    geno_df = pd.read_csv(qc / "QC_genotype_matrix.csv", index_col=0)
    snp_map = pd.read_csv(qc / "QC_snp_map.csv")
    pheno_df = pd.read_csv(qc / "QC_phenotype.csv")
    return geno_df, snp_map, pheno_df


def _scan_pcs(geno_imputed, sid, chroms, chroms_num, positions, iid,
              pheno_reader, K0, pcs_full, arm_flags,
              band_lo=BAND_LO, band_hi=1.05, parsimony_tol=0.02):
    """Scan k=0..max (ALL k, no early-stop) for the full lambda trace, then
    select k via the production band strategy + deflation guard.

    Returns (best_k, lambdas, guard_fired).
    """
    max_k = min(MAX_PCS_SCAN, pcs_full.shape[1])
    lambdas = []
    for k in range(max_k + 1):
        covar_k = None
        if k > 0:
            covar_k = CovarData(iid=iid, val=pcs_full[:, :k],
                                names=[f"PC{i+1}" for i in range(k)])
        try:
            df_k, _, _ = run_farmcpu(
                geno_imputed=geno_imputed, sid=sid, chroms=chroms,
                chroms_num=chroms_num, positions=positions, iid=iid,
                pheno_reader=pheno_reader, K0=K0, covar_reader=covar_k,
                verbose=False, **arm_flags)
            lam = compute_lambda_gc(df_k["PValue"].values, trim=False)
        except Exception:
            lam = np.nan
        lambdas.append(lam)
        print(f"    k={k}: lambda={lam:.4f}" if np.isfinite(lam)
              else f"    k={k}: lambda=NaN")

    best_k = select_best_pc_from_lambdas(
        lambdas, strategy="band", band_lo=band_lo, band_hi=band_hi,
        parsimony_tolerance=parsimony_tol)
    guard_fired = bool(np.isfinite(lambdas[0]) and lambdas[0] < band_lo)
    # any_in_band matters for OLS: if NO k lands in [0.95,1.05], selection falls
    # back to "least inflated" -- typically k=10, a materially different covariate
    # structure from the 0-1 PCs the MLM arms pick.
    any_in_band = bool(any(np.isfinite(v) and band_lo <= v <= band_hi for v in lambdas))
    return best_k, lambdas, guard_fired, any_in_band


def run_single(run_name, arm, out_dir):
    """Run one arm on one dataset; write results to out_dir; return diagnostics."""
    info = RUNS[run_name]
    trait = info["trait"]
    arm_flags = ARMS[arm]
    print(f"\n{'='*60}\n  {run_name} (trait={trait})  arm={arm}\n{'='*60}")

    geno_df, snp_map, pheno_df = load_qc_data(run_name)

    # Align samples (identical to the cache-generation logic)
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
    positions_raw = snp_map["Pos"].values[idx_order].astype(int)
    chroms_canon = np.array([canon_chr(c) for c in chroms_raw], dtype=str)

    # Filter out ALT / unplaced chromosomes (non-numeric)
    numeric_mask = np.array([c.isdigit() for c in chroms_canon])
    if not numeric_mask.all():
        geno_imputed = geno_imputed[:, numeric_mask]
        sid = sid[numeric_mask]
        chroms_raw = chroms_raw[numeric_mask]
        positions_raw = positions_raw[numeric_mask]
        chroms_canon = chroms_canon[numeric_mask]
        n_snps = geno_imputed.shape[1]

    chroms = chroms_raw
    positions = positions_raw
    chroms_num = chroms_canon.astype(int)

    iid = np.array([[s, s] for s in common], dtype=str)
    y = pheno_df[trait].values.astype(float)
    pheno_reader = PhenoData(iid=iid, val=y.reshape(-1, 1))

    # Kinship (GRM)
    Z_grm = _standardize_geno_for_grm(geno_imputed)
    K = _build_grm_from_Z(Z_grm)
    from pysnptools.kernelreader import KernelData
    K0 = KernelData(iid=iid, val=K)

    # PCA — compute max PCs, then auto-select (per arm)
    from sklearn.decomposition import PCA
    max_k = min(MAX_PCS_SCAN, n_samples - 1, n_snps)
    # svd_solver="full": the exact SVD, so the PCs (and every result) are the same on a rerun.
    # The published run used sklearn's default, which picks a randomized solver for these
    # shapes and draws from the unseeded global random state.
    pca = PCA(n_components=max_k, svd_solver="full")
    pcs_full = pca.fit_transform(geno_imputed)

    # k fixed at the PC count of the published Table S10 run: no PC scan, no picker
    n_pcs = PUBLISHED_S10_K[run_name]
    print(f"  n={n_samples}, p={n_snps}; k={n_pcs}, fixed: the PC count of the published Table S10 run")

    covar_reader = None
    if n_pcs > 0:
        covar_reader = CovarData(iid=iid, val=pcs_full[:, :n_pcs],
                                 names=[f"PC{i+1}" for i in range(n_pcs)])

    # Final FarmCPU run with the selected PCs. final_scan comes from arm_flags
    # (per arm) -- NOT hardcoded -- so it matches the PC scan's final_scan.
    farmcpu_df, pseudo_qtn_table, conv = run_farmcpu(
        geno_imputed=geno_imputed, sid=sid, chroms=chroms,
        chroms_num=chroms_num, positions=positions, iid=iid,
        pheno_reader=pheno_reader, K0=K0, covar_reader=covar_reader,
        verbose=False, **arm_flags)
    lam_final = compute_lambda_gc(farmcpu_df["PValue"].values)
    print(f"  final: retention={conv['n_pseudo_qtns']} break={conv.get('break_site')} "
          f"n_iter={conv['n_iterations']} lambda_GC={lam_final:.4f}")

    # Save to the SAFE out dir (never qc_data/)
    out_run = out_dir / run_name
    out_run.mkdir(parents=True, exist_ok=True)
    out_path = out_run / f"platform_GWAS_FarmCPU_{arm}_{trait}.csv"
    farmcpu_df.to_csv(out_path, index=False)
    # Emit the pseudo-QTN identities (to test the self-conditioning hypothesis:
    # are the arm's Bonferroni-significant markers themselves pseudo-QTNs?).
    (out_run / f"pseudo_qtns_{arm}_{trait}.csv").write_text(
        pseudo_qtn_table.to_csv(index=False))
    print(f"  saved: {out_path}  (+ {len(pseudo_qtn_table)} pseudo-QTNs)")

    return {
        "run": run_name, "trait": trait, "arm": arm,
        "n_samples": int(n_samples), "n_snps": int(n_snps),
        "selected_k": int(n_pcs), "guard_fired": None,   # k fixed, no scan (see k_source)
        "any_k_in_band": None,
        "lambda_k0": None,
        "lambda_trace": None,
        "k_source": "fixed: the PC count of the published Table S10 run (farmcpu_gatefree_diagnostics.json "
                    "of 19 Aug 2026, arm E_F_nogate2_ols, field selected_k)",
        "retention": int(conv["n_pseudo_qtns"]),
        "break_site": conv.get("break_site"),
        "n_iterations": int(conv["n_iterations"]),
        "converged": bool(conv["converged"]),
        "lambda_gc_final": round(float(lam_final), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", default=["E_F_nogate2_ols"], choices=list(ARMS))
    ap.add_argument("--runs", nargs="+", default=list(RUNS), choices=list(RUNS))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    # SAFETY: never write into qc_data/ (the published A_F cache = Table S10 source).
    if out_dir == QC_DIR.resolve() or QC_DIR.resolve() in out_dir.parents:
        ap.error(f"--out-dir must NOT be under {QC_DIR} (the published A_F cache)")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"arms={args.arm}  runs={args.runs}  out_dir={out_dir}")
    diags = []
    for arm in args.arm:
        for run_name in args.runs:
            diags.append(run_single(run_name, arm, out_dir))

    # Merge (upsert by run+arm) so a partial run never clobbers other arms' rows.
    diag_path = out_dir / "farmcpu_gatefree_diagnostics.csv"
    json_path = out_dir / "farmcpu_gatefree_diagnostics.json"
    existing = []
    if json_path.exists():
        try:
            existing = json.loads(json_path.read_text())
        except Exception:
            existing = []
    new_keys = {(r["run"], r["arm"]) for r in diags}
    merged = [r for r in existing if (r["run"], r["arm"]) not in new_keys] + diags
    pd.DataFrame(merged).to_csv(diag_path, index=False)
    json_path.write_text(json.dumps(merged, indent=2))
    diag_df = pd.DataFrame(diags)  # print only THIS run's rows
    print(f"\ndiagnostics -> {diag_path}  ({len(merged)} total rows)")
    _cols = ["run", "arm", "selected_k", "guard_fired", "lambda_k0", "retention",
             "break_site", "n_iterations", "lambda_gc_final"]
    print(diag_df[_cols].to_string(index=False))


if __name__ == "__main__":
    main()
