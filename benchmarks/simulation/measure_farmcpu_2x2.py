"""FarmCPU 2x2 measurement (selection-kinship x final-scan, + Jaccard-carry axis).

Work order: docs/revision/farmcpu_workorder.md.  This is a MEASUREMENT (class b):
it changes no default and overwrites no published output.  It runs eight FarmCPU
configurations on the two committed FarmCPU simulation cells (h2=0.5 and 0.8, 5 QTN,
100 reps) and writes a SEPARATE summary CSV.

The eight configs are the full 2x2 (selection kinship {global, loco} x final scan
{mlm, ols}) crossed with the Jaccard-set-carry fix {F, T}:

    A = global selection + MLM final      (A_F = the published control / gate)
    B = global selection + OLS final
    C = LOCO   selection + OLS final       (C_T = the shipping candidate)
    D = LOCO   selection + MLM final

A_F reproduces the published `farmcpu_loco` numbers (power 0.186 / FDR 0.6131 /
lambdaGC 1.0891 at h2=0.5; 0.316 / 0.9431 / 1.2565 at h2=0.8).  It is the hard gate:
if it misses beyond Monte-Carlo error the harness has drifted and nothing else is
interpretable.

Per config per cell we record power, marker-FDR (+ locus-FDR), lambdaGC, the pseudo-QTN
count (mean + full distribution), and wall-clock.  The evaluation reuses
`evaluate_simulation.classify_detections` (500 kb window, Bonferroni 0.05/n_snps) and the
per-rep lambdaGC formula from `evaluate_simulation.evaluate_all_scenarios` so the numbers
are directly comparable to `power_fdr_summary_ci.csv`.

Runs are checkpointed per rep to <rep>/farmcpu_2x2_<config>/results.csv, so the job is
resumable and `--eval-only` re-aggregates without recomputing.

Usage (run from the tree whose sim_data you want; sim_data ships only in DEV):
    python benchmarks/simulation/measure_farmcpu_2x2.py               # full 8-config run
    python benchmarks/simulation/measure_farmcpu_2x2.py --configs A_F # gate only
    python benchmarks/simulation/measure_farmcpu_2x2.py --reps 3      # smoke test
    python benchmarks/simulation/measure_farmcpu_2x2.py --eval-only   # re-aggregate
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.simulation.run_simulation_platform import (  # noqa: E402
    precompute_kinship, run_farmcpu_gwas,
)
from benchmarks.simulation.evaluate_simulation import classify_detections  # noqa: E402

# ── The 8 configs. A_F FIRST so the gate is checkable before the rest finish. ──
CONFIGS = {
    "A_F": dict(selection_kinship="global", final_scan="mlm", carry_validated_set=False),
    "A_T": dict(selection_kinship="global", final_scan="mlm", carry_validated_set=True),
    "B_F": dict(selection_kinship="global", final_scan="ols", carry_validated_set=False),
    "B_T": dict(selection_kinship="global", final_scan="ols", carry_validated_set=True),
    "C_F": dict(selection_kinship="loco", final_scan="ols", carry_validated_set=False),
    "C_T": dict(selection_kinship="loco", final_scan="ols", carry_validated_set=True),
    "D_F": dict(selection_kinship="loco", final_scan="mlm", carry_validated_set=False),
    "D_T": dict(selection_kinship="loco", final_scan="mlm", carry_validated_set=True),
}

CELLS = ["h2_050_q005", "h2_080_q005"]

# Published farmcpu_loco values (power_fdr_summary_ci.csv rows 14/28) = the A_F gate.
GATE = {
    "h2_050_q005": dict(power=0.186, fdr=0.6131, lambda_gc=1.0891),
    "h2_080_q005": dict(power=0.316, fdr=0.9431, lambda_gc=1.2565),
}


def _lambda_gc(pvals):
    """Genomic-control lambda, matching evaluate_simulation.evaluate_all_scenarios."""
    pvals = np.asarray(pvals, dtype=float)
    pvals = pvals[(pvals > 0) & (pvals < 1)]
    if len(pvals) <= 10:
        return np.nan
    chisq = stats.chi2.ppf(1 - pvals, df=1)
    return float(np.median(chisq) / stats.chi2.ppf(0.5, df=1))


def _ci(vals):
    """Mean, SE, and normal-approx 95% CI (matching summarize_with_ci), clipped [0,1]."""
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    n = len(vals)
    mean = float(vals.mean()) if n else np.nan
    se = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    lo = max(0.0, mean - 1.96 * se)
    hi = min(1.0, mean + 1.96 * se)
    return mean, se, lo, hi


def _load_sim(sim_data_dir):
    geno = np.load(sim_data_dir / "geno_matrix.npy")
    snp_map = pd.read_csv(sim_data_dir / "snp_map.csv")
    sample_ids = np.load(sim_data_dir / "sample_ids.npy", allow_pickle=True)
    return geno, snp_map, sample_ids


def _determinism_check(old_results, new_results, rep_dir):
    """Compare a recomputed results.csv to the prior one (work order §3).

    The §1/§4 instrumentation is additive logging and must not move a p-value, so
    recomputation has to reproduce the prior p-values to float-summation noise.
    Returns a one-row dict; the caller enforces the gate.
    """
    o = old_results[["SNP", "PValue"]].rename(columns={"PValue": "p_old"})
    n = new_results[["SNP", "PValue"]].rename(columns={"PValue": "p_new"})
    m = o.merge(n, on="SNP", how="outer")
    po = m["p_old"].to_numpy(dtype=float)
    pn = m["p_new"].to_numpy(dtype=float)
    dp = np.abs(po - pn)
    finite = np.isfinite(dp)
    denom = np.minimum(np.abs(po), np.abs(pn))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(denom > 0, dp / denom, 0.0)
    # "top-100 ranking unchanged": compare the sorted top-100 p-VALUES, NOT the
    # SNP set. The set can differ by tie members at float precision (e.g. 21 SNPs
    # sharing the rank-100 p-value; 1e-16 noise reorders the tie) without the
    # p-value profile changing -- that is not a drift. A real change moves a
    # top-100 p-value and is caught here (and by max_rel_dp).
    top_old = np.sort(old_results["PValue"].to_numpy(dtype=float))[:100]
    top_new = np.sort(new_results["PValue"].to_numpy(dtype=float))[:100]
    top100_match = bool(
        len(top_old) == len(top_new)
        and np.allclose(top_old, top_new, rtol=1e-6, atol=0.0, equal_nan=True))
    return {
        "cell": rep_dir.parent.name,
        "rep": int(rep_dir.name.split("_")[-1]),
        "n_snps": int(len(m)),
        "max_abs_dp": float(np.nanmax(dp[finite])) if finite.any() else 0.0,
        "max_rel_dp": float(np.nanmax(rel[finite])) if finite.any() else 0.0,
        "n_differing": int((dp[finite] > 0).sum()),
        "top100_pvals_match": top100_match,
    }


def _run_one(rep_dir, config_name, cfg, geno, snp_map, sample_ids, precomputed,
             n_pcs, force, det_path=None):
    """Run (or load cached) one config on one rep. Returns (results_df, timing)."""
    out_dir = rep_dir / f"farmcpu_2x2_{config_name}"
    res_path = out_dir / "results.csv"
    tim_path = out_dir / "timing.json"
    if res_path.exists() and tim_path.exists() and not force:
        timing = json.loads(tim_path.read_text())
        if "break_site" in timing:           # fully instrumented (last payload key) -> reuse
            return pd.read_csv(res_path), timing
        # else: fall through and recompute so the full §1/§4 payload is written

    # §3 determinism gate: capture the prior p-values BEFORE recompute + overwrite.
    old_results = pd.read_csv(res_path) if res_path.exists() else None

    pheno_df = pd.read_csv(rep_dir / "phenotype.csv")
    y = pheno_df["SimTrait"].values.astype(np.float32)
    results, timing = run_farmcpu_gwas(
        geno, snp_map, sample_ids, y, precomputed,
        n_pcs=n_pcs, use_loco=True,  # 2x2 fixes use_loco=True; the axes are the knobs
        final_scan=cfg["final_scan"],
        selection_kinship=cfg["selection_kinship"],
        carry_validated_set=cfg["carry_validated_set"],
    )
    # §3: additive logging is not allowed to move a p-value. The recompute must
    # reproduce the prior results.csv. Log the check (incrementally, so it
    # survives a mid-run stop) and STOP loudly on drift.
    if old_results is not None and det_path is not None:
        chk = _determinism_check(old_results, results, rep_dir)
        pd.DataFrame([chk]).to_csv(
            det_path, mode="a", header=not det_path.exists(), index=False)
        if chk["max_rel_dp"] > 1e-6 or not chk["top100_pvals_match"]:
            raise RuntimeError(
                f"DETERMINISM GATE FAILED at {chk['cell']}/rep_{chk['rep']:03d}: "
                f"max_abs_dp={chk['max_abs_dp']:.3e} max_rel_dp={chk['max_rel_dp']:.3e} "
                f"top100_pvals_match={chk['top100_pvals_match']} -- STOP (work order §3)."
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    results[["SNP", "Chr", "Pos", "PValue"]].to_csv(res_path, index=False)
    tim_path.write_text(json.dumps(timing))
    return results, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-data-dir", default=None,
                    help="Defaults to <repo>/benchmarks/simulation/sim_data")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS),
                    choices=list(CONFIGS))
    ap.add_argument("--cells", nargs="+", default=CELLS,
                    help="Scenario cell dir names under --sim-data-dir "
                         "(default: the two published cells). Existence is "
                         "validated below; choices are not hardcoded so new "
                         "cells (e.g. h2_050_q015) are accepted.")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--n-pcs", type=int, default=2)
    ap.add_argument("--eval-only", action="store_true",
                    help="Do not run; aggregate existing per-rep results.csv only.")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if a cached results.csv exists.")
    ap.add_argument("--out", default=None,
                    help="Defaults to <sim_summary>/farmcpu_2x2_measurement.csv")
    args = ap.parse_args()

    sim_data_dir = Path(args.sim_data_dir) if args.sim_data_dir else \
        ROOT / "benchmarks" / "simulation" / "sim_data"
    # `choices=CELLS` was dropped so new cells are accepted; validate that each
    # requested cell has a scenario directory instead.
    for _c in args.cells:
        if not (sim_data_dir / _c).is_dir():
            ap.error(f"--cells: no scenario directory {_c!r} under {sim_data_dir}")
    sim_summary = ROOT / "benchmarks" / "simulation" / "sim_summary"
    sim_summary.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else sim_summary / "farmcpu_2x2_measurement.csv"
    perrep_path = out_path.with_name(out_path.stem + "_perrep.csv")

    print(f"sim_data: {sim_data_dir}")
    print(f"configs : {args.configs}")
    print(f"cells   : {args.cells}   reps: {args.reps}   n_pcs: {args.n_pcs}")

    geno, snp_map, sample_ids = _load_sim(sim_data_dir)
    n_snps = geno.shape[1]
    threshold = 0.05 / n_snps
    print(f"geno    : {geno.shape[0]} samples x {n_snps} SNPs; Bonferroni thr={threshold:.3e}")

    if not args.eval_only:
        precomputed = precompute_kinship(geno, snp_map, sample_ids, n_pcs=args.n_pcs)
        print(f"precompute: {precomputed['timing']}")
    else:
        precomputed = None

    perrep_rows = []
    det_path = sim_summary / "farmcpu_determinism_check.csv"
    if not args.eval_only:
        det_path.unlink(missing_ok=True)  # fresh determinism log per run (§3)
    for config_name in args.configs:  # A_F first -> gate checkable early
        cfg = CONFIGS[config_name]
        for cell in args.cells:
            cell_dir = sim_data_dir / cell
            t_cell = time.perf_counter()
            done = 0
            for rep in range(args.reps):
                rep_dir = cell_dir / f"rep_{rep:03d}"
                if not (rep_dir / "phenotype.csv").exists():
                    continue
                truth = json.loads((rep_dir / "truth.json").read_text())
                if args.eval_only:
                    res_path = rep_dir / f"farmcpu_2x2_{config_name}" / "results.csv"
                    tim_path = rep_dir / f"farmcpu_2x2_{config_name}" / "timing.json"
                    if not res_path.exists():
                        continue
                    results = pd.read_csv(res_path)
                    timing = json.loads(tim_path.read_text()) if tim_path.exists() else {}
                else:
                    results, timing = _run_one(
                        rep_dir, config_name, cfg, geno, snp_map, sample_ids,
                        precomputed, args.n_pcs, args.force,
                        det_path=det_path,
                    )
                cls = classify_detections(results, truth, snp_map, threshold,
                                          window_kb=500)
                perrep_rows.append({
                    "config": config_name, "cell": cell, "rep": rep,
                    "selection_kinship": cfg["selection_kinship"],
                    "final_scan": cfg["final_scan"],
                    "carry_validated_set": cfg["carry_validated_set"],
                    "power": cls["power"], "fdr": cls["fdr"],
                    "fdr_locus": cls["fdr_locus"],
                    "n_significant": cls["n_significant"],
                    "lambda_gc": _lambda_gc(results["PValue"].values),
                    "n_pseudo_qtns": timing.get("n_pseudo_qtns", np.nan),
                    "n_iterations": timing.get("n_iterations", np.nan),
                    "converged": timing.get("converged", None),
                    "association_sec": timing.get("association_sec", np.nan),
                })
                done += 1
            dt = time.perf_counter() - t_cell
            print(f"  [{config_name} / {cell}] {done} reps in {dt:.0f}s")

    if det_path.exists():
        _det = pd.read_csv(det_path)
        print(f"\ndeterminism (§3) -> {det_path}  ({len(_det)} recomputed reps; "
              f"worst max_abs_dp={_det['max_abs_dp'].max():.3e}, "
              f"worst max_rel_dp={_det['max_rel_dp'].max():.3e}, "
              f"all top100 pvals match={bool(_det['top100_pvals_match'].all())})")

    perrep = pd.DataFrame(perrep_rows)
    perrep.to_csv(perrep_path, index=False)
    print(f"\nper-rep -> {perrep_path}  ({len(perrep)} rows)")

    # ── Aggregate per (config, cell) ─────────────────────────────────────
    summary_rows = []
    for (config_name, cell), grp in perrep.groupby(["config", "cell"], sort=False):
        p_mean, p_se, p_lo, p_hi = _ci(grp["power"])
        f_mean, f_se, f_lo, f_hi = _ci(grp["fdr"])
        fl_mean, _, _, _ = _ci(grp["fdr_locus"])
        pq = grp["n_pseudo_qtns"].dropna().astype(float)
        summary_rows.append({
            "config": config_name, "cell": cell,
            "selection_kinship": grp["selection_kinship"].iloc[0],
            "final_scan": grp["final_scan"].iloc[0],
            "carry_validated_set": bool(grp["carry_validated_set"].iloc[0]),
            "n_reps": len(grp),
            "power_mean": round(p_mean, 4), "power_se": round(p_se, 4),
            "power_ci_lo": round(p_lo, 4), "power_ci_hi": round(p_hi, 4),
            "fdr_mean": round(f_mean, 4), "fdr_se": round(f_se, 4),
            "fdr_ci_lo": round(f_lo, 4), "fdr_ci_hi": round(f_hi, 4),
            "fdr_locus_mean": round(fl_mean, 4),
            "lambda_gc_mean": round(float(grp["lambda_gc"].mean()), 4),
            "n_pseudo_qtns_mean": round(float(pq.mean()), 3) if len(pq) else np.nan,
            "n_pseudo_qtns_min": int(pq.min()) if len(pq) else -1,
            "n_pseudo_qtns_max": int(pq.max()) if len(pq) else -1,
            "n_pseudo_qtns_median": float(pq.median()) if len(pq) else np.nan,
            "wall_clock_sec_mean": round(float(grp["association_sec"].mean()), 2),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_path, index=False)
    print(f"summary -> {out_path}  ({len(summary)} rows)")

    # ── Gate: A_F vs published (reported FIRST) ──────────────────────────
    print("\n" + "=" * 68)
    print("ARM A(F) REPRODUCTION GATE  (published farmcpu_loco vs measured A_F)")
    print("=" * 68)
    gate_ok = True
    a_f = summary[summary["config"] == "A_F"]
    if a_f.empty:
        print("  A_F not in this run; gate not evaluated.")
    else:
        for cell in args.cells:
            row = a_f[a_f["cell"] == cell]
            if row.empty:
                continue
            row = row.iloc[0]
            g = GATE.get(cell)
            if g is None:
                continue  # new cell -> no published gate value (work order §4)
            for metric, meas in (("power", row["power_mean"]),
                                 ("fdr", row["fdr_mean"]),
                                 ("lambda_gc", row["lambda_gc_mean"])):
                pub = g[metric]
                d = abs(meas - pub)
                flag = "OK" if d < 0.02 else "**DRIFT**"
                if d >= 0.02:
                    gate_ok = False
                print(f"  {cell:14s} {metric:10s} published={pub:.4f} "
                      f"measured={meas:.4f}  |d|={d:.4f}  {flag}")
    print("=" * 68)
    print("GATE:", "PASS" if gate_ok else "FAIL -- STOP AND REPORT (harness drifted)")
    print("=" * 68)


if __name__ == "__main__":
    main()
