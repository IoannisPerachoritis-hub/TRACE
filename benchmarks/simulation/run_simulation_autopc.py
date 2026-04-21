"""Run TRACE LOCO MLM with auto-PC selection on simulated phenotypes.

Two-phase approach:
  Phase 1 (calibration): Scan k=0..10 PCs on first 2 reps per scenario,
           select best k via band strategy [0.95, 1.05].
  Phase 2 (production):  Run remaining reps at the consensus k per scenario.

Results saved to platform_loco_autopc/ alongside existing platform_loco/.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.simulation.run_simulation_platform import (
    precompute_kinship,
    run_platform_gwas,
)
from gwas.plotting import compute_lambda_gc

# ── Band strategy parameters (same as gwas/models.py) ──
BAND_LO = 0.95
BAND_HI = 1.05
PARSIMONY_TOL = 0.02
MAX_PCS = 10


def select_best_k_band(lambda_scan: list[dict]) -> int:
    """Apply band strategy to a lambda scan. Returns selected k."""
    valid = [(r["k"], r["lambda_gc"], abs(r["lambda_gc"] - 1.0))
             for r in lambda_scan if np.isfinite(r["lambda_gc"])]
    if not valid:
        return 0

    # In-band: smallest k where BAND_LO <= lambda <= BAND_HI
    in_band = [(k, lam, delta) for k, lam, delta in valid
               if BAND_LO <= lam <= BAND_HI]
    if in_band:
        return min(in_band, key=lambda x: x[0])[0]

    # Fallback: closest-to-1 with adaptive parsimony tolerance
    best_delta = min(delta for _, _, delta in valid)
    tol = max(PARSIMONY_TOL, best_delta * 0.15)
    near_best = [(k, lam, delta) for k, lam, delta in valid
                 if delta <= best_delta + tol]
    return min(near_best, key=lambda x: x[0])[0]


def run_pc_scan(
    geno, snp_map, sample_ids, y, precomputed, max_pcs=MAX_PCS,
) -> list[dict]:
    """Scan k=0..max_pcs, return list of {k, lambda_gc} dicts."""
    scan = []
    for k in range(0, max_pcs + 1):
        try:
            results, _ = run_platform_gwas(
                geno, snp_map, sample_ids, y, precomputed,
                n_pcs=k, use_loco=True,
            )
            lam = compute_lambda_gc(results["PValue"].values, trim=False)
        except Exception:
            lam = float("nan")
        scan.append({"k": k, "lambda_gc": round(lam, 4)})
    return scan


def consensus_k(scan_list: list[list[dict]]) -> int:
    """Pick consensus k by averaging lambda profiles across calibration reps.

    Averages lambda_GC at each k, then applies band strategy to the
    averaged profile. Using the mean profile avoids sensitivity to a
    single rep and still picks the smallest k in-band.
    """
    # Average lambda at each k across reps
    from collections import defaultdict
    lambda_sums = defaultdict(list)
    for scan in scan_list:
        for r in scan:
            if np.isfinite(r["lambda_gc"]):
                lambda_sums[r["k"]].append(r["lambda_gc"])

    avg_scan = []
    for k in sorted(lambda_sums.keys()):
        vals = lambda_sums[k]
        avg_scan.append({"k": k, "lambda_gc": sum(vals) / len(vals)})

    return select_best_k_band(avg_scan)


def main(n_reps: int = 20, n_calibration: int = 2):
    sim_data_dir = ROOT / "benchmarks" / "simulation" / "sim_data"

    print("Loading genotype data...")
    geno = np.load(sim_data_dir / "geno_matrix.npy")
    snp_map = pd.read_csv(sim_data_dir / "snp_map.csv")
    sample_ids = np.load(sim_data_dir / "sample_ids.npy")
    print(f"  {geno.shape[0]} samples, {geno.shape[1]} SNPs")

    print(f"Pre-computing kinship with max_pcs={MAX_PCS}...")
    precomputed = precompute_kinship(geno, snp_map, sample_ids, n_pcs=MAX_PCS)
    print(f"  Precompute timing: {precomputed['timing']}")

    # Discover scenario groups (h2_XXX_qYYY)
    scenario_groups = {}
    for d in sorted(sim_data_dir.glob("h2_*/rep_*")):
        if d.is_dir() and (d / "phenotype.csv").exists():
            scenario = d.parent.name  # e.g., h2_080_q005
            scenario_groups.setdefault(scenario, []).append(d)

    # Limit to first n_reps
    for sc in scenario_groups:
        scenario_groups[sc] = sorted(scenario_groups[sc])[:n_reps]

    print(f"\n{len(scenario_groups)} scenarios, {n_reps} reps each")
    print(f"Phase 1: calibration scan on first {n_calibration} reps per scenario")
    print(f"Phase 2: production run on all {n_reps} reps at consensus k\n")

    # ── Phase 1: Calibration ──
    scenario_consensus = {}
    t_start = time.perf_counter()

    for sc_name, rep_dirs in sorted(scenario_groups.items()):
        cal_reps = rep_dirs[:n_calibration]
        k_list = []
        scan_list = []

        for rd in cal_reps:
            pheno_df = pd.read_csv(rd / "phenotype.csv")
            y = pheno_df["SimTrait"].values.astype(np.float32)

            scan = run_pc_scan(geno, snp_map, sample_ids, y, precomputed)
            best_k = select_best_k_band(scan)
            k_list.append(best_k)
            scan_list.append(scan)

            lam_at_k = next(r["lambda_gc"] for r in scan if r["k"] == best_k)
            print(f"  {sc_name}/{rd.name}: scan done, best k={best_k} "
                  f"(lambda={lam_at_k})")

        ck = consensus_k(scan_list)
        scenario_consensus[sc_name] = {
            "consensus_k": ck,
            "calibration_ks": k_list,
            "calibration_scans": scan_list,
        }
        agree = "agree" if len(set(k_list)) == 1 else "DISAGREE"
        print(f"  -> {sc_name}: consensus k={ck} "
              f"(reps selected {k_list}, {agree})")

    cal_elapsed = time.perf_counter() - t_start
    print(f"\nPhase 1 complete in {cal_elapsed:.0f}s")

    # Print summary table
    print(f"\n{'Scenario':<16} {'Consensus k':>12} {'Cal k0':>8} {'Cal k1':>8}")
    print("-" * 48)
    for sc_name, info in sorted(scenario_consensus.items()):
        ks = info["calibration_ks"]
        print(f"{sc_name:<16} {info['consensus_k']:>12} "
              f"{ks[0]:>8} {ks[1] if len(ks) > 1 else '-':>8}")

    # ── Phase 2: Production ──
    print(f"\n{'='*60}")
    print("Phase 2: Running all reps at consensus k")
    print(f"{'='*60}")

    t_prod = time.perf_counter()
    total_done = 0
    total_runs = sum(len(rds) for rds in scenario_groups.values())

    for sc_name, rep_dirs in sorted(scenario_groups.items()):
        ck = scenario_consensus[sc_name]["consensus_k"]
        scans = scenario_consensus[sc_name]["calibration_scans"]

        for i, rd in enumerate(rep_dirs):
            out_dir = rd / "platform_loco_autopc"
            out_dir.mkdir(parents=True, exist_ok=True)

            # For calibration reps, reuse the scan results at consensus k
            if i < n_calibration:
                scan = scans[i]
                # Re-run at consensus k to get the actual results CSV
                pheno_df = pd.read_csv(rd / "phenotype.csv")
                y = pheno_df["SimTrait"].values.astype(np.float32)
                results, timing = run_platform_gwas(
                    geno, snp_map, sample_ids, y, precomputed,
                    n_pcs=ck, use_loco=True,
                )
            else:
                pheno_df = pd.read_csv(rd / "phenotype.csv")
                y = pheno_df["SimTrait"].values.astype(np.float32)
                results, timing = run_platform_gwas(
                    geno, snp_map, sample_ids, y, precomputed,
                    n_pcs=ck, use_loco=True,
                )
                scan = None

            # Save results
            results[["SNP", "Chr", "Pos", "PValue"]].to_csv(
                out_dir / "results.csv", index=False,
            )

            # Save PC selection metadata
            meta = {
                "selected_k": ck,
                "source": "calibration_consensus",
                "scenario": sc_name,
            }
            if scan is not None:
                meta["lambda_scan"] = scan
            with open(out_dir / "pc_selection.json", "w") as f:
                json.dump(meta, f, indent=2)

            total_done += 1

        elapsed = time.perf_counter() - t_prod
        rate = total_done / elapsed if elapsed > 0 else 0
        remaining = (total_runs - total_done) / rate if rate > 0 else 0
        print(f"  {sc_name} (k={ck}): {len(rep_dirs)} reps done "
              f"[{total_done}/{total_runs}, ~{remaining:.0f}s left]")

    total_elapsed = time.perf_counter() - t_start
    print(f"\nAll done in {total_elapsed:.0f}s "
          f"({total_done} production runs)")

    # Save consensus summary
    summary = {sc: {"consensus_k": info["consensus_k"],
                     "calibration_ks": info["calibration_ks"]}
               for sc, info in scenario_consensus.items()}
    summary_path = sim_data_dir / "autopc_consensus_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Consensus summary saved to {summary_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run TRACE LOCO with auto-PC selection on simulations")
    parser.add_argument("--n-reps", type=int, default=20,
                        help="Number of replicates per scenario (default: 20)")
    parser.add_argument("--n-cal", type=int, default=2,
                        help="Calibration reps for PC scan (default: 2)")
    args = parser.parse_args()
    main(n_reps=args.n_reps, n_calibration=args.n_cal)
