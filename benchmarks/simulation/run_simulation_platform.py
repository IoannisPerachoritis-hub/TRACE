"""Run TRACE MLM (LOCO and Global) on simulated phenotypes.

Calls _impl functions from gwas/kinship.py and gwas/models.py directly,
bypassing VCF parsing and the CLI.
"""
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from gwas.kinship import (
    _build_grm_from_Z,
    _build_loco_kernels_impl,
    _compute_pcs_full_impl,
    _ld_prune_for_grm_by_chr_bp,
    _standardize_geno_for_grm,
)
from gwas.models import _run_gwas_impl, run_farmcpu
from gwas.utils import CovarData, PhenoData

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger(__name__)


def precompute_kinship(
    geno: np.ndarray,
    snp_map: pd.DataFrame,
    sample_ids: np.ndarray,
    n_pcs: int = 2,
):
    """Pre-compute kinship matrices and PCA (shared across all sims).

    Returns dict with all precomputed objects needed for GWAS.
    """
    chroms = snp_map["Chr"].values.astype(str)
    positions = snp_map["Pos"].values.astype(int)

    t0 = time.perf_counter()

    # Standardize for GRM
    Z = _standardize_geno_for_grm(geno)
    t_std = time.perf_counter() - t0

    # LD prune
    t1 = time.perf_counter()
    Z_pruned, keep_mask = _ld_prune_for_grm_by_chr_bp(
        chroms, positions, Z, r2_thresh=0.2,
        window_bp=500_000, step_bp=100_000, return_mask=True,
    )
    chroms_grm = chroms[keep_mask]
    t_prune = time.perf_counter() - t1

    # Build GRM
    t2 = time.perf_counter()
    K_base = _build_grm_from_Z(Z_pruned)
    t_grm = time.perf_counter() - t2

    # PCA
    t3 = time.perf_counter()
    pcs, _eigvals = _compute_pcs_full_impl(Z_pruned, max_pcs=max(n_pcs, 5))
    t_pca = time.perf_counter() - t3

    # LOCO kernels
    # iid must be (n, 2) for pysnptools
    iid_2d = np.c_[sample_ids, sample_ids].astype(str)

    t4 = time.perf_counter()
    K0, K_by_chr, _ = _build_loco_kernels_impl(
        iid_2d, Z_pruned, chroms_grm, K_base=K_base,
    )
    t_loco = time.perf_counter() - t4

    # Global kinship (same kernel for all chromosomes)
    K_global = {ch: K0 for ch in K_by_chr}

    total = time.perf_counter() - t0

    timing = {
        "standardize_sec": round(t_std, 2),
        "ld_prune_sec": round(t_prune, 2),
        "build_grm_sec": round(t_grm, 2),
        "compute_pca_sec": round(t_pca, 2),
        "build_loco_sec": round(t_loco, 2),
        "total_precompute_sec": round(total, 2),
    }

    return {
        "K0": K0,
        "K_by_chr_loco": K_by_chr,
        "K_by_chr_global": K_global,
        "pcs": pcs,
        "timing": timing,
    }


def run_platform_gwas(
    geno: np.ndarray,
    snp_map: pd.DataFrame,
    sample_ids: np.ndarray,
    y: np.ndarray,
    precomputed: dict,
    n_pcs: int = 2,
    use_loco: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Run TRACE MLM on a single phenotype.

    Returns (gwas_results_df, timing_dict).
    """
    sid = snp_map["SNP_ID"].values.astype(str)
    chroms = snp_map["Chr"].values.astype(str)
    chroms_num = snp_map["Chr"].values.astype(int)
    positions = snp_map["Pos"].values.astype(int)

    iid = np.c_[sample_ids, sample_ids]
    pheno_reader = PhenoData(sample_ids, y)

    K0 = precomputed["K0"]
    K_by_chr = (precomputed["K_by_chr_loco"] if use_loco
                else precomputed["K_by_chr_global"])
    pcs = precomputed["pcs"]

    # Mean-impute genotypes (should already be imputed, but ensure no NaN)
    geno_imp = geno.copy()
    col_means = np.nanmean(geno_imp, axis=0)
    for j in range(geno_imp.shape[1]):
        mask = np.isnan(geno_imp[:, j])
        if mask.any():
            geno_imp[mask, j] = col_means[j]

    t0 = time.perf_counter()
    results = _run_gwas_impl(
        geno_imputed=geno_imp,
        y=y,
        pcs_full=pcs,
        n_pcs=n_pcs,
        sid=sid,
        positions=positions,
        chroms=chroms,
        chroms_num=chroms_num,
        iid=iid,
        _K0=K0,
        _K_by_chr=K_by_chr,
        pheno_reader=pheno_reader,
        trait_name="SimTrait",
    )
    assoc_time = time.perf_counter() - t0

    timing = {"association_sec": round(assoc_time, 2)}
    return results, timing


def run_farmcpu_gwas(
    geno: np.ndarray,
    snp_map: pd.DataFrame,
    sample_ids: np.ndarray,
    y: np.ndarray,
    precomputed: dict,
    n_pcs: int = 2,
    use_loco: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Run FarmCPU on a single phenotype.

    Returns (gwas_results_df, timing_dict).
    """
    sid = snp_map["SNP_ID"].values.astype(str)
    chroms = snp_map["Chr"].values.astype(str)
    chroms_num = snp_map["Chr"].values.astype(int)
    positions = snp_map["Pos"].values.astype(int)

    iid = np.c_[sample_ids, sample_ids]
    pheno_reader = PhenoData(sample_ids, y)

    K0 = precomputed["K0"]
    pcs = precomputed["pcs"]

    # Build CovarData for PCs
    covar_reader = None
    if n_pcs > 0 and pcs is not None:
        covar_reader = CovarData(
            iid=iid,
            val=pcs[:, :n_pcs].astype(np.float64),
            names=[f"PC{i+1}" for i in range(n_pcs)],
        )

    # Mean-impute genotypes (should already be imputed, but ensure no NaN)
    geno_imp = geno.copy()
    col_means = np.nanmean(geno_imp, axis=0)
    for j in range(geno_imp.shape[1]):
        mask = np.isnan(geno_imp[:, j])
        if mask.any():
            geno_imp[mask, j] = col_means[j]

    t0 = time.perf_counter()
    farmcpu_df, _pseudo_qtn_table, convergence_info = run_farmcpu(
        geno_imputed=geno_imp,
        sid=sid,
        chroms=chroms,
        chroms_num=chroms_num,
        positions=positions,
        iid=iid,
        pheno_reader=pheno_reader,
        K0=K0,
        covar_reader=covar_reader,
        verbose=False,
        use_loco=use_loco,
    )
    assoc_time = time.perf_counter() - t0

    timing = {
        "association_sec": round(assoc_time, 2),
        "n_pseudo_qtns": convergence_info["n_pseudo_qtns"],
        "converged": convergence_info["converged"],
    }
    return farmcpu_df, timing


def run_single_scenario(args_tuple):
    """Worker function for parallel execution. Unpacks args tuple."""
    (scenario_dir, geno, snp_map, sample_ids,
     precomputed, n_pcs, use_loco, mode_name) = args_tuple

    pheno_path = scenario_dir / "phenotype.csv"
    if not pheno_path.exists():
        return None

    pheno_df = pd.read_csv(pheno_path)
    y = pheno_df["SimTrait"].values.astype(np.float32)

    try:
        if mode_name.startswith("farmcpu"):
            results, timing = run_farmcpu_gwas(
                geno, snp_map, sample_ids, y, precomputed,
                n_pcs=n_pcs, use_loco=use_loco,
            )
        else:
            results, timing = run_platform_gwas(
                geno, snp_map, sample_ids, y, precomputed,
                n_pcs=n_pcs, use_loco=use_loco,
            )
        # Save standardized results
        out_dir = scenario_dir / mode_name
        out_dir.mkdir(parents=True, exist_ok=True)
        results[["SNP", "Chr", "Pos", "PValue"]].to_csv(
            out_dir / "results.csv", index=False,
        )
        # Save timing
        with open(out_dir / "timing.json", "w") as f:
            json.dump(timing, f)

        return str(scenario_dir.name)
    except Exception as e:
        log.error("Failed %s/%s: %s", scenario_dir.name, mode_name, e)
        return None


def run_all_platform(
    sim_data_dir: str | None = None,
    n_pcs: int = 2,
    modes: tuple[str, ...] = ("platform_loco", "platform_global"),
    max_workers: int = 1,
):
    """Run TRACE MLM on all simulated phenotypes."""
    if sim_data_dir is None:
        sim_data_dir = ROOT / "benchmarks" / "simulation" / "sim_data"
    sim_data_dir = Path(sim_data_dir)

    print("Loading genotype data...")
    geno = np.load(sim_data_dir / "geno_matrix.npy")
    snp_map = pd.read_csv(sim_data_dir / "snp_map.csv")
    sample_ids = np.load(sim_data_dir / "sample_ids.npy")
    print(f"  {geno.shape[0]} samples, {geno.shape[1]} SNPs")

    print("Pre-computing kinship matrices and PCA...")
    precomputed = precompute_kinship(geno, snp_map, sample_ids, n_pcs=n_pcs)
    print(f"  Precompute timing: {precomputed['timing']}")

    # Find all scenario directories
    scenario_dirs = sorted([
        d for d in sim_data_dir.glob("h2_*/rep_*")
        if d.is_dir() and (d / "phenotype.csv").exists()
    ])
    print(f"\nFound {len(scenario_dirs)} scenarios")

    for mode_name in modes:
        use_loco = mode_name in ("platform_loco", "farmcpu_loco")
        print(f"\n{'='*60}")
        print(f"  Running mode: {mode_name} ({'LOCO' if use_loco else 'Global'} kinship)")
        print(f"{'='*60}")

        done = 0
        t_start = time.perf_counter()

        # Sequential execution (FastLMM not always fork-safe)
        for sd in scenario_dirs:
            result = run_single_scenario((
                sd, geno, snp_map, sample_ids,
                precomputed, n_pcs, use_loco, mode_name,
            ))
            done += 1
            if done % 50 == 0 or done == len(scenario_dirs):
                elapsed = time.perf_counter() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (len(scenario_dirs) - done) / rate if rate > 0 else 0
                print(f"  Progress: {done}/{len(scenario_dirs)} "
                      f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

        elapsed = time.perf_counter() - t_start
        print(f"  Completed {done} runs in {elapsed:.1f}s "
              f"({elapsed/done:.1f}s/run)")

    # Save precompute timing for fair runtime comparison
    timing_path = sim_data_dir / "platform_precompute_timing.json"
    with open(timing_path, "w") as f:
        json.dump(precomputed["timing"], f, indent=2)
    print(f"\nPrecompute timing saved to {timing_path}")


def run_null_platform(
    null_data_dir: str | None = None,
    n_pcs: int = 2,
):
    """Run TRACE MLM on null (permuted) phenotypes for type I error calibration."""
    if null_data_dir is None:
        null_data_dir = ROOT / "benchmarks" / "simulation" / "null_data"
    null_data_dir = Path(null_data_dir)
    sim_data_dir = ROOT / "benchmarks" / "simulation" / "sim_data"

    print("Loading genotype data...")
    geno = np.load(sim_data_dir / "geno_matrix.npy")
    snp_map = pd.read_csv(sim_data_dir / "snp_map.csv")
    sample_ids = np.load(sim_data_dir / "sample_ids.npy")

    print("Pre-computing kinship matrices...")
    precomputed = precompute_kinship(geno, snp_map, sample_ids, n_pcs=n_pcs)

    perm_dirs = sorted([
        d for d in null_data_dir.glob("perm_*")
        if d.is_dir() and (d / "phenotype.csv").exists()
    ])
    print(f"Found {len(perm_dirs)} null phenotypes")

    t_start = time.perf_counter()
    for i, pd_dir in enumerate(perm_dirs):
        pheno_df = pd.read_csv(pd_dir / "phenotype.csv")
        y = pheno_df["SimTrait"].values.astype(np.float32)

        # Match samples
        valid = ~np.isnan(y)
        y_valid = y[valid]
        geno_valid = geno[valid]
        ids_valid = sample_ids[valid]

        # Recompute kinship for valid samples if different count
        if valid.sum() != len(sample_ids):
            pc_local = precompute_kinship(geno_valid, snp_map, ids_valid, n_pcs)
        else:
            pc_local = precomputed

        results, _ = run_platform_gwas(
            geno_valid, snp_map, ids_valid, y_valid, pc_local,
            n_pcs=n_pcs, use_loco=True,
        )

        out_dir = pd_dir / "platform_loco"
        out_dir.mkdir(parents=True, exist_ok=True)
        results[["SNP", "Chr", "Pos", "PValue"]].to_csv(
            out_dir / "results.csv", index=False,
        )

        if (i + 1) % 10 == 0 or i + 1 == len(perm_dirs):
            elapsed = time.perf_counter() - t_start
            print(f"  Null progress: {i+1}/{len(perm_dirs)} ({elapsed:.0f}s)")

    print(f"Null calibration complete in {time.perf_counter()-t_start:.0f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sim", "null", "both"], default="both")
    parser.add_argument("--n-pcs", type=int, default=2)
    parser.add_argument("--modes", nargs="+",
                        default=["platform_loco", "platform_global"],
                        help="Modes to run. Options: platform_loco, "
                             "platform_global, farmcpu_loco")
    args = parser.parse_args()

    if args.mode in ("sim", "both"):
        run_all_platform(n_pcs=args.n_pcs, modes=tuple(args.modes))
    if args.mode in ("null", "both"):
        run_null_platform(n_pcs=args.n_pcs)
