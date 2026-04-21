"""Generate simulated phenotypes from real Varitome genotypes.

Draws QTNs from the real tomato SNP matrix, assigns effect sizes,
and scales noise to hit a target h². Saves phenotype CSVs + truth
JSONs. Simulation design follows GAPIT3 (Wang & Zhang 2021).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


def load_qc_data(run_name: str = "tomato_locule_number"):
    """Load QC'd genotype matrix and SNP map from benchmarks/qc_data/."""
    qc_dir = ROOT / "benchmarks" / "qc_data" / run_name
    geno_df = pd.read_csv(qc_dir / "QC_genotype_matrix.csv")
    snp_map = pd.read_csv(qc_dir / "QC_snp_map.csv")
    pheno_df = pd.read_csv(qc_dir / "QC_phenotype.csv")

    sample_ids = geno_df.iloc[:, 0].values.astype(str)
    geno = geno_df.iloc[:, 1:].values.astype(np.float32)

    # Filter to numeric chromosomes only (GAPIT3/rMVP requirement)
    chr_mask = snp_map["Chr"].apply(
        lambda x: str(x).isdigit()
    ).values
    geno = geno[:, chr_mask]
    snp_map = snp_map[chr_mask].reset_index(drop=True)

    return geno, snp_map, sample_ids, pheno_df


def select_qtns(
    snp_map: pd.DataFrame,
    geno: np.ndarray,
    n_qtns: int,
    min_maf: float = 0.10,
    min_chromosomes: int = 3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Select causal SNP indices spread across chromosomes.

    Returns array of column indices into the genotype matrix.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = geno.shape[0]
    # Compute MAF for each SNP
    af = np.nanmean(geno, axis=0) / 2.0
    maf = np.minimum(af, 1 - af)

    # Eligible SNPs: MAF >= threshold, no NaN
    eligible = np.where(maf >= min_maf)[0]
    if len(eligible) < n_qtns:
        raise ValueError(
            f"Only {len(eligible)} SNPs with MAF >= {min_maf}, need {n_qtns}"
        )

    # Ensure spread across chromosomes
    chroms = snp_map["Chr"].values.astype(str)
    unique_chroms = np.unique(chroms[eligible])

    if len(unique_chroms) < min_chromosomes:
        # Not enough chromosomes — just pick randomly
        return rng.choice(eligible, size=n_qtns, replace=False)

    # Strategy: assign QTNs to chromosomes round-robin, then pick randomly within
    n_per_chr = max(1, n_qtns // len(unique_chroms))
    selected = []

    chrom_order = rng.permutation(unique_chroms)
    for ch in chrom_order:
        if len(selected) >= n_qtns:
            break
        ch_eligible = eligible[chroms[eligible] == ch]
        if len(ch_eligible) == 0:
            continue
        pick_n = min(n_per_chr, len(ch_eligible), n_qtns - len(selected))
        picked = rng.choice(ch_eligible, size=pick_n, replace=False)
        selected.extend(picked.tolist())

    # Fill remaining if needed
    remaining = n_qtns - len(selected)
    if remaining > 0:
        pool = np.setdiff1d(eligible, selected)
        extra = rng.choice(pool, size=remaining, replace=False)
        selected.extend(extra.tolist())

    return np.array(selected[:n_qtns])


def simulate_phenotype(
    geno: np.ndarray,
    qtn_indices: np.ndarray,
    h2: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Simulate a phenotype with known causal architecture.

    y = G_causal @ beta + noise
    where noise variance is scaled to achieve target heritability.

    Returns (y, truth_dict).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = geno.shape[0]
    q = len(qtn_indices)

    G_causal = geno[:, qtn_indices].copy()
    # Mean-impute any NaN in causal SNPs
    col_means = np.nanmean(G_causal, axis=0)
    for j in range(q):
        mask = np.isnan(G_causal[:, j])
        if mask.any():
            G_causal[mask, j] = col_means[j]

    # Draw effect sizes
    beta = rng.standard_normal(q)

    # Genetic values
    g = G_causal @ beta
    var_g = np.var(g)

    if var_g < 1e-10:
        # Degenerate — monomorphic QTNs
        var_g = 1.0

    # Scale noise to achieve target h²
    var_e = var_g * (1 - h2) / h2
    noise = rng.normal(0, np.sqrt(var_e), size=n)

    y = g + noise
    h2_realized = var_g / (var_g + np.var(noise))

    truth = {
        "qtn_indices": qtn_indices.tolist(),
        "betas": beta.tolist(),
        "h2_target": h2,
        "h2_realized": float(h2_realized),
        "var_g": float(var_g),
        "var_e": float(var_e),
        "n_qtns": q,
    }
    return y, truth


def generate_all_scenarios(
    h2_levels: list[float] = (0.3, 0.5, 0.8),
    qtn_counts: list[int] = (5, 15, 50),
    n_reps: int = 100,
    run_name: str = "tomato_locule_number",
    output_dir: str | None = None,
    seed: int = 2026,
):
    """Generate all phenotype files and truth JSONs for the simulation study."""
    if output_dir is None:
        output_dir = ROOT / "benchmarks" / "simulation" / "sim_data"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading QC'd genotypes from {run_name}...")
    geno, snp_map, sample_ids, _ = load_qc_data(run_name)
    print(f"  Samples: {geno.shape[0]}, SNPs: {geno.shape[1]}")

    # Save SNP map for downstream evaluation
    snp_map.to_csv(output_dir / "snp_map.csv", index=False)
    np.save(output_dir / "geno_matrix.npy", geno)
    np.save(output_dir / "sample_ids.npy", sample_ids)

    # Also save QTN SNP IDs for each scenario
    rng = np.random.default_rng(seed)
    scenario_count = 0

    for h2 in h2_levels:
        for n_qtns in qtn_counts:
            # Select QTNs once per scenario (shared across reps)
            qtn_indices = select_qtns(snp_map, geno, n_qtns, rng=rng)
            qtn_snp_ids = snp_map.iloc[qtn_indices]["SNP_ID"].tolist()
            qtn_chrs = snp_map.iloc[qtn_indices]["Chr"].tolist()
            qtn_positions = snp_map.iloc[qtn_indices]["Pos"].tolist()

            scenario_name = f"h2_{int(h2*100):03d}_q{n_qtns:03d}"
            print(f"\n  Scenario: {scenario_name} "
                  f"(h2={h2}, {n_qtns} QTNs across "
                  f"{len(set(qtn_chrs))} chromosomes)")

            for rep in range(n_reps):
                rep_dir = output_dir / scenario_name / f"rep_{rep:03d}"
                rep_dir.mkdir(parents=True, exist_ok=True)

                y, truth = simulate_phenotype(geno, qtn_indices, h2, rng=rng)
                truth["qtn_snp_ids"] = qtn_snp_ids
                truth["qtn_chrs"] = [str(c) for c in qtn_chrs]
                truth["qtn_positions"] = [int(p) for p in qtn_positions]
                truth["scenario"] = scenario_name
                truth["rep"] = rep

                # Save phenotype CSV
                pheno_df = pd.DataFrame({
                    "SampleID": sample_ids,
                    "SimTrait": y,
                })
                pheno_df.to_csv(rep_dir / "phenotype.csv", index=False)

                # Save truth JSON
                with open(rep_dir / "truth.json", "w") as f:
                    json.dump(truth, f, indent=2)

                scenario_count += 1

    print(f"\nGenerated {scenario_count} simulations in {output_dir}")
    return scenario_count


def generate_null_phenotypes(
    n_perms: int = 100,
    run_name: str = "tomato_locule_number",
    output_dir: str | None = None,
    seed: int = 2026,
):
    """Generate permuted (null) phenotypes for type I error calibration."""
    if output_dir is None:
        output_dir = ROOT / "benchmarks" / "simulation" / "null_data"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, sample_ids, pheno_df = load_qc_data(run_name)
    # Use the first numeric column as trait
    trait_col = [c for c in pheno_df.columns if c != "SampleID"][0]
    y_real = pheno_df[trait_col].values.astype(float)

    # Drop NaN
    valid = ~np.isnan(y_real)
    y_real = y_real[valid]
    sample_ids = sample_ids[valid]

    rng = np.random.default_rng(seed)

    for i in range(n_perms):
        perm_dir = output_dir / f"perm_{i:03d}"
        perm_dir.mkdir(parents=True, exist_ok=True)

        y_perm = rng.permutation(y_real)
        pheno_perm = pd.DataFrame({
            "SampleID": sample_ids,
            "SimTrait": y_perm,
        })
        pheno_perm.to_csv(perm_dir / "phenotype.csv", index=False)

    print(f"Generated {n_perms} null phenotypes in {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate simulated phenotypes")
    parser.add_argument("--mode", choices=["sim", "null", "both"], default="both")
    parser.add_argument("--n-reps", type=int, default=50)
    parser.add_argument("--n-perms", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.mode in ("sim", "both"):
        generate_all_scenarios(n_reps=args.n_reps, seed=args.seed)
    if args.mode in ("null", "both"):
        generate_null_phenotypes(n_perms=args.n_perms, seed=args.seed)
