"""Per-QTN detection analysis across tools and scenarios.

Reports which QTNs each tool detects, enabling set-difference
comparisons (e.g. LOCO-only vs GAPIT-only) and recovery curves
stratified by effect size. Re-reads existing p-value CSVs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
SIM_DATA = ROOT / "benchmarks" / "simulation" / "sim_data"
SIM_SUMMARY = ROOT / "benchmarks" / "simulation" / "sim_summary"

TOOLS = (
    "platform_loco", "platform_global", "gapit", "rmvp",
    "farmcpu_loco", "gapit_farmcpu",
)
# Friendly names for display / figures
TOOL_LABELS = {
    "platform_loco": "TRACE-LOCO",
    "platform_global": "TRACE-Global",
    "gapit": "GAPIT3",
    "rmvp": "rMVP",
    "farmcpu_loco": "TRACE-FarmCPU",
    "gapit_farmcpu": "GAPIT3-FarmCPU",
}
WINDOW_KB = 500


def per_qtn_detection(
    results_csv: Path,
    truth: dict,
    threshold: float,
    window_kb: float = WINDOW_KB,
) -> list[dict]:
    """For one rep x tool, return per-QTN detection status."""
    df = pd.read_csv(results_csv).dropna(subset=["PValue"])
    sig = df[df["PValue"] < threshold]

    window_bp = window_kb * 1000
    qtn_rows = []

    for qi in range(truth["n_qtns"]):
        qc = str(truth["qtn_chrs"][qi])
        qp = int(truth["qtn_positions"][qi])
        beta = truth["betas"][qi]

        detected = False
        best_p = 1.0
        n_sig_nearby = 0

        if len(sig) > 0:
            sig_chrs = sig["Chr"].astype(str).values
            sig_pos = sig["Pos"].astype(int).values
            sig_pvals = sig["PValue"].values

            for si in range(len(sig)):
                if sig_chrs[si] == qc and abs(sig_pos[si] - qp) <= window_bp:
                    detected = True
                    n_sig_nearby += 1
                    if sig_pvals[si] < best_p:
                        best_p = sig_pvals[si]

        qtn_rows.append({
            "qtn_idx": qi,
            "qtn_chr": qc,
            "qtn_pos": qp,
            "qtn_beta": beta,
            "qtn_abs_beta": abs(beta),
            "detected": detected,
            "best_pvalue": best_p if detected else np.nan,
            "n_sig_nearby": n_sig_nearby,
        })

    return qtn_rows


def evaluate_qtn_recovery(
    sim_data_dir: Path = SIM_DATA,
    tools: tuple[str, ...] = TOOLS,
    window_kb: float = WINDOW_KB,
) -> pd.DataFrame:
    """Run per-QTN detection across all scenarios, reps, and tools."""
    snp_map = pd.read_csv(sim_data_dir / "snp_map.csv")
    n_snps = len(snp_map)
    bonf_threshold = 0.05 / n_snps

    # Precompute per-SNP MAF for PVE computation
    geno = np.load(sim_data_dir / "geno_matrix.npy")
    af = np.nanmean(geno, axis=0) / 2.0
    maf_all = np.minimum(af, 1.0 - af)

    all_rows = []
    scenario_dirs = sorted(sim_data_dir.glob("h2_*/rep_*"))
    n_dirs = len(scenario_dirs)

    for i, sd in enumerate(scenario_dirs):
        truth_path = sd / "truth.json"
        if not truth_path.exists():
            continue
        with open(truth_path) as f:
            truth = json.load(f)

        if (i + 1) % 100 == 0:
            print(f"  Processing {i + 1}/{n_dirs} ...")

        var_total = truth["var_g"] + truth["var_e"]

        for tool in tools:
            results_path = sd / tool / "results.csv"
            if not results_path.exists():
                continue

            qtn_rows = per_qtn_detection(
                results_path, truth, bonf_threshold, window_kb,
            )
            for row in qtn_rows:
                qi = row["qtn_idx"]
                p = float(maf_all[truth["qtn_indices"][qi]])
                # Marginal PVE: per-QTN fraction of total phenotypic
                # variance under HWE, ignoring LD with other QTNs
                row["qtn_maf"] = p
                row["qtn_pve"] = (
                    (row["qtn_beta"] ** 2) * 2.0 * p * (1.0 - p) / var_total
                )
                row.update({
                    "scenario": truth["scenario"],
                    "rep": truth["rep"],
                    "h2": truth["h2_target"],
                    "n_qtns": truth["n_qtns"],
                    "tool": tool,
                })
                all_rows.append(row)

    return pd.DataFrame(all_rows)


def compute_set_differences(qtn_df: pd.DataFrame) -> pd.DataFrame:
    """For each rep x scenario, compute TRACE-only vs competitor-only QTNs.

    Compares platform_loco against each other MLM tool (gapit, rmvp,
    platform_global) separately.
    """
    comparisons = [
        ("platform_loco", "gapit", "TRACE-LOCO vs GAPIT3"),
        ("platform_loco", "rmvp", "TRACE-LOCO vs rMVP"),
        ("platform_loco", "platform_global", "LOCO vs Global"),
    ]

    rows = []
    for scenario, s_grp in qtn_df.groupby("scenario"):
        for rep, r_grp in s_grp.groupby("rep"):
            h2 = r_grp["h2"].iloc[0]
            n_qtns = r_grp["n_qtns"].iloc[0]

            for tool_a, tool_b, label in comparisons:
                a = r_grp[r_grp["tool"] == tool_a]
                b = r_grp[r_grp["tool"] == tool_b]
                if len(a) == 0 or len(b) == 0:
                    continue

                a_detected = set(a[a["detected"]]["qtn_idx"])
                b_detected = set(b[b["detected"]]["qtn_idx"])

                only_a = a_detected - b_detected
                only_b = b_detected - a_detected
                both = a_detected & b_detected
                neither = set(range(n_qtns)) - a_detected - b_detected

                # Effect sizes of exclusively-detected QTNs
                a_betas = a[a["qtn_idx"].isin(only_a)]["qtn_abs_beta"]
                b_betas = b[b["qtn_idx"].isin(only_b)]["qtn_abs_beta"]

                rows.append({
                    "scenario": scenario,
                    "rep": rep,
                    "h2": h2,
                    "n_qtns": n_qtns,
                    "comparison": label,
                    "tool_a": tool_a,
                    "tool_b": tool_b,
                    "n_only_a": len(only_a),
                    "n_only_b": len(only_b),
                    "n_both": len(both),
                    "n_neither": len(neither),
                    "power_a": len(a_detected) / n_qtns,
                    "power_b": len(b_detected) / n_qtns,
                    "mean_abs_beta_only_a": a_betas.mean() if len(a_betas) > 0 else np.nan,
                    "mean_abs_beta_only_b": b_betas.mean() if len(b_betas) > 0 else np.nan,
                })

    return pd.DataFrame(rows)


def compute_effect_size_recovery(qtn_df: pd.DataFrame) -> pd.DataFrame:
    """Compute recovery rate binned by QTN effect size."""
    # Create effect size bins
    df = qtn_df.copy()
    df["abs_beta_bin"] = pd.cut(
        df["qtn_abs_beta"],
        bins=[0, 0.1, 0.3, 0.5, 1.0, 2.0, np.inf],
        labels=["<0.1", "0.1-0.3", "0.3-0.5", "0.5-1.0", "1.0-2.0", ">2.0"],
    )

    rows = []
    for (tool, h2, nq, beta_bin), grp in df.groupby(
        ["tool", "h2", "n_qtns", "abs_beta_bin"], observed=True
    ):
        n = len(grp)
        n_detected = grp["detected"].sum()
        rows.append({
            "tool": tool,
            "h2": h2,
            "n_qtns": nq,
            "abs_beta_bin": beta_bin,
            "n_qtns_in_bin": n,
            "n_detected": int(n_detected),
            "recovery_rate": n_detected / n if n > 0 else 0.0,
        })

    return pd.DataFrame(rows)


def compute_pve_recovery(qtn_df: pd.DataFrame) -> pd.DataFrame:
    """Compute recovery rate binned by per-QTN PVE (% variance explained)."""
    df = qtn_df.copy()
    # PVE expressed as percentage for readable bin labels
    df["pve_pct"] = df["qtn_pve"] * 100.0
    df["pve_bin"] = pd.cut(
        df["pve_pct"],
        bins=[0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, np.inf],
        labels=["<0.5%", "0.5-1%", "1-2%", "2-5%", "5-10%", "10-20%", ">20%"],
    )

    rows = []
    for (tool, h2, nq, pve_bin), grp in df.groupby(
        ["tool", "h2", "n_qtns", "pve_bin"], observed=True
    ):
        n = len(grp)
        n_detected = grp["detected"].sum()
        rows.append({
            "tool": tool,
            "h2": h2,
            "n_qtns": nq,
            "pve_bin": pve_bin,
            "n_qtns_in_bin": n,
            "n_detected": int(n_detected),
            "recovery_rate": n_detected / n if n > 0 else 0.0,
        })

    return pd.DataFrame(rows)


def print_summary(set_diff_df: pd.DataFrame):
    """Print human-readable summary of set-difference results."""
    print("\n" + "=" * 80)
    print("  QTN-LEVEL RECOVERY: SET DIFFERENCES")
    print("=" * 80)

    for comparison, c_grp in set_diff_df.groupby("comparison"):
        print(f"\n  {comparison}:")
        print(f"  {'Scenario':<20s} {'Only-A':>8s} {'Only-B':>8s} "
              f"{'Both':>8s} {'Neither':>8s} {'PowerA':>8s} {'PowerB':>8s}")
        print("  " + "-" * 72)

        for (h2, nq), s_grp in c_grp.groupby(["h2", "n_qtns"]):
            label = f"h2={h2}, q={nq}"
            oa = s_grp["n_only_a"].mean()
            ob = s_grp["n_only_b"].mean()
            bo = s_grp["n_both"].mean()
            ne = s_grp["n_neither"].mean()
            pa = s_grp["power_a"].mean()
            pb = s_grp["power_b"].mean()
            print(f"  {label:<20s} {oa:>8.2f} {ob:>8.2f} "
                  f"{bo:>8.2f} {ne:>8.2f} {pa:>8.1%} {pb:>8.1%}")

    # Effect size of exclusively-detected QTNs
    print("\n\n  EFFECT SIZE OF EXCLUSIVELY-DETECTED QTNs:")
    print("  " + "-" * 60)
    for comparison, c_grp in set_diff_df.groupby("comparison"):
        valid_a = c_grp.dropna(subset=["mean_abs_beta_only_a"])
        valid_b = c_grp.dropna(subset=["mean_abs_beta_only_b"])
        if len(valid_a) > 0:
            beta_a = valid_a["mean_abs_beta_only_a"].mean()
        else:
            beta_a = float("nan")
        if len(valid_b) > 0:
            beta_b = valid_b["mean_abs_beta_only_b"].mean()
        else:
            beta_b = float("nan")
        print(f"  {comparison}:")
        print(f"    Mean |beta| exclusive to A: {beta_a:.4f}")
        print(f"    Mean |beta| exclusive to B: {beta_b:.4f}")


def main():
    print("QTN-Level Recovery Analysis")
    print("=" * 40)

    # Step 1: Per-QTN detection
    print("\nStep 1: Computing per-QTN detection across all reps/tools ...")
    qtn_df = evaluate_qtn_recovery()
    print(f"  Total rows: {len(qtn_df):,}")

    SIM_SUMMARY.mkdir(parents=True, exist_ok=True)
    qtn_df.to_csv(SIM_SUMMARY / "qtn_recovery.csv", index=False)
    print(f"  Saved: {SIM_SUMMARY / 'qtn_recovery.csv'}")

    # Step 2: Set differences
    print("\nStep 2: Computing set differences ...")
    set_diff_df = compute_set_differences(qtn_df)
    set_diff_df.to_csv(SIM_SUMMARY / "qtn_set_differences.csv", index=False)
    print(f"  Saved: {SIM_SUMMARY / 'qtn_set_differences.csv'}")

    print_summary(set_diff_df)

    # Step 3: Effect size recovery
    print("\nStep 3: Computing effect-size-stratified recovery ...")
    es_df = compute_effect_size_recovery(qtn_df)
    es_df.to_csv(SIM_SUMMARY / "qtn_effect_size_recovery.csv", index=False)
    print(f"  Saved: {SIM_SUMMARY / 'qtn_effect_size_recovery.csv'}")

    # Print effect size recovery for key tools
    print("\n  Recovery rate by effect size (MLM tools, 5-QTN scenarios):")
    mlm_tools = ["platform_loco", "platform_global", "gapit", "rmvp"]
    sub = es_df[(es_df["tool"].isin(mlm_tools)) & (es_df["n_qtns"] == 5)]
    if len(sub) > 0:
        pivot = sub.groupby(["tool", "abs_beta_bin"])["recovery_rate"].mean()
        print(pivot.unstack().to_string())

    # Step 4: PVE-stratified recovery
    print("\nStep 4: Computing PVE-stratified recovery ...")
    pve_df = compute_pve_recovery(qtn_df)
    pve_df.to_csv(SIM_SUMMARY / "qtn_pve_recovery.csv", index=False)
    print(f"  Saved: {SIM_SUMMARY / 'qtn_pve_recovery.csv'}")

    print("\n  Recovery rate by PVE (MLM tools, 5-QTN scenarios):")
    sub = pve_df[(pve_df["tool"].isin(mlm_tools)) & (pve_df["n_qtns"] == 5)]
    if len(sub) > 0:
        pivot = sub.groupby(["tool", "pve_bin"])["recovery_rate"].mean()
        print(pivot.unstack().to_string())

    print("\n  Done!")


if __name__ == "__main__":
    main()
