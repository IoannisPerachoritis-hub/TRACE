"""Evaluate simulation results: Power, FDR, Type I error.

Reads truth JSONs and standardized result CSVs from each tool,
classifies detections as true/false positives, and aggregates
across replicates.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent


def _cluster_significant_snps(
    sig_chrs: np.ndarray,
    sig_pos: np.ndarray,
    window_bp: float,
) -> list[tuple[str, float]]:
    """Merge significant SNPs into loci (clusters within *window_bp* on same chr).

    Returns a list of (chr, midpoint) for each locus.
    """
    if len(sig_chrs) == 0:
        return []
    order = np.lexsort((sig_pos, sig_chrs))
    loci: list[tuple[str, float]] = []
    cur_chr = sig_chrs[order[0]]
    cur_start = sig_pos[order[0]]
    cur_end = sig_pos[order[0]]
    for idx in order[1:]:
        c, p = sig_chrs[idx], sig_pos[idx]
        if c == cur_chr and p - cur_end <= window_bp:
            cur_end = p
        else:
            loci.append((cur_chr, (cur_start + cur_end) / 2))
            cur_chr, cur_start, cur_end = c, p, p
    loci.append((cur_chr, (cur_start + cur_end) / 2))
    return loci


def classify_detections(
    results_df: pd.DataFrame,
    truth: dict,
    snp_map: pd.DataFrame,
    threshold: float,
    window_kb: float = 500,
) -> dict:
    """Classify significant SNPs as true/false positives.

    A detection is a true positive if the SNP is within `window_kb` kb
    of a QTN on the same chromosome.

    Reports both SNP-level and locus-level FDR.  Locus-level FDR groups
    nearby significant SNPs into clusters first, then checks whether each
    cluster overlaps a QTN — this avoids inflated FDR from LD hitchhiking.
    """
    sig = results_df[results_df["PValue"] < threshold].copy()
    if len(sig) == 0:
        return {
            "n_significant": 0,
            "n_true_positives": 0,
            "n_false_positives": 0,
            "n_false_negatives": truth["n_qtns"],
            "power": 0.0,
            "fdr": 0.0,
            "n_loci": 0,
            "n_loci_tp": 0,
            "n_loci_fp": 0,
            "fdr_locus": 0.0,
        }

    qtn_chrs = [str(c) for c in truth["qtn_chrs"]]
    qtn_positions = truth["qtn_positions"]
    window_bp = window_kb * 1000

    # For each QTN, check if any significant SNP is nearby
    qtns_detected = set()
    tp_snps = set()

    sig_chrs = sig["Chr"].astype(str).values
    sig_pos = sig["Pos"].astype(int).values

    for qi, (qc, qp) in enumerate(zip(qtn_chrs, qtn_positions)):
        for si in range(len(sig)):
            if sig_chrs[si] == qc and abs(sig_pos[si] - qp) <= window_bp:
                qtns_detected.add(qi)
                tp_snps.add(si)

    n_tp = len(qtns_detected)  # Number of QTNs detected (power numerator)
    n_fp = len(sig) - len(tp_snps)  # Significant SNPs not near any QTN
    n_fn = truth["n_qtns"] - n_tp

    power = n_tp / truth["n_qtns"] if truth["n_qtns"] > 0 else 0.0
    fdr = n_fp / len(sig) if len(sig) > 0 else 0.0

    # ── Locus-level FDR ─────────────────────────────────────
    loci = _cluster_significant_snps(sig_chrs, sig_pos, window_bp)
    n_loci = len(loci)
    n_loci_tp = 0
    for lc, lmid in loci:
        for qc, qp in zip(qtn_chrs, qtn_positions):
            if lc == qc and abs(lmid - qp) <= window_bp:
                n_loci_tp += 1
                break
    n_loci_fp = n_loci - n_loci_tp
    fdr_locus = n_loci_fp / n_loci if n_loci > 0 else 0.0

    return {
        "n_significant": len(sig),
        "n_true_positives": n_tp,
        "n_false_positives": n_fp,
        "n_false_negatives": n_fn,
        "power": power,
        "fdr": fdr,
        "n_loci": n_loci,
        "n_loci_tp": n_loci_tp,
        "n_loci_fp": n_loci_fp,
        "fdr_locus": fdr_locus,
    }


def power_fdr_curve(
    results_df: pd.DataFrame,
    truth: dict,
    snp_map: pd.DataFrame,
    thresholds: np.ndarray | None = None,
    window_kb: float = 500,
) -> pd.DataFrame:
    """Compute power and FDR at multiple thresholds for one replicate."""
    if thresholds is None:
        thresholds = np.logspace(-1, -8, 50)

    rows = []
    for t in thresholds:
        result = classify_detections(results_df, truth, snp_map, t, window_kb)
        result["threshold"] = t
        rows.append(result)

    return pd.DataFrame(rows)


def evaluate_all_scenarios(
    sim_data_dir: str | None = None,
    tools: tuple[str, ...] = (
        "platform_loco", "platform_global", "gapit", "rmvp",
        "farmcpu_loco", "gapit_farmcpu",
    ),
    window_kb: float = 500,
):
    """Evaluate all simulation results across all tools and scenarios."""
    if sim_data_dir is None:
        sim_data_dir = ROOT / "benchmarks" / "simulation" / "sim_data"
    sim_data_dir = Path(sim_data_dir)

    snp_map = pd.read_csv(sim_data_dir / "snp_map.csv")
    n_snps = len(snp_map)
    bonf_threshold = 0.05 / n_snps

    all_results = []
    all_curves = []

    scenario_dirs = sorted(sim_data_dir.glob("h2_*/rep_*"))

    for sd in scenario_dirs:
        truth_path = sd / "truth.json"
        if not truth_path.exists():
            continue
        with open(truth_path) as f:
            truth = json.load(f)

        scenario = truth["scenario"]
        rep = truth["rep"]
        h2 = truth["h2_target"]
        n_qtns = truth["n_qtns"]

        for tool in tools:
            results_path = sd / tool / "results.csv"
            if not results_path.exists():
                continue

            results_df = pd.read_csv(results_path)
            results_df = results_df.dropna(subset=["PValue"])

            # Bonferroni classification
            bonf = classify_detections(
                results_df, truth, snp_map, bonf_threshold, window_kb,
            )
            bonf.update({
                "scenario": scenario,
                "rep": rep,
                "h2": h2,
                "n_qtns": n_qtns,
                "tool": tool,
                "threshold_type": "bonferroni",
                "threshold": bonf_threshold,
            })

            # Lambda GC
            pvals = results_df["PValue"].values
            pvals = pvals[(pvals > 0) & (pvals < 1)]
            if len(pvals) > 10:
                chisq = stats.chi2.ppf(1 - pvals, df=1)
                bonf["lambda_gc"] = float(
                    np.median(chisq) / stats.chi2.ppf(0.5, df=1)
                )

            all_results.append(bonf)

            # Power/FDR curve (for a subset of reps to save time)
            if rep < 10:
                curve = power_fdr_curve(
                    results_df, truth, snp_map, window_kb=window_kb,
                )
                curve["scenario"] = scenario
                curve["rep"] = rep
                curve["h2"] = h2
                curve["n_qtns"] = n_qtns
                curve["tool"] = tool
                all_curves.append(curve)

    results_df = pd.DataFrame(all_results)
    curves_df = pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame()

    # Save
    out_dir = sim_data_dir.parent / "sim_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(out_dir / "bonferroni_results.csv", index=False)
    if len(curves_df) > 0:
        curves_df.to_csv(out_dir / "power_fdr_curves.csv", index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("  SIMULATION EVALUATION SUMMARY")
    print("=" * 60)

    if len(results_df) == 0:
        print("  No results found!")
        return results_df, curves_df

    for (h2, nq), grp in results_df.groupby(["h2", "n_qtns"]):
        print(f"\n  h2={h2}, QTNs={nq}:")
        for tool, tgrp in grp.groupby("tool"):
            power = tgrp["power"].mean()
            fdr = tgrp["fdr"].mean()
            lam = tgrp["lambda_gc"].mean() if "lambda_gc" in tgrp else float("nan")
            print(f"    {tool:20s}: Power={power:.3f}, FDR={fdr:.3f}, "
                  f"Lambda={lam:.3f}")

    return results_df, curves_df


def evaluate_null(
    null_data_dir: str | None = None,
):
    """Evaluate null calibration results."""
    if null_data_dir is None:
        null_data_dir = ROOT / "benchmarks" / "simulation" / "null_data"
    null_data_dir = Path(null_data_dir)

    all_pvalues = []
    lambdas = []

    perm_dirs = sorted(null_data_dir.glob("perm_*"))
    for pd_dir in perm_dirs:
        results_path = pd_dir / "platform_loco" / "results.csv"
        if not results_path.exists():
            continue
        df = pd.read_csv(results_path)
        pvals = df["PValue"].dropna().values
        pvals = pvals[(pvals > 0) & (pvals < 1)]
        all_pvalues.extend(pvals.tolist())

        # Lambda GC per permutation
        if len(pvals) > 10:
            chisq = stats.chi2.ppf(1 - pvals, df=1)
            lam = float(np.median(chisq) / stats.chi2.ppf(0.5, df=1))
            lambdas.append(lam)

    if not all_pvalues:
        print("No null results found!")
        return

    all_pvalues = np.array(all_pvalues)
    n_total = len(all_pvalues)

    # Empirical type I error rates
    # Include M_eff threshold (0.05/250 = 2e-4) alongside standard levels
    thresholds = [0.05, 0.01, 0.001, 2e-4, 5e-5, 0.05 / 43974]
    print("\n" + "=" * 60)
    print("  NULL CALIBRATION RESULTS")
    print("=" * 60)
    print(f"  Total p-values: {n_total:,} ({len(lambdas)} permutations)")
    print(f"  Lambda GC: mean={np.mean(lambdas):.4f}, "
          f"SD={np.std(lambdas):.4f}, "
          f"range=[{min(lambdas):.4f}, {max(lambdas):.4f}]")

    print("\n  Empirical Type I Error:")
    type1_rows = []
    for t in thresholds:
        n_sig = (all_pvalues < t).sum()
        observed_rate = n_sig / n_total
        expected_rate = t
        ratio = observed_rate / expected_rate if expected_rate > 0 else float("nan")
        if t < 1e-4:
            label = "Bonferroni"
        elif abs(t - 2e-4) < 1e-10:
            label = "M_eff (0.05/250)"
        else:
            label = f"alpha={t}"
        print(f"    {label:20s}: observed={observed_rate:.6f}, "
              f"expected={expected_rate:.6f}, ratio={ratio:.3f}")
        type1_rows.append({
            "threshold": t,
            "label": label,
            "n_significant": n_sig,
            "n_total": n_total,
            "observed_rate": observed_rate,
            "expected_rate": expected_rate,
            "ratio": ratio,
        })

    # Save
    out_dir = ROOT / "benchmarks" / "simulation" / "sim_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(type1_rows).to_csv(
        out_dir / "null_type1_error.csv", index=False,
    )
    pd.DataFrame({"lambda_gc": lambdas}).to_csv(
        out_dir / "null_lambda_gc.csv", index=False,
    )

    # Save QQ data for plotting
    n_qq = min(len(all_pvalues), 100_000)  # Subsample for QQ plot
    if len(all_pvalues) > n_qq:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(all_pvalues), size=n_qq, replace=False)
        qq_pvals = np.sort(all_pvalues[idx])
    else:
        qq_pvals = np.sort(all_pvalues)

    expected = -np.log10(np.linspace(1 / len(qq_pvals), 1, len(qq_pvals)))
    observed = -np.log10(qq_pvals)

    pd.DataFrame({"expected": expected, "observed": observed}).to_csv(
        out_dir / "null_qq_data.csv", index=False,
    )


def summarize_with_ci(
    results_path: str | None = None,
    output_dir: str | None = None,
):
    """Compute per-scenario summary with 95% CIs from replicates."""
    if results_path is None:
        results_path = (
            ROOT / "benchmarks" / "simulation" / "sim_summary"
            / "bonferroni_results.csv"
        )
    if output_dir is None:
        output_dir = ROOT / "benchmarks" / "simulation" / "sim_summary"
    output_dir = Path(output_dir)

    df = pd.read_csv(results_path)
    if len(df) == 0:
        print("No results for CI summary!")
        return

    rows = []
    for (h2, nq, tool), grp in df.groupby(["h2", "n_qtns", "tool"]):
        n = len(grp)
        power_mean = grp["power"].mean()
        power_se = grp["power"].std(ddof=1) / np.sqrt(n) if n > 1 else 0
        fdr_mean = grp["fdr"].mean()
        fdr_se = grp["fdr"].std(ddof=1) / np.sqrt(n) if n > 1 else 0
        lam_mean = (grp["lambda_gc"].mean()
                    if "lambda_gc" in grp and grp["lambda_gc"].notna().any()
                    else float("nan"))

        row = {
            "h2": h2,
            "n_qtns": nq,
            "tool": tool,
            "n_reps": n,
            "power_mean": round(power_mean, 4),
            "power_se": round(power_se, 4),
            "power_ci_lo": round(max(0, power_mean - 1.96 * power_se), 4),
            "power_ci_hi": round(min(1, power_mean + 1.96 * power_se), 4),
            "fdr_mean": round(fdr_mean, 4),
            "fdr_se": round(fdr_se, 4),
            "fdr_ci_lo": round(max(0, fdr_mean - 1.96 * fdr_se), 4),
            "fdr_ci_hi": round(min(1, fdr_mean + 1.96 * fdr_se), 4),
            "lambda_gc_mean": round(lam_mean, 4) if not np.isnan(lam_mean) else None,
        }

        # Locus-level FDR (if available in results)
        if "fdr_locus" in grp.columns:
            fdr_l_mean = grp["fdr_locus"].mean()
            fdr_l_se = grp["fdr_locus"].std(ddof=1) / np.sqrt(n) if n > 1 else 0
            row["fdr_locus_mean"] = round(fdr_l_mean, 4)
            row["fdr_locus_se"] = round(fdr_l_se, 4)
            row["fdr_locus_ci_lo"] = round(max(0, fdr_l_mean - 1.96 * fdr_l_se), 4)
            row["fdr_locus_ci_hi"] = round(min(1, fdr_l_mean + 1.96 * fdr_l_se), 4)

        rows.append(row)

    summary = pd.DataFrame(rows)
    out_path = output_dir / "power_fdr_summary_ci.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nSaved CI summary: {out_path}")

    # Print nicely
    print("\n" + "=" * 80)
    print("  POWER & FDR WITH 95% CI")
    print("=" * 80)
    for (h2, nq), grp in summary.groupby(["h2", "n_qtns"]):
        print(f"\n  h2={h2}, QTNs={nq}:")
        for _, r in grp.iterrows():
            print(
                f"    {r['tool']:20s}: "
                f"Power={r['power_mean']:.3f} [{r['power_ci_lo']:.3f}, {r['power_ci_hi']:.3f}]  "
                f"FDR={r['fdr_mean']:.3f} [{r['fdr_ci_lo']:.3f}, {r['fdr_ci_hi']:.3f}]  "
                f"(n={r['n_reps']})"
            )


def evaluate_window_sensitivity(
    sim_data_dir: str | None = None,
    windows_kb: tuple[float, ...] = (50, 100, 250, 500, 1000),
    tools: tuple[str, ...] = (
        "platform_loco", "platform_global", "gapit", "rmvp",
        "farmcpu_loco", "gapit_farmcpu",
    ),
):
    """Evaluate power/FDR at multiple TP-classification windows.

    No new GWAS runs needed — just re-classifies existing p-values
    with different window sizes.
    """
    if sim_data_dir is None:
        sim_data_dir = ROOT / "benchmarks" / "simulation" / "sim_data"
    sim_data_dir = Path(sim_data_dir)

    out_dir = sim_data_dir.parent / "sim_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for wkb in windows_kb:
        print(f"\n  Evaluating window = {wkb} kb ...")
        results_df, _ = evaluate_all_scenarios(
            sim_data_dir=str(sim_data_dir),
            tools=tools,
            window_kb=wkb,
        )

        if len(results_df) == 0:
            continue

        # Aggregate per tool × scenario
        for (h2, nq, tool), grp in results_df.groupby(["h2", "n_qtns", "tool"]):
            n = len(grp)
            power_mean = grp["power"].mean()
            power_se = grp["power"].std(ddof=1) / np.sqrt(n) if n > 1 else 0
            fdr_mean = grp["fdr"].mean()
            fdr_se = grp["fdr"].std(ddof=1) / np.sqrt(n) if n > 1 else 0

            all_rows.append({
                "window_kb": wkb,
                "h2": h2,
                "n_qtns": nq,
                "tool": tool,
                "n_reps": n,
                "power_mean": round(power_mean, 4),
                "power_se": round(power_se, 4),
                "power_ci_lo": round(max(0, power_mean - 1.96 * power_se), 4),
                "power_ci_hi": round(min(1, power_mean + 1.96 * power_se), 4),
                "fdr_mean": round(fdr_mean, 4),
                "fdr_se": round(fdr_se, 4),
                "fdr_ci_lo": round(max(0, fdr_mean - 1.96 * fdr_se), 4),
                "fdr_ci_hi": round(min(1, fdr_mean + 1.96 * fdr_se), 4),
            })

    ws_df = pd.DataFrame(all_rows)
    ws_path = out_dir / "window_sensitivity.csv"
    ws_df.to_csv(ws_path, index=False)
    print(f"\nSaved window sensitivity: {ws_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("  WINDOW SENSITIVITY SUMMARY")
    print("=" * 70)
    for tool in tools:
        tsub = ws_df[ws_df["tool"] == tool]
        if len(tsub) == 0:
            continue
        print(f"\n  {tool}:")
        for wkb in windows_kb:
            wsub = tsub[tsub["window_kb"] == wkb]
            if len(wsub) == 0:
                continue
            overall_power = wsub["power_mean"].mean()
            overall_fdr = wsub["fdr_mean"].mean()
            print(f"    {wkb:4.0f} kb: mean power={overall_power:.3f}, "
                  f"mean FDR={overall_fdr:.3f}")

    return ws_df


def evaluate_fp_chromosomal_distribution(
    sim_data_dir: str | None = None,
    tools: tuple[str, ...] = (
        "platform_loco", "platform_global", "gapit", "rmvp",
    ),
    window_kb: float = 250,
):
    """Analyse whether false-positive SNPs cluster on QTN-harboring chromosomes.

    For each scenario/tool/replicate, significant SNPs are classified as TP
    or FP (reusing the same window logic as ``classify_detections``).  Each FP
    is then labelled as residing on a chromosome that contains at least one
    QTN ("QTN-chr") or not ("non-QTN-chr").  The fraction of FPs on QTN
    chromosomes is aggregated with 95 % CIs across replicates.

    No new GWAS runs are needed — only existing p-value files are re-read.
    """
    if sim_data_dir is None:
        sim_data_dir = ROOT / "benchmarks" / "simulation" / "sim_data"
    sim_data_dir = Path(sim_data_dir)

    snp_map = pd.read_csv(sim_data_dir / "snp_map.csv")
    n_snps = len(snp_map)
    bonf_threshold = 0.05 / n_snps
    window_bp = window_kb * 1000

    # Pre-compute per-chromosome SNP counts for expected-fraction calc
    chr_snp_counts = snp_map["Chr"].astype(str).value_counts().to_dict()

    per_rep_rows: list[dict] = []

    scenario_dirs = sorted(sim_data_dir.glob("h2_*/rep_*"))
    for sd in scenario_dirs:
        truth_path = sd / "truth.json"
        if not truth_path.exists():
            continue
        with open(truth_path) as f:
            truth = json.load(f)

        scenario = truth["scenario"]
        rep = truth["rep"]
        h2 = truth["h2_target"]
        n_qtns = truth["n_qtns"]
        qtn_chrs = {str(c) for c in truth["qtn_chrs"]}
        qtn_positions = {
            (str(c), int(p))
            for c, p in zip(truth["qtn_chrs"], truth["qtn_positions"])
        }

        # Expected fraction: SNPs on QTN chromosomes / total SNPs
        n_snps_on_qtn_chr = sum(
            chr_snp_counts.get(c, 0) for c in qtn_chrs
        )
        expected_frac = n_snps_on_qtn_chr / n_snps

        for tool in tools:
            results_path = sd / tool / "results.csv"
            if not results_path.exists():
                continue

            results_df = pd.read_csv(results_path).dropna(subset=["PValue"])
            sig = results_df[results_df["PValue"] < bonf_threshold]
            if len(sig) == 0:
                continue

            sig_chrs = sig["Chr"].astype(str).values
            sig_pos = sig["Pos"].astype(int).values

            # Classify each significant SNP as TP or FP
            tp_indices: set[int] = set()
            for qc, qp in qtn_positions:
                for si in range(len(sig)):
                    if sig_chrs[si] == qc and abs(sig_pos[si] - qp) <= window_bp:
                        tp_indices.add(si)

            # Count FPs by chromosome type
            n_fp_qtn_chr = 0
            n_fp_non_qtn_chr = 0
            for si in range(len(sig)):
                if si in tp_indices:
                    continue  # skip TPs
                if sig_chrs[si] in qtn_chrs:
                    n_fp_qtn_chr += 1
                else:
                    n_fp_non_qtn_chr += 1

            n_fp_total = n_fp_qtn_chr + n_fp_non_qtn_chr
            frac_on_qtn_chr = n_fp_qtn_chr / n_fp_total if n_fp_total > 0 else float("nan")

            per_rep_rows.append({
                "scenario": scenario,
                "rep": rep,
                "h2": h2,
                "n_qtns": n_qtns,
                "tool": tool,
                "n_significant": len(sig),
                "n_tp": len(tp_indices),
                "n_fp_total": n_fp_total,
                "n_fp_qtn_chr": n_fp_qtn_chr,
                "n_fp_non_qtn_chr": n_fp_non_qtn_chr,
                "frac_fp_on_qtn_chr": frac_on_qtn_chr,
                "expected_frac_qtn_chr": expected_frac,
                "n_qtn_chrs": len(qtn_chrs),
            })

    per_rep_df = pd.DataFrame(per_rep_rows)

    # Aggregate per (h2, n_qtns, tool)
    summary_rows: list[dict] = []
    for (h2, nq, tool), grp in per_rep_df.groupby(["h2", "n_qtns", "tool"]):
        valid = grp.dropna(subset=["frac_fp_on_qtn_chr"])
        n = len(valid)
        if n == 0:
            continue
        frac_mean = valid["frac_fp_on_qtn_chr"].mean()
        frac_se = valid["frac_fp_on_qtn_chr"].std(ddof=1) / np.sqrt(n) if n > 1 else 0
        exp_mean = valid["expected_frac_qtn_chr"].mean()
        summary_rows.append({
            "h2": h2,
            "n_qtns": nq,
            "tool": tool,
            "n_reps_total": len(grp),
            "n_reps_with_fp": n,
            "frac_fp_qtn_chr_mean": round(frac_mean, 4),
            "frac_fp_qtn_chr_se": round(frac_se, 4),
            "frac_fp_qtn_chr_ci_lo": round(max(0, frac_mean - 1.96 * frac_se), 4),
            "frac_fp_qtn_chr_ci_hi": round(min(1, frac_mean + 1.96 * frac_se), 4),
            "expected_frac_mean": round(exp_mean, 4),
            "mean_fp_per_rep": round(valid["n_fp_total"].mean(), 1),
        })

    summary_df = pd.DataFrame(summary_rows)

    # Save both per-rep and summary
    out_dir = sim_data_dir.parent / "sim_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_rep_df.to_csv(out_dir / "fp_chromosomal_per_rep.csv", index=False)
    summary_df.to_csv(out_dir / "fp_chromosomal_distribution.csv", index=False)

    # Print
    print("\n" + "=" * 80)
    print("  FALSE-POSITIVE CHROMOSOMAL DISTRIBUTION")
    print("=" * 80)
    for (h2, nq), grp in summary_df.groupby(["h2", "n_qtns"]):
        print(f"\n  h2={h2}, QTNs={nq}:")
        for _, r in grp.iterrows():
            print(
                f"    {r['tool']:20s}: "
                f"FP on QTN-chr = {r['frac_fp_qtn_chr_mean']:.1%} "
                f"[{r['frac_fp_qtn_chr_ci_lo']:.1%}, {r['frac_fp_qtn_chr_ci_hi']:.1%}] "
                f"(expected {r['expected_frac_mean']:.1%})  "
                f"mean FPs/rep = {r['mean_fp_per_rep']:.1f}  "
                f"(n={r['n_reps_with_fp']}/{r['n_reps_total']})"
            )

    return summary_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["sim", "null", "both", "window", "ci", "fp-chrom"],
        default="both",
    )
    args = parser.parse_args()

    if args.mode in ("sim", "both"):
        evaluate_all_scenarios()
    if args.mode in ("null", "both"):
        evaluate_null()
    if args.mode == "ci":
        summarize_with_ci()
    if args.mode == "window":
        evaluate_window_sensitivity()
    if args.mode == "fp-chrom":
        evaluate_fp_chromosomal_distribution()
