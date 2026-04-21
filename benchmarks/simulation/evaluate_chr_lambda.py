"""Chromosome-specific lambda_GC analysis.

Splits p-values into QTN-bearing vs null chromosomes and computes
lambda_GC for each subset. If null-chromosome lambda ~ 1.0 even
when genome-wide lambda >> 1.0, the inflation is from recovered
signal on causal chromosomes, not from miscalibration.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
SIM_DATA = ROOT / "benchmarks" / "simulation" / "sim_data"
SIM_SUMMARY = ROOT / "benchmarks" / "simulation" / "sim_summary"

TOOLS = (
    "platform_loco", "platform_global", "gapit", "rmvp",
    "farmcpu_loco", "gapit_farmcpu",
)


def compute_lambda_gc(pvals: np.ndarray) -> float:
    """Compute genomic inflation factor from p-values."""
    pvals = pvals[(pvals > 0) & (pvals < 1)]
    if len(pvals) < 10:
        return np.nan
    chisq = stats.chi2.ppf(1 - pvals, df=1)
    return float(np.median(chisq) / stats.chi2.ppf(0.5, df=1))


def evaluate_chr_lambda(
    sim_data_dir: Path = SIM_DATA,
    tools: tuple[str, ...] = TOOLS,
) -> pd.DataFrame:
    """Compute per-chromosome lambda_GC across all reps and tools."""
    all_rows = []
    scenario_dirs = sorted(sim_data_dir.glob("h2_*/rep_*"))
    n_dirs = len(scenario_dirs)

    for i, sd in enumerate(scenario_dirs):
        truth_path = sd / "truth.json"
        if not truth_path.exists():
            continue
        with open(truth_path) as f:
            truth = json.load(f)

        qtn_chrs = set(str(c) for c in truth["qtn_chrs"])

        if (i + 1) % 100 == 0:
            print(f"  Processing {i + 1}/{n_dirs} ...")

        for tool in tools:
            results_path = sd / tool / "results.csv"
            if not results_path.exists():
                continue

            df = pd.read_csv(results_path).dropna(subset=["PValue"])
            df["Chr"] = df["Chr"].astype(str)

            # Global lambda — pooled median over all SNPs.
            lambda_global = compute_lambda_gc(df["PValue"].values)

            # QTN-chr / Null-chr lambdas — pooled medians over the SNP
            # subset living on QTN-bearing vs null chromosomes. Same
            # statistic as Global, applied to disjoint subsets, so within
            # every rep min(qtn, null) ≤ global ≤ max(qtn, null) holds
            # (median-of-union bound). Previous version averaged
            # per-chromosome lambdas, which is a different statistic and
            # produced apparent paradoxes at low h² where Global fell
            # below both subset means due to chromosome-size weighting.
            mask_qtn = df["Chr"].isin(qtn_chrs)
            qtn_pvals = df.loc[mask_qtn, "PValue"].values
            null_pvals = df.loc[~mask_qtn, "PValue"].values
            lambda_qtn = compute_lambda_gc(qtn_pvals)
            lambda_null = compute_lambda_gc(null_pvals)

            n_null_chrs_present = df.loc[~mask_qtn, "Chr"].nunique()

            all_rows.append({
                "scenario": truth["scenario"],
                "rep": truth["rep"],
                "h2": truth["h2_target"],
                "n_qtns": truth["n_qtns"],
                "tool": tool,
                "lambda_gc_global": lambda_global,
                "lambda_gc_qtn_chr": lambda_qtn,
                "lambda_gc_null_chr": lambda_null,
                "n_qtn_chrs": len(qtn_chrs),
                "n_null_chrs": n_null_chrs_present,
            })

    return pd.DataFrame(all_rows)


def print_summary(df: pd.DataFrame):
    """Print human-readable summary."""
    print("\n" + "=" * 100)
    print("  CHROMOSOME-SPECIFIC LAMBDA_GC: QTN-BEARING vs NULL CHROMOSOMES")
    print("=" * 100)

    # Focus on MLM tools
    mlm_tools = ["platform_loco", "platform_global", "gapit", "rmvp"]
    tool_labels = {
        "platform_loco": "TRACE-LOCO",
        "platform_global": "TRACE-Global",
        "gapit": "GAPIT3",
        "rmvp": "rMVP",
    }

    print(f"\n  {'Scenario':<20s} {'Tool':<16s} "
          f"{'Global':>8s} {'QTN-chr':>8s} {'Null-chr':>8s} {'Ratio':>8s}")
    print("  " + "-" * 72)

    for (h2, nq), sg in df.groupby(["h2", "n_qtns"]):
        for tool in mlm_tools:
            tg = sg[sg["tool"] == tool]
            if len(tg) == 0:
                continue
            label = f"h2={h2}, q={int(nq)}"
            gl = tg["lambda_gc_global"].mean()
            ql = tg["lambda_gc_qtn_chr"].mean()
            nl = tg["lambda_gc_null_chr"].mean()
            ratio = ql / nl if nl > 0 else np.nan
            print(f"  {label:<20s} {tool_labels[tool]:<16s} "
                  f"{gl:>8.2f} {ql:>8.2f} {nl:>8.2f} {ratio:>8.1f}x")
        print()

    # Key result: null-chromosome lambda across all scenarios for LOCO
    loco = df[df["tool"] == "platform_loco"]
    print("\n  KEY RESULT: TRACE-LOCO null-chromosome lambda_GC")
    print("  " + "-" * 50)
    for (h2, nq), sg in loco.groupby(["h2", "n_qtns"]):
        nl = sg["lambda_gc_null_chr"]
        print(f"  h2={h2}, q={int(nq)}: "
              f"mean={nl.mean():.3f}, "
              f"median={nl.median():.3f}, "
              f"95%CI=[{nl.quantile(0.025):.3f}, {nl.quantile(0.975):.3f}]")


def main():
    print("Chromosome-Specific Lambda_GC Analysis")
    print("=" * 40)

    df = evaluate_chr_lambda()
    print(f"  Total rows: {len(df):,}")

    SIM_SUMMARY.mkdir(parents=True, exist_ok=True)
    out_path = SIM_SUMMARY / "chr_specific_lambda.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    print_summary(df)


if __name__ == "__main__":
    main()
