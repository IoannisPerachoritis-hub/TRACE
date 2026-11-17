"""Table S10's tail recovery: how much of one tool's top 1% falls in the other's top 5%.

For each run whose QC folder is in the data root, TRACE-FarmCPU's table (rerun_trace_farmcpu.py,
the E_F_nogate2_ols arm) and GAPIT3 FarmCPU's table are merged on SNP (inner merge). Then:
- gapit_to_trace: the share of GAPIT3's top max(1, int(1% of N)) SNPs by p-value that are among
  TRACE's top int(5% of N), N the merged SNPs, ties taken in table order (pandas nsmallest,
  keep="first");
- trace_to_gapit: the same the other way.
It is computed twice: on GAPIT3's standardized table, which is FarmCPU's own table (the "(Kansas)"
file, and the file behind the rest of Table S10), and on GAPIT3's "(NYC)" file, where GAPIT's
regional re-scan has replaced the p-values of the markers near each significant bin.

    python benchmarks/tail_recovery.py [--data-root DIR]

Input: results/farmcpu_gatefree_concordance/<run>/platform_GWAS_FarmCPU_E_F_nogate2_ols_<trait>.csv
and results/gapit_farmcpu/<run>/ (gapit_farmcpu_results_standardized.csv and the (NYC) file).
Output: benchmarks/table_s10_tail_recovery.csv in the data root, one row per run and file. The
variant column names the definition: std (the standardized, Kansas, file) or nyc, then tl (the
TRACE table on the left of the merge), int (whole-number set sizes) and first.first (the tie rule
of each side).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks._options import options, runs_in_root  # noqa: E402

OPT = options()
ROOT = OPT.root   # the data root: this repository unless --data-root is given
RUNS = runs_in_root({"pepper_FWe": "FWe", "pepper_BX": "BX", "tomato_weight_g": "weight_g",
                     "tomato_locule_number": "locule_number"})
VARIANTS = ("std.tl.int.first.first", "nyc.tl.int.first.first")


def _top(m, col, k, rule):
    """The SNP set of the top k by p-value under a tie rule."""
    if rule in ("first", "last", "all"):
        return set(m.nsmallest(k, col, keep=rule)["SNP"])
    if rule in ("min", "max", "average"):
        return set(m.loc[m[col].rank(method=rule) <= k, "SNP"])
    if rule == "quicksort":
        return set(m.sort_values(col, kind="quicksort").head(k)["SNP"])
    if rule == "logdesc":          # -log10 p, descending, stable sort
        lp = -np.log10(m[col].clip(lower=1e-300))
        return set(m.assign(_lp=lp).sort_values("_lp", ascending=False, kind="mergesort").head(k)["SNP"])
    raise ValueError(rule)


def recovery(m, ref_col, tgt_col, sizes="int", ref_rule="first", tgt_rule="first"):
    n = len(m)
    f = int if sizes == "int" else (lambda x: int(round(x)))
    n_ref, n_tgt = max(1, f(n * 0.01)), f(n * 0.05)
    ref = _top(m, ref_col, n_ref, ref_rule)
    tgt = _top(m, tgt_col, n_tgt, tgt_rule)
    return len(ref & tgt) / len(ref), len(ref & tgt), len(ref), n_ref, n_tgt


def merged(trace, gapit, merge):
    a = trace[["SNP", "PValue"]].dropna().rename(columns={"PValue": "Pt"})
    b = gapit[["SNP", "PValue"]].dropna().rename(columns={"PValue": "Pg"})
    return pd.merge(a, b, on="SNP") if merge == "tl" else pd.merge(b, a, on="SNP")[["SNP", "Pt", "Pg"]]


def main():
    trace_dir = ROOT / "benchmarks" / "results" / "farmcpu_gatefree_concordance"
    rows = []
    for run, t in RUNS.items():
        trace = pd.read_csv(trace_dir / run / f"platform_GWAS_FarmCPU_E_F_nogate2_ols_{t}.csv")
        gdir = ROOT / "benchmarks" / "results" / "gapit_farmcpu" / run
        files = {"std": pd.read_csv(gdir / "gapit_farmcpu_results_standardized.csv"),
                 "nyc": pd.read_csv(gdir / f"GAPIT.Association.GWAS_Results.FarmCPU.{t}(NYC).csv"
                                    ).rename(columns={"P.value": "PValue"})}
        for v in VARIANTS:
            src, merge, sizes, gr, tr = v.split(".")
            m = merged(trace, files[src], merge)
            one = recovery(m, "Pg", "Pt", sizes, gr, tr)
            other = recovery(m, "Pt", "Pg", sizes, tr, gr)
            rows.append(dict(run=run, variant=v, n_merged=len(m), gapit_to_trace=one[0], hits=one[1],
                             n_ref=one[2], n_top5=one[4], trace_to_gapit=other[0], hits_other=other[1]))
    df = pd.DataFrame(rows)
    out = ROOT / "benchmarks" / "table_s10_tail_recovery.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    for v, g in df.groupby("variant"):
        print(f"{v}: mean one way {g['gapit_to_trace'].mean():.4f}, the other {g['trace_to_gapit'].mean():.4f}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
