"""Table S14's three added columns, computed from Table S14's published values (no runs).

For each scenario row of Table S14 (the subsampling filter on the simulated LD blocks): the whole
numbers a (TP blocks passing the 50% filter) in 0..TP and b (FP blocks passing) in 0..FP whose
percentages round to the published pass rates (one decimal; half up, half even and Python's own
percent format are all tried), how many fit, the gap from the unrounded fractions,
100 (a/TP - b/FP), against the published gap, then:
- precision without the filter: TP / (TP + FP);
- precision with the filter: a / (a + b);
- the share of true blocks kept: a / TP.

    python benchmarks/simulation/table_s14_columns.py

Input: the four rows below, as Table S14 prints them. Output:
benchmarks/simulation/sim_summary/table_s14_columns.csv.
"""
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "sim_summary" / "table_s14_columns.csv"

# Table S14 as printed: scenario (h2 / QTNs), TP blocks, FP blocks, TP pass >50%, FP pass >50%, gap (pp)
TABLE_S14 = [
    ("0.5 / 5", 87, 850, "78.2", "45.6", "32.5"),
    ("0.8 / 5", 120, 1564, "75.8", "52.3", "23.5"),
    ("0.8 / 15", 131, 1492, "59.5", "47.4", "12.2"),
    ("0.8 / 50", 94, 595, "28.7", "34.5", "-5.7"),
]


def fits(n, pub):
    """The whole numbers a in 0..n whose percentage of n prints as ``pub`` (one decimal)."""
    out = []
    for a in range(n + 1):
        x = Decimal(a) * 100 / Decimal(n)
        forms = {str(x.quantize(Decimal("0.1"), ROUND_HALF_UP)), str(x.quantize(Decimal("0.1"), ROUND_HALF_EVEN)),
                 f"{a / n:.1%}"[:-1]}
        if pub in forms:
            out.append(a)
    return out


def main():
    rows = []
    for scenario, tp, fp, tp_pass, fp_pass, gap_pub in TABLE_S14:
        ft, ff = fits(tp, tp_pass), fits(fp, fp_pass)
        for a in ft:
            for b in ff:
                gap = 100 * (a / tp - b / fp)
                p0, p1, kept = tp / (tp + fp), a / (a + b), a / tp
                rows.append(dict(scenario=scenario, tp_blocks=tp, fp_blocks=fp,
                                 tp_pass_published=tp_pass, fp_pass_published=fp_pass,
                                 tp_pass_n=a, fp_pass_n=b, n_tp_fits=len(ft), n_fp_fits=len(ff),
                                 gap_from_fractions=round(gap, 4), gap_published=gap_pub,
                                 gap_equal=f"{gap:.1f}" == gap_pub,
                                 precision_no_filter=round(p0, 4), precision_filter=round(p1, 4),
                                 true_blocks_kept=round(kept, 4), mark="computed from Table S14"))
                print(f"{scenario}: a = {a}/{tp}, b = {b}/{fp}; gap {gap:.4f} (published {gap_pub}); "
                      f"precision {p0:.4f} -> {p1:.4f}; true blocks kept {kept:.4f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
