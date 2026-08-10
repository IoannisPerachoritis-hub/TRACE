"""T-19 / T-50 — Omission census (counts only).

Sizes the two silent-disappearance paths on the tomato locule-number demo: how
many reporting-threshold-significant SNPs never reach the post-GWAS/haplotype
view because they either (Route A, "block_formation") seeded but formed no
>=min_snps block, or (Route B, "seeding_threshold") were significant under the
*reporting* rule yet fell above the *seeding* threshold (1e-5) and outside the
top-N, so were never offered to the detector at all.

Pure set-difference — no detector call, no rescue code (Batch C). Inputs: the
committed MLM GWAS table (carries `Significant_{Meff,Bonf,FDR}`) and the
reproduced six-block Table S8 set (`tests/golden/tomato_locule/ld_blocks.csv`,
from T-61). Coverage is positional and inclusive (`cli.py:1110`,
`gwas/ld.py:678-681`).

Run:  .venv/Scripts/python benchmarks/omission_census.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from annotation import canon_chr  # noqa: E402

QC_DIR = ROOT / "benchmarks" / "qc_data" / "tomato_locule_number"
GWAS_CSV = QC_DIR / "platform_GWAS_locule_number.csv"
BLOCKS_CSV = ROOT / "tests" / "golden" / "tomato_locule" / "ld_blocks.csv"

SEED_P = 1e-5          # --ld-seed-p default (cli.py:133)
TOP_N = 10             # --ld-top-n default (cli.py:135)
MEFF = 242.0           # genome-wide Li&Ji M_eff for this run (T-61 manifest); manuscript states ~250
MEFF_THRESH = 0.05 / MEFF   # ~2.07e-4 (manuscript ~2e-4)


def _covered_mask(df, blocks):
    """Positional, inclusive coverage against the block intervals."""
    ivs = [(canon_chr(r["Chr"]), int(r["Start (bp)"]), int(r["End (bp)"]))
           for _, r in blocks.iterrows()]
    chr_ = df["Chr"].map(lambda c: canon_chr(c)).values
    pos = df["Pos"].astype(int).values
    covered = np.zeros(len(df), bool)
    for bc, s, e in ivs:
        covered |= (chr_ == bc) & (pos >= s) & (pos <= e)
    return covered


def _snpid_member_mask(df, blocks):
    members = set()
    for ids in blocks["SNP_IDs"].dropna():
        members.update(x for x in str(ids).split(",") if x)
    return df["SNP"].astype(str).isin(members).values


def main():
    gwas = pd.read_csv(GWAS_CSV)
    blocks = pd.read_csv(BLOCKS_CSV)
    n_scanned = len(gwas)

    top_set = set(gwas.nsmallest(TOP_N, "PValue")["SNP"].astype(str))
    gwas = gwas.copy()
    gwas["_covered"] = _covered_mask(gwas, blocks)
    gwas["_member"] = _snpid_member_mask(gwas, blocks)
    gwas["_seeded"] = (gwas["PValue"] < SEED_P) | (gwas["SNP"].astype(str).isin(top_set))

    print("=" * 72)
    print("  T-19 / T-50 omission census — tomato locule-number (MLM)")
    print("=" * 72)
    print(f"  scanned SNPs: {n_scanned:,} | blocks: {len(blocks)} (Table S8 reproduced) | "
          f"seed_p={SEED_P:g} top_n={TOP_N}")
    print(f"  M_eff={MEFF:g}  ->  reporting threshold 0.05/M_eff = {MEFF_THRESH:.3e} "
          f"(manuscript ~2e-4); Bonferroni 0.05/m = {0.05/n_scanned:.3e}")

    rules = {
        "Significant_Meff": ("M_eff (reporting; manuscript rule)", MEFF_THRESH),
        "Significant_Bonf": ("Bonferroni (committed one-click rule)", 0.05 / n_scanned),
        "Significant_FDR": ("FDR<0.05", None),
    }

    report = {}
    for col, (label, thr) in rules.items():
        S = gwas[gwas[col]]
        nS = len(S)
        covered = int(S["_covered"].sum())
        uncovered = nS - covered
        unc = S[~S["_covered"]]
        route_b = int((~unc["_seeded"]).sum())       # never a seed
        route_a = int(unc["_seeded"].sum())           # seeded but no block
        report[col] = dict(n=nS, covered=covered, uncovered=uncovered,
                           route_a=route_a, route_b=route_b)
        print(f"\n  [{label}]  significant = {nS}")
        print(f"    (a) covered by an emitted block : {covered}")
        print(f"    (b) NOT covered                 : {uncovered}")
        print(f"        - Route A block_formation (seeded, <min_snps): {route_a}")
        print(f"        - Route B seeding_threshold (never a seed)   : {route_b}")

    # T-19 (c)/(d) diagnostics on the M_eff set
    S = gwas[gwas["Significant_Meff"]]
    band = S[(S["PValue"] >= SEED_P) & (S["PValue"] < MEFF_THRESH) & (~S["SNP"].astype(str).isin(top_set))]
    topn_only = S[(S["PValue"] >= SEED_P) & (S["SNP"].astype(str).isin(top_set))]
    print("\n  T-19 four-count summary (M_eff reporting rule, N=44):")
    print(f"    (a) covered by a block                         : {report['Significant_Meff']['covered']}")
    print(f"    (b) not covered                                : {report['Significant_Meff']['uncovered']}")
    print(f"    (c) in [1e-5, {MEFF_THRESH:.2e}) & not top-N -> never a seed : {len(band)}")
    print(f"    (d) entered ONLY via top-N (p>=1e-5, in top-{TOP_N})        : {len(topn_only)}")
    print(f"    check: (a)+(b) = {report['Significant_Meff']['covered']}+"
          f"{report['Significant_Meff']['uncovered']} = "
          f"{report['Significant_Meff']['covered']+report['Significant_Meff']['uncovered']} (== 44 expected)")

    print("\n  NOTE (discrepancy to flag): the committed one-click run used Bonferroni "
          f"({0.05/n_scanned:.2e}), which is STRICTER than the 1e-5 seed threshold, so its "
          "Route-B band is empty. The manuscript's 44 / M_eff framing is the benchmark run; "
          "under M_eff the Route-B band is populated.")
    return report, len(band), len(topn_only)


if __name__ == "__main__":
    main()
