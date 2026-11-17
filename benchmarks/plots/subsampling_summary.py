"""Figure S8's numbers: the subsampling of tomato locule number, summed up.

From the CLI run's subsampling tables (in its results ZIP) and the full-data MLM table:
- the resamples (all, and those with status ok) and the resample size;
- the full-data top SNP (the smallest MLM p) and its discovery frequency, and the SNPs with the
  highest frequency;
- the frequency of SL25ch02p47301921, the lead SNP of Table S8's lead block;
- the Bonferroni set (p < 0.05/m in the MLM table), each SNP's frequency, the lowest;
- the SNPs never found (count, percent), the SNPs found at least once, the ones among them that
  are not Bonferroni SNPs, and how many of those lie below 0.20.

    python benchmarks/plots/subsampling_summary.py --zip RESULTS.zip [--data-root DIR]

The ZIP comes from the CLI run in benchmarks/README.md (Figure S8). Input also:
qc_data/tomato_locule_number/platform_GWAS_locule_number.csv. Output:
benchmarks/plots/figure_s8_subsampling_summary.json in the data root.
"""
import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks._options import options  # noqa: E402

OPT = options()
ROOT = OPT.root   # the data root: this repository unless --data-root is given
S8_LEAD = "SL25ch02p47301921"


def summary(zpath, table_path):
    z = zipfile.ZipFile(zpath)
    disc = pd.read_csv(io.BytesIO(z.read("tables/Subsampling_SNP_stability.csv")))
    meta = pd.read_csv(io.BytesIO(z.read("tables/Subsampling_rep_metadata.csv")))
    mlm = pd.read_csv(table_path)
    m = len(mlm)
    bonf = mlm[mlm["PValue"] < 0.05 / m]
    top = mlm.loc[mlm["PValue"].idxmin()]
    f = dict(zip(disc["SNP"], disc["DiscoveryFreq"]))
    freq = disc["DiscoveryFreq"]
    nz = disc[freq > 0]
    nz_nonsig = nz[~nz["SNP"].isin(bonf["SNP"])]
    return dict(
        n_snps_table=m, n_snps_subsampling=len(disc),
        n_reps=len(meta), n_reps_ok=int((meta["status"] == "ok").sum()) if "status" in meta else None,
        resample_sizes=sorted(map(int, meta["n_samples"].unique())),
        top_snp=str(top["SNP"]), top_snp_p=float(top["PValue"]), top_snp_freq=f.get(top["SNP"]),
        max_freq=float(freq.max()), snps_at_max_freq=sorted(disc.loc[freq == freq.max(), "SNP"]),
        s8_lead=S8_LEAD, s8_lead_freq=f.get(S8_LEAD),
        bonferroni_threshold=0.05 / m, n_bonferroni=len(bonf),
        bonferroni_freqs={s: f.get(s) for s in bonf.sort_values("PValue")["SNP"]},
        bonferroni_min_freq=min(f.get(s, 0.0) for s in bonf["SNP"]) if len(bonf) else None,
        n_zero=int((freq == 0).sum()), pct_zero=100.0 * float((freq == 0).mean()),
        n_nonzero=len(nz), n_nonzero_bonferroni=int(nz["SNP"].isin(bonf["SNP"]).sum()),
        n_nonzero_nonsig=len(nz_nonsig),
        n_nonzero_nonsig_below_020=int((nz_nonsig["DiscoveryFreq"] < 0.20).sum()),
        share_nonzero_nonsig_below_020=float((nz_nonsig["DiscoveryFreq"] < 0.20).mean()) if len(nz_nonsig) else None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="the CLI run's results ZIP (with --subsampling)")
    a = ap.parse_args()
    table = ROOT / "benchmarks" / "qc_data" / "tomato_locule_number" / "platform_GWAS_locule_number.csv"
    out = ROOT / "benchmarks" / "plots" / "figure_s8_subsampling_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    s = summary(Path(a.zip), table)
    out.write_text(json.dumps(s, indent=1), encoding="utf-8")
    for k, v in s.items():
        print(f"{k}: {v}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
