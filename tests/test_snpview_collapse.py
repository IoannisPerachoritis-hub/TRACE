"""collapse_snps_for_plotting builds its set of SNP IDs once, with the same result as before.

Before, the function rebuilt the set of all SNP IDs for every representative and for every
candidate SNP near it. That was the same set each time, so the result did not depend on it, but
with thousands of clustered significant SNPs and 185,618 SNPs (the rice run) the step took hours.
These tests compare the function with the earlier body, kept below, on clustered cases with many
candidates per representative: with the table in memory and with the table on disk.
"""
import numpy as np
import pandas as pd

from annotation import canon_chr
from gwas.genostore import LazyColumns, write_store
from gwas.plotting import compute_r2_to_lead
from gwas.snpview import collapse_snps_for_plotting


def _collapse_before(sig_df, geno_dosage_raw, sid, *, r2_threshold=0.9, flank_bp=300_000, min_pair_n=15):
    """The body of collapse_snps_for_plotting at release af83e6a, with its set built per SNP."""
    sid_arr = np.asarray(sid).astype(str)
    order = sig_df.assign(_c=sig_df["Chr"].map(lambda c: (0, int(canon_chr(c))) if str(canon_chr(c)).isdigit()
                                               else (1, str(canon_chr(c))))) \
                  .sort_values(["PValue", "_c", "Pos", "SNP"], kind="stable").index
    rep_of, rep_r2, reps = {}, {}, []
    pos_by_snp = dict(zip(sig_df["SNP"].astype(str), sig_df["Pos"].astype(int)))
    chr_by_snp = dict(zip(sig_df["SNP"].astype(str), sig_df["Chr"].map(canon_chr)))
    for idx in order:
        snp = str(sig_df.at[idx, "SNP"])
        if snp in rep_of:
            continue
        rep_of[snp] = snp
        rep_r2[snp] = 1.0
        reps.append(snp)
        if geno_dosage_raw is None or snp not in set(sid_arr):
            continue
        cc, cpos = chr_by_snp[snp], pos_by_snp[snp]
        cand = [str(s) for s in sig_df["SNP"].astype(str)
                if s not in rep_of and chr_by_snp.get(str(s)) == cc
                and abs(pos_by_snp.get(str(s), cpos) - cpos) <= flank_bp and str(s) in set(sid_arr)]
        if not cand:
            continue
        mask = np.isin(sid_arr, cand)
        try:
            r2, _ = compute_r2_to_lead(geno_dosage_raw, sid_arr, snp, mask, min_pair_n)
        except ValueError:
            continue
        for s, val in zip(sid_arr[mask], np.asarray(r2, float)):
            if np.isfinite(val) and val >= r2_threshold and str(s) not in rep_of:
                rep_of[str(s)] = snp
                rep_r2[str(s)] = float(val)
    counts = {}
    for s, rep in rep_of.items():
        counts[rep] = counts.get(rep, 0) + 1
    out = pd.DataFrame({
        "SNP": sig_df["SNP"].astype(str).values,
        "Representative_SNP": [rep_of.get(str(s), str(s)) for s in sig_df["SNP"]],
        "r2_to_representative": [rep_r2.get(str(s), 1.0) for s in sig_df["SNP"]],
    })
    out["Representative_Of_Count"] = out["Representative_SNP"].map(lambda r: counts.get(r, 1)).astype(int)
    return out


def _clustered_case(seed):
    """Samples x SNPs calls with groups of near-identical SNPs, 1% missing calls, and a table of
    significant SNPs crowded into 200 kb on chromosome 1 (every one a candidate of every other),
    a looser cluster on chromosome 2, one significant SNP that is not in the table, a chromosome
    name that is not a number and tied p-values."""
    rng = np.random.default_rng(seed)
    n = 150
    cols, sid, chrom, pos = [], [], [], []
    for c, n_snps, span in (("1", 120, 200_000), ("2", 60, 2_000_000), ("Un", 6, 50_000)):
        hap = rng.integers(0, 2, size=(n, 2))
        for j in range(n_snps):
            if j % 4 == 0:                       # a new haplotype every fourth SNP
                hap = rng.integers(0, 2, size=(n, 2))
            h = hap.copy()
            flip = rng.random((n, 2)) < (0.01 if j % 3 else 0.2)
            h[flip] = 1 - h[flip]
            cols.append(h.sum(axis=1).astype(float))
            sid.append(f"{c}:{j}")
            chrom.append(c)
            pos.append(int(1_000 + j * span // n_snps))
    G = np.column_stack(cols)
    G[rng.random(G.shape) < 0.01] = np.nan
    sid = np.array(sid)
    pick = np.r_[np.arange(0, 120), np.arange(120, 180, 2), np.arange(180, 186)]
    sig = pd.DataFrame({"SNP": sid[pick], "Chr": np.array(chrom)[pick], "Pos": np.array(pos)[pick],
                        "PValue": np.round(rng.uniform(1e-12, 1e-6, size=len(pick)), 14)})
    sig.loc[sig.index[5:9], "PValue"] = sig["PValue"].iloc[5]          # ties
    absent = pd.DataFrame({"SNP": ["1:absent"], "Chr": ["1"], "Pos": [50_000], "PValue": [1e-13]})
    sig = pd.concat([sig, absent], ignore_index=True).sample(frac=1.0, random_state=seed)
    return sig.reset_index(drop=True), G, sid


def test_same_result_as_before_in_memory():
    for seed in (0, 1, 2):
        sig, G, sid = _clustered_case(seed)
        for thr in (0.9, 0.5):
            new = collapse_snps_for_plotting(sig, G, sid, r2_threshold=thr)
            old = _collapse_before(sig, G, sid, r2_threshold=thr)
            pd.testing.assert_frame_equal(new, old)
        assert (new["Representative_SNP"] != new["SNP"]).any()      # some SNPs did collapse
        assert new.loc[new["SNP"] == "1:absent", "Representative_SNP"].iloc[0] == "1:absent"


def test_same_result_as_before_on_disk(tmp_path):
    sig, G, sid = _clustered_case(3)
    lazy = LazyColumns(write_store(tmp_path / "s", G), np.arange(G.shape[1]), fill=None)
    new = collapse_snps_for_plotting(sig, lazy, sid)
    old = _collapse_before(sig, G, sid)
    pd.testing.assert_frame_equal(new, old)
    assert len(new) == len(sig) and list(new["SNP"]) == list(sig["SNP"])


def test_no_table_gives_each_snp_its_own_representative():
    sig, _, sid = _clustered_case(4)
    new = collapse_snps_for_plotting(sig, None, sid)
    pd.testing.assert_frame_equal(new, _collapse_before(sig, None, sid))
    assert (new["Representative_SNP"] == new["SNP"]).all()
