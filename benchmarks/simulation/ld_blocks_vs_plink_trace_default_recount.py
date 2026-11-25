"""The recount of ld_blocks_vs_plink_trace_default.py's TRACE rows: only the candidate intervals
that hold at least one Bonferroni-significant SNP (no new block detection).

"Hold" is TRACE's own positional rule (gwas/isolated.py, find_unblocked_significant_snps): a block
holds a SNP when the SNP lies within the block's Start to End. An interval of unblocked SNPs holds
its significant SNPs by construction. A causal SNP is captured when it lies inside a counted
candidate interval (start <= position <= end, as before).

The simulation's saved intervals keep each interval's candidate bounds (from the marker strictly
before the block's first SNP to the marker strictly after its last, with the edge flank at a
chromosome end and the 5 Mb clamp), not the block's own Start and End. With no interval clamped,
the markers strictly inside the candidate bounds are the block's Start to End markers, so a block
holds a significant SNP when one of its replicate's significant SNPs lies at start < pos < end.
The script checks that rule first:
(a) on the real runs' tables, which keep both bounds, over every block: the markers strictly
    inside the candidate interval equal the markers within Start to End (the run's GWAS table
    gives the markers);
(b) on the simulation: every block with a significant member holds a significant SNP by the rule,
    and no interval is wider than 5 Mb.
Then a self-check: with every interval counted, the recount's aggregation (the committed
module's summarise) reproduces the as-run TRACE rows of the committed tables exactly. Any failed
check stops the script.

    python benchmarks/simulation/ld_blocks_vs_plink_trace_default_recount.py --work DIR --out DIR \\
        --tomato-zip TOMATO.zip --rice-zip RICE.zip --rice-gene-model locus.gff [--code TREE]

--work: the simulation's work folder (intervals.csv, causal_by_method.csv, sim_facts.json).
Writes, in --out: ld_blocks_vs_plink_trace_default_recount_main.csv, ..._recount.csv (per
scenario, then the pooled 5-QTN rows) and ..._recount_loci.csv, with two TRACE rows (the as-run
rows, method trace_default, copied from the committed tables, and the recount, method
trace_default_bonferroni) and the four PLINK rows copied unchanged; recount_facts.json.
"""
import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import ld_blocks_vs_plink_trace_default as td   # the committed P-2c module (it imports --code's gwas)

p2, annotation = td.p2, td.annotation
HERE = Path(__file__).resolve().parent
RECOUNT = "trace_default_bonferroni"
MAX_BP = 5_000_000


def lines(path):
    return [x.rstrip(b"\r") for x in Path(path).read_bytes().split(b"\n") if x.strip()]


def field(line, header, name):
    return line.decode().split(",")[header.decode().split(",").index(name)]


def csv_lines(rows, cols):
    buf = io.StringIO()
    pd.DataFrame(rows, columns=cols).to_csv(buf, index=False, header=False, lineterminator="\n")
    return [x.encode() for x in buf.getvalue().split("\n") if x]


def stop(msg):
    raise SystemExit(f"stop: {msg}")


# ----------------------------------------------------------------------------------------------
# (a) the rule on the real runs' tables

def check_rule_on_runs(zips):
    out = {}
    for name, zpath in zips.items():
        meta, g, sig, tblocks, bframe, samples = p2.zip_run(zpath)
        pos_by_chr = {c: np.unique(gg["Pos"].astype(np.int64).to_numpy()) for c, gg in g.groupby("Chr")}
        bad = []
        for r in bframe.to_dict("records"):
            pos = pos_by_chr.get(str(r["Chr"]), np.array([], dtype=np.int64))
            inside = pos[(pos > int(r["Interval_start"])) & (pos < int(r["Interval_end"]))]
            within = pos[(pos >= int(r["Start (bp)"])) & (pos <= int(r["End (bp)"]))]
            if not np.array_equal(inside, within):
                bad.append(f"chr{r['Chr']}:{r['Start (bp)']}-{r['End (bp)']}")
        out[name] = dict(blocks=int(len(bframe)), equal=int(len(bframe)) - len(bad), differ=bad)
    return out


# ----------------------------------------------------------------------------------------------
# the simulation

def sim_recount(work, sim_data, thr):
    ivd = pd.read_csv(work / "intervals.csv", dtype={"chr": str})
    cd = pd.read_csv(work / "causal_by_method.csv", dtype={"chr": str})
    ivd["holds"] = ivd["n_blocks"] == 0
    held_by_rule, n_sig_but_not_held = 0, 0
    for (scen, rep), idx in ivd.groupby(["scenario", "rep"]).groups.items():
        res = pd.read_csv(Path(sim_data) / scen / rep / "platform_loco" / "results.csv").dropna(subset=["PValue"])
        res["Chr"] = res["Chr"].astype(str)
        sig = res[res["PValue"] < thr]
        by_chr = {c: np.sort(s["Pos"].astype(np.int64).to_numpy()) for c, s in sig.groupby("Chr")}
        for i in idx:
            if ivd.at[i, "n_blocks"] != 1:
                continue
            p = by_chr.get(ivd.at[i, "chr"])
            h = bool(p is not None and ((p > ivd.at[i, "start"]) & (p < ivd.at[i, "end"])).any())
            ivd.at[i, "holds"] = h
            held_by_rule += h
            if ivd.at[i, "n_sig"] >= 1 and not h:
                n_sig_but_not_held += 1
    return ivd, cd, held_by_rule, n_sig_but_not_held


def captured(ivd, cd):
    """Each causal instance captured by the given intervals (start <= pos <= end)."""
    spans = {k: g[["chr", "start", "end"]].to_numpy() for k, g in ivd.groupby(["scenario", "rep"])}
    out = []
    for r in cd.itertuples(index=False):
        s = spans.get((r.scenario, r.rep))
        out.append(bool(s is not None and any(c == r.chr and a <= r.pos <= b for c, a, b in s)))
    return np.array(out, dtype=bool)


def rows_for(ivd, cd):
    per = []
    for scen in sorted(cd.scenario.unique()):
        per += td.summarise(ivd[ivd.scenario == scen], cd[cd.scenario == scen], scen)
    pooled = td.summarise(ivd[ivd.scenario.str.endswith("_q005")], cd[cd.scenario.str.endswith("_q005")],
                          "pooled_5qtn")
    return per + pooled


# ----------------------------------------------------------------------------------------------
# the loci

def loci_rows(a, count_all):
    """The loci rows of TRACE, as the committed module builds them, over every candidate interval
    (count_all) or only those that hold a Bonferroni SNP (the run's Significant_Bonf column; a
    block holds a SNP within its Start to End)."""
    rows, facts = [], {}
    loci = [("tomato_locule_chr2", a.tomato_zip, None), ("rice_GS3", a.rice_zip, "GS3"), ("rice_GL7", a.rice_zip, "GL7")]
    gene_models = {"tomato": p2.gene_counter(annotation.load_gene_model(p2.REPO / "data" / "Sol_genes_SL3.csv")),
                   "rice": p2.gene_counter(annotation.load_gene_model(a.rice_gene_model))}
    for locus, zpath, gene in loci:
        meta, g, sig, tblocks, bframe, samples = p2.zip_run(zpath)
        zf = zipfile.ZipFile(zpath)
        iso_name = [n for n in zf.namelist() if n.endswith("Isolated_SNP_intervals_MLM.csv")]
        isot = pd.read_csv(io.BytesIO(zf.read(iso_name[0]))) if iso_name else pd.DataFrame()
        bonf = g[g["Significant_Bonf"].astype(str) == "True"]
        bonf_by_chr = {c: np.sort(s["Pos"].astype(np.int64).to_numpy()) for c, s in bonf.groupby("Chr")}
        cand = []
        for r in bframe.to_dict("records"):
            ch = annotation.canon_chr(r["Chr"])
            p = bonf_by_chr.get(ch)
            holds = bool(p is not None and ((p >= int(r["Start (bp)"])) & (p <= int(r["End (bp)"]))).any())
            cand.append(dict(chr=ch, start=int(r["Interval_start"]), end=int(r["Interval_end"]),
                             members=td.snp_list(r["SNP_IDs"]), kind="LD block", holds=holds))
        cand += [dict(chr=annotation.canon_chr(r["Chr"]), start=int(r["Start (bp)"]), end=int(r["End (bp)"]),
                      members=td.snp_list(r["SNP_IDs"]), kind="unblocked SNP", holds=True)
                 for r in isot.to_dict("records")]
        if not count_all:
            cand = [c for c in cand if c["holds"]]
        sig_ids = set(sig["SNP"].astype(str))
        if gene is None:
            ch = "2"
            b2 = bframe[bframe["Chr"].astype(str) == "2"]
            lead = str(b2.loc[b2["lead_snp_pvalue"].astype(float).idxmin()]["lead_snp"]) if len(b2) else None
            here = [c for c in cand if c["chr"] == ch]
            count_genes, gene_iv, causal = gene_models["tomato"], None, None
        else:
            gid, ch, s, e = p2.RICE_GENES[gene]
            lo_, hi_ = s - p2.RICE_WINDOW, e + p2.RICE_WINDOW
            w = g[(g["Chr"] == ch) & (g["Pos"] >= lo_) & (g["Pos"] <= hi_)]
            lead = w.sort_values(["PValue", "Pos"]).iloc[0]["SNP"] if len(w) else None
            here = [c for c in cand if c["chr"] == ch and c["end"] >= lo_ and c["start"] <= hi_]
            count_genes, gene_iv = gene_models["rice"], (ch, s, e)
            causal = p2.GS3_CAUSAL if gene == "GS3" else None
            win_sig = set(sig[(sig["Chr"] == ch) & (sig["Pos"] >= lo_) & (sig["Pos"] <= hi_)]["SNP"].astype(str))
            facts[f"{locus}_window"] = dict(
                window=f"chr{ch}:{lo_}-{hi_}", candidate_intervals_overlapping=len(here),
                holding_a_significant_snp_of_the_window=sum(any(m in win_sig for m in c["members"]) for c in here),
                significant_snps_in_window=len(win_sig))
        held = {}
        for c in here:
            for m in c["members"]:
                if m in sig_ids:
                    held[m] = held.get(m, 0) + 1
        multi = sum(1 for v in held.values() if v > 1)
        for c in sorted(here, key=lambda d: (d["start"], d["end"])):
            rows.append(dict(locus=locus, method=td.METHOD if count_all else RECOUNT, chr=c["chr"], start=c["start"],
                             end=c["end"], width_kb=(c["end"] - c["start"]) / 1000.0, n_snps=len(c["members"]),
                             n_sig=sum(m in sig_ids for m in c["members"]),
                             n_genes=count_genes(c["chr"], c["start"], c["end"]),
                             holds_lead_snp=lead in c["members"],
                             gene_inside=(None if gene_iv is None else
                                          bool(c["chr"] == gene_iv[0] and c["start"] <= gene_iv[2] and c["end"] >= gene_iv[1])),
                             causal_inside=(None if causal is None else
                                            bool(c["chr"] == causal[0] and c["start"] <= causal[1] <= c["end"])),
                             sig_snps_in_two_blocks=multi))
        facts[locus] = dict(lead_snp=lead, candidate_intervals=len(here),
                            holding_lead=sum(lead in c["members"] for c in here),
                            blocks_excluded=0 if count_all else None,
                            bonferroni_set_equals_significant_table=set(bonf["SNP"].astype(str)) == sig_ids)
    return rows, facts


def main(argv=None):
    ap = argparse.ArgumentParser(description="The recount of TRACE's default rows over intervals holding a Bonferroni SNP")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--code", default=None)
    ap.add_argument("--summary", default=str(HERE / "sim_summary"))
    ap.add_argument("--sim-data", default=str(p2.SIM))
    ap.add_argument("--tomato-zip", required=True)
    ap.add_argument("--rice-zip", required=True)
    ap.add_argument("--rice-gene-model", required=True)
    a = ap.parse_args(argv)
    work, out, summ = Path(a.work), Path(a.out), Path(a.summary)
    out.mkdir(parents=True, exist_ok=True)
    facts = dict(gwas_file=td.gwas.__file__, module=td.__file__, script=str(Path(__file__).resolve()))
    sim_facts = json.loads((work / "sim_facts.json").read_text(encoding="utf-8"))
    thr = float(sim_facts["rule"]["p_threshold"])
    facts["threshold"] = repr(thr)

    # (a) the rule on the real runs' tables
    facts["check_a"] = check_rule_on_runs({"tomato": a.tomato_zip, "rice": a.rice_zip})
    if any(v["differ"] for v in facts["check_a"].values()):
        (out / "recount_facts.json").write_text(json.dumps(facts, indent=1, default=str), encoding="utf-8")
        stop("check (a): the markers inside a candidate interval differ from the block's Start to End markers")

    # the simulation: which blocks hold a Bonferroni SNP
    ivd, cd, held, nsig_not_held = sim_recount(work, a.sim_data, thr)
    blocks = ivd["n_blocks"] == 1
    facts["check_b"] = dict(blocks_with_a_significant_member=int((blocks & (ivd.n_sig >= 1)).sum()),
                            of_them_not_holding_by_the_rule=int(nsig_not_held),
                            widest_interval_bp=int((ivd.end - ivd.start).max()),
                            intervals_wider_than_5_mb=int(((ivd.end - ivd.start) > MAX_BP).sum()),
                            unblocked_intervals_without_a_significant_member=int(((ivd.n_blocks == 0) & (ivd.n_sig < 1)).sum()))
    if nsig_not_held or facts["check_b"]["intervals_wider_than_5_mb"]:
        (out / "recount_facts.json").write_text(json.dumps(facts, indent=1, default=str), encoding="utf-8")
        stop("check (b) failed")

    # self-check: every interval counted reproduces the as-run TRACE rows
    header = lines(summ / "ld_blocks_vs_plink_trace_default.csv")[0]
    cols = header.decode().split(",")
    cd_all = cd.copy()
    cd_all["captured"] = captured(ivd, cd)
    all_rows = rows_for(ivd, cd_all)
    as_run = {}
    for name in ("ld_blocks_vs_plink_trace_default.csv", "ld_blocks_vs_plink_trace_default_main.csv"):
        ls = lines(summ / name)
        as_run[name] = [x for x in ls[1:] if field(x, ls[0], "method") == td.METHOD]
    mine_all = csv_lines(all_rows, cols)
    mine_pooled = [x for x in mine_all if x.startswith(b"pooled_5qtn,")]
    loci_header = lines(summ / "ld_blocks_vs_plink_trace_default_loci.csv")[0]
    loci_cols = loci_header.decode().split(",")
    lr_all, _ = loci_rows(a, count_all=True)
    as_run_loci = [x for x in lines(summ / "ld_blocks_vs_plink_trace_default_loci.csv")[1:]
                   if field(x, loci_header, "method") == td.METHOD]
    mine_loci_all = [x.replace(b"\r", b"") for x in csv_lines(lr_all, loci_cols)]
    facts["self_check"] = dict(
        captured_recomputed_equals_saved=bool((cd_all["captured"].to_numpy() == cd["captured"].astype(bool).to_numpy()).all()),
        per_scenario_rows_equal=mine_all == as_run["ld_blocks_vs_plink_trace_default.csv"],
        main_rows_equal=mine_pooled == as_run["ld_blocks_vs_plink_trace_default_main.csv"],
        loci_rows_equal=mine_loci_all == as_run_loci,
        rows=dict(per_scenario=len(mine_all), main=len(mine_pooled), loci=len(mine_loci_all)))
    if not all(v for k, v in facts["self_check"].items() if k != "rows"):
        (out / "recount_facts.json").write_text(json.dumps(facts, indent=1, default=str), encoding="utf-8")
        stop("the self-check failed: every interval counted does not reproduce the as-run rows")

    # the recount
    counted = ivd[ivd["holds"]]
    cd_rec = cd.copy()
    cd_rec["captured"] = captured(counted, cd)
    rec_rows = rows_for(counted, cd_rec)
    for r in rec_rows:
        r["method"] = RECOUNT
    excluded = ivd[blocks & ~ivd["holds"]]
    facts["counts"] = dict(
        block_intervals=int(blocks.sum()), counted_blocks=int((blocks & ivd["holds"]).sum()),
        excluded_blocks=int(len(excluded)),
        sum_equals_block_intervals=int((blocks & ivd["holds"]).sum()) + int(len(excluded)) == int(blocks.sum()),
        unblocked_intervals=int((ivd.n_blocks == 0).sum()), counted_intervals=int(len(counted)),
        causal_captured_as_run=int(cd_all["captured"].sum()), causal_captured_recount=int(cd_rec["captured"].sum()),
        causal_captured_as_run_not_in_recount=int((cd_all["captured"] & ~cd_rec["captured"]).sum()),
        causal_captured_in_recount_not_as_run=int((~cd_all["captured"] & cd_rec["captured"]).sum()),
        true_hits_in_excluded_blocks=int(excluded["true_hit"].sum()),
        true_hits_as_run=int(ivd["true_hit"].sum()), true_hits_recount=int(counted["true_hit"].sum()))
    pd.DataFrame(rec_rows).to_csv(out / "trace_default_bonferroni_rows.csv", index=False)
    rec_lines = csv_lines(rec_rows, cols)
    rec_by = {(field(x, header, "scenario"), field(x, header, "ld_group")): x for x in rec_lines}

    # the tables: as-run TRACE rows, the recount, the four PLINK rows, all but the recount copied
    for name, out_name in (("ld_blocks_vs_plink_trace_default.csv", "ld_blocks_vs_plink_trace_default_recount.csv"),
                           ("ld_blocks_vs_plink_trace_default_main.csv", "ld_blocks_vs_plink_trace_default_recount_main.csv")):
        ls = lines(summ / name)
        body = []
        for x in ls[1:]:
            body.append(x)
            if field(x, ls[0], "method") == td.METHOD and field(x, ls[0], "ld_group") == "all":
                scen = field(x, ls[0], "scenario")
                body += [rec_by[(scen, g)] for g in ("short", "long", "all")]
        (out / out_name).write_bytes(b"\r\n".join([ls[0]] + body) + b"\r\n")
    lr_rec, lfacts = loci_rows(a, count_all=False)
    ls = lines(summ / "ld_blocks_vs_plink_trace_default_loci.csv")
    rec_loci = [x.replace(b"\r", b"") for x in csv_lines(lr_rec, loci_cols)]
    body = [x for x in ls[1:] if field(x, ls[0], "method") == td.METHOD] + rec_loci + \
           [x for x in ls[1:] if field(x, ls[0], "method") != td.METHOD]
    (out / "ld_blocks_vs_plink_trace_default_recount_loci.csv").write_bytes(b"\r\n".join([ls[0]] + body) + b"\r\n")
    facts["loci_recount"] = lfacts
    # every copied line equals its committed source
    copied = {}
    for name, out_name in (("ld_blocks_vs_plink_trace_default.csv", "ld_blocks_vs_plink_trace_default_recount.csv"),
                           ("ld_blocks_vs_plink_trace_default_main.csv", "ld_blocks_vs_plink_trace_default_recount_main.csv"),
                           ("ld_blocks_vs_plink_trace_default_loci.csv", "ld_blocks_vs_plink_trace_default_recount_loci.csv")):
        src = lines(summ / name)
        new = [x for x in lines(out / out_name)[1:] if field(x, src[0], "method") != RECOUNT]
        copied[out_name] = dict(lines_other_than_the_recount=len(new), equal_to_the_committed_table=new == src[1:])
    facts["copied_rows"] = copied
    (out / "recount_facts.json").write_text(json.dumps(facts, indent=1, default=str), encoding="utf-8")
    print(json.dumps(facts, indent=1, default=str))


if __name__ == "__main__":
    main()
