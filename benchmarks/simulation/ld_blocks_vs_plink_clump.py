"""TRACE's LD blocks and PLINK's haplotype blocks against PLINK 1.9 clumping, from the same
significant SNPs.

ld_blocks_vs_plink.py compares TRACE's blocks with PLINK --blocks, a genome-wide partition made
from LD alone. The conventional method that, like TRACE, starts from the GWAS results is PLINK
clumping (--clump). This script adds it, on the same data and with the same metrics, as two
further methods:
- plink_clump_c1: --clump with --clump-r2, --clump-kb and --clump-p2 at PLINK's defaults (r2 0.5,
  250 kb and 0.01 in PLINK 1.9's own help text, `plink --help clump`);
- plink_clump_c2: --clump-r2 0.6 --clump-kb 300, TRACE's r2 threshold and flank in the published
  simulation evaluation, with --clump-p2 at its default.
In both, --clump-p1 is the analysis's Bonferroni threshold, so the index SNPs are the significant
SNPs. There is no --clump-allow-overlap and no --clump-range. PLINK reads the genotype data of
ld_blocks_vs_plink.py with its input options (--double-id --vcf-half-call missing --chr-set 12
--memory 1024; for rice its --bfile, --keep, --extract and --chr 3,7).

    python benchmarks/simulation/ld_blocks_vs_plink_clump.py prepare --vcf varitome_filtered.vcf \\
        --plink PATH_TO_PLINK --out DIR
    python benchmarks/simulation/ld_blocks_vs_plink_clump.py sim --plink PATH_TO_PLINK --out DIR [--jobs 4]
    python benchmarks/simulation/ld_blocks_vs_plink_clump.py loci --plink PATH_TO_PLINK --out DIR \\
        --vcf varitome_filtered.vcf --tomato-zip TOMATO.zip --rice-zip RICE.zip \\
        --rice-bfile PREFIX --rice-gene-model locus.gff
    python benchmarks/simulation/ld_blocks_vs_plink_clump.py accept --out DIR

prepare: ld_blocks_vs_plink.py's vcf and plink steps into DIR (the template check, the template's
raw calls as template_raw.vcf, PLINK's blocks at 200 and 600 kb).
sim: for each replicate, its p-values as a whitespace separated "SNP P" file (each value at full
precision, unchanged), PLINK --clump in both settings on template_raw.vcf, and then
ld_blocks_vs_plink.py's own replicate function with the replicate's clumps beside PLINK's blocks,
so the TRACE and --blocks rows come from that script's code. Writes DIR/ld_blocks_vs_plink_clump.csv
(per scenario, then the three 5-QTN scenarios pooled), DIR/ld_blocks_vs_plink_clump_main.csv (the
pooled rows), DIR/clump_runs.csv (per run: the clump options PLINK logged, the counts and the
checks), DIR/clumps_digest.csv (every clump) and DIR/clump_facts.json.
loci: ld_blocks_vs_plink.py's loci step, then the clumps of tomato chromosome 2 and of rice
chromosomes 3 and 7 from the runs' p-values. Writes DIR/ld_blocks_vs_plink_clump_loci.csv
(ld_blocks_vs_plink.py's rows as it wrote them, then the clump rows).
accept: the TRACE and --blocks rows of the three new tables against ld_blocks_vs_plink.py's
committed tables (sim_summary/), bytes first, then values; writes DIR/acceptance_p2_rows.json.

Definitions (fixed before the run):
- A clump's interval runs from the smallest to the largest position among its index SNP and the
  SNPs PLINK lists in SP2 (their "(1)" file suffix stripped). A clump of the index SNP alone has
  width 0. A significant SNP's interval is the clump that holds it, as index or as member
  (ld_blocks_vs_plink.intervals, with the clump's SNPs as its members).
- Metrics, LD groups and gene models: ld_blocks_vs_plink.py's.
- Checks: every significant SNP (p below the index threshold) is in exactly one clump; every
  clump's index SNP is significant; the clump options of each PLINK log.
- A run with no significant SNP has no .clumped file: zero clumps.
"""
import argparse
import csv
import io
import json
import logging
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import ld_blocks_vs_plink as p2   # the P-2 comparison, in this folder, unchanged

CLUMP_SETTINGS = {"plink_clump_c1": [], "plink_clump_c2": ["--clump-r2", "0.6", "--clump-kb", "300"]}
EXPECTED_OPTIONS = {"plink_clump_c1": {}, "plink_clump_c2": {"--clump-r2": "0.6", "--clump-kb": "300"}}
METHODS = p2.METHODS + list(CLUMP_SETTINGS)
VCF_INPUT = ["--double-id", "--vcf-half-call", "missing"]     # ld_blocks_vs_plink.py's VCF options
HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------------------------
# PLINK --clump

def write_assoc(path, snps, pvals):
    """The p-values as --clump reads them: "SNP P", whitespace separated, each value at full
    precision (Python's repr of the float, which reads back to the same value)."""
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("SNP P\n")
        for s, p in zip(snps, pvals):
            f.write(f"{s} {float(p)!r}\n")


def run_clump(plink, inputs, assoc, p1, out_prefix, extra):
    """PLINK 1.9 --clump with ld_blocks_vs_plink.py's options (--chr-set 12, 1,024 MB of workspace)."""
    cmd = ([str(plink)] + [str(x) for x in inputs] + ["--chr-set", "12", "--memory", "1024",
                                                       "--clump", str(assoc), "--clump-p1", repr(float(p1))]
           + extra + ["--out", str(out_prefix)])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"plink failed ({r.returncode}): {' '.join(cmd)}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return cmd


def read_log(log_path):
    """PLINK's log: the options in effect, the "--clump:" line and the warnings."""
    lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    opts, inside = {}, False
    for x in lines:
        if x.startswith("Options in effect:"):
            inside = True
            continue
        if inside:
            m = re.match(r"^\s+(--\S+)\s*(.*)$", x)
            if m:
                opts[m.group(1)] = m.group(2).strip()
                continue
            inside = False
    clump_line = " ".join(x for x in lines if x.startswith("--clump"))
    warnings = [x for x in lines if x.lower().startswith("warning")]
    return opts, clump_line, warnings, lines


def clump_options(opts):
    """The clump options of a log other than the report file and the index threshold."""
    return {k: v for k, v in opts.items() if k.startswith("--clump") and k not in ("--clump", "--clump-p1")}


def read_clumps(clumped, chr_pos):
    """PLINK's .clumped: one clump per row. Its SNPs are the index SNP and SP2's ("NONE" when
    there are none, each with its "(1)" file suffix stripped); its span runs from the smallest to
    the largest of their positions. Stops when a SNP of SP2 is not in the genotype data."""
    path = Path(clumped)
    if not path.is_file():
        return []
    df = pd.read_csv(path, sep=r"\s+", dtype={"CHR": str, "SNP": str, "SP2": str})
    clumps = []
    for k, r in enumerate(df.itertuples(index=False)):
        sp2 = [] if r.SP2 == "NONE" else [re.sub(r"\(\d+\)$", "", t) for t in r.SP2.split(",") if t]
        snps = [r.SNP] + sp2
        unknown = [s for s in snps if s not in chr_pos]
        if unknown:
            raise SystemExit(f"stop: {path.name}: SNPs not in the genotype data: {unknown[:5]}")
        chrs = {chr_pos[s][0] for s in snps}
        if chrs != {str(r.CHR)}:
            raise SystemExit(f"stop: {path.name}: a clump of chromosome {r.CHR} holds SNPs of {sorted(chrs)}")
        pos = [chr_pos[s][1] for s in snps]
        clumps.append(dict(id=k, chr=str(r.CHR), start=min(pos), end=max(pos), members=snps, index=r.SNP))
    return clumps


def clump_checks(sig_ids, clumps, pval, thr):
    """Each significant SNP in exactly one clump; each clump's index SNP significant."""
    count = {s: 0 for s in sig_ids}
    for c in clumps:
        for s in c["members"]:
            if s in count:
                count[s] += 1
    none = sorted(s for s, n in count.items() if n == 0)
    many = sorted(s for s, n in count.items() if n > 1)
    nonsig_index = sorted(c["index"] for c in clumps if not pval[c["index"]] < thr)
    return none, many, nonsig_index


def clump_run(plink, inputs, assoc, thr, prefix, setting, sig_ids, pval, chr_pos):
    """One --clump run: its command, clumps, logged options and checks."""
    cmd = run_clump(plink, inputs, assoc, thr, prefix, CLUMP_SETTINGS[setting])
    opts, line, warns, _ = read_log(f"{prefix}.log")
    clumps = read_clumps(f"{prefix}.clumped", chr_pos)
    none, many, nonsig = clump_checks(sig_ids, clumps, pval, thr)
    got = clump_options(opts)
    fact = dict(setting=setting, n_sig=len(sig_ids), n_clumps=len(clumps),
                sig_in_no_clump=len(none), sig_in_two_or_more=len(many), index_not_significant=len(nonsig),
                clump_p1_logged=opts.get("--clump-p1"), other_clump_options_logged=json.dumps(got, sort_keys=True),
                options_as_designed=(got == EXPECTED_OPTIONS[setting] and opts.get("--clump-p1") == repr(float(thr))),
                clump_line=line, warnings=" | ".join(warns),
                exception_snps=" ".join((none + many + nonsig)[:20]))
    return cmd, clumps, fact


# ----------------------------------------------------------------------------------------------
# metrics (ld_blocks_vs_plink.py's, over the five methods)

def summarise(iv, cs, label):
    rows = []
    for meth in METHODS:
        for grp in ["short", "long", "all"]:
            ci = cs[(cs.method == meth) & ((cs.group == grp) | (grp == "all"))]
            ii = iv[(iv.method == meth) & ((iv.group == grp) | (grp == "all"))] if len(iv) else iv
            tt = ii[ii.true_hit] if len(ii) else ii
            rows.append(dict(scenario=label, method=meth, ld_group=grp, n_causal=len(ci),
                             n_captured=int(ci.captured.sum()),
                             captured=round(float(ci.captured.mean()), 4) if len(ci) else None,
                             n_intervals=len(ii), n_true_hits=int(ii.true_hit.sum()) if len(ii) else 0,
                             true_hits=round(float(ii.true_hit.mean()), 4) if len(ii) else None,
                             median_width_kb=round(float(ii.width_kb.median()), 1) if len(ii) else None,
                             median_genes=float(ii.n_genes.median()) if len(ii) else None,
                             median_genes_true_hits=float(tt.n_genes.median()) if len(tt) else None))
    return rows


# ----------------------------------------------------------------------------------------------
# commands

def cmd_prepare(a):
    p2.cmd_vcf(a)
    p2.cmd_plink(a)


_C = {}   # one process's clump inputs, set once by _init


def _init(sim, out, bonf, span, group_of, plink, vcf, work):
    p2._init_worker(sim, out, bonf, span, group_of)
    snp_map = pd.read_csv(Path(sim) / "snp_map.csv")
    _C.update(plink=plink, inputs=["--vcf", vcf] + VCF_INPUT, work=Path(work), base=dict(p2._W["plink"]),
              chr_pos={s: (str(c), int(p)) for s, c, p in zip(snp_map["SNP_ID"], snp_map["Chr"], snp_map["Pos"])})


def one_replicate(task):
    """A replicate's clumps in both settings, then ld_blocks_vs_plink.one_replicate with them
    beside PLINK's blocks."""
    scen, rep, res_path, truth = task
    W = p2._W
    d = _C["work"] / scen / rep
    d.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(res_path).dropna(subset=["PValue"])
    write_assoc(d / "assoc.txt", res["SNP"], res["PValue"])
    pval = dict(zip(res["SNP"].astype(str), res["PValue"].astype(float)))
    sig_ids = list(res.loc[res["PValue"] < W["bonf"], "SNP"].astype(str))
    clumps, facts, digest = {}, [], []
    for setting in CLUMP_SETTINGS:
        _, cl, fact = clump_run(_C["plink"], _C["inputs"], d / "assoc.txt", W["bonf"], d / setting, setting,
                                sig_ids, pval, _C["chr_pos"])
        clumps[setting] = cl
        facts.append(dict(scenario=scen, rep=rep, **fact))
        sig_set = set(sig_ids)
        digest += [dict(scenario=scen, rep=rep, setting=setting, index_snp=c["index"], chr=c["chr"],
                        start=c["start"], end=c["end"], n_snps=len(c["members"]),
                        n_sig=sum(s in sig_set for s in c["members"])) for c in cl]
    W["plink"] = {**_C["base"], **clumps}
    try:
        result = p2.one_replicate(task)
    finally:
        W["plink"] = _C["base"]
    return result, facts, digest


def cmd_sim(a):
    logging.getLogger("gwas").setLevel(logging.WARNING)
    sim, out = Path(a.sim_data), Path(a.out)
    geno, ids, snp_map = p2.load_template(sim)
    bonf = 0.05 / len(snp_map)
    reps_all = list(p2.replicates(sim))
    truths = {(s, r): json.loads(t.read_text(encoding="utf-8")) for s, r, _, t in reps_all}
    causal_all = sorted({c for t in truths.values() for c in t["qtn_snp_ids"]})
    span, check = p2.ld_context(geno, snp_map, causal_all)
    five = [span[c] for (s, _), t in truths.items() if s.endswith("_q005") for c in t["qtn_snp_ids"]]
    split = float(np.median(five))
    group_of = {c: ("short" if v <= split else "long") for c, v in span.items()}
    reps = [x for x in reps_all if not a.only or f"{x[0]}/{x[1]}" in a.only]
    tasks = [(s, r, str(p), truths[(s, r)]) for s, r, p, _ in reps]
    init = (str(sim), str(out), bonf, span, group_of, a.plink, str(out / "template_raw.vcf"), str(out / "clump"))
    if a.jobs > 1:
        import multiprocessing
        with multiprocessing.get_context("spawn").Pool(a.jobs, initializer=_init, initargs=init) as pool:
            results = []
            for k, r in enumerate(pool.imap(one_replicate, tasks, chunksize=1)):
                results.append(r)
                if (k + 1) % 100 == 0:
                    print(f"{k + 1}/{len(tasks)} replicates", flush=True)
    else:
        _init(*init)
        results = []
        for k, t in enumerate(tasks):
            results.append(one_replicate(t))
            if (k + 1) % 100 == 0:
                print(f"{k + 1}/{len(tasks)} replicates", flush=True)
    rep_results = [r[0] for r in results]
    iv_rows = [row for r in rep_results for row in r[0]]
    causal_rows = [row for r in rep_results for row in r[1]]
    rep_rows = [r[2] for r in rep_results]
    runs = pd.DataFrame([f for r in results for f in r[1]])
    digest = pd.DataFrame([d for r in results for d in r[2]])
    ivd, cd = pd.DataFrame(iv_rows), pd.DataFrame(causal_rows)
    ivd.to_csv(out / "intervals.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(out / "replicates.csv", index=False)
    cd.to_csv(out / "causal_by_method.csv", index=False)
    runs.to_csv(out / "clump_runs.csv", index=False)
    digest.to_csv(out / "clumps_digest.csv", index=False)
    per = []
    for scen in sorted(cd.scenario.unique()):
        per += summarise(ivd[ivd.scenario == scen] if len(ivd) else ivd, cd[cd.scenario == scen], scen)
    pooled = summarise(ivd[ivd.scenario.str.endswith("_q005")] if len(ivd) else ivd,
                       cd[cd.scenario.str.endswith("_q005")], "pooled_5qtn")
    pd.DataFrame(per + pooled).to_csv(out / "ld_blocks_vs_plink_clump.csv", index=False)
    pd.DataFrame(pooled).to_csv(out / "ld_blocks_vs_plink_clump_main.csv", index=False)
    widest = {}
    for s in CLUMP_SETTINGS:
        ds = digest[digest.setting == s] if len(digest) else digest
        widest[s] = float(((ds.end - ds.start) / 1000.0).max()) if len(ds) else None
    facts = dict(template=f"{geno.shape[0]} x {geno.shape[1]}", bonferroni=bonf, bonferroni_repr=repr(bonf),
                 replicates=len(reps), causal_unique=len(causal_all), ld_split_kb=split, ld_split_from=len(five),
                 ld_check=check, clump_runs=len(runs),
                 clump_runs_without_clumped_file=int((runs.n_clumps == 0).sum()) if len(runs) else 0,
                 clumps={s: int((digest.setting == s).sum()) if len(digest) else 0 for s in CLUMP_SETTINGS},
                 widest_clump_kb=widest,
                 sig_in_no_clump=int(runs.sig_in_no_clump.sum()) if len(runs) else 0,
                 sig_in_two_or_more=int(runs.sig_in_two_or_more.sum()) if len(runs) else 0,
                 index_not_significant=int(runs.index_not_significant.sum()) if len(runs) else 0,
                 runs_with_options_as_designed=int(runs.options_as_designed.sum()) if len(runs) else 0,
                 runs_with_warnings=int((runs.warnings.fillna("") != "").sum()) if len(runs) else 0,
                 warnings_seen=sorted(set(w for x in runs.warnings.fillna("") for w in x.split(" | ") if w)),
                 trace_widest_block_kb=max((r[4] for r in rep_results), key=lambda w: w[0])[0],
                 sig_snps_in_two_trace_blocks=sum(r[3] for r in rep_results),
                 plink_blocks={k: len(p2.read_plink_blocks(out / f"{k}.blocks.det")) for k in p2.PLINK_SETTINGS},
                 gwas_file=p2.gwas.__file__, script=str(Path(__file__).resolve()))
    (out / "clump_facts.json").write_text(json.dumps(facts, indent=1, default=str), encoding="utf-8")
    print(json.dumps(facts, indent=1, default=str))
    print(pd.DataFrame(pooled).to_string(index=False))


def _locus_clumps(a, loc, inputs, assoc, thr, sig_ids, pval, chr_pos, out):
    clumps, runs, cmds = {}, [], {}
    for setting in CLUMP_SETTINGS:
        cmd, cl, fact = clump_run(a.plink, inputs, assoc, thr, out / f"{loc}_{setting}", setting,
                                  sig_ids, pval, chr_pos)
        clumps[setting] = cl
        runs.append(dict(locus=loc, **fact))
        cmds[f"{loc}_{setting}"] = cmd
    return clumps, runs, cmds


def cmd_loci(a):
    logging.getLogger("gwas").setLevel(logging.WARNING)
    p2.cmd_loci(a)          # ld_blocks_vs_plink.py's loci step: its rows, its PLINK files under DIR/loci
    out = Path(a.out) / "loci"
    rows, facts, runs, cmds = [], {}, [], {}
    # tomato: chromosome 2 of the locule number run, PLINK on tomato_chr2.vcf as written by the loci step
    meta, g, sig, tblocks, bframe, samples = p2.zip_run(a.tomato_zip)
    thr = 0.05 / len(g)
    g2 = g[g["Chr"] == "2"].sort_values("Pos")
    write_assoc(out / "tomato_assoc.txt", g2["SNP"], g2["PValue"])
    pval = dict(zip(g2["SNP"].astype(str), g2["PValue"].astype(float)))
    sig_ids = list(g2.loc[g2["PValue"] < thr, "SNP"].astype(str))
    chr_pos = {str(s): ("2", int(p)) for s, p in zip(g2["SNP"], g2["Pos"])}
    clumps, r, c = _locus_clumps(a, "tomato", ["--vcf", out / "tomato_chr2.vcf"] + VCF_INPUT,
                                 out / "tomato_assoc.txt", thr, sig_ids, pval, chr_pos, out)
    runs += r
    cmds.update(c)
    sig2 = sig[sig["Chr"] == "2"][["SNP", "Chr", "Pos", "PValue"]]
    b2 = bframe[bframe["Chr"] == "2"]
    lead_row = b2.loc[b2["lead_snp_pvalue"].astype(float).idxmin()] if len(b2) else None
    lead = None if lead_row is None else str(lead_row["lead_snp"])
    tomato_genes = p2.gene_counter(p2.annotation.load_gene_model(p2.REPO / "data" / "Sol_genes_SL3.csv"))
    rows += p2.locus_rows("tomato_locule_chr2", sig2, clumps, lead, tomato_genes)
    facts["tomato"] = dict(snps_tested=len(g), bonferroni=thr, bonferroni_repr=repr(thr), chr2_snps=len(g2),
                           chr2_significant_by_p=len(sig_ids), chr2_significant_table=len(sig2),
                           significant_sets_equal=set(sig_ids) == set(sig2["SNP"].astype(str)), lead_snp=lead,
                           clumps={k: len(v) for k, v in clumps.items()})
    # rice: chromosomes 3 and 7 of the grain length run, PLINK on the PLINK files with the loci step's lists
    meta, g, sig, tblocks, bframe, samples = p2.zip_run(a.rice_zip)
    thr = 0.05 / len(g)
    g37 = g[g["Chr"].isin(["3", "7"])]
    write_assoc(out / "rice_assoc.txt", g37["SNP"], g37["PValue"])
    pval = dict(zip(g37["SNP"].astype(str), g37["PValue"].astype(float)))
    sig_ids = list(g37.loc[g37["PValue"] < thr, "SNP"].astype(str))
    bim = pd.read_csv(f"{a.rice_bfile}.bim", sep=r"\s+", header=None, dtype={0: str, 1: str})
    want = set(g37["SNP"].astype(str))
    bim = bim[bim[1].isin(want)]
    chr_pos = {s: (c, int(p)) for c, s, p in zip(bim[0], bim[1], bim[3])}
    pos_agree = all(chr_pos.get(str(s)) == (str(ch), int(p)) for s, ch, p in zip(g37["SNP"], g37["Chr"], g37["Pos"]))
    clumps, r, c = _locus_clumps(a, "rice", ["--bfile", a.rice_bfile, "--keep", out / "rice_keep.txt",
                                             "--extract", out / "rice_extract.txt", "--chr", "3,7"],
                                 out / "rice_assoc.txt", thr, sig_ids, pval, chr_pos, out)
    runs += r
    cmds.update(c)
    rice_genes = p2.gene_counter(p2.annotation.load_gene_model(a.rice_gene_model))
    facts["rice"] = dict(snps_tested=len(g), bonferroni=thr, bonferroni_repr=repr(thr), snps_chr3_chr7=len(g37),
                         significant_chr3_chr7=len(sig_ids), bim_positions_equal_run_positions=pos_agree,
                         clumps={k: len(v) for k, v in clumps.items()})
    for name, (gid, ch, s, e) in p2.RICE_GENES.items():
        w = g[(g["Chr"] == ch) & (g["Pos"] >= s - p2.RICE_WINDOW) & (g["Pos"] <= e + p2.RICE_WINDOW)]
        sw = sig[(sig["Chr"] == ch) & (sig["Pos"] >= s - p2.RICE_WINDOW) & (sig["Pos"] <= e + p2.RICE_WINDOW)]
        lead = w.sort_values(["PValue", "Pos"]).iloc[0]["SNP"] if len(w) else None
        rows += p2.locus_rows(f"rice_{name}", sw[["SNP", "Chr", "Pos", "PValue"]],
                              {k: [b for b in v if b["chr"] == ch] for k, v in clumps.items()},
                              lead, rice_genes, gene=(ch, s, e), causal=p2.GS3_CAUSAL if name == "GS3" else None)
        facts["rice"][name] = dict(window_snps=len(w), window_significant=len(sw), lead_snp=lead)
    for loc in ("tomato", "rice"):   # PLINK's logs, without the computer's name and folder
        for setting in CLUMP_SETTINGS:
            lines = (out / f"{loc}_{setting}.log").read_text(encoding="utf-8", errors="replace").splitlines()
            facts[f"plink_log_{loc}_{setting}"] = [x for x in lines
                                                   if not x.startswith(("Hostname:", "Working directory:"))]
    p2_csv = (Path(a.out) / "ld_blocks_vs_plink_loci.csv").read_bytes()   # the loci step's rows, as written
    buf = io.StringIO()
    pd.DataFrame(rows, columns=pd.read_csv(io.BytesIO(p2_csv), nrows=0).columns).to_csv(
        buf, index=False, header=False, lineterminator="\r\n" if b"\r\n" in p2_csv else "\n")
    (Path(a.out) / "ld_blocks_vs_plink_clump_loci.csv").write_bytes(p2_csv + buf.getvalue().encode("utf-8"))
    pd.DataFrame(runs).to_csv(Path(a.out) / "clump_runs_loci.csv", index=False)
    (Path(a.out) / "clump_commands_loci.json").write_text(json.dumps({k: [str(x) for x in v] for k, v in cmds.items()},
                                                                     indent=1), encoding="utf-8")
    facts["gwas_file"] = p2.gwas.__file__
    (out / "clump_loci_facts.json").write_text(json.dumps(facts, indent=1, default=str), encoding="utf-8")
    print(pd.DataFrame(rows).drop(columns=["sig_snps_in_two_blocks"]).to_string(index=False))
    print(pd.DataFrame(runs).to_string(index=False))


def _rows_of(path, methods):
    """The header and the lines of a CSV whose method is one of ``methods``, as bytes."""
    lines = Path(path).read_bytes().split(b"\n")
    header = lines[0]
    col = next(csv.reader([header.decode().strip()])).index("method")
    keep = [x for x in lines[1:] if x.strip() and next(csv.reader([x.decode().strip()]))[col] in methods]
    return header, keep


def cmd_accept(a):
    out, ref = Path(a.out), Path(a.ref)
    result = {}
    for new, old in [("ld_blocks_vs_plink_clump_main.csv", "ld_blocks_vs_plink_main.csv"),
                     ("ld_blocks_vs_plink_clump.csv", "ld_blocks_vs_plink.csv"),
                     ("ld_blocks_vs_plink_clump_loci.csv", "ld_blocks_vs_plink_loci.csv")]:
        if not (out / new).is_file():
            result[new] = "missing"
            continue
        hn, rn = _rows_of(out / new, p2.METHODS)
        ho, ro = _rows_of(ref / old, p2.METHODS)
        rec = dict(reference=str(ref / old), rows_new=len(rn), rows_reference=len(ro),
                   header_equal=hn == ho, rows_bytes_equal=(hn == ho and rn == ro))
        if not rec["rows_bytes_equal"]:
            dn = pd.read_csv(io.BytesIO(b"\n".join([hn] + rn)))
            do = pd.read_csv(io.BytesIO(b"\n".join([ho] + ro)))
            rec["same_shape"] = dn.shape == do.shape
            if dn.shape == do.shape:
                rec["columns_equal_values"] = {c: bool(dn[c].equals(do[c])) for c in do.columns}
        result[new] = rec
    (out / "acceptance_p2_rows.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description="TRACE's LD blocks and PLINK's blocks against PLINK clumping")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "sim", "loci", "accept"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, help="the work folder (VCF, PLINK files, results)")
        p.add_argument("--sim-data", default=str(p2.SIM), help="the published simulation folder")
        if name in ("prepare", "loci"):
            p.add_argument("--vcf", required=True, help="the Varitome VCF (raw calls)")
        if name in ("prepare", "sim", "loci"):
            p.add_argument("--plink", required=True, help="the PLINK 1.9 executable")
        if name == "sim":
            p.add_argument("--jobs", type=int, default=1,
                           help="replicates run side by side in this many processes (the result is the same)")
            p.add_argument("--only", nargs="*", default=None, help="only these SCENARIO/REP (a check)")
        if name == "loci":
            p.add_argument("--tomato-zip", required=True, help="the CLI results ZIP of tomato locule number")
            p.add_argument("--rice-zip", required=True, help="the CLI results ZIP of rice grain length")
            p.add_argument("--rice-bfile", required=True, help="PLINK files (.bed/.bim/.fam) of the rice VCF")
            p.add_argument("--rice-gene-model", required=True, help="the rice gene model of the run (locus.gff)")
        if name == "accept":
            p.add_argument("--ref", default=str(HERE / "sim_summary"),
                           help="the folder of ld_blocks_vs_plink.py's committed tables")
    a = ap.parse_args(argv)
    print(f"gwas: {p2.gwas.__file__}", flush=True)
    {"prepare": cmd_prepare, "sim": cmd_sim, "loci": cmd_loci, "accept": cmd_accept}[a.cmd](a)


if __name__ == "__main__":
    main()
