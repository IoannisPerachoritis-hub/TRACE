"""Candidate regions around true signals: for each signal, the whole region that a method's intervals
highlight, scored for TRACE's 1.2.0 default candidate intervals that hold a Bonferroni SNP, PLINK's
haplotype blocks (200 and 600 kb) and PLINK's clumps (C1 and C2). There is no block detection and no
PLINK run: every interval comes from the saved outputs of the committed scripts, used unchanged.

Definitions:
- Significant: p below 0.05 / 43,974 (1.1370355209896758e-06), as ld_blocks_vs_plink.py.
- True positive (TP) SNP: a significant SNP on the chromosome of a causal SNP of its replicate, at
  most 500 kb from it (evaluate_simulation.py's rule). A SNP within 500 kb of two causal SNPs belongs
  to the nearer one (tie: the lower causal position).
- Signal: a causal SNP of a replicate with at least one TP SNP.
- Holding. TRACE: the counted intervals of ld_blocks_vs_plink_trace_default_recount.sim_recount
  (P-2c's saved intervals) hold a SNP at start < pos < end of their saved bounds (for an interval of
  unblocked SNPs, its SNP list is the template markers strictly inside its bounds, checked below).
  PLINK blocks and clumps: the interval of ld_blocks_vs_plink.intervals whose significant SNPs hold
  the SNP (a SNP in no block or clump is its own interval, width 0), with ld_blocks_vs_plink_clump's
  read_clumps for the clumps.
- Candidate region: the union of the method's intervals that hold at least one of the signal's TP
  SNPs, overlapping or touching intervals merged (closed intervals sorted by start, the next joins
  when its start is at most the current end). Width: the summed length of the merged segments (kb).
  Genes: the distinct gene model rows overlapping any of its intervals (gene start <= end and gene
  end >= start). Causal SNP inside: start <= causal position <= end for some of its intervals.

Checks, each of which stops the script when it fails (the facts file says which):
(a) the real runs' intervals of unblocked SNPs (the runs' tables and P-2c's driver tables): the
    markers strictly inside the bounds equal the SNP list;
(b) every simulated interval of unblocked SNPs: the template markers strictly inside its bounds
    number n_snps and are all significant;
the PLINK block and clump intervals recomputed here equal the saved rows of ld_blocks_vs_plink.py
and ld_blocks_vs_plink_clump.py (intervals.csv, every column); every TP SNP is held by an interval of
each method; every TRACE interval holding a TP SNP is a counted one; a significant causal SNP is
inside every method's region; a region holding the causal SNP has the saved capture flag True (P-2c,
P-2 and P-2b's causal_by_method.csv); a one interval region has the saved width and gene count of its
interval; the signals and their TP SNPs are the same for every method.

    python benchmarks/simulation/ld_blocks_vs_plink_per_signal.py --out DIR --p2 P2 --p2b P2B --p2c P2C \\
        --sim-data SIM --tomato-zip TOMATO.zip --rice-zip RICE.zip --rice-gene-model locus.gff \\
        [--only SCENARIO/REP ...] [--code TREE]

P2, P2B and P2C are the work folders of ld_blocks_vs_plink.py, ld_blocks_vs_plink_clump.py and
ld_blocks_vs_plink_trace_default.py. A missing input stops the script. Writes, in DIR:
ld_blocks_vs_plink_per_signal_main.csv (all replicates pooled), ld_blocks_vs_plink_per_signal_5qtn.csv
(the 5-QTN scenarios pooled), ld_blocks_vs_plink_per_signal.csv (per scenario),
ld_blocks_vs_plink_per_signal_loci.csv, regions.csv (one row per signal and method),
loci_regions.csv and per_signal_facts.json.
"""
import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import ld_blocks_vs_plink_trace_default_recount as rc   # first: P-2c's module puts --code's tree on sys.path

td = rc.td
p2 = td.p2
annotation = td.annotation
import ld_blocks_vs_plink_clump as pb  # noqa: E402  P-2b's clump rule (read_clumps), unchanged

TRACE = "trace_default_bonferroni"
BLOCKS = ["plink_200kb", "plink_600kb"]
CLUMPS = ["plink_clump_c1", "plink_clump_c2"]
PLINK = BLOCKS + CLUMPS
METHODS = [TRACE] + PLINK
FLAG_METHOD = {TRACE: "trace_default", **{m: m for m in PLINK}}   # the method of the saved capture flags
TP_BP = 500_000
IV_COLS = ["scenario", "rep", "method", "chr", "start", "end", "width_kb", "n_snps", "n_sig", "n_blocks",
           "n_genes", "true_hit", "group"]
LOCI = [("tomato_locule_chr2", "tomato", "SL25ch02p47301921", "lead"),
        ("rice_GS3", "rice", "3:16733441", "causal"),
        ("rice_GL7", "rice", "7:24558720", "lead")]
DET_SHA256 = {"plink_200kb": "df7586ed079cc076f585c45ed00b1585001fa890ea77f5a80003f308ba86151b",
              "plink_600kb": "ada20eb021af644aa46aba005806f8ac6e4a6493e694926766973c41ae99b57c"}


class Stop(Exception):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_lines(rows, cols):
    """Rows as pandas writes them in a CSV (ld_blocks_vs_plink.py's intervals.csv), as bytes."""
    buf = io.StringIO()
    pd.DataFrame(rows, columns=cols).to_csv(buf, index=False, header=False, lineterminator="\n")
    return [x.encode() for x in buf.getvalue().split("\n") if x]


def saved_lines(path, methods):
    """{(scenario, rep, method): [line, ...]} of a saved intervals.csv, for the given methods."""
    raw = [x.rstrip(b"\r") for x in Path(path).read_bytes().split(b"\n")]
    if raw[0].decode() != ",".join(IV_COLS):
        raise Stop(f"{path}: header {raw[0][:200]!r}")
    out = {}
    for x in raw[1:]:
        if not x:
            continue
        s, r, m = x.split(b",", 3)[:3]
        m = m.decode()
        if m in methods:
            out.setdefault((s.decode(), r.decode(), m), []).append(x)
    return out


def capture_flags(path, methods):
    d = pd.read_csv(path, dtype={"chr": str, "snp": str})
    d = d[d["method"].isin(methods)]
    return {(s, r, m, c): bool(f) for s, r, m, c, f in zip(d["scenario"], d["rep"], d["method"], d["snp"], d["captured"])}


class Genes:
    """A gene model's rows by chromosome, with ld_blocks_vs_plink.gene_counter's overlap rule."""

    def __init__(self, gene_model):
        g = gene_model[["Chr", "Start", "End", "Gene_ID"]].copy()
        g["Chr"] = g["Chr"].astype(str)
        self.by = {c: (x["Start"].to_numpy(), x["End"].to_numpy(), x["Gene_ID"].astype(str).to_numpy())
                   for c, x in g.groupby("Chr")}

    def count(self, ch, spans):
        """The distinct rows, and the distinct Gene_IDs, overlapping any of the spans."""
        if ch not in self.by or not spans:
            return 0, 0
        s, e, ids = self.by[ch]
        hit = np.zeros(len(s), dtype=bool)
        for a, b in spans:
            hit |= (s <= b) & (e >= a)
        return int(hit.sum()), int(len(set(ids[hit])))


def merge(spans):
    segs = []
    for a, b in sorted(spans):
        if segs and a <= segs[-1][1]:
            segs[-1][1] = max(segs[-1][1], b)
        else:
            segs.append([a, b])
    return segs


def tp_snps(sig, causal):
    """{causal SNP: [(snp, chr, pos), ...]}, with the TP SNPs within 500 kb of two or more causal
    SNPs and the ties among them."""
    by_chr = {}
    for c, ch, p in causal:
        by_chr.setdefault(ch, []).append((int(p), c))
    tp, near_two, ties = {}, 0, 0
    for snp, ch, pos in zip(sig["SNP"].astype(str).tolist(), sig["Chr"].astype(str).tolist(),
                            sig["Pos"].astype(int).tolist()):
        near = sorted((abs(pos - p), p, c) for p, c in by_chr.get(ch, []) if abs(pos - p) <= TP_BP)
        if not near:
            continue
        if len(near) > 1:
            near_two += 1
            ties += near[0][0] == near[1][0]
        tp.setdefault(near[0][2], []).append((snp, ch, pos))
    return tp, near_two, ties


def p2_rows(scen, rep, meth, ivs, causal, group_of, count_genes):
    """ld_blocks_vs_plink.one_replicate's interval rows."""
    rows = []
    for iv in ivs:
        held = [c for c, ch, p in causal if ch == iv["chr"] and iv["start"] <= p <= iv["end"]]
        if held:
            grp = "short" if any(group_of[c] == "short" for c in held) else "long"
        else:
            same = [(min(abs(p - iv["start"]), abs(p - iv["end"])), p, c) for c, ch, p in causal if ch == iv["chr"]]
            grp = group_of[sorted(same)[0][2]] if same else "none"
        rows.append((scen, rep, meth, iv["chr"], iv["start"], iv["end"], (iv["end"] - iv["start"]) / 1000.0,
                     iv["n_snps"], iv["n_sig"], iv["n_blocks"], count_genes(iv["chr"], iv["start"], iv["end"]),
                     bool(held), grp))
    return rows


def region(spans, genes, ch):
    segs = merge(spans)
    n_rows, n_ids = genes.count(ch, spans)
    return dict(segments=";".join(f"{a}-{b}" for a, b in segs), width_kb=sum(b - a for a, b in segs) / 1000.0,
                genes=n_rows, gene_ids=n_ids)


# ----------------------------------------------------------------------------------------------
# check (a): the real runs' intervals of unblocked SNPs

def check_a(zips, p2c):
    out = {}
    for name, zpath in zips.items():
        meta, g, sig, tblocks, bframe, samples = p2.zip_run(zpath)
        markers = {c: x.sort_values("Pos", kind="stable") for c, x in g.groupby("Chr")}
        tables = {"run": p2.zip_table(zipfile.ZipFile(zpath), "tables/Isolated_SNP_intervals_MLM.csv"),
                  "driver": pd.read_csv(Path(p2c) / f"loci_{name}_driver_unblocked.csv")}
        for tname, t in tables.items():
            bad = []
            for r in t.to_dict("records"):
                m = markers.get(annotation.canon_chr(r["Chr"]))
                inside = [] if m is None else m[(m["Pos"] > int(r["Start (bp)"])) & (m["Pos"] < int(r["End (bp)"]))]["SNP"]
                if sorted(map(str, inside)) != sorted(td.snp_list(r["SNP_IDs"])):
                    bad.append(f"chr{r['Chr']}:{r['Start (bp)']}-{r['End (bp)']}")
            out[f"{name}_{tname}"] = dict(intervals=int(len(t)), equal=int(len(t)) - len(bad), differ=bad)
    return out


# ----------------------------------------------------------------------------------------------
# the simulation

def run_sim(a, facts):
    sim, p2d, p2bd, p2cd = Path(a.sim_data), Path(a.p2), Path(a.p2b), Path(a.p2c)
    t0 = time.time()
    sim_facts = json.loads((p2cd / "sim_facts.json").read_text(encoding="utf-8"))
    thr = float(sim_facts["rule"]["p_threshold"])
    geno, ids, snp_map = p2.load_template(sim)
    facts["threshold"] = repr(thr)
    if thr != 0.05 / len(snp_map):
        raise Stop(f"the threshold {thr!r} is not 0.05 / {len(snp_map)}")
    # every replicate's inputs
    missing = [str(d / f) for sd in sorted(sim.glob("h2_*")) if sd.is_dir() for d in sorted(sd.glob("rep_*"))
               if d.is_dir() for f in ("platform_loco/results.csv", "truth.json") if not (d / f).is_file()]
    reps = list(p2.replicates(sim))
    facts["replicates_found"] = len(reps)
    if missing or len(reps) != sim_facts["replicates"]:
        raise Stop(f"missing simulation inputs: {missing[:10]}, {len(reps)} replicates found, "
                   f"{sim_facts['replicates']} in P-2c's run")
    selected = [x for x in reps if not a.only or f"{x[0]}/{x[1]}" in a.only]
    if a.only and len(selected) != len(set(a.only)):
        raise Stop(f"--only names replicates that are not there: {sorted(set(a.only) - {f'{x[0]}/{x[1]}' for x in selected})}")
    keys = {(s, r) for s, r, _, _ in selected}
    facts["scenarios"] = sorted({s for s, _ in keys})
    truths = {(s, r): json.loads(t.read_text(encoding="utf-8")) for s, r, _, t in reps}
    causal_all = sorted({c for t in truths.values() for c in t["qtn_snp_ids"]})
    span, _ = p2.ld_context(geno, snp_map, causal_all)          # ld_blocks_vs_plink.cmd_compare's LD groups
    five = [span[c] for (s, _), t in truths.items() if s.endswith("_q005") for c in t["qtn_snp_ids"]]
    split = float(np.median(five))
    group_of = {c: ("short" if v <= split else "long") for c, v in span.items()}
    facts["ld_split_kb"] = split
    del geno
    # TRACE: the recount's counted intervals
    ivd, _, _, _ = rc.sim_recount(p2cd, sim, thr)
    ivd = ivd[[(s, r) in keys for s, r in zip(ivd["scenario"], ivd["rep"])]]
    facts["trace_intervals"] = dict(all=int(len(ivd)), counted=int(ivd["holds"].sum()),
                                    counted_blocks=int(((ivd["n_blocks"] == 1) & ivd["holds"]).sum()),
                                    unblocked=int((ivd["n_blocks"] == 0).sum()))
    trace_by = {}
    for (s, r, ch), g in ivd.groupby(["scenario", "rep", "chr"], sort=False):
        trace_by.setdefault((s, r), {})[ch] = dict(
            start=g["start"].to_numpy(np.int64), end=g["end"].to_numpy(np.int64), holds=g["holds"].to_numpy(bool),
            width_kb=g["width_kb"].to_numpy(float), n_genes=g["n_genes"].to_numpy(np.int64))
    # the saved PLINK intervals and capture flags, the PLINK blocks, the genes
    saved = {**saved_lines(p2d / "intervals.csv", BLOCKS), **saved_lines(p2bd / "intervals.csv", CLUMPS)}
    flags = {**capture_flags(p2cd / "causal_by_method.csv", ["trace_default"]),
             **capture_flags(p2d / "causal_by_method.csv", BLOCKS),
             **capture_flags(p2bd / "causal_by_method.csv", CLUMPS)}
    plink_blocks = {m: p2.read_plink_blocks(p2d / f"{m}.blocks.det") for m in BLOCKS}
    gene_model = annotation.load_gene_model(p2.REPO / "data" / "Sol_genes_SL3.csv")
    genes, count_genes = Genes(gene_model), p2.gene_counter(gene_model)
    chr_pos = {s: (str(c), int(p)) for s, c, p in zip(snp_map["SNP_ID"], snp_map["Chr"], snp_map["Pos"])}
    facts["seconds_setup"] = round(time.time() - t0, 1)
    # each replicate's significant SNPs (one read of its results)
    rep_sig = {}
    for s, r, res_path, _ in selected:
        res = pd.read_csv(res_path).dropna(subset=["PValue"])
        res["Chr"] = res["Chr"].astype(str)
        sig = res[res["PValue"] < thr][["SNP", "Chr", "Pos", "PValue"]]
        pv = dict(zip(res["SNP"].astype(str), res["PValue"].astype(float)))
        rep_sig[(s, r)] = (sig, {str(c): pv.get(str(c)) for c in truths[(s, r)]["qtn_snp_ids"]},
                           set(sig["SNP"].astype(str)))
    # check (b): every interval of unblocked SNPs, against the template's markers
    markers = {}
    for ch, g in snp_map.groupby("Chr"):
        g = g.sort_values("Pos", kind="stable")
        markers[ch] = (g["Pos"].to_numpy(np.int64), g["SNP_ID"].astype(str).to_numpy())
    n_b, bad_b = 0, []
    for x in ivd[ivd["n_blocks"] == 0].itertuples(index=False):
        n_b += 1
        pos, sid = markers[x.chr]
        lo, hi = int(np.searchsorted(pos, x.start, side="right")), int(np.searchsorted(pos, x.end, side="left"))
        if hi - lo != x.n_snps or not all(v in rep_sig[(x.scenario, x.rep)][2] for v in sid[lo:hi]):
            bad_b.append(f"{x.scenario}/{x.rep} chr{x.chr}:{x.start}-{x.end}")
    facts["check_b"] = dict(unblocked_intervals=n_b, equal=n_b - len(bad_b), differ=bad_b[:20], n_differ=len(bad_b))
    if bad_b:
        raise Stop("check (b): an interval of unblocked SNPs does not hold n_snps significant template markers")
    # the signals
    c = dict(replicates=len(selected), significant_snps=0, signals=0, tp_snps=0, tp_near_two_causal=0,
             tp_near_two_causal_ties=0, tp_unheld={m: 0 for m in METHODS}, tp_held_twice={m: 0 for m in METHODS},
             trace_holders_not_counted=0, self_check_miss={m: 0 for m in METHODS},
             cross_check_fail={m: 0 for m in METHODS}, one_interval_regions={m: 0 for m in METHODS},
             width_check_fail={m: 0 for m in METHODS}, genes_check_fail={m: 0 for m in METHODS},
             genes_differ_by_gene_id={m: 0 for m in METHODS})
    eq = dict(compared=0, equal=0, differ=[], saved_keys_not_compared=0)
    compared, rows = set(), []
    for s, r, res_path, _ in selected:
        sig, causal_p, _ = rep_sig[(s, r)]
        truth = truths[(s, r)]
        causal = list(zip(truth["qtn_snp_ids"], [str(x) for x in truth["qtn_chrs"]], truth["qtn_positions"]))
        tp, near_two, ties = tp_snps(sig, causal)
        c["significant_snps"] += len(sig)
        c["tp_near_two_causal"] += near_two
        c["tp_near_two_causal_ties"] += ties
        ivs = {}
        for m in BLOCKS:
            ivs[m] = p2.intervals(sig, plink_blocks[m])[0] if len(sig) else []
        for m in CLUMPS:
            path = p2bd / "clump" / s / r / f"{m}.clumped"
            if len(sig) and not path.is_file():
                raise Stop(f"missing input: {path}")
            ivs[m] = p2.intervals(sig, pb.read_clumps(path, chr_pos))[0] if len(sig) else []
        ok = {}
        for m in PLINK:   # the recomputed intervals against the saved rows, every column
            key = (s, r, m)
            compared.add(key)
            ok[m] = csv_lines(p2_rows(s, r, m, ivs[m], causal, group_of, count_genes), IV_COLS) == saved.get(key, [])
            eq["compared"] += 1
            eq["equal"] += ok[m]
            if not ok[m]:
                eq["differ"].append(f"{s}/{r}/{m}")
        where = {m: {} for m in PLINK}
        for m in PLINK:
            for i, iv in enumerate(ivs[m]):
                for x in iv["sig_snps"]:
                    where[m].setdefault(str(x), []).append(i)
        tr = trace_by.get((s, r), {})
        for cid, ch, cpos in causal:
            if cid not in tp:
                continue
            lst = tp[cid]
            cpos = int(cpos)
            p_c = causal_p.get(str(cid))
            sig_causal = bool(p_c is not None and p_c < thr)
            tp_hash = hashlib.sha1(",".join(sorted(x[0] for x in lst)).encode()).hexdigest()[:16]
            c["signals"] += 1
            c["tp_snps"] += len(lst)
            for m in METHODS:
                held, twice, unheld = set(), 0, 0
                arr = tr.get(ch) if m == TRACE else None
                for snp, sch, pos in lst:
                    if m == TRACE:
                        if arr is None:
                            hs = []
                        else:
                            hit = (arr["start"] < pos) & (arr["end"] > pos)
                            c["trace_holders_not_counted"] += int((hit & ~arr["holds"]).sum())
                            hs = np.flatnonzero(hit & arr["holds"]).tolist()
                    else:
                        hs = where[m].get(snp, [])
                    twice += len(hs) > 1
                    unheld += not hs
                    held.update(hs)
                held = sorted(held)
                if m == TRACE:
                    spans = [(int(arr["start"][i]), int(arr["end"][i])) for i in held]
                else:
                    if any(ivs[m][i]["chr"] != ch for i in held):
                        raise Stop(f"{s}/{r} {m}: a region spans two chromosomes")
                    spans = [(int(ivs[m][i]["start"]), int(ivs[m][i]["end"])) for i in held]
                reg = region(spans, genes, ch)
                inside = any(x <= cpos <= y for x, y in spans)
                flag = flags.get((s, r, FLAG_METHOD[m], str(cid)))
                c["tp_unheld"][m] += unheld
                c["tp_held_twice"][m] += twice
                c["genes_differ_by_gene_id"][m] += reg["genes"] != reg["gene_ids"]
                c["self_check_miss"][m] += sig_causal and not inside
                c["cross_check_fail"][m] += inside and flag is not True
                if len(held) == 1:
                    c["one_interval_regions"][m] += 1
                    i = held[0]
                    if m == TRACE:
                        w0, g0 = float(arr["width_kb"][i]), int(arr["n_genes"][i])
                    elif ok[m]:
                        fields = saved[(s, r, m)][i].split(b",")
                        w0, g0 = float(fields[6]), int(fields[10])
                    else:
                        w0, g0 = None, None   # the rows differ: the script stops on that
                    c["width_check_fail"][m] += w0 is None or abs(reg["width_kb"] - w0) > 1e-9
                    c["genes_check_fail"][m] += g0 is None or reg["genes"] != g0
                rows.append(dict(scenario=s, rep=r, causal_snp=cid, chr=ch, pos=cpos, causal_significant=sig_causal,
                                 n_tp=len(lst), tp_hash=tp_hash, method=m, n_intervals=len(held), n_tp_held_twice=twice,
                                 n_tp_unheld=unheld, segments=reg["segments"], width_kb=reg["width_kb"],
                                 genes=reg["genes"], gene_ids=reg["gene_ids"], causal_inside=inside,
                                 saved_captured=flag))
    eq["saved_keys_not_compared"] = sum(1 for k in saved if k not in compared and (not a.only or (k[0], k[1]) in keys))
    facts["recomputed_plink_intervals"] = dict(compared=eq["compared"], equal=eq["equal"], n_differ=len(eq["differ"]),
                                               differ=eq["differ"][:20],
                                               saved_keys_not_compared=eq["saved_keys_not_compared"])
    facts["counts"] = c
    facts["seconds_sim"] = round(time.time() - t0, 1)
    return pd.DataFrame(rows)


def accept_sim(reg, facts):
    c, fails = facts["counts"], []
    for k in ("tp_unheld", "self_check_miss", "cross_check_fail", "width_check_fail", "genes_check_fail"):
        if any(c[k].values()):
            fails.append(f"{k} {c[k]}")
    if c["trace_holders_not_counted"]:
        fails.append(f"trace_holders_not_counted {c['trace_holders_not_counted']}")
    e = facts["recomputed_plink_intervals"]
    if e["n_differ"] or e["saved_keys_not_compared"]:
        fails.append(f"recomputed intervals: {e['n_differ']} differ, {e['saved_keys_not_compared']} saved not compared")
    sig_cols = ["scenario", "rep", "causal_snp", "n_tp", "tp_hash"]
    ref = reg[reg["method"] == TRACE][sig_cols].reset_index(drop=True)
    same = {m: bool(reg[reg["method"] == m][sig_cols].reset_index(drop=True).equals(ref)) for m in METHODS}
    facts["signals_identical_across_methods"] = dict(
        identical=all(same.values()), per_method=same,
        signals={m: int((reg["method"] == m).sum()) for m in METHODS},
        tp_snps={m: int(reg.loc[reg["method"] == m, "n_tp"].sum()) for m in METHODS})
    if not all(same.values()):
        fails.append("the signals or their TP SNPs differ between methods")
    if fails:
        raise Stop("; ".join(fails))


def summarise(reg, scope):
    def q(x, nd=None):
        if not len(x):
            return [None, None, None]
        v = [float(x.quantile(p)) for p in (0.25, 0.5, 0.75)]
        return [round(y, nd) for y in v] if nd is not None else v
    out = []
    for m in METHODS:
        d = reg[reg["method"] == m]
        n = len(d)
        dn, di = d[~d["causal_significant"]], d[d["causal_inside"]]
        nsig, nin, nin_ns = int(d["causal_significant"].sum()), int(d["causal_inside"].sum()), int(dn["causal_inside"].sum())
        w, g, gi = q(d["width_kb"], 1), q(d["genes"]), q(di["genes"])
        out.append(dict(scenario=scope, method=m, n_signals=n, n_causal_distinct=int(d["causal_snp"].nunique()),
                        n_causal_significant=nsig, causal_significant=round(nsig / n, 4) if n else None,
                        n_causal_inside=nin, causal_inside=round(nin / n, 4) if n else None,
                        n_not_significant=len(dn), n_causal_inside_not_significant=nin_ns,
                        causal_inside_not_significant=round(nin_ns / len(dn), 4) if len(dn) else None,
                        width_kb_q1=w[0], width_kb_median=w[1], width_kb_q3=w[2],
                        genes_q1=g[0], genes_median=g[1], genes_q3=g[2],
                        genes_inside_q1=gi[0], genes_inside_median=gi[1], genes_inside_q3=gi[2]))
    if len({(r["n_causal_significant"], r["causal_significant"]) for r in out}) != 1:
        raise Stop(f"{scope}: the share of significant causal SNPs differs between methods")
    return out


# ----------------------------------------------------------------------------------------------
# the loci

def run_loci(a, facts):
    p2d, p2bd, p2cd = Path(a.p2), Path(a.p2b), Path(a.p2c)
    lf = json.loads((p2bd / "loci" / "clump_loci_facts.json").read_text(encoding="utf-8"))
    if lf.get("rice", {}).get("bim_positions_equal_run_positions") is not True:
        raise Stop("P-2b's loci facts do not record the rice .bim positions equal to the run's")
    need = [p2d / "loci" / f"{run}_{m}.blocks.det" for run in ("tomato", "rice") for m in BLOCKS] + \
           [p2bd / "loci" / f"{run}_{m}.clumped" for run in ("tomato", "rice") for m in CLUMPS] + \
           [p2cd / f"loci_{run}_driver_{t}.csv" for run in ("tomato", "rice") for t in ("blocks", "unblocked")]
    missing = [str(x) for x in need if not x.is_file()]
    if missing:
        raise Stop(f"missing loci inputs: {missing}")
    models = {"tomato": Genes(annotation.load_gene_model(p2.REPO / "data" / "Sol_genes_SL3.csv")),
              "rice": Genes(annotation.load_gene_model(a.rice_gene_model))}
    runs = {"tomato": p2.zip_run(a.tomato_zip), "rice": p2.zip_run(a.rice_zip)}
    gs3_id, gs3_chr, gs3_start, gs3_end = p2.RICE_GENES["GS3"]
    rows, details = [], []
    for locus, run, anchor, atype in LOCI:
        meta, g, sig, tblocks, bframe, samples = runs[run]
        hit = g[g["SNP"].astype(str) == anchor]
        if len(hit) != 1:
            raise Stop(f"{locus}: the anchor {anchor} is in the run's GWAS table {len(hit)} times")
        ach, apos = str(hit.iloc[0]["Chr"]), int(hit.iloc[0]["Pos"])
        bonf = g[(g["Significant_Bonf"].astype(str) == "True") & (g["Chr"] == ach)][["SNP", "Chr", "Pos", "PValue"]]
        tpd = bonf[(bonf["Pos"] - apos).abs() <= TP_BP]
        tp = list(zip(tpd["SNP"].astype(str).tolist(), tpd["Pos"].astype(int).tolist()))
        is_gs3 = locus == "rice_GS3"
        anchor_sig = anchor in set(bonf["SNP"].astype(str))
        if not tp:
            for m in METHODS:
                rows.append(dict(locus=locus, anchor=anchor, anchor_type=atype, n_tp_snps=0, method=m, width_kb=None,
                                 genes=None, gene_inside=None, causal_inside=None, note="no signal"))
                details.append(dict(locus=locus, method=m, n_tp=0, anchor_significant=anchor_sig, n_intervals=0,
                                    tp_held_twice=0, tp_unheld=0, segments="", gene_ids=None))
            continue
        blocks = pd.read_csv(p2cd / f"loci_{run}_driver_blocks.csv")
        unb = pd.read_csv(p2cd / f"loci_{run}_driver_unblocked.csv")
        ivs = {TRACE: [dict(chr=annotation.canon_chr(b["Chr"]), start=int(b["Interval_start"]), end=int(b["Interval_end"]),
                            members=None) for b in blocks.to_dict("records")] +
                      [dict(chr=annotation.canon_chr(u["Chr"]), start=int(u["Start (bp)"]), end=int(u["End (bp)"]),
                            members=set(td.snp_list(u["SNP_IDs"]))) for u in unb.to_dict("records")]}
        for m in BLOCKS:
            ivs[m] = p2.intervals(bonf, [b for b in p2.read_plink_blocks(p2d / "loci" / f"{run}_{m}.blocks.det")
                                         if b["chr"] == ach])[0]
        chr_pos = {str(s): (str(ch), int(p)) for s, ch, p in zip(g["SNP"], g["Chr"], g["Pos"])}
        for m in CLUMPS:
            ivs[m] = p2.intervals(bonf, [x for x in pb.read_clumps(p2bd / "loci" / f"{run}_{m}.clumped", chr_pos)
                                         if x["chr"] == ach])[0]
        for m in METHODS:
            held, twice, unheld = set(), 0, 0
            for snp, pos in tp:
                if m == TRACE:
                    hs = [i for i, iv in enumerate(ivs[m]) if iv["chr"] == ach and (
                        iv["start"] < pos < iv["end"] if iv["members"] is None else snp in iv["members"])]
                else:
                    hs = [i for i, iv in enumerate(ivs[m]) if snp in iv["sig_snps"]]
                twice += len(hs) > 1
                unheld += not hs
                held.update(hs)
            spans = [(int(ivs[m][i]["start"]), int(ivs[m][i]["end"])) for i in sorted(held)]
            reg = region(spans, models[run], ach)
            gene_in = any(ach == gs3_chr and x <= gs3_end and y >= gs3_start for x, y in spans) if is_gs3 else None
            causal_in = any(ach == p2.GS3_CAUSAL[0] and x <= p2.GS3_CAUSAL[1] <= y for x, y in spans) if is_gs3 else None
            if unheld:
                raise Stop(f"{locus} {m}: {unheld} TP SNPs held by no interval")
            if is_gs3 and anchor_sig and not causal_in:
                raise Stop(f"{locus} {m}: the significant causal SNP is not inside the region")
            rows.append(dict(locus=locus, anchor=anchor, anchor_type=atype, n_tp_snps=len(tp), method=m,
                             width_kb=reg["width_kb"], genes=reg["genes"], gene_inside=gene_in, causal_inside=causal_in,
                             note=""))
            details.append(dict(locus=locus, method=m, n_tp=len(tp), anchor_significant=anchor_sig,
                                n_intervals=len(held), tp_held_twice=twice, tp_unheld=unheld, segments=reg["segments"],
                                gene_ids=reg["gene_ids"]))
    rows, details = pd.DataFrame(rows), pd.DataFrame(details)
    rows["genes"] = rows["genes"].astype("Int64")            # counts, blank where there is no signal
    details["gene_ids"] = details["gene_ids"].astype("Int64")
    return rows, details


def main(argv=None):
    ap = argparse.ArgumentParser(description="Candidate regions around true signals, per method")
    ap.add_argument("--out", required=True)
    ap.add_argument("--code", default=None, help="the TRACE tree whose gwas and annotation are used")
    ap.add_argument("--sim-data", default=str(p2.SIM))
    ap.add_argument("--p2", required=True, help="ld_blocks_vs_plink.py's work folder")
    ap.add_argument("--p2b", required=True, help="ld_blocks_vs_plink_clump.py's work folder")
    ap.add_argument("--p2c", required=True, help="ld_blocks_vs_plink_trace_default.py's work folder")
    ap.add_argument("--tomato-zip", required=True)
    ap.add_argument("--rice-zip", required=True)
    ap.add_argument("--rice-gene-model", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="only these SCENARIO/REP (a check)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    facts = dict(script=str(Path(__file__).resolve()), script_sha256=sha256(__file__), gwas_file=td.gwas.__file__,
                 modules={m.__name__: m.__file__ for m in (rc, td, p2, pb)}, only=a.only)
    try:
        p2d, p2bd, p2cd, sim = Path(a.p2), Path(a.p2b), Path(a.p2c), Path(a.sim_data)
        need = [p2d / f"{m}.blocks.det" for m in BLOCKS] + [
            p2d / "intervals.csv", p2d / "causal_by_method.csv", p2bd / "intervals.csv", p2bd / "causal_by_method.csv",
            p2bd / "loci" / "clump_loci_facts.json", p2cd / "intervals.csv", p2cd / "causal_by_method.csv",
            p2cd / "sim_facts.json", sim / "snp_map.csv", sim / "geno_matrix.npy", sim / "sample_ids.npy",
            Path(a.tomato_zip), Path(a.rice_zip), Path(a.rice_gene_model)]
        missing = [str(x) for x in need if not x.is_file()]
        if missing:
            raise Stop(f"missing inputs: {missing}")
        facts["blocks_det_sha256"] = {m: sha256(p2d / f"{m}.blocks.det") for m in BLOCKS}
        if facts["blocks_det_sha256"] != DET_SHA256:
            raise Stop("a .blocks.det differs from P-2's")
        facts["check_a"] = check_a({"tomato": a.tomato_zip, "rice": a.rice_zip}, p2cd)
        if any(v["differ"] for v in facts["check_a"].values()):
            raise Stop("check (a): the markers inside an interval of unblocked SNPs differ from its SNP list")
        reg = run_sim(a, facts)
        accept_sim(reg, facts)
        reg.to_csv(out / "regions.csv", index=False)
        per = []
        for scen in facts["scenarios"]:
            per += summarise(reg[reg["scenario"] == scen], scen)
        pd.DataFrame(per).to_csv(out / "ld_blocks_vs_plink_per_signal.csv", index=False)
        pd.DataFrame(summarise(reg[reg["scenario"].str.endswith("_q005")], "pooled_5qtn")).to_csv(
            out / "ld_blocks_vs_plink_per_signal_5qtn.csv", index=False)
        pd.DataFrame(summarise(reg, "pooled_all")).to_csv(out / "ld_blocks_vs_plink_per_signal_main.csv", index=False)
        loci, details = run_loci(a, facts)
        loci.to_csv(out / "ld_blocks_vs_plink_per_signal_loci.csv", index=False)
        details.to_csv(out / "loci_regions.csv", index=False)
        facts["status"] = "ok"
    except Stop as e:
        facts["status"] = f"stop: {e}"
    facts["seconds"] = round(time.time() - t0, 1)
    (out / "per_signal_facts.json").write_text(json.dumps(facts, indent=1, default=str), encoding="utf-8")
    print(json.dumps(facts, indent=1, default=str))
    if facts["status"] != "ok":
        raise SystemExit(facts["status"])
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(pd.read_csv(out / "ld_blocks_vs_plink_per_signal_main.csv").to_string(index=False))
        print(pd.read_csv(out / "ld_blocks_vs_plink_per_signal_loci.csv").to_string(index=False))


if __name__ == "__main__":
    main()
