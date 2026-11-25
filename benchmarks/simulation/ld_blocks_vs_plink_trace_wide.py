"""TRACE's 1.2.0 LD blocks with the LD window doubled: the GUI's "LD window around each lead SNP (kb)"
set to twice the run's automatic window, one extra TRACE setting for ld_blocks_vs_plink_per_signal.py.

The setting (fixed before any result). D is the run's automatic LD decay and W = gwas.windows.ld_window_kb(D)
its automatic window, both from ld_blocks_vs_plink_trace_default.ld_decay_window.
- Blocks: ld_blocks_vs_plink_trace_default.trace_blocks with the decay D and the window 2W. The GUI's
  window field calls the block search this way: it keeps the run's decay, so pairs link up to
  min(max(D, 200 kb), 2W) (gwas/ld.py, find_ld_clusters_genomewide).
- Candidate intervals: ld_blocks_vs_plink_trace_default.candidates (at the loci also unblocked_intervals
  and gwas.isolated.block_candidate_intervals) with the window W, so their edge flank stays W, as in the
  GUI. The 5 Mb rule is unchanged.
- Everything else: TRACE 1.2.0's defaults, the calls of ld_blocks_vs_plink_trace_default.py (unchanged).
--block-factor 1 is that script's default setting (blocks with the window W), for the self-checks.

    python benchmarks/simulation/ld_blocks_vs_plink_trace_wide.py sim --out DIR --p2c P2C [--jobs 4] \\
        [--only SCENARIO/REP ...] [--block-factor 2] [--compare] [--filter-check] [--sim-data SIM] [--code TREE]
    python benchmarks/simulation/ld_blocks_vs_plink_trace_wide.py loci --out DIR --p2c P2C --name NAME \\
        --vcf VCF --pheno PHENO --trait TRAIT --zip RUN.zip [--geno-store-dir DIR] [--block-factor 2] \\
        [--compare] [--filter-check] [--code TREE]
    python benchmarks/simulation/ld_blocks_vs_plink_trace_wide.py tables --out DIR --wide SCORE --repro SCORE \\
        --p2d-regions regions.csv [--summary DIR]

P2C is ld_blocks_vs_plink_trace_default.py's work folder: D and W must equal its sim_facts.json and
loci_NAME.json, and --compare checks the output against its files.
sim: the published simulation, as ld_blocks_vs_plink_trace_default.py sim, with the setting above.
Writes, in DIR and in that script's layout (method trace_default, so ld_blocks_vs_plink_per_signal.py
scores the folder unchanged): intervals.csv, causal_by_method.csv, replicates.csv, membership.csv,
sim_facts.json (that script's keys and the setting), and candidates.csv (scenario, rep, then each
replicate's gwas.isolated.candidate_interval_table, with the SNP lists).
loci: the driver on a real run, as ld_blocks_vs_plink_trace_default.py loci (the CLI's inputs, the
run's GWAS table from its ZIP), with the setting above and no comparison with the run's own tables.
Writes loci_NAME_driver_blocks.csv, loci_NAME_driver_unblocked.csv, loci_NAME_driver_candidates.csv
and loci_NAME.json.
Both stop when an unblocked interval's SNP list differs from the markers strictly inside its bounds
(the template's in the simulation, the GWAS table's at the loci). --compare (with --block-factor 1)
checks that the output equals P2C's: the replicates' rows of its four CSVs (sim), its three driver
tables byte for byte (loci). --filter-check makes td.trace_blocks's two calls directly
(gwas.ld.find_ld_clusters_genomewide, then gwas.ld.filter_contained_blocks) and counts the blocks
the filter removes.
tables: the four tables of ld_blocks_vs_plink_per_signal.py with this setting's TRACE rows added:
the committed lines of sim_summary/ld_blocks_vs_plink_per_signal*.csv byte for byte, each default
TRACE line followed by the line of the run on this setting's folder (--wide), relabelled
trace_wide_bonferroni. Checks that the run on the committed inputs (--repro) reproduces the committed
tables byte for byte, that --wide's PLINK lines equal the committed ones and that its signals and their
TP SNPs equal those of --p2d-regions. Writes ld_blocks_vs_plink_per_signal_wide*.csv and tables_facts.json.
"""
import argparse
import hashlib
import io
import json
import logging
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import ld_blocks_vs_plink_trace_default as td   # P-2c's module, unchanged: it puts --code's tree first on sys.path

p2, ld, iso, annotation = td.p2, td.ld, td.iso, td.annotation
HERE = Path(__file__).resolve().parent
METHOD = td.METHOD                   # "trace_default" inside the work folder
DEFAULT, WIDE = "trace_default_bonferroni", "trace_wide_bonferroni"
DIST_FLOOR_BP = 200_000              # find_ld_clusters_genomewide's pair distance floor (gwas/ld.py)
TABLES = ["ld_blocks_vs_plink_per_signal_main.csv", "ld_blocks_vs_plink_per_signal_5qtn.csv",
          "ld_blocks_vs_plink_per_signal.csv", "ld_blocks_vs_plink_per_signal_loci.csv"]


class Stop(Exception):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setting(decay, window, factor, args):
    """The setting as logged: D, W, the block window, the pair cap (find_ld_clusters_genomewide's
    rule, gwas/ld.py) and the candidate intervals' edge flank and 5 Mb rule."""
    block_window = factor * window
    flank_bp = int(round(float(block_window) * 1000.0))
    cap = (min(max(int(decay * 1000), DIST_FLOOR_BP), flank_bp) if decay is not None and np.isfinite(decay)
           else flank_bp)
    edge, maxbp = td.interval_rules(args, window)
    return dict(block_factor=factor, ld_decay_kb=decay, ld_window_kb=window, block_window_kb=block_window,
                pair_cap_bp=cap, interval_window_kb=window, edge_flank_bp=edge, max_interval_bp=maxbp)


def filter_check(gwas_df, chroms, positions, geno_float, sid, decay, block_window, args, rule, blocks):
    """td.trace_blocks's two calls, made directly: the blocks found, the blocks filter_contained_blocks
    removes, and whether the blocks kept equal td.trace_blocks's."""
    seed_p, top_n = td.seed_settings(args, rule)
    found = ld.find_ld_clusters_genomewide(
        gwas_df=gwas_df, chroms=chroms, positions=positions, geno_imputed=geno_float, sid=sid,
        ld_threshold=args.ld_r2, flank_kb=block_window, ld_decay_kb=decay, min_snps=2,
        top_n=top_n, sig_thresh=seed_p, ld_merge_r2=args.ld_merge_r2)
    kept, removed = ld.filter_contained_blocks(found, min_contained=2)
    return dict(found=int(len(found)), kept=int(len(kept)), removed=int(removed),
                kept_equal_trace_blocks=kept.to_csv(index=False) == blocks.to_csv(index=False))


def marker_index(chroms, positions, sid):
    """{chromosome: (sorted positions, their SNP IDs)}."""
    chroms = np.asarray([annotation.canon_chr(c) for c in chroms])
    positions, sid = np.asarray(positions).astype(np.int64), np.asarray(sid).astype(str)
    out = {}
    for ch in np.unique(chroms):
        m = chroms == ch
        order = np.argsort(positions[m], kind="stable")
        out[str(ch)] = (positions[m][order], sid[m][order])
    return out


def unblocked_lists(table, markers, start, end):
    """The unblocked intervals of a table: each SNP list against the markers strictly inside its
    bounds. Returns the count and the intervals that differ."""
    n, bad = 0, []
    for r in table.to_dict("records"):
        if r.get("Type", "unblocked SNP") != "unblocked SNP":
            continue
        n += 1
        pos, ids = markers.get(annotation.canon_chr(r["Chr"]), (np.array([], dtype=np.int64), np.array([], dtype=str)))
        lo = int(np.searchsorted(pos, int(r[start]), side="right"))
        hi = int(np.searchsorted(pos, int(r[end]), side="left"))
        if sorted(map(str, ids[lo:hi])) != sorted(td.snp_list(r["SNP_IDs"])):
            bad.append(f"chr{r['Chr']}:{r[start]}-{r[end]}")
    return n, bad


def lines(path):
    return [x.rstrip(b"\r") for x in Path(path).read_bytes().split(b"\n") if x.strip()]


# ----------------------------------------------------------------------------------------------
# sim

_W = {}   # this setting's block window and the checks, beside td._S


def _init(sim, decay, window, meff, span, group_of, block_window, check_filter):
    td._init(sim, decay, window, meff, span, group_of)
    S = td._S
    _W.update(block_window=block_window, check_filter=check_filter,
              markers=marker_index(S["chroms"], S["positions"], S["sid"]))


def one_replicate(task):
    """ld_blocks_vs_plink_trace_default.one_replicate with the blocks at this setting's block window
    and the candidate intervals at the run's window, with the candidate table, the SNP list check and,
    when asked, the filter check."""
    scen, rep, res_path, truth = task
    S = td._S
    res = pd.read_csv(res_path).dropna(subset=["PValue"])
    res["Chr"] = res["Chr"].astype(str)
    blocks, err = td.trace_blocks(res, S["chroms"], S["positions"], S["geno_float"], S["sid"], S["decay"],
                                  _W["block_window"], S["args"], S["rule"])
    cand = td.candidates(res, blocks, S["rule"], S["chroms"], S["positions"], S["sid"], S["args"], S["window"])
    # from here on as ld_blocks_vs_plink_trace_default.one_replicate
    sig = res.loc[S["rule"].significant_mask(res)]
    sig_ids = sig["SNP"].astype(str).tolist()
    sig_pos = dict(zip(sig_ids, zip(sig["Chr"].astype(str), sig["Pos"].astype(int))))
    ivs = []
    for r in cand.itertuples(index=False):
        ivs.append(dict(type=r.Type, chr=str(r.Chr), start=int(r.Interval_start), end=int(r.Interval_end),
                        members=td.snp_list(r.SNP_IDs), block_start=r.Block_start, block_end=r.Block_end))
    count = {s: 0 for s in sig_ids}
    for iv in ivs:
        for s in iv["members"]:
            if s in count:
                count[s] += 1
    none = [s for s, n in count.items() if n == 0]
    many = [s for s, n in count.items() if n > 1]
    reasons = {}
    for s in none:
        ch, p = sig_pos[s]
        in_span = any(iv["type"] == "LD block" and iv["chr"] == ch and int(iv["block_start"]) <= p <= int(iv["block_end"])
                      for iv in ivs)
        key = "inside an LD block's span, not a member" if in_span else "in no interval's SNP list"
        reasons[key] = reasons.get(key, 0) + 1
    for s in many:
        reasons["in two or more SNP lists"] = reasons.get("in two or more SNP lists", 0) + 1
    in_two_spans = sum(1 for s in sig_ids
                       if sum(iv["chr"] == sig_pos[s][0] and iv["start"] <= sig_pos[s][1] <= iv["end"] for iv in ivs) > 1)
    overlapping_pairs = 0
    for ch in {iv["chr"] for iv in ivs}:
        spans = sorted((iv["start"], iv["end"]) for iv in ivs if iv["chr"] == ch)
        for k in range(1, len(spans)):
            overlapping_pairs += sum(1 for j in range(k) if spans[j][1] >= spans[k][0])
    causal = list(zip(truth["qtn_snp_ids"], [str(c) for c in truth["qtn_chrs"]], truth["qtn_positions"]))
    group_of, span = S["group_of"], S["span"]
    causal_rows, iv_rows = [], []
    for c, ch, p in causal:
        inside = any(iv["chr"] == ch and iv["start"] <= p <= iv["end"] for iv in ivs)
        causal_rows.append(dict(scenario=scen, rep=rep, method=METHOD, snp=c, chr=ch, pos=p, ld_kb=span[c],
                                group=group_of[c], captured=inside))
    sig_set = set(sig_ids)
    for iv in ivs:
        held = [c for c, ch, p in causal if ch == iv["chr"] and iv["start"] <= p <= iv["end"]]
        if held:
            grp = "short" if any(group_of[c] == "short" for c in held) else "long"
        else:
            same = [(min(abs(p - iv["start"]), abs(p - iv["end"])), p, c) for c, ch, p in causal if ch == iv["chr"]]
            grp = group_of[sorted(same)[0][2]] if same else "none"
        iv_rows.append(dict(scenario=scen, rep=rep, method=METHOD, chr=iv["chr"], start=iv["start"], end=iv["end"],
                            width_kb=(iv["end"] - iv["start"]) / 1000.0, n_snps=len(iv["members"]),
                            n_sig=sum(s in sig_set for s in iv["members"]),
                            n_blocks=1 if iv["type"] == "LD block" else 0,
                            n_genes=S["count_genes"](iv["chr"], iv["start"], iv["end"]),
                            true_hit=bool(held), group=grp))
    n_block_iv = sum(iv["type"] == "LD block" for iv in ivs)
    rep_row = dict(scenario=scen, rep=rep, n_sig=len(sig_ids), n_blocks=int(len(blocks)), n_candidate_intervals=len(ivs),
                   n_block_intervals=n_block_iv, n_unblocked_intervals=len(ivs) - n_block_iv,
                   n_block_intervals_without_significant=sum(
                       iv["type"] == "LD block" and not any(s in sig_set for s in iv["members"]) for iv in ivs),
                   block_error=err or "")
    check = dict(scenario=scen, rep=rep, n_sig=len(sig_ids), sig_in_no_interval=len(none),
                 sig_in_two_or_more=len(many), reasons=json.dumps(reasons, sort_keys=True),
                 exception_snps=" ".join((none + many)[:20]), sig_in_two_or_more_spans=in_two_spans,
                 overlapping_interval_pairs=overlapping_pairs)
    # this script's additions
    table = cand.copy()
    table.insert(0, "rep", rep)
    table.insert(0, "scenario", scen)
    n_lists, bad_lists = unblocked_lists(cand, _W["markers"], "Interval_start", "Interval_end")
    extra = dict(scenario=scen, rep=rep, unblocked=n_lists, lists_differ=bad_lists,
                 types=sorted(set(cand["Type"])) if len(cand) else [],
                 truncated=int(cand["Interval_truncated"].astype(bool).sum()) if len(cand) else 0)
    if _W["check_filter"]:
        extra["filter"] = filter_check(res, S["chroms"], S["positions"], S["geno_float"], S["sid"], S["decay"],
                                       _W["block_window"], S["args"], S["rule"], blocks)
    return iv_rows, causal_rows, rep_row, check, table, extra


def compare_lines(frame, path, keys):
    """A frame's CSV lines against the lines of a saved file (the same header) whose scenario and rep
    are in keys, in order."""
    saved = lines(path)
    mine = [x.encode() for x in frame.to_csv(index=False, lineterminator="\n").split("\n") if x]
    rows = [x for x in saved[1:] if tuple(v.decode() for v in x.split(b",", 2)[:2]) in keys]
    return dict(header_equal=mine[0] == saved[0], rows=len(mine) - 1, rows_saved=len(rows),
                equal=mine[0] == saved[0] and mine[1:] == rows)


def cmd_sim(a):
    logging.getLogger("gwas").setLevel(logging.WARNING)
    sim, out, ref_dir = Path(a.sim_data), Path(a.out), Path(a.p2c)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    facts = dict(script=str(Path(__file__).resolve()), script_sha256=sha256(__file__), only=a.only)
    try:
        ref = json.loads((ref_dir / "sim_facts.json").read_text(encoding="utf-8"))
        geno, ids, snp_map = p2.load_template(sim)
        chroms, positions = snp_map["Chr"].to_numpy(), snp_map["Pos"].astype(int).to_numpy()
        meff, _ = td._compute_meff_li_ji_impl(geno)
        decay, window, dinfo = td.ld_decay_window(geno, chroms, positions)
        args = td.cli_default_options()
        rule = td.rule_from_cli_args(args, len(snp_map), meff)
        st = setting(decay, window, a.block_factor, args)
        facts["setting"] = st
        facts["setting_equals_p2c"] = dict(ld_decay_kb=decay == ref["ld_decay_kb"], ld_window_kb=window == ref["ld_window_kb"],
                                           p2c_ld_decay_kb=ref["ld_decay_kb"], p2c_ld_window_kb=ref["ld_window_kb"])
        if decay != ref["ld_decay_kb"] or window != ref["ld_window_kb"]:
            raise Stop(f"D {decay!r} and W {window!r} differ from P-2c's ({ref['ld_decay_kb']!r}, {ref['ld_window_kb']!r})")
        print(f"setting: D {decay!r} kb, W {window} kb, block window {st['block_window_kb']} kb, "
              f"pair cap {st['pair_cap_bp']} bp, edge flank {st['edge_flank_bp']} bp", flush=True)
        reps_all = list(p2.replicates(sim))
        truths = {(s, r): json.loads(t.read_text(encoding="utf-8")) for s, r, _, t in reps_all}
        causal_all = sorted({c for t in truths.values() for c in t["qtn_snp_ids"]})
        span, check = p2.ld_context(geno, snp_map, causal_all)
        five = [span[c] for (s, _), t in truths.items() if s.endswith("_q005") for c in t["qtn_snp_ids"]]
        split = float(np.median(five))
        group_of = {c: ("short" if v <= split else "long") for c, v in span.items()}
        setup_s = time.time() - t0
        reps = [x for x in reps_all if not a.only or f"{x[0]}/{x[1]}" in a.only]
        if a.only and len(reps) != len(set(a.only)):
            raise Stop("--only names replicates that are not there")
        tasks = [(s, r, str(p), truths[(s, r)]) for s, r, p, _ in reps]
        init = (str(sim), decay, window, meff, span, group_of, st["block_window_kb"], a.filter_check)
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
        ivd = pd.DataFrame([row for r in results for row in r[0]])
        cd = pd.DataFrame([row for r in results for row in r[1]])
        reps_df = pd.DataFrame([r[2] for r in results])
        checks = pd.DataFrame([r[3] for r in results])
        extra = [r[5] for r in results]
        ivd.to_csv(out / "intervals.csv", index=False)
        cd.to_csv(out / "causal_by_method.csv", index=False)
        reps_df.to_csv(out / "replicates.csv", index=False)
        checks.to_csv(out / "membership.csv", index=False)
        pd.concat([r[4] for r in results], ignore_index=True).to_csv(out / "candidates.csv", index=False)
        p2_causal = td.HERE / "sim_summary" / "ld_blocks_vs_plink_causal.csv"
        ctx = pd.DataFrame([dict(snp=c, ld_kb=v, group=group_of[c]) for c, v in sorted(span.items())])
        reasons = {}
        for x in checks.reasons:
            for k, v in json.loads(x).items():
                reasons[k] = reasons.get(k, 0) + v
        edge, maxbp = st["edge_flank_bp"], st["max_interval_bp"]
        facts.update(   # ld_blocks_vs_plink_trace_default.py's keys
            template=f"{geno.shape[0]} x {geno.shape[1]}", template_dtype=str(geno.dtype), replicates=len(reps),
            meff=int(meff), rule=dict(rule=rule.rule, p_threshold=rule.p_threshold, p_threshold_repr=repr(rule.p_threshold),
                                      n_tests=rule.n_tests),
            ld_decay_kb=decay, ld_window_kb=window, ld_decay_detail=dinfo,
            ld_decay_function="ld_decay_window (gwas/pipeline.py S19: gwas.ld.pairwise_r2, gwas.ld.ld_decay("
                              "ld_threshold=0.2, max_dist_kb=5000), the median) and gwas.windows.ld_window_kb",
            edge_flank_bp=edge, max_interval_bp=maxbp, seed_p=args.ld_seed_p, top_n=args.ld_top_n, ld_r2=args.ld_r2,
            ld_merge_r2=args.ld_merge_r2, seed_mode=args.ld_seed_mode,
            causal_unique=len(causal_all), ld_split_kb=split, ld_split_from=len(five), ld_check=check,
            ld_groups_equal_p2=(ctx.to_csv(index=False, lineterminator="\r\n").encode()
                                == p2_causal.read_bytes()) if p2_causal.is_file() else None,
            block_errors=int((reps_df.block_error != "").sum()) if len(reps_df) else 0,
            candidate_intervals=int(len(ivd)), block_intervals=int(reps_df.n_block_intervals.sum()) if len(reps_df) else 0,
            unblocked_intervals=int(reps_df.n_unblocked_intervals.sum()) if len(reps_df) else 0,
            block_intervals_without_significant=int(reps_df.n_block_intervals_without_significant.sum()) if len(reps_df) else 0,
            widest_interval_kb=float(ivd.width_kb.max()) if len(ivd) else None,
            sig_in_no_interval=int(checks.sig_in_no_interval.sum()) if len(checks) else 0,
            sig_in_two_or_more=int(checks.sig_in_two_or_more.sum()) if len(checks) else 0,
            exception_reasons=reasons,
            sig_in_two_or_more_spans=int(checks.sig_in_two_or_more_spans.sum()) if len(checks) else 0,
            overlapping_interval_pairs=int(checks.overlapping_interval_pairs.sum()) if len(checks) else 0,
            setup_s=setup_s, gwas_file=td.gwas.__file__, cli_file=td.cli.__file__)
        bad = [f"{e['scenario']}/{e['rep']} {x}" for e in extra for x in e["lists_differ"]]
        facts["unblocked_snp_lists"] = dict(intervals=sum(e["unblocked"] for e in extra),
                                            equal=sum(e["unblocked"] for e in extra) - len(bad), differ=bad[:20],
                                            n_differ=len(bad))
        facts["candidate_types"] = sorted({t for e in extra for t in e["types"]})
        facts["intervals_truncated_5mb"] = sum(e["truncated"] for e in extra)
        if a.filter_check:
            facts["filter_check"] = dict(per_replicate=[dict(scenario=e["scenario"], rep=e["rep"], **e["filter"])
                                                        for e in extra],
                                         removed=sum(e["filter"]["removed"] for e in extra),
                                         kept_equal_trace_blocks=all(e["filter"]["kept_equal_trace_blocks"] for e in extra))
        if a.compare:
            keys = {(s, r) for s, r, _, _ in reps}
            facts["compare_p2c"] = {name: compare_lines(frame, ref_dir / name, keys) for name, frame in
                                    (("intervals.csv", ivd), ("causal_by_method.csv", cd), ("replicates.csv", reps_df),
                                     ("membership.csv", checks))}
        facts["seconds"] = round(time.time() - t0, 1)
        fails = []
        if bad:
            fails.append("an unblocked interval's SNP list differs from the markers strictly inside its bounds")
        if set(facts["candidate_types"]) - {"LD block", "unblocked SNP"}:
            fails.append(f"candidate types {facts['candidate_types']}")
        if a.filter_check and (facts["filter_check"]["removed"] or not facts["filter_check"]["kept_equal_trace_blocks"]):
            fails.append("the filter check")
        if a.compare and not all(v["equal"] for v in facts["compare_p2c"].values()):
            fails.append("the rows differ from P-2c's")
        if fails:
            raise Stop("; ".join(fails))
        facts["status"] = "ok"
    except Stop as e:
        facts["status"] = f"stop: {e}"
    (out / "sim_facts.json").write_text(json.dumps(facts, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in facts.items() if k != "filter_check"}, indent=1, default=str))
    if "filter_check" in facts:
        print(json.dumps({k: v for k, v in facts["filter_check"].items()}, indent=1, default=str))
    if facts["status"] != "ok":
        raise SystemExit(facts["status"])


# ----------------------------------------------------------------------------------------------
# loci

def cmd_loci(a):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log = logging.getLogger("cli")
    out, ref_dir = Path(a.out), Path(a.p2c)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    facts = dict(name=a.name, zip=str(a.zip), gwas_file=td.gwas.__file__, cli_file=td.cli.__file__,
                 script=str(Path(__file__).resolve()), script_sha256=sha256(__file__))
    try:
        ref = json.loads((ref_dir / f"loci_{a.name}.json").read_text(encoding="utf-8"))
        args = td.cli_default_options(vcf=a.vcf, pheno=a.pheno, trait=a.trait, output=str(out),
                                      geno_store_dir=a.geno_store_dir)
        zf = zipfile.ZipFile(a.zip)
        meta = json.loads(zf.read("run_metadata.json"))
        gwas_df = pd.read_csv(io.BytesIO(zf.read(f"tables/GWAS_{a.trait}.csv")))
        times = {}
        t = time.time()
        inp = td.cli_inputs(args, log)
        times["inputs_s"] = time.time() - t
        try:
            t = time.time()
            meff, _ = td._compute_meff_li_ji_impl(inp["geno_imputed"])
            rule = td.rule_from_cli_args(args, len(inp["sid"]), meff)
            times["meff_s"] = time.time() - t
            t = time.time()
            decay, window, dinfo = td.ld_decay_window(inp["geno_imputed"], inp["chroms"], inp["positions"])
            times["ld_decay_s"] = time.time() - t
            st = setting(decay, window, a.block_factor, args)
            facts["setting"] = st
            facts["setting_equals_p2c"] = dict(ld_decay_kb=decay == ref["ld_decay_kb"],
                                               ld_window_kb=window == ref["ld_window_kb"],
                                               p2c_ld_decay_kb=ref["ld_decay_kb"], p2c_ld_window_kb=ref["ld_window_kb"])
            if decay != ref["ld_decay_kb"] or window != ref["ld_window_kb"]:
                raise Stop(f"D {decay!r} and W {window!r} differ from P-2c's ({ref['ld_decay_kb']!r}, "
                           f"{ref['ld_window_kb']!r})")
            log.info(f"setting: D {decay!r} kb, W {window} kb, block window {st['block_window_kb']} kb, "
                     f"pair cap {st['pair_cap_bp']} bp, edge flank {st['edge_flank_bp']} bp")
            t = time.time()
            geno_float = inp["geno_imputed"].astype(float)
            blocks, err = td.trace_blocks(gwas_df, inp["chroms"], inp["positions"], geno_float, inp["sid"], decay,
                                          st["block_window_kb"], args, rule)
            times["blocks_s"] = time.time() - t
            if a.filter_check:
                t = time.time()
                facts["filter_check"] = filter_check(gwas_df, inp["chroms"], inp["positions"], geno_float, inp["sid"],
                                                     decay, st["block_window_kb"], args, rule, blocks)
                times["filter_check_s"] = time.time() - t
            t = time.time()
            edge, maxbp = td.interval_rules(args, window)
            biv = iso.block_candidate_intervals(blocks, inp["chroms"], inp["positions"], inp["sid"],
                                                edge_flank_bp=edge, max_interval_bp=maxbp)
            drv_blocks = blocks.copy()
            if len(drv_blocks):
                drv_blocks["Interval_start"] = biv["Interval_start"].to_numpy()
                drv_blocks["Interval_end"] = biv["Interval_end"].to_numpy()
            drv_iso = td.unblocked_intervals(gwas_df, blocks, rule, inp["chroms"], inp["positions"], inp["sid"], args,
                                             window)
            cand = td.candidates(gwas_df, blocks, rule, inp["chroms"], inp["positions"], inp["sid"], args, window)
            times["intervals_s"] = time.time() - t
        finally:
            if inp.get("store_dir") is not None:
                from gwas import genostore
                genostore.remove_store(inp["store_dir"], inp["parsed"])
        bcols = ["Chr", "Start (bp)", "End (bp)", "Interval_start", "Interval_end", "lead_snp", "SNP_IDs"]
        icols = ["Chr", "Start (bp)", "End (bp)", "lead_snp", "SNP_IDs"]
        names = {t: f"loci_{a.name}_driver_{t}.csv" for t in ("blocks", "unblocked", "candidates")}
        drv_blocks[[c for c in bcols if c in drv_blocks.columns]].to_csv(out / names["blocks"], index=False)
        drv_iso[[c for c in icols if c in drv_iso.columns]].to_csv(out / names["unblocked"], index=False)
        cand.to_csv(out / names["candidates"], index=False)
        markers = marker_index(gwas_df["Chr"], gwas_df["Pos"], gwas_df["SNP"])
        n_u, bad_u = unblocked_lists(drv_iso, markers, "Start (bp)", "End (bp)")
        n_c, bad_c = unblocked_lists(cand, markers, "Interval_start", "Interval_end")
        facts.update(
            table_mode=inp["plan_mode"], samples=inp["samples"], snps=int(len(inp["sid"])), meff=int(meff),
            rule=dict(rule=rule.rule, p_threshold=rule.p_threshold, n_tests=rule.n_tests),
            ld_decay_kb=decay, ld_window_kb=window, ld_decay_detail=dinfo,
            run_metadata={k: meta.get(k) for k in ("LD decay (kb)", "LD flank (kb)", "LD blocks (MLM)", "Samples",
                                                   "SNPs (post-QC)", "Significance", "Genotype table",
                                                   "Isolated SNPs (no LD block)")},
            block_error=err, blocks=int(len(blocks)), unblocked_intervals=int(len(drv_iso)),
            candidate_intervals=int(len(cand)),
            candidate_types=sorted(set(cand["Type"])) if len(cand) else [],
            intervals_truncated_5mb=int(cand["Interval_truncated"].astype(bool).sum()) if len(cand) else 0,
            unblocked_snp_lists=dict(unblocked_table=dict(intervals=n_u, equal=n_u - len(bad_u), differ=bad_u),
                                     candidate_table=dict(intervals=n_c, equal=n_c - len(bad_c), differ=bad_c),
                                     markers="the run's GWAS table"),
            times_s=times)
        if a.compare:
            facts["compare_p2c"] = {t: dict(equal=(out / n).read_bytes() == (ref_dir / n).read_bytes(),
                                            sha256=sha256(out / n), p2c_sha256=sha256(ref_dir / n))
                                    for t, n in names.items()}
        fails = []
        if bad_u or bad_c:
            fails.append("an unblocked interval's SNP list differs from the markers strictly inside its bounds")
        if a.filter_check and (facts["filter_check"]["removed"] or not facts["filter_check"]["kept_equal_trace_blocks"]):
            fails.append("the filter check")
        if a.compare and not all(v["equal"] for v in facts["compare_p2c"].values()):
            fails.append("the driver tables differ from P-2c's")
        if fails:
            raise Stop("; ".join(fails))
        facts["status"] = "ok"
    except Stop as e:
        facts["status"] = f"stop: {e}"
    facts["seconds"] = round(time.time() - t0, 1)
    (out / f"loci_{a.name}.json").write_text(json.dumps(facts, indent=1, default=str), encoding="utf-8")
    print(json.dumps(facts, indent=1, default=str))
    if facts["status"] != "ok":
        raise SystemExit(facts["status"])


# ----------------------------------------------------------------------------------------------
# tables

def cmd_tables(a):
    out, src, wide, repro = Path(a.out), Path(a.summary), Path(a.wide), Path(a.repro)
    out.mkdir(parents=True, exist_ok=True)
    facts = dict(script=str(Path(__file__).resolve()), script_sha256=sha256(__file__), summary=str(src),
                 wide=str(wide), repro=str(repro))
    try:
        facts["reproduction"] = {n: dict(equal=(repro / n).read_bytes() == (src / n).read_bytes(),
                                         sha256=sha256(repro / n), committed_sha256=sha256(src / n)) for n in TABLES}
        wf = json.loads((wide / "per_signal_facts.json").read_text(encoding="utf-8"))
        rf = json.loads((repro / "per_signal_facts.json").read_text(encoding="utf-8"))
        facts["runs_status"] = dict(wide=wf["status"], repro=rf["status"])
        for name in TABLES:
            mcol = 4 if name.endswith("_loci.csv") else 1
            old, new = lines(src / name), lines(wide / name)
            if old[0] != new[0]:
                raise Stop(f"{name}: the headers differ")
            trace_new = {}
            for x in new[1:]:
                f = x.split(b",")
                if f[mcol].decode() == DEFAULT:
                    trace_new[f[0]] = f
            plink_old = [x for x in old[1:] if x.split(b",")[mcol].decode() != DEFAULT]
            plink_new = [x for x in new[1:] if x.split(b",")[mcol].decode() != DEFAULT]
            body = []
            for x in old[1:]:
                body.append(x)
                f = x.split(b",")
                if f[mcol].decode() == DEFAULT:
                    y = list(trace_new[f[0]])
                    y[mcol] = WIDE.encode()
                    body.append(b",".join(y))
            out_name = name.replace("per_signal", "per_signal_wide")
            (out / out_name).write_bytes(b"\r\n".join([old[0]] + body) + b"\r\n")
            facts[out_name] = dict(lines=len(body), committed_lines=len(old) - 1, trace_wide_lines=len(body) - len(old) + 1,
                                   plink_lines_equal_committed=plink_new == plink_old, plink_lines=len(plink_old),
                                   committed_lines_kept=[x for x in body if x.split(b",")[mcol].decode() != WIDE] == old[1:])
        cols = ["scenario", "rep", "causal_snp", "n_tp", "tp_hash"]
        rw = pd.read_csv(wide / "regions.csv")
        rd = pd.read_csv(a.p2d_regions)
        same = {}
        for m in sorted(set(rd["method"])):
            x = rw[rw["method"] == m][cols].reset_index(drop=True)
            y = rd[rd["method"] == m][cols].reset_index(drop=True)
            same[m] = dict(signals=int(len(x)), p2d_signals=int(len(y)), equal=bool(x.equals(y)))
        facts["same_signals"] = same
        fails = [n for n, v in facts["reproduction"].items() if not v["equal"]]
        fails += [k for k, v in facts["runs_status"].items() if v != "ok"]
        fails += [n for n in TABLES if not facts[n.replace("per_signal", "per_signal_wide")]["plink_lines_equal_committed"]]
        fails += [m for m, v in same.items() if not v["equal"]]
        if fails:
            raise Stop(f"failed: {fails}")
        facts["status"] = "ok"
    except Stop as e:
        facts["status"] = f"stop: {e}"
    (out / "tables_facts.json").write_text(json.dumps(facts, indent=1, default=str), encoding="utf-8")
    print(json.dumps(facts, indent=1, default=str))
    if facts["status"] != "ok":
        raise SystemExit(facts["status"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="TRACE's LD blocks with the LD window doubled")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sim", "loci", "tables"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, help="the work folder")
        p.add_argument("--code", default=None, help="the TRACE tree to import (default: this repository)")
        if name in ("sim", "loci"):
            p.add_argument("--p2c", required=True, help="ld_blocks_vs_plink_trace_default.py's work folder")
            p.add_argument("--block-factor", type=int, default=2, help="the block window in automatic windows")
            p.add_argument("--compare", action="store_true", help="check the output against P2C's (factor 1)")
            p.add_argument("--filter-check", action="store_true", help="count the blocks the filter removes")
        if name == "sim":
            p.add_argument("--sim-data", default=str(p2.SIM), help="the published simulation folder")
            p.add_argument("--jobs", type=int, default=1)
            p.add_argument("--only", nargs="*", default=None, help="only these SCENARIO/REP (a check)")
        if name == "loci":
            p.add_argument("--name", required=True)
            p.add_argument("--vcf", required=True)
            p.add_argument("--pheno", required=True)
            p.add_argument("--trait", required=True)
            p.add_argument("--zip", required=True, help="the run's results ZIP")
            p.add_argument("--geno-store-dir", default=None, help="as the run's --geno-store-dir")
        if name == "tables":
            p.add_argument("--summary", default=str(HERE / "sim_summary"), help="the committed per signal tables")
            p.add_argument("--wide", required=True, help="ld_blocks_vs_plink_per_signal.py's output on this setting")
            p.add_argument("--repro", required=True, help="its output on the committed inputs")
            p.add_argument("--p2d-regions", required=True, help="the committed run's regions.csv")
    a = ap.parse_args(argv)
    print(f"gwas: {td.gwas.__file__}; cli: {td.cli.__file__}", flush=True)
    {"sim": cmd_sim, "loci": cmd_loci, "tables": cmd_tables}[a.cmd](a)


if __name__ == "__main__":
    main()
