"""TRACE's 1.2.0 default LD blocks and candidate intervals against PLINK's blocks and clumps.

ld_blocks_vs_plink.py (P-2) and ld_blocks_vs_plink_clump.py (P-2b) compared PLINK with TRACE's
blocks under the published simulation evaluation's settings. This script forms TRACE's blocks as
TRACE 1.2.0 does by default (the CLI and the GUI alike) and takes the intervals TRACE reports,
the candidate intervals: one per LD block and one per run of unblocked significant SNPs. It calls
the CLI's own functions with the CLI's default arguments (cli._build_parser, through
gwas.pipeline.cli_default_options), in the order gwas/pipeline.py calls them:
- the LD decay (stage S19): for each chromosome with at least 50 SNPs, at most 1,500 SNPs taken
  by np.linspace, gwas.ld.pairwise_r2 and gwas.ld.ld_decay(ld_threshold=0.2, max_dist_kb=5000),
  then the median; the window is gwas.windows.ld_window_kb(decay);
- the blocks (S24): gwas.ld.find_ld_clusters_genomewide with --ld-r2, the window, the decay, at
  least 2 SNPs, the seeds of --ld-seed-mode (suggestive: --ld-seed-p and the --ld-top-n floor) and
  --ld-merge-r2, then gwas.ld.filter_contained_blocks(min_contained=2);
- the reporting rule gwas.significance.rule_from_cli_args (Bonferroni by default), with M_eff
  from gwas.plotting._compute_meff_li_ji_impl as the CLI passes it;
- the candidate intervals: gwas.isolated.block_candidate_intervals, find_unblocked_significant_snps
  with build_flanking_intervals, and candidate_interval_table, with the CLI's edge flank
  (gwas.isolated.default_edge_flank_bp) and its 5 Mb rule.
The PLINK rows are not recomputed: they are copied from the committed tables of
ld_blocks_vs_plink.py (--blocks at 200 and 600 kb) and ld_blocks_vs_plink_clump.py (--clump C1
and C2).

    python benchmarks/simulation/ld_blocks_vs_plink_trace_default.py loci --out DIR --name NAME \\
        --vcf VCF --pheno PHENO --trait TRAIT --zip RUN.zip [--geno-store-dir DIR] [--code TREE]
    python benchmarks/simulation/ld_blocks_vs_plink_trace_default.py sim --out DIR [--jobs 4] \\
        [--only SCENARIO/REP ...] [--code TREE]
    python benchmarks/simulation/ld_blocks_vs_plink_trace_default.py tables --out DIR \\
        --tomato-zip TOMATO.zip --rice-zip RICE.zip --rice-gene-model locus.gff [--code TREE]

--code: the TRACE tree whose gwas, annotation and cli are imported (default: this repository).
loci: the driver on a real CLI run. The CLI's inputs come from its own loading (gwas.storeplan.plan,
then gwas.genostore.ingest_vcf on disk or gwas.io.load_vcf_cached and ParsedGenotypes in memory,
the phenotype file as the CLI reads it) and the steps S05 to S09 of gwas/pipeline.py (the same
functions, the TraitStore ones on disk). The GWAS table is the run's own, from its ZIP. The
driver's block table and unblocked intervals are compared, row by row, with the run's
LD_blocks_annotated_MLM.csv and Isolated_SNP_intervals_MLM.csv. Writes DIR/loci_NAME.json and
the driver's tables.
sim: the published simulation (sim_data/, ld_blocks_vs_plink.py's inputs: the template as it
gives TRACE, each replicate's TRACE LOCO MLM p-values and truth.json). The template's LD decay,
window and M_eff once, then per replicate the blocks and candidate intervals; each causal SNP's
capture and each interval's metrics as ld_blocks_vs_plink.py defines them (every candidate
interval is an interval; position inside its span; genes over its span; the same LD groups).
Writes DIR/trace_default_rows.csv, DIR/intervals.csv, DIR/causal_by_method.csv,
DIR/replicates.csv, DIR/membership.csv and DIR/sim_facts.json.
tables: ld_blocks_vs_plink_trace_default_main.csv, ld_blocks_vs_plink_trace_default.csv (per
scenario, then the pooled 5-QTN rows) and ld_blocks_vs_plink_trace_default_loci.csv, with five
methods: TRACE's 1.2.0 defaults and the four PLINK methods, whose rows are copied byte for byte
from sim_summary/; writes DIR/tables_facts.json. The loci rows of TRACE come from the runs' own
tables: every candidate interval at the locus (tomato chromosome 2; rice GS3 and GL7, the gene
with 500 kb on each side), flagged when it holds the locus's lead SNP (as ld_blocks_vs_plink.py
picks it).
"""
import argparse
import io
import json
import logging
import sys
import time
import zipfile
from pathlib import Path


def _code_dir():
    a = sys.argv
    if "--code" in a:
        return Path(a[a.index("--code") + 1]).resolve()
    return Path(__file__).resolve().parents[2]


CODE = _code_dir()
sys.path.insert(0, str(CODE))   # gwas, annotation and cli from this tree

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import annotation  # noqa: E402
import cli  # noqa: E402
import gwas  # noqa: E402
import gwas.ld as ld  # noqa: E402
from gwas import isolated as iso  # noqa: E402
from gwas.genostore import is_lazy  # noqa: E402
from gwas.pipeline import cli_default_options  # noqa: E402
from gwas.plotting import _compute_meff_li_ji_impl  # noqa: E402
from gwas.significance import rule_from_cli_args  # noqa: E402
from gwas.windows import ld_window_kb  # noqa: E402

import ld_blocks_vs_plink as p2  # noqa: E402  P-2's comparison, in this folder, unchanged

HERE = Path(__file__).resolve().parent
METHOD = "trace_default"
PLINK_SOURCES = {"plink_200kb": "ld_blocks_vs_plink", "plink_600kb": "ld_blocks_vs_plink",
                 "plink_clump_c1": "ld_blocks_vs_plink_clump", "plink_clump_c2": "ld_blocks_vs_plink_clump"}
METHODS = [METHOD] + list(PLINK_SOURCES)


# ----------------------------------------------------------------------------------------------
# the CLI's steps

def ld_decay_window(geno_imputed, chroms, positions):
    """The LD decay and window of gwas/pipeline.py (stage S19), the same calls with the same
    thinning and median; with the per-chromosome values and the count of grid-limited ones."""
    ld_distances, n_censored = [], 0
    chroms, positions = np.asarray(chroms), np.asarray(positions)
    geno_float = geno_imputed.astype(float)
    for ch in np.unique(chroms):
        ch_mask = chroms == ch
        pos_ch = positions[ch_mask]
        if is_lazy(geno_float):
            if len(pos_ch) < 50:
                continue
            ci = np.flatnonzero(ch_mask)
            if len(pos_ch) > 1500:
                idx = np.linspace(0, len(pos_ch) - 1, 1500, dtype=int)
                pos_ch, ci = pos_ch[idx], ci[idx]
            geno_ch = geno_float[:, ci]
        else:
            geno_ch = geno_float[:, ch_mask]
            if len(pos_ch) < 50:
                continue
            if len(pos_ch) > 1500:
                idx = np.linspace(0, len(pos_ch) - 1, 1500, dtype=int)
                pos_ch, geno_ch = pos_ch[idx], geno_ch[:, idx]
        r2_mat = ld.pairwise_r2(geno_ch)
        dk, _, dfld = ld.ld_decay(pos_ch, r2_mat, ld_threshold=0.2, max_dist_kb=5000)
        if np.isfinite(dk):
            ld_distances.append(float(dk))
            if dfld.attrs.get("ld_decay_censored"):
                n_censored += 1
    decay = float(np.median(ld_distances)) if ld_distances else None
    return decay, ld_window_kb(decay), dict(per_chromosome=ld_distances, grid_limited=n_censored)


def seed_settings(args, rule):
    """The block seeds of gwas/pipeline.py: suggestive (the default) = --ld-seed-p and the
    --ld-top-n floor; significant = the genome-wide threshold and no floor."""
    if args.ld_seed_mode == "significant":
        return (rule.p_threshold if rule.p_threshold is not None else None), 0
    return args.ld_seed_p, args.ld_top_n


def trace_blocks(gwas_df, chroms, positions, geno_float, sid, decay, window, args, rule):
    """The blocks of gwas/pipeline.py (stage S24): find_ld_clusters_genomewide, then
    filter_contained_blocks(min_contained=2). An exception gives no blocks, as in the CLI."""
    seed_p, top_n = seed_settings(args, rule)
    try:
        blocks = ld.find_ld_clusters_genomewide(
            gwas_df=gwas_df, chroms=chroms, positions=positions, geno_imputed=geno_float, sid=sid,
            ld_threshold=args.ld_r2, flank_kb=window, ld_decay_kb=decay, min_snps=2,
            top_n=top_n, sig_thresh=seed_p, ld_merge_r2=args.ld_merge_r2)
        blocks, _ = ld.filter_contained_blocks(blocks, min_contained=2)
        return blocks, None
    except Exception as e:   # the CLI logs a warning and goes on without blocks
        return pd.DataFrame(), f"{type(e).__name__}: {e}"


def interval_rules(args, window):
    edge = iso.default_edge_flank_bp(getattr(args, "isolated_edge_flank_kb", None), window)
    return edge, int(float(getattr(args, "isolated_max_interval_mb", 5.0)) * 1_000_000)


def unblocked_intervals(gwas_df, blocks, rule, chroms, positions, sid, args, window):
    """The isolated-SNP rescue's intervals (gwas/pipeline.py, stage S25): the unblocked
    significant SNPs, then their flanking intervals."""
    edge, maxbp = interval_rules(args, window)
    unb = iso.find_unblocked_significant_snps(gwas_df, blocks, rule, seed_p_used=args.ld_seed_p,
                                              top_n_used=args.ld_top_n)
    sig_ids = set(gwas_df.loc[rule.significant_mask(gwas_df), "SNP"].astype(str))
    return iso.build_flanking_intervals(unb, chroms, positions, sid, edge_flank_bp=edge,
                                        max_interval_bp=maxbp, significant_snp_ids=sig_ids)


def candidates(gwas_df, blocks, rule, chroms, positions, sid, args, window):
    """Every candidate interval: gwas.isolated.candidate_interval_table."""
    edge, maxbp = interval_rules(args, window)
    return iso.candidate_interval_table(blocks, gwas_df, rule, chroms, positions, sid, edge_flank_bp=edge,
                                        max_interval_bp=maxbp, seed_p_used=args.ld_seed_p,
                                        top_n_used=args.ld_top_n)


def snp_list(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [str(s).strip() for s in x if str(s).strip()]
    return [s.strip() for s in str(x).split(",") if s.strip() and str(x) != "nan"]


# ----------------------------------------------------------------------------------------------
# loci: the driver against a real CLI run

def cli_inputs(args, log):
    """The CLI's inputs to its post-GWAS steps: cli.run_pipeline's loading and
    gwas.pipeline._run_trait's steps S05 to S09, the same calls in the same order."""
    from gwas import storeplan
    from gwas.pipeline import ParsedGenotypes
    from gwas.qc import (_pipeline_build_geno_matrices, _pipeline_harmonize_ids,
                         _pipeline_phenotype_qc, _pipeline_snp_qc)
    vcf_path, pheno_path = Path(args.vcf), Path(args.pheno)
    store_base = Path(args.geno_store_dir) if getattr(args, "geno_store_dir", None) else Path(args.output)
    plan = storeplan.plan(vcf_path, choice=getattr(args, "geno_store", "auto"), store_base=store_base,
                          models=args.model, subsampling=getattr(args, "subsampling", False),
                          boot_jobs=getattr(args, "boot_jobs", 1), export_qc=getattr(args, "export_qc", False),
                          impute=getattr(args, "impute", "mean"))
    log.info(plan.message)
    if plan.stop:
        raise SystemExit("stop: " + plan.stop)
    store_dir = None
    if plan.mode == "disk":
        from gwas import genostore
        store_dir = genostore.new_store_dir(store_base)
        parsed = genostore.ingest_vcf(vcf_path, store_dir, log=log)
    else:
        from gwas.io import load_vcf_cached
        parsed = ParsedGenotypes.from_callset(load_vcf_cached(vcf_path.read_bytes()))
    sep = "\t" if pheno_path.suffix in (".tsv", ".txt") else ","
    pheno = pd.read_csv(pheno_path, sep=sep, index_col=0, encoding="utf-8-sig").copy()
    f_ids, f_pqc, f_sqc, f_mat = (_pipeline_harmonize_ids, _pipeline_phenotype_qc, _pipeline_snp_qc,
                                  _pipeline_build_geno_matrices)
    if getattr(parsed, "is_store", False):
        from gwas.storepipe import TraitStore
        st = TraitStore(parsed)
        f_ids, f_pqc, f_sqc, f_mat = st.harmonize_ids, st.phenotype_qc, st.snp_qc, st.build_geno_matrices
    geno_df, pheno, chroms, positions, sid, info_scores, allele_map = f_ids(
        parsed.genotypes, parsed.samples, parsed.chroms, parsed.positions, parsed.vid, parsed.ref, parsed.alt,
        pheno, parsed.info_scores)
    geno_df, pheno, y = f_pqc(geno_df, pheno, args.trait, args.norm, args.ind_miss)
    canonical = (tuple(str(i) for i in range(1, args.n_chromosomes + 1)) if args.n_chromosomes else None)
    geno_df, chroms, chroms_num, positions, sid, n_raw, qc_snp, info_scores = f_sqc(
        geno_df, chroms, positions, sid, args.maf, args.miss, args.mac, True, info_scores, args.info_thresh,
        canonical=canonical, max_het=args.max_het, het_excess_p=args.het_excess_p)
    geno_dosage_raw, rate, geno_imputed, iid = f_mat(geno_df, method=args.impute, impute_k=args.impute_k,
                                                     impute_l=args.impute_l)
    return dict(plan_mode=plan.mode, parsed=parsed, store_dir=store_dir, samples=int(len(iid)),
                chroms=chroms, positions=positions, sid=sid, geno_imputed=geno_imputed)


def _canon_rows(df, cols):
    out = []
    for r in df.to_dict("records"):
        row = []
        for c in cols:
            v = r[c]
            if c == "Chr":
                v = annotation.canon_chr(v)
            elif c == "SNP_IDs":
                v = ",".join(snp_list(v))
            elif c in ("lead_snp",):
                v = str(v)
            else:
                v = int(v)
            row.append(v)
        out.append(tuple(row))
    return out


def compare_rows(driver, run, cols):
    a, b = _canon_rows(driver, cols), _canon_rows(run, cols)
    return dict(rows_driver=len(a), rows_run=len(b), same_rows_same_order=a == b,
                same_rows_any_order=sorted(a) == sorted(b),
                only_driver=[list(x) for x in sorted(set(a) - set(b))][:10],
                only_run=[list(x) for x in sorted(set(b) - set(a))][:10])


def cmd_loci(a):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log = logging.getLogger("cli")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    args = cli_default_options(vcf=a.vcf, pheno=a.pheno, trait=a.trait, output=str(out),
                               geno_store_dir=a.geno_store_dir)
    zf = zipfile.ZipFile(a.zip)
    meta = json.loads(zf.read("run_metadata.json"))
    gwas_df = pd.read_csv(io.BytesIO(zf.read(f"tables/GWAS_{a.trait}.csv")))
    run_blocks = pd.read_csv(io.BytesIO(zf.read("tables/LD_blocks_annotated_MLM.csv")))
    iso_name = "tables/Isolated_SNP_intervals_MLM.csv"
    run_iso = (pd.read_csv(io.BytesIO(zf.read(iso_name))) if iso_name in zf.namelist()
               else pd.DataFrame(columns=["Chr", "Start (bp)", "End (bp)", "lead_snp", "SNP_IDs"]))
    times, facts = {}, dict(name=a.name, zip=str(a.zip), gwas_file=gwas.__file__, cli_file=cli.__file__)
    t = time.time()
    inp = cli_inputs(args, log)
    times["inputs_s"] = time.time() - t
    try:
        t = time.time()
        meff, _ = _compute_meff_li_ji_impl(inp["geno_imputed"])
        rule = rule_from_cli_args(args, len(inp["sid"]), meff)
        times["meff_s"] = time.time() - t
        t = time.time()
        decay, window, dinfo = ld_decay_window(inp["geno_imputed"], inp["chroms"], inp["positions"])
        times["ld_decay_s"] = time.time() - t
        t = time.time()
        geno_float = inp["geno_imputed"].astype(float)
        blocks, err = trace_blocks(gwas_df, inp["chroms"], inp["positions"], geno_float, inp["sid"], decay, window,
                                   args, rule)
        times["blocks_s"] = time.time() - t
        t = time.time()
        edge, maxbp = interval_rules(args, window)
        biv = iso.block_candidate_intervals(blocks, inp["chroms"], inp["positions"], inp["sid"],
                                            edge_flank_bp=edge, max_interval_bp=maxbp)
        drv_blocks = blocks.copy()
        if len(drv_blocks):
            drv_blocks["Interval_start"] = biv["Interval_start"].to_numpy()
            drv_blocks["Interval_end"] = biv["Interval_end"].to_numpy()
        drv_iso = unblocked_intervals(gwas_df, blocks, rule, inp["chroms"], inp["positions"], inp["sid"], args, window)
        cand = candidates(gwas_df, blocks, rule, inp["chroms"], inp["positions"], inp["sid"], args, window)
        times["intervals_s"] = time.time() - t
    finally:
        if inp.get("store_dir") is not None:
            from gwas import genostore
            genostore.remove_store(inp["store_dir"], inp["parsed"])
    bcols = ["Chr", "Start (bp)", "End (bp)", "Interval_start", "Interval_end", "lead_snp", "SNP_IDs"]
    icols = ["Chr", "Start (bp)", "End (bp)", "lead_snp", "SNP_IDs"]
    drv_blocks[[c for c in bcols if c in drv_blocks.columns]].to_csv(out / f"loci_{a.name}_driver_blocks.csv", index=False)
    drv_iso[[c for c in icols if c in drv_iso.columns]].to_csv(out / f"loci_{a.name}_driver_unblocked.csv", index=False)
    cand.to_csv(out / f"loci_{a.name}_driver_candidates.csv", index=False)
    facts.update(
        table_mode=inp["plan_mode"], samples=inp["samples"], snps=int(len(inp["sid"])), meff=int(meff),
        rule=dict(rule=rule.rule, p_threshold=rule.p_threshold, n_tests=rule.n_tests),
        ld_decay_kb=decay, ld_window_kb=window, ld_decay_detail=dinfo,
        run_metadata={k: meta.get(k) for k in ("LD decay (kb)", "LD flank (kb)", "LD blocks (MLM)", "Samples",
                                               "SNPs (post-QC)", "Significance", "Genotype table",
                                               "Isolated SNPs (no LD block)")},
        decay_matches_run=(round(decay, 1) == float(meta.get("LD decay (kb)"))) if decay is not None else False,
        window_matches_run=int(window) == int(meta.get("LD flank (kb)")),
        block_error=err, blocks=int(len(blocks)), unblocked_intervals=int(len(drv_iso)),
        candidate_intervals=int(len(cand)),
        blocks_vs_run=compare_rows(drv_blocks, run_blocks, bcols) if len(drv_blocks) or len(run_blocks) else "both empty",
        unblocked_vs_run=compare_rows(drv_iso, run_iso, icols) if len(drv_iso) or len(run_iso) else "both empty",
        times_s=times)
    ok = (facts["decay_matches_run"] and facts["window_matches_run"]
          and isinstance(facts["blocks_vs_run"], dict) and facts["blocks_vs_run"]["same_rows_any_order"]
          and (facts["unblocked_vs_run"] == "both empty" or facts["unblocked_vs_run"]["same_rows_any_order"]))
    facts["driver_equals_cli"] = bool(ok)
    (out / f"loci_{a.name}.json").write_text(json.dumps(facts, indent=1, default=str), encoding="utf-8")
    print(json.dumps(facts, indent=1, default=str))
    if not ok:
        raise SystemExit(f"stop: the driver differs from the CLI run ({a.name}); see loci_{a.name}.json")


# ----------------------------------------------------------------------------------------------
# sim: the published simulation

_S = {}   # one process's inputs, set once by _init


def _init(sim, decay, window, meff, span, group_of):
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass
    logging.getLogger("gwas").setLevel(logging.WARNING)
    geno, _, snp_map = p2.load_template(Path(sim))
    args = cli_default_options()
    _S.update(geno_float=geno.astype(float), chroms=snp_map["Chr"].to_numpy(),
              positions=snp_map["Pos"].astype(int).to_numpy(), sid=snp_map["SNP_ID"].astype(str).to_numpy(),
              args=args, rule=rule_from_cli_args(args, len(snp_map), meff), decay=decay, window=window,
              span=span, group_of=group_of,
              count_genes=p2.gene_counter(annotation.load_gene_model(p2.REPO / "data" / "Sol_genes_SL3.csv")))


def one_replicate(task):
    """One replicate: TRACE's default blocks and candidate intervals, the membership check, each
    causal SNP's capture and each interval's metrics (ld_blocks_vs_plink.py's definitions)."""
    scen, rep, res_path, truth = task
    S = _S
    res = pd.read_csv(res_path).dropna(subset=["PValue"])
    res["Chr"] = res["Chr"].astype(str)
    blocks, err = trace_blocks(res, S["chroms"], S["positions"], S["geno_float"], S["sid"], S["decay"], S["window"],
                               S["args"], S["rule"])
    cand = candidates(res, blocks, S["rule"], S["chroms"], S["positions"], S["sid"], S["args"], S["window"])
    sig = res.loc[S["rule"].significant_mask(res)]
    sig_ids = sig["SNP"].astype(str).tolist()
    sig_pos = dict(zip(sig_ids, zip(sig["Chr"].astype(str), sig["Pos"].astype(int))))
    ivs = []
    for r in cand.itertuples(index=False):
        ivs.append(dict(type=r.Type, chr=str(r.Chr), start=int(r.Interval_start), end=int(r.Interval_end),
                        members=snp_list(r.SNP_IDs), block_start=r.Block_start, block_end=r.Block_end))
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
    return iv_rows, causal_rows, rep_row, check


def summarise(iv, cs, label):
    """ld_blocks_vs_plink.py's summary rows, for TRACE's default intervals."""
    rows = []
    for grp in ["short", "long", "all"]:
        ci = cs[(cs.group == grp) | (grp == "all")]
        ii = iv[(iv.group == grp) | (grp == "all")] if len(iv) else iv
        tt = ii[ii.true_hit] if len(ii) else ii
        rows.append(dict(scenario=label, method=METHOD, ld_group=grp, n_causal=len(ci),
                         n_captured=int(ci.captured.sum()),
                         captured=round(float(ci.captured.mean()), 4) if len(ci) else None,
                         n_intervals=len(ii), n_true_hits=int(ii.true_hit.sum()) if len(ii) else 0,
                         true_hits=round(float(ii.true_hit.mean()), 4) if len(ii) else None,
                         median_width_kb=round(float(ii.width_kb.median()), 1) if len(ii) else None,
                         median_genes=float(ii.n_genes.median()) if len(ii) else None,
                         median_genes_true_hits=float(tt.n_genes.median()) if len(tt) else None))
    return rows


def cmd_sim(a):
    logging.getLogger("gwas").setLevel(logging.WARNING)
    sim, out = Path(a.sim_data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    geno, ids, snp_map = p2.load_template(sim)
    chroms, positions = snp_map["Chr"].to_numpy(), snp_map["Pos"].astype(int).to_numpy()
    meff, _ = _compute_meff_li_ji_impl(geno)
    decay, window, dinfo = ld_decay_window(geno, chroms, positions)
    args = cli_default_options()
    rule = rule_from_cli_args(args, len(snp_map), meff)
    edge, maxbp = interval_rules(args, window)
    reps_all = list(p2.replicates(sim))
    truths = {(s, r): json.loads(t.read_text(encoding="utf-8")) for s, r, _, t in reps_all}
    causal_all = sorted({c for t in truths.values() for c in t["qtn_snp_ids"]})
    span, check = p2.ld_context(geno, snp_map, causal_all)
    five = [span[c] for (s, _), t in truths.items() if s.endswith("_q005") for c in t["qtn_snp_ids"]]
    split = float(np.median(five))
    group_of = {c: ("short" if v <= split else "long") for c, v in span.items()}
    setup_s = time.time() - t0
    reps = [x for x in reps_all if not a.only or f"{x[0]}/{x[1]}" in a.only]
    tasks = [(s, r, str(p), truths[(s, r)]) for s, r, p, _ in reps]
    init = (str(sim), decay, window, meff, span, group_of)
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
    ivd.to_csv(out / "intervals.csv", index=False)
    cd.to_csv(out / "causal_by_method.csv", index=False)
    reps_df.to_csv(out / "replicates.csv", index=False)
    checks.to_csv(out / "membership.csv", index=False)
    per = []
    for scen in sorted(cd.scenario.unique()):
        per += summarise(ivd[ivd.scenario == scen] if len(ivd) else ivd, cd[cd.scenario == scen], scen)
    pooled = summarise(ivd[ivd.scenario.str.endswith("_q005")] if len(ivd) else ivd,
                       cd[cd.scenario.str.endswith("_q005")], "pooled_5qtn")
    pd.DataFrame(per + pooled).to_csv(out / "trace_default_rows.csv", index=False)
    p2_causal = HERE / "sim_summary" / "ld_blocks_vs_plink_causal.csv"
    ctx = pd.DataFrame([dict(snp=c, ld_kb=v, group=group_of[c]) for c, v in sorted(span.items())])
    reasons = {}
    for x in checks.reasons:
        for k, v in json.loads(x).items():
            reasons[k] = reasons.get(k, 0) + v
    facts = dict(
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
        setup_s=setup_s, gwas_file=gwas.__file__, cli_file=cli.__file__, script=str(Path(__file__).resolve()))
    (out / "sim_facts.json").write_text(json.dumps(facts, indent=1, default=str), encoding="utf-8")
    print(json.dumps(facts, indent=1, default=str))
    print(pd.DataFrame(pooled).to_string(index=False))


# ----------------------------------------------------------------------------------------------
# tables

def _lines(path):
    return [x for x in Path(path).read_bytes().split(b"\n") if x.strip()]


def _key(line, header):
    vals = line.decode().rstrip("\r").split(",")
    cols = header.decode().rstrip("\r").split(",")
    return dict(zip(cols, vals))


def cmd_tables(a):
    out = Path(a.out)
    src = Path(a.summary)
    facts = {}
    trace = pd.read_csv(out / "trace_default_rows.csv")
    for kind, suffix in (("main", "_main"), ("per_scenario", "")):
        header = _lines(src / f"ld_blocks_vs_plink{suffix}.csv")[0]
        plink = {}
        copied = 0
        for meth, stem in PLINK_SOURCES.items():
            lines = _lines(src / f"{stem}{suffix}.csv")
            if lines[0] != header:
                raise SystemExit(f"stop: {stem}{suffix}.csv has another header")
            for x in lines[1:]:
                k = _key(x, header)
                if k["method"] == meth:
                    plink[(k["scenario"], meth, k["ld_group"])] = x
        tr = trace if kind == "per_scenario" else trace[trace.scenario == "pooled_5qtn"]
        scenarios = list(dict.fromkeys(tr.scenario))
        buf = io.StringIO()
        tr.to_csv(buf, index=False, header=False, lineterminator="\n")
        tlines = {(r.scenario, r.ld_group): line.encode() for r, line in
                  zip(tr.itertuples(index=False), buf.getvalue().split("\n"))}
        body = []
        missing = []
        for scen in scenarios:
            for meth in METHODS:
                for grp in ("short", "long", "all"):
                    if meth == METHOD:
                        body.append(tlines[(scen, grp)])
                    else:
                        x = plink.get((scen, meth, grp))
                        if x is None:
                            missing.append(f"{scen}/{meth}/{grp}")
                            continue
                        body.append(x.rstrip(b"\r"))
                        copied += 1
        name = "ld_blocks_vs_plink_trace_default" + ("_main" if kind == "main" else "") + ".csv"
        (out / name).write_bytes(b"\r\n".join([header.rstrip(b"\r")] + body) + b"\r\n")
        facts[name] = dict(rows=len(body), plink_rows_copied=copied, plink_rows_missing=missing)
    # acceptance 4: every PLINK line of the new tables against its source table
    for name, suffix in (("ld_blocks_vs_plink_trace_default_main.csv", "_main"),
                         ("ld_blocks_vs_plink_trace_default.csv", "")):
        new = _lines(out / name)
        header = new[0]
        srcs = {stem: set(_lines(src / f"{stem}{suffix}.csv")[1:]) for stem in set(PLINK_SOURCES.values())}
        n, equal = 0, 0
        for x in new[1:]:
            meth = _key(x, header)["method"]
            if meth in PLINK_SOURCES:
                n += 1
                equal += (x + b"\r") in srcs[PLINK_SOURCES[meth]] or x in srcs[PLINK_SOURCES[meth]]
        facts[name]["plink_rows_equal_to_source"] = f"{equal} of {n}"
    facts["loci"] = loci_table(a, out, src)
    (out / "tables_facts.json").write_text(json.dumps(facts, indent=1, default=str), encoding="utf-8")
    print(json.dumps(facts, indent=1, default=str))


def loci_table(a, out, src):
    """The loci rows: TRACE's from each run's own tables, the PLINK rows copied from sim_summary/."""
    rows, facts = [], {}
    loci = [("tomato_locule_chr2", a.tomato_zip, None), ("rice_GS3", a.rice_zip, "GS3"), ("rice_GL7", a.rice_zip, "GL7")]
    gene_models = {"tomato": p2.gene_counter(annotation.load_gene_model(p2.REPO / "data" / "Sol_genes_SL3.csv")),
                   "rice": p2.gene_counter(annotation.load_gene_model(a.rice_gene_model))}
    for locus, zpath, gene in loci:
        meta, g, sig, tblocks, bframe, samples = p2.zip_run(zpath)
        zf = zipfile.ZipFile(zpath)
        iso_name = [n for n in zf.namelist() if n.endswith("Isolated_SNP_intervals_MLM.csv")]
        isot = pd.read_csv(io.BytesIO(zf.read(iso_name[0]))) if iso_name else pd.DataFrame()
        cand = [dict(chr=annotation.canon_chr(r["Chr"]), start=int(r["Interval_start"]), end=int(r["Interval_end"]),
                     members=snp_list(r["SNP_IDs"]), kind="LD block") for r in bframe.to_dict("records")]
        cand += [dict(chr=annotation.canon_chr(r["Chr"]), start=int(r["Start (bp)"]), end=int(r["End (bp)"]),
                      members=snp_list(r["SNP_IDs"]), kind="unblocked SNP") for r in isot.to_dict("records")]
        sig_ids = set(sig["SNP"].astype(str))
        if gene is None:
            ch, lo_, hi_ = "2", None, None
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
            rows.append(dict(locus=locus, method=METHOD, chr=c["chr"], start=c["start"], end=c["end"],
                             width_kb=(c["end"] - c["start"]) / 1000.0, n_snps=len(c["members"]),
                             n_sig=sum(m in sig_ids for m in c["members"]),
                             n_genes=count_genes(c["chr"], c["start"], c["end"]),
                             holds_lead_snp=lead in c["members"],
                             gene_inside=(None if gene_iv is None else
                                          bool(c["chr"] == gene_iv[0] and c["start"] <= gene_iv[2] and c["end"] >= gene_iv[1])),
                             causal_inside=(None if causal is None else
                                            bool(c["chr"] == causal[0] and c["start"] <= causal[1] <= c["end"])),
                             sig_snps_in_two_blocks=multi))
        facts[locus] = dict(lead_snp=lead, candidate_intervals=len(here),
                            holding_lead=sum(lead in c["members"] for c in here))
    header = _lines(src / "ld_blocks_vs_plink_loci.csv")[0]
    cols = header.decode().rstrip("\r").split(",")
    buf = io.StringIO()
    pd.DataFrame(rows, columns=cols).to_csv(buf, index=False, header=False, lineterminator="\r\n")
    body = [x.encode() for x in buf.getvalue().split("\r\n") if x]
    copied, equal = 0, 0
    for meth, stem in PLINK_SOURCES.items():
        lines = _lines(src / f"{stem}_loci.csv")
        if lines[0] != header:
            raise SystemExit(f"stop: {stem}_loci.csv has another header")
        for x in lines[1:]:
            if _key(x, header)["method"] == meth:
                body.append(x.rstrip(b"\r"))
                copied += 1
                equal += 1
    (out / "ld_blocks_vs_plink_trace_default_loci.csv").write_bytes(b"\r\n".join([header.rstrip(b"\r")] + body) + b"\r\n")
    facts["rows_trace_default"] = len(rows)
    facts["plink_rows_copied"] = copied
    return facts


def main(argv=None):
    ap = argparse.ArgumentParser(description="TRACE's 1.2.0 default LD blocks against PLINK")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("loci", "sim", "tables"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, help="the work folder")
        p.add_argument("--code", default=None, help="the TRACE tree to import (default: this repository)")
        if name == "loci":
            p.add_argument("--name", required=True)
            p.add_argument("--vcf", required=True)
            p.add_argument("--pheno", required=True)
            p.add_argument("--trait", required=True)
            p.add_argument("--zip", required=True, help="the run's results ZIP")
            p.add_argument("--geno-store-dir", default=None, help="as the run's --geno-store-dir")
        if name == "sim":
            p.add_argument("--sim-data", default=str(p2.SIM), help="the published simulation folder")
            p.add_argument("--jobs", type=int, default=1)
            p.add_argument("--only", nargs="*", default=None, help="only these SCENARIO/REP (a check)")
        if name == "tables":
            p.add_argument("--summary", default=str(HERE / "sim_summary"), help="the committed PLINK tables")
            p.add_argument("--tomato-zip", required=True)
            p.add_argument("--rice-zip", required=True)
            p.add_argument("--rice-gene-model", required=True)
    a = ap.parse_args(argv)
    print(f"gwas: {gwas.__file__}; cli: {cli.__file__}", flush=True)
    {"loci": cmd_loci, "sim": cmd_sim, "tables": cmd_tables}[a.cmd](a)


if __name__ == "__main__":
    main()
