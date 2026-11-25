"""TRACE's LD blocks against PLINK's haplotype blocks, both from the same significant SNPs.

Three methods: TRACE's graph LD blocks (gwas/ld.py at its defaults, as the published simulation
evaluation called it), PLINK 1.9 --blocks (Gabriel et al. 2002, Haploview's definition) at its
default 200 kb limit, and PLINK with --blocks-max-kb 600. 600 kb is the widest window of TRACE's
block search: each seed's window is the seed plus and minus the 300 kb flank (gwas/ld.py).
Overlapping blocks of different seeds can still merge into a wider block.

    python benchmarks/simulation/ld_blocks_vs_plink.py vcf --vcf varitome_filtered.vcf --out DIR
    python benchmarks/simulation/ld_blocks_vs_plink.py plink --plink PATH_TO_PLINK --out DIR
    python benchmarks/simulation/ld_blocks_vs_plink.py compare --out DIR
    python benchmarks/simulation/ld_blocks_vs_plink.py loci --plink PATH_TO_PLINK --out DIR \\
        --vcf varitome_filtered.vcf --tomato-zip TOMATO.zip --rice-zip RICE.zip \\
        --rice-bfile PREFIX --rice-gene-model locus.gff

Input: the published simulation in benchmarks/simulation/sim_data/ (or --sim-data DIR): the
template (geno_matrix.npy, sample_ids.npy, snp_map.csv, the earlier 165 x 43,974 template of the
simulation section) and, per scenario and replicate, truth.json and platform_loco/results.csv.
No GWAS is rerun.

vcf: the template's samples and SNPs as a VCF of their raw calls (chromosomes 1 to 12), taken
from the Varitome VCF, for PLINK. Where a call is present the template must hold its ALT count,
else the script stops. Where a call is missing, PLINK sees it missing (the template holds 0).
plink: --blocks no-pheno-req at PLINK's defaults, then with --blocks-max-kb 600. PLINK's blocks
do not depend on the phenotype, so each setting runs once on the template.
compare: per replicate, the SNPs with p < 0.05/43,974 (Bonferroni, as Table S2) and each
method's intervals, the LD context of each causal SNP, and the metrics. Writes
DIR/ld_blocks_vs_plink.csv (one row per scenario, method and LD group, counts included),
DIR/ld_blocks_vs_plink_main.csv (the three 5-QTN scenarios pooled), DIR/ld_blocks_vs_plink_causal.csv
(each causal SNP's LD context) and DIR/compare_facts.json.
loci: the worked examples. Tomato locule number on chromosome 2 (the CLI run's blocks, PLINK on
the raw calls of the run's samples and SNPs) and rice GS3 and GL7 (the rice run's blocks, PLINK
on the PLINK files of the same VCF with the run's samples and SNPs). Writes
DIR/ld_blocks_vs_plink_loci.csv.

Definitions (fixed before any result):
- TRACE: gwas.ld.find_ld_clusters_genomewide on the template with r2 0.6, a 300 kb flank, at
  least 2 SNPs per block and no top-N floor, seeded by the Bonferroni SNPs, as the published
  simulation evaluation called it. The worked examples keep the blocks of their CLI runs.
- A significant SNP's interval is the block whose SNP list holds it (TRACE: SNP_IDs; PLINK: the
  SNPS of .blocks.det). A significant SNP in no block is an interval of its own (start = end = its
  position). Significant SNPs in the same block form one interval. A TRACE interval spans its
  block's Start to End (the range of its members), a PLINK interval its block's BP1 to BP2. When
  one significant SNP is in two TRACE blocks, the two join into one interval (counted).
- LD context of a causal SNP: the distance between the farthest SNP upstream and the farthest SNP
  downstream on its chromosome with r2 >= 0.5 to it, in kb. r2 is gwas.ld.pairwise_r2's
  estimator on the template (no missing values there, so its fast path: Pearson r2 over all
  samples, monomorphic SNPs excluded). The SNP itself counts, so the value is 0 when no other SNP
  reaches 0.5. The causal SNPs of the three 5-QTN scenarios (300 replicates x 5) are split at
  their median: short LD (at most the median) and long LD (above it). The 15- and 50-QTN
  scenarios use the same split value.
- Metrics per method and LD group: captured, the share of causal SNPs inside one of the method's
  intervals of their replicate (same chromosome, start <= position <= end); true hits, the share
  of intervals that hold a causal SNP; the median interval width (kb); the median number of genes
  per interval, over all intervals and over true hits only. Genes are the rows of TRACE's default
  tomato gene model (data/Sol_genes_SL3.csv) that overlap the interval. This is positional only:
  the markers are on SL2.5 and the model on SL3.1, as the paper says for tomato.
- An interval's LD group is that of the causal SNP it holds (short when it holds both kinds),
  else that of the nearest causal SNP on its chromosome. An interval on a chromosome without a
  causal SNP counts in the "all" rows only.
"""
import argparse
import gzip
import json
import logging
import subprocess
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

import annotation
import gwas
from gwas.ld import find_ld_clusters_genomewide, pairwise_r2

REPO = Path(gwas.__file__).resolve().parents[1]
SIM = REPO / "benchmarks" / "simulation" / "sim_data"
FLANK_KB, LD_R2, MIN_SNPS = 300, 0.6, 2
PLINK_SETTINGS = {"plink_200kb": [], "plink_600kb": ["--blocks-max-kb", "600"]}
METHODS = ["trace", "plink_200kb", "plink_600kb"]
CONTEXT_R2 = 0.5
GT = {"0/0": 0, "0/1": 1, "1/0": 1, "1/1": 2, "0|0": 0, "0|1": 1, "1|0": 1, "1|1": 2}
# rice genes of the worked examples (RAP-DB IRGSP-1.0 locus.gff) and GS3's causal site
RICE_GENES = {"GS3": ("Os03g0407400", "3", 16_729_501, 16_735_109),
              "GL7": ("Os07g0603300", "7", 24_664_328, 24_669_321)}
RICE_WINDOW = 500_000
GS3_CAUSAL = ("3", 16_733_441)


# ----------------------------------------------------------------------------------------------
# inputs

def load_template(sim):
    geno = np.load(sim / "geno_matrix.npy")
    ids = [str(s) for s in np.load(sim / "sample_ids.npy", allow_pickle=True)]
    snp_map = pd.read_csv(sim / "snp_map.csv")
    snp_map["Chr"] = snp_map["Chr"].astype(str)
    return geno, ids, snp_map


def replicates(sim, max_reps=None):
    for scen in sorted(p for p in sim.iterdir() if p.is_dir() and p.name.startswith("h2_")):
        reps = sorted(p for p in scen.iterdir() if p.is_dir() and p.name.startswith("rep_"))
        for rep in reps[:max_reps]:
            res = rep / "platform_loco" / "results.csv"
            if res.is_file() and (rep / "truth.json").is_file():
                yield scen.name, rep.name, res, rep / "truth.json"


def read_vcf_calls(path, samples, snp_ids):
    """ALT counts (NaN where missing) of the given samples and SNP IDs, in the given orders, with
    each record's CHROM, POS, REF and ALT."""
    want = {s: j for j, s in enumerate(snp_ids)}
    calls = np.full((len(samples), len(snp_ids)), np.nan, dtype=np.float32)
    meta = [None] * len(snp_ids)
    other = set()
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        cols = None
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                header = line.rstrip("\n").split("\t")
                missing = [s for s in samples if s not in header[9:]]
                if missing:
                    raise SystemExit(f"{len(missing)} samples not in {path}: {missing[:5]}")
                cols = [header.index(s) for s in samples]
                continue
            p = line.rstrip("\n").split("\t")
            j = want.get(p[2])
            if j is None:
                continue
            meta[j] = (p[0], int(p[1]), p[3], p[4])
            for i, c in enumerate(cols):
                g = p[c].split(":")[0]
                v = GT.get(g)
                if v is None:
                    other.add(g)
                else:
                    calls[i, j] = v
    absent = [snp_ids[j] for j, m in enumerate(meta) if m is None]
    if absent:
        raise SystemExit(f"{len(absent)} SNPs not in {path}: {absent[:5]}")
    unknown = other - {"./.", ".|.", "."}
    if unknown:
        raise SystemExit(f"genotype codes other than biallelic calls: {sorted(unknown)[:10]}")
    return calls, meta


def write_vcf(path, samples, chroms, positions, ids, refs, alts, calls):
    code = {0: "0/0", 1: "0/1", 2: "1/1"}
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("##fileformat=VCFv4.2\n")
        for c in sorted(set(chroms), key=lambda x: int(x)):
            f.write(f"##contig=<ID={c}>\n")
        f.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples) + "\n")
        for j in range(len(ids)):
            gts = ["./." if np.isnan(v) else code[int(v)] for v in calls[:, j]]
            f.write(f"{chroms[j]}\t{positions[j]}\t{ids[j]}\t{refs[j]}\t{alts[j]}\t.\t.\t.\tGT\t"
                    + "\t".join(gts) + "\n")


def run_plink(plink, inputs, out_prefix, extra):
    """PLINK 1.9 --blocks (1,024 MB of workspace, which changes no result)."""
    cmd = ([str(plink)] + inputs + ["--chr-set", "12", "--memory", "1024", "--blocks", "no-pheno-req"]
           + extra + ["--out", str(out_prefix)])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"plink failed ({r.returncode}): {' '.join(cmd)}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return cmd


def read_plink_blocks(det_path):
    """PLINK's .blocks.det: one row per block (CHR BP1 BP2 KB NSNPS SNPS)."""
    det = pd.read_csv(det_path, sep=r"\s+")
    det["CHR"] = det["CHR"].astype(str)
    blocks = []
    for b, r in enumerate(det.itertuples(index=False)):
        blocks.append(dict(id=b, chr=r.CHR, start=int(r.BP1), end=int(r.BP2), members=r.SNPS.split("|")))
    return blocks


def trace_blocks_from_frame(df):
    """TRACE's blocks (Chr, Start (bp), End (bp), SNP_IDs as a list or a comma separated text)."""
    blocks = []
    for b, r in enumerate(df.to_dict("records")):
        ids = r["SNP_IDs"]
        members = list(ids) if isinstance(ids, (list, tuple, np.ndarray)) else str(ids).split(",")
        blocks.append(dict(id=b, chr=str(r["Chr"]), start=int(r["Start (bp)"]), end=int(r["End (bp)"]),
                           members=[str(m).strip() for m in members if str(m).strip()]))
    return blocks


def gene_counter(gene_model):
    genes = gene_model[["Chr", "Start", "End"]].copy()
    genes["Chr"] = genes["Chr"].astype(str)
    by_chr = {c: (g["Start"].to_numpy(), g["End"].to_numpy()) for c, g in genes.groupby("Chr")}

    def count(ch, start, end):
        if ch not in by_chr:
            return 0
        s, e = by_chr[ch]
        return int(((s <= end) & (e >= start)).sum())
    return count


# ----------------------------------------------------------------------------------------------
# intervals

def intervals(sig, blocks):
    """The intervals of the significant SNPs ``sig`` (SNP, Chr, Pos) under ``blocks``: SNPs in the
    same block form one interval; a SNP in no block is its own interval; a SNP in two blocks joins
    them. Returns a list of dicts and the count of SNPs found in more than one block."""
    holder = {}
    for b in blocks:
        for m in b["members"]:
            holder.setdefault(m, []).append(b["id"])
    by_id = {b["id"]: b for b in blocks}
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    multi = 0
    for r in sig.itertuples(index=False):
        key = ("snp", r.SNP)
        parent.setdefault(key, key)
        hs = holder.get(r.SNP, [])
        multi += len(hs) > 1
        for h in hs:
            parent.setdefault(("block", h), ("block", h))
            union(key, ("block", h))
    groups = {}
    for node in list(parent):
        groups.setdefault(find(node), []).append(node)
    out = []
    pos = {r.SNP: (str(r.Chr), int(r.Pos)) for r in sig.itertuples(index=False)}
    for nodes in groups.values():
        snps = [n[1] for n in nodes if n[0] == "snp"]
        bids = [n[1] for n in nodes if n[0] == "block"]
        if bids:
            bl = [by_id[i] for i in bids]
            members = set().union(*[set(b["members"]) for b in bl])
            out.append(dict(chr=bl[0]["chr"], start=min(b["start"] for b in bl), end=max(b["end"] for b in bl),
                            n_snps=len(members), n_sig=len(snps), n_blocks=len(bl), sig_snps=sorted(snps)))
        else:
            ch, p = pos[snps[0]]
            out.append(dict(chr=ch, start=p, end=p, n_snps=1, n_sig=1, n_blocks=0, sig_snps=snps))
    return sorted(out, key=lambda d: (int(d["chr"]) if d["chr"].isdigit() else 99, d["start"])), multi


# ----------------------------------------------------------------------------------------------
# LD context

def r2_rows(geno_chr, ks):
    """r2 of each column k in ``ks`` with every column of ``geno_chr``: gwas.ld.pairwise_r2's fast
    path (no missing values): Pearson r2 over all samples, NaN for monomorphic columns, 1 on the
    diagonal."""
    G = np.asarray(geno_chr, float)
    G0 = G - G.mean(axis=0)
    std = G.std(axis=0)
    valid = std > 1e-8
    Z = np.zeros_like(G0)
    Z[:, valid] = G0[:, valid] / std[valid]
    n = G.shape[0]
    out = {}
    for k in ks:
        r2 = np.full(G.shape[1], np.nan)
        if valid[k]:
            r = Z.T @ Z[:, k] / n
            r2 = np.clip(r, -1.0, 1.0) ** 2
            r2[~valid] = np.nan
        r2[k] = 1.0
        out[k] = r2
    return out


def ld_context(geno, snp_map, causal_ids, check_n=25, seed=1):
    """kb between the farthest SNPs with r2 >= 0.5 to each causal SNP (itself included), and the
    largest difference against gwas.ld.pairwise_r2 on windows of at most 2,000 SNPs."""
    idx = {s: j for j, s in enumerate(snp_map["SNP_ID"])}
    chroms = snp_map["Chr"].to_numpy()
    pos = snp_map["Pos"].to_numpy()
    span, rows = {}, {}
    for ch in sorted(set(snp_map.loc[[idx[s] for s in causal_ids], "Chr"])):
        cols = np.flatnonzero(chroms == ch)
        local = {idx[s]: int(np.searchsorted(cols, idx[s])) for s in causal_ids if chroms[idx[s]] == ch}
        r2 = r2_rows(geno[:, cols], list(local.values()))
        for j, k in local.items():
            sel = np.flatnonzero(np.nan_to_num(r2[k], nan=0.0) >= CONTEXT_R2)
            p = pos[cols][sel]
            span[snp_map["SNP_ID"][j]] = float(p.max() - p.min()) / 1000.0
            rows[snp_map["SNP_ID"][j]] = (cols, r2[k])
    rng = np.random.default_rng(seed)
    pick = rng.choice(sorted(rows), size=min(check_n, len(rows)), replace=False)
    worst, nan_agree = 0.0, True
    for s in pick:
        cols, mine = rows[s]
        k = int(np.searchsorted(cols, idx[s]))
        lo = max(0, min(k - 999, len(cols) - 1999))
        win = slice(lo, min(len(cols), lo + 1999))
        ref = pairwise_r2(geno[:, cols[win]])[k - lo]
        a, b = mine[win], ref
        nan_agree &= bool(np.array_equal(np.isnan(a), np.isnan(b)))
        both = ~np.isnan(a) & ~np.isnan(b)
        if both.any():
            worst = max(worst, float(np.max(np.abs(a[both] - b[both]))))
    return span, dict(checked_snps=len(pick), max_abs_diff_vs_pairwise_r2=worst, nan_pattern_equal=nan_agree)


# ----------------------------------------------------------------------------------------------
# commands

def cmd_vcf(a):
    sim, out = Path(a.sim_data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    geno, ids, snp_map = load_template(sim)
    calls, meta = read_vcf_calls(a.vcf, ids, list(snp_map["SNP_ID"]))
    bad_pos = [s for s, m, p in zip(snp_map["SNP_ID"], meta, snp_map["Pos"]) if m[1] != int(p)]
    bad_allele = [s for s, m, r, al in zip(snp_map["SNP_ID"], meta, snp_map["Ref"], snp_map["Alt"])
                  if (m[2], m[3]) != (r, al)]
    present = ~np.isnan(calls)
    differ = int((geno[present] != calls[present]).sum())
    facts = dict(samples=len(ids), snps=len(snp_map), calls=int(calls.size), present=int(present.sum()),
                 missing=int((~present).sum()), present_equal_to_template=int(present.sum()) - differ,
                 present_different=differ, position_mismatches=len(bad_pos), allele_mismatches=len(bad_allele),
                 template_values_at_missing_calls={str(k): int(v) for k, v in
                                                   zip(*np.unique(geno[~present], return_counts=True))})
    (out / "template_check.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")
    print(json.dumps(facts, indent=1))
    if differ or bad_pos or bad_allele:
        raise SystemExit("stop: the template and the raw calls differ (template_check.json)")
    write_vcf(out / "template_raw.vcf", ids, list(snp_map["Chr"]), list(snp_map["Pos"]),
              list(snp_map["SNP_ID"]), list(snp_map["Ref"]), list(snp_map["Alt"]), calls)
    print(f"wrote {out / 'template_raw.vcf'}")


def cmd_plink(a):
    out = Path(a.out)
    vcf = out / "template_raw.vcf"
    cmds = {}
    for name, extra in PLINK_SETTINGS.items():
        cmds[name] = run_plink(a.plink, ["--vcf", str(vcf), "--double-id", "--vcf-half-call", "missing"],
                               out / name, extra)
    (out / "plink_commands.json").write_text(json.dumps(cmds, indent=1), encoding="utf-8")
    for name in PLINK_SETTINGS:
        print(name, len(read_plink_blocks(out / f"{name}.blocks.det")), "blocks")


_W = {}   # one process's shared inputs, set once by _init_worker


def _init_worker(sim, out, bonf, span, group_of):
    """Load the template, PLINK's blocks and the gene model once per process, with one BLAS thread
    (so the result does not depend on --jobs)."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass
    logging.getLogger("gwas").setLevel(logging.WARNING)
    geno, _, snp_map = load_template(Path(sim))
    _W.update(geno=geno, chroms=snp_map["Chr"].to_numpy(), positions=snp_map["Pos"].astype(int).to_numpy(),
              sid=snp_map["SNP_ID"].to_numpy(), bonf=bonf, span=span, group_of=group_of,
              plink={name: read_plink_blocks(Path(out) / f"{name}.blocks.det") for name in PLINK_SETTINGS},
              count_genes=gene_counter(annotation.load_gene_model(REPO / "data" / "Sol_genes_SL3.csv")))


def one_replicate(task):
    """One replicate: its Bonferroni SNPs, TRACE's blocks, each method's intervals, the causal SNPs
    captured. Returns the rows and the replicate's facts."""
    scen, rep, res_path, truth = task
    W = _W
    res = pd.read_csv(res_path).dropna(subset=["PValue"])
    res["Chr"] = res["Chr"].astype(str)
    sig = res[res["PValue"] < W["bonf"]][["SNP", "Chr", "Pos", "PValue"]]
    causal = list(zip(truth["qtn_snp_ids"], [str(c) for c in truth["qtn_chrs"]], truth["qtn_positions"]))
    tb = find_ld_clusters_genomewide(gwas_df=res, chroms=W["chroms"], positions=W["positions"],
                                     geno_imputed=W["geno"], sid=W["sid"], ld_threshold=LD_R2, flank_kb=FLANK_KB,
                                     sig_thresh=W["bonf"], top_n=0, min_snps=MIN_SNPS)
    tblocks = trace_blocks_from_frame(tb) if len(tb) else []
    widest = max(((b["end"] - b["start"]) / 1000.0, f"{scen}/{rep} chr{b['chr']}:{b['start']}-{b['end']}")
                 for b in tblocks) if tblocks else (0.0, None)
    iv_rows, causal_rows, multi_trace = [], [], 0
    group_of, span = W["group_of"], W["span"]
    for meth, blocks in {"trace": tblocks, **W["plink"]}.items():
        ivs, multi = intervals(sig, blocks) if len(sig) else ([], 0)
        if meth == "trace":
            multi_trace = multi
        for c, ch, p in causal:
            inside = any(iv["chr"] == ch and iv["start"] <= p <= iv["end"] for iv in ivs)
            causal_rows.append(dict(scenario=scen, rep=rep, method=meth, snp=c, chr=ch, pos=p,
                                    ld_kb=span[c], group=group_of[c], captured=inside))
        for iv in ivs:
            held = [c for c, ch, p in causal if ch == iv["chr"] and iv["start"] <= p <= iv["end"]]
            if held:
                grp = "short" if any(group_of[c] == "short" for c in held) else "long"
            else:
                same = [(min(abs(p - iv["start"]), abs(p - iv["end"])), p, c) for c, ch, p in causal if ch == iv["chr"]]
                grp = group_of[sorted(same)[0][2]] if same else "none"
            iv_rows.append(dict(scenario=scen, rep=rep, method=meth, chr=iv["chr"], start=iv["start"],
                                end=iv["end"], width_kb=(iv["end"] - iv["start"]) / 1000.0, n_snps=iv["n_snps"],
                                n_sig=iv["n_sig"], n_blocks=iv["n_blocks"],
                                n_genes=W["count_genes"](iv["chr"], iv["start"], iv["end"]),
                                true_hit=bool(held), group=grp))
    rep_row = dict(scenario=scen, rep=rep, n_sig=len(sig), n_trace_blocks=len(tblocks))
    return iv_rows, causal_rows, rep_row, multi_trace, widest


def cmd_compare(a):
    logging.getLogger("gwas").setLevel(logging.WARNING)
    sim, out = Path(a.sim_data), Path(a.out)
    geno, ids, snp_map = load_template(sim)
    bonf = 0.05 / len(snp_map)
    plink = {name: read_plink_blocks(out / f"{name}.blocks.det") for name in PLINK_SETTINGS}
    reps = list(replicates(sim, a.max_reps))
    truths = {(s, r): json.loads(t.read_text(encoding="utf-8")) for s, r, _, t in reps}
    causal_all = sorted({c for t in truths.values() for c in t["qtn_snp_ids"]})
    span, check = ld_context(geno, snp_map, causal_all)
    five = [span[c] for (s, _), t in truths.items() if s.endswith("_q005") for c in t["qtn_snp_ids"]]
    split = float(np.median(five))
    group_of = {c: ("short" if v <= split else "long") for c, v in span.items()}
    tasks = [(s, r, str(p), truths[(s, r)]) for s, r, p, _ in reps]
    init = (str(sim), str(out), bonf, span, group_of)
    if a.jobs > 1:
        import multiprocessing
        with multiprocessing.get_context("spawn").Pool(a.jobs, initializer=_init_worker, initargs=init) as pool:
            results = []
            for k, r in enumerate(pool.imap(one_replicate, tasks, chunksize=1)):
                results.append(r)
                if (k + 1) % 100 == 0:
                    print(f"{k + 1}/{len(tasks)} replicates", flush=True)
    else:
        _init_worker(*init)
        results = []
        for k, t in enumerate(tasks):
            results.append(one_replicate(t))
            if (k + 1) % 100 == 0:
                print(f"{k + 1}/{len(tasks)} replicates", flush=True)
    iv_rows = [row for r in results for row in r[0]]
    causal_rows = [row for r in results for row in r[1]]
    rep_rows = [r[2] for r in results]
    multi_total = sum(r[3] for r in results)
    widest_trace = max((r[4] for r in results), key=lambda w: w[0])
    ivd, cd = pd.DataFrame(iv_rows), pd.DataFrame(causal_rows)
    ivd.to_csv(out / "intervals.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(out / "replicates.csv", index=False)
    cd.to_csv(out / "causal_by_method.csv", index=False)
    ctx = pd.DataFrame([dict(snp=c, ld_kb=v, group=group_of[c]) for c, v in sorted(span.items())])
    ctx.to_csv(out / "ld_blocks_vs_plink_causal.csv", index=False)

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
    per = []
    for scen in sorted(cd.scenario.unique()):
        per += summarise(ivd[ivd.scenario == scen] if len(ivd) else ivd, cd[cd.scenario == scen], scen)
    pooled = summarise(ivd[ivd.scenario.str.endswith("_q005")] if len(ivd) else ivd,
                       cd[cd.scenario.str.endswith("_q005")], "pooled_5qtn")
    pd.DataFrame(per + pooled).to_csv(out / "ld_blocks_vs_plink.csv", index=False)
    pd.DataFrame(pooled).to_csv(out / "ld_blocks_vs_plink_main.csv", index=False)
    facts = dict(template=f"{geno.shape[0]} x {geno.shape[1]}", bonferroni=bonf, replicates=len(reps),
                 causal_unique=len(causal_all), ld_split_kb=split, ld_split_from=len(five),
                 ld_check=check, trace_widest_block_kb=widest_trace[0], trace_widest_block=widest_trace[1],
                 sig_snps_in_two_trace_blocks=multi_total,
                 plink_blocks={k: len(v) for k, v in plink.items()},
                 plink_widest_kb={k: max((b["end"] - b["start"]) / 1000.0 for b in v) for k, v in plink.items()},
                 gwas_file=gwas.__file__)
    (out / "compare_facts.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")
    print(json.dumps(facts, indent=1))
    print(pd.DataFrame(pooled).to_string(index=False))


def zip_table(zf, suffix):
    names = [n for n in zf.namelist() if n.endswith(suffix)]
    if len(names) != 1:
        raise SystemExit(f"{suffix}: {len(names)} members")
    return pd.read_csv(BytesIO(zf.read(names[0])))


def zip_run(path):
    zf = zipfile.ZipFile(path)
    meta = json.loads(zf.read([n for n in zf.namelist() if n.endswith("run_metadata.json")][0]))
    g = zip_table(zf, f"tables/GWAS_{meta['Trait']}.csv")
    g["Chr"] = g["Chr"].astype(str)
    sig = zip_table(zf, "tables/Significant_SNPs_MLM.csv")
    sig["Chr"] = sig["Chr"].astype(str)
    blocks = zip_table(zf, "tables/LD_blocks_annotated_MLM.csv")
    blocks["Chr"] = blocks["Chr"].astype(str)
    samples = zip_table(zf, "tables/QC_sample_heterozygosity.csv")["sample"].astype(str).tolist()
    return meta, g, sig, trace_blocks_from_frame(blocks) if len(blocks) else [], blocks, samples


def locus_rows(locus, sig, methods, lead_snp, count_genes, gene=None, causal=None):
    rows = []
    for meth, blocks in methods.items():
        ivs, multi = intervals(sig, blocks)
        for iv in ivs:
            rows.append(dict(locus=locus, method=meth, chr=iv["chr"], start=iv["start"], end=iv["end"],
                             width_kb=(iv["end"] - iv["start"]) / 1000.0, n_snps=iv["n_snps"], n_sig=iv["n_sig"],
                             n_genes=count_genes(iv["chr"], iv["start"], iv["end"]),
                             holds_lead_snp=lead_snp in iv["sig_snps"],
                             gene_inside=(None if gene is None else
                                          bool(iv["chr"] == gene[0] and iv["start"] <= gene[2] and iv["end"] >= gene[1])),
                             causal_inside=(None if causal is None else
                                            bool(iv["chr"] == causal[0] and iv["start"] <= causal[1] <= iv["end"])),
                             sig_snps_in_two_blocks=multi))
    return rows


def cmd_loci(a):
    logging.getLogger("gwas").setLevel(logging.WARNING)
    out = Path(a.out) / "loci"
    out.mkdir(parents=True, exist_ok=True)
    rows, facts = [], {}
    # tomato: chromosome 2 of the locule number run
    meta, g, sig, tblocks, bframe, samples = zip_run(a.tomato_zip)
    g2 = g[g["Chr"] == "2"].sort_values("Pos")
    calls, vmeta = read_vcf_calls(a.vcf, samples, list(g2["SNP"]))
    write_vcf(out / "tomato_chr2.vcf", samples, ["2"] * len(g2), list(g2["Pos"]), list(g2["SNP"]),
              [m[2] for m in vmeta], [m[3] for m in vmeta], calls)
    plink = {}
    for name, extra in PLINK_SETTINGS.items():
        run_plink(a.plink, ["--vcf", str(out / "tomato_chr2.vcf"), "--double-id", "--vcf-half-call", "missing"],
                  out / f"tomato_{name}", extra)
        plink[name] = read_plink_blocks(out / f"tomato_{name}.blocks.det")
    sig2 = sig[sig["Chr"] == "2"][["SNP", "Chr", "Pos", "PValue"]]
    tb2 = [b for b in tblocks if b["chr"] == "2"]
    b2 = bframe[bframe["Chr"] == "2"]
    # the lead block of Table S8: the block whose lead SNP has the smallest p; its lead SNP
    lead_row = b2.loc[b2["lead_snp_pvalue"].astype(float).idxmin()] if len(b2) else None
    lead = None if lead_row is None else str(lead_row["lead_snp"])
    tomato_genes = gene_counter(annotation.load_gene_model(REPO / "data" / "Sol_genes_SL3.csv"))
    rows += locus_rows("tomato_locule_chr2", sig2, {"trace": tb2, **plink}, lead, tomato_genes)
    facts["tomato"] = dict(zip=str(a.tomato_zip), samples=len(samples), chr2_snps=len(g2),
                           chr2_missing_calls=int(np.isnan(calls).sum()), significant_chr2=len(sig2),
                           lead_block=None if lead_row is None else
                           f"{int(lead_row['Start (bp)'])}-{int(lead_row['End (bp)'])}",
                           lead_snp=lead, trace_blocks_chr2=len(tb2))
    # rice: GS3 (chromosome 3) and GL7 (chromosome 7) of the grain length run
    meta, g, sig, tblocks, bframe, samples = zip_run(a.rice_zip)
    fam = pd.read_csv(f"{a.rice_bfile}.fam", sep=r"\s+", header=None, dtype=str)
    keep = fam[fam[1].isin(samples)]
    if len(keep) != len(samples):
        raise SystemExit(f"rice: {len(samples)} run samples, {len(keep)} found in the .fam")
    keep[[0, 1]].to_csv(out / "rice_keep.txt", sep=" ", header=False, index=False)
    g37 = g[g["Chr"].isin(["3", "7"])]
    g37[["SNP"]].to_csv(out / "rice_extract.txt", header=False, index=False)
    for name, extra in PLINK_SETTINGS.items():
        run_plink(a.plink, ["--bfile", a.rice_bfile, "--keep", str(out / "rice_keep.txt"),
                            "--extract", str(out / "rice_extract.txt"), "--chr", "3,7"],
                  out / f"rice_{name}", extra)
        plink[name] = read_plink_blocks(out / f"rice_{name}.blocks.det")
    rice_genes = gene_counter(annotation.load_gene_model(a.rice_gene_model))
    facts["rice"] = dict(zip=str(a.rice_zip), samples=len(samples), snps_chr3_chr7=len(g37))
    for name, (gid, ch, s, e) in RICE_GENES.items():
        w = g[(g["Chr"] == ch) & (g["Pos"] >= s - RICE_WINDOW) & (g["Pos"] <= e + RICE_WINDOW)]
        sw = sig[(sig["Chr"] == ch) & (sig["Pos"] >= s - RICE_WINDOW) & (sig["Pos"] <= e + RICE_WINDOW)]
        lead = w.sort_values(["PValue", "Pos"]).iloc[0]["SNP"] if len(w) else None
        tbw = [b for b in tblocks if b["chr"] == ch]
        rows += locus_rows(f"rice_{name}", sw[["SNP", "Chr", "Pos", "PValue"]],
                           {"trace": tbw, **{k: [b for b in v if b["chr"] == ch] for k, v in plink.items()}},
                           lead, rice_genes, gene=(ch, s, e), causal=GS3_CAUSAL if name == "GS3" else None)
        facts["rice"][name] = dict(window_snps=len(w), window_significant=len(sw), lead_snp=lead)
    for loc in ("tomato", "rice"):   # PLINK's logs, without the computer's name and folder
        for name in PLINK_SETTINGS:
            lines = (out / f"{loc}_{name}.log").read_text(encoding="utf-8", errors="replace").splitlines()
            facts[f"plink_log_{loc}_{name}"] = [x for x in lines
                                                if not x.startswith(("Hostname:", "Working directory:"))]
    pd.DataFrame(rows).to_csv(Path(a.out) / "ld_blocks_vs_plink_loci.csv", index=False)
    facts["gwas_file"] = gwas.__file__
    (out / "loci_facts.json").write_text(json.dumps(facts, indent=1, default=str), encoding="utf-8")
    print(pd.DataFrame(rows).drop(columns=["sig_snps_in_two_blocks"]).to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description="TRACE's LD blocks against PLINK's haplotype blocks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("vcf", "plink", "compare", "loci"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, help="the work folder (VCF, PLINK files, results)")
        p.add_argument("--sim-data", default=str(SIM), help="the published simulation folder")
        if name == "compare":
            p.add_argument("--max-reps", type=int, default=None,
                           help="only the first N replicates of each scenario (a quick check)")
            p.add_argument("--jobs", type=int, default=1,
                           help="replicates run side by side in this many processes (the result is the same)")
        if name in ("vcf", "loci"):
            p.add_argument("--vcf", required=True, help="the Varitome VCF (raw calls)")
        if name in ("plink", "loci"):
            p.add_argument("--plink", required=True, help="the PLINK 1.9 executable")
        if name == "loci":
            p.add_argument("--tomato-zip", required=True, help="the CLI results ZIP of tomato locule number")
            p.add_argument("--rice-zip", required=True, help="the CLI results ZIP of rice grain length")
            p.add_argument("--rice-bfile", required=True, help="PLINK files (.bed/.bim/.fam) of the rice VCF")
            p.add_argument("--rice-gene-model", required=True, help="the rice gene model of the run (locus.gff)")
    a = ap.parse_args(argv)
    print(f"gwas: {gwas.__file__}", flush=True)
    {"vcf": cmd_vcf, "plink": cmd_plink, "compare": cmd_compare, "loci": cmd_loci}[a.cmd](a)


if __name__ == "__main__":
    main()
