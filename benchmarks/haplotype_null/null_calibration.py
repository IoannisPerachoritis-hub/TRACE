"""The null calibration of the haplotype tests: how often each test rejects a true null.

One process: the data, the band rule, the blocks, the null phenotypes, the tests, the report.

    python benchmarks/haplotype_null/null_calibration.py --dataset tomato --vcf VCF --pheno CSV --out DIR
    python benchmarks/haplotype_null/null_calibration.py --dataset rice --vcf VCF --pheno CSV --out DIR --band-k 0
    python benchmarks/haplotype_null/null_calibration.py --dataset example --out DIR   (a self test, minutes)

Options: --n-pheno 200, --h2 0 0.3 0.6, --n-perm 1000, --seed 20261117 (the defaults of the
published run), --n-control, --band-k K (skip the band rule's scan), --skip-checks.
Inputs: tomato, the Varitome VCF and its phenotype file (locule_number; the table in memory);
rice, the 3K Rice Genomes Core SNPs of the 2,260 matched accessions and GRLT (the table on disk,
its work folder under DIR); example, the repository's tests/data/example_v1.

The data, the pipeline's way: its front stages with the CLI's default options (the same
functions in the same order), the IDs, phenotype QC, SNP QC, the imputed table, the kinship
matrix, the 20 PCs and the LOCO kernels.

The PC arm, once per dataset on the real trait: the strict band rule behind the paper's PC counts
(smallest k with 0.95 <= lambda_GC <= 1.05; commit 03095b7): LOCO MLM scans at k = 0..10, the
untrimmed lambda_GC rounded to 4 decimals; the smallest k in the band, else the smallest k within
max(0.02, 0.15 x the best |lambda - 1|) of the best; then the deflation guard (lambda at k = 0
below 0.95 gives k = 0). The scan goes to band_rule.csv, with the later edge-tolerant pick of
benchmarks/_pc_band_select.py beside it, labelled.

Blocks: the four Table S8 blocks (tomato: tests/golden/tomato_locule/ld_blocks.csv, their members)
and control blocks, runs of 2 to 6 contiguous SNPs at random positions, the i-th on chromosome
i mod 12, drawn with numpy default_rng(--seed). Each is grouped by TRACE's own function
(run_haplotype_block_gwas, min_hap_count 5, min_group_size 3) and kept only with at least two
tested groups (blocks.csv).

Null phenotypes: for each h2 and j = 0..n-1, chromosome j mod 12 is held out; 200 causal SNPs are
drawn from the other chromosomes with N(0, 1) effects on the standardised genotypes, the genetic
value is scaled to variance h2, and y = g + e with e ~ N(0, 1 - h2). Phenotype j is tested only
on its held-out chromosome's blocks, so every test is a true null. The seeds and the causal SNPs
are in phenotypes.csv.

Tests on every (block, phenotype), 1,000 permutations each with the pipeline's seed: kin, the
kinship-aware test (k = 0, the block's chromosome's LOCO kernel); pcs0, the PCs-only test at k = 0;
pcsband, the PCs-only test at the band rule's k (the same as pcs0 when k = 0, not run twice);
rice also pcs4 (4 PCs). pvalues.csv holds one row per test; a kin test the engine refuses keeps
its reason and no p-value.

Report: rates.csv, the rates at alpha 0.05 and 0.01 for P_perm and PValue_param per h2 and test,
with the test counts and Clopper-Pearson 95% intervals (they treat the tests as independent;
tests that share a phenotype are not); stop_rule.json, the check that no 95% interval of kin lies
entirely above alpha; QQ plots (PNG and PDF); meta.json, the run's facts, seeds and timings.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

TREE = Path(__file__).resolve().parents[2]   # the repository (gwas, tests/golden, tests/data)
sys.path.insert(0, str(TREE))
ALPHAS = (0.05, 0.01)
N_CAUSAL = 200
MAX_PCS = 10


def datasets():
    ex = TREE / "tests" / "data" / "example_v1"
    return {
        "tomato": dict(vcf=None, pheno=None, trait="locule_number",
                       store=False, n_control=46, table_s8=True, expect=(164, 43749), pcs_fixed=[]),
        "rice": dict(vcf=None, pheno=None, trait="GRLT",
                     store=True, n_control=50, table_s8=False, expect=(2260, 160932), pcs_fixed=[4]),
        "example": dict(vcf=ex / "example.vcf.gz", pheno=ex / "example_pheno.csv", trait="Trait1",
                        store=False, n_control=6, table_s8=False, expect=None, pcs_fixed=[2]),
        "example_disk": dict(vcf=ex / "example.vcf.gz", pheno=ex / "example_pheno.csv",
                             trait="Trait1", store=True, n_control=6, table_s8=False, expect=None,
                             pcs_fixed=[2]),
    }


# ------------------------------------------------------------------------------- the data
def front_stages(ds, work):
    """The pipeline's front stages (gwas.pipeline._run_trait, S05-S12) with the CLI's defaults:
    a dict of the table, its labels, the trait's y, the kernels and the PCs."""
    from gwas import qc as _qc
    from gwas.kinship import _build_loco_kernels_impl, _compute_pcs_full_impl
    from gwas.pipeline import cli_default_options

    a = cli_default_options()
    if ds["store"]:
        from gwas.genostore import ingest_vcf, new_store_dir
        from gwas.storepipe import TraitStore

        store_dir = new_store_dir(work)
        parsed = ingest_vcf(Path(ds["vcf"]), store_dir)
        st = TraitStore(parsed)
        H, PQ, SQ, BG, BK = (st.harmonize_ids, st.phenotype_qc, st.snp_qc, st.build_geno_matrices,
                             st.build_kinship)
    else:
        from gwas.io import load_vcf_cached
        from gwas.pipeline import ParsedGenotypes

        store_dir = None
        parsed = ParsedGenotypes.from_callset(load_vcf_cached(Path(ds["vcf"]).read_bytes()))
        H, PQ, SQ, BG, BK = (_qc._pipeline_harmonize_ids, _qc._pipeline_phenotype_qc,
                             _qc._pipeline_snp_qc, _qc._pipeline_build_geno_matrices,
                             _qc._pipeline_build_kinship)
    p = Path(ds["pheno"])
    pheno = pd.read_csv(p, sep="\t" if p.suffix in (".tsv", ".txt") else ",", index_col=0,
                        encoding="utf-8-sig")
    geno_df, pheno, chroms, positions, sid, info_scores, _am = H(
        parsed.genotypes, parsed.samples, parsed.chroms, parsed.positions, parsed.vid, parsed.ref,
        parsed.alt, pheno, parsed.info_scores)
    geno_df, pheno, y = PQ(geno_df, pheno, ds["trait"], a.norm, a.ind_miss)
    canonical = (tuple(str(i) for i in range(1, a.n_chromosomes + 1)) if a.n_chromosomes else None)
    geno_df, chroms, chroms_num, positions, sid, n_raw, qc_snp, info_scores = SQ(
        geno_df, chroms, positions, sid, a.maf, a.miss, a.mac, True, info_scores, a.info_thresh,
        canonical=canonical, max_het=a.max_het, het_excess_p=a.het_excess_p)
    _raw, _rate, geno_imputed, iid = BG(geno_df, method=a.impute, impute_k=a.impute_k,
                                        impute_l=a.impute_l)
    K, Z_for_pca, chroms_grm, _pos_grm, _km = BK(geno_df, geno_imputed, chroms, positions)
    pcs_full, _eig = _compute_pcs_full_impl(Z_for_pca, max_pcs=20)
    K0, K_by_chr, _diag = _build_loco_kernels_impl(iid, Z_for_pca, chroms_grm, K)
    return dict(parsed=parsed, store_dir=store_dir, geno_df=geno_df, pheno=pheno,
                y=np.asarray(y, float).ravel(), chroms=np.asarray(chroms).astype(str),
                chroms_num=chroms_num, positions=np.asarray(positions), sid=np.asarray(sid).astype(str),
                geno_imputed=geno_imputed, iid=iid, K0=K0, K_by_chr=K_by_chr, pcs_full=pcs_full,
                # the table as the pipeline gives it to the haplotype test (_geno_float)
                geno_float=geno_imputed.astype(float),
                n_raw=int(n_raw), qc_snp=dict(qc_snp))


def _columns(d, idx):
    """Columns ``idx`` of the imputed table as a float64 array (n x len(idx))."""
    from gwas.genostore import as_columns

    return np.asarray(as_columns(d["geno_imputed"], float)[:, np.asarray(idx)], dtype=float)


# --------------------------------------------------------------------------- the band rule
def band_rule(d, trait, out):
    """The strict band rule (03095b7) on the real trait: the lambda scan and its pick."""
    from gwas.models import _run_gwas_impl
    from gwas.plotting import compute_lambda_gc
    from gwas.utils import PhenoData

    pheno_reader = PhenoData(iid=d["iid"], val=d["y"])
    rows = []
    for k in range(0, min(MAX_PCS, d["pcs_full"].shape[1]) + 1):
        t0 = time.time()
        g = _run_gwas_impl(d["geno_imputed"], d["y"], d["pcs_full"], k, d["sid"], d["positions"],
                           d["chroms"], d["chroms_num"], d["iid"], d["K0"], d["K_by_chr"],
                           pheno_reader, trait)
        lam = compute_lambda_gc(g["PValue"].values, trim=False)
        rows.append({"n_pcs": k, "lambda_gc": round(lam, 4) if np.isfinite(lam) else np.nan,
                     "lambda_gc_unrounded": lam,
                     "delta_from_1": round(abs(lam - 1.0), 4) if np.isfinite(lam) else np.nan,
                     "seconds": round(time.time() - t0, 1)})
        print(f"  band rule: k={k} lambdaGC={lam:.4f} ({rows[-1]['seconds']} s)", flush=True)
    df = pd.DataFrame(rows)
    k_strict, how = strict_pick(df)
    df["strict_pick"] = df["n_pcs"] == k_strict
    try:
        sys.path.insert(0, str(TREE / "benchmarks"))
        from _pc_band_select import select_best_pc_from_lambdas
        k_edge = int(select_best_pc_from_lambdas(df["lambda_gc"].tolist()))
    except Exception as e:  # noqa: BLE001 - beside the rule, labelled; never the rule
        k_edge, how = None, how + f"; edge-tolerant pick unavailable ({type(e).__name__}: {e})"
    df["edge_tolerant_pick_benchmarks_pc_band_select"] = df["n_pcs"] == k_edge
    df.to_csv(out / "band_rule.csv", index=False)
    return {"k": int(k_strict), "how": how, "k_edge_tolerant": k_edge,
            "lambda_scan": df[["n_pcs", "lambda_gc"]].values.tolist()}


def strict_pick(df, band_lo=0.95, band_hi=1.05, parsimony_tolerance=0.02):
    """03095b7's auto_select_pcs choice (strategy "band") on a lambda scan (rounded lambdas)."""
    valid = df["lambda_gc"].notna()
    if not valid.any():
        return 0, "no finite lambda: k = 0"
    in_band = df[valid & (df["lambda_gc"] >= band_lo) & (df["lambda_gc"] <= band_hi)]
    if len(in_band):
        k, how = int(df.loc[in_band["n_pcs"].idxmin(), "n_pcs"]), "smallest k in [0.95, 1.05]"
    else:
        best = df.loc[valid, "delta_from_1"].min()
        tol = max(parsimony_tolerance, best * 0.15)
        near = df[valid & (df["delta_from_1"] <= best + tol)]
        k = int(df.loc[near["n_pcs"].idxmin(), "n_pcs"])
        how = f"none in band: smallest k within {tol:.4f} of the best |lambda - 1| ({best:.4f})"
    lam0 = df.loc[df["n_pcs"] == 0, "lambda_gc"].iloc[0]
    if pd.notna(lam0) and lam0 < band_lo:
        k, how = 0, f"deflation guard: lambda at k = 0 is {lam0:.4f} < {band_lo} (before it: {how})"
    return k, how


# ------------------------------------------------------------------------------ the blocks
def _group(d, blocks):
    """TRACE's own grouping of ``blocks`` (gwas.haplotype.run_haplotype_block_gwas, the PCs-only
    test with one permutation on a placeholder phenotype: the grouping does not depend on the
    phenotype's values, and the null phenotypes have no missing value): per tested block, its
    tested samples (in the test's order) and their labels."""
    from gwas.haplotype import run_haplotype_block_gwas

    ids = np.asarray(d["geno_df"].index, dtype=str)
    placeholder = pd.DataFrame({"t": np.arange(len(ids), dtype=float)}, index=ids)
    res, tables = run_haplotype_block_gwas(
        haplo_df=blocks, chroms=d["chroms"], positions=d["positions"],
        geno_imputed=d["geno_float"],
        sid=d["sid"], geno_df=pd.DataFrame(index=ids), pheno_df=placeholder, trait_col="t",
        pcs=None, min_hap_count=5, min_group_size=3, n_perm=1, hap_test="pcs")
    out = {}
    for _, r in res.iterrows():
        key = f"{r['Chr']}:{int(r['Start'])}-{int(r['End'])}"
        t = tables[key]
        t = t[t["in_haplotype_test"]]
        out[key] = dict(samples=t["Sample"].astype(str).to_numpy(), groups=t["Haplotype"].astype(str).to_numpy(),
                        n_snps=int(r["n_snps"]), n_groups=int(r["n_tested_haplotypes"]))
    return out


def choose_blocks(d, ds, n_control, seed, out):
    rng = np.random.default_rng(seed)
    chrom_list = sorted(set(d["chroms"]), key=lambda c: (len(c), c))
    rows, picked = [], {}
    if ds["table_s8"]:
        gold = pd.read_csv(TREE / "tests" / "golden" / "tomato_locule" / "ld_blocks.csv")
        tb = pd.DataFrame({"Chr": gold["Chr"].astype(str), "Start (bp)": gold["Start (bp)"],
                           "End (bp)": gold["End (bp)"], "lead_snp": gold["lead_snp"],
                           "SNP_IDs": gold["SNP_IDs"]})
        # the golden's chromosome labels are canonical; the table's may not be
        from annotation import canon_chr
        lab = {canon_chr(c): c for c in chrom_list}
        tb["Chr"] = [lab.get(canon_chr(c), c) for c in tb["Chr"]]
        g = _group(d, tb)
        for i, r in tb.iterrows():
            key = f"{r['Chr']}:{int(r['Start (bp)'])}-{int(r['End (bp)'])}"
            if key not in g or g[key]["n_groups"] < 2:
                raise SystemExit(f"Table S8 block {key} is not tested with two groups")
            picked[key] = dict(g[key], kind="table_s8", Chr=r["Chr"], Start=int(r["Start (bp)"]),
                               End=int(r["End (bp)"]), SNP_IDs=r["SNP_IDs"], draws=0)
    by_chr = {c: np.flatnonzero(d["chroms"] == c) for c in chrom_list}
    for c in by_chr:
        by_chr[c] = by_chr[c][np.argsort(d["positions"][by_chr[c]], kind="stable")]
    draws = 0
    for i in range(n_control):
        c = chrom_list[i % len(chrom_list)]
        idx = by_chr[c]
        for attempt in range(1, 501):
            draws += 1
            L = int(rng.integers(2, 7))
            if idx.size < L:
                raise SystemExit(f"chromosome {c} has fewer than {L} SNPs")
            s = int(rng.integers(0, idx.size - L + 1))
            cols = idx[s:s + L]
            start, end = int(d["positions"][cols[0]]), int(d["positions"][cols[-1]])
            key = f"{c}:{start}-{end}"
            if key in picked:
                continue
            blk = pd.DataFrame({"Chr": [c], "Start (bp)": [start], "End (bp)": [end],
                                "lead_snp": [d["sid"][cols[0]]], "SNP_IDs": [",".join(d["sid"][cols])]})
            g = _group(d, blk)
            if key in g and g[key]["n_groups"] >= 2:
                picked[key] = dict(g[key], kind="control", Chr=c, Start=start, End=end,
                                   SNP_IDs=",".join(d["sid"][cols]), draws=attempt)
                break
        else:
            raise SystemExit(f"no testable control block on chromosome {c} in 500 draws")
    for k, b in picked.items():
        sizes = pd.Series(b["groups"]).value_counts().sort_index()
        rows.append({"block": k, "kind": b["kind"], "Chr": b["Chr"], "Start": b["Start"], "End": b["End"],
                     "SNP_IDs": b["SNP_IDs"], "n_snps_tested": b["n_snps"], "n_tested": len(b["samples"]),
                     "n_groups": b["n_groups"], "group_sizes": ";".join(f"{h}:{n}" for h, n in sizes.items()),
                     "draws": b["draws"]})
    pd.DataFrame(rows).to_csv(out / "blocks.csv", index=False)
    return picked, {"seed": int(seed), "draws_total": int(draws), "chromosomes": chrom_list}


# ------------------------------------------------------------------------- the phenotypes
def null_phenotype(d, h2, j, chrom_list, dataset):
    from gwas.utils import stable_seed

    c_out = chrom_list[j % len(chrom_list)]
    # a dataset's table on disk gets the same phenotypes as in memory (same seeds)
    # "CC63" is the seed label of the published run: changing it changes every phenotype
    seed = stable_seed("CC63", dataset.replace("_disk", ""), f"{h2:g}", j)
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(d["chroms"] != c_out)
    causal = np.sort(rng.choice(pool, size=N_CAUSAL, replace=False))
    Z = _columns(d, causal)
    sd = Z.std(axis=0)
    Z = (Z - Z.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    beta = rng.normal(0.0, 1.0, N_CAUSAL)
    g = Z @ beta
    vg = float(g.var())
    g = g * np.sqrt(h2 / vg) if (h2 > 0 and vg > 0) else np.zeros_like(g)
    y = g + rng.normal(0.0, np.sqrt(1.0 - h2), g.size)
    assert not np.any(d["chroms"][causal] == c_out)
    return y, dict(h2=h2, j=j, chr_out=c_out, seed=int(seed), var_g=float(g.var()),
                   var_y=float(y.var()), causal=";".join(d["sid"][causal]))


# ------------------------------------------------------------------------------- the tests
def run_tests(d, ds, picked, k_band, n_pheno, h2s, n_perm, dataset, out):
    from gwas.haplotype import block_test_lm_with_pcs, freedman_lane_perm_pvalue
    from gwas.haplotype_kinship import KinshipTestError, kernel_for_samples, kinship_block_test
    from gwas.utils import stable_seed

    chrom_list = sorted(set(d["chroms"]), key=lambda c: (len(c), c))
    row_of = pd.Index(np.asarray(d["geno_df"].index, dtype=str))
    pcs_sets = {"pcs0": 0}
    if k_band > 0:
        pcs_sets["pcsband"] = k_band
    for k in ds["pcs_fixed"]:
        pcs_sets[f"pcs{k}"] = k
    phenos, rows, timing = [], [], {}
    pv_path, ph_path = out / "pvalues.csv", out / "phenotypes.csv"
    for p_ in (pv_path, ph_path):
        if p_.exists():
            p_.unlink()
    y_all = {}
    for h2 in h2s:
        for j in range(n_pheno):
            y, info = null_phenotype(d, h2, j, chrom_list, dataset)
            y_all[(h2, j)] = y
            phenos.append(info)
    pd.DataFrame(phenos).to_csv(ph_path, index=False)
    blocks_by_chr = {}
    for key, b in picked.items():
        blocks_by_chr.setdefault(b["Chr"], []).append(key)
    for c in chrom_list:
        keys = blocks_by_chr.get(c, [])
        if not keys:
            continue
        eig = {}
        for key in keys:
            t0 = time.time()
            Kr, _ = kernel_for_samples(d["K_by_chr"][c], picked[key]["samples"])
            eig[key] = np.linalg.eigh(Kr)
            timing["eigh"] = timing.get("eigh", 0.0) + time.time() - t0
        for info in phenos:
            if info["chr_out"] != c:
                continue
            y = y_all[(info["h2"], info["j"])]
            name = f"null_{dataset.replace('_disk', '')}_h{info['h2']:g}_{info['j']}"
            for key in keys:
                b = picked[key]
                rows_i = row_of.get_indexer(b["samples"])
                yt, g = y[rows_i], b["groups"]
                seed = stable_seed(str(b["Chr"]), int(b["Start"]), int(b["End"]), name, "FL")
                base = dict(h2=info["h2"], j=info["j"], chr_out=c, block=key, kind=b["kind"],
                            n=len(yt), n_groups=b["n_groups"])
                t0 = time.time()
                try:
                    r = kinship_block_test(yt, g, None, eig=eig[key], n_perm=n_perm, seed=seed)
                    kin = dict(df1=r["df1"], df2=r["df2"], F=r["F_param"],
                               p_param=r["PValue_param"], p_perm=r["P_perm"], h2_null=r["h2"],
                               failed="")
                except KinshipTestError as e:       # the pipeline's "failed: <reason>" row
                    kin = dict(df1=np.nan, df2=np.nan, F=np.nan, p_param=np.nan,
                               p_perm=np.nan, h2_null=np.nan, failed=str(e))
                timing["kin"] = timing.get("kin", 0.0) + time.time() - t0
                rows.append(dict(base, test="kin", **kin))
                for tname, k in pcs_sets.items():
                    C = d["pcs_full"][rows_i, :k] if k > 0 else None
                    t0 = time.time()
                    F, pp, df1, df2 = block_test_lm_with_pcs(yt, g, pcs=C)
                    _F, pperm = freedman_lane_perm_pvalue(yt, g, pcs=C, n_perm=n_perm, seed=seed)
                    timing[tname] = timing.get(tname, 0.0) + time.time() - t0
                    rows.append(dict(base, test=tname, df1=df1, df2=df2, F=F, p_param=pp,
                                     p_perm=pperm, h2_null=np.nan, failed=""))
        pd.DataFrame(rows).to_csv(pv_path, index=False)            # after each chromosome
        print(f"  chromosome {c}: {len(rows)} test rows so far", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(phenos), timing, list(pcs_sets)


# ------------------------------------------------------------------------------ the report
def clopper_pearson(x, n, conf=0.95):
    from scipy.stats import beta

    a = 1.0 - conf
    lo = 0.0 if x == 0 else float(beta.ppf(a / 2, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(1 - a / 2, x + 1, n - x))
    return lo, hi


def report(pv, out, h2s, tests):
    rows = []
    for h2 in h2s:
        for t in tests:
            s = pv[(pv["h2"] == h2) & (pv["test"] == t)]
            for pcol in ("p_perm", "p_param"):
                p = s[pcol].astype(float)
                p = p[np.isfinite(p)]
                for a in ALPHAS:
                    x, n = int((p <= a).sum()), int(p.size)
                    lo, hi = clopper_pearson(x, n) if n else (np.nan, np.nan)
                    rows.append({"h2": h2, "test": t, "pvalue": pcol, "alpha": a, "n_tests": n,
                                 "n_rejected": x, "rate": x / n if n else np.nan,
                                 "ci95_low": lo, "ci95_high": hi,
                                 "interval_entirely_above_alpha": bool(n and lo > a)})
    rates = pd.DataFrame(rows)
    rates.to_csv(out / "rates.csv", index=False)
    kin = rates[rates["test"] == "kin"]
    failed = kin[kin["interval_entirely_above_alpha"]]
    stop = {"rule": "kin: a 95% Clopper-Pearson interval entirely above alpha, at alpha 0.05 or 0.01, "
                    "for P_perm or PValue_param, at any h2, fails the check",
            "stop": bool(len(failed)), "failing": failed.to_dict(orient="records"),
            "checked": int(len(kin))}
    (out / "stop_rule.json").write_text(json.dumps(stop, indent=1))
    qq(pv, out, h2s, tests)
    return rates, stop


def qq(pv, out, h2s, tests):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import beta

    for h2 in h2s:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
        for ax, pcol, lab in zip(axes, ("p_perm", "p_param"), ("P_perm", "PValue_param")):
            n_max = 0
            for t in tests:
                p = pv[(pv["h2"] == h2) & (pv["test"] == t)][pcol].astype(float)
                p = np.sort(p[np.isfinite(p)].to_numpy())
                n = p.size
                if not n:
                    continue
                n_max = max(n_max, n)
                e = (np.arange(1, n + 1) - 0.5) / n
                ax.plot(-np.log10(e), -np.log10(p), ".", ms=3, label=f"{t} (n={n})")
            if n_max:
                i = np.arange(1, n_max + 1)
                lo, hi = beta.ppf(0.975, i, n_max - i + 1), beta.ppf(0.025, i, n_max - i + 1)
                e = (i - 0.5) / n_max
                ax.fill_between(-np.log10(e), -np.log10(lo), -np.log10(hi), color="0.85", lw=0,
                                label="95% band (independent tests)")
                m = -np.log10(0.5 / n_max)
                ax.plot([0, m], [0, m], "k-", lw=0.8)
            ax.set_xlabel("expected -log10 p")
            ax.set_ylabel(f"observed -log10 {lab}")
            ax.set_title(f"h2 = {h2:g}: {lab}")
            ax.legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"qq_h2_{h2:g}.{ext}", dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------------- checks
def check_against_pipeline(d, ds):
    """The front stages give the kernels the pipeline itself hands the haplotype test (a
    pipeline run on the same inputs, its call recorded): K0 and every LOCO kernel equal."""
    import gwas.haplotype as gh
    import gwas.pipeline as pl

    seen = []
    real = gh.run_haplotype_block_gwas

    def rec(*a, **k):
        seen.append(k)
        return real(*a, **k)

    gh.run_haplotype_block_gwas = rec
    try:
        p = Path(ds["pheno"])
        pheno = pd.read_csv(p, index_col=0, encoding="utf-8-sig")
        parsed = d["parsed"]
        pl.run_trait_pipeline(parsed, pheno, ds["trait"], pl.cli_default_options(
            vcf=str(ds["vcf"]), pheno=str(p), output=None, model=["mlm"], no_plots=True,
            no_report=True, no_annotation=True, ld_r2=0.1, ld_flank_kb=200))
    finally:
        gh.run_haplotype_block_gwas = real
    if not seen:
        return {"pipeline_kernels_equal": None, "note": "the pipeline tested no block"}
    K0, K_by_chr = seen[0]["kernels"]
    same = (np.array_equal(np.asarray(K0.val), np.asarray(d["K0"].val))
            and set(K_by_chr) == set(d["K_by_chr"])
            and all(np.array_equal(np.asarray(K_by_chr[c].val), np.asarray(d["K_by_chr"][c].val))
                    for c in K_by_chr))
    return {"pipeline_kernels_equal": bool(same)}


# ----------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(datasets()))
    ap.add_argument("--out", required=True)
    ap.add_argument("--vcf", default=None, help="the VCF (tomato and rice)")
    ap.add_argument("--pheno", default=None, help="the phenotype file (tomato and rice)")
    ap.add_argument("--n-pheno", type=int, default=200)
    ap.add_argument("--h2", type=float, nargs="+", default=[0.0, 0.3, 0.6])
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-control", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--band-k", type=int, default=None, help="skip the band-rule scan (this k)")
    ap.add_argument("--skip-checks", action="store_true")
    a = ap.parse_args(argv)
    import gwas

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = datasets()[a.dataset]
    if a.vcf or a.pheno:
        ds.update(vcf=Path(a.vcf), pheno=Path(a.pheno))
    if ds["vcf"] is None or ds["pheno"] is None:
        raise SystemExit(f"--dataset {a.dataset} needs --vcf and --pheno")
    t_all = time.time()
    head = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True,
                          cwd=str(TREE)).stdout.strip()
    meta = {"dataset": a.dataset, "args": vars(a), "gwas_file": gwas.__file__, "tree": str(TREE),
            "tree_head": head, "vcf": str(ds["vcf"]), "pheno": str(ds["pheno"]), "trait": ds["trait"]}
    print("gwas.__file__:", gwas.__file__, "tree", TREE, head, flush=True)
    if str(TREE.resolve()).lower() not in str(Path(gwas.__file__).resolve()).lower():
        raise SystemExit("gwas was imported from outside the tree")
    t0 = time.time()
    d = front_stages(ds, out)
    n, m = d["geno_imputed"].shape
    meta.update(n_samples=int(n), n_snps=int(m), n_raw=d["n_raw"], store=bool(ds["store"]),
                seconds_front=round(time.time() - t0, 1),
                kernels={c: ("K0" if K is d["K0"] else "LOCO") for c, K in d["K_by_chr"].items()})
    print(f"data: {n} x {m} ({meta['seconds_front']} s); expected {ds['expect']}", flush=True)
    if ds["expect"] is not None and (n, m) != tuple(ds["expect"]):
        raise SystemExit(f"the table is {n} x {m}, not {ds['expect']}")
    try:
        if not a.skip_checks and a.dataset.startswith("example"):
            meta["check"] = check_against_pipeline(d, ds)
            print("check against the pipeline:", meta["check"], flush=True)
        t0 = time.time()
        if a.band_k is None:
            meta["band_rule"] = band_rule(d, ds["trait"], out)
        else:
            meta["band_rule"] = {"k": int(a.band_k), "how": "given (--band-k)"}
        meta["seconds_band_rule"] = round(time.time() - t0, 1)
        k_band = int(meta["band_rule"]["k"])
        print("band rule:", meta["band_rule"], flush=True)
        t0 = time.time()
        n_control = ds["n_control"] if a.n_control is None else a.n_control
        picked, binfo = choose_blocks(d, ds, n_control, a.seed, out)
        meta["blocks"] = dict(binfo, n_blocks=len(picked),
                              n_table_s8=sum(b["kind"] == "table_s8" for b in picked.values()),
                              seconds=round(time.time() - t0, 1))
        print(f"blocks: {len(picked)} ({meta['blocks']['seconds']} s)", flush=True)
        t0 = time.time()
        pv, ph, timing, tests = run_tests(d, ds, picked, k_band, a.n_pheno, a.h2, a.n_perm,
                                          a.dataset, out)
        meta["seconds_tests"] = round(time.time() - t0, 1)
        meta["timing_by_test_s"] = {k: round(v, 1) for k, v in timing.items()}
        meta["tests"] = ["kin"] + tests
        # kin tests the engine refused (KinshipTestError; a "failed: <reason>" row in the
        # pipeline): no p-value, so not in the rates' counts; their reasons
        _f = pv[(pv["test"] == "kin") & (pv["failed"].astype(str) != "")]
        meta["kin_failed"] = {"n": int(len(_f)),
                              "reasons": _f["failed"].value_counts().to_dict()}
        # every tested block lies on its phenotype's held-out chromosome, and no causal SNP there
        causal_chr = {(r.h2, r.j): {d["chroms"][i] for i in np.flatnonzero(np.isin(d["sid"], r.causal.split(";")))}
                      for r in ph.itertuples()}
        bad = sum(r.chr_out in causal_chr[(r.h2, r.j)] or picked[r.block]["Chr"] != r.chr_out
                  for r in pv.itertuples())
        meta["null_checks"] = {"tests": int(len(pv)), "violations": int(bad)}
        rates, stop = report(pv, out, a.h2, ["kin"] + tests)
        meta["stop_rule"] = stop
        print(rates.to_string(index=False), flush=True)
        print("STOP RULE:", "FAILS" if stop["stop"] else "passes", flush=True)
    finally:
        meta["seconds_total"] = round(time.time() - t_all, 1)
        (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
        if d.get("store_dir") is not None:
            from gwas.genostore import remove_store
            remove_store(d["store_dir"], d["parsed"])
    return 3 if meta.get("stop_rule", {}).get("stop") else 0


if __name__ == "__main__":
    sys.exit(main())
