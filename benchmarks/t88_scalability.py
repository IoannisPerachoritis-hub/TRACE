"""T-88 - post-GWAS scalability envelope (measured points; + memory, cache-call
counter, a pepper sample point, and a fixed-signal-set row).

Times the three post-GWAS stages - LD-block detection (find_ld_clusters_genomewide
+ filter_contained_blocks), haplotype-block testing (run_haplotype_block_gwas,
n_perm=1000), and annotation (annotate_ld_blocks) - as a function of size, on the
committed QC data. Also reports:
  * peak process RSS (psutil) + the exact genotype-matrix bytes (n*m*8) per size;
  * pairwise_r2 call counts WITH vs WITHOUT the shared _r2_window_cache (the
    machine-independent evidence that cache injection reduces recomputation);
  * a second panel (pepper_BX, 350 samples x 5,862 SNPs);
  * a fixed-signal row (all significant SNPs kept, background subsampled) that holds
    the block set fixed while varying the scanned-marker count.

MUST be run from the TRACE-release tree (imports the shipping, committed code). The
DEV working tree's --ld-dedup WIP thins the timed post-GWAS path and must not be
measured. Reuses benchmarks/capture_golden.py::_load_qc. Params = the T-61 config
that reproduces Table S8 (flank_kb=144, ld_decay_kb=72.17).

Run (from TRACE-release):  .../python.exe benchmarks/t88_scalability.py
"""
import gc
import hashlib as _real_hashlib
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import psutil
    _PROC = psutil.Process()
except Exception:  # noqa: BLE001
    _PROC = None

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))
from gwas import ld  # noqa: E402
from gwas.haplotype import run_haplotype_block_gwas  # noqa: E402
from annotation import annotate_ld_blocks, load_gene_annotation  # noqa: E402
from capture_golden import _load_qc, _rename_start_end  # noqa: E402

TOMATO_QC = ROOT / "benchmarks" / "qc_data" / "tomato_locule_number"
TOMATO_GWAS = TOMATO_QC / "platform_GWAS_locule_number.csv"
PEPPER_QC = ROOT / "benchmarks" / "qc_data" / "pepper_BX"
PEPPER_GWAS = PEPPER_QC / "platform_GWAS_BX.csv"
GENES_CSV = ROOT / "data" / "Sol_genes_SL3.csv"

FLANK_KB = 144.0
LD_DECAY_KB = 72.17
N_PERM = 1000
SIZES = [5000, 10000, 20000, None]   # None = full
REPS = 3
SEED = 42


def _rss_mb():
    return _PROC.memory_info().rss / 1e6 if _PROC else float("nan")


def _git_head():
    """Actual committed HEAD of the tree that runs this script (machine-emitted, not hand-typed).

    The provenance block used to print a hardcoded "TRACE-release committed HEAD" string, which is
    how the note's SHA drifted stale (c3a4bdf) -- emit the real hash so it can never be mis-attributed.
    """
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _run_ld(gwas_df, chroms, positions, geno, sid):
    blocks = ld.find_ld_clusters_genomewide(
        gwas_df=gwas_df, chroms=chroms, positions=positions,
        geno_imputed=geno.astype(float), sid=sid,
        ld_threshold=0.6, flank_kb=FLANK_KB, ld_decay_kb=LD_DECAY_KB,
        min_snps=3, top_n=10, sig_thresh=1e-5, adj_r2_min=0.2,
        merge_iou=0.3, gap_factor=10.0,
    )
    blocks, _ = ld.filter_contained_blocks(blocks, min_contained=2)
    return blocks


def _run_hap(blocks, chroms, positions, geno, sid, geno_df, pheno_df, tcol):
    if blocks.empty:
        return 0
    haplo_df = _rename_start_end(blocks.copy())
    hap, _ = run_haplotype_block_gwas(
        haplo_df=haplo_df, chroms=chroms, positions=positions,
        geno_imputed=geno.astype(float), sid=sid,
        geno_df=geno_df, pheno_df=pheno_df, trait_col=tcol,
        pcs=None, n_perm=N_PERM, n_pcs_used=0, min_hap_count=5, min_group_size=3,
    )
    return 0 if hap is None else len(hap)


def _timed(label, gwas_df, chroms, positions, geno, sid, geno_df, pheno_df, tcol, genes):
    gc.collect()
    base_rss = _rss_mb()
    peak_rss = base_rss
    tl, th, ta = [], [], []
    nb = nh = n_seed = 0
    n_seed = int((gwas_df["PValue"] < 1e-5).sum())
    for _ in range(REPS):
        t0 = time.perf_counter(); blocks = _run_ld(gwas_df, chroms, positions, geno, sid)
        tl.append(time.perf_counter() - t0); peak_rss = max(peak_rss, _rss_mb())
        t1 = time.perf_counter(); nh = _run_hap(blocks, chroms, positions, geno, sid, geno_df, pheno_df, tcol)
        th.append(time.perf_counter() - t1); peak_rss = max(peak_rss, _rss_mb())
        t2 = time.perf_counter()
        if genes is not None and not blocks.empty:
            annotate_ld_blocks(blocks.copy(), genes, n_flank=2, max_flank_dist_bp=500_000)
        ta.append(time.perf_counter() - t2); peak_rss = max(peak_rss, _rss_mb())
        nb = len(blocks)
    n, m = geno.shape
    return dict(label=label, n_samp=n, n_snps=m, seed=n_seed, blocks=nb, hap=nh,
                t_ld=float(np.mean(tl)), t_hap=float(np.mean(th)), t_annot=float(np.mean(ta)),
                total=float(np.mean(tl) + np.mean(th) + np.mean(ta)),
                peak_rss=peak_rss, d_rss=peak_rss - base_rss, geno_mb=n * m * 8 / 1e6)


def _subset(geno, sid, chroms, positions, geno_df, gwas_full, idx):
    idx = np.sort(idx)
    sid_s = sid[idx]
    gwas_s = gwas_full.set_index("SNP").loc[sid_s].reset_index()
    return geno[:, idx], sid_s, chroms[idx], positions[idx], geno_df.iloc[:, idx], gwas_s


# ---- pairwise_r2 call counter (monkeypatch) + cache-defeat shim ----
class _UniqDigest:
    __slots__ = ("_n",)
    def __init__(self, n): self._n = n
    def hexdigest(self): return f"u{self._n}"


class _ShimHashlib:
    def __init__(self): self._c = 0
    def blake2b(self, *a, **k):
        self._c += 1
        return _UniqDigest(self._c)


def count_pairwise_r2(gwas_df, chroms, positions, geno, sid, *, defeat_cache):
    orig = ld.pairwise_r2
    n = {"c": 0}
    def counting(*a, **k):
        n["c"] += 1
        return orig(*a, **k)
    ld.pairwise_r2 = counting
    if defeat_cache:
        ld.hashlib = _ShimHashlib()      # every cache_key becomes unique -> cache always misses
    try:
        _run_ld(gwas_df, chroms, positions, geno, sid)
    finally:
        ld.pairwise_r2 = orig
        ld.hashlib = _real_hashlib
    return n["c"]


def main():
    proc_cpu = platform.processor() or platform.machine()
    _sha = _git_head()
    _ram_gb = psutil.virtual_memory().total / 1e9 if _PROC is not None else float("nan")
    _phys = psutil.cpu_count(logical=False) if _PROC is not None else None
    print("=" * 96)
    print("  T-88 post-GWAS scalability envelope (measured points + memory + cache counts)")
    print("=" * 96)
    print(f"  machine  : {proc_cpu} | cores = {_phys} physical / {os.cpu_count()} logical | "
          f"RAM {_ram_gb:.1f} GB | {platform.system()} {platform.release()}")
    print(f"  tree     : TRACE-release, git HEAD = {_sha}")
    print(f"  timed ld : {ld.__file__}")
    print(f"  params   : flank=144kb, ld_decay=72.17kb, n_perm={N_PERM}, reps={REPS}, seed={SEED}")
    if _PROC is None:
        print("  WARNING: psutil unavailable - RSS columns will be nan")

    # --- tomato ---
    geno, geno_df, pheno_df, sid, chroms, positions = _load_qc(TOMATO_QC)
    sid = np.asarray(sid).astype(str)
    gwas_full = pd.read_csv(TOMATO_GWAS)
    genes = load_gene_annotation(str(GENES_CSV)) if GENES_CSV.exists() else None
    tcol = next(c for c in pheno_df.columns if c.lower() not in ("fid", "iid", "sampleid"))
    m_full = geno.shape[1]

    rows = []
    rng = np.random.default_rng(SEED)
    for K in SIZES:
        idx = np.arange(m_full) if K is None else rng.choice(m_full, size=K, replace=False)
        g, s, c, p, gd, gw = _subset(geno, sid, chroms, positions, geno_df, gwas_full, idx)
        rows.append(_timed(f"tomato/{'full' if K is None else K}", gw, c, p, g, s, gd, pheno_df, tcol, genes))

    # fixed-signal row: keep ALL significant SNPs, subsample background to ~20k
    sig_snps = set(gwas_full.loc[gwas_full["PValue"] < 1e-5, "SNP"].astype(str))
    is_sig = np.array([x in sig_snps for x in sid])
    sig_idx = np.where(is_sig)[0]
    bg_idx = np.where(~is_sig)[0]
    n_bg = max(0, 20000 - len(sig_idx))
    fixed_idx = np.concatenate([sig_idx, np.random.default_rng(SEED).choice(bg_idx, size=n_bg, replace=False)])
    g, s, c, p, gd, gw = _subset(geno, sid, chroms, positions, geno_df, gwas_full, fixed_idx)
    rows.append(_timed("tomato/20k-fixed-signal", gw, c, p, g, s, gd, pheno_df, tcol, genes))

    # --- pepper (second real panel; 350 x 5,862; annotation skipped - no gene model in release) ---
    try:
        pg, pgd, pph, psid, pch, ppos = _load_qc(PEPPER_QC)
        psid = np.asarray(psid).astype(str)
        pgw = pd.read_csv(PEPPER_GWAS)
        ptcol = next(cc for cc in pph.columns if cc.lower() not in ("fid", "iid", "sampleid"))
        rows.append(_timed("pepper_BX/full", pgw, pch, ppos, pg, psid, pgd, pph, ptcol, None))
    except Exception as e:  # noqa: BLE001
        print(f"  (pepper point skipped: {e})")

    # --- print table ---
    print(f"\n  {'panel/size':<24}{'samp':>5}{'SNPs':>7}{'seed':>5}{'blk':>4}{'hap':>4}"
          f"{'LD(s)':>8}{'hap(s)':>8}{'ann(s)':>8}{'tot(s)':>8}{'peakRSS':>9}{'dRSS':>7}{'genoMB':>8}")
    for r in rows:
        print(f"  {r['label']:<24}{r['n_samp']:>5}{r['n_snps']:>7}{r['seed']:>5}{r['blocks']:>4}{r['hap']:>4}"
              f"{r['t_ld']:>8.3f}{r['t_hap']:>8.3f}{r['t_annot']:>8.3f}{r['total']:>8.3f}"
              f"{r['peak_rss']:>9.0f}{r['d_rss']:>7.0f}{r['geno_mb']:>8.1f}")

    # --- counted criterion (full tomato) ---
    idx_full = np.arange(m_full)
    g, s, c, p, gd, gw = _subset(geno, sid, chroms, positions, geno_df, gwas_full, idx_full)
    n_with = count_pairwise_r2(gw, c, p, g, s, defeat_cache=False)
    n_without = count_pairwise_r2(gw, c, p, g, s, defeat_cache=True)
    print(f"\n  counted criterion (full tomato, LD-block detection):")
    print(f"    pairwise_r2 calls WITH shared _r2_window_cache    : {n_with}")
    print(f"    pairwise_r2 calls WITHOUT the cache (forced miss) : {n_without}")
    print(f"    saved by cache = {n_without - n_with}  (with <= without: {n_with <= n_without})")

    # --- SNP-sweep ratios (tomato random subsamples only) ---
    sweep = [r for r in rows if r["label"].startswith("tomato/") and "fixed" not in r["label"]]
    print("\n  tomato SNP-sweep ratios (SNP-count ratio -> total-time ratio):")
    for a, b in zip(sweep[:-1], sweep[1:]):
        sr = b["n_snps"] / a["n_snps"]; tr = b["total"] / a["total"] if a["total"] > 0 else float("nan")
        print(f"    {a['n_snps']} -> {b['n_snps']}: SNPs x{sr:.2f}  ->  total x{tr:.2f}")

    return rows, n_with, n_without


if __name__ == "__main__":
    main()
