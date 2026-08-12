"""T-88 - post-GWAS scalability envelope (measured points, not a fitted law).

Times the three post-GWAS stages - LD-block detection (find_ld_clusters_genomewide
+ filter_contained_blocks), haplotype-block testing (run_haplotype_block_gwas,
n_perm=1000), and annotation (annotate_ld_blocks) - as a function of SNP count, on
the committed tomato locule QC data (165 samples x 43,974 SNPs). SNPs are
random-subsampled (seed=42, sorted so within-chromosome order is preserved), which
thins the signal proportionally; the block/haplotype counts are reported alongside
the times so the WORK done is visible.

MUST be run from the TRACE-release tree (imports the shipping, committed code). The
DEV working tree's --ld-dedup WIP thins the timed post-GWAS path and must not be
measured. Reuses benchmarks/capture_golden.py::_load_qc. Params = the T-61 config
that reproduces Table S8 (flank_kb=144, ld_decay_kb=72.17).

Run (from TRACE-release):  .../python.exe benchmarks/t88_scalability.py
"""
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))
from gwas import ld  # noqa: E402
from gwas.haplotype import run_haplotype_block_gwas  # noqa: E402
from annotation import annotate_ld_blocks, load_gene_annotation  # noqa: E402
from capture_golden import _load_qc, _rename_start_end  # noqa: E402

QC_DIR = ROOT / "benchmarks" / "qc_data" / "tomato_locule_number"
GWAS_CSV = QC_DIR / "platform_GWAS_locule_number.csv"
GENES_CSV = ROOT / "data" / "Sol_genes_SL3.csv"

# T-61 reproducing config
FLANK_KB = 144.0
LD_DECAY_KB = 72.17
N_PERM = 1000
SIZES = [5000, 10000, 20000, None]   # None = full (43,974)
REPS = 3


def _one_run(gwas_df, chroms, positions, geno, sid, geno_df, pheno_df, trait_col, genes):
    t0 = time.perf_counter()
    blocks = ld.find_ld_clusters_genomewide(
        gwas_df=gwas_df, chroms=chroms, positions=positions,
        geno_imputed=geno.astype(float), sid=sid,
        ld_threshold=0.6, flank_kb=FLANK_KB, ld_decay_kb=LD_DECAY_KB,
        min_snps=3, top_n=10, sig_thresh=1e-5, adj_r2_min=0.2,
        merge_iou=0.3, gap_factor=10.0,
    )
    blocks, _ = ld.filter_contained_blocks(blocks, min_contained=2)
    t_ld = time.perf_counter() - t0

    n_hap = 0
    t1 = time.perf_counter()
    if not blocks.empty:
        haplo_df = _rename_start_end(blocks.copy())
        hap, _ = run_haplotype_block_gwas(
            haplo_df=haplo_df, chroms=chroms, positions=positions,
            geno_imputed=geno.astype(float), sid=sid,
            geno_df=geno_df, pheno_df=pheno_df, trait_col=trait_col,
            pcs=None, n_perm=N_PERM, n_pcs_used=0, min_hap_count=5, min_group_size=3,
        )
        n_hap = 0 if hap is None else len(hap)
    t_hap = time.perf_counter() - t1

    t2 = time.perf_counter()
    if not blocks.empty and genes is not None:
        annotate_ld_blocks(blocks.copy(), genes, n_flank=2, max_flank_dist_bp=500_000)
    t_annot = time.perf_counter() - t2

    return len(blocks), n_hap, t_ld, t_hap, t_annot


def main():
    geno, geno_df, pheno_df, sid, chroms, positions = _load_qc(QC_DIR)
    gwas_full = pd.read_csv(GWAS_CSV)
    genes = load_gene_annotation(str(GENES_CSV)) if GENES_CSV.exists() else None
    tcol = next(c for c in pheno_df.columns if c.lower() not in ("fid", "iid", "sampleid"))
    n_samp, m_full = geno.shape
    sid = np.asarray(sid).astype(str)

    print("=" * 84)
    print("  T-88 post-GWAS scalability envelope - tomato locule QC (measured points)")
    print("=" * 84)
    print(f"  machine  : {platform.processor() or platform.machine()} | "
          f"logical procs (os) = {__import__('os').cpu_count()} | {platform.system()} {platform.release()}")
    print(f"  tree     : TRACE-release committed HEAD (run this script FROM TRACE-release)")
    print(f"  timed ld : {ld.__file__}")
    print(f"  data     : {n_samp} samples x {m_full} SNPs | post-GWAS params: flank=144kb, "
          f"ld_decay=72.17kb, n_perm={N_PERM}")
    print(f"  stages   : LD-block detection -> haplotype testing (n_perm={N_PERM}) -> annotation\n")

    hdr = (f"  {'n_SNPs':>8}{'n_seed':>7}{'blocks':>7}{'hap':>5}"
           f"{'LD(s)':>9}{'hap(s)':>9}{'annot(s)':>10}{'total(s)':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows = []
    rng = np.random.default_rng(42)
    for K in SIZES:
        if K is None:
            idx = np.arange(m_full)
        else:
            idx = np.sort(rng.choice(m_full, size=K, replace=False))
        geno_s = geno[:, idx]
        sid_s = sid[idx]
        chroms_s = chroms[idx]
        positions_s = positions[idx]
        geno_df_s = geno_df.iloc[:, idx]
        gwas_s = gwas_full.set_index("SNP").loc[sid_s].reset_index()
        n_seed = int((gwas_s["PValue"] < 1e-5).sum())

        best = None  # keep the min-total rep (least noise); also collect all for the mean
        times = []
        for _ in range(REPS):
            nb, nh, tl, th, ta = _one_run(
                gwas_s, chroms_s, positions_s, geno_s, sid_s, geno_df_s, pheno_df, tcol, genes)
            times.append((tl, th, ta))
        arr = np.array(times)
        tl, th, ta = arr.mean(axis=0)
        tot = tl + th + ta
        rows.append(dict(n_snps=len(idx), n_seed=n_seed, blocks=nb, hap=nh,
                         t_ld=tl, t_hap=th, t_annot=ta, total=tot))
        print(f"  {len(idx):>8}{n_seed:>7}{nb:>7}{nh:>5}"
              f"{tl:>9.3f}{th:>9.3f}{ta:>10.3f}{tot:>10.3f}")

    # ratios between consecutive sizes (measured, not fitted)
    print("\n  ratios (consecutive sizes; SNP-count ratio -> total-time ratio):")
    for a, b in zip(rows[:-1], rows[1:]):
        snp_r = b["n_snps"] / a["n_snps"]
        tot_r = b["total"] / a["total"] if a["total"] > 0 else float("nan")
        print(f"    {a['n_snps']} -> {b['n_snps']}: SNPs x{snp_r:.2f}  ->  total time x{tot_r:.2f}")

    return rows


if __name__ == "__main__":
    main()
