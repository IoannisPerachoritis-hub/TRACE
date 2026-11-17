"""Figure S8: how often each SNP is found in the subsampling of tomato locule number.

A histogram of the per-SNP discovery frequency over the resamples, for the SNPs found at least
once, with the Bonferroni SNPs of the full-data MLM table marked. It reads the CLI's subsampling
table, or runs the subsampling itself when no table is given.

    python benchmarks/plots/subsampling_frequency.py --csv Subsampling_SNP_stability.csv [--data-root DIR]

The table comes from the CLI run in benchmarks/README.md (Figure S8): tables/Subsampling_SNP_stability.csv
in its ZIP. Input also: qc_data/tomato_locule_number/platform_GWAS_locule_number.csv (the Bonferroni
set). Output: benchmarks/plots/figure_s8_subsampling.png and .pdf, in the data root.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks._options import options, runs_in_root  # noqa: E402
OPT = options()
ROOT = OPT.root   # the data root: this repository unless --data-root is given

from utils.pub_theme import apply_matplotlib_theme, PALETTE
apply_matplotlib_theme()

PAPER_FIGURES = ROOT / "benchmarks" / "plots"
PAPER_FIGURES.mkdir(parents=True, exist_ok=True)

QC_DIR = ROOT / "benchmarks" / "qc_data" / "tomato_locule_number"


def load_tomato_data():
    """Load QC'd tomato locule data and return arrays."""
    geno_df = pd.read_csv(QC_DIR / "QC_genotype_matrix.csv", index_col=0)
    snp_map = pd.read_csv(QC_DIR / "QC_snp_map.csv")
    pheno_df = pd.read_csv(QC_DIR / "QC_phenotype.csv", index_col=0)

    geno_imputed = geno_df.values.astype(np.float32)
    col_means = np.nanmean(geno_imputed, axis=0)
    nan_mask = np.isnan(geno_imputed)
    if nan_mask.any():
        geno_imputed[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    sid = snp_map["SNP_ID"].values if "SNP_ID" in snp_map.columns else snp_map.iloc[:, 0].values

    from annotation import canon_chr
    chroms_raw = snp_map["Chr"].values if "Chr" in snp_map.columns else snp_map.iloc[:, 1].values
    chroms_str = np.array([str(canon_chr(c)) for c in chroms_raw])

    chr_map = {}
    for c in np.unique(chroms_str):
        try:
            chr_map[c] = int(c)
        except ValueError:
            chr_map[c] = hash(c) % 1000
    chroms_num = np.array([chr_map[c] for c in chroms_str])

    positions = snp_map["Pos"].values.astype(float) if "Pos" in snp_map.columns else snp_map.iloc[:, 2].values.astype(float)

    trait_col = [c for c in pheno_df.columns if c.lower() not in ("fid", "iid", "sampleid")][0]
    y = pheno_df[trait_col].values.astype(float)

    geno_df.index = geno_df.index.astype(str)
    pheno_df.index = pheno_df.index.astype(str)
    common = geno_df.index.intersection(pheno_df.index)
    geno_idx = [i for i, x in enumerate(geno_df.index) if x in set(common)]
    pheno_idx = pheno_df.index.get_indexer(geno_df.index[geno_idx])

    geno_imputed = geno_imputed[geno_idx, :]
    y = y[pheno_idx]

    sample_ids = geno_df.index[geno_idx].astype(str).values
    iid = np.column_stack([sample_ids, sample_ids])

    return geno_imputed, y, sid, chroms_str, chroms_num, positions, iid


def run_subsampling(geno_imputed, y, sid, chroms_str, chroms_num, positions, iid):
    """Run subsampling GWAS and return discovery DataFrame."""
    from gwas.kinship import (
        _standardize_geno_for_grm, _build_grm_from_Z,
        _ld_prune_for_grm_by_chr_bp,
    )
    from gwas.subsampling import subsample_gwas_resampling
    from scipy.linalg import eigh

    Z_pruned, keep_mask = _ld_prune_for_grm_by_chr_bp(
        chroms_str, positions, geno_imputed, return_mask=True,
    )
    Z_grm = _standardize_geno_for_grm(Z_pruned)
    K_base = _build_grm_from_Z(Z_grm)

    eigvals, eigvecs = eigh(K_base)
    idx_sort = np.argsort(eigvals)[::-1]
    pcs_full = eigvecs[:, idx_sort[:10]]

    print("Running subsampling GWAS (100 resamples)...")
    disc_df, raw_pvals, rep_meta = subsample_gwas_resampling(
        geno_imputed=geno_imputed,
        y=y,
        sid=sid,
        chroms=chroms_str,
        chroms_num=chroms_num,
        positions=positions,
        iid=iid,
        Z_for_grm=Z_grm,
        pcs_full=pcs_full,
        n_pcs=0,
        n_reps=100,
        sample_frac=0.80,
        discovery_thresh=1e-4,
        seed=42,
        progress_callback=lambda rep, total: print(f"  Replicate {rep}/{total}..."),
    )

    return disc_df


def load_significant_snps():
    """Load MLM GWAS results to identify Bonferroni-significant SNPs."""
    gwas_path = QC_DIR / "platform_GWAS_locule_number.csv"
    if not gwas_path.exists():
        return set()

    gwas_df = pd.read_csv(gwas_path)
    p_col = "PValue" if "PValue" in gwas_df.columns else "pvalue"
    snp_col = "SNP" if "SNP" in gwas_df.columns else "sid"

    # Use Bonferroni significance column if available, otherwise compute
    if "Significant_Bonf" in gwas_df.columns:
        sig = gwas_df[gwas_df["Significant_Bonf"] == True]
    else:
        n_tests = len(gwas_df)
        bonf = 0.05 / n_tests
        sig = gwas_df[gwas_df[p_col] < bonf]

    return set(sig[snp_col].values)


def plot_subsampling_histogram(disc_df, sig_snps):
    """Create the discovery frequency histogram (freq > 0 only)."""
    freq = disc_df["DiscoveryFreq"].values
    is_sig = disc_df["SNP"].isin(sig_snps).values

    # Split into zero and non-zero
    n_total = len(freq)
    n_zero = int((freq == 0).sum())
    n_nonzero = n_total - n_zero

    # Filter to non-zero only
    mask_nz = freq > 0
    freq_nz = freq[mask_nz]
    is_sig_nz = is_sig[mask_nz]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bins = np.arange(0, 1.075, 0.05)
    # Shift bins slightly so lowest bin starts just above 0
    bins[0] = 0.001

    # Non-significant SNPs with non-zero discovery freq
    ax.hist(
        freq_nz[~is_sig_nz], bins=bins,
        color=PALETTE["grey"], alpha=0.6,
        label=f"Non-significant (n = {(~is_sig_nz).sum()})",
        edgecolor="white", linewidth=0.5,
    )

    # Significant SNPs
    if is_sig_nz.any():
        ax.hist(
            freq_nz[is_sig_nz], bins=bins,
            color=PALETTE["red"], alpha=0.85,
            label=f"Bonferroni-significant (n = {is_sig_nz.sum()})",
            edgecolor="white", linewidth=0.5,
        )

    # 50% stability line
    ax.axvline(x=0.5, color=PALETTE["blue"], linestyle="--", linewidth=1.2,
               alpha=0.7, label="50% stability threshold")

    ax.set_xlabel("Discovery frequency across 100 subsampling resamples")
    ax.set_ylabel("Number of SNPs")
    ax.set_title("Subsampling GWAS Stability (Tomato Locule Number)")

    # Text annotation for zero-frequency SNPs
    pct_zero = 100.0 * n_zero / n_total
    ax.text(
        0.98, 0.95,
        f"{n_zero:,} SNPs ({pct_zero:.1f}%) with zero\ndiscovery frequency (not shown)",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        color=PALETTE["grey"],
        fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ddd", alpha=0.9),
    )

    ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#ccc")

    fig.tight_layout()

    for ext in ["png", "pdf"]:
        out = PAPER_FIGURES / f"figure_s8_subsampling.{ext}"
        fig.savefig(out, dpi=300)
        print(f"Saved: {out}")

    plt.close(fig)

    # Print summary stats
    print(f"\nSummary:")
    print(f"  Total SNPs: {n_total:,}")
    print(f"  Zero freq: {n_zero:,} ({pct_zero:.1f}%)")
    print(f"  Non-zero freq: {n_nonzero}")
    print(f"  Significant SNPs (Bonferroni): {is_sig.sum()}")
    if is_sig_nz.any():
        sig_freqs = freq_nz[is_sig_nz]
        print(f"  Significant SNP freq range: {sig_freqs.min():.3f} - {sig_freqs.max():.3f}")
        n_stable = (sig_freqs >= 0.5).sum()
        print(f"  Significant SNPs with freq >= 50%: {n_stable}/{is_sig_nz.sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure S8: Subsampling histogram")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to pre-computed subsampling discovery CSV")
    args = parser.parse_args()

    if args.csv:
        disc_df = pd.read_csv(args.csv)
        print(f"Loaded pre-computed data: {len(disc_df)} SNPs")
    else:
        geno, y, sid, chroms_str, chroms_num, positions, iid = load_tomato_data()
        print(f"Loaded: {geno.shape[0]} samples, {geno.shape[1]} SNPs")
        disc_df = run_subsampling(geno, y, sid, chroms_str, chroms_num, positions, iid)

        csv_out = PAPER_FIGURES / "subsampling_discovery_table.csv"
        disc_df.to_csv(csv_out, index=False)
        print(f"Saved table: {csv_out}")

    sig_snps = load_significant_snps()
    print(f"Significant SNPs (Bonferroni): {len(sig_snps)}")

    plot_subsampling_histogram(disc_df, sig_snps)
    print("\nDone.")
