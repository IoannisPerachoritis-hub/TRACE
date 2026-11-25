"""Evaluate FDR reduction through TRACE downstream pipeline.

Runs LD block detection, haplotype testing, and subsampling stability
on the worst-case FDR scenario (h2=0.8, 5 QTNs, LOCO kinship) to
empirically measure how each pipeline stage reduces false discovery rate.

Block-level classification: a block is TP if any QTN falls within
its [Start, End] boundaries; otherwise FP.

Usage:
    python benchmarks/simulation/evaluate_fdr_reduction.py [--n-reps 10] [--n-sub 20] [--n-jobs 1]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from annotation import canon_chr
from gwas.haplotype import run_haplotype_block_gwas
from gwas.kinship import (
    _compute_pcs_full_impl,
    _ld_prune_for_grm_by_chr_bp,
    _standardize_geno_for_grm,
)
from gwas.ld import find_ld_clusters_genomewide
from gwas.subsampling import subsample_gwas_resampling

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

SIM_DIR = Path(__file__).resolve().parent / "sim_data"
OUT_DIR = Path(__file__).resolve().parent / "sim_summary"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_shared_data():
    """Load genotype matrix, SNP map, and sample IDs (shared across reps)."""
    geno = np.load(SIM_DIR / "geno_matrix.npy")
    sample_ids = np.load(SIM_DIR / "sample_ids.npy", allow_pickle=True).astype(str)
    snp_map = pd.read_csv(SIM_DIR / "snp_map.csv")
    return geno, sample_ids, snp_map


def precompute_grm_basis(geno, snp_map, n_pcs=2):
    """Compute LD-pruned Z matrix, PCs, and chromosome labels for GRM."""
    chroms = snp_map["Chr"].values.astype(str)
    positions = snp_map["Pos"].values.astype(int)

    Z = _standardize_geno_for_grm(geno)
    Z_pruned, keep_mask = _ld_prune_for_grm_by_chr_bp(
        chroms, positions, Z,
        r2_thresh=0.2, window_bp=500_000, step_bp=100_000,
        return_mask=True,
    )
    chroms_grm = chroms[keep_mask]
    pcs, _ = _compute_pcs_full_impl(Z_pruned, max_pcs=max(n_pcs, 5))
    return Z_pruned, chroms_grm, pcs


def classify_blocks(ld_blocks, truth, window_kb=500):
    """Classify each LD block as TP or FP.

    A block is TP if any QTN falls within *window_kb* of the block
    boundaries [Start - window, End + window].  This matches the
    simulation's TP classification window and accounts for the fact
    that LD blocks are narrow (~300-600 kb) and may not physically
    contain the causal SNP even when they tag it.
    """
    qtn_chrs = [str(c) for c in truth["qtn_chrs"]]
    qtn_pos = truth["qtn_positions"]
    window_bp = window_kb * 1000

    is_tp = []
    for _, block in ld_blocks.iterrows():
        block_chr = str(block["Chr"])
        start = int(block["Start (bp)"])
        end = int(block["End (bp)"])
        tp = False
        for qc, qp in zip(qtn_chrs, qtn_pos):
            if (canon_chr(qc) == canon_chr(block_chr)
                    and (start - window_bp) <= qp <= (end + window_bp)):
                tp = True
                break
        is_tp.append(tp)
    return np.array(is_tp, dtype=bool)


def build_cascade(ld_blocks, is_tp, hap_results, discovery_df):
    """Build the FDR cascade through pipeline stages.

    Returns list of dicts with stage, n_blocks, n_tp, n_fp, fdr.
    """
    rows = []
    n = len(ld_blocks)
    n_tp = int(is_tp.sum())
    n_fp = n - n_tp
    fdr = n_fp / n if n > 0 else np.nan
    rows.append({"stage": "gwas_blocks", "n_blocks": n, "n_tp": n_tp,
                 "n_fp": n_fp, "fdr": fdr})

    # --- Haplotype filter: BH-FDR < 0.05 ---
    hap_mask = np.zeros(n, dtype=bool)
    if hap_results is not None and len(hap_results) > 0:
        # Match blocks by lead_snp, using BH-corrected p-value
        for i, (_, block) in enumerate(ld_blocks.iterrows()):
            lead = str(block.get("lead_snp", ""))
            match = hap_results[hap_results["lead_snp"].astype(str) == lead]
            if len(match) > 0:
                fdr_bh = match.iloc[0].get("FDR_BH", np.nan)
                if pd.notna(fdr_bh) and fdr_bh < 0.05:
                    hap_mask[i] = True

    n_h = int(hap_mask.sum())
    n_tp_h = int((is_tp & hap_mask).sum())
    n_fp_h = n_h - n_tp_h
    fdr_h = n_fp_h / n_h if n_h > 0 else np.nan
    rows.append({"stage": "haplotype_filtered", "n_blocks": n_h,
                 "n_tp": n_tp_h, "n_fp": n_fp_h, "fdr": fdr_h})

    # --- Subsampling filter: seed SNP DiscoveryFreq > 0.50 ---
    sub_mask = np.zeros(n, dtype=bool)
    if discovery_df is not None and len(discovery_df) > 0:
        freq_map = dict(zip(
            discovery_df["SNP"].astype(str),
            discovery_df["DiscoveryFreq"].values,
        ))
        for i, (_, block) in enumerate(ld_blocks.iterrows()):
            lead = str(block.get("lead_snp", ""))
            freq = freq_map.get(lead, 0.0)
            if freq > 0.50:
                sub_mask[i] = True

    n_s = int(sub_mask.sum())
    n_tp_s = int((is_tp & sub_mask).sum())
    n_fp_s = n_s - n_tp_s
    fdr_s = n_fp_s / n_s if n_s > 0 else np.nan
    rows.append({"stage": "subsampling_filtered", "n_blocks": n_s,
                 "n_tp": n_tp_s, "n_fp": n_fp_s, "fdr": fdr_s})

    # --- Both filters ---
    both_mask = hap_mask & sub_mask
    n_b = int(both_mask.sum())
    n_tp_b = int((is_tp & both_mask).sum())
    n_fp_b = n_b - n_tp_b
    fdr_b = n_fp_b / n_b if n_b > 0 else np.nan
    rows.append({"stage": "both_filtered", "n_blocks": n_b,
                 "n_tp": n_tp_b, "n_fp": n_fp_b, "fdr": fdr_b})

    return rows


# ------------------------------------------------------------------
# Per-replicate processing
# ------------------------------------------------------------------

def process_replicate(
    rep_idx,
    geno, sample_ids, snp_map,
    Z_pruned, chroms_grm, pcs,
    n_pcs, n_sub, n_jobs_sub,
    scenario="h2_080_q005",
):
    """Process one simulation replicate through the full cascade."""
    rep_dir = SIM_DIR / scenario / f"rep_{rep_idx:03d}"
    log.info(f"Processing rep {rep_idx}...")

    # Load replicate data
    with open(rep_dir / "truth.json") as f:
        truth = json.load(f)
    gwas_df = pd.read_csv(rep_dir / "platform_loco" / "results.csv")
    pheno_raw = pd.read_csv(rep_dir / "phenotype.csv")

    chroms = snp_map["Chr"].values.astype(str)
    positions = snp_map["Pos"].values.astype(int)
    sid = snp_map["SNP_ID"].values.astype(str)
    n_snps = len(snp_map)
    bonf_thresh = 0.05 / n_snps

    # Ensure GWAS df has correct column types
    gwas_df["Chr"] = gwas_df["Chr"].astype(str)

    # ---- 1. LD Block Detection ----
    # Pre-cluster Bonferroni-significant SNPs into loci so that we
    # get a manageable number of seed SNPs (~20-50 loci instead of
    # ~1600 individual SNPs).  For each locus, pick the best-p SNP
    # as the seed.  This mirrors real pipeline usage where users
    # inspect top loci, not individual SNPs.
    t0 = time.perf_counter()
    sig_df = gwas_df[gwas_df["PValue"] < bonf_thresh].copy()
    sig_df["Chr"] = sig_df["Chr"].astype(str).map(canon_chr)
    sig_df = sig_df.sort_values(["Chr", "Pos"])

    # Cluster within 500 kb on same chromosome
    leads = []
    if len(sig_df) > 0:
        cur_chr = sig_df.iloc[0]["Chr"]
        cur_end = sig_df.iloc[0]["Pos"]
        cluster = [sig_df.iloc[0]]
        for _, row in sig_df.iloc[1:].iterrows():
            if row["Chr"] == cur_chr and row["Pos"] - cur_end <= 500_000:
                cur_end = max(cur_end, row["Pos"])
                cluster.append(row)
            else:
                best = min(cluster, key=lambda r: r["PValue"])
                leads.append(best)
                cur_chr = row["Chr"]
                cur_end = row["Pos"]
                cluster = [row]
        best = min(cluster, key=lambda r: r["PValue"])
        leads.append(best)

    if not leads:
        log.warning(f"  Rep {rep_idx}: no significant SNPs")
        return [], [], None

    # Build a filtered GWAS df with only lead SNPs so find_ld_clusters
    # seeds from ~20-50 loci rather than ~1600 individual SNPs
    lead_df = pd.DataFrame(leads)
    log.info(f"  Rep {rep_idx}: {len(sig_df)} Bonf-sig SNPs clustered into {len(lead_df)} loci")

    ld_blocks = find_ld_clusters_genomewide(
        gwas_df=lead_df,
        chroms=chroms,
        positions=positions,
        geno_imputed=geno,
        sid=sid,
        ld_threshold=0.6,
        flank_kb=300,
        min_snps=3,
        sig_thresh=1.0,  # all rows in lead_df are seeds (already filtered)
        top_n=0,
    )
    t_ld = time.perf_counter() - t0

    if ld_blocks is None or len(ld_blocks) == 0:
        log.warning(f"  Rep {rep_idx}: no LD blocks formed")
        return [], {}, None

    # Classify blocks
    is_tp = classify_blocks(ld_blocks, truth)
    log.info(f"  Rep {rep_idx}: {len(ld_blocks)} blocks ({is_tp.sum()} TP, "
             f"{(~is_tp).sum()} FP) in {t_ld:.1f}s")

    # ---- 2. Haplotype Testing ----
    t0 = time.perf_counter()

    # Build geno_df and pheno_df for haplotype function
    geno_df = pd.DataFrame(geno, index=sample_ids.astype(str), columns=sid)
    pheno_df = pheno_raw.set_index("SampleID")
    pheno_df.index = pheno_df.index.astype(str)

    hap_results = None
    try:
        hap_results, _ = run_haplotype_block_gwas(
            haplo_df=ld_blocks,
            chroms=chroms,
            positions=positions,
            geno_imputed=geno,
            sid=sid,
            geno_df=geno_df,
            pheno_df=pheno_df,
            trait_col="SimTrait",
            pcs=pcs[:, :n_pcs],
            n_perm=1000,
            min_hap_count=5,
            min_group_size=3,
            n_pcs_used=n_pcs,
        )
    except Exception as e:
        log.warning(f"  Rep {rep_idx}: haplotype test failed: {e}")

    t_hap = time.perf_counter() - t0
    log.info(f"  Rep {rep_idx}: haplotype testing done in {t_hap:.1f}s")

    # ---- 3. Subsampling ----
    t0 = time.perf_counter()

    # Build chroms_num for subsampling
    unique_chrs = sorted(set(chroms), key=lambda c: int(canon_chr(c) or 0) if canon_chr(c) and canon_chr(c).isdigit() else 99)
    chr_to_num = {c: i + 1 for i, c in enumerate(unique_chrs)}
    chroms_num = np.array([chr_to_num.get(c, 0) for c in chroms], dtype=int)

    iid_2d = np.c_[sample_ids, sample_ids].astype(str)
    y = pheno_df.reindex(sample_ids.astype(str))["SimTrait"].values.astype(np.float32)

    discovery_df = None
    try:
        discovery_df, _raw_pvals, _meta = subsample_gwas_resampling(
            geno_imputed=geno,
            y=y,
            sid=sid,
            chroms=chroms,
            chroms_num=chroms_num,
            positions=positions,
            iid=iid_2d,
            Z_for_grm=Z_pruned,
            n_pcs=n_pcs,
            n_reps=n_sub,
            sample_frac=0.80,
            discovery_thresh=bonf_thresh,
            seed=rep_idx * 1000,
            use_loco=True,
            chroms_grm=chroms_grm,
            n_jobs=n_jobs_sub,
        )
    except Exception as e:
        log.warning(f"  Rep {rep_idx}: subsampling failed: {e}")

    t_sub = time.perf_counter() - t0
    log.info(f"  Rep {rep_idx}: subsampling done in {t_sub:.1f}s")

    # ---- 4. Build cascade ----
    cascade = build_cascade(ld_blocks, is_tp, hap_results, discovery_df)
    for row in cascade:
        row["rep"] = rep_idx

    # Collect per-block detail for figure
    block_detail = []
    for i, (_, block) in enumerate(ld_blocks.iterrows()):
        lead = str(block.get("lead_snp", ""))
        freq = 0.0
        if discovery_df is not None and len(discovery_df) > 0:
            freq_match = discovery_df[discovery_df["SNP"].astype(str) == lead]
            if len(freq_match) > 0:
                freq = float(freq_match.iloc[0]["DiscoveryFreq"])

        hap_p = np.nan
        hap_fdr = np.nan
        if hap_results is not None and len(hap_results) > 0:
            match = hap_results[hap_results["lead_snp"].astype(str) == lead]
            if len(match) > 0:
                hap_p = float(match.iloc[0].get("P_perm", match.iloc[0].get("PValue", np.nan)))
                hap_fdr = float(match.iloc[0].get("FDR_BH", np.nan))

        block_detail.append({
            "rep": rep_idx,
            "lead_snp": lead,
            "chr": str(block["Chr"]),
            "start": int(block["Start (bp)"]),
            "end": int(block["End (bp)"]),
            "is_tp": bool(is_tp[i]),
            "hap_p_perm": hap_p,
            "hap_fdr_bh": hap_fdr,
            "subsampling_freq": freq,
        })

    return cascade, block_detail, discovery_df


# ------------------------------------------------------------------
# Summary and plotting
# ------------------------------------------------------------------

def make_summary(all_cascade):
    """Aggregate cascade across replicates."""
    df = pd.DataFrame(all_cascade)
    summary = df.groupby("stage").agg(
        median_blocks=("n_blocks", "median"),
        median_tp=("n_tp", "median"),
        median_fp=("n_fp", "median"),
        median_fdr=("fdr", "median"),
        q25_fdr=("fdr", lambda x: np.nanpercentile(x, 25)),
        q75_fdr=("fdr", lambda x: np.nanpercentile(x, 75)),
        mean_fdr=("fdr", "mean"),
    ).reindex(["gwas_blocks", "haplotype_filtered",
               "subsampling_filtered", "both_filtered"])
    return summary


def plot_results(all_cascade, all_block_detail, scenario="h2_080_q005"):
    """Create two-panel figure: FDR cascade + TP vs FP discovery frequency."""
    from utils.pub_theme import apply_matplotlib_theme
    apply_matplotlib_theme()
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel A: FDR cascade boxplot ---
    cascade_df = pd.DataFrame(all_cascade)
    stage_order = ["gwas_blocks", "haplotype_filtered",
                   "subsampling_filtered", "both_filtered"]
    stage_labels = ["GWAS\nBlocks", "+ Haplotype\n(FDR<0.05)",
                    "+ Subsampling\n(>50%)", "+ Both\nFilters"]

    data_a = [cascade_df[cascade_df["stage"] == s]["fdr"].dropna().values
              for s in stage_order]

    bp = ax1.boxplot(data_a, labels=stage_labels, patch_artist=True,
                     widths=0.5, medianprops=dict(color="black", linewidth=1.5))

    colors = ["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_ylabel("Block-Level FDR")
    ax1.set_title("A) FDR Reduction Through Pipeline Stages")
    ax1.set_ylim(-0.05, 1.05)
    ax1.axhline(0.05, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)

    # --- Panel B: TP vs FP discovery frequency ---
    detail_df = pd.DataFrame(all_block_detail)
    if len(detail_df) > 0 and "subsampling_freq" in detail_df.columns:
        tp_freq = detail_df[detail_df["is_tp"]]["subsampling_freq"].values
        fp_freq = detail_df[~detail_df["is_tp"]]["subsampling_freq"].values

        data_b = [tp_freq, fp_freq]
        bp2 = ax2.boxplot(data_b, labels=["TP Blocks", "FP Blocks"],
                          patch_artist=True, widths=0.4,
                          medianprops=dict(color="black", linewidth=1.5))
        bp2["boxes"][0].set_facecolor("#5cb85c")
        bp2["boxes"][0].set_alpha(0.7)
        bp2["boxes"][1].set_facecolor("#d9534f")
        bp2["boxes"][1].set_alpha(0.7)

        ax2.axhline(0.50, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
        ax2.set_ylabel("Subsampling Discovery Frequency")
        ax2.set_title("B) Seed SNP Stability: TP vs FP Blocks")
        ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fig.savefig(OUT_DIR / f"figure_s7_fdr_reduction_{scenario}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"figure_s7_fdr_reduction_{scenario}.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figure saved to {OUT_DIR / f'figure_s7_fdr_reduction_{scenario}.png'}")


def plot_combined_subsampling():
    """Create 2x2 figure showing only TP vs FP subsampling frequency.

    Reads pre-computed block detail CSVs — no computation needed.
    """
    from scipy.stats import mannwhitneyu
    from utils.pub_theme import apply_matplotlib_theme
    apply_matplotlib_theme()
    import matplotlib.pyplot as plt

    scenarios = VIABLE_SCENARIOS
    labels = [
        r"(A)  $h^2$=0.5, 5 QTNs",
        r"(B)  $h^2$=0.8, 5 QTNs",
        r"(C)  $h^2$=0.8, 15 QTNs",
        r"(D)  $h^2$=0.8, 50 QTNs",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, (scenario, label) in enumerate(zip(scenarios, labels)):
        ax = axes[idx]
        csv_path = OUT_DIR / f"fdr_reduction_blocks_{scenario}.csv"
        if not csv_path.exists():
            ax.set_title(f"{label}\n(no data)")
            continue

        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["subsampling_freq"])
        # Deduplicate: one entry per (rep, lead_snp) — multiple LD
        # sub-blocks share the same lead SNP, freq, and TP status.
        df = df.drop_duplicates(subset=["rep", "lead_snp"])

        tp_freq = df[df["is_tp"]]["subsampling_freq"].values
        fp_freq = df[~df["is_tp"]]["subsampling_freq"].values

        bp = ax.boxplot(
            [tp_freq, fp_freq],
            tick_labels=["TP Blocks", "FP Blocks"],
            patch_artist=True, widths=0.4,
            medianprops=dict(color="black", linewidth=1.5),
        )
        bp["boxes"][0].set_facecolor("#5cb85c")
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor("#d9534f")
        bp["boxes"][1].set_alpha(0.7)

        ax.axhline(0.50, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Discovery Frequency")
        ax.set_title(label)

        # Mann-Whitney test + pass rate gap
        if len(tp_freq) > 0 and len(fp_freq) > 0:
            stat, p = mannwhitneyu(tp_freq, fp_freq, alternative="two-sided")
            tp_pass = (tp_freq > 0.50).mean() * 100
            fp_pass = (fp_freq > 0.50).mean() * 100
            gap = tp_pass - fp_pass
            p_str = "< 0.0001" if p < 0.0001 else f"= {p:.4f}"
            ax.text(
                0.50, 0.02, f"MW p {p_str}\nGap: {gap:+.1f} pp",
                transform=ax.transAxes, fontsize=8,
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="grey", alpha=0.8),
            )
            # Sample sizes
            ax.text(
                0.50, 0.95,
                f"n(TP)={len(tp_freq)}, n(FP)={len(fp_freq)}",
                transform=ax.transAxes, fontsize=7,
                ha="center", va="top", color="grey",
            )

    plt.tight_layout()
    out_png = OUT_DIR / "figure_s7_fdr_reduction_combined.png"
    out_pdf = OUT_DIR / "figure_s7_fdr_reduction_combined.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Combined figure saved to {out_png}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run_scenario(scenario, geno, sample_ids, snp_map, Z_pruned, chroms_grm,
                 pcs, n_pcs, n_reps, n_sub, n_jobs):
    """Run the full FDR reduction benchmark for one scenario."""
    log.info(f"\n{'='*60}")
    log.info(f"Scenario: {scenario}, {n_reps} reps, {n_sub} subsamples")
    log.info(f"{'='*60}")

    # Check how many reps actually have results
    scenario_dir = SIM_DIR / scenario
    available_reps = sorted([
        int(d.name.split("_")[1])
        for d in scenario_dir.iterdir()
        if d.is_dir() and d.name.startswith("rep_")
        and (d / "platform_loco" / "results.csv").exists()
    ])
    n_reps = min(n_reps, len(available_reps))
    if n_reps == 0:
        log.warning(f"  No results found for {scenario}, skipping")
        return None

    all_cascade = []
    all_block_detail = []
    skipped = 0

    for rep_idx in available_reps[:n_reps]:
        t_rep = time.perf_counter()
        cascade, block_detail, _ = process_replicate(
            rep_idx, geno, sample_ids, snp_map,
            Z_pruned, chroms_grm, pcs,
            n_pcs, n_sub, n_jobs,
            scenario=scenario,
        )
        if not cascade:
            skipped += 1
            continue
        all_cascade.extend(cascade)
        all_block_detail.extend(block_detail)
        log.info(f"Rep {rep_idx} complete in {time.perf_counter()-t_rep:.1f}s")

    if not all_cascade:
        log.warning(f"  No blocks formed in any replicate for {scenario}")
        return None

    if skipped > 0:
        log.info(f"  Skipped {skipped} reps with no blocks")

    # Save raw results
    cascade_df = pd.DataFrame(all_cascade)
    cascade_df.to_csv(OUT_DIR / f"fdr_reduction_{scenario}.csv", index=False)

    detail_df = pd.DataFrame(all_block_detail)
    detail_df.to_csv(OUT_DIR / f"fdr_reduction_blocks_{scenario}.csv", index=False)

    # Summary table
    summary = make_summary(all_cascade)
    summary.to_csv(OUT_DIR / f"fdr_reduction_summary_{scenario}.csv")
    log.info(f"\nFDR Reduction Summary ({scenario}):\n{summary.to_string()}")

    # Plot
    try:
        plot_results(all_cascade, all_block_detail, scenario)
    except Exception as e:
        log.warning(f"Plotting failed for {scenario}: {e}")

    return {
        "scenario": scenario,
        "n_reps_used": n_reps - skipped,
        "cascade_df": cascade_df,
        "detail_df": detail_df,
        "summary": summary,
    }


# Scenarios with enough Bonferroni-significant SNPs to form LD blocks
VIABLE_SCENARIOS = [
    "h2_050_q005",   # moderate h2, oligogenic
    "h2_080_q005",   # high h2, oligogenic (worst-case FDR)
    "h2_080_q015",   # high h2, moderate QTNs
    "h2_080_q050",   # high h2, polygenic
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-reps", type=int, default=30,
                        help="Number of simulation replicates (default: 30)")
    parser.add_argument("--n-sub", type=int, default=20,
                        help="Subsampling iterations per replicate (default: 20)")
    parser.add_argument("--n-pcs", type=int, default=2,
                        help="Number of PCs (default: 2, matching simulation)")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel workers for subsampling (default: 1)")
    parser.add_argument("--scenario", type=str, default="all",
                        help="Scenario name or 'all' for viable scenarios (default: all)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only regenerate the combined figure from existing CSVs")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        log.info("Regenerating combined subsampling figure from existing data...")
        plot_combined_subsampling()
        log.info("Done.")
        return

    # Determine which scenarios to run
    if args.scenario == "all":
        scenarios = VIABLE_SCENARIOS
    else:
        scenarios = [args.scenario]

    log.info(f"Scenarios to run: {scenarios}")

    # Load shared data (once)
    log.info("Loading shared data...")
    geno, sample_ids, snp_map = load_shared_data()

    log.info("Precomputing GRM basis (LD pruning + PCA)...")
    t0 = time.perf_counter()
    Z_pruned, chroms_grm, pcs = precompute_grm_basis(geno, snp_map, args.n_pcs)
    log.info(f"GRM basis ready: {Z_pruned.shape[1]} pruned SNPs in {time.perf_counter()-t0:.1f}s")

    # Run each scenario
    all_results = {}
    for scenario in scenarios:
        result = run_scenario(
            scenario, geno, sample_ids, snp_map,
            Z_pruned, chroms_grm, pcs,
            args.n_pcs, args.n_reps, args.n_sub, args.n_jobs,
        )
        if result is not None:
            all_results[scenario] = result

    # Combined cross-scenario summary
    if len(all_results) > 1:
        log.info(f"\n{'='*60}")
        log.info("CROSS-SCENARIO SUMMARY")
        log.info(f"{'='*60}")
        for sc, res in all_results.items():
            s = res["summary"]
            gwas_fdr = s.loc["gwas_blocks", "median_fdr"] if "gwas_blocks" in s.index else np.nan
            sub_fdr = s.loc["subsampling_filtered", "median_fdr"] if "subsampling_filtered" in s.index else np.nan
            both_fdr = s.loc["both_filtered", "median_fdr"] if "both_filtered" in s.index else np.nan
            log.info(f"  {sc}: GWAS FDR={gwas_fdr:.3f} -> Subsamp={sub_fdr:.3f} -> Both={both_fdr:.3f}")

    log.info("\nAll done.")


if __name__ == "__main__":
    main()
