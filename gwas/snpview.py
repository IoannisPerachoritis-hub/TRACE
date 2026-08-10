"""T-21 / T-23 / T-85 — the per-SNP view layer (effects, gene evidence, plot budget).

Display/evidence helpers over the significant-SNP table. Pure or near-pure and
headlessly testable. Like the rescue, this module must never call the block
detector — it may read r² (``compute_r2_to_lead``) and η²
(``anova_eta_sq_from_labels``) but not ``find_ld_clusters_genomewide`` /
``find_ld_blocks_graph`` / ``find_ld_blocks_from_genotypes`` /
``filter_contained_blocks`` (integration §1.7).

Spec: post_gwas_visibility_design.md §2.2-2.4; integration_and_testing.md §3.2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from annotation import canon_chr

EFFECT_FLAGS = ("sign-flip", "structure-sensitive", "concordant", "undetermined")


def effect_flag(beta_mlm, beta_ols, *, ols_floor=1e-6):
    """Raw-vs-adjusted effect flag (T-21 §2.3). Computed only when both β are
    finite and |β_OLS| exceeds a small floor; never a bare ratio."""
    bm = float(beta_mlm) if beta_mlm is not None else np.nan
    bo = float(beta_ols) if beta_ols is not None else np.nan
    if not np.isfinite(bm) or not np.isfinite(bo) or abs(bo) <= ols_floor:
        return "undetermined"
    if np.sign(bm) != np.sign(bo):
        return "sign-flip"          # strongest structure warning
    if abs(bm / bo) < 0.5:
        return "structure-sensitive"
    return "concordant"


def genotype_class_summary(geno_col, y):
    """Per-genotype-class counts/means + per-SNP η² for the boxplot (T-21 §2.2).

    Classes are hard calls (0/1/2) from the raw dosage; missing are counted, not
    dropped; every class is reported including n=0. η² reuses
    ``anova_eta_sq_from_labels`` so it is directly comparable to Table S8's block η².
    """
    from gwas.ld import anova_eta_sq_from_labels

    g = np.asarray(geno_col, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(g) & np.isfinite(y)
    classes = {}
    for cls in (0, 1, 2):
        m = valid & (np.rint(g) == cls)
        classes[cls] = {"n": int(m.sum()),
                        "mean": float(np.mean(y[m])) if m.any() else np.nan,
                        "draw_box": int(m.sum()) >= 5}      # box only when class n >= 5
    n_missing = int((~np.isfinite(g)).sum())
    labels = np.rint(g[valid]).astype(int)
    eta2, n_groups, n_used = anova_eta_sq_from_labels(y[valid], labels)
    return {"classes": classes, "n_missing": n_missing,
            "eta2": float(eta2), "n_groups": int(n_groups), "n_used": int(n_used)}


def ld_supported_interval(lead_snp_id, geno_dosage_raw, sid, chroms, positions,
                          *, r2_thresh=0.6, flank_bp=300_000, min_pair_n=15):
    """Span of markers with r² ≥ ``r2_thresh`` to the lead SNP within its flank
    window (T-23). Returns ``(start, end)`` bp, or ``None`` if the lead is absent /
    r² is uncomputable. r² is genotype-derived, so this interval is immune to the
    assembly-coordinate problem (design §2.4)."""
    from gwas.plotting import compute_r2_to_lead

    sid = np.asarray(sid).astype(str)
    positions = np.asarray(positions, dtype=np.int64)
    chroms_c = np.array([canon_chr(c) for c in np.asarray(chroms)], dtype=object)
    matches = np.where(sid == str(lead_snp_id))[0]
    if len(matches) == 0:
        return None
    li = int(matches[0])
    lc, lpos = chroms_c[li], int(positions[li])
    win_mask = (chroms_c == lc) & (positions >= lpos - flank_bp) & (positions <= lpos + flank_bp)
    try:
        r2, _lead_idx = compute_r2_to_lead(geno_dosage_raw, sid, str(lead_snp_id), win_mask, min_pair_n)
    except ValueError:
        return None
    win_idx = np.where(win_mask)[0]
    supported = win_idx[np.nan_to_num(np.asarray(r2, float), nan=0.0) >= r2_thresh]
    if len(supported) == 0:
        return (lpos, lpos)     # only the lead supports itself
    sp = positions[supported]
    return (int(min(int(sp.min()), lpos)), int(max(int(sp.max()), lpos)))


def label_genes_by_ld(genes_df, ld_interval):
    """Label each gene ``ld_supported`` (overlaps the LD-supported interval) or
    ``distance_only`` (T-23). Nothing is filtered — labels are added. When the
    interval is ``None`` (e.g. an isolated SNP) every gene is ``distance_only``."""
    labels = {}
    if genes_df is None or len(genes_df) == 0:
        return labels
    if ld_interval is None:
        return {str(g): "distance_only" for g in genes_df["Gene_ID"].astype(str)}
    s, e = ld_interval
    for _, g in genes_df.iterrows():
        gid = str(g["Gene_ID"])
        overlaps = not (int(g["End"]) < s or int(g["Start"]) > e)
        labels[gid] = "ld_supported" if overlaps else "distance_only"
    return labels


def collapse_snps_for_plotting(sig_df, geno_dosage_raw, sid,
                               *, r2_threshold=0.9, flank_bp=300_000, min_pair_n=15):
    """Greedy collapse of near-redundant SNPs to a representative — **plots only**
    (T-85). Never filters or reorders rows: returns one row per input SNP with
    ``SNP, Representative_SNP, Representative_Of_Count, r2_to_representative``.
    Reuses the ``_thin_block_snps`` r²=0.9 threshold and the flank_kb=300 scale so
    "redundant" means one thing across the code."""
    from gwas.plotting import compute_r2_to_lead

    sid_arr = np.asarray(sid).astype(str)
    positions = None
    # rank order: PValue asc, then Chr(canon), Pos, SNP (total, deterministic)
    order = sig_df.assign(_c=sig_df["Chr"].map(lambda c: (0, int(canon_chr(c))) if str(canon_chr(c)).isdigit()
                                               else (1, str(canon_chr(c))))) \
                  .sort_values(["PValue", "_c", "Pos", "SNP"], kind="stable").index
    rep_of = {}          # snp -> representative
    rep_r2 = {}          # snp -> r2 to its representative
    reps = []            # chosen representatives in order
    pos_by_snp = dict(zip(sig_df["SNP"].astype(str), sig_df["Pos"].astype(int)))
    chr_by_snp = dict(zip(sig_df["SNP"].astype(str), sig_df["Chr"].map(canon_chr)))

    for idx in order:
        snp = str(sig_df.at[idx, "SNP"])
        if snp in rep_of:
            continue
        rep_of[snp] = snp
        rep_r2[snp] = 1.0
        reps.append(snp)
        if geno_dosage_raw is None or snp not in set(sid_arr):
            continue
        # find unassigned sig SNPs within flank on same chr with r2 >= threshold
        cc, cpos = chr_by_snp[snp], pos_by_snp[snp]
        cand = [str(s) for s in sig_df["SNP"].astype(str)
                if s not in rep_of and chr_by_snp.get(str(s)) == cc
                and abs(pos_by_snp.get(str(s), cpos) - cpos) <= flank_bp and str(s) in set(sid_arr)]
        if not cand:
            continue
        mask = np.isin(sid_arr, cand)
        try:
            r2, _ = compute_r2_to_lead(geno_dosage_raw, sid_arr, snp, mask, min_pair_n)
        except ValueError:
            continue
        for s, val in zip(sid_arr[mask], np.asarray(r2, float)):
            if np.isfinite(val) and val >= r2_threshold and str(s) not in rep_of:
                rep_of[str(s)] = snp
                rep_r2[str(s)] = float(val)

    counts = {}
    for s, rep in rep_of.items():
        counts[rep] = counts.get(rep, 0) + 1
    out = pd.DataFrame({
        "SNP": sig_df["SNP"].astype(str).values,
        "Representative_SNP": [rep_of.get(str(s), str(s)) for s in sig_df["SNP"]],
        "r2_to_representative": [rep_r2.get(str(s), 1.0) for s in sig_df["SNP"]],
    })
    out["Representative_Of_Count"] = out["Representative_SNP"].map(
        lambda r: counts.get(r, 1)).astype(int)
    return out
