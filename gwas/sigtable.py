"""T-20 / T-23 — the significant-SNP table (the no-omission guarantee).

Every SNP passing the **reporting** threshold appears as one row — never the
seeding threshold, never truncated — marked ``in_block`` or one of four
``unblocked_*`` states, with its candidate interval, gene count, and gene-evidence
label. This is the table that surfaces the SNPs the post-GWAS view silently
dropped (23 of 44 on the tomato demo).

It is a **presentation layer over C1's rescue**: it calls
``gwas.isolated.run_isolated_snp_rescue`` for the unblocked rows and reads the
emitted block table for the in-block rows. It never re-runs the block detector
(``gwas.isolated`` has no ``gwas.ld`` import; this module only reads block
coordinates and calls the public annotator).

Coverage authority: a SNP is ``in_block`` iff it is **positionally covered**
(``Start ≤ Pos ≤ End``) by an emitted block — the same rule as the census and the
rescue, so the in-block / unblocked split equals the measured 23/21. (The T-77
spec also describes an ``SNP_IDs``-membership rule; where the two differ,
``sigtable_membership_divergence`` records it rather than letting them disagree
silently.)

Spec: post_gwas_visibility_design.md §2.1; integration_and_testing.md §2.1 (T-77).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from annotation import canon_chr
from gwas import isolated as _iso

# 38-column normative order (integration §2.1).
SIG_TABLE_COLUMNS = [
    "SNP", "Chr", "Pos", "PValue", "-log10p",
    "Beta_MLM", "SE_MLM", "Beta_OLS", "SE_OLS", "Effect_Source",
    "MAF", "ImputationRate", "Sig_Rule", "Sig_Threshold",
    "Seed_Threshold", "Passes_Seed_Threshold", "Passes_Meff", "Passes_Bonf", "Passes_FDR", "FDR",
    "Block_ID", "Block_Status", "r2_to_block_lead", "N_SNPs_in_Block", "Max_r2_in_Window",
    "Nearest_Typed_Upstream_SNP", "Nearest_Typed_Upstream_bp",
    "Nearest_Typed_Downstream_SNP", "Nearest_Typed_Downstream_bp",
    "Interval_Start_bp", "Interval_End_bp", "Interval_Bounded_By",
    "N_Genes_Interval", "Candidate_Genes", "Gene_Evidence", "Genome_Build",
    "Plot_Rendered", "Withheld_Reason",
]

BLOCK_STATUS_VALUES = (
    "in_block", "unblocked_not_seeded", "unblocked_isolated",
    "unblocked_monomorphic_window", "unblocked_low_ld",
)
GENE_EVIDENCE_VALUES = (
    "no_annotation_loaded", "no_gene_within_interval", "overlapping_interval",
    "flanking_within_500kb", "custom_gene_model",
)

# annotate_ld_blocks / annotate_isolated_intervals annotation_status -> Gene_Evidence
_STATUS_TO_EVIDENCE = {
    "overlapping": "overlapping_interval",
    "intergenic": "flanking_within_500kb",
    "no_genes_nearby": "no_gene_within_interval",
}

# Unblocked_SNPs projection = these columns (1-5, 13-16, 22, 25-36) of the table.
UNBLOCKED_PROJECTION = [
    "SNP", "Chr", "Pos", "PValue", "-log10p",
    "Sig_Rule", "Sig_Threshold", "Seed_Threshold", "Passes_Seed_Threshold",
    "Block_Status", "Max_r2_in_Window",
    "Nearest_Typed_Upstream_SNP", "Nearest_Typed_Upstream_bp",
    "Nearest_Typed_Downstream_SNP", "Nearest_Typed_Downstream_bp",
    "Interval_Start_bp", "Interval_End_bp", "Interval_Bounded_By",
    "N_Genes_Interval", "Candidate_Genes", "Gene_Evidence", "Genome_Build",
]


def _empty_table():
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in SIG_TABLE_COLUMNS})
    return df


def _flank_gene_ids(row):
    ids = []
    for direction in ("upstream", "downstream"):
        for i in (1, 2):
            g = str(row.get(f"{direction}_gene_{i}", "") or "")
            if g:
                ids.append(g)
    return ids


def build_significant_snp_table(
    gwas_df,
    blocks_df,
    sig_rule,
    chroms,
    positions,
    sid,
    geno_dosage_raw=None,
    genes=None,
    *,
    seed_p_used,
    top_n_used=0,
    edge_flank_bp,
    max_interval_bp=5_000_000,
    low_res_bp=None,
    ld_decay_bp=None,
    genome_build="SL3",
    species="tomato",
    geno_encoding="dosage012",
):
    """Build the 38-column significant-SNP table. Pure; does not mutate inputs."""
    from annotation import annotate_ld_blocks

    sig_mask = sig_rule.significant_mask(gwas_df)
    sig = gwas_df.loc[sig_mask].copy()
    if sig.empty:
        return _empty_table()

    # reduced 4-column GUI-fallback frame => effects are unavailable, never silently dropped
    has_mlm = "Beta_MLM" in gwas_df.columns
    has_ols = "Beta_OLS" in gwas_df.columns
    mlm_all_nan = has_mlm and not np.isfinite(pd.to_numeric(gwas_df["Beta_MLM"], errors="coerce")).any()
    reduced = not has_mlm and not has_ols

    # ── C1 rescue for the unblocked (positionally uncovered) rows ──
    rescue = _iso.run_isolated_snp_rescue(
        gwas_df, blocks_df, sig_rule, chroms, positions, sid, genes=genes,
        seed_p_used=seed_p_used, top_n_used=top_n_used,
        edge_flank_bp=edge_flank_bp, max_interval_bp=max_interval_bp,
        low_res_bp=low_res_bp, ld_decay_bp=ld_decay_bp,
    )
    unblocked_path = {str(r["snp_id"]): str(r["omission_path"]) for _, r in rescue.unblocked.iterrows()}
    # map each unblocked SNP -> the interval that contains it (SNP_IDs membership)
    snp_to_interval = {}
    for _, iv in rescue.intervals.iterrows():
        for s in str(iv.get("SNP_IDs", "")).split(","):
            if s:
                snp_to_interval[s] = iv

    # ── in-block annotation (positional blocks, annotated once) ──
    block_rows = []
    if blocks_df is not None and len(blocks_df) > 0:
        sc = "Start (bp)" if "Start (bp)" in blocks_df.columns else "Start"
        ec = "End (bp)" if "End (bp)" in blocks_df.columns else "End"
        for _, b in blocks_df.iterrows():
            block_rows.append((canon_chr(b["Chr"]), int(b[sc]), int(b[ec]),
                               str(b.get("lead_snp", "")),
                               set(filter(None, str(b.get("SNP_IDs", "")).split(",")))))
    block_ann = {}
    if genes is not None and blocks_df is not None and len(blocks_df) > 0:
        ann = annotate_ld_blocks(blocks_df.copy(), genes, n_flank=2, max_flank_dist_bp=500_000)
        for _, a in ann.iterrows():
            key = (canon_chr(a["Chr"]), int(a["Start (bp)"]), int(a["End (bp)"]))
            block_ann[key] = a

    # ── MAF per SNP ──
    maf_map = {}
    if geno_dosage_raw is not None:
        try:
            from gwas.ld import maf_from_matrix
            maf_vec = maf_from_matrix(np.asarray(geno_dosage_raw, float), geno_encoding)
            maf_map = {str(s): float(m) for s, m in zip(np.asarray(sid).astype(str), maf_vec)}
        except Exception:
            maf_map = {}

    # ── per-chromosome sorted marker axis for nearest-typed lookups + window facts ──
    chroms_c = np.array([canon_chr(c) for c in np.asarray(chroms)], dtype=object)
    positions = np.asarray(positions, dtype=np.int64)
    sid_arr = np.asarray(sid).astype(str)
    by_chr = {}
    for cc in np.unique(chroms_c):
        m = chroms_c == cc
        order = np.argsort(positions[m], kind="stable")
        cols = np.where(m)[0][order]
        by_chr[cc] = (positions[m][order], sid_arr[m][order], cols)

    def _nearest(cc, pos):
        if cc not in by_chr:
            return ("", np.nan, "", np.nan)
        mpos, msid, _ = by_chr[cc]
        i = int(np.searchsorted(mpos, pos, "left"))
        up = (msid[i - 1], int(pos - mpos[i - 1])) if i - 1 >= 0 and mpos[i - 1] < pos else ("", np.nan)
        # first strictly-greater
        j = int(np.searchsorted(mpos, pos, "right"))
        dn = (msid[j], int(mpos[j] - pos)) if j < len(mpos) and mpos[j] > pos else ("", np.nan)
        return (up[0], up[1], dn[0], dn[1])

    def _window_facts(cc, pos):
        if cc not in by_chr:
            return 0, 0
        mpos, _, cols = by_chr[cc]
        lo = int(np.searchsorted(mpos, pos - edge_flank_bp, "left"))
        hi = int(np.searchsorted(mpos, pos + edge_flank_bp, "right"))
        n_window = hi - lo
        n_poly = n_window
        if geno_dosage_raw is not None and n_window > 0:
            g = np.asarray(geno_dosage_raw, float)[:, cols[lo:hi]]
            n_poly = int((np.nanvar(g, axis=0) > 0).sum())
        return n_window, n_poly

    rows = []
    for _, r in sig.iterrows():
        snp = str(r["SNP"])
        cc = canon_chr(r["Chr"])
        pos = int(r["Pos"])
        p = float(r["PValue"])

        # containing block (positional)
        containing = None
        for (bc, bs, be, lead, ids) in block_rows:
            if bc == cc and bs <= pos <= be:
                containing = (bc, bs, be, lead, ids)
                break

        beta_mlm = float(r["Beta_MLM"]) if has_mlm and pd.notna(r.get("Beta_MLM")) else np.nan
        beta_ols = float(r["Beta_OLS"]) if has_ols and pd.notna(r.get("Beta_OLS")) else np.nan
        se_mlm = float(r["SE_MLM"]) if "SE_MLM" in gwas_df.columns and pd.notna(r.get("SE_MLM")) else (
            float(r["SnpWeightSE"]) if "SnpWeightSE" in gwas_df.columns and pd.notna(r.get("SnpWeightSE")) else np.nan)
        se_ols = float(r["SE_OLS"]) if "SE_OLS" in gwas_df.columns and pd.notna(r.get("SE_OLS")) else np.nan
        if reduced:
            effect_source = "unavailable"
        elif not has_mlm or mlm_all_nan:
            effect_source = "OLS_only"
        elif not has_ols:
            effect_source = "MLM_only"
        else:
            effect_source = "both"

        up_snp, up_bp, dn_snp, dn_bp = _nearest(cc, pos)

        if containing is not None:
            block_status = "in_block"
            block_id = f"{cc}:{containing[1]}-{containing[2]}"
            n_in_block = len(containing[4]) if containing[4] else pd.NA
            interval_start, interval_end = containing[1], containing[2]
            bounded_by = "typed_snps"
            if genes is None:
                n_genes = pd.NA
                cand_genes = ""
                gene_ev = "no_annotation_loaded"
            else:
                a = block_ann.get((containing[0], containing[1], containing[2]))
                if a is None:
                    n_genes, cand_genes, gene_ev = pd.NA, "", "no_gene_within_interval"
                else:
                    n_genes = int(a.get("n_genes_overlapping", 0))
                    ov = [g for g in str(a.get("overlapping_genes", "")).split(";") if g]
                    cand_genes = ";".join(ov + _flank_gene_ids(a))
                    gene_ev = _STATUS_TO_EVIDENCE.get(str(a.get("annotation_status", "")), "no_gene_within_interval")
        else:
            path = unblocked_path.get(snp, "block_formation")
            if path == "seeding_threshold":
                block_status = "unblocked_not_seeded"
            else:
                n_window, n_poly = _window_facts(cc, pos)
                if n_window < 2:
                    block_status = "unblocked_isolated"
                elif n_poly < 2:
                    block_status = "unblocked_monomorphic_window"
                else:
                    block_status = "unblocked_low_ld"
            block_id = ""
            n_in_block = pd.NA
            iv = snp_to_interval.get(snp)
            if iv is not None:
                interval_start = int(iv["Start (bp)"])
                interval_end = int(iv["End (bp)"])
                bounded_by = {"": "typed_snps", "chr_start": "chromosome_start",
                              "chr_end": "chromosome_end", "chr_both": "chromosome_both"}.get(
                    str(iv.get("edge_flag", "")), "typed_snps")
                if str(iv.get("flank_up_snp_id", "")):
                    up_snp = str(iv["flank_up_snp_id"])
                if str(iv.get("flank_dn_snp_id", "")):
                    dn_snp = str(iv["flank_dn_snp_id"])
                if genes is None:
                    n_genes, cand_genes, gene_ev = pd.NA, "", "no_annotation_loaded"
                else:
                    ng = iv.get("n_genes_in_interval", pd.NA)
                    n_genes = pd.NA if pd.isna(ng) else int(ng)
                    ov = [g for g in str(iv.get("overlapping_genes", "")).split(";") if g]
                    cand_genes = ";".join(ov + _flank_gene_ids(iv))
                    gene_ev = _STATUS_TO_EVIDENCE.get(str(iv.get("annotation_status", "")), "no_gene_within_interval")
            else:
                interval_start = interval_end = pd.NA
                bounded_by = "typed_snps"
                n_genes, cand_genes = pd.NA, ""
                gene_ev = "no_annotation_loaded" if genes is None else "no_gene_within_interval"

        if species == "custom":
            gene_ev = "custom_gene_model"
        gb = "custom" if species == "custom" else genome_build

        rows.append({
            "SNP": snp, "Chr": cc, "Pos": pos, "PValue": p,
            "-log10p": float(r["-log10p"]) if "-log10p" in gwas_df.columns and pd.notna(r.get("-log10p"))
                       else (float(-np.log10(p)) if p > 0 else np.inf),
            "Beta_MLM": beta_mlm, "SE_MLM": se_mlm, "Beta_OLS": beta_ols, "SE_OLS": se_ols,
            "Effect_Source": effect_source,
            "MAF": maf_map.get(snp, np.nan),
            "ImputationRate": float(r["ImputationRate"]) if "ImputationRate" in gwas_df.columns and pd.notna(r.get("ImputationRate")) else np.nan,
            "Sig_Rule": sig_rule.rule,
            "Sig_Threshold": np.nan if sig_rule.p_threshold is None else float(sig_rule.p_threshold),
            "Seed_Threshold": float(seed_p_used),
            "Passes_Seed_Threshold": bool(p < float(seed_p_used)),
            "Passes_Meff": bool(r["Significant_Meff"]) if "Significant_Meff" in gwas_df.columns and pd.notna(r.get("Significant_Meff")) else pd.NA,
            "Passes_Bonf": bool(r["Significant_Bonf"]) if "Significant_Bonf" in gwas_df.columns and pd.notna(r.get("Significant_Bonf")) else pd.NA,
            "Passes_FDR": bool(r["Significant_FDR"]) if "Significant_FDR" in gwas_df.columns and pd.notna(r.get("Significant_FDR")) else pd.NA,
            "FDR": float(r["FDR"]) if "FDR" in gwas_df.columns and pd.notna(r.get("FDR")) else np.nan,
            "Block_ID": block_id, "Block_Status": block_status,
            "r2_to_block_lead": np.nan,   # [lazy] populated in the display layer (2b)
            "N_SNPs_in_Block": n_in_block,
            "Max_r2_in_Window": np.nan,   # [lazy]
            "Nearest_Typed_Upstream_SNP": up_snp, "Nearest_Typed_Upstream_bp": up_bp,
            "Nearest_Typed_Downstream_SNP": dn_snp, "Nearest_Typed_Downstream_bp": dn_bp,
            "Interval_Start_bp": interval_start, "Interval_End_bp": interval_end,
            "Interval_Bounded_By": bounded_by,
            "N_Genes_Interval": n_genes, "Candidate_Genes": cand_genes,
            "Gene_Evidence": gene_ev, "Genome_Build": gb,
            "Plot_Rendered": True, "Withheld_Reason": "",   # rendering budget applies in 2b
        })

    out = pd.DataFrame(rows, columns=SIG_TABLE_COLUMNS)
    # nullable integer dtypes where the schema calls for them
    for col in ("N_SNPs_in_Block", "N_Genes_Interval", "Nearest_Typed_Upstream_bp",
                "Nearest_Typed_Downstream_bp", "Interval_Start_bp", "Interval_End_bp"):
        out[col] = out[col].astype("Int64")
    return out.sort_values("PValue", kind="stable").reset_index(drop=True)


def project_unblocked(sig_table):
    """The Unblocked_SNPs projection — rows where Block_Status != 'in_block',
    exactly the UNBLOCKED_PROJECTION columns of the significant-SNP table."""
    unb = sig_table.loc[sig_table["Block_Status"] != "in_block", UNBLOCKED_PROJECTION]
    return unb.reset_index(drop=True)
