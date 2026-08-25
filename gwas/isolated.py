"""Isolated-SNP rescue (T-42 … T-44, T-48).

Genome-wide-significant SNPs that form no LD block are currently dropped from the
post-GWAS view. This module recovers them as a **parallel** output — never by
re-running or relaxing the block detector. The unblocked set is a positional
**set difference** against the FINAL emitted block table.

The rule that makes that guarantee mechanical: **this module must never import
``gwas.ld``.** All coverage arithmetic is local; the block table is an input, not
something recomputed here. (Asserted by ``tests/test_isolated.py`` and the C1
negative-acceptance suite, AC-N4.)

Contents
--------
- ``find_unblocked_significant_snps`` (T-42) — the set-difference detector.
- ``build_flanking_intervals`` (T-43) — the nearest-typed-marker interval.
- ``annotate_isolated_intervals`` (T-44) — gene mining via the PUBLIC
  ``annotation.annotate_ld_blocks`` (unchanged) + evidence labelling.
- ``run_isolated_snp_rescue`` — orchestrator returning a small result object.
- ``isolated_interval_caption`` (T-48) — the guarded, denylist-clean caption.

Spec: ``docs/revision/specs/isolated_snp_spec.md``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from annotation import canon_chr

# 12-column contract for find_unblocked_significant_snps (T-42).
UNBLOCKED_COLUMNS = [
    "snp_id", "chr", "pos", "p_value", "neg_log10_p",
    "reporting_rule", "reporting_threshold", "beta_ols", "beta_mlm",
    "omission_path", "n_blocks_on_chr", "nearest_block_bp",
]


def _chr_sort_key(c):
    c = str(c)
    return (0, int(c)) if c.isdigit() else (1, len(c), c)


def _block_intervals_by_chr(blocks_df):
    """{canon_chr: [(start, end), ...]} from the FINAL block table. Reads only the
    Chr / Start (bp) / End (bp) columns — never SNP_IDs, never a detector call."""
    out: dict[str, list[tuple[int, int]]] = {}
    if blocks_df is None or len(blocks_df) == 0:
        return out
    start_col = "Start (bp)" if "Start (bp)" in blocks_df.columns else "Start"
    end_col = "End (bp)" if "End (bp)" in blocks_df.columns else "End"
    for _, b in blocks_df.iterrows():
        ch = canon_chr(b["Chr"])
        out.setdefault(ch, []).append((int(b[start_col]), int(b[end_col])))
    return out


def _nearest_block_bp(pos, intervals):
    """Signed gap (bp) to the nearest block edge on this chromosome; negative if the
    SNP lies 5' of the nearest block, positive if 3'. NaN when the chr has none."""
    if not intervals:
        return np.nan
    best = None
    for s, e in intervals:
        if pos < s:
            gap = pos - s          # negative (upstream of block)
        elif pos > e:
            gap = pos - e          # positive (downstream of block)
        else:
            gap = 0                # inside (should not occur for uncovered SNPs)
        if best is None or abs(gap) < abs(best):
            best = gap
    return float(best)


def find_unblocked_significant_snps(
    gwas_df,
    blocks_df,
    sig_rule,
    *,
    seed_p_used,
    top_n_used=0,
    pad_bp=0,
):
    """Reporting-significant SNPs not covered by any emitted LD block (T-42).

    Pure: does not modify ``gwas_df`` or ``blocks_df``, and does not call
    ``gwas.ld``. The significant set is ``sig_rule.significant_mask(gwas_df)`` and
    nothing else — ``seed_p_used`` / ``top_n_used`` only *classify* the omission
    path, they never select.

    Coverage is **positional and inclusive** on both ends (matching
    ``cli.py`` block membership and ``gwas.ld.get_block_snp_mask``): a SNP is
    covered iff some block on its (canon) chromosome has
    ``Start - pad_bp <= Pos <= End + pad_bp``.

    Returns one row per uncovered SNP, columns ``UNBLOCKED_COLUMNS``, sorted by
    ``(chr, pos)``.
    """
    sig_mask = sig_rule.significant_mask(gwas_df)
    sig = gwas_df.loc[sig_mask]

    intervals_by_chr = _block_intervals_by_chr(blocks_df)

    # seed classification set (mirrors gwas/ld.py:839 + :844-849 top-N union)
    if top_n_used and top_n_used > 0:
        top_ids = set(gwas_df.nsmallest(int(top_n_used), "PValue")["SNP"].astype(str))
    else:
        top_ids = set()

    has_beta_ols = "Beta_OLS" in gwas_df.columns
    has_beta_mlm = "Beta_MLM" in gwas_df.columns

    # count blocks per chr once
    n_blocks_by_chr = {ch: len(ivs) for ch, ivs in intervals_by_chr.items()}

    rows = []
    for _, r in sig.iterrows():
        ch = canon_chr(r["Chr"])
        pos = int(r["Pos"])
        intervals = intervals_by_chr.get(ch, [])
        covered = any((s - pad_bp) <= pos <= (e + pad_bp) for s, e in intervals)
        if covered:
            continue
        p = float(r["PValue"])
        snp_id = str(r["SNP"])
        seeded = (p < float(seed_p_used)) or (snp_id in top_ids)
        rows.append({
            "snp_id": snp_id,
            "chr": ch,
            "pos": pos,
            "p_value": p,
            "neg_log10_p": float(-np.log10(p)) if p > 0 else np.inf,
            "reporting_rule": sig_rule.rule,
            "reporting_threshold": (np.nan if sig_rule.p_threshold is None
                                    else float(sig_rule.p_threshold)),
            "beta_ols": float(r["Beta_OLS"]) if has_beta_ols and pd.notna(r.get("Beta_OLS")) else np.nan,
            "beta_mlm": float(r["Beta_MLM"]) if has_beta_mlm and pd.notna(r.get("Beta_MLM")) else np.nan,
            "omission_path": "block_formation" if seeded else "seeding_threshold",
            "n_blocks_on_chr": int(n_blocks_by_chr.get(ch, 0)),
            "nearest_block_bp": _nearest_block_bp(pos, intervals),
        })

    if not rows:
        return pd.DataFrame({c: pd.Series(dtype=_UNBLOCKED_DTYPE[c]) for c in UNBLOCKED_COLUMNS})

    out = pd.DataFrame(rows, columns=UNBLOCKED_COLUMNS)
    out = out.sort_values(
        by=["chr", "pos"], key=lambda col: col.map(_chr_sort_key) if col.name == "chr" else col
    ).reset_index(drop=True)
    return out.astype(_UNBLOCKED_DTYPE)


_UNBLOCKED_DTYPE = {
    "snp_id": "object", "chr": "object", "pos": "int64", "p_value": "float64",
    "neg_log10_p": "float64", "reporting_rule": "object", "reporting_threshold": "float64",
    "beta_ols": "float64", "beta_mlm": "float64", "omission_path": "object",
    "n_blocks_on_chr": "int64", "nearest_block_bp": "float64",
}


# ---------------------------------------------------------------------------
# T-43 — flanking-marker intervals
# ---------------------------------------------------------------------------
INTERVAL_COLUMNS = [
    "interval_id", "region_type", "Chr", "Start (bp)", "End (bp)", "lead_snp", "SNP_IDs",
    "n_snps_in_run", "lead_snp_pvalue", "lead_neg_log10_p", "omission_path",
    "flank_up_snp_id", "flank_dn_snp_id", "flank_up_pos", "flank_dn_pos",
    "flank_up_is_significant", "flank_dn_is_significant",
    "dist_to_flank_up_bp", "dist_to_flank_dn_bp",
    "interval_bp", "interval_bp_untruncated", "interval_truncated", "edge_flag",
    "n_typed_markers_in_interval", "n_typed_markers_interior", "marker_set",
]


def build_flanking_intervals(
    uncovered_df,
    chroms,
    positions,
    sid,
    *,
    edge_flank_bp,
    max_interval_bp=5_000_000,
    merge_runs=True,
    significant_snp_ids=None,
):
    """Recombination-bounded interval for each uncovered SNP (T-43).

    The interval is ``[nearest typed marker strictly upstream, nearest strictly
    downstream]`` over the sorted-unique POST-QC marker positions, closed on both
    ends; membership is independent of p-value. Uncovered SNPs occupying
    *consecutive* marker slots merge into one run (parameter-free). Six edge cases
    (``chr_start`` / ``chr_end`` / ``chr_both`` / very-wide clamp / significant
    flank / run) each get an explicit rule. Emits ``INTERVAL_COLUMNS`` (block-schema
    names, so T-44 can call the public ``annotate_ld_blocks`` unchanged).

    ``significant_snp_ids`` (optional) is the full reporting-significant id set, used
    only to fill ``flank_*_is_significant`` (a flanking marker may itself be a
    significant SNP that landed in a block); ``None`` -> those flags are False.
    """
    sig_ids = set(significant_snp_ids or ())
    chroms = np.asarray([canon_chr(c) for c in np.asarray(chroms)], dtype=object)
    positions = np.asarray(positions, dtype=np.int64)
    sid = np.asarray(sid).astype(str)

    if uncovered_df is None or len(uncovered_df) == 0:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in INTERVAL_COLUMNS})

    rows = []
    for ch, sub in uncovered_df.groupby("chr", sort=False):
        cmask = chroms == canon_chr(ch)
        mpos = positions[cmask]
        msid = sid[cmask]
        order = np.argsort(mpos, kind="stable")
        mpos, msid = mpos[order], msid[order]
        uniq_pos, first_idx = np.unique(mpos, return_index=True)
        uniq_sid = msid[first_idx]
        pos_to_mi = {int(p): i for i, p in enumerate(uniq_pos)}

        s2 = sub.sort_values("pos")
        recs = [(int(r["pos"]), str(r["snp_id"]), float(r["p_value"]), str(r["omission_path"]))
                for _, r in s2.iterrows()]
        mis = [pos_to_mi.get(p) for p, _, _, _ in recs]

        runs, cur = [], []
        for k, mi in enumerate(mis):
            if not cur:
                cur = [k]
            elif merge_runs and mi is not None and mis[cur[-1]] is not None and mi == mis[cur[-1]] + 1:
                cur.append(k)
            else:
                runs.append(cur); cur = [k]
        if cur:
            runs.append(cur)

        for run in runs:
            members = [recs[k] for k in run]
            run_pos = [m[0] for m in members]
            run_ids = [m[1] for m in members]
            run_ps = [m[2] for m in members]
            run_paths = {m[3] for m in members}
            lead_k = int(np.argmin(run_ps))
            lead_pos, lead_id, lead_p = run_pos[lead_k], run_ids[lead_k], run_ps[lead_k]

            first_mi = pos_to_mi.get(run_pos[0])
            last_mi = pos_to_mi.get(run_pos[-1])
            up_mi = (first_mi - 1) if first_mi is not None and first_mi - 1 >= 0 else None
            dn_mi = (last_mi + 1) if last_mi is not None and last_mi + 1 < len(uniq_pos) else None

            if up_mi is None and dn_mi is None:
                edge = "chr_both"
                base_start = max(0, run_pos[0] - int(edge_flank_bp))
                base_end = run_pos[-1] + int(edge_flank_bp)
                flank_up_id = flank_dn_id = ""
                flank_up_pos = flank_dn_pos = np.nan
            elif up_mi is None:
                edge = "chr_start"
                base_start = max(0, run_pos[0] - int(edge_flank_bp))
                base_end = int(uniq_pos[dn_mi])
                flank_up_id = ""; flank_up_pos = np.nan
                flank_dn_id = str(uniq_sid[dn_mi]); flank_dn_pos = int(uniq_pos[dn_mi])
            elif dn_mi is None:
                edge = "chr_end"
                base_start = int(uniq_pos[up_mi])
                base_end = run_pos[-1] + int(edge_flank_bp)
                flank_up_id = str(uniq_sid[up_mi]); flank_up_pos = int(uniq_pos[up_mi])
                flank_dn_id = ""; flank_dn_pos = np.nan
            else:
                edge = ""
                base_start = int(uniq_pos[up_mi]); base_end = int(uniq_pos[dn_mi])
                flank_up_id = str(uniq_sid[up_mi]); flank_up_pos = int(uniq_pos[up_mi])
                flank_dn_id = str(uniq_sid[dn_mi]); flank_dn_pos = int(uniq_pos[dn_mi])

            untrunc = int(base_end - base_start)
            truncated = False
            start, end = base_start, base_end
            if untrunc > int(max_interval_bp):
                half = int(max_interval_bp) // 2
                start = max(base_start, lead_pos - half)
                end = min(base_end, lead_pos + half)
                truncated = True

            interior = int(((uniq_pos > start) & (uniq_pos < end)).sum()) - len(set(run_pos))
            n_in_interval = int(((uniq_pos >= start) & (uniq_pos <= end)).sum())
            omission = run_paths.pop() if len(run_paths) == 1 else "mixed"
            up_ok = flank_up_id != "" and flank_up_pos == flank_up_pos
            dn_ok = flank_dn_id != "" and flank_dn_pos == flank_dn_pos

            rows.append({
                "interval_id": f"ISO_{canon_chr(ch)}_{start}_{end}",
                "region_type": "isolated_snp_interval",
                "Chr": canon_chr(ch), "Start (bp)": int(start), "End (bp)": int(end),
                "lead_snp": lead_id, "SNP_IDs": ",".join(run_ids),
                "n_snps_in_run": len(run_ids),
                "lead_snp_pvalue": lead_p,
                "lead_neg_log10_p": float(-np.log10(lead_p)) if lead_p > 0 else np.inf,
                "omission_path": omission,
                "flank_up_snp_id": flank_up_id, "flank_dn_snp_id": flank_dn_id,
                "flank_up_pos": flank_up_pos, "flank_dn_pos": flank_dn_pos,
                "flank_up_is_significant": bool(flank_up_id in sig_ids),
                "flank_dn_is_significant": bool(flank_dn_id in sig_ids),
                "dist_to_flank_up_bp": (int(run_pos[0] - flank_up_pos) if up_ok else np.nan),
                "dist_to_flank_dn_bp": (int(flank_dn_pos - run_pos[-1]) if dn_ok else np.nan),
                "interval_bp": int(end - start),
                "interval_bp_untruncated": untrunc,
                "interval_truncated": truncated,
                "edge_flag": edge,
                "n_typed_markers_in_interval": n_in_interval,
                "n_typed_markers_interior": max(0, interior),
                "marker_set": "post_qc",
            })

    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in INTERVAL_COLUMNS})
    return pd.DataFrame(rows, columns=INTERVAL_COLUMNS).sort_values(
        ["Chr", "Start (bp)"], key=lambda col: col.map(_chr_sort_key) if col.name == "Chr" else col
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# T-48 — wording + denylist
# ---------------------------------------------------------------------------
# Phrases forbidden from any user-facing string this module emits. Enforced
# word-boundary-wise so "candidate region"/"candidate interval" stay legal.
DENYLIST = (
    "causal", "the causal gene", "candidate gene identified", "top candidate",
    "prioritised", "prioritized", "the gene responsible", "confirms", "validates",
    "pinpoints", "resolves the locus", "likely causal",
)


def isolated_interval_caption(row) -> str:
    """Guarded, denylist-clean caption for one isolated-SNP interval (T-48)."""
    ch = row["Chr"]
    start, end = int(row["Start (bp)"]), int(row["End (bp)"])
    kb = (end - start) / 1000.0
    interior = int(row.get("n_typed_markers_interior", 0))
    n_genes = row.get("n_genes_in_interval", None)
    genes_txt = "annotation not loaded" if (n_genes is None or pd.isna(n_genes)) else f"{int(n_genes)} gene(s)"
    parts = [
        f"Chr{ch}:{start:,}-{end:,} ({kb:.0f} kb; {interior} interior typed marker(s); {genes_txt}).",
        "This SNP is significant under the analysis reporting threshold but formed no LD block, "
        "so there is no LD-delimited interval.",
        "The bounds are the nearest typed post-QC markers on each side; interval width reflects "
        "marker spacing here, not association strength.",
        "Genes are listed by physical position only, in genomic order, not by any measure of importance.",
    ]
    if bool(row.get("low_resolution", False)):
        parts.append("The interval is wide (low marker resolution); treat gene membership as provisional.")
    if bool(row.get("interval_truncated", False)):
        parts.append("The mining window was clamped to a maximum width; the untyped region extends further.")
    if str(row.get("edge_flag", "")):
        parts.append("The interval abuts a chromosome edge and is bounded by a fixed flank rather than a typed marker.")
    if str(row.get("omission_path", "")) in ("seeding_threshold", "mixed"):
        parts.append("This SNP was above the block-seeding p-threshold, so it was never offered to the "
                     "block detector (the reporting and seeding thresholds differ).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# T-44 — gene mining + evidence labelling (calls the PUBLIC annotate_ld_blocks)
# ---------------------------------------------------------------------------
LONG_COLUMNS = [
    "interval_id", "Chr", "interval_start_bp", "interval_end_bp", "lead_snp_id", "lead_p_value",
    "gene_id", "gene_start", "gene_end", "gene_strand", "gene_description", "gene_relation",
    "dist_to_interval_bp", "dist_to_lead_snp_bp", "region_type", "interval_evidence",
    "localization_support", "evidence_note", "rank",
]
_WIDE_EXTRA = [
    "n_genes_in_interval", "gene_density_per_Mb", "low_res_bp_used", "low_resolution",
    "resolution_note", "n_genes_displayed", "genes_truncated_for_display",
    "interval_evidence", "localization_support", "interval_caption",
]


def _resolve_low_res_bp(low_res_bp, ld_decay_bp):
    if low_res_bp is not None:
        return int(low_res_bp)
    if ld_decay_bp:
        return int(max(2 * int(ld_decay_bp), 400_000))
    return 400_000


def annotate_isolated_intervals(
    intervals_df,
    genes,
    *,
    n_flank=2,
    max_flank_dist_bp=500_000,
    low_res_bp=None,
    ld_decay_bp=None,
    display_gene_cap=25,
):
    """Mine genes for each isolated interval via the PUBLIC ``annotate_ld_blocks``
    (unchanged) and attach the isolated-specific descriptors + evidence labels
    (T-44). Returns ``(wide_df, genes_long_df)``. All evidence is hard-coded to
    ``distance_only`` / ``flanking_markers_only`` — an isolated SNP has, by
    construction, no LD-supported interval.
    """
    from annotation import annotate_ld_blocks  # local import: keep module import-light

    low = _resolve_low_res_bp(low_res_bp, ld_decay_bp)

    if intervals_df is None or len(intervals_df) == 0:
        return (pd.DataFrame(columns=list(INTERVAL_COLUMNS) + _WIDE_EXTRA),
                pd.DataFrame(columns=LONG_COLUMNS))

    if genes is None:
        wide = intervals_df.copy()
        wide["n_genes_in_interval"] = pd.array([pd.NA] * len(wide), dtype="Int64")
        wide["gene_density_per_Mb"] = np.nan
        wide["low_res_bp_used"] = int(low)
        wide["low_resolution"] = wide["interval_bp"].astype(int) > int(low)
        wide["resolution_note"] = ["wide_interval" if lr else "" for lr in wide["low_resolution"]]
        wide["n_genes_displayed"] = pd.array([pd.NA] * len(wide), dtype="Int64")
        wide["genes_truncated_for_display"] = False
        wide["interval_evidence"] = "distance_only"
        wide["localization_support"] = "flanking_markers_only"
        wide["interval_caption"] = [isolated_interval_caption(r) for _, r in wide.iterrows()]
        return wide, pd.DataFrame(columns=LONG_COLUMNS)

    wide = annotate_ld_blocks(intervals_df, genes, n_flank=n_flank, max_flank_dist_bp=max_flank_dist_bp)
    wide["n_genes_in_interval"] = wide["n_genes_overlapping"].astype("Int64")
    ibp = wide["interval_bp"].astype(float).replace(0, np.nan)
    wide["gene_density_per_Mb"] = wide["n_genes_in_interval"].astype(float) / (ibp / 1e6)
    wide["low_res_bp_used"] = int(low)
    wide["low_resolution"] = wide["interval_bp"].astype(int) > int(low)
    wide["resolution_note"] = [
        ("wide_interval_truncated" if t else ("wide_interval" if lr else ("edge_clamped" if e else "")))
        for t, lr, e in zip(wide["interval_truncated"], wide["low_resolution"],
                            wide["edge_flag"].astype(str) != "")
    ]
    _ng = wide["n_genes_in_interval"].fillna(0).astype(int)
    wide["n_genes_displayed"] = _ng.clip(upper=display_gene_cap)
    wide["genes_truncated_for_display"] = _ng > display_gene_cap
    wide["interval_evidence"] = "distance_only"
    wide["localization_support"] = "flanking_markers_only"
    wide["interval_caption"] = [isolated_interval_caption(r) for _, r in wide.iterrows()]

    # gene-level long table
    desc_col = "Description" if "Description" in genes.columns else None
    gmap = {}
    for _, g in genes.iterrows():
        gmap[str(g["Gene_ID"])] = (
            int(g["Start"]), int(g["End"]), str(g.get("Strand", "")),
            str(g[desc_col]) if desc_col else "",
        )
    long_rows = []
    for _, r in wide.iterrows():
        imid = (int(r["Start (bp)"]) + int(r["End (bp)"])) // 2
        entries = []
        for gid in str(r.get("overlapping_genes", "")).split(";"):
            if gid:
                entries.append((gid, "overlapping", 0.0))
        for i in (1, 2):
            for direction, rel in (("upstream", "flanking_upstream"), ("downstream", "flanking_downstream")):
                gid = str(r.get(f"{direction}_gene_{i}", "") or "")
                if gid:
                    entries.append((gid, rel, r.get(f"{direction}_dist_{i}", np.nan)))
        for gid, rel, dist in entries:
            info = gmap.get(gid)
            gstart = info[0] if info else np.nan
            gend = info[1] if info else np.nan
            gmid = (gstart + gend) // 2 if info else np.nan
            long_rows.append({
                "interval_id": r["interval_id"], "Chr": r["Chr"],
                "interval_start_bp": int(r["Start (bp)"]), "interval_end_bp": int(r["End (bp)"]),
                "lead_snp_id": r["lead_snp"], "lead_p_value": r["lead_snp_pvalue"],
                "gene_id": gid, "gene_start": gstart, "gene_end": gend,
                "gene_strand": info[2] if info else "", "gene_description": info[3] if info else "",
                "gene_relation": rel, "dist_to_interval_bp": dist,
                "dist_to_lead_snp_bp": (int(gmid - imid) if info else np.nan),
                "region_type": "isolated_snp_interval", "interval_evidence": "distance_only",
                "localization_support": "flanking_markers_only",
                "evidence_note": "physical position only; no ranking applied", "rank": 0,
            })
    long_df = pd.DataFrame(long_rows, columns=LONG_COLUMNS)
    if len(long_df):
        long_df = long_df.sort_values(["interval_id", "gene_start"]).reset_index(drop=True)
        long_df["rank"] = long_df.groupby("interval_id").cumcount()
    return wide, long_df


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
class IsolatedRescueResult:
    """Small result carrier for the CLI/GUI wiring (T-45/T-47)."""

    def __init__(self, unblocked, intervals, genes_long):
        self.unblocked = unblocked
        self.intervals = intervals
        self.genes_long = genes_long
        self.n_uncovered = int(len(unblocked))
        self.n_intervals = int(len(intervals))
        vc = unblocked["omission_path"].value_counts() if len(unblocked) else {}
        self.n_seeding_path = int(vc.get("seeding_threshold", 0)) if len(unblocked) else 0
        self.n_block_path = int(vc.get("block_formation", 0)) if len(unblocked) else 0


def run_isolated_snp_rescue(
    gwas_df, blocks_df, sig_rule, chroms, positions, sid,
    genes=None, *,
    seed_p_used, top_n_used=0, pad_bp=0,
    edge_flank_bp, max_interval_bp=5_000_000, low_res_bp=None, ld_decay_bp=None,
):
    """End-to-end rescue for one model: detect uncovered significant SNPs → build
    flanking intervals → annotate. Never touches the block detector."""
    unblocked = find_unblocked_significant_snps(
        gwas_df, blocks_df, sig_rule, seed_p_used=seed_p_used,
        top_n_used=top_n_used, pad_bp=pad_bp,
    )
    sig_ids = set(gwas_df.loc[sig_rule.significant_mask(gwas_df), "SNP"].astype(str))
    intervals = build_flanking_intervals(
        unblocked, chroms, positions, sid,
        edge_flank_bp=edge_flank_bp, max_interval_bp=max_interval_bp,
        significant_snp_ids=sig_ids,
    )
    wide, genes_long = annotate_isolated_intervals(
        intervals, genes, low_res_bp=low_res_bp, ld_decay_bp=ld_decay_bp,
    )
    return IsolatedRescueResult(unblocked, wide, genes_long)
