"""The LD Blocks & Haplotypes tab's one block table.

``merge_block_table`` gives one row per detected LD block (in detection order), with its
haplotype test when the block was tested and its LD coherence, joined on the block's
chromosome, start and end. A ``Tested`` column says "yes", or why the block was not
tested. Columns that repeat a value under two names are shown once:
- the test's ``PValue`` is its ``P_perm``: "P (permutation)";
- the function's ``FDR_BH`` and the tab's ``FDR_qvalue`` are one Benjamini-Hochberg: "FDR q";
- the block's ``Mean r2`` is the coherence table's ``ldq_r2_mean``: "Mean r²".
When a pair differs (it should not), both are kept, labelled. The only η² is the haplotype
test's ``eta2`` (on its samples and markers, by its model), with its interval and the
PCs-only test's value for comparison. ``Window_R2`` (gwas.guipost.block_qc_effects: the
share of the PC-residualised trait's variance that the haplotype groups of every SNP in the
block's coordinate window explain, built from the older window-based groups) is another
quantity: it is labelled "Window R² (all SNPs in the block window)" and ``screen_view``, what
the tab shows, leaves it to the CSV.
"""
import numpy as np
import pandas as pd

COMPACT_COLUMNS = [
    "Chr", "Start", "End", "Lead SNP", "SNPs", "Length (kb)", "Tested",
    "Samples tested", "Haplotypes tested", "P (permutation)", "FDR q",
    "η² (haplotype test)", "η² 95% CI low", "η² 95% CI high", "Mean r²",
    "Kinship model",
]

_BLOCK_NAMES = {"lead_snp": "Lead SNP", "lead_snp_pvalue": "Lead SNP p-value",
                "SNP_IDs": "SNP IDs", "Mean r2": "Mean r²", "merge_r2": "Merge r²"}
_HAP_NAMES = {
    "n_samples_tested": "Samples tested", "n_tested_haplotypes": "Haplotypes tested",
    "n_haplotypes": "Haplotypes observed", "n_samples_block": "Samples in the block",
    "n_snps": "SNPs in the test (after thinning)", "P_perm": "P (permutation)",
    "F_perm": "F (permutation)", "F_param": "F (parametric)",
    "PValue_param": "P (parametric)", "eta2": "η² (haplotype test)",
    "eta2_ci_low": "η² 95% CI low", "eta2_ci_high": "η² 95% CI high", "omega2": "ω²",
    "lead_snp_r2": "Lead SNP R²", "n_permutations": "Permutations",
    "Block_length_kb": "Block length (kb)", "Frac_other": "Fraction of haplotypes in Other",
    "maf_median": "Median MAF",
    # the test's kernel ("none" for the PCs-only test, or why the kinship-aware test could
    # not test the block), its null model's h2, and the PCs-only test's eta2 for comparison
    "kinship_model": "Kinship model", "h2_null": "h² (null model)",
    "eta2_pcs_only": "η² (PCs-only test)",
    # gwas.guipost.block_qc_effects' window value: not the test's η²
    "Window_R2": "Window R² (all SNPs in the block window)",
}
# the "Download block table (CSV)" button's tooltip: the CSV's one column the screen leaves out
WINDOW_R2_HELP = (
    "The CSV also has Window R² (all SNPs in the block window): the share of the trait's "
    "variance, after the PCs, that the haplotype groups of every SNP in the block's window "
    "explain over all phenotyped samples. It is not the test's η².")
_COH_NAMES = {"ldq_r2_min": "Min r²", "ldq_frac_members_r2_lead_ge": "Frac r² ≥ 0.6 to lead"}


def _canon(chrom):
    from annotation import canon_chr
    return canon_chr(str(chrom))


def _same(a, b):
    """Two columns hold the same values (NaN equal to NaN; numbers to 1e-12)."""
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    try:
        return bool(np.allclose(a.astype(float), b.astype(float), rtol=0, atol=1e-12,
                                equal_nan=True))
    except (TypeError, ValueError):
        return bool((a.astype(str) == b.astype(str)).all())


def _keep_one(df, keep, other, name, other_name):
    """Show ``keep`` as ``name``; drop ``other`` when it repeats it, else label it."""
    if keep not in df.columns and other in df.columns:
        keep, other = other, keep
    if keep not in df.columns:
        return df
    if other in df.columns:
        if _same(df[keep], df[other]):
            df = df.drop(columns=[other])
        else:
            df = df.rename(columns={other: other_name})
    return df.rename(columns={keep: name})


def merge_block_table(blocks, hap_result=None, coherence=None, untested=None, seed_p=None,
                      intervals=None):
    """One row per detected block: the block, its haplotype test and its LD coherence.

    ``blocks`` is the detected blocks (Chr, Start (bp), End (bp), ...); ``hap_result``
    the haplotype test's table (Chr, Start, End, ...; the tested blocks only);
    ``coherence`` the LD-coherence table (Chr, Start (bp), End (bp), ldq_*); ``untested``
    {"Chr:Start-End": reason} for the blocks the test left out; ``seed_p`` adds "Seed
    significant" (the lead SNP's p-value below the seed threshold); ``intervals``
    (gwas.isolated.block_candidate_intervals, aligned to ``blocks``) adds "Interval start"
    and "Interval end" after "End": the region the gene search uses.
    """
    b = blocks.reset_index(drop=True).copy()
    key = pd.DataFrame({"_c": b["Chr"].map(_canon), "_s": b["Start (bp)"].astype(int),
                        "_e": b["End (bp)"].astype(int)})
    out = pd.DataFrame({"Chr": b["Chr"].astype(str), "Start": key["_s"], "End": key["_e"]})
    rest = b.drop(columns=["Chr", "Start (bp)", "End (bp)"])
    out = pd.concat([out, rest.rename(columns=_BLOCK_NAMES)], axis=1)
    if "SNP IDs" in out.columns:
        out.insert(4, "SNPs", out["SNP IDs"].fillna("").astype(str).map(
            lambda v: len([x for x in v.split(",") if x.strip()])))
    elif "n_snps" in out.columns:
        out = out.rename(columns={"n_snps": "SNPs"})
    out.insert(min(5, out.shape[1]), "Length (kb)", (out["End"] - out["Start"]) / 1000.0)
    if seed_p is not None and "Lead SNP p-value" in out.columns:
        out["Seed significant"] = pd.to_numeric(out["Lead SNP p-value"], errors="coerce") < float(seed_p)

    # the haplotype test, where the block was tested
    tested = np.zeros(len(out), dtype=bool)
    if hap_result is not None and len(hap_result):
        h = hap_result.copy()
        h["_c"] = h["Chr"].map(_canon)
        h["_s"] = h["Start"].astype(int)
        h["_e"] = h["End"].astype(int)
        h = h.drop(columns=[c for c in ("Chr", "Start", "End", "lead_snp", "MidPos")
                            if c in h.columns])
        merged = key.merge(h, on=["_c", "_s", "_e"], how="left", indicator=True)
        tested = (merged["_merge"] == "both").to_numpy()
        hcols = merged.drop(columns=["_c", "_s", "_e", "_merge"])
        hcols = _keep_one(hcols, "P_perm", "PValue", "P_perm", "PValue (differs from P_perm)")
        hcols = _keep_one(hcols, "FDR_qvalue", "FDR_BH", "FDR q", "FDR_BH (differs)")
        out = pd.concat([out, hcols.rename(columns=_HAP_NAMES)], axis=1)

    # why a block was not tested
    reasons = untested or {}
    out.insert(out.columns.get_loc("Length (kb)") + 1, "Tested", [
        "yes" if t else reasons.get(f"{c}:{s}-{e}", "not tested")
        for t, c, s, e in zip(tested, b["Chr"].astype(str), key["_s"], key["_e"])])

    # the block's LD coherence
    if coherence is not None and len(coherence):
        q = coherence.copy()
        q["_c"] = q["Chr"].map(_canon)
        q["_s"] = q["Start (bp)"].astype(int)
        q["_e"] = q["End (bp)"].astype(int)
        q = q.drop(columns=["Chr", "Start (bp)", "End (bp)"])
        qcols = key.merge(q, on=["_c", "_s", "_e"], how="left").drop(columns=["_c", "_s", "_e"])
        if "Mean r²" in out.columns and "ldq_r2_mean" in qcols.columns:
            if _same(out["Mean r²"], qcols["ldq_r2_mean"]):
                qcols = qcols.drop(columns=["ldq_r2_mean"])
            else:
                qcols = qcols.rename(columns={"ldq_r2_mean": "Mean r² (coherence table)"})
        elif "ldq_r2_mean" in qcols.columns:
            qcols = qcols.rename(columns={"ldq_r2_mean": "Mean r²"})
        out = pd.concat([out, qcols.rename(columns=_COH_NAMES)], axis=1)

    # the block's candidate interval: the marker before its first SNP to the marker after
    # its last (the block's own Start/End stay its member span)
    if intervals is not None and len(intervals) == len(out):
        iv = intervals.reset_index(drop=True)
        at = out.columns.get_loc("End") + 1
        out.insert(at, "Interval start", iv["Interval_start"].to_numpy())
        out.insert(at + 1, "Interval end", iv["Interval_end"].to_numpy())
    return out


def compact_view(table):
    """The table's default view: the COMPACT_COLUMNS it has, in that order."""
    return table[[c for c in COMPACT_COLUMNS if c in table.columns]]


# in the CSV, not on screen: the window R² comes from the older coordinate-window
# haplotype groups, which often differ from the test's, so on screen it only confuses
CSV_ONLY_COLUMNS = ["Window R² (all SNPs in the block window)"]


def screen_view(table, all_columns=False):
    """What the tab shows: the compact view, or (all columns) every column except the
    CSV_ONLY_COLUMNS. The CSV is written from the whole table."""
    if not all_columns:
        return compact_view(table)
    return table.drop(columns=[c for c in CSV_ONLY_COLUMNS if c in table.columns])
