"""T-34 — per-block two-view sample ledger (pure; importable without Streamlit).

Makes "the two panels use different sample sets" visible as data rather than
implied: one row per sample stating whether it is in the lead-SNP view, in the
haplotype (MLG) view, or excluded — and, when excluded, why (a closed
vocabulary, each value traceable to one exclusion in run_haplotype_block_gwas).
numpy + pandas only. Spec: docs/revision/specs/ld_triage_spec.md §7.2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEDGER_COLUMNS = [
    "Sample", "has_phenotype", "lead_call", "lead_genotype_class", "in_lead_view",
    "mlg_string", "mlg_label", "in_hap_view", "exclusion_reason",
]

# closed exclusion vocabulary (§7.2) — each value maps to one code site
EXCLUSION_REASONS = (
    "", "phenotype_missing", "lead_call_missing", "rare_mlg_below_min_hap_count",
    "group_below_min_group_size", "pc_row_missing", "block_missingness_gt_0.30",
)


def build_view_sample_ledger(sample_ids, y_raw, lead_calls, hap_table, tested_samples,
                             *, min_hap_count=5, min_group_size=3):
    """One row per sample reconciling the lead-SNP and haplotype views.

    Parameters
    ----------
    sample_ids     : sequence of sample IDs (str), the reference order.
    y_raw          : phenotype aligned to sample_ids (NaN allowed).
    lead_calls     : dosage at the lead SNP aligned to sample_ids (NaN allowed);
                     pass None if the lead SNP is unavailable.
    hap_table      : DataFrame with columns Sample / Haplotype (H1.. or "Other") /
                     Allele_sequence — the MLG assignment for every sample that
                     reached labelling (hap_tables[block_key]).
    tested_samples : the samples in the final tested set (df_test["Sample"]).
    min_hap_count / min_group_size : the exclusion-chain thresholds (for the reason).

    Returns
    -------
    DataFrame with LEDGER_COLUMNS, one row per sample, order = sample_ids.
    """
    sids = [str(s) for s in sample_ids]
    n = len(sids)
    y = np.asarray(y_raw, dtype=float) if y_raw is not None else np.full(n, np.nan)
    lc = np.asarray(lead_calls, dtype=float) if lead_calls is not None else np.full(n, np.nan)

    seq_by = {}
    lab_by = {}
    if hap_table is not None and len(hap_table):
        seq_by = dict(zip(hap_table["Sample"].astype(str), hap_table["Allele_sequence"].astype(str)))
        lab_by = dict(zip(hap_table["Sample"].astype(str), hap_table["Haplotype"].astype(str)))
    tested = {str(s) for s in (tested_samples if tested_samples is not None else [])}

    # tested-group label counts -> which labels are a valid tested group
    if tested and lab_by:
        tested_labels = pd.Series([lab_by.get(s, "") for s in tested])
        counts = tested_labels[tested_labels != ""].value_counts()
        valid_labels = set(counts[counts >= int(min_group_size)].index)
    else:
        valid_labels = set()

    rows = []
    for i, s in enumerate(sids):
        has_pheno = bool(np.isfinite(y[i]))
        lead_call = lc[i]
        has_lead = bool(np.isfinite(lead_call))
        lead_class = int(round(lead_call)) if has_lead else None
        in_lead = has_pheno and has_lead
        in_hap = s in tested
        label = lab_by.get(s, "")

        if in_lead and in_hap:
            reason = ""
        elif not has_pheno:
            reason = "phenotype_missing"
        elif not in_hap:
            if label == "Other":
                reason = "rare_mlg_below_min_hap_count"
            elif s not in lab_by:
                reason = "block_missingness_gt_0.30"
            elif label not in valid_labels:
                reason = "group_below_min_group_size"
            else:
                reason = "pc_row_missing"
        else:  # in the haplotype view + has phenotype, but no lead call
            reason = "lead_call_missing"

        rows.append({
            "Sample": s, "has_phenotype": has_pheno,
            "lead_call": lead_call if has_lead else np.nan,
            "lead_genotype_class": lead_class if lead_class is not None else np.nan,
            "in_lead_view": in_lead, "mlg_string": seq_by.get(s, ""),
            "mlg_label": label, "in_hap_view": in_hap, "exclusion_reason": reason,
        })
    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def view_sample_counts(ledger) -> dict:
    """The three-cell strip aggregates (§7.2): the two view sizes + their overlap
    as an asymmetric split (a sample can be in the haplotype view while missing
    the lead call, so the sets are not nested)."""
    if ledger is None or len(ledger) == 0:
        return {"n_samples_lead": 0, "n_samples_tested": 0,
                "n_both": 0, "n_lead_only": 0, "n_hap_only": 0}
    lead = ledger["in_lead_view"].astype(bool)
    hap = ledger["in_hap_view"].astype(bool)
    return {
        "n_samples_lead": int(lead.sum()),
        "n_samples_tested": int(hap.sum()),
        "n_both": int((lead & hap).sum()),
        "n_lead_only": int((lead & ~hap).sum()),
        "n_hap_only": int((~lead & hap).sum()),
    }
