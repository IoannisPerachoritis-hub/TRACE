"""T-34 — build_view_sample_ledger acceptance (pure; synthetic)."""
import numpy as np
import pandas as pd

from gwas.views import (
    EXCLUSION_REASONS,
    LEDGER_COLUMNS,
    build_view_sample_ledger,
    view_sample_counts,
)


def _scenario():
    sids = [f"s{i}" for i in range(10)]
    y = np.array([np.nan, 1, 2, 3, 4, 5, 6, 7, 8, 9], float)         # s0 pheno missing
    lead = np.array([1, np.nan, 0, 1, 2, 0, 1, 2, 0, 1], float)      # s1 lead missing
    hap = pd.DataFrame({                                             # s8 absent -> block_missingness
        "Sample": ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s9"],
        "Haplotype": ["H1", "H1", "H1", "H1", "H1", "Other", "H2", "H1", "H1"],
        "Allele_sequence": ["AA"] * 9,
    })
    tested = ["s1", "s2", "s3", "s4", "s9"]                          # H1 tested group (>=3)
    return sids, y, lead, hap, tested


def test_ledger_covers_every_sample_and_reason():
    sids, y, lead, hap, tested = _scenario()
    led = build_view_sample_ledger(sids, y, lead, hap, tested, min_hap_count=5, min_group_size=3)
    assert list(led.columns) == LEDGER_COLUMNS
    assert len(led) == len(sids) and list(led["Sample"]) == sids       # every sample, in order
    r = dict(zip(led["Sample"], led["exclusion_reason"]))
    assert r["s0"] == "phenotype_missing"
    assert r["s1"] == "lead_call_missing"                              # in hap view, no lead -> n_hap_only
    assert r["s2"] == r["s3"] == r["s4"] == r["s9"] == ""              # in both
    assert r["s5"] == "rare_mlg_below_min_hap_count"
    assert r["s6"] == "group_below_min_group_size"
    assert r["s7"] == "pc_row_missing"
    assert r["s8"] == "block_missingness_gt_0.30"
    # every sample not in both views carries a non-empty reason from the closed vocab
    for _, row in led.iterrows():
        if not (row["in_lead_view"] and row["in_hap_view"]):
            assert row["exclusion_reason"] and row["exclusion_reason"] in EXCLUSION_REASONS


def test_view_counts_are_consistent_and_nonnested():
    sids, y, lead, hap, tested = _scenario()
    led = build_view_sample_ledger(sids, y, lead, hap, tested)
    c = view_sample_counts(led)
    assert c["n_both"] + c["n_lead_only"] == c["n_samples_lead"]       # exact (§7.2)
    assert c["n_both"] + c["n_hap_only"] == c["n_samples_tested"]
    assert c["n_hap_only"] == 1                                        # s1: in hap, missing lead
    assert c["n_samples_lead"] == 8 and c["n_samples_tested"] == 5


def test_empty_and_missing_inputs():
    led = build_view_sample_ledger([], None, None, None, None)
    assert len(led) == 0 and list(led.columns) == LEDGER_COLUMNS
    assert view_sample_counts(led)["n_both"] == 0
    # no lead calls at all -> every sample lead_call_missing or phenotype_missing, never crashes
    led2 = build_view_sample_ledger(["a", "b"], [1.0, 2.0], None,
                                    pd.DataFrame({"Sample": ["a"], "Haplotype": ["H1"],
                                                  "Allele_sequence": ["A"]}), ["a"])
    assert (~led2["in_lead_view"]).all()
    assert led2.set_index("Sample").loc["a", "exclusion_reason"] == "lead_call_missing"
