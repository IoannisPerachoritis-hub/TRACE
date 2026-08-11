"""T-32 — Layer 3 router acceptance (pure; dict-literal rows, no fixtures)."""
import numpy as np
import pandas as pd

from gwas.triage import (
    TRIAGE_COLUMNS,
    TriageThresholds,
    triage_blocks,
    triage_locus_view,
    triage_thresholds_json,
)

THR = TriageThresholds(r2_coherent=0.6)


def _row(**kw):
    """A fully-passing merged row (OK_BOTH); override fields per case."""
    base = dict(ldq_r2_mean=0.90, ldq_frac_members_r2_lead_ge=0.90, ldq_r2_pairs_finite=10,
                ldq_r2_pairs_total=10, mlg_frac_retained=0.90, mlg_eff_inv_simpson=3.0,
                n_lead_classes_ge=3, lead_maf=0.30, ldq_lead_in_block=True, ldq_n_members=5,
                n_samples_tested=150, n_samples_block=160, mlg_top1_freq=0.4,
                n_tested_haplotypes=3, mlg_n_other=5, ldq_lead_snp="s0")
    base.update(kw)
    return base


def _v(**kw):
    return triage_locus_view(_row(**kw), THR)


def test_decision_table_rows():
    assert (_v()["triage_view"], _v()["triage_primary"], _v()["triage_reason_code"]) == \
        ("both", "haplotype", "OK_BOTH")
    r = _v(mlg_frac_retained=0.50)                          # B1 fail
    assert (r["triage_view"], r["triage_primary"], r["triage_reason_code"]) == \
        ("both", "lead_snp", "MLG_SAMPLE_LOSS")
    r = _v(mlg_eff_inv_simpson=1.5)                         # B2 fail
    assert r["triage_reason_code"] == "MLG_CONCENTRATED" and r["triage_primary"] == "lead_snp"
    r = _v(ldq_r2_mean=0.30)                                # A fail
    assert (r["triage_view"], r["triage_primary"], r["triage_reason_code"]) == \
        ("both", "lead_snp", "BLOCK_INCOHERENT")
    r = _v(ldq_r2_mean=0.30, mlg_eff_inv_simpson=1.5)       # A fail + B fail
    assert (r["triage_view"], r["triage_reason_code"]) == ("lead_snp", "BLOCK_INCOHERENT_AND_FRAGMENTED")
    r = _v(n_lead_classes_ge=1)                             # L fail, A/B pass
    assert (r["triage_view"], r["triage_primary"], r["triage_reason_code"]) == \
        ("haplotype", "haplotype", "LEAD_UNUSABLE")
    r = _v(ldq_r2_mean=0.30, n_lead_classes_ge=1)           # L fail + A fail
    assert (r["triage_view"], r["triage_primary"], r["triage_reason_code"]) == \
        ("neither", "none", "NO_INTERPRETABLE_VIEW")


def test_ld_undetermined_and_not_tested_and_insufficient():
    r = _v(ldq_r2_pairs_finite=2)                           # < 3 finite pairs
    assert r["triage_reason_code"] == "LD_UNDETERMINED" and r["triage_axis_ld"] == "undetermined"
    r = _v(ldq_r2_pairs_finite=4, ldq_r2_pairs_total=10)    # coverage < 0.5
    assert r["triage_reason_code"] == "LD_UNDETERMINED"
    r = _v(mlg_frac_retained=np.nan, mlg_eff_inv_simpson=np.nan)   # no Layer 2
    assert r["triage_reason_code"] == "HAPLOTYPE_NOT_TESTED" and r["triage_view"] == "both"
    r = _v(ldq_r2_mean=np.nan)                              # core metric NaN
    assert r["triage_reason_code"] == "INSUFFICIENT_METRICS" and r["triage_primary"] == "lead_snp"


def test_lead_outside_override():
    r = _v(ldq_lead_in_block=False)                         # OK_BOTH but lead is a non-member seed
    assert r["triage_flag_lead_outside_block"] is True
    assert r["triage_primary"] == "lead_snp"                # forced from haplotype
    assert r["triage_reason_code"] == "OK_BOTH;LEAD_OUTSIDE_BLOCK"
    assert "seeding SNP" in r["triage_reason"]
    # never suppresses: view still both
    assert r["triage_view"] == "both"


def test_disabled_null_schema_and_json_roundtrip():
    import json
    off = TriageThresholds(r2_coherent=0.6, enabled=False)
    r = triage_locus_view(_row(), off)
    assert set(r.keys()) == set(TRIAGE_COLUMNS)
    assert r["triage_view"] == "" and r["triage_reason_code"] == ""
    # thresholds_json round-trips to the dataclass field values
    d = json.loads(triage_thresholds_json(THR))
    assert d["r2_coherent"] == 0.6 and d["lead_r2_frac"] == 0.5 and d["enabled"] is True


def test_triage_blocks_row_count_and_order_invariant():
    df = pd.DataFrame([_row(ldq_lead_snp="a"), _row(ldq_r2_mean=0.3, ldq_lead_snp="b"),
                       _row(n_lead_classes_ge=1, ldq_lead_snp="c")])
    out = triage_blocks(df, THR)
    assert len(out) == 3 and list(out["ldq_lead_snp"]) == ["a", "b", "c"]   # order preserved
    for c in TRIAGE_COLUMNS:
        assert c in out.columns
    assert list(out["triage_reason_code"])[:1] == ["OK_BOTH"]
    # empty frame -> columns present
    empty = triage_blocks(df.iloc[0:0], THR)
    assert len(empty) == 0 and all(c in empty.columns for c in TRIAGE_COLUMNS)


def test_property_raising_r2_coherent_never_flips_lead_to_hap():
    # a block near the coherence boundary
    row = _row(ldq_r2_mean=0.65)
    prims = [triage_locus_view(row, TriageThresholds(r2_coherent=c))["triage_primary"]
             for c in (0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9)]
    # once lead_snp, never returns to haplotype as the gate tightens
    seen_lead = False
    for p in prims:
        if p == "lead_snp":
            seen_lead = True
        if seen_lead:
            assert p != "haplotype", f"lead_snp -> haplotype as r2_coherent rose: {prims}"
