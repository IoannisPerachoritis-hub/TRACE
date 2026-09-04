"""T-38 — LD-triage additive-guarantee negative controls (N1-N9).

The guard for Sub-batch 3: it exists BEFORE the engine (T-30/31/32) lands, the
same rule Batch A followed. Sub-batch 3 is the first work that touches the core
files, so the guarantee is no longer "the file was never edited" but "the file
was edited and nothing published moved". These tests encode that.

Coverage map (some tighten as layers land — noted per test):
  N1 --no-triage byte identity ............ lands with the CLI flag (T-36)
  N2/N5 golden bitwise + eta2 frozen ...... already enforced by
                                            tests/test_golden_published.py
                                            (test_haplotype_blocks_bitwise_stable)
  N3 detection defaults untouched ......... here (introspection)
  N4 no column collision .................. here (set op over literal name lists)
  N6 no suppression ....................... lands with the router wiring (T-32/34)
  N7 upstream invariants .................. tightens once mlg_* land (T-31)
  N8 eta2 never alone ..................... here (intended column companions)
  N9 append-only column order ............. tightens once mlg_* land (T-31)
"""
import inspect

import gwas.ld as ld


# ── the intended new-name registry (single source of truth for N4/N8/N9) ──────
# Every column Sub-batch 3 adds. Kept here so N4 fails loudly if a new name ever
# collides with an existing table column.
LDQ_COLUMNS = [
    "ldq_n_members", "ldq_r2_mean", "ldq_r2_median", "ldq_r2_min",
    "ldq_r2_pairs_finite", "ldq_r2_pairs_total", "ldq_adj_r2_min",
    "ldq_adj_r2_median", "ldq_adj_thr_used", "ldq_lead_snp", "ldq_n_leads",
    "ldq_lead_in_block", "ldq_r2_lead_median", "ldq_r2_lead_min",
    "ldq_frac_members_r2_lead_ge", "ldq_span_kb", "ldq_meff_block",
    "ldq_meff_status", "ldq_r2_estimator", "ldq_r2_lead_estimator",
]
MLG_COLUMNS = [
    "mlg_n_observed", "mlg_top1_freq", "mlg_eff_inv_simpson", "mlg_n_other",
    "mlg_frac_retained", "mlg_min_group_n", "n_samples_tested",
]
TRIAGE_COLUMNS = [
    "triage_view", "triage_primary", "triage_reason_code", "triage_reason",
    "triage_axis_ld", "triage_flag_lead_outside_block", "triage_thresholds_json",
]
ETA2_COLUMNS = ["eta2_null_expected", "eta2_adj", "eta2_F_rank_delta"]

NEW_COLUMNS = LDQ_COLUMNS + MLG_COLUMNS + TRIAGE_COLUMNS + ETA2_COLUMNS

# existing inventories (spec §2.7); if any of these change, other guards catch it
BLOCK_TABLE_COLUMNS = ["Chr", "Start (bp)", "End (bp)", "lead_snp", "lead_snp_pvalue",
                       "SNP_IDs", "n_contained_blocks"]
HAPLOTYPE_TABLE_COLUMNS = [
    "Chr", "Start", "End", "lead_snp", "n_samples_block", "n_snps",
    "n_haplotypes", "n_tested_haplotypes", "df1", "df2", "n_permutations",
    "permutation_type", "F_param", "PValue_param", "F_perm", "P_perm", "PValue",
    "eta2", "hap_stats_json", "MidPos", "-log10p", "FDR_BH",
]
CLI_HAP_COLUMNS = ["Hap_PValue", "Hap_FDR_BH", "Hap_F_perm", "Hap_F_param",
                   "Hap_n_haplotypes", "Hap_n_tested", "Hap_n_samples",
                   "Hap_eta2", "Hap_n_perms"]


# ── N3 — detection defaults untouched (introspection, not a reviewer) ─────────
def test_n3_detection_defaults_untouched():
    defs = {p.name: p.default
            for p in inspect.signature(ld.find_ld_clusters_genomewide).parameters.values()}
    expect = {"ld_threshold": 0.6, "flank_kb": 300, "min_snps": 2, "top_n": 0,
              "sig_thresh": 1e-5, "adj_r2_min": 0.2, "min_pair_n": 20, "merge_iou": 0.3}
    for k, v in expect.items():
        assert k in defs, f"detection default {k} vanished from find_ld_clusters_genomewide"
        assert defs[k] == v, f"detection default {k} = {defs[k]!r}, expected {v!r} (frozen)"


def test_n3_pairwise_and_graph_signatures_present():
    # the reused primitives must keep their contract; triage calls, never edits them
    for fn in ("pairwise_r2", "pairwise_r", "get_block_snp_mask", "maf_from_matrix"):
        assert hasattr(ld, fn), f"gwas.ld.{fn} missing — triage reuses it"


# ── N4 — the new names never collide with an existing table column ────────────
def test_n4_no_column_collision():
    existing = set(BLOCK_TABLE_COLUMNS + HAPLOTYPE_TABLE_COLUMNS + CLI_HAP_COLUMNS)
    clash = existing & set(NEW_COLUMNS)
    assert not clash, f"new triage columns collide with existing table columns: {clash}"


def test_n4_new_names_unique():
    assert len(NEW_COLUMNS) == len(set(NEW_COLUMNS)), "duplicate name in the triage registry"


# ── N8 — eta2 is never displayed without its denominator companions ───────────
def test_n8_eta2_comparability_names_defined():
    # the eta2-comparability columns exist in the registry and travel with the
    # sample-count companions the display rules require (N8 full check binds to
    # show_cols/keep once T-35 wires them).
    assert "eta2_adj" in ETA2_COLUMNS and "eta2_null_expected" in ETA2_COLUMNS
    assert "n_samples_tested" in MLG_COLUMNS and "n_tested_haplotypes" in HAPLOTYPE_TABLE_COLUMNS
