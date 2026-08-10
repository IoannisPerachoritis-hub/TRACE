"""C1 — gwas/isolated.py unit tests (T-42/43/44/48 + AC-42.6, AC-N4)."""
import ast
import pathlib
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gwas.significance import rule_from_cli_args
from gwas import isolated
from gwas.isolated import (
    DENYLIST,
    INTERVAL_COLUMNS,
    UNBLOCKED_COLUMNS,
    annotate_isolated_intervals,
    build_flanking_intervals,
    find_unblocked_significant_snps,
    isolated_interval_caption,
    run_isolated_snp_rescue,
)

# Bonferroni over 100 tests -> p_threshold 5e-4, and no Significant_* column present,
# so significance is controlled purely by the PValues in each test frame.
RULE = rule_from_cli_args(SimpleNamespace(sig_thresh="bonferroni"), n_snps=100, meff_val=None)


def _gwas(rows):
    return pd.DataFrame(rows, columns=["SNP", "Chr", "Pos", "PValue"])


def _blocks(rows):
    return pd.DataFrame(rows, columns=["Chr", "Start (bp)", "End (bp)", "Lead SNP", "SNP_IDs"])


# ---- T-42 find_unblocked_significant_snps ----
def test_find_unblocked_basic_coverage():
    gwas = _gwas([
        ["s_in", "2", 1500, 1e-8],     # significant, inside the block -> covered
        ["s_out", "2", 5000, 1e-8],    # significant, outside -> uncovered
        ["s_ns", "2", 5200, 0.5],      # not significant -> ignored
        ["s_c2", "3", 100, 1e-8],      # significant, chr3 has no block -> uncovered
    ])
    blocks = _blocks([["2", 1000, 2000, "s_in", "s_in"]])
    out = find_unblocked_significant_snps(gwas, blocks, RULE, seed_p_used=1e-5, top_n_used=0)
    assert list(out.columns) == UNBLOCKED_COLUMNS
    assert set(out["snp_id"]) == {"s_out", "s_c2"}
    assert out.loc[out["snp_id"] == "s_c2", "n_blocks_on_chr"].iloc[0] == 0


def test_coverage_is_inclusive_on_both_ends():
    gwas = _gwas([["a", "1", 1000, 1e-8], ["b", "1", 2000, 1e-8], ["c", "1", 2001, 1e-8]])
    blocks = _blocks([["1", 1000, 2000, "x", "x"]])
    out = find_unblocked_significant_snps(gwas, blocks, RULE, seed_p_used=1e-5)
    assert set(out["snp_id"]) == {"c"}   # exactly-at-Start and exactly-at-End are covered


def test_omission_path_split():
    gwas = _gwas([
        ["seed", "1", 100, 1e-8],   # p < seed_p -> block_formation
        ["gap", "1", 200, 1e-4],    # seed_p <= p < reporting thr, not top-n -> seeding_threshold
    ])
    # widen the reporting threshold so both are significant
    rule = rule_from_cli_args(SimpleNamespace(sig_thresh="bonferroni"), n_snps=100, meff_val=None)
    out = find_unblocked_significant_snps(gwas, _blocks([]), rule, seed_p_used=1e-5, top_n_used=0)
    paths = dict(zip(out["snp_id"], out["omission_path"]))
    assert paths == {"seed": "block_formation", "gap": "seeding_threshold"}


def test_top_n_promotes_to_block_formation():
    gwas = _gwas([["g", "1", 200, 1e-4], ["x", "1", 900, 0.9]])
    out = find_unblocked_significant_snps(gwas, _blocks([]), RULE, seed_p_used=1e-5, top_n_used=1)
    # g is the top-1 by PValue, so although p >= seed_p it counts as seeded
    assert out.loc[out["snp_id"] == "g", "omission_path"].iloc[0] == "block_formation"


def test_blocks_none_all_uncovered():
    gwas = _gwas([["a", "1", 1, 1e-8], ["b", "2", 2, 1e-8]])
    out = find_unblocked_significant_snps(gwas, None, RULE, seed_p_used=1e-5)
    assert set(out["snp_id"]) == {"a", "b"}
    assert out["nearest_block_bp"].isna().all()


def test_input_purity_ac_42_6():
    gwas = _gwas([["a", "1", 1, 1e-8], ["b", "2", 5000, 1e-8]])
    blocks = _blocks([["1", 1, 100, "a", "a"]])
    gh = pd.util.hash_pandas_object(gwas).sum()
    bh = pd.util.hash_pandas_object(blocks).sum()
    out = find_unblocked_significant_snps(gwas, blocks, RULE, seed_p_used=1e-5)
    assert pd.util.hash_pandas_object(gwas).sum() == gh
    assert pd.util.hash_pandas_object(blocks).sum() == bh
    out.loc[:, "pos"] = -1  # mutating the result must not touch inputs
    assert pd.util.hash_pandas_object(gwas).sum() == gh


def test_isolated_never_imports_gwas_ld():
    """AC-N4 (static half): the rescue module must not import the block detector."""
    tree = ast.parse(pathlib.Path("gwas/isolated.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(not a.name.startswith("gwas.ld") for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("gwas.ld")


# ---- T-43 build_flanking_intervals ----
def _axis(positions, chrom="1"):
    positions = list(positions)
    return (np.array([chrom] * len(positions)),
            np.array(positions, dtype=int),
            np.array([f"m{p}" for p in positions]))


def test_interval_is_nearest_markers_both_sides():
    chroms, positions, sid = _axis([100, 200, 300, 400, 500])
    unc = find_unblocked_significant_snps(
        _gwas([["m300", "1", 300, 1e-8]]), _blocks([]), RULE, seed_p_used=1e-5)
    iv = build_flanking_intervals(unc, chroms, positions, sid, edge_flank_bp=1000)
    assert list(iv.columns) == INTERVAL_COLUMNS
    r = iv.iloc[0]
    assert (int(r["Start (bp)"]), int(r["End (bp)"])) == (200, 400)
    assert r["n_typed_markers_interior"] == 0   # defining isolated signature
    assert r["edge_flag"] == ""


def test_run_merge_adjacent_markers():
    chroms, positions, sid = _axis([100, 200, 300, 400])
    unc = find_unblocked_significant_snps(
        _gwas([["m200", "1", 200, 1e-8], ["m300", "1", 300, 1e-9]]), _blocks([]), RULE, seed_p_used=1e-5)
    iv = build_flanking_intervals(unc, chroms, positions, sid, edge_flank_bp=1000)
    assert len(iv) == 1                                  # merged into one run
    r = iv.iloc[0]
    assert (int(r["Start (bp)"]), int(r["End (bp)"])) == (100, 400)
    assert r["n_snps_in_run"] == 2
    assert r["Lead SNP"] == "m300"                       # lowest p-value


def test_separated_snps_make_two_overlapping_intervals():
    chroms, positions, sid = _axis([100, 200, 300, 400, 500])
    unc = find_unblocked_significant_snps(
        _gwas([["m200", "1", 200, 1e-8], ["m400", "1", 400, 1e-8]]), _blocks([]), RULE, seed_p_used=1e-5)
    iv = build_flanking_intervals(unc, chroms, positions, sid, edge_flank_bp=1000)
    assert len(iv) == 2                                  # marker 300 between them -> separate
    assert (100, 300) in list(zip(iv["Start (bp)"], iv["End (bp)"]))
    assert (300, 500) in list(zip(iv["Start (bp)"], iv["End (bp)"]))


def test_edge_flags_chr_start_and_end():
    chroms, positions, sid = _axis([100, 200, 300])
    unc = find_unblocked_significant_snps(
        _gwas([["m100", "1", 100, 1e-8], ["m300", "1", 300, 1e-8]]), _blocks([]), RULE, seed_p_used=1e-5)
    iv = build_flanking_intervals(unc, chroms, positions, sid, edge_flank_bp=50).set_index("Lead SNP")
    assert iv.loc["m100", "edge_flag"] == "chr_start"
    assert int(iv.loc["m100", "Start (bp)"]) == 50       # clamped by edge_flank_bp
    assert iv.loc["m300", "edge_flag"] == "chr_end"


def test_interval_truncation():
    chroms, positions, sid = _axis([0, 1_000_000, 20_000_000])
    unc = find_unblocked_significant_snps(
        _gwas([["m1", "1", 1_000_000, 1e-8]]), _blocks([]), RULE, seed_p_used=1e-5)
    iv = build_flanking_intervals(unc, chroms, positions, sid, edge_flank_bp=1000, max_interval_bp=5_000_000)
    r = iv.iloc[0]
    assert bool(r["interval_truncated"]) is True
    assert int(r["interval_bp"]) <= 5_000_000 < int(r["interval_bp_untruncated"])


# ---- T-44 annotate_isolated_intervals ----
def _genes():
    return pd.DataFrame({
        "Gene_ID": ["G1", "G2", "G3"],
        "Chr": ["1", "1", "1"],
        "Start": [210, 250, 900],
        "End": [240, 290, 950],
        "Strand": ["+", "-", "+"],
        "Description": ["desc1", "desc2", "desc3"],
    })


def test_annotate_no_genes_loaded_gives_na_not_zero():
    chroms, positions, sid = _axis([100, 200, 300, 400])
    unc = find_unblocked_significant_snps(_gwas([["m300", "1", 300, 1e-8]]), _blocks([]), RULE, seed_p_used=1e-5)
    iv = build_flanking_intervals(unc, chroms, positions, sid, edge_flank_bp=1000)
    wide, long = annotate_isolated_intervals(iv, None)
    assert str(wide["n_genes_in_interval"].dtype) == "Int64"
    assert wide["n_genes_in_interval"].isna().all()      # <NA>, never 0
    assert len(long) == 0


def test_annotate_with_genes_labels_distance_only():
    chroms, positions, sid = _axis([100, 200, 300, 400])
    unc = find_unblocked_significant_snps(_gwas([["m200", "1", 200, 1e-8]]), _blocks([]), RULE, seed_p_used=1e-5)
    iv = build_flanking_intervals(unc, chroms, positions, sid, edge_flank_bp=1000)  # interval [100,300]
    wide, long = annotate_isolated_intervals(iv, _genes())
    assert (wide["interval_evidence"] == "distance_only").all()
    assert (wide["localization_support"] == "flanking_markers_only").all()
    # G1 (210-240) and G2 (250-290) overlap [100,300]; every long row is distance_only, positional rank
    assert set(long["gene_id"]) >= {"G1", "G2"}
    assert (long["interval_evidence"] == "distance_only").all()
    assert list(long.sort_values("gene_start")["rank"]) == sorted(long["rank"])


# ---- T-48 denylist ----
def test_module_captions_are_denylist_clean():
    # exercise every conditional clause in the caption
    base = dict(Chr="2", n_typed_markers_interior=0, n_genes_in_interval=5)
    rows = [
        {**base, "Start (bp)": 1000, "End (bp)": 2000, "low_resolution": False,
         "interval_truncated": False, "edge_flag": "", "omission_path": "block_formation"},
        {**base, "Start (bp)": 0, "End (bp)": 9_000_000, "low_resolution": True,
         "interval_truncated": True, "edge_flag": "chr_start", "omission_path": "seeding_threshold"},
        {**base, "Start (bp)": 10, "End (bp)": 20, "low_resolution": False,
         "interval_truncated": False, "edge_flag": "chr_both", "omission_path": "mixed",
         "n_genes_in_interval": None},
    ]
    strings = [isolated_interval_caption(pd.Series(r)) for r in rows]
    strings.append("physical position only; no ranking applied")  # the long-table evidence_note
    for s in strings:
        for phrase in DENYLIST:
            assert not re.search(r"\b" + re.escape(phrase) + r"\b", s, re.I), \
                f"denylisted phrase {phrase!r} in: {s}"
    # a legal phrase must survive the same check
    assert re.search(r"\bcandidate region\b", "a candidate region here", re.I)


def test_end_to_end_orchestrator_smoke():
    chroms, positions, sid = _axis([100, 200, 300, 400, 5000, 5100])
    gwas = _gwas([["m300", "1", 300, 1e-8], ["m5000", "1", 5000, 1e-4]])
    res = run_isolated_snp_rescue(
        gwas, _blocks([]), RULE, chroms, positions, sid, genes=_genes(),
        seed_p_used=1e-5, top_n_used=0, edge_flank_bp=1000,
    )
    assert res.n_uncovered == 2
    assert res.n_block_path + res.n_seeding_path == 2
    assert res.n_intervals >= 1
    assert list(res.unblocked.columns) == UNBLOCKED_COLUMNS
