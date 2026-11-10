"""One candidate-interval rule for LD blocks and unblocked SNPs (gwas/isolated.py).

An LD block's candidate interval runs from the nearest post-QC marker strictly upstream
of its first SNP to the nearest strictly downstream of its last: the rule unblocked SNPs
already follow (build_flanking_intervals), with the same chromosome-edge flank and the
same very-wide clamp around the lead SNP. The block's own Start/End are not changed.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from gwas.isolated import (
    BLOCK_INTERVAL_COLUMNS,
    CANDIDATE_INTERVAL_COLUMNS,
    block_candidate_intervals,
    build_flanking_intervals,
    candidate_interval_table,
    default_edge_flank_bp,
    edge_flank_bp_from_decay,
)
from gwas.significance import rule_from_cli_args

EDGE = 300_000


def _axis(chrom_positions):
    """chroms, positions, sid for {chrom: [positions]}; ids m<chrom>_<pos>."""
    chroms, positions, sid = [], [], []
    for ch, poss in chrom_positions.items():
        for p in poss:
            chroms.append(ch)
            positions.append(p)
            sid.append(f"m{ch}_{p}")
    return np.array(chroms), np.array(positions, dtype=np.int64), np.array(sid)


AXIS = _axis({"1": [100, 200, 300, 400, 500],
              "2": [1_000_000, 10_000_000, 10_000_100, 20_000_000]})


def _blocks(rows):
    return pd.DataFrame(rows, columns=["Chr", "Start (bp)", "End (bp)", "lead_snp", "SNP_IDs"])


def _iv(blocks):
    return block_candidate_intervals(blocks, *AXIS, edge_flank_bp=EDGE)


def test_a_block_mid_chromosome_runs_from_the_marker_before_to_the_marker_after():
    blocks = _blocks([["1", 200, 400, "m1_300", "m1_200,m1_300,m1_400"]])
    iv = _iv(blocks)
    assert list(iv.columns) == BLOCK_INTERVAL_COLUMNS
    assert iv.iloc[0].tolist() == [100, 500, "", False]
    assert blocks[["Start (bp)", "End (bp)"]].values.tolist() == [[200, 400]]   # unchanged


def test_a_block_at_a_chromosome_edge_takes_the_edge_flank():
    blocks = _blocks([["1", 100, 300, "m1_100", "m1_100,m1_200,m1_300"],     # first marker
                      ["1", 300, 500, "m1_500", "m1_300,m1_400,m1_500"]])    # last marker
    iv = _iv(blocks)
    assert iv.iloc[0].tolist() == [0, 400, "chr_start", False]               # max(0, 100 - edge)
    assert iv.iloc[1].tolist() == [200, 500 + EDGE, "chr_end", False]


def test_a_very_wide_gap_is_clamped_around_the_lead_and_flagged():
    blocks = _blocks([["2", 10_000_000, 10_000_100, "m2_10000000", "m2_10000000,m2_10000100"]])
    iv = _iv(blocks)
    # flanks at 1 Mb and 20 Mb: 19 Mb > 5 Mb, so 5 Mb around the lead at 10 Mb
    assert iv.iloc[0].tolist() == [7_500_000, 12_500_000, "", True]


def test_an_unblocked_snp_gets_build_flanking_intervals_interval():
    for ch, pos in (("1", 300), ("1", 100), ("1", 500), ("2", 10_000_000)):
        uncovered = pd.DataFrame({"chr": [ch], "pos": [pos], "snp_id": [f"m{ch}_{pos}"],
                                  "p_value": [1e-9], "omission_path": ["block_formation"]})
        ref = build_flanking_intervals(uncovered, *AXIS, edge_flank_bp=EDGE).iloc[0]
        one = _iv(_blocks([[ch, pos, pos, f"m{ch}_{pos}", f"m{ch}_{pos}"]])).iloc[0]
        assert (one["Interval_start"], one["Interval_end"]) == (ref["Start (bp)"], ref["End (bp)"])
        assert one["Interval_edge_flag"] == ref["edge_flag"]
        assert bool(one["Interval_truncated"]) == bool(ref["interval_truncated"])


def test_the_table_lists_blocks_and_unblocked_snps_by_one_rule():
    rule = rule_from_cli_args(SimpleNamespace(sig_thresh="bonferroni"), n_snps=100, meff_val=None)
    gwas = pd.DataFrame({"SNP": ["m1_300", "m2_20000000"], "Chr": ["1", "2"],
                         "Pos": [300, 20_000_000], "PValue": [1e-9, 1e-8]})
    blocks = _blocks([["1", 200, 400, "m1_300", "m1_200,m1_300,m1_400"]])
    t = candidate_interval_table(blocks, gwas, rule, *AXIS, edge_flank_bp=EDGE)
    assert list(t.columns) == CANDIDATE_INTERVAL_COLUMNS
    assert t["Type"].tolist() == ["LD block", "unblocked SNP"]
    assert t.iloc[0][["Chr", "Interval_start", "Interval_end", "Lead_SNP"]].tolist() == [
        "1", 100, 500, "m1_300"]
    assert (t.iloc[0]["Block_start"], t.iloc[0]["Block_end"]) == (200, 400)
    # the unblocked SNP is the chromosome's last marker: chr_end, the edge flank past it;
    # 10,000,100 to 20,300,000 is over 5 Mb, so it is clamped around the SNP (and flagged)
    assert t.iloc[1][["Chr", "Interval_start", "Interval_end", "Lead_SNP",
                      "Interval_edge_flag", "Interval_truncated"]].tolist() == [
        "2", 17_500_000, 20_000_000 + EDGE, "m2_20000000", "chr_end", True]
    assert pd.isna(t.iloc[1]["Block_start"])


def test_the_edge_flank_is_the_clis_rule():
    assert default_edge_flank_bp(None, None) == 300_000
    assert default_edge_flank_bp(None, 144) == 144_000          # the run's LD flank
    assert default_edge_flank_bp(50, 144) == 50_000             # --isolated-edge-flank-kb wins
    assert edge_flank_bp_from_decay(72.17) == 144_000           # 2 x LD decay, as the CLI
    assert edge_flank_bp_from_decay(None) == 300_000
