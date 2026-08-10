"""T-20/T-77 — significant-SNP table unit tests."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gwas.significance import rule_from_cli_args
from gwas.sigtable import (
    BLOCK_STATUS_VALUES,
    SIG_TABLE_COLUMNS,
    UNBLOCKED_PROJECTION,
    build_significant_snp_table,
    project_unblocked,
)

RULE = rule_from_cli_args(SimpleNamespace(sig_thresh="bonferroni"), n_snps=100, meff_val=None)  # p_thr=5e-4


def _gwas(rows, extra=None):
    df = pd.DataFrame(rows, columns=["SNP", "Chr", "Pos", "PValue"])
    if extra:
        for k, v in extra.items():
            df[k] = v
    return df


def _blocks(rows):
    return pd.DataFrame(rows, columns=["Chr", "Start (bp)", "End (bp)", "Lead SNP", "SNP_IDs"])


def _axis(positions, chrom="1"):
    positions = list(positions)
    return (np.array([chrom] * len(positions)), np.array(positions, int),
            np.array([f"m{p}" for p in positions]))


def _full_gwas():
    # 3 significant SNPs (p<5e-4): one inside a block, two unblocked
    g = _gwas(
        [["m1500", "1", 1500, 1e-8], ["m5000", "1", 5000, 1e-8], ["m5200", "1", 5200, 1e-4]],
        extra={"Beta_MLM": [0.5, -0.3, 0.2], "SE_MLM": [0.1, 0.1, 0.1],
               "Beta_OLS": [0.6, -0.2, 0.25], "SE_OLS": [0.1, 0.1, 0.1]},
    )
    return g


def _run(gwas, blocks, genes=None, **kw):
    chroms, positions, sid = _axis([1000, 1500, 2000, 5000, 5200, 8000])
    return build_significant_snp_table(
        gwas, blocks, RULE, chroms, positions, sid, genes=genes,
        seed_p_used=1e-5, top_n_used=0, edge_flank_bp=5000, **kw)


def test_columns_and_order():
    tab = _run(_full_gwas(), _blocks([["1", 1000, 2000, "m1500", "m1000,m1500,m2000"]]))
    assert list(tab.columns) == SIG_TABLE_COLUMNS
    assert len(tab.columns) == 38


def test_in_block_vs_unblocked_split():
    tab = _run(_full_gwas(), _blocks([["1", 1000, 2000, "m1500", "m1000,m1500,m2000"]]))
    by = dict(zip(tab["SNP"], tab["Block_Status"]))
    assert by["m1500"] == "in_block"
    assert by["m5000"] != "in_block" and by["m5200"] != "in_block"
    assert tab.loc[tab["SNP"] == "m1500", "Block_ID"].iloc[0] == "1:1000-2000"


def test_block_status_only_five_values():
    tab = _run(_full_gwas(), _blocks([["1", 1000, 2000, "m1500", "m1000,m1500,m2000"]]))
    assert set(tab["Block_Status"]) <= set(BLOCK_STATUS_VALUES)


def test_unblocked_not_seeded_when_p_above_seed():
    # m5200 has p=1e-4 >= seed 1e-5 and not top-n -> never a seed
    tab = _run(_full_gwas(), _blocks([["1", 1000, 2000, "m1500", "m1000,m1500,m2000"]]))
    assert tab.loc[tab["SNP"] == "m5200", "Block_Status"].iloc[0] == "unblocked_not_seeded"


def test_reduced_frame_effect_source_unavailable():
    # 4-column GUI fallback frame -> effects unavailable on every row (never silently dropped)
    g = _gwas([["m5000", "1", 5000, 1e-8]])  # only SNP/Chr/Pos/PValue
    tab = _run(g, _blocks([]))
    assert (tab["Effect_Source"] == "unavailable").all()
    assert {"Beta_MLM", "Beta_OLS", "Effect_Source"} <= set(tab.columns)


def test_no_annotation_gives_na_not_zero():
    tab = _run(_full_gwas(), _blocks([["1", 1000, 2000, "m1500", "m1000,m1500,m2000"]]), genes=None)
    assert str(tab["N_Genes_Interval"].dtype) == "Int64"
    assert tab["N_Genes_Interval"].isna().all()               # <NA>, never 0
    assert (tab["Gene_Evidence"] == "no_annotation_loaded").all()


def test_project_unblocked_equals_projection():
    tab = _run(_full_gwas(), _blocks([["1", 1000, 2000, "m1500", "m1000,m1500,m2000"]]))
    unb = project_unblocked(tab)
    assert list(unb.columns) == UNBLOCKED_PROJECTION
    # every non-in_block row, and only those
    assert len(unb) == int((tab["Block_Status"] != "in_block").sum())
    manual = tab.loc[tab["Block_Status"] != "in_block", UNBLOCKED_PROJECTION].reset_index(drop=True)
    pd.testing.assert_frame_equal(unb, manual)


def test_row_count_equals_significant_count():
    g = _full_gwas()
    tab = _run(g, _blocks([]))
    assert len(tab) == int(RULE.significant_mask(g).sum())


def test_empty_when_no_significant():
    g = _gwas([["a", "1", 100, 0.9]], extra={"Beta_MLM": [0.1], "Beta_OLS": [0.1]})
    tab = _run(g, _blocks([]))
    assert len(tab) == 0
    assert list(tab.columns) == SIG_TABLE_COLUMNS


def test_with_genes_evidence_vocab():
    genes = pd.DataFrame({
        "Gene_ID": ["G1"], "Chr": ["1"], "Start": [1500], "End": [1600],
        "Strand": ["+"], "Description": ["d"],
    })
    tab = _run(_full_gwas(), _blocks([["1", 1000, 2000, "m1500", "m1000,m1500,m2000"]]), genes=genes)
    from gwas.sigtable import GENE_EVIDENCE_VALUES
    assert set(tab["Gene_Evidence"]) <= set(GENE_EVIDENCE_VALUES)
    # G1 overlaps the block [1000,2000] -> the in_block SNP is overlapping_interval
    assert tab.loc[tab["SNP"] == "m1500", "Gene_Evidence"].iloc[0] == "overlapping_interval"
