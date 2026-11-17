"""The LD Blocks & Haplotypes tab's one block table (pages/_ld_tabs/block_table.py).

merge_block_table gives one row per detected block, with its haplotype test (when tested)
and its LD coherence. Every value must equal its source frame, a "Tested" column must say
"yes" or why not, and a value under two names must be shown once. run_haplotype_block_gwas
records the reasons (in its result's attrs, never written to a file).
"""
import numpy as np
import pandas as pd
import pytest

from gwas.haplotype import run_haplotype_block_gwas
from pages._ld_tabs.block_table import (COMPACT_COLUMNS, CSV_ONLY_COLUMNS, WINDOW_R2_HELP,
                                        compact_view, merge_block_table, screen_view)

BLOCKS = pd.DataFrame({
    "Chr": ["2", "2", "chr03"], "Start (bp)": [1000, 5000, 200], "End (bp)": [3000, 9000, 800],
    "lead_snp": ["a", "b", "c"], "lead_snp_pvalue": [1e-8, 3e-5, 2e-6],
    "SNP_IDs": ["a,a1,a2", "b,b1", "c,c1,c2,c3"], "Mean r2": [0.8, 0.5, 0.7],
})
HAP = pd.DataFrame({   # blocks 1 and 3 tested (Chr written differently on purpose)
    "Chr": ["2", "3"], "Start": [1000, 200], "End": [3000, 800], "lead_snp": ["a", "c"],
    "n_snps": [3, 2], "n_samples_tested": [48, 40], "n_tested_haplotypes": [3, 2],
    "P_perm": [0.004, 0.2], "PValue": [0.004, 0.2], "FDR_BH": [0.008, 0.2],
    "FDR_qvalue": [0.008, 0.2], "eta2": [0.31, 0.05], "eta2_ci_low": [0.1, 0.0],
    "eta2_ci_high": [0.5, 0.2], "Window_R2": [0.29, 0.06], "MidPos": [2000.0, 500.0],
    "kinship_model": ["LOCO kernel", "failed: no kernel for chromosome 3"],
    "h2_null": [0.42, float("nan")], "eta2_pcs_only": [0.33, 0.06],
})
COH = pd.DataFrame({
    "Chr": ["2", "2", "3"], "Start (bp)": [1000, 5000, 200], "End (bp)": [3000, 9000, 800],
    "ldq_r2_mean": [0.8, 0.5, 0.7], "ldq_r2_min": [0.6, 0.2, 0.4],
    "ldq_frac_members_r2_lead_ge": [1.0, 0.5, 0.75],
})
WHY = {"2:5000-9000": "fewer than 2 haplotype groups of at least 3 samples"}


def test_one_row_per_detected_block_with_every_value_from_its_source():
    t = merge_block_table(BLOCKS, HAP, COH, untested=WHY, seed_p=1e-5)
    assert len(t) == 3
    assert t["Chr"].tolist() == ["2", "2", "chr03"]
    assert t["Start"].tolist() == [1000, 5000, 200] and t["End"].tolist() == [3000, 9000, 800]
    assert t["Lead SNP"].tolist() == ["a", "b", "c"]
    assert t["SNPs"].tolist() == [3, 2, 4]
    assert t["Length (kb)"].tolist() == [2.0, 4.0, 0.6]
    assert t["Seed significant"].tolist() == [True, False, True]
    assert t["Tested"].tolist() == ["yes", WHY["2:5000-9000"], "yes"]
    for col, src in [("Samples tested", "n_samples_tested"), ("Haplotypes tested",
                     "n_tested_haplotypes"), ("P (permutation)", "P_perm"),
                     ("FDR q", "FDR_qvalue"), ("η² (haplotype test)", "eta2"),
                     ("η² 95% CI low", "eta2_ci_low"), ("η² 95% CI high", "eta2_ci_high"),
                     ("Window R² (all SNPs in the block window)", "Window_R2"),
                     ("SNPs in the test (after thinning)", "n_snps"),
                     ("Kinship model", "kinship_model"), ("η² (PCs-only test)", "eta2_pcs_only")]:
        assert t.loc[[0, 2], col].tolist() == HAP[src].tolist(), col
        assert pd.isna(t.loc[1, col]), col                          # the untested block
    assert t["Mean r²"].tolist() == BLOCKS["Mean r2"].tolist()
    assert t["Min r²"].tolist() == COH["ldq_r2_min"].tolist()
    assert t["Frac r² ≥ 0.6 to lead"].tolist() == COH["ldq_frac_members_r2_lead_ge"].tolist()


def test_values_under_two_names_are_shown_once():
    t = merge_block_table(BLOCKS, HAP, COH, untested=WHY)
    for gone in ("PValue", "FDR_BH", "FDR_qvalue", "P_perm", "eta2", "Window_R2", "ldq_r2_mean",
                 "lead_snp", "MidPos"):
        assert gone not in t.columns, gone
    assert list(t.columns).count("Chr") == 1
    # a pair that differs is kept, labelled
    odd = HAP.assign(PValue=[0.01, 0.3])
    t2 = merge_block_table(BLOCKS, odd, COH)
    assert "P (permutation)" in t2.columns and "PValue (differs from P_perm)" in t2.columns
    # the window R² is another quantity, not the test's η² under a second name: it keeps its
    # own column even where the two agree
    same = merge_block_table(BLOCKS, HAP.assign(Window_R2=HAP["eta2"]))
    assert "Window R² (all SNPs in the block window)" in same.columns
    assert "η² (haplotype test)" in same.columns


def test_the_only_eta2_on_the_table_is_the_tests():
    """Every column the table labels η² is the haplotype test's (its value, its interval,
    and the PCs-only test's value for comparison); the window value is an R²."""
    t = merge_block_table(BLOCKS, HAP, COH, untested=WHY)
    eta = [c for c in t.columns if "η²" in c]
    src = {"η² (haplotype test)": "eta2", "η² 95% CI low": "eta2_ci_low",
           "η² 95% CI high": "eta2_ci_high", "η² (PCs-only test)": "eta2_pcs_only"}
    assert sorted(eta) == sorted(src)
    for col in eta:
        assert t.loc[[0, 2], col].tolist() == HAP[src[col]].tolist(), col
    assert t.loc[[0, 2], "Window R² (all SNPs in the block window)"].tolist() == \
        HAP["Window_R2"].tolist()
    # the download button's tooltip names the CSV's window column in one line
    assert "\n" not in WINDOW_R2_HELP and CSV_ONLY_COLUMNS[0] in WINDOW_R2_HELP
    assert "not the test's η²" in WINDOW_R2_HELP


def test_the_compact_view_and_untested_tables():
    t = merge_block_table(BLOCKS, HAP, COH, untested=WHY)
    assert list(compact_view(t).columns) == COMPACT_COLUMNS
    none = merge_block_table(BLOCKS, None, COH, untested=WHY)      # nothing tested
    assert none["Tested"].tolist() == ["not tested", WHY["2:5000-9000"], "not tested"]
    assert len(none) == 3


def test_the_candidate_interval_sits_after_end():
    iv = pd.DataFrame({"Interval_start": [900, 4000, 100], "Interval_end": [3500, 9800, 900]})
    t = merge_block_table(BLOCKS, HAP, COH, untested=WHY, intervals=iv)
    cols = list(t.columns)
    at = cols.index("End")
    assert cols[at + 1: at + 3] == ["Interval start", "Interval end"]
    assert t["Interval start"].tolist() == [900, 4000, 100]
    assert t["Interval end"].tolist() == [3500, 9800, 900]
    assert t["Start"].tolist() == [1000, 5000, 200]                # the block span is kept
    assert list(compact_view(t).columns) == COMPACT_COLUMNS         # the default view as before


def test_the_window_r2_is_in_the_csv_only():
    t = merge_block_table(BLOCKS, HAP, COH, untested=WHY)
    window = "Window R² (all SNPs in the block window)"
    assert CSV_ONLY_COLUMNS == [window] and window in t.columns       # the CSV keeps it
    for all_columns in (False, True):
        shown = screen_view(t, all_columns)
        assert window not in shown.columns
        assert "η² (haplotype test)" in shown.columns
    assert list(screen_view(t, False).columns) == list(compact_view(t).columns)
    assert list(screen_view(t, True).columns) == [c for c in t.columns if c != window]


def _blocks_for_reasons():
    rng = np.random.default_rng(3)
    n = 60
    ids = np.array([f"S{i:02d}" for i in range(n)])
    grp = np.repeat([0, 1, 2], 20)
    G = np.zeros((n, 9))
    G[:, 0:3] = np.array([[0, 0, 2], [2, 2, 0], [0, 2, 2]])[grp]        # 3 groups of 20
    G[:, 3:6] = np.array([[(i // 3 ** k) % 3 for k in range(3)] for i in range(n)])  # unique
    G[:4, 6:8] = 2.0                         # one group of 56; 4 rare samples -> Other
    pos = np.arange(1, 10) * 1000.0
    sid = np.array([f"s{j}" for j in range(9)])
    blocks = pd.DataFrame({
        "Chr": ["1"] * 4, "Start (bp)": [1000, 4000, 7000, 20000],
        "End (bp)": [3000, 6000, 9000, 21000], "lead_snp": ["s0", "s3", "s6", "x"],
        "SNP_IDs": [",".join(sid[0:3]), ",".join(sid[3:6]), ",".join(sid[6:9]), "x"],
    })
    y = pd.DataFrame({"t": grp + rng.normal(0, 0.5, n)}, index=ids)
    return blocks, np.array(["1"] * 9), pos, G, sid, pd.DataFrame(G, index=ids), y


def test_the_haplotype_test_records_why_a_block_was_left_out():
    blocks, chroms, pos, G, sid, geno_df, pheno = _blocks_for_reasons()
    res, _tables = run_haplotype_block_gwas(
        haplo_df=blocks, chroms=chroms, positions=pos, geno_imputed=G, sid=sid,
        geno_df=geno_df, pheno_df=pheno, trait_col="t", pcs=None, min_hap_count=5,
        min_group_size=5, n_perm=20, hap_test="pcs")
    assert res[["Start", "End"]].values.tolist() == [[1000, 3000]]
    why = res.attrs["untested"]
    assert why == {
        "1:4000-6000": "no haplotype in at least 5 samples",
        "1:7000-9000": "fewer than 2 haplotype groups of at least 5 samples",
        "1:20000-21000": "fewer than 2 SNPs",
    }
    table = merge_block_table(blocks, res, None, untested=why)
    assert table["Tested"].tolist() == ["yes", why["1:4000-6000"], why["1:7000-9000"],
                                        why["1:20000-21000"]]


@pytest.mark.parametrize("bad", ["Chr", "Start (bp)"])
def test_missing_join_columns_are_refused(bad):
    with pytest.raises(KeyError):
        merge_block_table(BLOCKS.drop(columns=[bad]), HAP)
