"""T-40 — SignificanceRule unit tests (AC-40.1 … AC-40.4)."""
import dataclasses
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gwas.significance import (
    REPORTING_RULES,
    SignificanceRule,
    rule_from_cli_args,
    rule_from_streamlit,
)

N_SNPS = 43974
MEFF = 242


# --- AC-40.1: reproduces primary_thresh + _sig_col_name for all three rules ---
def test_rule_from_cli_reproduces_thresholds_and_columns():
    meff = rule_from_cli_args(SimpleNamespace(sig_thresh="meff"), N_SNPS, MEFF)
    assert meff.rule == "meff"
    assert meff.p_threshold == pytest.approx(0.05 / MEFF)       # cli.py:835/:853
    assert meff.boolean_column == "Significant_Meff"            # cli.py:1091-1094
    assert meff.n_tests == MEFF

    bonf = rule_from_cli_args(SimpleNamespace(sig_thresh="bonferroni"), N_SNPS, MEFF)
    assert bonf.rule == "bonferroni"
    assert bonf.p_threshold == pytest.approx(0.05 / N_SNPS)     # cli.py:842/:847
    assert bonf.boolean_column == "Significant_Bonf"
    assert bonf.n_tests == N_SNPS

    fdr = rule_from_cli_args(SimpleNamespace(sig_thresh="fdr"), N_SNPS, MEFF)
    assert fdr.rule == "fdr"
    assert fdr.p_threshold is None                              # cli.py:850
    assert fdr.boolean_column == "Significant_FDR"
    assert fdr.n_tests is None


def test_meff_failure_fallback_is_naive_bonferroni():
    """cli.py:837-840 — when M_eff can't be computed the rule stays meff but the
    threshold falls back to 0.05 / n_snps."""
    r = rule_from_cli_args(SimpleNamespace(sig_thresh="meff"), N_SNPS, None)
    assert r.rule == "meff"
    assert r.boolean_column == "Significant_Meff"
    assert r.p_threshold == pytest.approx(0.05 / N_SNPS)
    assert r.n_tests == N_SNPS


def test_default_rule_is_meff():
    assert rule_from_cli_args(SimpleNamespace(), N_SNPS, MEFF).rule == "meff"
    assert rule_from_cli_args(SimpleNamespace(sig_thresh="garbage"), N_SNPS, MEFF).rule == "meff"


def test_rule_from_streamlit_label_mapping():
    assert rule_from_streamlit("FDR q<0.05", N_SNPS, MEFF).rule == "fdr"
    assert rule_from_streamlit(f"M_eff Bonferroni (M={MEFF})", N_SNPS, MEFF).rule == "meff"
    assert rule_from_streamlit("Bonferroni α=0.05", N_SNPS, MEFF).rule == "bonferroni"


# --- AC-40.2: the boolean column is preferred, and agrees with the p-value path ---
def test_mask_prefers_boolean_column_and_agrees_with_pvalue():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=200)
    p[:5] = [1e-9, 1e-6, 5e-5, 1e-4, 3e-4]
    r = rule_from_cli_args(SimpleNamespace(sig_thresh="meff"), N_SNPS, MEFF)
    df = pd.DataFrame({"PValue": p})
    df["Significant_Meff"] = df["PValue"] < r.p_threshold      # as cli.py:872 writes it
    from_col = r.significant_mask(df)
    from_p = df["PValue"].to_numpy() < r.p_threshold
    assert np.array_equal(from_col, from_p)
    # and it truly reads the column, not the p-values: corrupt the column, mask follows it
    df2 = df.copy()
    df2["Significant_Meff"] = False
    assert not r.significant_mask(df2).any()


def test_mask_falls_back_to_pvalue_when_column_absent():
    r = rule_from_cli_args(SimpleNamespace(sig_thresh="bonferroni"), 100, MEFF)
    df = pd.DataFrame({"PValue": [1e-9, 0.5, 0.05 / 100 * 0.9, 0.05 / 100 * 1.1]})
    m = r.significant_mask(df)  # no Significant_Bonf column present
    assert m.tolist() == [True, False, True, False]


# --- AC-40.3: fdr rule without its column raises and never touches PValue ---
def test_fdr_without_column_raises_and_does_not_use_pvalue():
    r = rule_from_cli_args(SimpleNamespace(sig_thresh="fdr"), N_SNPS, MEFF)
    df = pd.DataFrame({"PValue": [1e-30, 1e-30, 1e-30]})  # all "significant" by p, but no FDR column
    with pytest.raises(ValueError, match="Significant_FDR"):
        r.significant_mask(df)


# --- AC-40.4: frozen ---
def test_rule_is_frozen():
    r = rule_from_cli_args(SimpleNamespace(sig_thresh="meff"), N_SNPS, MEFF)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.p_threshold = 1e-5


# --- construction validation ---
def test_post_init_rejects_inconsistent_construction():
    with pytest.raises(ValueError):
        SignificanceRule("nonsense", 1e-4, "Significant_Meff", 1, "x")
    with pytest.raises(ValueError):  # meff must have a p_threshold
        SignificanceRule("meff", None, "Significant_Meff", 1, "x")
    with pytest.raises(ValueError):  # fdr must NOT have a p_threshold
        SignificanceRule("fdr", 1e-4, "Significant_FDR", None, "x")
    with pytest.raises(ValueError):  # column must match rule
        SignificanceRule("meff", 1e-4, "Significant_Bonf", 1, "x")


def test_reporting_rules_constant():
    assert REPORTING_RULES == ("meff", "bonferroni", "fdr")


# --- numeric --sig-thresh (custom rule) ---

def test_numeric_sig_thresh_builds_custom_rule():
    r = rule_from_cli_args(SimpleNamespace(sig_thresh=5e-8), N_SNPS, MEFF)
    assert r.rule == "custom"
    assert r.p_threshold == 5e-8
    assert r.boolean_column == "Significant_Custom"
    assert r.n_tests is None
    assert r.label == "p < 5.0e-08"


def test_custom_rule_mask_uses_pvalue_threshold():
    r = rule_from_cli_args(SimpleNamespace(sig_thresh=1e-6), 100, MEFF)
    df = pd.DataFrame({"PValue": [1e-9, 1e-7, 1e-5, 0.4]})
    assert list(r.significant_mask(df)) == [True, True, False, False]


def test_custom_rule_prefers_boolean_column_when_present():
    r = rule_from_cli_args(SimpleNamespace(sig_thresh=5e-8), 100, MEFF)
    df = pd.DataFrame({"PValue": [1e-9, 0.5], "Significant_Custom": [False, True]})
    assert list(r.significant_mask(df)) == [False, True]


def test_rule_from_streamlit_custom():
    r = rule_from_streamlit("Custom p-value (e.g. 5e-8)", N_SNPS, MEFF, custom_thresh=5e-8)
    assert r.rule == "custom" and r.p_threshold == 5e-8
    assert r.boolean_column == "Significant_Custom"
    # named labels still resolve (no threshold needed)
    assert rule_from_streamlit("FDR (q<0.05)", N_SNPS, MEFF).rule == "fdr"
    assert rule_from_streamlit("Bonferroni (alpha=0.05)", N_SNPS, MEFF).rule == "bonferroni"
    assert rule_from_streamlit("M_eff - Li & Ji", N_SNPS, MEFF).rule == "meff"


# --- add_significance_columns: one rule for every model's table (2-INT.6a) ---

def _pvals(n=400, seed=3):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, size=n) ** 3
    p[:6] = [1e-12, 1e-8, 2e-6, 4e-5, 1e-4, 9e-4]
    return p


def test_significance_columns_use_the_given_thresholds_and_bh_within_the_table():
    from statsmodels.stats.multitest import multipletests

    from gwas.significance import add_significance_columns
    p = _pvals()
    df = pd.DataFrame({"SNP": [f"s{i}" for i in range(p.size)], "PValue": p})
    bonf, meff, custom = 0.05 / 1000, 0.05 / 300, 1e-3     # MLM's numbers, not this table's size
    out = add_significance_columns(df, bonf_thresh=bonf, meff_thresh=meff, custom_thresh=custom)
    assert list(out.columns) == ["SNP", "PValue", "FDR", "Significant_FDR", "Significant_Bonf",
                                 "Significant_Meff", "Significant_Custom"]
    assert list(df.columns) == ["SNP", "PValue"], "the input frame must not be modified"
    reject, q, _, _ = multipletests(p, method="fdr_bh")
    assert np.array_equal(out["FDR"].to_numpy(), q)
    assert np.array_equal(out["Significant_FDR"].to_numpy(), reject)
    assert np.array_equal(out["Significant_Bonf"].to_numpy(), p < bonf)
    assert np.array_equal(out["Significant_Meff"].to_numpy(), p < meff)
    assert np.array_equal(out["Significant_Custom"].to_numpy(), p < custom)
    assert out["Significant_Bonf"].sum() < out["Significant_Meff"].sum() < out["Significant_Custom"].sum()


def test_significance_columns_equal_the_former_inline_mlm_code():
    """With no missing p-value the helper writes exactly what gwas/pipeline.py wrote
    for MLM before it existed (MLM output byte-identical), in the same column order."""
    from statsmodels.stats.multitest import multipletests

    from gwas.significance import add_significance_columns
    p = _pvals(seed=11)
    base = pd.DataFrame({"SNP": [f"s{i}" for i in range(p.size)], "PValue": p, "-log10p": -np.log10(p)})
    bonf, meff, custom = 0.05 / p.size, 0.05 / 120, 5e-5
    old = base.copy()
    _rej, _fdr, _, _ = multipletests(old["PValue"].values, method="fdr_bh")
    old["FDR"] = _fdr
    old["Significant_FDR"] = _rej
    old["Significant_Bonf"] = old["PValue"] < bonf
    old["Significant_Meff"] = old["PValue"] < meff
    old["Significant_Custom"] = old["PValue"] < custom
    new = add_significance_columns(base, bonf_thresh=bonf, meff_thresh=meff, custom_thresh=custom)
    pd.testing.assert_frame_equal(new, old)
    assert new.to_csv(index=False) == old.to_csv(index=False)


def test_missing_pvalues_are_never_significant_and_bh_counts_only_the_others():
    from statsmodels.stats.multitest import multipletests

    from gwas.significance import add_significance_columns
    p = _pvals(seed=5)
    p[[0, 7, 50, 51]] = np.nan          # p[0] would have been the strongest hit
    out = add_significance_columns(pd.DataFrame({"PValue": p}), bonf_thresh=1.0,
                                   meff_thresh=1.0, custom_thresh=1.0)
    miss = np.isnan(p)
    for col in ("Significant_FDR", "Significant_Bonf", "Significant_Meff", "Significant_Custom"):
        assert out[col].dtype == bool
        assert not out.loc[miss, col].any(), col
        assert out.loc[~miss, col].all() or col == "Significant_FDR", col   # thresholds of 1
    assert out.loc[miss, "FDR"].isna().all()
    reject, q, _, _ = multipletests(p[~miss], method="fdr_bh")
    assert np.array_equal(out.loc[~miss, "FDR"].to_numpy(), q)
    assert np.array_equal(out.loc[~miss, "Significant_FDR"].to_numpy(), reject)


def test_optional_columns_and_degenerate_tables():
    from gwas.significance import add_significance_columns
    df = pd.DataFrame({"PValue": [1e-9, 0.2, 0.9]})
    out = add_significance_columns(df, bonf_thresh=1e-3)
    assert list(out.columns) == ["PValue", "FDR", "Significant_FDR", "Significant_Bonf"]
    stricter = add_significance_columns(df, bonf_thresh=1e-3, fdr_alpha=1e-12)
    assert out["Significant_FDR"].tolist() == [True, False, False]
    assert stricter["Significant_FDR"].tolist() == [False, False, False]
    for frame in (pd.DataFrame({"PValue": pd.Series([], dtype=float)}),
                  pd.DataFrame({"PValue": [np.nan, np.nan]})):
        got = add_significance_columns(frame, bonf_thresh=0.05, meff_thresh=0.05)
        assert not got["Significant_FDR"].any() and not got["Significant_Meff"].any()
        assert got["FDR"].isna().all() and len(got) == len(frame)


def test_rule_mask_selects_what_the_rule_says_on_the_helpers_columns():
    """SignificanceRule.significant_mask (the rescue / significant-SNP tables) on a table
    the helper wrote selects exactly PValue < the rule's threshold, or the
    Benjamini-Hochberg decision for FDR, so an MLMM or FarmCPU table gives the SNPs its
    rule names."""
    from statsmodels.stats.multitest import multipletests

    from gwas.significance import add_significance_columns
    p = _pvals(seed=8)
    n_snps, meff = 900, 250
    bh = multipletests(p, method="fdr_bh")[0]
    for sig in ("bonferroni", "meff", "fdr", 2e-4):
        r = rule_from_cli_args(SimpleNamespace(sig_thresh=sig), n_snps, meff)
        out = add_significance_columns(
            pd.DataFrame({"PValue": p}), bonf_thresh=0.05 / n_snps, meff_thresh=0.05 / meff,
            custom_thresh=sig if isinstance(sig, float) else None)
        expected = bh if sig == "fdr" else p < r.p_threshold
        assert np.array_equal(r.significant_mask(out), expected), sig
        assert expected.any() and not expected.all(), sig


def test_a_pvalue_equal_to_a_threshold_is_not_significant_and_rows_keep_their_labels():
    """Strict '<' (as the former MLM code), and values stay with their rows on a
    shuffled, gapped index such as the tables get after dropna."""
    from statsmodels.stats.multitest import multipletests

    from gwas.significance import add_significance_columns
    p = _pvals(seed=3)
    bonf, meff, custom = 0.05 / 1000, 0.05 / 300, 1e-3
    p[:4] = [bonf, meff, custom, 1.0]                    # exact ties: not below
    df = pd.DataFrame({"SNP": [f"s{i}" for i in range(p.size)], "PValue": p})
    df = df.sample(frac=1, random_state=0)
    df.index = df.index * 2 + 1                           # gapped, shuffled labels
    out = add_significance_columns(df, bonf_thresh=bonf, meff_thresh=meff, custom_thresh=custom)
    by_snp = out.set_index("SNP")
    assert not by_snp.loc["s0", "Significant_Bonf"] and by_snp.loc["s0", "Significant_Meff"]
    assert not by_snp.loc["s1", "Significant_Meff"] and by_snp.loc["s1", "Significant_Custom"]
    assert not by_snp.loc["s2", "Significant_Custom"]
    pv = df["PValue"].to_numpy()
    reject, q, _, _ = multipletests(pv, method="fdr_bh")
    assert list(out.index) == list(df.index)
    np.testing.assert_array_equal(out["FDR"].to_numpy(), q)
    np.testing.assert_array_equal(out["Significant_FDR"].to_numpy(), reject)
    for col, thr in (("Significant_Bonf", bonf), ("Significant_Meff", meff), ("Significant_Custom", custom)):
        np.testing.assert_array_equal(out[col].to_numpy(), pv < thr)


def test_a_missing_value_in_a_nullable_column_is_not_significant():
    from gwas.significance import add_significance_columns
    df = pd.DataFrame({"PValue": pd.array([pd.NA, 1e-9, 0.5], dtype="Float64")})
    out = add_significance_columns(df, bonf_thresh=1e-3, meff_thresh=1e-3, custom_thresh=1e-3)
    for col in ("Significant_FDR", "Significant_Bonf", "Significant_Meff", "Significant_Custom"):
        assert out[col].dtype == bool, col
        assert out[col].tolist() == [False, True, False], col
    assert np.isnan(out["FDR"].iloc[0])


def test_frame_for_rule_lets_a_tab_custom_threshold_judge_the_pvalues():
    """A Post-GWAS tab's own custom p-value must not be overridden by a stored
    Significant_Custom column written with the run's threshold (MLMM / FarmCPU tables
    carry one); other rules keep their stored column."""
    from gwas.significance import frame_for_rule
    p = np.array([1e-9, 1e-7, 1e-6, 0.3])
    df = pd.DataFrame({"PValue": p, "Significant_Custom": p < 5e-8,
                       "Significant_Bonf": [True, False, False, False]})
    tab = rule_from_streamlit("Custom p-value", 4, None, custom_thresh=1e-5)
    assert tab.significant_mask(df).tolist() == [True, False, False, False]      # the stale column
    assert tab.significant_mask(frame_for_rule(df, tab)).tolist() == [True, True, True, False]
    assert "Significant_Custom" in df.columns                                       # input unchanged
    bonf = rule_from_streamlit("Bonferroni (alpha=0.05)", 4, None)
    assert frame_for_rule(df, bonf) is df
