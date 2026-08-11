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
