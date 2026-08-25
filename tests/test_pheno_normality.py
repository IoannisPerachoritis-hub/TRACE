"""
Tests for the phenotype normality-check page's pure statistics and loader.

Covers the house conventions: happy-path (normal data), monomorphic/constant
column, small-n guards, the Shapiro (n>5000) and D'Agostino (n<8) skips,
transform round-trips, NaN preservation, and the file loader's ID detection.
"""
import io

import numpy as np
import pandas as pd
import pytest

from pages._pheno_qc import load_phenotype_file
from pages._pheno_qc.stats import (
    TRANSFORM_INT,
    TRANSFORM_LOG10,
    TRANSFORM_NONE,
    TRANSFORM_YEOJOHNSON,
    apply_transform,
    normality_summary_table,
    recommend_uniform_method,
    run_normality_tests,
    transform_whole_table,
)


# ── run_normality_tests ───────────────────────────────────────

def test_normality_normal_data(rng):
    """Clean normal data → 'Normal' verdict, high Shapiro p."""
    x = rng.normal(10.0, 2.0, size=300)
    r = run_normality_tests(x)
    assert r["n"] == 300
    assert r["verdict"] == "Normal"
    assert r["shapiro_p"] > 0.05
    assert abs(r["skew"]) < 0.5


def test_normality_skewed_data(rng):
    """Strongly right-skewed positive data → 'Non-normal', low p, high skew."""
    x = rng.lognormal(mean=0.0, sigma=1.0, size=300)
    r = run_normality_tests(x)
    assert r["verdict"] == "Non-normal"
    assert r["shapiro_p"] < 0.05
    assert r["skew"] > 1.0


def test_normality_constant_column():
    """Monomorphic/constant column is guarded, not crashed."""
    r = run_normality_tests(np.full(30, 5.0))
    assert r["verdict"] == "Constant (no variance)"
    assert r["n"] == 30
    assert np.isnan(r["shapiro_p"])


def test_normality_small_n():
    """Fewer than 3 finite values → insufficient-data guard."""
    r = run_normality_tests([1.0, 2.0])
    assert r["verdict"] == "Insufficient (n<3)"
    assert r["n"] == 2


def test_normality_drops_nan(rng):
    """NaN/inf entries are dropped before testing; n reflects finite count."""
    x = rng.normal(0, 1, size=50)
    x[[1, 7, 20]] = np.nan
    x[3] = np.inf
    r = run_normality_tests(x)
    assert r["n"] == 46


def test_shapiro_skipped_large_n(rng):
    """n>5000 → Shapiro skipped (NaN + note), K² and A-D still produced."""
    x = rng.normal(0, 1, size=6000)
    r = run_normality_tests(x)
    assert np.isnan(r["shapiro_p"])
    assert "Shapiro" in r["note"]
    assert np.isfinite(r["k2_p"])
    assert r["ad_normal"] is not None
    assert r["verdict"] in ("Normal", "Non-normal")


def test_dagostino_skipped_small_n(rng):
    """3 <= n < 8 → D'Agostino K² skipped but Shapiro present."""
    x = rng.normal(0, 1, size=5)
    r = run_normality_tests(x)
    assert np.isnan(r["k2_p"])
    assert "D'Agostino" in r["note"]
    assert np.isfinite(r["shapiro_p"])


# ── apply_transform ───────────────────────────────────────────

def test_apply_none_is_identity(rng):
    x = rng.normal(0, 1, size=20)
    np.testing.assert_array_equal(apply_transform(x, TRANSFORM_NONE), x)


def test_apply_log10_positive(rng):
    """All-positive input → pure log10 (no shift, preserves ratios)."""
    x = rng.uniform(1.0, 100.0, size=50)
    out = apply_transform(x, TRANSFORM_LOG10)
    np.testing.assert_allclose(out, np.log10(x))


def test_apply_log10_refuses_nonpositive():
    """Zeros/negatives are REFUSED (raise + note), not auto-shifted -- the panel
    must not preview a log the GWAS run (same helper) would never apply.
    Reconciled with gwas.qc.normalise_phenotype (b850af2 / 4e8991d)."""
    x = np.array([0.0, -2.0, 3.0, 5.0])
    with pytest.raises(ValueError, match="non-positive"):
        apply_transform(x, TRANSFORM_LOG10)


def test_apply_yeojohnson_reduces_skew(rng):
    from scipy.stats import skew
    x = rng.lognormal(0.0, 1.0, size=300)
    out = apply_transform(x, TRANSFORM_YEOJOHNSON)
    assert np.all(np.isfinite(out))
    assert abs(skew(out)) < abs(skew(x))


def test_apply_int_is_symmetric(rng):
    from scipy.stats import skew
    x = rng.lognormal(0.0, 1.0, size=300)
    out = apply_transform(x, TRANSFORM_INT)
    assert abs(skew(out)) < 0.3  # rank-INT yields near-symmetric output


def test_apply_unknown_raises(rng):
    with pytest.raises(ValueError, match="Unknown transform"):
        apply_transform(rng.normal(0, 1, size=10), "bogus")


@pytest.mark.parametrize("method", [TRANSFORM_YEOJOHNSON, TRANSFORM_INT])
def test_transform_preserves_nan(rng, method):
    """Transforms keep NaN positions so output stays index-aligned."""
    x = rng.uniform(1.0, 10.0, size=40)
    x[[2, 9, 30]] = np.nan
    out = apply_transform(x, method)
    assert np.isnan(out[[2, 9, 30]]).all()
    assert np.isfinite(out[np.isfinite(x)]).all()


# ── normality_summary_table ───────────────────────────────────

def test_summary_table_mixed(rng):
    """Table has one row per trait with the expected verdicts and columns."""
    df = pd.DataFrame({
        "normal_trait": rng.normal(0, 1, size=300),
        "skewed_trait": rng.lognormal(0, 1, size=300),
    })
    tbl = normality_summary_table(df, ["normal_trait", "skewed_trait"])
    assert list(tbl["Trait"]) == ["normal_trait", "skewed_trait"]
    assert set(tbl["Verdict"]) == {"Normal", "Non-normal"}
    for col in ("Trait", "N", "Shapiro p", "Skew", "Verdict"):
        assert col in tbl.columns
    assert "Suggested transform" not in tbl.columns


# ── transform_whole_table (one uniform method for all traits) ─

def _panel_df(rng):
    """A 3-trait panel: normal, positive-skewed (with NaN + a zero), neg-skewed."""
    pos_skew = rng.lognormal(0.0, 1.0, size=300)
    pos_skew[[4, 40]] = np.nan   # missing values must survive
    pos_skew[7] = 0.0            # a zero → exercises log10 auto-shift
    neg_skew = rng.gamma(2.0, size=300)
    neg_skew = neg_skew - neg_skew.mean()  # has negatives
    return pd.DataFrame(
        {
            "normal_trait": rng.normal(0.0, 1.0, size=300),
            "pos_skew": pos_skew,
            "neg_skew": neg_skew,
        },
        index=[f"S{i:03d}" for i in range(300)],
    )


def test_transform_whole_table_int_normalizes_all(rng):
    """rank-INT applied uniformly → every trait becomes Normal; index/NaN kept."""
    df = _panel_df(rng)
    cols = list(df.columns)
    transformed, report = transform_whole_table(df, cols, TRANSFORM_INT)

    assert list(report.columns) == ["Trait", "N", "Applied", "Note"]
    assert list(report["Trait"]) == cols
    assert (report["Applied"] == TRANSFORM_INT).all()
    # shape + index preserved
    assert transformed.shape == df.shape
    assert transformed.index.equals(df.index)
    # NaN positions preserved
    np.testing.assert_array_equal(
        np.isnan(transformed["pos_skew"].to_numpy()),
        np.isnan(df["pos_skew"].to_numpy()),
    )
    # every trait now passes normality
    tbl = normality_summary_table(transformed, cols)
    assert set(tbl["Verdict"]) == {"Normal"}


def test_transform_whole_table_log10_refuses_negatives(rng):
    """log10 uniform: a column with a zero/negative is REFUSED and noted (left
    raw), matching the pipeline's refuse-not-shift semantics."""
    df = _panel_df(rng)
    transformed, report = transform_whole_table(df, list(df.columns), TRANSFORM_LOG10)
    rep = report.set_index("Trait")
    assert "non-positive" in rep.loc["neg_skew", "Note"].lower()
    assert "skipped" in rep.loc["neg_skew", "Applied"].lower()
    # refused -> left raw, so still finite
    assert np.isfinite(transformed["neg_skew"].to_numpy()).all()


def test_recommend_uniform_method_int_maxes(rng):
    """rank-INT normalizes every (non-degenerate) trait by construction."""
    df = _panel_df(rng)
    cols = list(df.columns)
    counts = recommend_uniform_method(df, cols)
    assert set(counts.keys()) == {TRANSFORM_LOG10, TRANSFORM_YEOJOHNSON, TRANSFORM_INT}
    assert counts[TRANSFORM_INT] == len(cols)
    assert counts[TRANSFORM_INT] >= counts[TRANSFORM_LOG10]


# ── load_phenotype_file ───────────────────────────────────────

def test_loader_detects_named_id_column():
    csv = b"Accession,Trait1,Trait2\nS1,1.0,2.0\nS2,3.0,4.0\n"
    pheno, id_col, latin1 = load_phenotype_file(io.BytesIO(csv))
    assert id_col == "Accession"
    assert list(pheno.columns) == ["Trait1", "Trait2"]
    assert pheno.index.tolist() == ["S1", "S2"]
    assert latin1 is False


def test_loader_falls_back_to_first_column():
    csv = b"strain,Trait1\nS1,1.0\nS2,2.0\n"
    pheno, id_col, _ = load_phenotype_file(io.BytesIO(csv))
    assert id_col == "strain"
    assert pheno.index.tolist() == ["S1", "S2"]


def test_loader_tab_separated():
    tsv = b"Sample\tYield\nS1\t5.0\nS2\t6.0\n"
    pheno, id_col, _ = load_phenotype_file(io.BytesIO(tsv))
    assert id_col == "Sample"
    assert list(pheno.columns) == ["Yield"]
