"""T-35 — η² comparability columns (pure arithmetic on eta2/df1/df2/F_perm)."""
import numpy as np
import pandas as pd

from gwas.triage import ETA2_COMPARABILITY_COLUMNS, add_eta2_comparability


def _tbl(eta2, f_perm, df1=(2, 3, 4), df2=(160, 159, 158)):
    return pd.DataFrame({"eta2": eta2, "F_perm": f_perm, "df1": df1, "df2": df2})


def test_eta2_bit_identical_and_closed_forms():
    df = _tbl([0.10, 0.30, 0.05], [3.0, 9.0, 1.5])
    out = add_eta2_comparability(df)
    # eta2 untouched (N5)
    assert (out["eta2"].values == df["eta2"].values).all()
    # closed forms, exact
    assert np.allclose(out["eta2_null_expected"], df["df1"] / (df["df1"] + df["df2"]))
    assert np.allclose(out["eta2_adj"], 1.0 - (1.0 - df["eta2"]) * (df["df1"] + df["df2"]) / df["df2"])


def test_eta2_adj_negative_is_stored_not_clipped():
    # eta2 below its null expectation -> negative adjusted effect, kept
    df = _tbl([0.005], [0.3], df1=(4,), df2=(160,))
    out = add_eta2_comparability(df)
    assert out["eta2_adj"].iloc[0] < 0.0


def test_rank_delta_zero_when_concordant_signed_when_not():
    # eta2 and F_perm order the three blocks identically -> all deltas 0
    conc = _tbl([0.1, 0.2, 0.3], [1.0, 2.0, 3.0])
    assert (add_eta2_comparability(conc)["eta2_F_rank_delta"] == 0).all()
    # swap the top two on F only -> the disagreement shows with opposite signs
    disc = _tbl([0.1, 0.2, 0.3], [1.0, 3.0, 2.0])
    d = add_eta2_comparability(disc)["eta2_F_rank_delta"]
    assert d.iloc[0] == 0
    assert d.iloc[1] < 0 and d.iloc[2] > 0            # block2 over-ranked by F; block3 under
    assert d.sum() == 0                               # ranks are a permutation


def test_empty_table_has_columns():
    out = add_eta2_comparability(pd.DataFrame(columns=["eta2", "df1", "df2", "F_perm"]))
    for c in ETA2_COMPARABILITY_COLUMNS:
        assert c in out.columns
    assert len(out) == 0
