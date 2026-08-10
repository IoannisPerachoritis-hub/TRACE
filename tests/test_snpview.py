"""T-21/T-23/T-85 — per-SNP view layer unit tests."""
import numpy as np
import pandas as pd

from gwas.snpview import (
    EFFECT_FLAGS,
    collapse_snps_for_plotting,
    effect_flag,
    genotype_class_summary,
    label_genes_by_ld,
    ld_supported_interval,
)


# ---- T-21 §2.3 effect flag ----
def test_effect_flag_all_branches():
    assert effect_flag(0.5, 0.6) == "concordant"            # same sign, |ratio| 0.83
    assert effect_flag(0.2, 0.6) == "structure-sensitive"   # |ratio| 0.33 < 0.5
    assert effect_flag(-0.3, 0.6) == "sign-flip"            # opposite sign
    assert effect_flag(np.nan, 0.6) == "undetermined"       # beta_mlm missing
    assert effect_flag(0.5, 1e-9) == "undetermined"         # |beta_ols| below floor
    assert set([effect_flag(0.5, 0.6)]) <= set(EFFECT_FLAGS)


# ---- T-21 §2.2 genotype class summary ----
def test_genotype_class_summary_counts_and_box_rule():
    g = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, np.nan])
    y = np.arange(10.0)
    s = genotype_class_summary(g, y)
    assert s["classes"][0]["n"] == 5 and s["classes"][0]["draw_box"] is True   # n>=5
    assert s["classes"][1]["n"] == 2 and s["classes"][1]["draw_box"] is False  # n<5 -> points only
    assert s["classes"][2]["n"] == 2
    assert s["n_missing"] == 1                                                  # NaN counted
    assert np.isfinite(s["eta2"])


# ---- T-23 LD-supported interval + gene labelling ----
def _corr_block(rng, n, k, flip=0.02):
    hap = rng.integers(0, 2, size=(n, 2))
    cols = []
    for _ in range(k):
        h = hap.copy()
        fmask = rng.random((n, 2)) < flip
        h[fmask] = 1 - h[fmask]
        cols.append(h.sum(axis=1).astype(float))
    return np.column_stack(cols)


def test_ld_supported_interval_and_labels():
    rng = np.random.default_rng(0)
    n = 120
    # markers at 100,200,300,400,500 on chr1; 200/300/400 correlated to lead(300), 100/500 independent
    block = _corr_block(rng, n, 3)                 # cols -> markers 200,300,400
    indep = rng.binomial(2, 0.3, size=(n, 2)).astype(float)  # 100, 500
    geno = np.column_stack([indep[:, 0], block[:, 0], block[:, 1], block[:, 2], indep[:, 1]])
    positions = np.array([100, 200, 300, 400, 500])
    sid = np.array(["m100", "m200", "m300", "m400", "m500"])
    chroms = np.array(["1"] * 5)
    iv = ld_supported_interval("m300", geno, sid, chroms, positions, r2_thresh=0.6, flank_bp=1000)
    assert iv is not None
    assert iv[0] <= 200 and iv[1] >= 400          # spans the correlated markers
    assert iv[0] >= 200 and iv[1] <= 400          # excludes the independent 100/500

    genes = pd.DataFrame({"Gene_ID": ["Ginside", "Goutside"], "Chr": ["1", "1"],
                          "Start": [250, 900], "End": [350, 950]})
    labels = label_genes_by_ld(genes, iv)
    assert labels["Ginside"] == "ld_supported"
    assert labels["Goutside"] == "distance_only"


def test_label_genes_none_interval_all_distance_only():
    genes = pd.DataFrame({"Gene_ID": ["A", "B"], "Chr": ["1", "1"], "Start": [1, 2], "End": [3, 4]})
    labels = label_genes_by_ld(genes, None)
    assert labels == {"A": "distance_only", "B": "distance_only"}


def test_ld_supported_interval_lead_absent_returns_none():
    geno = np.random.default_rng(1).binomial(2, 0.3, size=(50, 3)).astype(float)
    assert ld_supported_interval("nope", geno, np.array(["a", "b", "c"]),
                                 np.array(["1", "1", "1"]), np.array([1, 2, 3])) is None


# ---- T-85 collapse for plotting ----
def test_collapse_one_row_per_snp_never_reorders():
    rng = np.random.default_rng(2)
    n = 120
    block = _corr_block(rng, n, 2, flip=0.0)       # m200, m300 identical -> r2=1.0
    indep = rng.binomial(2, 0.3, size=(n, 1)).astype(float)
    geno = np.column_stack([block[:, 0], block[:, 1], indep[:, 0]])
    sid = np.array(["m200", "m300", "m900"])
    sig = pd.DataFrame({"SNP": ["m200", "m300", "m900"], "Chr": ["1", "1", "1"],
                        "Pos": [200, 300, 900], "PValue": [1e-9, 1e-8, 1e-7]})
    out = collapse_snps_for_plotting(sig, geno, sid, r2_threshold=0.9, flank_bp=1000)
    assert list(out["SNP"]) == ["m200", "m300", "m900"]        # order preserved, one row each
    assert len(out) == 3
    # m200/m300 collapse to one representative; m900 is its own
    assert out.loc[out["SNP"] == "m300", "Representative_SNP"].iloc[0] == "m200"
    assert out.loc[out["SNP"] == "m900", "Representative_SNP"].iloc[0] == "m900"
