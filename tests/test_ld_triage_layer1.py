"""T-30 — Layer 1 (predictive LD-quality) acceptance. Pure/synthetic; no fixtures."""
import numpy as np
import pandas as pd

from gwas.ld import compute_block_ld_quality, meff_li_ji_from_corr


# ---- meff_li_ji_from_corr (acceptance 2/3/4) ----
def test_meff_from_corr_branches():
    m = 5
    meff, st = meff_li_ji_from_corr(np.ones((m, m)))          # perfectly correlated
    assert st == "ok" and abs(meff - 1.0) < 1e-6
    meff, st = meff_li_ji_from_corr(np.eye(m))                # independent
    assert st == "ok" and abs(meff - m) < 1e-6
    R = np.eye(3); R[0, 1] = R[1, 0] = np.nan                 # non-finite off-diagonal
    meff, st = meff_li_ji_from_corr(R)
    assert st == "nonfinite_r" and np.isnan(meff)
    meff, st = meff_li_ji_from_corr(np.array([[1.0]]))        # too few
    assert st == "too_few_snps" and np.isnan(meff)


def _synthetic():
    rng = np.random.default_rng(0)
    n = 120
    a = rng.integers(0, 3, n).astype(float)                  # block A: 4 identical -> r2=1
    A = np.column_stack([a, a, a, a])
    B = rng.integers(0, 3, (n, 4)).astype(float)             # block B: 4 independent
    geno = np.column_stack([A, B])
    sid = np.array([f"a{i}" for i in range(4)] + [f"b{i}" for i in range(4)])
    positions = np.array([100, 200, 300, 400, 1100, 1200, 1300, 1400])
    chroms = np.array(["1"] * 8)
    blocks = pd.DataFrame({
        "Chr": ["1", "1"], "Start (bp)": [100, 1100], "End (bp)": [400, 1400],
        "Lead SNP": ["a0", "b0"], "SNP_IDs": ["a0,a1,a2,a3", "b0,b1,b2,b3"],
    })
    gwas = pd.DataFrame({"SNP": sid, "PValue": [1e-9, 0.2, 0.3, 0.4, 1e-8, 0.2, 0.3, 0.4]})
    return blocks, chroms, positions, sid, geno, gwas


def test_block_ld_quality_correlated_vs_independent():
    blocks, chroms, positions, sid, geno, gwas = _synthetic()
    out = compute_block_ld_quality(blocks, chroms, positions, sid, geno, gwas)
    assert list(out["Start (bp)"]) == [100, 1100]            # order preserved
    A = out.iloc[0]
    assert abs(A["ldq_r2_mean"] - 1.0) < 1e-9                # perfectly correlated
    assert abs(A["ldq_meff_block"] - 1.0) < 1e-6
    assert A["ldq_lead_in_block"] and A["ldq_n_members"] == 4
    assert A["ldq_r2_estimator"] == "pairwise_r2_lowmiss_imputed"
    assert A["ldq_r2_pairs_finite"] == A["ldq_r2_pairs_total"] == 6
    B = out.iloc[1]
    assert B["ldq_meff_block"] > 3.0                          # ~4 independent SNPs
    assert B["ldq_r2_mean"] < 0.5


def test_lead_absent_no_raise():
    blocks, chroms, positions, sid, geno, gwas = _synthetic()
    blocks.loc[0, "Lead SNP"] = "not_a_snp"                   # lead absent from sid
    blocks.loc[0, "SNP_IDs"] = "a1,a2,a3"                     # and not a member
    out = compute_block_ld_quality(blocks, chroms, positions, sid, geno, gwas)
    assert out.iloc[0]["ldq_lead_in_block"] == False          # noqa: E712 — pandas bool


def test_multi_lead_deterministic():
    blocks, chroms, positions, sid, geno, gwas = _synthetic()
    blocks.loc[0, "Lead SNP"] = "a3;a1;a0"                    # merged multi-lead
    out = compute_block_ld_quality(blocks, chroms, positions, sid, geno, gwas)
    row = out.iloc[0]
    assert row["ldq_n_leads"] == 3
    assert row["ldq_lead_snp"] == "a0"                        # smallest PValue among tokens


def test_lead_estimator_raw_vs_imputed_differ():
    blocks, chroms, positions, sid, geno, gwas = _synthetic()
    raw = geno.copy()
    raw[:20, 1] = np.nan                                      # missingness on a member
    o_imp = compute_block_ld_quality(blocks, chroms, positions, sid, geno, gwas)
    o_raw = compute_block_ld_quality(blocks, chroms, positions, sid, geno, gwas, geno_dosage_raw=raw)
    assert o_imp.iloc[0]["ldq_r2_lead_estimator"] == "imputed_fallback"
    assert o_raw.iloc[0]["ldq_r2_lead_estimator"] == "pairwise_complete_raw"
