"""Tests for gwas/haplotype.py — haplotype block tests and permutation."""
import numpy as np
import pandas as pd
import pytest
from gwas.haplotype import (
    block_test_lm_with_pcs,
    freedman_lane_perm_pvalue,
    compute_haplotype_effects,
    run_haplotype_block_gwas,
    _eta2_confint,
    _lead_snp_partial_r2,
    _omega2_from_f,
)
from gwas.utils import stable_seed


# ── block_test_lm_with_pcs ───────────────────────────────────

class TestBlockTestLM:
    def test_strong_effect(self):
        # y perfectly determined by group → large F, small p
        y = np.array([1.0]*10 + [5.0]*10 + [9.0]*10)
        groups = np.array(["A"]*10 + ["B"]*10 + ["C"]*10)
        F, p, df1, df2 = block_test_lm_with_pcs(y, groups)
        assert F > 50
        assert p < 0.001

    def test_no_effect(self):
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 30)
        groups = np.array(["A"]*10 + ["B"]*10 + ["C"]*10)
        F, p, df1, df2 = block_test_lm_with_pcs(y, groups)
        # Should usually not be significant with random data
        assert np.isfinite(F)
        assert 0 <= p <= 1

    def test_single_group_returns_nan(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        groups = np.array(["A", "A", "A", "A"])
        F, p, df1, df2 = block_test_lm_with_pcs(y, groups)
        assert np.isnan(F)
        assert df1 == 0

    def test_with_pcs(self):
        rng = np.random.default_rng(42)
        n = 30
        y = np.array([1.0]*10 + [5.0]*10 + [9.0]*10)
        groups = np.array(["A"]*10 + ["B"]*10 + ["C"]*10)
        pcs = rng.normal(0, 1, (n, 2))
        F, p, df1, df2 = block_test_lm_with_pcs(y, groups, pcs=pcs)
        assert np.isfinite(F)
        assert 0 <= p <= 1


# ── freedman_lane_perm_pvalue ────────────────────────────────

class TestFreedmanLane:
    def test_deterministic_with_seed(self):
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 20)
        groups = np.array(["A"]*7 + ["B"]*7 + ["C"]*6)
        F1, p1 = freedman_lane_perm_pvalue(y, groups, n_perm=100, seed=99)
        F2, p2 = freedman_lane_perm_pvalue(y, groups, n_perm=100, seed=99)
        assert F1 == pytest.approx(F2)
        assert p1 == pytest.approx(p2)

    def test_strong_signal_low_p(self):
        # y is perfectly separated by group → p should be very small
        y = np.array([0.0]*10 + [10.0]*10 + [20.0]*10)
        groups = np.array(["A"]*10 + ["B"]*10 + ["C"]*10)
        F, p = freedman_lane_perm_pvalue(y, groups, n_perm=200, seed=0)
        assert p < 0.05

    def test_null_signal_high_p(self):
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 30)
        groups = np.array(["A"]*10 + ["B"]*10 + ["C"]*10)
        F, p = freedman_lane_perm_pvalue(y, groups, n_perm=200, seed=0)
        # With random data, p should generally not be tiny
        assert p > 0.01

    def test_with_covariates(self):
        rng = np.random.default_rng(42)
        n = 30
        y = np.array([0.0]*10 + [10.0]*10 + [20.0]*10)
        groups = np.array(["A"]*10 + ["B"]*10 + ["C"]*10)
        pcs = rng.normal(0, 1, (n, 2))
        F, p = freedman_lane_perm_pvalue(y, groups, pcs=pcs, n_perm=100, seed=0)
        assert np.isfinite(F)
        assert 0 <= p <= 1

    def test_single_group_nan(self):
        y = np.array([1.0, 2.0, 3.0])
        groups = np.array(["A", "A", "A"])
        F, p = freedman_lane_perm_pvalue(y, groups, n_perm=50, seed=0)
        assert np.isnan(F)


# ── compute_haplotype_effects ────────────────────────────────

class TestComputeHaplotypeEffects:
    def test_perfect_separation(self):
        y = np.array([1.0, 1.0, 5.0, 5.0, 9.0, 9.0])
        g = np.array(["A", "A", "B", "B", "C", "C"])
        result = compute_haplotype_effects(y, g)
        assert result["eta2"] == pytest.approx(1.0, abs=1e-10)

    def test_no_effect(self):
        y = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        g = np.array(["A", "A", "B", "B", "C", "C"])
        result = compute_haplotype_effects(y, g)
        # All values identical → ss_total = 0 → eta2 = 0/0 = NaN or 0
        assert result["eta2"] == 0.0 or np.isnan(result["eta2"])

    def test_eta2_range(self):
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 30)
        g = np.array(["A"]*10 + ["B"]*10 + ["C"]*10)
        result = compute_haplotype_effects(y, g)
        assert 0.0 <= result["eta2"] <= 1.0

    def test_hap_stats_structure(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        g = np.array(["A", "A", "A", "B", "B", "B"])
        result = compute_haplotype_effects(y, g)
        assert "eta2" in result
        assert "hap_stats" in result
        for hap, stats in result["hap_stats"].items():
            assert "n" in stats
            assert "mean" in stats
            assert "se" in stats

    def test_few_samples_nan(self):
        y = np.array([1.0, 2.0])
        g = np.array(["A", "B"])
        result = compute_haplotype_effects(y, g)
        assert np.isnan(result["eta2"])


# ── run_haplotype_block_gwas: one design for every statistic ───────

def _cov_panel(seed=7, n=90):
    """Two tested blocks on chromosome 1 (three and four haplotype groups; four samples of
    block 1 carry a rare haplotype, "Other"), three PC columns and one covariate, a phenotype
    that depends on all of them."""
    rng = np.random.default_rng(seed)
    ids = np.array([f"S{i:03d}" for i in range(n)])
    g1 = np.repeat([0, 1, 2], n // 3)
    g2 = rng.integers(0, 4, n)
    G = np.zeros((n, 7))
    G[:, 0:3] = np.array([[0, 0, 2], [2, 2, 0], [0, 2, 2]], float)[g1]
    G[:4, 0:3] = [2.0, 0.0, 2.0]                         # a rare haplotype -> Other
    G[:, 3:6] = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [2, 2, 2]], float)[g2]
    G[:, 6] = rng.integers(0, 3, n)                      # a SNP outside both blocks
    sid = np.array([f"s{j}" for j in range(7)])
    pos = np.array([1000, 2000, 3000, 6000, 7000, 8000, 20000], float)
    blocks = pd.DataFrame({
        "Chr": ["1", "1"], "Start (bp)": [1000, 6000], "End (bp)": [3000, 8000],
        "lead_snp": ["s1", "s4"], "SNP_IDs": ["s0,s1,s2", "s3,s4,s5"],
    })
    pcs = rng.normal(0, 1, (n, 3))
    cov = rng.normal(0, 1, (n, 1))
    y = 0.4 * g1 + 0.3 * g2 + 0.8 * pcs[:, 0] + 0.6 * cov[:, 0] + rng.normal(0, 1, n)
    return dict(blocks=blocks, chroms=np.array(["1"] * 7), pos=pos, G=G, sid=sid,
                geno_df=pd.DataFrame(G, index=ids), pheno=pd.DataFrame({"t": y}, index=ids),
                pcs=pcs, cov=cov, ids=ids)


def _run_cov_panel(p, user_covar=None, n_perm=199):
    return run_haplotype_block_gwas(
        haplo_df=p["blocks"], chroms=p["chroms"], positions=p["pos"], geno_imputed=p["G"],
        sid=p["sid"], geno_df=p["geno_df"], pheno_df=p["pheno"], trait_col="t", pcs=p["pcs"],
        n_pcs_used=2, n_perm=n_perm, user_covar=user_covar, hap_test="pcs")


def _expected_from_design(p, res, tables, design, n_perm=199):
    """Each tested block's statistics recomputed from its tested samples (the test's own
    table, in its order) with the functions the test calls, on one ``design`` (a callable:
    the block's samples -> the covariate columns, intercept excluded)."""
    rows = []
    for _, r in res.iterrows():
        key = f"{r['Chr']}:{int(r['Start'])}-{int(r['End'])}"
        tab = tables[key]
        tab = tab[tab["in_haplotype_test"]]
        samples = tab["Sample"].astype(str).to_numpy()
        g = np.asarray(tab["Haplotype"].astype(str).to_numpy(dtype=object), dtype=str)
        y = p["pheno"]["t"].reindex(samples).to_numpy(dtype=float)
        C = design(samples)
        F, pv, df1, df2 = block_test_lm_with_pcs(y, g, pcs=C)
        F_obs, p_perm = freedman_lane_perm_pvalue(
            y, g, pcs=C, n_perm=n_perm,
            seed=stable_seed(str(r["Chr"]), int(r["Start"]), int(r["End"]), "t", "FL"))
        X = np.column_stack([np.ones(len(y)), C])
        resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        lo, hi = _eta2_confint(F, df1, df2)
        rows.append({
            "F_param": F, "PValue_param": pv, "df1": df1, "df2": df2, "F_perm": F_obs,
            "P_perm": p_perm, "eta2": compute_haplotype_effects(resid, g)["eta2"],
            "eta2_ci_low": lo, "eta2_ci_high": hi, "omega2": _omega2_from_f(F, df1, df2),
            "lead_snp_r2": _lead_snp_partial_r2(r["lead_snp"], p["sid"], p["G"],
                                                p["geno_df"].index.astype(str), samples,
                                                resid, X),
        })
    return pd.DataFrame(rows)


_STATS = ["F_param", "PValue_param", "df1", "df2", "F_perm", "P_perm", "eta2",
          "eta2_ci_low", "eta2_ci_high", "omega2", "lead_snp_r2"]


class TestBlockGwasCovariates:
    def test_without_covariates_every_statistic_is_the_pcs_composition(self):
        """With no user covariates every statistic is computed exactly as before the fix:
        the parametric test, the permutation, eta2 and lead_snp_r2 all on [1, PCs]. The
        values are compared exactly (the same functions on the same arrays)."""
        p = _cov_panel()
        res, tables = _run_cov_panel(p, user_covar=None)
        assert len(res) == 2
        pcs_df = pd.DataFrame(p["pcs"], index=p["ids"])
        exp = _expected_from_design(p, res, tables,
                                    lambda s: pcs_df.reindex(s).iloc[:, :2].to_numpy(float))
        for c in _STATS:
            assert res[c].tolist() == exp[c].tolist(), c

    def test_with_covariates_every_statistic_uses_pcs_and_covariates(self):
        """With user covariates the parametric test, the permutation, eta2 (with its CI and
        omega2) and lead_snp_r2 all use one design, [1, PCs, covariates]."""
        p = _cov_panel()
        res, tables = _run_cov_panel(p, user_covar=p["cov"])
        pcs_df = pd.DataFrame(p["pcs"], index=p["ids"])
        cov_df = pd.DataFrame(p["cov"], index=p["ids"])
        exp = _expected_from_design(
            p, res, tables,
            lambda s: np.column_stack([pcs_df.reindex(s).iloc[:, :2].to_numpy(float),
                                       cov_df.reindex(s).to_numpy(float)]))
        for c in _STATS:
            assert res[c].tolist() == exp[c].tolist(), c
        # and the covariate changes the test (it explains part of the phenotype)
        base, _ = _run_cov_panel(p, user_covar=None)
        assert not np.allclose(res["F_param"], base["F_param"])

    def test_a_missing_covariate_value_is_an_error(self):
        """A tested sample without a covariate value stops the test (it used to fall back to
        the PCs alone for that block, silently)."""
        p = _cov_panel()
        cov = p["cov"].copy()
        cov[10, 0] = np.nan                                  # a sample of both blocks' tests
        with pytest.raises(ValueError, match="covariate"):
            _run_cov_panel(p, user_covar=cov)
