"""Tests for gwas/plotting.py — statistical functions only (not visual)."""
import numpy as np
import pytest
from gwas.plotting import compute_lambda_gc, compute_r2_to_lead, compute_meff_li_ji


# ── compute_lambda_gc ────────────────────────────────────────

class TestLambdaGC:
    def test_null_distribution(self):
        # Uniform p-values → λ ≈ 1.0
        rng = np.random.default_rng(42)
        pvals = rng.uniform(0, 1, 5000)
        lam = compute_lambda_gc(pvals)
        assert 0.85 < lam < 1.15

    def test_inflated(self):
        # All very small p-values → λ > 1
        pvals = np.full(500, 1e-10)
        lam = compute_lambda_gc(pvals)
        assert lam > 5.0

    def test_few_snps_nan(self):
        pvals = np.array([0.01, 0.05, 0.1])
        lam = compute_lambda_gc(pvals)
        assert np.isnan(lam)

    def test_invalid_pvals_filtered(self):
        # Mix of valid and invalid p-values
        rng = np.random.default_rng(42)
        pvals = rng.uniform(0, 1, 200)
        pvals_dirty = np.concatenate([pvals, [np.nan, 0.0, -1.0, 2.0]])
        lam_clean = compute_lambda_gc(pvals)
        lam_dirty = compute_lambda_gc(pvals_dirty)
        # Should give similar results after filtering
        assert abs(lam_clean - lam_dirty) < 0.1

    def test_boundary_fifty_snps(self):
        rng = np.random.default_rng(42)
        pvals = rng.uniform(0, 1, 50)
        lam = compute_lambda_gc(pvals)
        assert np.isfinite(lam)  # exactly 50 should work


# ── compute_r2_to_lead ───────────────────────────────────────

class TestComputeR2ToLead:
    def test_lead_self_is_one(self):
        rng = np.random.default_rng(42)
        n_samples, n_snps = 50, 20
        G = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps))
        sid = np.array([f"snp_{i}" for i in range(n_snps)])
        snp_mask = np.ones(n_snps, dtype=bool)
        lead = "snp_5"

        r2_vals, lead_idx = compute_r2_to_lead(G, sid, lead, snp_mask)
        # r² of lead SNP to itself should be 1.0
        # lead_idx is the global index; within snp_mask it maps to position
        mask_indices = np.where(snp_mask)[0]
        local_lead = np.where(mask_indices == lead_idx)[0][0]
        assert r2_vals[local_lead] == pytest.approx(1.0, abs=1e-10)

    def test_output_length(self):
        rng = np.random.default_rng(42)
        G = rng.choice([0.0, 1.0, 2.0], size=(30, 15))
        sid = np.array([f"s{i}" for i in range(15)])
        snp_mask = np.array([True]*10 + [False]*5)
        r2_vals, _ = compute_r2_to_lead(G, sid, "s3", snp_mask)
        assert len(r2_vals) == 10  # sum of mask

    def test_unknown_lead_raises(self):
        G = np.ones((10, 5))
        sid = np.array(["a", "b", "c", "d", "e"])
        snp_mask = np.ones(5, dtype=bool)
        with pytest.raises(ValueError):
            compute_r2_to_lead(G, sid, "nonexistent", snp_mask)


# ── compute_meff_li_ji ───────────────────────────────────────

class TestComputeMeffLiJi:

    def test_independent_snps_returns_near_m(self):
        """Independent columns → meff close to m (Li & Ji may slightly exceed m)."""
        rng = np.random.default_rng(42)
        G = rng.choice([0.0, 1.0, 2.0], size=(100, 20))
        meff, eigs = compute_meff_li_ji(G)
        assert meff >= 15  # at least 75% of m
        assert meff <= 22  # Li & Ji can slightly exceed m due to fractional contributions

    def test_perfectly_correlated_returns_low(self):
        """All columns identical → meff very small (1-2)."""
        col = np.arange(50, dtype=float)
        G = np.tile(col.reshape(-1, 1), (1, 20))
        meff, eigs = compute_meff_li_ji(G)
        assert meff <= 2  # one dominant eigenvalue + possible fractional remainder

    def test_returns_int(self):
        rng = np.random.default_rng(42)
        G = rng.choice([0.0, 1.0, 2.0], size=(50, 30))
        meff, eigs = compute_meff_li_ji(G)
        assert isinstance(meff, int)

    def test_meff_between_1_and_m(self):
        rng = np.random.default_rng(42)
        G = rng.choice([0.0, 1.0, 2.0], size=(50, 30))
        meff, _ = compute_meff_li_ji(G)
        assert 1 <= meff <= 30

    def test_eigenvalues_nonnegative_descending(self):
        rng = np.random.default_rng(42)
        G = rng.choice([0.0, 1.0, 2.0], size=(50, 30))
        _, eigs = compute_meff_li_ji(G)
        assert np.all(eigs >= -1e-10)  # clipped to 0
        assert np.all(np.diff(eigs) <= 1e-10)  # descending

    def test_dual_matrix_large_m(self):
        """m >> n uses the dual-matrix trick — should give exact result."""
        rng = np.random.default_rng(42)
        G = rng.choice([0.0, 1.0, 2.0], size=(30, 500))
        meff, eigs = compute_meff_li_ji(G)
        assert 1 <= meff <= 500
        # Eigenvalues should have at most n=30 non-zero entries
        assert eigs.shape[0] == 30  # dual matrix is 30×30

    def test_dual_matches_direct_when_n_lt_m(self):
        """Dual-matrix result equals direct m×m computation for moderate sizes."""
        rng = np.random.default_rng(42)
        G = rng.choice([0.0, 1.0, 2.0], size=(40, 80))
        meff_dual, _ = compute_meff_li_ji(G)
        # Brute-force: compute via full m×m correlation matrix
        mu = G.mean(axis=0)
        sd = G.std(axis=0)
        sd[sd < 1e-8] = 1.0
        Z = (G - mu) / sd
        C = (Z.T @ Z) / 40.0
        np.fill_diagonal(C, 1.0)
        eigs_full = np.linalg.eigvalsh(C)
        eigs_full = np.clip(eigs_full, 0.0, None)[::-1]
        meff_direct = 0.0
        for lam in eigs_full:
            if lam >= 1.0:
                meff_direct += 1.0 + (lam % 1.0)
            elif lam > 0:
                meff_direct += lam
        # Allow ±1 due to floating-point rounding at ceil boundary
        assert abs(meff_dual - int(np.ceil(meff_direct))) <= 1
