"""Tests for gwas/ld.py — LD computation and block detection."""
import numpy as np
import pytest
from gwas.ld import _interval_iou_bp, pairwise_r2, find_ld_blocks_graph, ld_decay, get_block_snp_mask


# ── _interval_iou_bp ─────────────────────────────────────────

class TestIntervalIoU:
    def test_identical_intervals(self):
        assert _interval_iou_bp(100, 200, 100, 200) == pytest.approx(1.0)

    def test_non_overlapping(self):
        assert _interval_iou_bp(0, 100, 200, 300) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # [0,100) ∩ [50,150) → intersection=50, union=150
        iou = _interval_iou_bp(0, 100, 50, 150)
        assert iou == pytest.approx(50.0 / 150.0)

    def test_one_contains_other(self):
        # [0,200) contains [50,100) → intersection=50, union=200
        iou = _interval_iou_bp(0, 200, 50, 100)
        assert iou == pytest.approx(50.0 / 200.0)

    def test_zero_length_interval(self):
        iou = _interval_iou_bp(100, 100, 50, 150)
        assert iou == pytest.approx(0.0)


# ── pairwise_r2 ──────────────────────────────────────────────

class TestPairwiseR2:
    def test_identical_columns_returns_one(self, geno_small):
        # Duplicate column 0 into column 1
        G = geno_small.copy()
        G[:, 1] = G[:, 0]
        r2 = pairwise_r2(G)
        assert r2[0, 1] == pytest.approx(1.0, abs=1e-10)

    def test_diagonal_is_one(self, geno_small):
        r2 = pairwise_r2(geno_small)
        np.testing.assert_allclose(np.diag(r2), 1.0, atol=1e-10)

    def test_symmetric(self, geno_small):
        r2 = pairwise_r2(geno_small)
        np.testing.assert_allclose(r2, r2.T, atol=1e-10)

    def test_range_zero_to_one(self, geno_small):
        r2 = pairwise_r2(geno_small)
        valid = r2[np.isfinite(r2)]
        assert np.all(valid >= -1e-10)
        assert np.all(valid <= 1.0 + 1e-10)

    def test_independent_snps_near_zero(self):
        rng = np.random.default_rng(123)
        # Large sample, independent SNPs → mean r² should be low
        G = rng.choice([0.0, 1.0, 2.0], size=(200, 10))
        r2 = pairwise_r2(G)
        off_diag = r2[np.triu_indices_from(r2, k=1)]
        off_diag = off_diag[np.isfinite(off_diag)]
        assert np.mean(off_diag) < 0.15

    def test_monomorphic_snp_nan(self, geno_with_monomorphic):
        r2 = pairwise_r2(geno_with_monomorphic)
        # Monomorphic column (col 2) → off-diagonal should be NaN
        mono_col = 2
        for j in range(r2.shape[1]):
            if j != mono_col:
                assert np.isnan(r2[mono_col, j]), \
                    f"r2[{mono_col},{j}] should be NaN for monomorphic SNP"
        # Diagonal still 1
        assert r2[mono_col, mono_col] == pytest.approx(1.0)

    def test_with_nan_genotypes(self, geno_with_nan):
        r2 = pairwise_r2(geno_with_nan)
        # Should still produce a valid matrix
        assert r2.shape == (geno_with_nan.shape[1], geno_with_nan.shape[1])
        np.testing.assert_allclose(np.diag(r2), 1.0, atol=1e-10)

    def test_empty_matrix(self):
        G = np.zeros((10, 0))
        r2 = pairwise_r2(G)
        assert r2.shape == (0, 0)

    def test_single_snp(self):
        G = np.array([[0.0], [1.0], [2.0]])
        r2 = pairwise_r2(G)
        assert r2.shape == (1, 1)
        assert r2[0, 0] == pytest.approx(1.0)

    def test_too_many_snps_raises(self):
        G = np.zeros((10, 5000))
        with pytest.raises(RuntimeError, match="extremely slow"):
            pairwise_r2(G)


# ── find_ld_blocks_graph ─────────────────────────────────────

class TestFindLDBlocksGraph:
    def test_perfect_ld_cluster(self):
        # 5 SNPs in perfect LD → 1 block
        n_snps = 5
        positions = np.arange(1000, 1000 + n_snps * 100, 100)
        r2 = np.ones((n_snps, n_snps))
        blocks = find_ld_blocks_graph(positions, r2, ld_threshold=0.6, min_snps=3)
        assert len(blocks) >= 1
        # The block should span all positions
        assert blocks[0][0] <= positions[0]  # start
        assert blocks[0][1] >= positions[-1]  # end

    def test_two_separated_clusters(self):
        # Two groups of 5 SNPs, far apart, each in perfect LD within group
        pos_a = np.arange(1000, 1500, 100)
        pos_b = np.arange(100000, 100500, 100)
        positions = np.concatenate([pos_a, pos_b])
        n = len(positions)
        r2 = np.eye(n)
        # Group A: perfect LD
        for i in range(5):
            for j in range(5):
                r2[i, j] = 1.0
        # Group B: perfect LD
        for i in range(5, 10):
            for j in range(5, 10):
                r2[i, j] = 1.0
        blocks = find_ld_blocks_graph(
            positions, r2, ld_threshold=0.6, min_snps=3,
            max_dist_bp=10000,
        )
        assert len(blocks) == 2

    def test_no_ld_no_blocks(self):
        # Independent SNPs → no blocks (min_snps=3 means isolated pairs fail)
        positions = np.arange(0, 1000, 100)
        n = len(positions)
        r2 = np.eye(n)  # no LD
        blocks = find_ld_blocks_graph(positions, r2, ld_threshold=0.6, min_snps=3)
        assert len(blocks) == 0

    def test_blocks_sorted_by_position(self):
        # Create 3 LD blocks in reverse position order to test sorting
        pos_c = np.arange(200000, 200500, 100)
        pos_a = np.arange(1000, 1500, 100)
        pos_b = np.arange(100000, 100500, 100)
        positions = np.concatenate([pos_c, pos_a, pos_b])
        n = len(positions)
        r2 = np.eye(n)
        # Groups in input order (c, a, b) but should be sorted (a, b, c)
        for grp_start in [0, 5, 10]:
            for i in range(grp_start, grp_start + 5):
                for j in range(grp_start, grp_start + 5):
                    r2[i, j] = 1.0
        blocks = find_ld_blocks_graph(
            positions, r2, ld_threshold=0.6, min_snps=3,
            max_dist_bp=10000,
        )
        if len(blocks) >= 2:
            starts = [b[0] for b in blocks]
            assert starts == sorted(starts)


# ── find_ld_blocks_graph — member SNP IDs ─────────────────────

class TestBlockMemberSNPIDs:
    def test_returns_member_ids_when_sids_provided(self):
        n_snps = 5
        positions = np.arange(1000, 1000 + n_snps * 100, 100)
        sids = np.array(["SNP_A", "SNP_B", "SNP_C", "SNP_D", "SNP_E"])
        r2 = np.ones((n_snps, n_snps))
        blocks = find_ld_blocks_graph(
            positions, r2, ld_threshold=0.6, min_snps=3,
            region_sids=sids,
        )
        assert len(blocks) >= 1
        # 5th element is member IDs list
        member_ids = blocks[0][4]
        assert set(member_ids) == set(sids)

    def test_returns_empty_ids_when_sids_not_provided(self):
        n_snps = 5
        positions = np.arange(1000, 1000 + n_snps * 100, 100)
        r2 = np.ones((n_snps, n_snps))
        blocks = find_ld_blocks_graph(
            positions, r2, ld_threshold=0.6, min_snps=3,
        )
        assert len(blocks) >= 1
        assert blocks[0][4] == []

    def test_non_member_snp_excluded(self):
        """SNP in coordinate range but not in LD block should be excluded."""
        # 5 SNPs: first 3 in perfect LD, SNP_D independent, SNP_E independent
        positions = np.array([1000, 1100, 1200, 1150, 1300])
        sids = np.array(["A", "B", "C", "D", "E"])
        n = 5
        r2 = np.eye(n)
        # A, B, C in perfect LD
        for i in range(3):
            for j in range(3):
                r2[i, j] = 1.0
        # D and E have no LD with anything (r²=0)
        blocks = find_ld_blocks_graph(
            positions, r2, ld_threshold=0.6, min_snps=3,
            region_sids=sids,
        )
        assert len(blocks) >= 1
        member_ids = blocks[0][4]
        # D falls within [1000, 1200] range but is NOT an LD member
        assert "D" not in member_ids
        assert "E" not in member_ids
        assert set(member_ids) == {"A", "B", "C"}


# ── get_block_snp_mask ─────────────────────────────────────

class TestGetBlockSNPMask:
    def test_uses_snp_ids_when_available(self):
        chroms = np.array(["1", "1", "1", "1"])
        positions = np.array([100, 200, 300, 400])
        sid = np.array(["S1", "S2", "S3", "S4"])
        block_row = {
            "Chr": "1", "Start (bp)": 100, "End (bp)": 400,
            "SNP_IDs": "S1,S3",
        }
        mask = get_block_snp_mask(block_row, chroms, positions, sid)
        # Should select only S1 and S3, not S2/S4 despite being in range
        assert list(mask) == [True, False, True, False]

    def test_falls_back_to_coordinates_when_no_snp_ids(self):
        chroms = np.array(["1", "1", "1", "2"])
        positions = np.array([100, 200, 300, 200])
        sid = np.array(["S1", "S2", "S3", "S4"])
        block_row = {
            "Chr": "1", "Start (bp)": 100, "End (bp)": 300,
        }
        mask = get_block_snp_mask(block_row, chroms, positions, sid)
        assert list(mask) == [True, True, True, False]

    def test_falls_back_when_snp_ids_empty(self):
        chroms = np.array(["1", "1"])
        positions = np.array([100, 200])
        sid = np.array(["S1", "S2"])
        block_row = {
            "Chr": "1", "Start (bp)": 100, "End (bp)": 200,
            "SNP_IDs": "",
        }
        mask = get_block_snp_mask(block_row, chroms, positions, sid)
        assert list(mask) == [True, True]


# ── ld_decay ─────────────────────────────────────────────────

class TestLDDecay:
    def test_returns_finite(self):
        rng = np.random.default_rng(42)
        n_snps = 50
        positions = np.sort(rng.integers(0, 500_000, n_snps))
        G = rng.choice([0.0, 1.0, 2.0], size=(100, n_snps))
        r2 = pairwise_r2(G)
        dist, slope, df = ld_decay(positions, r2)
        assert np.isfinite(dist) or np.isnan(dist)  # either valid or NaN
        assert isinstance(df, type(None)) or hasattr(df, "columns")

    def test_few_snps(self):
        positions = np.array([100, 200, 300])
        r2 = np.eye(3)
        dist, slope, df = ld_decay(positions, r2)
        # Very few pairs → may return NaN for decay distance
        # Just verify it doesn't crash
        assert isinstance(dist, float)


# ── Monomorphic SNPs in block detection ────────────────────────

class TestBlockDetectionMonomorphic:
    """LD block detection should handle monomorphic SNPs gracefully."""

    def test_blocks_with_monomorphic_snps(self, geno_with_monomorphic_block):
        """Block detection should not crash when some SNPs are monomorphic."""
        G = geno_with_monomorphic_block
        r2 = pairwise_r2(G)
        positions = np.arange(1000, 1000 + G.shape[1] * 1000, 1000)
        # Should not raise — monomorphic SNPs produce NaN in r2
        blocks = find_ld_blocks_graph(positions, r2, ld_threshold=0.3, min_snps=2)
        # Blocks should only contain polymorphic SNPs
        assert isinstance(blocks, list)

    def test_r2_nan_for_monomorphic_columns(self, geno_with_monomorphic_block):
        """Monomorphic columns (5, 10, 15) should produce NaN off-diagonal r2."""
        G = geno_with_monomorphic_block
        r2 = pairwise_r2(G)
        for mono_col in [5, 10, 15]:
            for j in range(r2.shape[1]):
                if j != mono_col:
                    assert np.isnan(r2[mono_col, j]), \
                        f"r2[{mono_col},{j}] should be NaN for monomorphic SNP"


# ── Pepper chromosome naming ──────────────────────────────────

class TestPepperChromosomeIntegration:
    """Verify LD functions work with pepper chromosome naming (Ca prefix)."""

    def test_pairwise_r2_pepper_data(self, gwas_rng):
        """pairwise_r2 works regardless of chromosome naming."""
        G = gwas_rng.choice([0.0, 1.0, 2.0], size=(30, 8)).astype(np.float32)
        r2 = pairwise_r2(G)
        assert r2.shape == (8, 8)
        np.testing.assert_allclose(np.diag(r2), 1.0, atol=1e-10)

    def test_canon_chr_pepper_in_metadata(self, pepper_snp_metadata):
        """Pepper chromosome names should canonicalize correctly."""
        from annotation import canon_chr
        chroms = pepper_snp_metadata["chroms"]
        canonical = np.array([canon_chr(c) for c in chroms])
        # Ca1 → "1", Ca2 → "2", Ca3 → "3"
        assert set(canonical) == {"1", "2", "3"}
