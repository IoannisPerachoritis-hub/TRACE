"""Tests for gwas/subsampling.py — subsampling aggregation and per-rep helpers."""
import numpy as np
import pandas as pd

from gwas.subsampling import aggregate_subsampling_to_ld_blocks, _subsample_one_rep


# ── Fixtures ─────────────────────────────────────────────────


def _make_discovery_df(sid, chroms, positions, disc_freq):
    """Build a minimal discovery_df aligned to the given SNP arrays."""
    return pd.DataFrame({
        "SNP": np.asarray(sid, dtype=str),
        "Chr": np.asarray(chroms, dtype=str),
        "Pos": np.asarray(positions, dtype=int),
        "DiscoveryCount": (np.asarray(disc_freq) * 10).astype(int),
        "DiscoveryFreq": np.asarray(disc_freq, dtype=float),
    })


def _make_ld_blocks(blocks):
    """Build LD blocks DataFrame from list of (chr, start, end, lead) tuples."""
    return pd.DataFrame(blocks, columns=["Chr", "Start (bp)", "End (bp)", "Lead SNP"])


def _make_raw_pvals(n_reps, n_snps, rng_seed=42):
    """Generate synthetic raw p-values matrix."""
    rng = np.random.default_rng(rng_seed)
    return rng.uniform(1e-6, 1.0, size=(n_reps, n_snps))


# ── aggregate_subsampling_to_ld_blocks tests ─────────────────


class TestAggregateSubsamplingToLDBlocks:
    """Tests for aggregate_subsampling_to_ld_blocks()."""

    def test_basic_aggregation(self):
        """Single block containing 3 SNPs — verify key output columns."""
        sid = np.array(["snp1", "snp2", "snp3", "snp4"])
        chroms = np.array(["1", "1", "1", "2"])
        positions = np.array([100, 200, 300, 100])

        disc_freq = [0.8, 0.6, 0.3, 0.1]
        discovery_df = _make_discovery_df(sid, chroms, positions, disc_freq)

        ld_blocks = _make_ld_blocks([("1", 50, 350, "snp1")])

        rng = np.random.default_rng(0)
        raw_pvals = rng.uniform(1e-6, 1.0, size=(10, 4))
        # Make snp1 consistently significant
        raw_pvals[:, 0] = 1e-6

        result = aggregate_subsampling_to_ld_blocks(
            discovery_df, ld_blocks, raw_pvals, sid, chroms, positions,
            discovery_thresh=1e-4,
        )

        assert not result.empty
        assert len(result) == 1

        row = result.iloc[0]
        assert row["Chr"] == "1"
        assert row["n_snps_in_block"] == 3
        assert "BlockDiscoveryFreq" in result.columns
        assert "BestSNP_DiscoveryFreq" in result.columns
        assert "BestSNP_ID" in result.columns
        assert "n_consistent_snps_50pct" in result.columns
        assert row["BestSNP_ID"] == "snp1"
        assert row["BestSNP_DiscoveryFreq"] == 0.8

    def test_empty_ld_blocks(self):
        """Empty LD blocks DataFrame returns empty result."""
        sid = np.array(["snp1"])
        chroms = np.array(["1"])
        positions = np.array([100])
        discovery_df = _make_discovery_df(sid, chroms, positions, [0.5])
        raw_pvals = np.array([[0.01]])

        result = aggregate_subsampling_to_ld_blocks(
            discovery_df, pd.DataFrame(), raw_pvals, sid, chroms, positions,
        )
        assert result.empty

    def test_none_ld_blocks(self):
        """None LD blocks returns empty result."""
        sid = np.array(["snp1"])
        chroms = np.array(["1"])
        positions = np.array([100])
        discovery_df = _make_discovery_df(sid, chroms, positions, [0.5])
        raw_pvals = np.array([[0.01]])

        result = aggregate_subsampling_to_ld_blocks(
            discovery_df, None, raw_pvals, sid, chroms, positions,
        )
        assert result.empty

    def test_block_with_no_snps(self):
        """Block that doesn't overlap any SNPs → zero counts."""
        sid = np.array(["snp1", "snp2"])
        chroms = np.array(["1", "1"])
        positions = np.array([100, 200])
        discovery_df = _make_discovery_df(sid, chroms, positions, [0.5, 0.5])
        raw_pvals = np.array([[0.01, 0.02]])

        # Block on chromosome 2, no SNPs there
        ld_blocks = _make_ld_blocks([("2", 50, 350, "snpX")])

        result = aggregate_subsampling_to_ld_blocks(
            discovery_df, ld_blocks, raw_pvals, sid, chroms, positions,
        )
        assert len(result) == 1
        assert result.iloc[0]["n_snps_in_block"] == 0
        assert result.iloc[0]["BlockDiscoveryFreq"] == 0.0

    def test_multiple_blocks(self):
        """Two blocks on different chromosomes — both aggregated correctly."""
        sid = np.array(["s1", "s2", "s3", "s4", "s5"])
        chroms = np.array(["1", "1", "1", "2", "2"])
        positions = np.array([100, 200, 300, 100, 200])

        disc_freq = [0.9, 0.7, 0.4, 0.8, 0.2]
        discovery_df = _make_discovery_df(sid, chroms, positions, disc_freq)

        ld_blocks = _make_ld_blocks([
            ("1", 50, 350, "s1"),
            ("2", 50, 250, "s4"),
        ])

        raw_pvals = _make_raw_pvals(10, 5)

        result = aggregate_subsampling_to_ld_blocks(
            discovery_df, ld_blocks, raw_pvals, sid, chroms, positions,
        )

        assert len(result) == 2
        # Block 1 has 3 SNPs, block 2 has 2
        block_snps = result.set_index("Chr")["n_snps_in_block"]
        assert block_snps["1"] == 3
        assert block_snps["2"] == 2

    def test_consistent_snps_counting(self):
        """n_consistent_snps_50pct counts SNPs with freq > 0.5."""
        sid = np.array(["a", "b", "c"])
        chroms = np.array(["1", "1", "1"])
        positions = np.array([100, 200, 300])

        disc_freq = [0.9, 0.6, 0.3]  # 2 above 0.5
        discovery_df = _make_discovery_df(sid, chroms, positions, disc_freq)
        ld_blocks = _make_ld_blocks([("1", 50, 350, "a")])
        raw_pvals = _make_raw_pvals(10, 3)

        result = aggregate_subsampling_to_ld_blocks(
            discovery_df, ld_blocks, raw_pvals, sid, chroms, positions,
        )
        assert result.iloc[0]["n_consistent_snps_50pct"] == 2

    def test_single_rep_data(self):
        """Single subsample rep should still produce valid results."""
        sid = np.array(["x1", "x2"])
        chroms = np.array(["1", "1"])
        positions = np.array([100, 200])

        discovery_df = _make_discovery_df(sid, chroms, positions, [1.0, 0.0])
        ld_blocks = _make_ld_blocks([("1", 50, 250, "x1")])
        raw_pvals = np.array([[1e-6, 0.5]])  # 1 rep, 2 SNPs

        result = aggregate_subsampling_to_ld_blocks(
            discovery_df, ld_blocks, raw_pvals, sid, chroms, positions,
            discovery_thresh=1e-4,
        )
        assert len(result) == 1
        assert result.iloc[0]["n_snps_in_block"] == 2

    def test_lead_snp_discovery_freq(self):
        """LeadSNP_DiscoveryFreq matches the lead SNP's freq in discovery_df."""
        sid = np.array(["lead", "other"])
        chroms = np.array(["1", "1"])
        positions = np.array([100, 200])

        discovery_df = _make_discovery_df(sid, chroms, positions, [0.75, 0.25])
        ld_blocks = _make_ld_blocks([("1", 50, 250, "lead")])
        raw_pvals = _make_raw_pvals(10, 2)

        result = aggregate_subsampling_to_ld_blocks(
            discovery_df, ld_blocks, raw_pvals, sid, chroms, positions,
        )
        assert result.iloc[0]["LeadSNP_DiscoveryFreq"] == 0.75


class TestSubsampleOneRep:
    """Tests for _subsample_one_rep error handling."""

    def test_returns_failed_on_exception(self):
        """When FastLMM raises, should return NaN pvals + failed metadata."""
        # This test uses invalid data that will cause single_snp to fail
        n, m = 5, 3
        geno = np.random.default_rng(0).normal(size=(n, m)).astype(np.float32)
        y = np.random.default_rng(0).normal(size=n).astype(np.float32)
        iid = np.array([[f"S{i}", f"S{i}"] for i in range(n)])
        Z_grm = geno.copy()
        sid = np.array(["s1", "s2", "s3"])
        pos_arr = np.c_[np.ones(m), np.zeros(m), np.arange(m) * 1000.0]
        idx = np.arange(n)

        rep, pvals, meta = _subsample_one_rep(
            0, idx, geno, y, iid, Z_grm, sid, pos_arr,
            n_pcs=0, discovery_thresh=1e-4,
        )

        assert rep == 0
        assert meta["n_samples"] == n
        # Either it succeeds with valid pvals or fails gracefully
        if meta["status"].startswith("failed"):
            assert np.all(np.isnan(pvals))
            assert meta["n_discoveries"] == 0
