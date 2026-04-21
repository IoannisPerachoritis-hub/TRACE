"""Tests for gwas/qc.py — QC filters and allele frequency."""
import numpy as np
import pandas as pd
import pytest
from gwas.qc import allele_freq_from_called_dosage, valid_af_mask_from_called_dosage, _pipeline_snp_qc


# ── allele_freq_from_called_dosage ───────────────────────────

class TestAlleleFreq:
    def test_all_homref(self):
        G = np.zeros((10, 1))
        p = allele_freq_from_called_dosage(G)
        assert p[0] == pytest.approx(0.0)

    def test_all_homalt(self):
        G = np.full((10, 1), 2.0)
        p = allele_freq_from_called_dosage(G)
        assert p[0] == pytest.approx(1.0)

    def test_all_het(self):
        G = np.ones((10, 1))
        p = allele_freq_from_called_dosage(G)
        assert p[0] == pytest.approx(0.5)

    def test_known_mix(self):
        # [0, 1, 2] → AC = 3, AN = 6, p = 0.5
        G = np.array([[0.0], [1.0], [2.0]])
        p = allele_freq_from_called_dosage(G)
        assert p[0] == pytest.approx(0.5)

    def test_with_nan(self):
        # NaN sample excluded: [0, NaN, 2] → AC=2, AN=4, p=0.5
        G = np.array([[0.0], [np.nan], [2.0]])
        p = allele_freq_from_called_dosage(G)
        assert p[0] == pytest.approx(0.5)

    def test_all_nan_column(self):
        G = np.full((5, 1), np.nan)
        p = allele_freq_from_called_dosage(G)
        assert np.isnan(p[0])

    def test_multiple_snps(self):
        G = np.array([
            [0.0, 2.0],
            [0.0, 2.0],
            [0.0, 2.0],
        ])
        p = allele_freq_from_called_dosage(G)
        assert p[0] == pytest.approx(0.0)
        assert p[1] == pytest.approx(1.0)


# ── valid_af_mask_from_called_dosage ─────────────────────────

class TestValidAFMask:
    def test_monomorphic_false(self):
        G = np.zeros((10, 1))
        mask = valid_af_mask_from_called_dosage(G)
        assert mask[0] == False

    def test_polymorphic_true(self):
        G = np.array([[0.0], [1.0], [2.0], [1.0], [0.0]])
        mask = valid_af_mask_from_called_dosage(G)
        assert mask[0] == True

    def test_too_few_calls_false(self):
        # Only 1 called genotype (AN=2 < 4)
        G = np.array([[1.0], [np.nan], [np.nan], [np.nan]])
        mask = valid_af_mask_from_called_dosage(G)
        assert mask[0] == False

    def test_all_nan_false(self):
        G = np.full((5, 1), np.nan)
        mask = valid_af_mask_from_called_dosage(G)
        assert mask[0] == False


# ── Edge cases ───────────────────────────────────────────────

class TestQCEdgeCases:
    def test_all_monomorphic_panel(self):
        """Panel where every SNP is monomorphic → all filtered out."""
        G = np.zeros((20, 10))  # all homozygous ref
        mask = valid_af_mask_from_called_dosage(G)
        assert not mask.any()

    def test_all_monomorphic_alt(self):
        """All homozygous alt → also monomorphic."""
        G = np.full((20, 5), 2.0)
        mask = valid_af_mask_from_called_dosage(G)
        assert not mask.any()

    def test_high_missingness_column(self):
        """A SNP with >50% missing data but enough calls should still be valid."""
        G = np.full((20, 1), np.nan)
        G[:4, 0] = [0.0, 1.0, 1.0, 2.0]  # 4 calls, AN=8 >= 4
        mask = valid_af_mask_from_called_dosage(G)
        assert mask[0] == True

    def test_single_call_rejected(self):
        """A SNP with only 1 called genotype → AN=2 < 4, rejected."""
        G = np.full((10, 1), np.nan)
        G[0, 0] = 1.0
        mask = valid_af_mask_from_called_dosage(G)
        assert mask[0] == False

    def test_dosage_clip_out_of_range(self):
        """Dosages slightly outside [0,2] should be handled via clip."""
        G = np.array([[2.0001], [-0.0001], [1.0]])
        p = allele_freq_from_called_dosage(G)
        assert p[0] >= 0.0
        assert p[0] <= 1.0

    def test_empty_matrix(self):
        """Zero-SNP matrix should return empty arrays."""
        G = np.zeros((10, 0))
        p = allele_freq_from_called_dosage(G)
        assert len(p) == 0
        mask = valid_af_mask_from_called_dosage(G)
        assert len(mask) == 0


# ── Chromosome guard tests ──────────────────────────────────


def _make_snp_qc_inputs(n_samples=20, chroms=None, n_snps=10):
    """Build minimal inputs for _pipeline_snp_qc."""
    if chroms is None:
        chroms = np.array(["chr1"] * n_snps)
    n_snps = len(chroms)
    rng = np.random.RandomState(42)
    G = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps))
    geno_df = pd.DataFrame(G, columns=[f"snp{i}" for i in range(n_snps)])
    positions = np.arange(1000, 1000 + n_snps * 1000, 1000)
    sid = np.array([f"snp{i}" for i in range(n_snps)])
    return geno_df, chroms, positions, sid


class TestChromosomeGuard:
    def test_all_unrecognized_raises(self):
        """All chromosome names unrecognized → clear ValueError."""
        geno_df, _, positions, sid = _make_snp_qc_inputs(
            chroms=np.array(["NC_001", "NC_002", "NC_003"]),
        )
        with pytest.raises(ValueError, match="None of the chromosome names"):
            _pipeline_snp_qc(geno_df, np.array(["NC_001", "NC_002", "NC_003"]),
                             positions, sid, maf_thresh=0.0, miss_thresh=1.0,
                             mac_thresh=0, drop_alt=True)

    def test_all_alt_drop_alt_raises(self):
        """All → ALT with drop_alt=True → ValueError about no SNPs remaining."""
        chroms = np.array(["scaffold_A", "scaffold_B", "scaffold_C"])
        geno_df, _, positions, sid = _make_snp_qc_inputs(chroms=chroms)
        with pytest.raises(ValueError, match="None of the chromosome names"):
            _pipeline_snp_qc(geno_df, chroms, positions, sid,
                             maf_thresh=0.0, miss_thresh=1.0,
                             mac_thresh=0, drop_alt=True)

    def test_partial_alt_tracked_in_qc(self):
        """Mix of valid + scaffold → qc_snp tracks ALT count."""
        chroms = np.array(["chr1", "chr1", "scaffold_99", "chr2"])
        geno_df, _, positions, sid = _make_snp_qc_inputs(chroms=chroms)
        result = _pipeline_snp_qc(geno_df, chroms, positions, sid,
                                  maf_thresh=0.0, miss_thresh=1.0,
                                  mac_thresh=0, drop_alt=True)
        qc_snp = result[6]
        assert qc_snp["ALT chromosomes"] == 1

    def test_partial_alt_no_drop(self):
        """With drop_alt=False, ALT variants are kept."""
        chroms = np.array(["chr1", "scaffold_99", "chr2"])
        geno_df, _, positions, sid = _make_snp_qc_inputs(chroms=chroms)
        result = _pipeline_snp_qc(geno_df, chroms, positions, sid,
                                  maf_thresh=0.0, miss_thresh=1.0,
                                  mac_thresh=0, drop_alt=False)
        out_chroms = result[1]
        assert "ALT" in out_chroms
        assert len(out_chroms) == 3

    def test_error_shows_original_labels(self):
        """Error message includes the actual CHROM values from the VCF."""
        chroms = np.array(["1A", "1B", "1D", "2A", "2B"])
        geno_df, _, positions, sid = _make_snp_qc_inputs(chroms=chroms)
        with pytest.raises(ValueError, match="1A"):
            _pipeline_snp_qc(geno_df, chroms, positions, sid,
                             maf_thresh=0.0, miss_thresh=1.0,
                             mac_thresh=0, drop_alt=True)
