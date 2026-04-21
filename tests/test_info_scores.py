"""Tests for imputation quality (INFO/DR2) awareness."""
import numpy as np
import pandas as pd
import pytest

from gwas.io import _extract_info_scores
from gwas.qc import _pipeline_snp_qc


# ── _extract_info_scores ──────────────────────────────────

class TestExtractInfoScores:
    def test_dr2_found(self):
        callset = {"variants/DR2": np.array([0.95, 0.80, 0.60], dtype=np.float32)}
        scores, name = _extract_info_scores(callset)
        assert name == "DR2"
        assert scores.shape == (3,)
        np.testing.assert_array_almost_equal(scores, [0.95, 0.80, 0.60])

    def test_r2_found(self):
        callset = {"variants/R2": np.array([0.9, 0.7], dtype=np.float32)}
        scores, name = _extract_info_scores(callset)
        assert name == "R2"
        assert len(scores) == 2

    def test_info_found(self):
        callset = {"variants/INFO": np.array([1.0, 0.5], dtype=np.float32)}
        scores, name = _extract_info_scores(callset)
        assert name == "INFO"

    def test_priority_order_dr2_first(self):
        """DR2 takes priority over R2 when both present."""
        callset = {
            "variants/DR2": np.array([0.9], dtype=np.float32),
            "variants/R2": np.array([0.5], dtype=np.float32),
        }
        _, name = _extract_info_scores(callset)
        assert name == "DR2"

    def test_no_quality_field(self):
        callset = {"variants/CHROM": np.array(["1", "2"])}
        scores, name = _extract_info_scores(callset)
        assert scores is None
        assert name is None

    def test_all_nan_skipped(self):
        """If the field exists but is all NaN, treat as absent."""
        callset = {"variants/DR2": np.array([np.nan, np.nan], dtype=np.float32)}
        scores, name = _extract_info_scores(callset)
        assert scores is None
        assert name is None

    # ── End-to-end VCF integration tests ─────────────────────
    # These read real VCF files through allel.read_vcf to verify
    # that scikit-allel correctly delivers INFO sub-fields into
    # the callset dict, closing the gap between mocked unit tests
    # and actual imputation tool output formats.

    _VCF_FIELDS = [
        "calldata/GT", "variants/CHROM", "variants/POS",
        "variants/REF", "variants/ALT",
        "variants/DR2", "variants/R2", "variants/INFO",
        "variants/AR2", "variants/IMP_QUAL",
    ]

    @staticmethod
    def _write_vcf(tmp_path, filename, info_header, info_values):
        """Helper: write a minimal 3-variant VCF with given INFO field."""
        lines = [
            "##fileformat=VCFv4.2",
            info_header,
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3",
        ]
        gts = [
            "0/0\t0/1\t1/1",
            "0/1\t0/0\t0/1",
            "1/1\t0/1\t0/0",
        ]
        for i, (val, gt) in enumerate(zip(info_values, gts)):
            lines.append(
                f"chr01\t{(i + 1) * 1000}\t.\tA\tG\t.\tPASS\t{val}\tGT\t{gt}"
            )
        vcf_path = tmp_path / filename
        vcf_path.write_text("\n".join(lines) + "\n")
        return vcf_path

    def test_e2e_beagle5_dr2(self, tmp_path):
        """End-to-end: Beagle 5.x VCF with DR2 field."""
        import allel
        vcf_path = self._write_vcf(
            tmp_path, "beagle5.vcf",
            '##INFO=<ID=DR2,Number=1,Type=Float,Description="Dosage R-Squared">',
            ["DR2=0.95", "DR2=0.42", "DR2=0.88"],
        )
        callset = allel.read_vcf(str(vcf_path), fields=self._VCF_FIELDS)
        scores, field = _extract_info_scores(callset)
        assert field == "DR2"
        assert scores is not None
        np.testing.assert_allclose(scores, [0.95, 0.42, 0.88], atol=1e-2)

    def test_e2e_minimac4_r2(self, tmp_path):
        """End-to-end: minimac4 VCF with R2 field."""
        import allel
        vcf_path = self._write_vcf(
            tmp_path, "minimac4.vcf",
            '##INFO=<ID=R2,Number=1,Type=Float,Description="Estimated Imputation Accuracy (R-squared)">',
            ["R2=0.91", "R2=0.55", "R2=0.99"],
        )
        callset = allel.read_vcf(str(vcf_path), fields=self._VCF_FIELDS)
        scores, field = _extract_info_scores(callset)
        assert field == "R2"
        np.testing.assert_allclose(scores, [0.91, 0.55, 0.99], atol=1e-2)

    def test_e2e_impute2_info(self, tmp_path):
        """End-to-end: IMPUTE2-style VCF with INFO quality field."""
        import allel
        # Note: the field name in the VCF header is "INFO" which is also
        # the column name in VCF format. scikit-allel distinguishes them:
        # the column is structural, the sub-field is parsed as variants/INFO.
        vcf_path = self._write_vcf(
            tmp_path, "impute2.vcf",
            '##INFO=<ID=INFO,Number=1,Type=Float,Description="IMPUTE2 info metric">',
            ["INFO=0.87", "INFO=0.63", "INFO=0.94"],
        )
        callset = allel.read_vcf(str(vcf_path), fields=self._VCF_FIELDS)
        scores, field = _extract_info_scores(callset)
        assert field == "INFO"
        np.testing.assert_allclose(scores, [0.87, 0.63, 0.94], atol=1e-2)

    def test_e2e_beagle4_ar2(self, tmp_path):
        """End-to-end: Beagle 4.x VCF with AR2 field."""
        import allel
        vcf_path = self._write_vcf(
            tmp_path, "beagle4.vcf",
            '##INFO=<ID=AR2,Number=1,Type=Float,Description="Allelic R-Squared">',
            ["AR2=0.93", "AR2=0.71", "AR2=0.86"],
        )
        callset = allel.read_vcf(str(vcf_path), fields=self._VCF_FIELDS)
        scores, field = _extract_info_scores(callset)
        assert field == "AR2"
        np.testing.assert_allclose(scores, [0.93, 0.71, 0.86], atol=1e-2)

    def test_e2e_priority_dr2_over_r2(self, tmp_path):
        """End-to-end: VCF with both DR2 and R2 — DR2 takes priority."""
        import allel
        lines = [
            "##fileformat=VCFv4.2",
            '##INFO=<ID=DR2,Number=1,Type=Float,Description="Dosage R-Squared">',
            '##INFO=<ID=R2,Number=1,Type=Float,Description="Minimac R-Squared">',
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2",
            "chr01\t1000\t.\tA\tG\t.\tPASS\tDR2=0.95;R2=0.50\tGT\t0/0\t0/1",
            "chr01\t2000\t.\tC\tT\t.\tPASS\tDR2=0.88;R2=0.40\tGT\t0/1\t1/1",
        ]
        vcf_path = tmp_path / "dual_field.vcf"
        vcf_path.write_text("\n".join(lines) + "\n")
        callset = allel.read_vcf(str(vcf_path), fields=self._VCF_FIELDS)
        scores, field = _extract_info_scores(callset)
        assert field == "DR2"
        np.testing.assert_allclose(scores, [0.95, 0.88], atol=1e-2)

    def test_e2e_no_quality_fields(self, tmp_path):
        """End-to-end: plain VCF with no imputation quality fields."""
        import allel
        lines = [
            "##fileformat=VCFv4.2",
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2",
            "chr01\t1000\t.\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1",
            "chr01\t2000\t.\tC\tT\t.\tPASS\t.\tGT\t0/1\t1/1",
        ]
        vcf_path = tmp_path / "plain.vcf"
        vcf_path.write_text("\n".join(lines) + "\n")
        callset = allel.read_vcf(str(vcf_path), fields=self._VCF_FIELDS)
        scores, field = _extract_info_scores(callset)
        assert scores is None
        assert field is None


# ── _pipeline_snp_qc with info_scores ─────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(99)


@pytest.fixture
def geno_20x10(rng):
    """20 samples, 10 SNPs, polymorphic."""
    n, m = 20, 10
    G = rng.choice([0.0, 1.0, 2.0], size=(n, m)).astype(float)
    samples = [f"S{i}" for i in range(n)]
    snps = [f"SNP_{i}" for i in range(m)]
    return pd.DataFrame(G, index=samples, columns=snps)


class TestSnpQcWithInfoScores:
    def test_info_filter_removes_low_quality(self, geno_20x10):
        m = geno_20x10.shape[1]
        chroms = np.array(["1"] * m)
        positions = np.arange(1, m + 1) * 1000
        sid = np.array(geno_20x10.columns)

        # First 5 SNPs high quality, last 5 low quality
        info_scores = np.array([0.95, 0.90, 0.85, 0.92, 0.88,
                                0.30, 0.20, 0.10, 0.40, 0.15], dtype=np.float32)

        geno, *_, qc, info_out = _pipeline_snp_qc(
            geno_20x10, chroms, positions, sid,
            maf_thresh=0.0, miss_thresh=1.0, mac_thresh=0, drop_alt=False,
            info_scores=info_scores, info_thresh=0.8,
        )
        assert qc["Fail INFO"] == 5
        assert geno.shape[1] == 5
        assert info_out is not None
        assert len(info_out) == 5
        assert np.all(info_out >= 0.8)

    def test_info_noop_when_thresh_zero(self, geno_20x10):
        """info_thresh=0.0 should not filter anything."""
        m = geno_20x10.shape[1]
        chroms = np.array(["1"] * m)
        positions = np.arange(1, m + 1) * 1000
        sid = np.array(geno_20x10.columns)
        info_scores = np.array([0.1] * m, dtype=np.float32)

        geno, *_, qc, _ = _pipeline_snp_qc(
            geno_20x10, chroms, positions, sid,
            maf_thresh=0.0, miss_thresh=1.0, mac_thresh=0, drop_alt=False,
            info_scores=info_scores, info_thresh=0.0,
        )
        assert qc["Fail INFO"] == 0
        assert geno.shape[1] == m

    def test_info_none_graceful(self, geno_20x10):
        """No info_scores → Fail INFO = 0, no filtering."""
        m = geno_20x10.shape[1]
        chroms = np.array(["1"] * m)
        positions = np.arange(1, m + 1) * 1000
        sid = np.array(geno_20x10.columns)

        geno, *_, qc, info_out = _pipeline_snp_qc(
            geno_20x10, chroms, positions, sid,
            maf_thresh=0.0, miss_thresh=1.0, mac_thresh=0, drop_alt=False,
            info_scores=None, info_thresh=0.8,
        )
        assert qc["Fail INFO"] == 0
        assert info_out is None

    def test_info_nan_treated_as_fail(self, geno_20x10):
        """NaN in info_scores should be treated as failing the threshold."""
        m = geno_20x10.shape[1]
        chroms = np.array(["1"] * m)
        positions = np.arange(1, m + 1) * 1000
        sid = np.array(geno_20x10.columns)

        info_scores = np.array([0.95] * m, dtype=np.float32)
        info_scores[0] = np.nan  # one NaN

        geno, *_, qc, _ = _pipeline_snp_qc(
            geno_20x10, chroms, positions, sid,
            maf_thresh=0.0, miss_thresh=1.0, mac_thresh=0, drop_alt=False,
            info_scores=info_scores, info_thresh=0.8,
        )
        assert qc["Fail INFO"] == 1
        assert geno.shape[1] == m - 1
