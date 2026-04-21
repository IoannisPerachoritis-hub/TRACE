"""Tests for gwas/io.py — VCF loading, chromosome cleaning, info scores."""
import gzip
import textwrap

import numpy as np
import pytest

from gwas.io import load_vcf_cached, _clean_chr_series, _extract_info_scores


# ── Minimal valid VCF ─────────────────────────────────────────


def _minimal_vcf(n_samples=3, n_snps=2, chroms=None):
    """Build a minimal valid VCF string."""
    sample_names = [f"S{i}" for i in range(n_samples)]
    header = textwrap.dedent("""\
        ##fileformat=VCFv4.2
        ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
    """).rstrip()
    col_line = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(sample_names)
    lines = [header, col_line]

    if chroms is None:
        chroms = ["1"] * n_snps

    for i in range(n_snps):
        ch = chroms[i] if i < len(chroms) else "1"
        pos = (i + 1) * 1000
        gts = "\t".join(["0/1"] * n_samples)
        lines.append(f"{ch}\t{pos}\tchr{ch}_{pos}\tA\tT\t.\tPASS\t.\tGT\t{gts}")

    return "\n".join(lines) + "\n"


# ── load_vcf_cached tests ────────────────────────────────────


class TestLoadVCFCached:
    def test_plain_text_vcf(self):
        """Plain-text VCF bytes are parsed correctly."""
        vcf_text = _minimal_vcf(n_samples=3, n_snps=4)
        callset = load_vcf_cached(vcf_text.encode("utf-8"))

        assert callset is not None
        assert "samples" in callset
        assert len(callset["samples"]) == 3
        assert "calldata/GT" in callset
        assert callset["calldata/GT"].shape[0] == 4  # 4 variants
        assert callset["calldata/GT"].shape[1] == 3  # 3 samples

    def test_gzip_vcf(self):
        """Gzip-compressed VCF bytes are detected and parsed."""
        vcf_text = _minimal_vcf(n_samples=2, n_snps=3)
        gz_bytes = gzip.compress(vcf_text.encode("utf-8"))

        callset = load_vcf_cached(gz_bytes)
        assert callset is not None
        assert len(callset["samples"]) == 2
        assert callset["calldata/GT"].shape[0] == 3

    def test_missing_chrom_header_raises(self):
        """VCF without #CHROM header line raises RuntimeError."""
        bad_vcf = "##fileformat=VCFv4.2\nsome data without header\n"
        with pytest.raises(RuntimeError, match="missing the mandatory.*#CHROM"):
            load_vcf_cached(bad_vcf.encode("utf-8"))

    def test_excel_quote_cleanup(self):
        """Excel-wrapped quotes are stripped before parsing."""
        vcf_text = _minimal_vcf(n_samples=2, n_snps=1)
        # Wrap each line in double quotes (Excel export artifact)
        wrapped = "\n".join(f'"{line}"' for line in vcf_text.splitlines()) + "\n"
        callset = load_vcf_cached(wrapped.encode("utf-8"))

        assert callset is not None
        assert len(callset["samples"]) == 2

    def test_multiple_chromosomes(self):
        """VCF with SNPs on different chromosomes."""
        vcf_text = _minimal_vcf(n_samples=3, n_snps=3, chroms=["1", "2", "3"])
        callset = load_vcf_cached(vcf_text.encode("utf-8"))

        chroms = callset["variants/CHROM"]
        assert len(chroms) == 3
        assert set(chroms) == {"1", "2", "3"}


# ── _clean_chr_series tests ──────────────────────────────────


class TestCleanChrSeries:
    def test_numeric_chromosomes(self):
        """Plain numeric chromosomes: '1', '5', '12'."""
        labels, codes, uniq, order = _clean_chr_series(["1", "5", "12"])
        assert list(labels) == ["1", "5", "12"]
        assert list(codes) == [1, 5, 12]

    def test_chr_prefix(self):
        """'chr1', 'chr12' → stripped to '1', '12'."""
        labels, codes, uniq, order = _clean_chr_series(["chr1", "chr12", "chr3"])
        assert list(labels) == ["1", "12", "3"]

    def test_chr_with_leading_zeros(self):
        """'chr01', 'chr02' → '1', '2'."""
        labels, codes, uniq, order = _clean_chr_series(["chr01", "chr02"])
        assert list(labels) == ["1", "2"]

    def test_tomato_sl4_format(self):
        """Tomato SL4.0ch01 → '1'."""
        labels, codes, uniq, order = _clean_chr_series(["SL4.0ch01", "SL4.0ch12"])
        assert list(labels) == ["1", "12"]

    def test_pepper_ca_format(self):
        """Pepper Ca01, Ca_01 → '1'."""
        labels, codes, uniq, order = _clean_chr_series(["Ca01", "Ca_02", "Ca03"])
        assert list(labels) == ["1", "2", "3"]

    def test_auto_detect_keeps_all_numeric(self):
        """Default auto-detect: any positive integer chromosome is kept."""
        labels, codes, uniq, order = _clean_chr_series(["1", "13", "20"])
        assert list(labels) == ["1", "13", "20"]
        assert "ALT" not in labels

    def test_auto_detect_non_contiguous(self):
        """Auto-detect handles non-contiguous numbering (e.g., 1, 2, 5, 8)."""
        labels, codes, uniq, order = _clean_chr_series(["chr1", "chr5", "chr8"])
        assert list(labels) == ["1", "5", "8"]
        assert uniq == ["1", "5", "8"]

    def test_scaffold_becomes_alt(self):
        """Non-numeric scaffold names → 'ALT'."""
        labels, codes, uniq, order = _clean_chr_series(["scaffold_1", "MT", "1"])
        assert labels[0] == "ALT"
        assert labels[1] == "ALT"
        assert labels[2] == "1"

    def test_auto_detect_mixed_numeric_and_scaffold(self):
        """Auto-detect keeps numeric chromosomes, ALTs scaffolds/organellar."""
        labels, codes, uniq, order = _clean_chr_series(
            ["chr1", "chr5", "chr20", "scaffold_1", "MT"],
        )
        assert list(labels) == ["1", "5", "20", "ALT", "ALT"]
        assert uniq == ["1", "5", "20", "ALT"]

    def test_alt_in_unique_order(self):
        """'ALT' appears at end of unique list."""
        labels, codes, uniq, order = _clean_chr_series(
            ["1", "scaffold_99", "2"],
        )
        assert uniq[-1] == "ALT"

    def test_custom_canonical(self):
        """Custom canonical set restricts to specified chromosomes only."""
        labels, codes, uniq, order = _clean_chr_series(
            ["1", "2", "3"],
            canonical=("1", "2"),
        )
        assert labels[2] == "ALT"  # "3" not in canonical

    def test_custom_canonical_13_becomes_alt(self):
        """Explicit canonical=(1..12) treats '13' as ALT."""
        canonical_12 = tuple(str(i) for i in range(1, 13))
        labels, codes, uniq, order = _clean_chr_series(
            ["1", "13"], canonical=canonical_12,
        )
        assert labels[0] == "1"
        assert labels[1] == "ALT"
        assert codes[1] == 0

    # ── Species-specific prefix tests ───────────────────────────

    def test_tomato_sl2_format(self):
        """Tomato SL2.50ch01 → '1'."""
        labels, *_ = _clean_chr_series(["SL2.50ch01", "SL2.50ch12"])
        assert list(labels) == ["1", "12"]

    def test_tomato_itag_format(self):
        """Tomato ITAG4.0ch01 → '1'."""
        labels, *_ = _clean_chr_series(["ITAG4.0ch01", "ITAG4.0ch06"])
        assert list(labels) == ["1", "6"]

    def test_potato_st4_format(self):
        """Potato ST4.03ch01 → '1'."""
        labels, *_ = _clean_chr_series(["ST4.03ch01", "ST4.03ch12"])
        assert list(labels) == ["1", "12"]

    def test_rice_os_format(self):
        """Rice Os01, Os_01 → '1'."""
        labels, *_ = _clean_chr_series(["Os01", "Os_02", "Os12"])
        assert list(labels) == ["1", "2", "12"]

    def test_soybean_gm_format(self):
        """Soybean Gm01, Gm_20 → '1', '20'."""
        labels, *_ = _clean_chr_series(["Gm01", "Gm_10", "Gm20"])
        assert list(labels) == ["1", "10", "20"]

    def test_arabidopsis_at_format(self):
        """Arabidopsis At1, At05 → '1', '5'."""
        labels, *_ = _clean_chr_series(["At1", "At02", "At5"])
        assert list(labels) == ["1", "2", "5"]

    def test_maize_zm_format(self):
        """Maize Zm01, Zm10 → '1', '10'."""
        labels, *_ = _clean_chr_series(["Zm01", "Zm10"])
        assert list(labels) == ["1", "10"]

    def test_common_bean_pv_format(self):
        """Common bean Pv01, Pv11 → '1', '11'."""
        labels, *_ = _clean_chr_series(["Pv01", "Pv11"])
        assert list(labels) == ["1", "11"]

    def test_pea_ps_format(self):
        """Pea Ps01, Ps07 → '1', '7'."""
        labels, *_ = _clean_chr_series(["Ps01", "Ps07"])
        assert list(labels) == ["1", "7"]

    def test_peanut_ah_format(self):
        """Peanut Ah01, Ah20 → '1', '20'."""
        labels, *_ = _clean_chr_series(["Ah01", "Ah20"])
        assert list(labels) == ["1", "20"]

    def test_pepper_cap_chr_format(self):
        """Pepper CaP_Chr01 → '1'."""
        labels, *_ = _clean_chr_series(["CaP_Chr01", "CaP_Chr12"])
        assert list(labels) == ["1", "12"]

    def test_pepper_ca_chr_format(self):
        """Pepper Ca_chr01 → '1'."""
        labels, *_ = _clean_chr_series(["Ca_chr01", "Ca_chr05"])
        assert list(labels) == ["1", "5"]

    def test_barley_trailing_h(self):
        """Barley 1H, 7H → '1', '7'."""
        labels, *_ = _clean_chr_series(["1H", "2H", "7H"])
        assert list(labels) == ["1", "2", "7"]

    def test_barley_chr_trailing_h(self):
        """Barley chr1H, chr7H → '1', '7'."""
        labels, *_ = _clean_chr_series(["chr1H", "chr7H"])
        assert list(labels) == ["1", "7"]

    def test_ncbi_chromosome_prefix(self):
        """NCBI chromosome_1, Chromosome01 → '1'."""
        labels, *_ = _clean_chr_series(["chromosome_1", "Chromosome01", "chromosome-5"])
        assert list(labels) == ["1", "1", "5"]

    def test_triple_leading_zeros(self):
        """'001', '012' → '1', '12'."""
        labels, *_ = _clean_chr_series(["001", "012"])
        assert list(labels) == ["1", "12"]

    def test_mixed_species_formats(self):
        """Different species formats in one call all normalize correctly."""
        labels, *_ = _clean_chr_series([
            "chr1", "SL4.0ch02", "Ca03", "Os04", "Gm05",
        ])
        assert list(labels) == ["1", "2", "3", "4", "5"]

    def test_organellar_sequences_become_alt(self):
        """MT, Pt, ChrUn, plastid → 'ALT'."""
        labels, *_ = _clean_chr_series(["MT", "Pt", "ChrUn", "plastid", "1"])
        assert list(labels) == ["ALT", "ALT", "ALT", "ALT", "1"]

    def test_unplaced_scaffolds_become_alt(self):
        """Various scaffold formats → 'ALT'."""
        labels, *_ = _clean_chr_series([
            "scaffold_1", "Scaffold123", "contig_42", "Un_random", "1",
        ])
        assert labels[0] == "ALT"
        assert labels[1] == "ALT"
        assert labels[2] == "ALT"
        assert labels[3] == "ALT"
        assert labels[4] == "1"

    def test_20_chromosome_soybean(self):
        """Soybean full set: Gm01..Gm20 → '1'..'20'."""
        inputs = [f"Gm{i:02d}" for i in range(1, 21)]
        labels, codes, uniq, order = _clean_chr_series(inputs)
        expected = [str(i) for i in range(1, 21)]
        assert list(labels) == expected
        assert uniq == expected
        assert list(codes) == list(range(1, 21))

    def test_returns_numpy_not_arrow_string_array(self):
        """Regression: must return np.ndarray, not pandas ArrowStringArray.

        With pandas>=2.0 + pyarrow installed, .values on a string Series can
        return pandas.arrays.ArrowStringArray, which Streamlit's @st.cache_data
        cannot hash. This caused 'MLM GWAS failed: Cannot hash argument chroms'
        in run_gwas_cached on a user's machine. Closes the bug at the source.
        """
        inp = np.array(["chr1", "chr2", "chrX", "1", "2"], dtype=str)
        labels, codes, uniq, order = _clean_chr_series(inp)
        assert isinstance(labels, np.ndarray), (
            f"labels should be np.ndarray, got {type(labels).__name__}"
        )
        assert isinstance(codes, np.ndarray), (
            f"codes should be np.ndarray, got {type(codes).__name__}"
        )
        # Specifically reject any pandas/Arrow extension array
        assert "Arrow" not in type(labels).__name__, (
            f"Got {type(labels).__name__} — Streamlit cache_data cannot hash this"
        )
        assert "Extension" not in type(labels).__name__


# ── _extract_info_scores tests ────────────────────────────────


class TestExtractInfoScores:
    def test_with_dr2_field(self):
        """Callset with variants/DR2 returns scores."""
        callset = {"variants/DR2": np.array([0.95, 0.88, 0.72])}
        scores, field = _extract_info_scores(callset)
        assert scores is not None
        assert field == "DR2"
        assert len(scores) == 3
        np.testing.assert_allclose(scores, [0.95, 0.88, 0.72], atol=1e-4)

    def test_with_r2_field(self):
        """Callset with variants/R2 (minimac4 style)."""
        callset = {"variants/R2": np.array([0.99, 0.85])}
        scores, field = _extract_info_scores(callset)
        assert field == "R2"
        assert len(scores) == 2

    def test_no_quality_fields(self):
        """Callset without any quality fields returns (None, None)."""
        callset = {"variants/CHROM": np.array(["1", "2"])}
        scores, field = _extract_info_scores(callset)
        assert scores is None
        assert field is None

    def test_all_nan_field_skipped(self):
        """Field with all NaN/non-finite is skipped."""
        callset = {
            "variants/DR2": np.array([np.nan, np.nan, np.nan]),
            "variants/R2": np.array([0.9, 0.8, 0.7]),
        }
        scores, field = _extract_info_scores(callset)
        assert field == "R2"  # DR2 skipped because all NaN

    def test_priority_order(self):
        """DR2 is checked before R2 (Beagle takes priority)."""
        callset = {
            "variants/DR2": np.array([0.95]),
            "variants/R2": np.array([0.90]),
        }
        scores, field = _extract_info_scores(callset)
        assert field == "DR2"
