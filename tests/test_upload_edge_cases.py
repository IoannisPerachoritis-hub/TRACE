"""Tests for file upload edge cases: BOM, accession detection, ID normalization,
partial overlap, non-numeric traits, encoding fallback, empty files."""
import io
import numpy as np
import pandas as pd
import pytest
import textwrap

from gwas.io import load_vcf_cached
from gwas.qc import _pipeline_harmonize_ids, _pipeline_phenotype_qc, _pipeline_parse_vcf_biallelic


# ── Helpers ────────────────────────────────────────────────


def _minimal_vcf_text(samples, n_snps=5):
    """Build a minimal VCF string with given sample names."""
    header = textwrap.dedent("""\
        ##fileformat=VCFv4.2
        ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
    """).rstrip()
    col_line = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples)
    lines = [header, col_line]
    for i in range(n_snps):
        pos = (i + 1) * 1000
        gts = "\t".join(["0/1"] * len(samples))
        lines.append(f"1\t{pos}\tsnp_{i}\tA\tT\t.\tPASS\t.\tGT\t{gts}")
    return "\n".join(lines) + "\n"


def _make_harmonize_inputs(geno_samples, pheno_samples, n_snps=5):
    """Create matching inputs for _pipeline_harmonize_ids."""
    n = len(geno_samples)
    genotypes = np.full((n, n_snps), 1.0)  # all het
    samples = np.array(geno_samples)
    chroms = np.array(["1"] * n_snps)
    positions = np.arange(1, n_snps + 1) * 1000
    vid = np.array([f"snp_{i}" for i in range(n_snps)])
    ref = np.array(["A"] * n_snps)
    alt = np.array(["T"] * n_snps)

    pheno = pd.DataFrame(
        {"trait": np.random.default_rng(42).normal(5.0, 1.0, len(pheno_samples))},
        index=pheno_samples,
    )
    return genotypes, samples, chroms, positions, vid, ref, alt, pheno


# ── Fix 1: UTF-8 BOM in VCF ──────────────────────────────


class TestVCFWithBOM:
    def test_bom_vcf_parses_correctly(self):
        """VCF with UTF-8 BOM should be parsed without errors."""
        vcf_text = _minimal_vcf_text(["S0", "S1", "S2"])
        bom = b"\xef\xbb\xbf"
        vcf_bytes = bom + vcf_text.encode("utf-8")

        callset = load_vcf_cached(vcf_bytes)
        assert callset is not None
        assert len(callset["samples"]) == 3
        assert callset["calldata/GT"].shape[0] == 5


# ── Fix 3: Numeric ID normalization ──────────────────────


class TestNumericIDNormalization:
    def test_leading_zeros_mismatch_recovers(self):
        """VCF '001','002' vs phenotype '1','2' should match after stripping."""
        args = _make_harmonize_inputs(
            geno_samples=["001", "002", "003"],
            pheno_samples=["1", "2", "3"],
        )
        geno_df, pheno, *_ = _pipeline_harmonize_ids(*args)
        assert geno_df.shape[0] == 3
        assert pheno.shape[0] == 3

    def test_leading_zeros_reversed(self):
        """Phenotype '001','002' vs VCF '1','2' should also match."""
        args = _make_harmonize_inputs(
            geno_samples=["1", "2", "3"],
            pheno_samples=["001", "002", "003"],
        )
        geno_df, pheno, *_ = _pipeline_harmonize_ids(*args)
        assert geno_df.shape[0] == 3

    def test_no_overlap_raises(self):
        """Completely different IDs should raise ValueError with example IDs."""
        args = _make_harmonize_inputs(
            geno_samples=["A", "B", "C"],
            pheno_samples=["X", "Y", "Z"],
        )
        with pytest.raises(ValueError, match="No overlapping sample IDs"):
            _pipeline_harmonize_ids(*args)

    def test_error_message_shows_example_ids(self):
        """Error message should contain example IDs from both files."""
        args = _make_harmonize_inputs(
            geno_samples=["geno_1", "geno_2", "geno_3"],
            pheno_samples=["pheno_A", "pheno_B", "pheno_C"],
        )
        with pytest.raises(ValueError, match="VCF samples.*first 5"):
            _pipeline_harmonize_ids(*args)


# ── Fix 4: Partial overlap warning ───────────────────────


class TestPartialOverlap:
    def test_partial_overlap_proceeds(self):
        """Partial overlap should succeed with matched subset."""
        args = _make_harmonize_inputs(
            geno_samples=["S0", "S1", "S2", "S3"],
            pheno_samples=["S1", "S2", "S5"],
        )
        geno_df, pheno, *_ = _pipeline_harmonize_ids(*args)
        assert geno_df.shape[0] == 2  # S1, S2
        assert pheno.shape[0] == 2


# ── Fix 5: Non-numeric trait detection ────────────────────


class TestNonNumericTrait:
    def test_all_non_numeric_raises(self):
        """Trait column with no numeric values should raise clear error."""
        n = 5
        geno_df = pd.DataFrame(
            np.ones((n, 3)),
            index=[f"S{i}" for i in range(n)],
            columns=["snp0", "snp1", "snp2"],
        )
        pheno = pd.DataFrame(
            {"trait": ["low", "medium", "high", "low", "medium"]},
            index=geno_df.index,
        )
        with pytest.raises(ValueError, match="no numeric values"):
            _pipeline_phenotype_qc(geno_df, pheno, "trait", "None (raw values)", 0.5)

    def test_mixed_numeric_proceeds(self):
        """Trait with some numeric and some non-numeric should proceed."""
        n = 5
        geno_df = pd.DataFrame(
            np.ones((n, 3)),
            index=[f"S{i}" for i in range(n)],
            columns=["snp0", "snp1", "snp2"],
        )
        pheno = pd.DataFrame(
            {"trait": [1.0, 2.0, "NA", 4.0, 5.0]},
            index=geno_df.index,
        )
        # Should succeed — "NA" treated as missing
        geno_out, pheno_out, y = _pipeline_phenotype_qc(
            geno_df, pheno, "trait", "None (raw values)", 0.5,
        )
        assert geno_out.shape[0] == 4  # one dropped (NA)
        assert not np.isnan(y).any()


# ── Fix 10: Phenotype CSV with BOM ───────────────────────


class TestPhenotypeBOM:
    def test_bom_csv_columns_clean(self):
        """CSV with UTF-8 BOM should have clean column names after read_csv."""
        csv_text = "Accession,Fruit_Weight\nLA0716,45.2\nLA1589,12.8\n"
        bom = b"\xef\xbb\xbf"
        buf = io.BytesIO(bom + csv_text.encode("utf-8"))

        df = pd.read_csv(buf, sep=None, engine="python", encoding="utf-8-sig")
        # Column name should NOT start with \ufeff
        assert df.columns[0] == "Accession"
        assert "\ufeff" not in df.columns[0]

    def test_bom_csv_without_sig_has_bom_prefix(self):
        """Without utf-8-sig, BOM contaminates the first column name."""
        csv_text = "Accession,Fruit_Weight\nLA0716,45.2\nLA1589,12.8\n"
        bom = b"\xef\xbb\xbf"
        buf = io.BytesIO(bom + csv_text.encode("utf-8"))

        df = pd.read_csv(buf, sep=None, engine="python")
        # First column name starts with BOM character
        assert df.columns[0].startswith("\ufeff")


# ── Fix 11: Phenotype encoding fallback ──────────────────


class TestPhenotypeEncoding:
    def test_latin1_csv_readable(self):
        """Latin-1 encoded CSV should be readable with encoding='latin-1'."""
        csv_text = "Accession,Weight\nP\xe9rez,45.2\nG\xf3mez,12.8\n"
        buf = io.BytesIO(csv_text.encode("latin-1"))

        # UTF-8 fails
        with pytest.raises((UnicodeDecodeError, UnicodeError)):
            pd.read_csv(buf, sep=None, engine="python", encoding="utf-8-sig")

        # Latin-1 succeeds
        buf.seek(0)
        df = pd.read_csv(buf, sep=None, engine="python", encoding="latin-1")
        assert len(df) == 2
        assert df.columns[0] == "Accession"


# ── Fix 12: Empty phenotype file ─────────────────────────


class TestEmptyPhenotype:
    def test_empty_bytes_raises(self):
        """Zero-byte phenotype file should raise an error."""
        import csv
        buf = io.BytesIO(b"")
        with pytest.raises((pd.errors.EmptyDataError, csv.Error)):
            pd.read_csv(buf, sep=None, engine="python", encoding="utf-8-sig")

    def test_header_only_empty_dataframe(self):
        """Header-only CSV produces empty DataFrame."""
        buf = io.BytesIO(b"Accession,Weight\n")
        df = pd.read_csv(buf, sep=None, engine="python", encoding="utf-8-sig")
        assert len(df) == 0
        assert "Accession" in df.columns


# ── Fix 13: Empty VCF body ───────────────────────────────


class TestEmptyVCFBody:
    def test_none_callset_raises(self):
        """None callset (empty VCF) should raise ValueError."""
        with pytest.raises(ValueError, match="no variant records"):
            _pipeline_parse_vcf_biallelic(None)

    def test_empty_gt_raises(self):
        """Callset with zero variants should raise ValueError."""
        callset = {
            "samples": np.array(["S0", "S1"]),
            "calldata/GT": np.empty((0, 2, 2), dtype=int),
            "variants/CHROM": np.array([]),
            "variants/POS": np.array([], dtype=int),
        }
        with pytest.raises(ValueError, match="no variant records"):
            _pipeline_parse_vcf_biallelic(callset)
