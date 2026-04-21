"""Tests for annotation.py — chromosome normalization and annotation."""
import pandas as pd
import pytest
from annotation import canon_chr, annotate_ld_blocks, consolidate_ld_block_table


# ── canon_chr ───────────────────────────────────────────────

class TestCanonChr:
    def test_numeric_passthrough(self):
        assert canon_chr("1") == "1"
        assert canon_chr("12") == "12"

    def test_chr_prefix(self):
        assert canon_chr("chr1") == "1"
        assert canon_chr("chr12") == "12"

    def test_Chr_prefix_with_leading_zero(self):
        assert canon_chr("Chr01") == "1"
        assert canon_chr("Chr12") == "12"

    def test_CHR_prefix(self):
        assert canon_chr("CHR5") == "5"

    def test_SL4_tomato_prefix(self):
        assert canon_chr("SL4.0ch01") == "1"
        assert canon_chr("SL4.0Ch12") == "12"

    def test_Ca_pepper_prefix(self):
        assert canon_chr("Ca1") == "1"
        assert canon_chr("Ca12") == "12"

    def test_Ca_no_digit_not_stripped(self):
        """Ca prefix without trailing digit should NOT be stripped (e.g. scaffold names)."""
        assert canon_chr("CaScaffold123") != "Scaffold123"
        # "CaALT" should keep the Ca since no digit follows
        result = canon_chr("CaALT")
        assert result == "CaALT", f"Expected 'CaALT' unchanged, got '{result}'"

    def test_empty_string(self):
        assert canon_chr("") == "0"

    def test_integer_input(self):
        assert canon_chr(5) == "5"

    def test_leading_zeros_stripped(self):
        assert canon_chr("007") == "7"
        assert canon_chr("chr007") == "7"

    # ── Species-specific prefix tests ───────────────────────────

    def test_ncbi_chromosome_prefix(self):
        assert canon_chr("chromosome_1") == "1"
        assert canon_chr("Chromosome01") == "1"
        assert canon_chr("chromosome-5") == "5"

    def test_tomato_sl2_format(self):
        assert canon_chr("SL2.50ch01") == "1"

    def test_tomato_itag_format(self):
        assert canon_chr("ITAG4.0ch06") == "6"

    def test_potato_st4_format(self):
        assert canon_chr("ST4.03ch01") == "1"
        assert canon_chr("ST4.03ch12") == "12"

    def test_rice_os_format(self):
        assert canon_chr("Os01") == "1"
        assert canon_chr("Os_02") == "2"
        assert canon_chr("Os12") == "12"

    def test_soybean_gm_format(self):
        assert canon_chr("Gm01") == "1"
        assert canon_chr("Gm_10") == "10"
        assert canon_chr("Gm20") == "20"

    def test_arabidopsis_at_format(self):
        assert canon_chr("At1") == "1"
        assert canon_chr("At05") == "5"

    def test_maize_zm_format(self):
        assert canon_chr("Zm01") == "1"
        assert canon_chr("Zm10") == "10"

    def test_common_bean_pv_format(self):
        assert canon_chr("Pv01") == "1"
        assert canon_chr("Pv11") == "11"

    def test_pea_ps_format(self):
        assert canon_chr("Ps01") == "1"
        assert canon_chr("Ps07") == "7"

    def test_peanut_ah_format(self):
        assert canon_chr("Ah01") == "1"
        assert canon_chr("Ah20") == "20"

    def test_pepper_cap_chr_format(self):
        assert canon_chr("CaP_Chr01") == "1"
        assert canon_chr("CaP_Chr12") == "12"

    def test_pepper_ca_chr_format(self):
        assert canon_chr("Ca_chr01") == "1"
        assert canon_chr("Ca_chr05") == "5"

    def test_barley_trailing_h(self):
        assert canon_chr("1H") == "1"
        assert canon_chr("7H") == "7"

    def test_barley_chr_trailing_h(self):
        assert canon_chr("chr1H") == "1"
        assert canon_chr("chr7H") == "7"


# ── annotate_ld_blocks ───────────────────────────────────────

class TestAnnotateLdBlocks:
    @pytest.fixture
    def genes_df(self):
        """Synthetic gene annotation: 5 genes on 2 chromosomes."""
        return pd.DataFrame({
            "Chr": ["1", "1", "1", "2", "2"],
            "Start": [100, 500, 1200, 100, 800],
            "End": [400, 900, 1500, 300, 1100],
            "Gene_ID": ["Gene1", "Gene2", "Gene3", "Gene4", "Gene5"],
            "Description": ["kinase", "transporter", "TF", "unknown", "heat shock"],
            "Strand": ["+", "-", "+", "+", "-"],
        })

    @pytest.fixture
    def ld_blocks_df(self):
        """Two LD blocks on chr 1 and chr 2."""
        return pd.DataFrame({
            "Chr": ["1", "2"],
            "Start": [200, 50],
            "End": [600, 200],
        })

    def test_overlapping_genes_found(self, ld_blocks_df, genes_df):
        out = annotate_ld_blocks(ld_blocks_df, genes_df, n_flank=2)
        assert "n_genes_overlapping" in out.columns
        # Block on chr1 [200,600] overlaps Gene1 [100,400] and Gene2 [500,900]
        assert out.iloc[0]["n_genes_overlapping"] == 2

    def test_flanking_gene_columns_present(self, ld_blocks_df, genes_df):
        out = annotate_ld_blocks(ld_blocks_df, genes_df, n_flank=2)
        for d in ("upstream", "downstream"):
            for i in (1, 2):
                assert f"{d}_gene_{i}" in out.columns
                assert f"{d}_dist_{i}" in out.columns

    def test_annotation_status(self, ld_blocks_df, genes_df):
        out = annotate_ld_blocks(ld_blocks_df, genes_df, n_flank=2)
        assert out.iloc[0]["annotation_status"] == "overlapping"

    def test_empty_blocks_returns_empty(self, genes_df):
        empty = pd.DataFrame(columns=["Chr", "Start", "End"])
        result = annotate_ld_blocks(empty, genes_df)
        assert result.empty

    def test_none_returns_none(self, genes_df):
        result = annotate_ld_blocks(None, genes_df)
        assert result is None

    def test_start_bp_column_variant(self, genes_df):
        """Handles 'Start (bp)' column naming."""
        blocks = pd.DataFrame({
            "Chr": ["1"],
            "Start (bp)": [200],
            "End (bp)": [600],
        })
        out = annotate_ld_blocks(blocks, genes_df, n_flank=1)
        assert "n_genes_overlapping" in out.columns
        assert out.iloc[0]["n_genes_overlapping"] >= 1

    def test_chr2_block_overlap(self, ld_blocks_df, genes_df):
        """Block on chr2 [50,200] should overlap Gene4 [100,300]."""
        out = annotate_ld_blocks(ld_blocks_df, genes_df, n_flank=2)
        assert out.iloc[1]["n_genes_overlapping"] == 1
        assert "Gene4" in out.iloc[1]["overlapping_genes"]

    def test_n_flank_parameter(self, genes_df):
        """Test with n_flank=3 produces more flanking columns."""
        blocks = pd.DataFrame({"Chr": ["1"], "Start": [200], "End": [600]})
        out = annotate_ld_blocks(blocks, genes_df, n_flank=3)
        assert "upstream_gene_3" in out.columns
        assert "downstream_gene_3" in out.columns


# ── consolidate_ld_block_table ──────────────────────────────

class TestConsolidateLdBlockTable:
    @pytest.fixture
    def ld_blocks(self):
        return pd.DataFrame({
            "Chr": ["1", "2"],
            "Start (bp)": [100, 500],
            "End (bp)": [400, 900],
            "Lead SNP": ["snp1", "snp2"],
            "SNP_IDs": ["snp1;snp1b", "snp2;snp2b"],
        })

    @pytest.fixture
    def hap_gwas(self):
        return pd.DataFrame({
            "Chr": ["1", "2"],
            "Start": [100, 500],
            "End": [400, 900],
            "PValue": [0.001, 0.05],
            "FDR_BH": [0.01, 0.10],
            "F_perm": [12.5, 3.1],
            "F_param": [13.0, 3.0],
            "n_haplotypes": [3, 2],
            "n_tested_haplotypes": [3, 2],
            "n_samples_block": [100, 95],
            "eta2": [0.35, 0.08],
            "n_permutations": [500, 500],
        })

    @pytest.fixture
    def annotated_blocks(self, ld_blocks):
        ann = ld_blocks.copy()
        ann["n_genes_overlapping"] = [2, 1]
        ann["overlapping_genes"] = ["GeneA;GeneB", "GeneC"]
        ann["annotation_status"] = ["overlapping", "overlapping"]
        return ann

    def test_ld_blocks_only(self, ld_blocks):
        result = consolidate_ld_block_table(ld_blocks)
        assert list(result.columns) == list(ld_blocks.columns)
        assert len(result) == 2

    def test_none_ld_blocks_returns_none(self):
        assert consolidate_ld_block_table(None) is None

    def test_ld_blocks_plus_haplotype(self, ld_blocks, hap_gwas):
        result = consolidate_ld_block_table(ld_blocks, hap_gwas=hap_gwas)
        assert "Hap_eta2" in result.columns
        assert "Hap_PValue" in result.columns
        assert "Hap_FDR_BH" in result.columns
        assert "Hap_n_haplotypes" in result.columns
        assert result.iloc[0]["Hap_eta2"] == pytest.approx(0.35)
        assert result.iloc[1]["Hap_eta2"] == pytest.approx(0.08)

    def test_full_merge(self, ld_blocks, hap_gwas, annotated_blocks):
        result = consolidate_ld_block_table(ld_blocks, hap_gwas, annotated_blocks)
        # Has annotation columns
        assert "n_genes_overlapping" in result.columns
        assert "overlapping_genes" in result.columns
        # Has haplotype columns
        assert "Hap_eta2" in result.columns
        assert "Hap_F_perm" in result.columns
        assert len(result) == 2

    def test_haplotype_columns_renamed(self, ld_blocks, hap_gwas):
        result = consolidate_ld_block_table(ld_blocks, hap_gwas=hap_gwas)
        # Original column names should NOT appear
        assert "eta2" not in result.columns
        assert "PValue" not in result.columns or result.columns.tolist().count("PValue") == 0
        # Renamed columns should appear
        assert "Hap_eta2" in result.columns
        assert "Hap_PValue" in result.columns
        assert "Hap_n_tested" in result.columns
        assert "Hap_n_samples" in result.columns
        assert "Hap_n_perms" in result.columns

    def test_no_haplotype_match(self, ld_blocks):
        """Haplotype data with non-matching coordinates gives NaN."""
        hap_nomatch = pd.DataFrame({
            "Chr": ["3"], "Start": [999], "End": [9999],
            "PValue": [0.01], "eta2": [0.5], "n_permutations": [100],
        })
        result = consolidate_ld_block_table(ld_blocks, hap_gwas=hap_nomatch)
        assert "Hap_eta2" in result.columns
        assert pd.isna(result.iloc[0]["Hap_eta2"])
