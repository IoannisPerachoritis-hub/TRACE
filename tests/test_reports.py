"""Tests for gwas/reports.py — HTML report generation."""
import pandas as pd
import pytest

from gwas.reports import generate_gwas_report, _fig_to_b64, _df_to_html


# ── Helper tests ──────────────────────────────────────────

class TestHelpers:
    def test_fig_to_b64_bytes(self):
        raw = b"\x89PNG\r\n\x1a\nfake_png_data"
        b64 = _fig_to_b64(raw)
        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_fig_to_b64_none(self):
        assert _fig_to_b64(None) is None

    def test_df_to_html_basic(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3.14, 2.72]})
        html = _df_to_html(df)
        assert "<table" in html
        assert "3.14" in html

    def test_df_to_html_empty(self):
        assert _df_to_html(pd.DataFrame()) is None

    def test_df_to_html_none(self):
        assert _df_to_html(None) is None

    def test_df_to_html_max_rows(self):
        df = pd.DataFrame({"A": range(50)})
        html = _df_to_html(df, max_rows=5)
        # Should only contain 5 data rows
        assert html.count("<tr>") <= 6  # 5 data + 1 header


# ── Report generation ─────────────────────────────────────

@pytest.fixture
def minimal_gwas_df():
    return pd.DataFrame({
        "SNP": ["SNP_1", "SNP_2", "SNP_3"],
        "Chr": ["1", "1", "2"],
        "Pos": [1000, 2000, 3000],
        "PValue": [1e-8, 0.01, 0.5],
        "FDR": [1e-6, 0.1, 0.9],
    })


@pytest.fixture
def qc_snp():
    return {
        "Total SNPs": 1000,
        "Fail MAF": 50,
        "Fail Missingness": 20,
        "Fail MAC": 10,
        "Fail INFO": 0,
        "Fail ANY": 70,
        "Pass ALL": 930,
    }


class TestGenerateGwasReport:
    def test_minimal_report(self, minimal_gwas_df, qc_snp):
        html = generate_gwas_report(
            trait_col="Yield",
            qc_snp=qc_snp,
            gwas_df=minimal_gwas_df,
        )
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "Yield" in html
        assert "Quality Control" in html

    def test_contains_qc_values(self, minimal_gwas_df, qc_snp):
        html = generate_gwas_report(
            trait_col="Height",
            qc_snp=qc_snp,
            gwas_df=minimal_gwas_df,
        )
        assert "1000" in html  # Total SNPs
        assert "930" in html   # Pass ALL

    def test_significant_count(self, minimal_gwas_df, qc_snp):
        html = generate_gwas_report(
            trait_col="Test",
            qc_snp=qc_snp,
            gwas_df=minimal_gwas_df,
            n_samples=200,
            n_snps=930,
        )
        # SNP_1 has FDR < 0.05
        assert "1" in html  # n_significant = 1

    def test_lambda_gc_displayed(self, minimal_gwas_df, qc_snp):
        html = generate_gwas_report(
            trait_col="Test",
            qc_snp=qc_snp,
            gwas_df=minimal_gwas_df,
            lambda_gc=1.05,
        )
        assert "1.050" in html

    def test_figures_embedded(self, minimal_gwas_df, qc_snp):
        fake_png = b"\x89PNG\r\n\x1a\nfake_image_data_here"
        html = generate_gwas_report(
            trait_col="Test",
            qc_snp=qc_snp,
            gwas_df=minimal_gwas_df,
            figures={"Manhattan.png": fake_png},
        )
        assert "data:image/png;base64," in html

    def test_metadata_included(self, minimal_gwas_df, qc_snp):
        meta = {"MAF_threshold": 0.05, "Model": "MLM"}
        html = generate_gwas_report(
            trait_col="Test",
            qc_snp=qc_snp,
            gwas_df=minimal_gwas_df,
            metadata=meta,
        )
        assert "MAF_threshold" in html
        assert "0.05" in html

    def test_info_field_shown(self, minimal_gwas_df, qc_snp):
        html = generate_gwas_report(
            trait_col="Test",
            qc_snp=qc_snp,
            gwas_df=minimal_gwas_df,
            info_field="DR2",
        )
        assert "DR2" in html

    def test_valid_html_structure(self, minimal_gwas_df, qc_snp):
        html = generate_gwas_report(
            trait_col="Test",
            qc_snp=qc_snp,
            gwas_df=minimal_gwas_df,
        )
        assert html.count("<html") == 1
        assert html.count("</html>") == 1
        assert html.count("<body>") == 1
        assert html.count("</body>") == 1

    def test_per_model_post_gwas_none(self, minimal_gwas_df, qc_snp):
        """Report still works when per_model_post_gwas is None."""
        html = generate_gwas_report(
            trait_col="Test",
            qc_snp=qc_snp,
            gwas_df=minimal_gwas_df,
            per_model_post_gwas=None,
        )
        assert "<!DOCTYPE html>" in html
        assert "Post-GWAS:" not in html

    def test_custom_sig_label(self, minimal_gwas_df, qc_snp):
        """Custom sig_label appears in the report summary card."""
        html = generate_gwas_report(
            trait_col="Test",
            qc_snp=qc_snp,
            gwas_df=minimal_gwas_df,
            sig_label="M_eff (M=8,000)",
            n_significant_override=5,
        )
        assert "M_eff (M=8,000)" in html
        assert ">5<" in html  # n_significant value in summary card

    def test_default_sig_label(self, minimal_gwas_df, qc_snp):
        """Without sig_label, defaults to FDR<0.05."""
        html = generate_gwas_report(
            trait_col="Test",
            qc_snp=qc_snp,
            gwas_df=minimal_gwas_df,
        )
        assert "FDR&lt;0.05" in html or "FDR<0.05" in html
