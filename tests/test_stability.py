"""Tests for gwas/stability.py — stability screen functions."""

import numpy as np
import pandas as pd
import pytest

from gwas.stability import (
    _stability_screen_topk,
    _make_run_manifest,
    ld_block_stability_screen,
)
from gwas.utils import CovarData


# ── _stability_screen_topk ─────────────────────────────────


class TestStabilityScreenTopk:
    """Tests for the repeated-split OLS stability screen."""

    def test_happy_path(self, gwas_geno, gwas_phenotype, gwas_snp_metadata):
        """Basic run returns correct DataFrame structure."""
        df = _stability_screen_topk(
            geno_imputed=gwas_geno,
            y_vec=gwas_phenotype,
            sid=gwas_snp_metadata["sid"],
            covar_reader=None,
            n_reps=10,
            top_k=20,
            seed=42,
        )
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["SNP", "StabilityCount", "StabilityFreq"]
        assert len(df) == gwas_geno.shape[1]
        assert df["StabilityFreq"].iloc[0] >= df["StabilityFreq"].iloc[-1]
        assert (df["StabilityCount"] >= 0).all()
        assert (df["StabilityFreq"] <= 1.0).all()

    def test_no_covariates(self, gwas_geno, gwas_phenotype, gwas_snp_metadata):
        """Works without covariates."""
        df = _stability_screen_topk(
            geno_imputed=gwas_geno,
            y_vec=gwas_phenotype,
            sid=gwas_snp_metadata["sid"],
            covar_reader=None,
            n_reps=5,
            seed=0,
        )
        assert len(df) == gwas_geno.shape[1]

    def test_with_covariates(
        self, gwas_geno, gwas_phenotype, gwas_snp_metadata,
        gwas_iid, gwas_covar_reader,
    ):
        """Works with covariates + iid_full."""
        df = _stability_screen_topk(
            geno_imputed=gwas_geno,
            y_vec=gwas_phenotype,
            sid=gwas_snp_metadata["sid"],
            covar_reader=gwas_covar_reader,
            iid_full=gwas_iid,
            n_reps=5,
            seed=0,
        )
        assert len(df) == gwas_geno.shape[1]

    def test_reproducibility(self, gwas_geno, gwas_phenotype, gwas_snp_metadata):
        """Same seed produces identical results."""
        kwargs = dict(
            geno_imputed=gwas_geno,
            y_vec=gwas_phenotype,
            sid=gwas_snp_metadata["sid"],
            covar_reader=None,
            n_reps=10,
            seed=99,
        )
        df1 = _stability_screen_topk(**kwargs)
        df2 = _stability_screen_topk(**kwargs)
        pd.testing.assert_frame_equal(df1, df2)

    def test_monomorphic_snp_low_stability(self, gwas_rng, gwas_snp_metadata):
        """A monomorphic SNP should have zero or near-zero stability."""
        n, m = 50, 100
        geno = gwas_rng.choice([0.0, 1.0, 2.0], size=(n, m)).astype(np.float32)
        geno[:, 0] = 1.0  # monomorphic
        y = gwas_rng.normal(10.0, 2.0, size=n).astype(np.float32)

        df = _stability_screen_topk(
            geno_imputed=geno,
            y_vec=y,
            sid=gwas_snp_metadata["sid"],
            covar_reader=None,
            n_reps=20,
            top_k=10,
            seed=42,
        )
        mono_row = df[df["SNP"] == gwas_snp_metadata["sid"][0]]
        assert mono_row["StabilityCount"].values[0] == 0

    def test_signal_recovery_causal_snp(self, gwas_rng, gwas_snp_metadata):
        """A strongly causal SNP should have high stability frequency."""
        n, m = 50, 100
        geno = gwas_rng.choice([0.0, 1.0, 2.0], size=(n, m)).astype(np.float64)
        noise = gwas_rng.normal(0, 0.5, size=n)
        y = 2.0 * geno[:, 0] + noise  # SNP 0 is causal with large effect

        df = _stability_screen_topk(
            geno_imputed=geno,
            y_vec=y,
            sid=gwas_snp_metadata["sid"],
            covar_reader=None,
            n_reps=50,
            top_k=10,
            seed=0,
        )
        causal_row = df[df["SNP"] == gwas_snp_metadata["sid"][0]]
        assert causal_row["StabilityFreq"].values[0] >= 0.8

    def test_covariates_without_iid_raises(
        self, gwas_geno, gwas_phenotype, gwas_snp_metadata, gwas_covar_reader,
    ):
        """Covariates without iid_full raises ValueError at split time."""
        with pytest.raises(ValueError, match="iid_full must be provided"):
            _stability_screen_topk(
                geno_imputed=gwas_geno,
                y_vec=gwas_phenotype,
                sid=gwas_snp_metadata["sid"],
                covar_reader=gwas_covar_reader,
                iid_full=None,
                n_reps=1,
            )

    def test_misaligned_covariates_raises(
        self, gwas_geno, gwas_phenotype, gwas_snp_metadata, gwas_iid,
    ):
        """Misaligned covariate rows raise ValueError."""
        bad_covar = CovarData(
            iid=gwas_iid[:10],
            val=np.zeros((10, 3)),
            names=["PC1", "PC2", "PC3"],
        )
        with pytest.raises(ValueError, match="misaligned"):
            _stability_screen_topk(
                geno_imputed=gwas_geno,
                y_vec=gwas_phenotype,
                sid=gwas_snp_metadata["sid"],
                covar_reader=bad_covar,
                iid_full=gwas_iid,
                n_reps=1,
            )


# ── _make_run_manifest ─────────────────────────────────────


class TestMakeRunManifest:
    """Tests for GWAS run manifest builder."""

    @pytest.fixture
    def base_kwargs(self):
        return dict(
            trait_col="fruit_weight",
            pheno_label="tomato_traits.csv",
            vcf_name="panel.vcf.gz",
            phe_name="pheno.csv",
            maf_thresh=0.05,
            mac_thresh=3.0,
            miss_thresh=0.1,
            ind_miss_thresh=0.3,
            drop_alt=False,
            norm_option="log",
            n_pcs=3,
            sig_rule="meff",
            lambda_gc=1.02,
            n_samples=150,
            n_snps=50000,
            qc_snp={"passed": 48000, "removed_maf": 1500, "removed_miss": 500},
        )

    def test_required_keys(self, base_kwargs):
        """Manifest contains expected top-level keys."""
        m = _make_run_manifest(**base_kwargs)
        assert isinstance(m, dict)
        for key in [
            "timestamp_local", "software_versions", "analysis_type",
            "trait", "inputs", "qc_thresholds", "diagnostics",
            "subsampling_gwas_resampling",
        ]:
            assert key in m, f"Missing key: {key}"
        assert m["trait"] == "fruit_weight"
        assert m["analysis_type"] == "GWAS_panel_LOCO_FaSTLMM"
        assert m["diagnostics"]["lambda_gc_bulk_5_95"] == pytest.approx(1.02)

    def test_lambda_gc_none(self, base_kwargs):
        """lambda_gc=None → None in diagnostics."""
        base_kwargs["lambda_gc"] = None
        m = _make_run_manifest(**base_kwargs)
        assert m["diagnostics"]["lambda_gc_bulk_5_95"] is None

    def test_lambda_gc_inf_and_nan(self, base_kwargs):
        """Non-finite lambda_gc → None."""
        for val in [np.inf, -np.inf, np.nan]:
            base_kwargs["lambda_gc"] = val
            m = _make_run_manifest(**base_kwargs)
            assert m["diagnostics"]["lambda_gc_bulk_5_95"] is None

    def test_optional_dicts_none(self, base_kwargs):
        """All optional dicts default to empty dict when None."""
        m = _make_run_manifest(**base_kwargs)
        assert m["phenotype_preprocessing"]["zero_handling"] == {}
        assert m["phenotype_preprocessing"]["transformations_section_2b"] == {}
        assert m["diagnostics"]["loco_fallback_by_chr"] == {}


# ── ld_block_stability_screen ──────────────────────────────


class TestLdBlockStabilityScreen:
    """Tests for the LD block subsampling stability screen."""

    @pytest.fixture
    def gwas_df_with_signal(self, gwas_snp_metadata, gwas_rng):
        """GWAS results with one significant hit on chr1."""
        m = len(gwas_snp_metadata["sid"])
        pvals = gwas_rng.uniform(0.01, 1.0, size=m)
        # Make first 5 SNPs on chr1 highly significant
        pvals[:5] = gwas_rng.uniform(1e-8, 1e-6, size=5)
        return pd.DataFrame({
            "SNP": gwas_snp_metadata["sid"],
            "Chr": gwas_snp_metadata["chroms"],
            "Pos": gwas_snp_metadata["positions"],
            "PValue": pvals,
        })

    def test_happy_path(
        self, gwas_df_with_signal, gwas_snp_metadata, gwas_geno,
    ):
        """Basic run returns DataFrame with stability columns."""
        df = ld_block_stability_screen(
            gwas_df=gwas_df_with_signal,
            chroms=gwas_snp_metadata["chroms"],
            positions=gwas_snp_metadata["positions"],
            geno_imputed=gwas_geno,
            sid=gwas_snp_metadata["sid"],
            n_reps=5,
            sig_thresh=1e-5,
            top_n=3,
            seed=42,
        )
        if not df.empty:
            assert "StabilityCount" in df.columns
            assert "StabilityFreq" in df.columns
            assert "LeadRetentionFreq" in df.columns
            assert (df["StabilityFreq"] <= 1.0).all()

    def test_no_significant_snps_returns_empty(
        self, gwas_snp_metadata, gwas_geno,
    ):
        """When no SNPs are significant, returns empty DataFrame."""
        gwas_df = pd.DataFrame({
            "SNP": gwas_snp_metadata["sid"],
            "Chr": gwas_snp_metadata["chroms"],
            "Pos": gwas_snp_metadata["positions"],
            "PValue": np.ones(len(gwas_snp_metadata["sid"])),  # all p=1
        })
        df = ld_block_stability_screen(
            gwas_df=gwas_df,
            chroms=gwas_snp_metadata["chroms"],
            positions=gwas_snp_metadata["positions"],
            geno_imputed=gwas_geno,
            sid=gwas_snp_metadata["sid"],
            n_reps=3,
            sig_thresh=1e-5,
            top_n=0,
            seed=0,
        )
        assert df.empty

    def test_tiny_sample_returns_empty(self, gwas_snp_metadata):
        """With only 1 sample, returns empty DataFrame (k < 2)."""
        geno_1 = np.array([[1.0] * len(gwas_snp_metadata["sid"])])
        gwas_df = pd.DataFrame({
            "SNP": gwas_snp_metadata["sid"],
            "Chr": gwas_snp_metadata["chroms"],
            "Pos": gwas_snp_metadata["positions"],
            "PValue": np.full(len(gwas_snp_metadata["sid"]), 1e-8),
        })
        df = ld_block_stability_screen(
            gwas_df=gwas_df,
            chroms=gwas_snp_metadata["chroms"],
            positions=gwas_snp_metadata["positions"],
            geno_imputed=geno_1,
            sid=gwas_snp_metadata["sid"],
            n_reps=3,
            seed=0,
        )
        assert df.empty

    def test_pheno_geno_alignment(
        self, gwas_df_with_signal, gwas_snp_metadata, gwas_geno,
    ):
        """When pheno_df + geno_df provided, alignment filters samples."""
        n = gwas_geno.shape[0]
        sample_ids = [f"sample_{i:03d}" for i in range(n)]
        pheno_df = pd.DataFrame({
            "sample": sample_ids[:40],  # only 40 of 50 samples
            "trait": np.random.default_rng(0).normal(size=40),
        })
        geno_df = pd.DataFrame(
            gwas_geno,
            index=sample_ids,
            columns=gwas_snp_metadata["sid"],
        )
        df = ld_block_stability_screen(
            gwas_df=gwas_df_with_signal,
            chroms=gwas_snp_metadata["chroms"],
            positions=gwas_snp_metadata["positions"],
            geno_imputed=gwas_geno,
            sid=gwas_snp_metadata["sid"],
            n_reps=3,
            sig_thresh=1e-5,
            top_n=3,
            seed=42,
            pheno_df=pheno_df,
            geno_df=geno_df,
            trait_col="trait",
        )
        # Should run without error; result may or may not be empty depending on LD structure
        assert isinstance(df, pd.DataFrame)
