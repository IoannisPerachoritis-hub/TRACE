"""
Integration tests for GWAS model functions: run_gwas_cached, run_farmcpu,
run_mlmm_research_grade_fast.

These tests use synthetic genotype/phenotype data (50 samples x 100 SNPs)
and monkeypatch Streamlit dependencies so they run without a Streamlit server.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock


# ── Streamlit monkeypatch ─────────────────────────────────────
# Must happen before importing gwas.models (module-level st imports)

@pytest.fixture(autouse=True)
def _mock_streamlit(monkeypatch):
    """Patch st.cache_data and st.cache_resource to be passthroughs,
    and st.session_state to be a plain dict."""
    import streamlit as st

    _session = {}
    monkeypatch.setattr(st, "session_state", _session)

    # Make @st.cache_data a no-op decorator
    def _passthrough_decorator(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda fn: fn
    monkeypatch.setattr(st, "cache_data", _passthrough_decorator)
    monkeypatch.setattr(st, "cache_resource", _passthrough_decorator)

    # Stub UI functions that models.py may call with verbose=True
    monkeypatch.setattr(st, "empty", lambda: MagicMock())
    monkeypatch.setattr(st, "progress", lambda *a, **k: MagicMock())

    yield _session


# ── run_gwas_cached ──────────────────────────────────────────

class TestRunGwasCached:

    def test_returns_dataframe_with_required_columns(
        self, _mock_streamlit,
        gwas_geno, gwas_phenotype, gwas_pcs, gwas_snp_metadata,
        gwas_iid, gwas_K0, gwas_K_by_chr, gwas_pheno_reader,
    ):
        from gwas.models import run_gwas_cached
        meta = gwas_snp_metadata

        # Store pheno_reader in mock session state
        _mock_streamlit["pheno_reader_key"] = gwas_pheno_reader

        df = run_gwas_cached(
            geno_imputed=gwas_geno,
            y=gwas_phenotype,
            pcs_full=gwas_pcs,
            n_pcs=3,
            sid=meta["sid"],
            positions=meta["positions"],
            chroms=meta["chroms"],
            chroms_num=meta["chroms_num"],
            iid=gwas_iid,
            _K0=gwas_K0,
            _K_by_chr=gwas_K_by_chr,
            _pheno_reader_key="pheno_reader_key",
        )

        assert isinstance(df, pd.DataFrame)
        for col in ["SNP", "Chr", "Pos", "PValue", "ChrNum"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_pvalues_in_valid_range(
        self, _mock_streamlit,
        gwas_geno, gwas_phenotype, gwas_pcs, gwas_snp_metadata,
        gwas_iid, gwas_K0, gwas_K_by_chr, gwas_pheno_reader,
    ):
        from gwas.models import run_gwas_cached
        meta = gwas_snp_metadata
        _mock_streamlit["pheno_reader_key"] = gwas_pheno_reader

        df = run_gwas_cached(
            geno_imputed=gwas_geno, y=gwas_phenotype, pcs_full=gwas_pcs,
            n_pcs=3, sid=meta["sid"], positions=meta["positions"],
            chroms=meta["chroms"], chroms_num=meta["chroms_num"],
            iid=gwas_iid, _K0=gwas_K0, _K_by_chr=gwas_K_by_chr,
            _pheno_reader_key="pheno_reader_key",
        )

        pvals = df["PValue"].values
        assert np.all(pvals > 0), "All p-values should be > 0"
        assert np.all(pvals <= 1), "All p-values should be <= 1"

    def test_all_snps_present(
        self, _mock_streamlit,
        gwas_geno, gwas_phenotype, gwas_pcs, gwas_snp_metadata,
        gwas_iid, gwas_K0, gwas_K_by_chr, gwas_pheno_reader,
    ):
        from gwas.models import run_gwas_cached
        meta = gwas_snp_metadata
        _mock_streamlit["pheno_reader_key"] = gwas_pheno_reader

        df = run_gwas_cached(
            geno_imputed=gwas_geno, y=gwas_phenotype, pcs_full=gwas_pcs,
            n_pcs=3, sid=meta["sid"], positions=meta["positions"],
            chroms=meta["chroms"], chroms_num=meta["chroms_num"],
            iid=gwas_iid, _K0=gwas_K0, _K_by_chr=gwas_K_by_chr,
            _pheno_reader_key="pheno_reader_key",
        )

        # All 100 SNPs should be in the result
        assert len(df) == len(meta["sid"])

    def test_all_chromosomes_represented(
        self, _mock_streamlit,
        gwas_geno, gwas_phenotype, gwas_pcs, gwas_snp_metadata,
        gwas_iid, gwas_K0, gwas_K_by_chr, gwas_pheno_reader,
    ):
        from gwas.models import run_gwas_cached
        meta = gwas_snp_metadata
        _mock_streamlit["pheno_reader_key"] = gwas_pheno_reader

        df = run_gwas_cached(
            geno_imputed=gwas_geno, y=gwas_phenotype, pcs_full=gwas_pcs,
            n_pcs=3, sid=meta["sid"], positions=meta["positions"],
            chroms=meta["chroms"], chroms_num=meta["chroms_num"],
            iid=gwas_iid, _K0=gwas_K0, _K_by_chr=gwas_K_by_chr,
            _pheno_reader_key="pheno_reader_key",
        )

        result_chroms = set(df["Chr"].unique())
        assert {"1", "2", "3"} == result_chroms

    def test_known_signal_detected(
        self, _mock_streamlit,
        gwas_geno, gwas_pcs, gwas_snp_metadata,
        gwas_iid, gwas_K0, gwas_K_by_chr,
    ):
        """Inject a strong signal: y = 5*G[:,0] + noise. SNP 0 should have smallest p."""
        from gwas.models import run_gwas_cached
        from gwas.utils import PhenoData
        meta = gwas_snp_metadata
        rng = np.random.default_rng(999)

        # Create phenotype driven by first SNP
        y = 5.0 * gwas_geno[:, 0] + rng.normal(0, 0.5, gwas_geno.shape[0])
        y = y.astype(np.float32)
        pheno_reader = PhenoData(iid=gwas_iid, val=y)
        _mock_streamlit["pheno_reader_key"] = pheno_reader

        df = run_gwas_cached(
            geno_imputed=gwas_geno, y=y, pcs_full=gwas_pcs,
            n_pcs=3, sid=meta["sid"], positions=meta["positions"],
            chroms=meta["chroms"], chroms_num=meta["chroms_num"],
            iid=gwas_iid, _K0=gwas_K0, _K_by_chr=gwas_K_by_chr,
            _pheno_reader_key="pheno_reader_key",
        )

        # The causal SNP should be among the top hits
        top_snp = df.sort_values("PValue").iloc[0]["SNP"]
        assert top_snp == meta["sid"][0]

    def test_run_gwas_cached_forwards_user_covar(
        self, _mock_streamlit,
        gwas_geno, gwas_pcs, gwas_snp_metadata, gwas_iid, gwas_K0, gwas_K_by_chr,
    ):
        """run_gwas_cached forwards user_covar to _run_gwas_impl: a covariate
        (near-)collinear with the phenotype absorbs the planted signal."""
        from gwas.models import run_gwas_cached
        from gwas.utils import PhenoData
        meta = gwas_snp_metadata
        rng = np.random.default_rng(2024)
        n = gwas_geno.shape[0]
        y = (2.0 * gwas_geno[:, 0].astype(float) + rng.normal(0, 1.0, n)).astype(np.float32)
        _mock_streamlit["pr"] = PhenoData(iid=gwas_iid, val=y)
        sid0 = str(meta["sid"][0])
        common = dict(
            geno_imputed=gwas_geno, y=y, pcs_full=gwas_pcs, n_pcs=3,
            sid=meta["sid"], positions=meta["positions"], chroms=meta["chroms"],
            chroms_num=meta["chroms_num"], iid=gwas_iid, _K0=gwas_K0,
            _K_by_chr=gwas_K_by_chr, _pheno_reader_key="pr",
        )
        base = run_gwas_cached(**common).set_index("SNP")
        covar = (y.astype(float) + rng.normal(0, 0.05 * float(np.std(y)), n)).reshape(-1, 1)
        withc = run_gwas_cached(**common, user_covar=covar).set_index("SNP")
        p_base = float(base.loc[sid0, "PValue"])
        p_cov = float(withc.loc[sid0, "PValue"]) if sid0 in withc.index else 1.0
        assert p_base < 1e-2
        assert p_cov > 100 * p_base
        assert p_cov > 0.05


# ── run_farmcpu ──────────────────────────────────────────────

class TestRunFarmCPU:

    def _run_farmcpu(self, geno, meta, iid, pheno_reader, K0, covar_reader):
        from gwas.models import run_farmcpu
        return run_farmcpu(
            geno_imputed=geno,
            sid=meta["sid"],
            chroms=meta["chroms"],
            chroms_num=meta["chroms_num"],
            positions=meta["positions"],
            iid=iid,
            pheno_reader=pheno_reader,
            K0=K0,
            covar_reader=covar_reader,
            p_threshold=0.05,   # relaxed for small data
            max_iterations=3,
            max_pseudo_qtns=5,
            verbose=False,
        )

    def test_returns_tuple_of_three(
        self, gwas_geno, gwas_snp_metadata, gwas_iid,
        gwas_pheno_reader, gwas_K0, gwas_covar_reader,
    ):
        result = self._run_farmcpu(
            gwas_geno, gwas_snp_metadata, gwas_iid,
            gwas_pheno_reader, gwas_K0, gwas_covar_reader,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_farmcpu_df_columns(
        self, gwas_geno, gwas_snp_metadata, gwas_iid,
        gwas_pheno_reader, gwas_K0, gwas_covar_reader,
    ):
        farmcpu_df, _, _ = self._run_farmcpu(
            gwas_geno, gwas_snp_metadata, gwas_iid,
            gwas_pheno_reader, gwas_K0, gwas_covar_reader,
        )
        assert isinstance(farmcpu_df, pd.DataFrame)
        for col in ["SNP", "Chr", "Pos", "PValue", "Model"]:
            assert col in farmcpu_df.columns

    def test_model_column_is_farmcpu(
        self, gwas_geno, gwas_snp_metadata, gwas_iid,
        gwas_pheno_reader, gwas_K0, gwas_covar_reader,
    ):
        farmcpu_df, _, _ = self._run_farmcpu(
            gwas_geno, gwas_snp_metadata, gwas_iid,
            gwas_pheno_reader, gwas_K0, gwas_covar_reader,
        )
        assert (farmcpu_df["Model"] == "FarmCPU").all()

    def test_convergence_info_keys(
        self, gwas_geno, gwas_snp_metadata, gwas_iid,
        gwas_pheno_reader, gwas_K0, gwas_covar_reader,
    ):
        _, _, conv = self._run_farmcpu(
            gwas_geno, gwas_snp_metadata, gwas_iid,
            gwas_pheno_reader, gwas_K0, gwas_covar_reader,
        )
        assert isinstance(conv, dict)
        assert "n_iterations" in conv
        assert "converged" in conv

    def test_pvalues_valid(
        self, gwas_geno, gwas_snp_metadata, gwas_iid,
        gwas_pheno_reader, gwas_K0, gwas_covar_reader,
    ):
        farmcpu_df, _, _ = self._run_farmcpu(
            gwas_geno, gwas_snp_metadata, gwas_iid,
            gwas_pheno_reader, gwas_K0, gwas_covar_reader,
        )
        pvals = farmcpu_df["PValue"].values
        assert np.all(np.isfinite(pvals))
        assert np.all(pvals > 0)
        assert np.all(pvals <= 1)

    def test_signal_detection(
        self, gwas_geno, gwas_snp_metadata, gwas_iid,
        gwas_K0, gwas_covar_reader,
    ):
        """Inject a strong signal: y = 5*G[:,0] + noise. Causal SNP should rank highly."""
        from gwas.models import run_farmcpu
        from gwas.utils import PhenoData
        meta = gwas_snp_metadata
        rng = np.random.default_rng(999)

        y = 5.0 * gwas_geno[:, 0] + rng.normal(0, 0.5, gwas_geno.shape[0])
        y = y.astype(np.float32)
        pheno_reader = PhenoData(iid=gwas_iid, val=y)

        farmcpu_df, pseudo_qtns, conv = run_farmcpu(
            geno_imputed=gwas_geno,
            sid=meta["sid"], chroms=meta["chroms"],
            chroms_num=meta["chroms_num"], positions=meta["positions"],
            iid=gwas_iid, pheno_reader=pheno_reader,
            K0=gwas_K0, covar_reader=gwas_covar_reader,
            p_threshold=0.05, max_iterations=5, max_pseudo_qtns=5,
            verbose=False,
        )

        # Causal SNP should be in the top 5
        top5 = set(farmcpu_df.sort_values("PValue").head(5)["SNP"].values)
        assert meta["sid"][0] in top5

        # Pseudo-QTN table should be a DataFrame (may be empty on small panels)
        assert isinstance(pseudo_qtns, pd.DataFrame)

    def test_farmcpu_golden_e_f_nogate2_deterministic(
        self, gwas_geno, gwas_snp_metadata, gwas_iid,
        gwas_pheno_reader, gwas_K0, gwas_covar_reader,
    ):
        """D-101 golden for the SHIPPED FarmCPU path (E_F_nogate2 defaults): it runs
        published Step 5, is deterministic on the seeded synthetic fixture (identical
        p-values on repeat), and its retained pseudo-QTN set respects the corrected
        bound sqrt(n/log10 n). Regression guard for the new default path."""
        from gwas import models
        r1 = self._run_farmcpu(gwas_geno, gwas_snp_metadata, gwas_iid,
                               gwas_pheno_reader, gwas_K0, gwas_covar_reader)
        r2 = self._run_farmcpu(gwas_geno, gwas_snp_metadata, gwas_iid,
                               gwas_pheno_reader, gwas_K0, gwas_covar_reader)
        p1 = r1[0].sort_values("SNP")["PValue"].to_numpy()
        p2 = r2[0].sort_values("SNP")["PValue"].to_numpy()
        np.testing.assert_array_equal(p1, p2)   # shipped path is deterministic
        # retained pseudo-QTNs within the published ceiling sqrt(n/log10 n)
        assert len(r1[1]) <= models._step5_accept_bound(gwas_geno.shape[0])


# ── run_mlmm_research_grade_fast ─────────────────────────────

class TestRunMLMM:

    def _run_mlmm(self, geno, meta, iid, pheno_reader, K0, covar_reader,
                   p_enter=1e-4, max_cof=10):
        from gwas.models import run_mlmm_research_grade_fast
        return run_mlmm_research_grade_fast(
            geno_imputed=geno,
            sid=meta["sid"],
            chroms=meta["chroms"],
            chroms_num=meta["chroms_num"],
            positions=meta["positions"],
            iid=iid,
            pheno_reader=pheno_reader,
            K0=K0,
            covar_reader=covar_reader,
            p_enter=p_enter,
            max_cof=max_cof,
            verbose=False,
        )

    def test_returns_tuple(
        self, gwas_geno, gwas_snp_metadata, gwas_iid,
        gwas_pheno_reader, gwas_K0, gwas_covar_reader,
    ):
        """MLMM returns (mlmm_df, cofactor_table)."""
        result = self._run_mlmm(
            gwas_geno, gwas_snp_metadata, gwas_iid,
            gwas_pheno_reader, gwas_K0, gwas_covar_reader,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        mlmm_df, cof_table = result
        assert isinstance(mlmm_df, pd.DataFrame)
        assert isinstance(cof_table, pd.DataFrame)

    def test_has_pvalue_column(
        self, gwas_geno, gwas_snp_metadata, gwas_iid,
        gwas_pheno_reader, gwas_K0, gwas_covar_reader,
    ):
        mlmm_df, _ = self._run_mlmm(
            gwas_geno, gwas_snp_metadata, gwas_iid,
            gwas_pheno_reader, gwas_K0, gwas_covar_reader,
        )
        assert "PValue" in mlmm_df.columns

    def test_pvalues_valid(
        self, gwas_geno, gwas_snp_metadata, gwas_iid,
        gwas_pheno_reader, gwas_K0, gwas_covar_reader,
    ):
        mlmm_df, _ = self._run_mlmm(
            gwas_geno, gwas_snp_metadata, gwas_iid,
            gwas_pheno_reader, gwas_K0, gwas_covar_reader,
        )
        pvals = mlmm_df["PValue"].dropna().values
        assert np.all(pvals > 0)
        assert np.all(pvals <= 1)

    def test_null_data_strict_threshold(
        self, gwas_snp_metadata, gwas_iid, gwas_K0, gwas_covar_reader,
    ):
        """With random phenotype and very strict p_enter, MLMM should select 0 cofactors."""
        from gwas.models import run_mlmm_research_grade_fast
        from gwas.utils import PhenoData
        rng = np.random.default_rng(77)
        n = gwas_iid.shape[0]
        geno = rng.choice([0.0, 1.0, 2.0], size=(n, 100)).astype(np.float32)
        y = rng.normal(0, 1, n).astype(np.float32)
        pheno_reader = PhenoData(iid=gwas_iid, val=y)

        mlmm_df, cof_table = run_mlmm_research_grade_fast(
            geno_imputed=geno,
            sid=gwas_snp_metadata["sid"],
            chroms=gwas_snp_metadata["chroms"],
            chroms_num=gwas_snp_metadata["chroms_num"],
            positions=gwas_snp_metadata["positions"],
            iid=gwas_iid,
            pheno_reader=pheno_reader,
            K0=gwas_K0,
            covar_reader=gwas_covar_reader,
            p_enter=1e-10,  # very strict → no cofactors selected
            max_cof=5,
            verbose=False,
        )
        assert isinstance(mlmm_df, pd.DataFrame)
        assert len(mlmm_df) > 0
        # With strict threshold, cofactor table should be empty
        assert len(cof_table) == 0
