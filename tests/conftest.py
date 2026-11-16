"""
Shared fixtures for the GWAS platform test suite.

All synthetic data uses np.random.default_rng(42) for reproducibility.
No external data files are required.
"""
import importlib

import numpy as np
import pytest

# Functions that AppTest scripts in this suite stub on the shared Post-GWAS tab modules.
# AppTest runs page scripts in the pytest process, so a stub would otherwise stay in
# place for every later test.
_TAB_STUBBED = {
    "pages._ld_tabs.tab_genome_wide": ("find_ld_clusters_genomewide", "_render_block_visualization",
                                        "_render_lead_snp_gallery"),
    "pages._ld_tabs.tab_local_ld": ("extract_block_geno_for_paper",),
}


@pytest.fixture(autouse=True)
def _restore_stubbed_tab_functions():
    """Put back any of those functions a test replaced."""
    saved = []
    for mod_name, names in _TAB_STUBBED.items():
        mod = importlib.import_module(mod_name)
        saved.append((mod, {n: getattr(mod, n) for n in names}))
    yield
    for mod, attrs in saved:
        for name, fn in attrs.items():
            setattr(mod, name, fn)


@pytest.fixture(autouse=True, scope="session")
def _private_streamlit_cache_folder(tmp_path_factory):
    """The GWAS page's "Clear session data" deletes TRACE's .memo files from Streamlit's cache folder
    (~/.streamlit/cache): in tests it looks in a folder of their own instead
    (TRACE_STREAMLIT_CACHE_DIR, gwas.guitable.streamlit_cache_folder), never in the home folder."""
    import os

    folder = tmp_path_factory.mktemp("streamlit_cache")
    old = os.environ.get("TRACE_STREAMLIT_CACHE_DIR")
    os.environ["TRACE_STREAMLIT_CACHE_DIR"] = str(folder)
    yield folder
    if old is None:
        os.environ.pop("TRACE_STREAMLIT_CACHE_DIR", None)
    else:
        os.environ["TRACE_STREAMLIT_CACHE_DIR"] = old


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def geno_small(rng):
    """20 samples x 10 SNPs, dosage {0,1,2}, no missing data."""
    return rng.choice([0.0, 1.0, 2.0], size=(20, 10))


@pytest.fixture
def geno_with_nan(geno_small):
    """Same as geno_small but with ~2% NaN entries."""
    G = geno_small.copy()
    G[0, 3] = G[5, 7] = G[12, 1] = G[18, 9] = np.nan
    return G


@pytest.fixture
def geno_with_monomorphic(rng):
    """10 samples x 5 SNPs, column 2 is monomorphic (all 1.0)."""
    G = rng.choice([0.0, 1.0, 2.0], size=(10, 5))
    G[:, 2] = 1.0
    return G


@pytest.fixture
def phenotype_continuous(rng):
    """20-sample continuous phenotype (normal distribution)."""
    return rng.normal(10.0, 2.0, size=20)


@pytest.fixture
def phenotype_with_nan(phenotype_continuous):
    """Continuous phenotype with 2 NaN entries."""
    y = phenotype_continuous.copy()
    y[3] = y[15] = np.nan
    return y


@pytest.fixture
def haplotype_groups():
    """20 samples assigned to 3 haplotype groups."""
    return np.array(["A"] * 7 + ["B"] * 7 + ["C"] * 6)


@pytest.fixture
def pcs_small(rng):
    """20 samples x 3 principal components."""
    return rng.normal(0, 1, size=(20, 3))


# ── GWAS-scale fixtures for integration tests ────────────────

@pytest.fixture
def gwas_rng():
    """Separate RNG for GWAS integration tests (avoids perturbing existing fixtures)."""
    return np.random.default_rng(123)


@pytest.fixture
def gwas_n_samples():
    return 50


@pytest.fixture
def gwas_n_snps():
    return 100


@pytest.fixture
def gwas_chroms_layout():
    """SNP count per chromosome: 3 chromosomes with 40/30/30 split."""
    return {"1": 40, "2": 30, "3": 30}


@pytest.fixture
def gwas_iid(gwas_n_samples):
    """(n, 2) string array of sample IDs for FastLMM."""
    ids = np.array([f"sample_{i:03d}" for i in range(gwas_n_samples)])
    return np.c_[ids, ids]


@pytest.fixture
def gwas_snp_metadata(gwas_chroms_layout):
    """sid, chroms, chroms_num, positions arrays for 100 SNPs across 3 chromosomes."""
    sid_list, chroms_list, chroms_num_list, pos_list = [], [], [], []
    snp_idx = 0
    for ch_str, n_snps in gwas_chroms_layout.items():
        ch_num = int(ch_str)
        for i in range(n_snps):
            sid_list.append(f"chr{ch_str}_snp_{i:04d}")
            chroms_list.append(ch_str)
            chroms_num_list.append(ch_num)
            pos_list.append((i + 1) * 10000)
            snp_idx += 1
    return {
        "sid": np.array(sid_list, dtype=str),
        "chroms": np.array(chroms_list, dtype=str),
        "chroms_num": np.array(chroms_num_list, dtype=int),
        "positions": np.array(pos_list, dtype=int),
    }


@pytest.fixture
def gwas_geno(gwas_rng, gwas_n_samples, gwas_n_snps):
    """50 samples x 100 SNPs, dosage {0,1,2}, float32."""
    return gwas_rng.choice([0.0, 1.0, 2.0], size=(gwas_n_samples, gwas_n_snps)).astype(np.float32)


@pytest.fixture
def gwas_phenotype(gwas_rng, gwas_n_samples):
    """Continuous phenotype for 50 samples (normal distribution)."""
    return gwas_rng.normal(10.0, 2.0, size=gwas_n_samples).astype(np.float32)


@pytest.fixture
def gwas_pcs(gwas_rng, gwas_n_samples):
    """50 samples x 3 PCs."""
    return gwas_rng.normal(0, 1, size=(gwas_n_samples, 3)).astype(np.float32)


@pytest.fixture
def gwas_pheno_reader(gwas_iid, gwas_phenotype):
    """PhenoData compatible with FastLMM."""
    from gwas.utils import PhenoData
    return PhenoData(iid=gwas_iid, val=gwas_phenotype)


@pytest.fixture
def gwas_covar_reader(gwas_iid, gwas_pcs):
    """CovarData compatible with FastLMM."""
    from gwas.utils import CovarData
    return CovarData(iid=gwas_iid, val=gwas_pcs, names=["PC1", "PC2", "PC3"])


@pytest.fixture
def gwas_K0(gwas_iid, gwas_geno):
    """Whole-genome kinship KernelData."""
    from pysnptools.kernelreader import KernelData as PSKernelData
    from gwas.kinship import _standardize_geno_for_grm, _build_grm_from_Z
    Z = _standardize_geno_for_grm(gwas_geno)
    K = _build_grm_from_Z(Z)
    return PSKernelData(iid=gwas_iid, val=K)


@pytest.fixture
def gwas_K_by_chr(gwas_iid, gwas_geno, gwas_snp_metadata):
    """Per-chromosome LOCO kinship dict."""
    from pysnptools.kernelreader import KernelData as PSKernelData
    from gwas.kinship import _standardize_geno_for_grm, _build_grm_from_Z
    chroms = gwas_snp_metadata["chroms"]
    Z = _standardize_geno_for_grm(gwas_geno)
    K_by_chr = {}
    for ch in np.unique(chroms):
        off_chr = chroms != ch
        if off_chr.sum() < 10:
            continue
        Z_off = Z[:, off_chr]
        K_chr = _build_grm_from_Z(Z_off)
        K_by_chr[str(ch)] = PSKernelData(iid=gwas_iid, val=K_chr)
    return K_by_chr


# ── Alphanumeric-chromosome fixtures (for naming-regex coverage) ──────

@pytest.fixture
def chroms_layout_alphanum():
    """SNP count per chromosome using letter-prefixed naming (e.g. Ca1)."""
    return {"Ca1": 40, "Ca2": 30, "Ca3": 30}


@pytest.fixture
def snp_metadata_alphanum(chroms_layout_alphanum):
    """sid, chroms, positions for 100 SNPs with letter-prefixed chromosome names."""
    sid_list, chroms_list, pos_list = [], [], []
    for ch_str, n_snps in chroms_layout_alphanum.items():
        for i in range(n_snps):
            sid_list.append(f"{ch_str}_snp_{i:04d}")
            chroms_list.append(ch_str)
            pos_list.append((i + 1) * 10000)
    return {
        "sid": np.array(sid_list, dtype=str),
        "chroms": np.array(chroms_list, dtype=str),
        "positions": np.array(pos_list, dtype=int),
    }


@pytest.fixture
def geno_with_monomorphic_block(gwas_rng, gwas_n_samples):
    """20 SNPs where 3 are monomorphic — for LD block detection edge cases."""
    G = gwas_rng.choice([0.0, 1.0, 2.0], size=(gwas_n_samples, 20)).astype(np.float32)
    G[:, 5] = 1.0   # monomorphic
    G[:, 10] = 0.0   # monomorphic
    G[:, 15] = 2.0   # monomorphic
    return G
