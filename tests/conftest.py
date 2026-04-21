"""
Shared fixtures for the GWAS platform test suite.

All synthetic data uses np.random.default_rng(42) for reproducibility.
No external data files are required.
"""
import numpy as np
import pytest


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


# ── Pepper-specific fixtures ─────────────────────────────────

@pytest.fixture
def pepper_chroms_layout():
    """SNP count per chromosome using pepper naming (Ca prefix)."""
    return {"Ca1": 40, "Ca2": 30, "Ca3": 30}


@pytest.fixture
def pepper_snp_metadata(pepper_chroms_layout):
    """sid, chroms, positions for 100 SNPs with pepper chromosome names."""
    sid_list, chroms_list, pos_list = [], [], []
    for ch_str, n_snps in pepper_chroms_layout.items():
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
