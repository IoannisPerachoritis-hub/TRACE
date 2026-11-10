"""β_MLM and SE_MLM per ALT allele copy, on planted SNPs.

FaST-LMM's single_snp fits each test SNP standardised, so its SnpWeight and SnpWeightSE
are per SD of the SNP's dosage; gwas.models.mlm_effects_per_copy divides both by that SD
(dosage_sd: ddof 0, over the samples tested), so Beta_MLM shares Beta_OLS's unit. No
structure is simulated, so the per-copy β_MLM should sit near the planted effect and
β_OLS, and the p-values must be single_snp's, bit for bit.
"""
import numpy as np
import pandas as pd
import pytest
from fastlmm.association import single_snp
from pysnptools.kernelreader import KernelData
from pysnptools.snpreader import SnpData

from gwas.kinship import _build_grm_from_Z, _standardize_geno_for_grm
from gwas.models import _run_gwas_impl, add_ols_effects_to_gwas, dosage_sd, mlm_effects_per_copy
from gwas.snpview import effect_flag
from gwas.utils import PhenoData

N, PER_CHR = 300, 100
COMMON, RARE = 10, 150      # planted: ALT frequency 0.40, 0.8 per copy; 0.04, 1.2 per copy


@pytest.fixture(scope="module")
def planted():
    rng = np.random.default_rng(7)
    m = 3 * PER_CHR
    p = rng.uniform(0.05, 0.5, m)
    p[COMMON], p[RARE] = 0.40, 0.04
    G = rng.binomial(2, p, size=(N, m)).astype(np.float32)
    y = 0.8 * G[:, COMMON] + 1.2 * G[:, RARE] + rng.normal(0, 1, N)
    sid = np.array([f"s{j}" for j in range(m)])
    chroms = np.repeat(["1", "2", "3"], PER_CHR)
    pos = np.tile(np.arange(1, PER_CHR + 1) * 10_000, 3)
    iid = np.array([f"i{k}" for k in range(N)])
    K0 = KernelData(iid=np.c_[iid, iid], val=_build_grm_from_Z(_standardize_geno_for_grm(G)))
    pheno = PhenoData(iid, y)
    gwas = _run_gwas_impl(G, y, None, 0, sid, pos, chroms, chroms.astype(int), iid, K0, {},
                          pheno)
    gwas = add_ols_effects_to_gwas(gwas, G, y, None, sid)
    return {"G": G, "y": y, "sid": sid, "pos": pos, "iid": iid, "K0": K0, "pheno": pheno,
            "gwas": gwas.set_index("SNP")}


def _single_snp(d, chrom, **kw):
    """single_snp on one chromosome's SNPs, as _run_gwas_impl calls it."""
    j = slice((chrom - 1) * PER_CHR, chrom * PER_CHR)
    snps = SnpData(iid=np.c_[d["iid"], d["iid"]], sid=d["sid"][j],
                   val=np.ascontiguousarray(d["G"][:, j]),
                   pos=np.c_[np.full(PER_CHR, float(chrom)), np.zeros(PER_CHR), d["pos"][j]])
    res = single_snp(test_snps=snps, pheno=d["pheno"], K0=d["K0"], covar=None,
                     leave_out_one_chrom=False, **kw)
    return res, d["G"][:, j], d["sid"][j]


def test_per_copy_is_the_ols_slope_when_h2_is_0(planted):
    """With h2 fixed at 0 the MLM is OLS on the intercept, so SnpWeight / dosage_sd is the
    OLS slope per ALT copy: this pins the SD (ddof 0, over the samples tested)."""
    res, G, sid = _single_snp(planted, 1, h2=0.0)
    res = mlm_effects_per_copy(res, G, sid).sort_values("sid_index")
    ols = np.array([np.polyfit(G[:, j].astype(float), planted["y"], 1)[0]
                    for j in range(PER_CHR)])
    np.testing.assert_allclose(res["SnpWeight"].to_numpy(), ols, rtol=1e-9)


def test_planted_effects_are_per_copy(planted):
    g = planted["gwas"]
    for j, beta in ((COMMON, 0.8), (RARE, 1.2)):
        r = g.loc[f"s{j}"]
        assert abs(r["Beta_MLM"] - beta) < 4 * r["SE_MLM"], (j, r["Beta_MLM"], r["SE_MLM"])
        assert abs(r["Beta_MLM"] - r["Beta_OLS"]) < 0.15 * abs(r["Beta_OLS"]), (
            j, r["Beta_MLM"], r["Beta_OLS"])
        assert 0.85 < r["SE_MLM"] / r["SE_OLS"] < 1.15, (j, r["SE_MLM"], r["SE_OLS"])


def test_a_low_frequency_snp_is_not_flagged(planted):
    """The planted SNP at ALT frequency 0.04 with no structure: per copy β_MLM agrees with
    β_OLS; per SD of its dosage it was about a quarter of β_OLS, which the flag called
    structure-sensitive."""
    r = planted["gwas"].loc[f"s{RARE}"]
    sd = dosage_sd(planted["G"][:, [RARE]])[0]
    assert effect_flag(r["Beta_MLM"], r["Beta_OLS"]) == "concordant"
    assert effect_flag(r["Beta_MLM"] * sd, r["Beta_OLS"]) == "structure-sensitive"


def test_only_the_effect_and_its_se_change(planted):
    """The p-values (and FaST-LMM's other columns) are single_snp's, bit for bit; Beta_MLM
    and SE_MLM are its SnpWeight and SnpWeightSE divided by the dosage SD."""
    g = planted["gwas"]
    assert {"Beta_MLM", "SE_MLM"} <= set(g.columns)
    assert not {"SnpWeight", "SnpWeightSE"} & set(g.columns)
    for chrom in (1, 2, 3):
        raw, G, sid = _single_snp(planted, chrom)
        raw = raw.set_index("SNP").loc[sid]
        mine = g.loc[sid]
        for col in ("PValue", "EffectSize", "Nullh2", "Mixing"):
            assert np.array_equal(mine[col].to_numpy(), raw[col].to_numpy()), (chrom, col)
        sd = dosage_sd(G)
        np.testing.assert_allclose(mine["Beta_MLM"].to_numpy() * sd, raw["SnpWeight"], rtol=1e-12)
        np.testing.assert_allclose(mine["SE_MLM"].to_numpy() * sd, raw["SnpWeightSE"], rtol=1e-12)


def test_the_sd_is_over_the_samples_single_snp_tests(planted):
    """single_snp drops samples without a phenotype before it standardises a SNP, so the
    per-copy division uses the SD over the same samples."""
    y = planted["y"].copy()
    y[:30] = np.nan
    d = dict(planted, pheno=PhenoData(planted["iid"], y))
    chroms = np.repeat(["1", "2", "3"], PER_CHR)
    g = _run_gwas_impl(planted["G"], y, None, 0, planted["sid"], planted["pos"], chroms,
                       chroms.astype(int), planted["iid"], planted["K0"], {}, d["pheno"])
    g = g.set_index("SNP")
    raw, G, sid = _single_snp(d, 1)
    raw = raw.set_index("SNP").loc[sid]
    sd = dosage_sd(G[30:])
    np.testing.assert_allclose(g.loc[sid, "Beta_MLM"].to_numpy() * sd, raw["SnpWeight"], rtol=1e-12)
    np.testing.assert_allclose(g.loc[sid, "SE_MLM"].to_numpy() * sd, raw["SnpWeightSE"], rtol=1e-12)
    assert np.array_equal(g.loc[sid, "PValue"].to_numpy(), raw["PValue"].to_numpy())


def test_dosage_sd_and_a_constant_snp():
    rng = np.random.default_rng(3)
    G = rng.integers(0, 3, size=(50, 9000)).astype(np.float32)     # more than one block
    G[:, 5] = 1.0
    sd = dosage_sd(G)
    np.testing.assert_allclose(sd, np.std(G.astype(np.float64), axis=0), rtol=1e-12)
    assert sd[5] == 0.0
    sid = np.array([f"x{j}" for j in range(G.shape[1])])
    res = pd.DataFrame({"sid_index": [5, 6], "SNP": ["x5", "x6"], "SnpWeight": [0.0, 0.5],
                        "SnpWeightSE": [0.1, 0.2], "PValue": [1.0, 0.01]})
    for frame in (res, res.drop(columns="sid_index")):       # by sid_index, or by SNP ID
        out = mlm_effects_per_copy(frame, G, sid)
        assert np.isnan(out["SnpWeight"].iloc[0]) and np.isnan(out["SnpWeightSE"].iloc[0])
        assert out["SnpWeight"].iloc[1] == 0.5 / sd[6]
        assert out["SnpWeightSE"].iloc[1] == 0.2 / sd[6]
        assert out["PValue"].tolist() == [1.0, 0.01]
