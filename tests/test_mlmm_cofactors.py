"""MLMM's forward search picks its cofactors by SNP: single_snp returns its rows sorted by
p-value, and MLMM used to read row i as SNP i, so its first cofactor was the genome's
first marker. Planted SNPs, no structure."""
import numpy as np
import pandas as pd
import pytest
from fastlmm.association import single_snp
from pysnptools.kernelreader import KernelData
from pysnptools.snpreader import SnpData

from gwas.kinship import _build_grm_from_Z, _standardize_geno_for_grm
from gwas.models import (_pvalues_in_snp_order, _standardize_genotypes_impl,
                         run_mlmm_research_grade_fast)
from gwas.utils import CovarData, PhenoData

N, PER_CHR, STEP = 300, 60, 10_000            # 3 chromosomes, a marker every 10 kb
A, C, B = (1, 30), (1, 35), (2, 20)           # (chromosome, marker): A strongest, C 50 kb
EFFECT = {A: 1.0, C: 0.7, B: 0.5}             # from A on the same chromosome, B elsewhere


def _id(cm):
    return f"c{cm[0] + 1}_{cm[1]}"


def _data(order=None):
    rng = np.random.default_rng(5)
    G = rng.binomial(2, 0.3, size=(N, 3 * PER_CHR)).astype(np.float32)
    y = sum(b * G[:, c * PER_CHR + k] for (c, k), b in EFFECT.items()) + rng.normal(0, 1, N)
    sid = np.array([_id((c, k)) for c in range(3) for k in range(PER_CHR)])
    chroms = np.repeat(["1", "2", "3"], PER_CHR)
    pos = np.tile((np.arange(PER_CHR) + 1) * STEP, 3)
    if order is not None:
        G, sid, chroms, pos = G[:, order], sid[order], chroms[order], pos[order]
    iid = np.array([f"i{k}" for k in range(N)])
    K0 = KernelData(iid=np.c_[iid, iid], val=_build_grm_from_Z(_standardize_geno_for_grm(G)))
    return G, y, sid, chroms, pos, iid, K0


def _cofactors(window_kb=250, order=None):
    G, y, sid, chroms, pos, iid, K0 = _data(order)
    _df, cof = run_mlmm_research_grade_fast(
        G, sid, chroms, chroms.astype(int), pos, np.c_[iid, iid], PhenoData(iid, y), K0,
        None, window_kb=window_kb, verbose=False)
    return list(cof["SNP"].astype(str)), sid


def test_the_strongest_snp_enters_first_and_the_window_holds():
    cof, sid = _cofactors()
    assert sid[0] != _id(A)                   # A is not the genome's first marker
    assert cof[0] == _id(A)                   # (the row-position bug chose sid[0])
    assert cof[1] == _id(B)                   # C, 50 kb from A, is inside the 250 kb window
    assert _id(C) not in cof


def test_outside_a_smaller_window_the_neighbour_enters():
    cof, _ = _cofactors(window_kb=10)
    assert cof[:2] == [_id(A), _id(C)]


def test_shuffling_the_snp_order_gives_the_same_cofactors():
    order = np.random.default_rng(1).permutation(3 * PER_CHR)
    assert _cofactors(order=order)[0] == _cofactors()[0]


def test_the_p_values_are_put_in_snp_order():
    sid = np.array(["a", "b", "c"])
    res = pd.DataFrame({"sid_index": [2, 0, 1], "SNP": ["c", "a", "b"],
                        "PValue": [1e-9, 1e-3, 0.5]})          # p-sorted, as single_snp
    assert _pvalues_in_snp_order(res, sid).tolist() == [1e-3, 0.5, 1e-9]
    by_id = res.drop(columns="sid_index")
    assert _pvalues_in_snp_order(by_id, sid).tolist() == [1e-3, 0.5, 1e-9]
    with pytest.raises(ValueError, match="do not cover every test SNP"):
        _pvalues_in_snp_order(res.iloc[:2], sid)               # a SNP missing
    with pytest.raises(ValueError, match="do not cover every test SNP"):
        _pvalues_in_snp_order(res.assign(sid_index=[2, 2, 1]), sid)   # one twice
    with pytest.raises(ValueError, match="not one of the test SNPs"):
        _pvalues_in_snp_order(by_id.assign(SNP=["c", "a", "x"]), sid)
    with pytest.raises(ValueError, match="neither sid_index nor SNP"):
        _pvalues_in_snp_order(res[["PValue"]], sid)


# ── each cofactor's reported p: a re-test with the other cofactors only ──────────────

def _run(window_kb=250, p_enter=1e-4):
    G, y, sid, chroms, pos, iid, K0 = _data()
    iid2, pheno = np.c_[iid, iid], PhenoData(iid, y)
    df, cof = run_mlmm_research_grade_fast(
        G, sid, chroms, chroms.astype(int), pos, iid2, pheno, K0, None,
        window_kb=window_kb, p_enter=p_enter, verbose=False)
    return df.set_index("SNP"), cof, (G, sid, chroms, pos, iid2, K0, pheno)


def _direct(ctx, test_ids, cov_ids):
    """single_snp of ``test_ids`` with ``cov_ids`` as covariates, built as MLMM builds
    its scans (the standardised genotypes, K0, no LOCO)."""
    G, sid, chroms, pos, iid2, K0, pheno = ctx
    Z = _standardize_genotypes_impl(G)
    at = {s: i for i, s in enumerate(sid)}
    t, c = [at[s] for s in test_ids], [at[s] for s in cov_ids]
    snps = SnpData(iid=iid2, sid=sid[t], val=np.ascontiguousarray(Z[:, t]),
                   pos=np.c_[chroms[t].astype(float), np.zeros(len(t)), pos[t].astype(float)])
    covar = (CovarData(iid=iid2, val=Z[:, c], names=[f"cof_{k + 1}" for k in range(len(c))])
             if c else None)
    r = single_snp(test_snps=snps, pheno=pheno, K0=K0, covar=covar, leave_out_one_chrom=False)
    return r.set_index("SNP")["PValue"]


def test_each_cofactor_is_tested_with_the_others_only():
    df, cof, ctx = _run()
    a, b = _id(A), _id(B)
    assert list(cof["SNP"]) == [a, b]
    pa, pb = _direct(ctx, [a], [b])[a], _direct(ctx, [b], [a])[b]
    assert df.loc[a, "PValue"] == pa and df.loc[b, "PValue"] == pb       # bit for bit
    assert cof["PValue_final"].tolist() == [pa, pb]
    assert cof["PValue_entry"].iloc[0] < 1e-10      # A entered with a tiny p
    assert pa < 1e-10 and pb < 1e-3                 # not the final scan's p = 1


def test_the_other_snps_keep_the_final_scans_p():
    df, cof, ctx = _run()
    sid = list(ctx[1])
    final = _direct(ctx, sid, list(cof["SNP"]))      # the final scan: both cofactors in
    others = [s for s in sid if s not in set(cof["SNP"])]
    assert np.array_equal(df.loc[others, "PValue"].to_numpy(), final.loc[others].to_numpy())


def test_a_single_cofactor_is_tested_without_covariates():
    df, cof, ctx = _run(p_enter=1e-10)              # only A passes the entry threshold
    assert list(cof["SNP"]) == [_id(A)]
    assert df.loc[_id(A), "PValue"] == _direct(ctx, [_id(A)], [])[_id(A)]


def test_a_failed_retest_is_nan_with_a_warning(monkeypatch):
    import gwas.models as gm
    real = gm.single_snp
    warnings = []

    def failing_for_one_snp(*a, **k):
        if k["test_snps"].sid_count == 1:
            raise RuntimeError("re-test failure")
        return real(*a, **k)

    monkeypatch.setattr(gm, "single_snp", failing_for_one_snp)
    monkeypatch.setattr(gm._log, "warning", lambda msg, *a: warnings.append(msg % a))
    df, cof, _ = _run()
    assert len(cof) == 2 and cof["PValue_final"].isna().all()
    assert df.loc[list(cof["SNP"]), "PValue"].isna().all()
    assert len(warnings) == 2 and all("re-test of cofactor" in w for w in warnings)
