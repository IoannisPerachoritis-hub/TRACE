"""MLMM's forward search picks its cofactors by SNP: single_snp returns its rows sorted by
p-value, and MLMM used to read row i as SNP i, so its first cofactor was the genome's
first marker. Planted SNPs, no structure."""
import numpy as np
import pandas as pd
import pytest
from pysnptools.kernelreader import KernelData

from gwas.kinship import _build_grm_from_Z, _standardize_geno_for_grm
from gwas.models import _pvalues_in_snp_order, run_mlmm_research_grade_fast
from gwas.utils import PhenoData

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
