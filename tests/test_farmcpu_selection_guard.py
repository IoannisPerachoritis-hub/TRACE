"""Guard tests for LOCO pseudo-QTN selection kinship (FarmCPU 2x2 measurement).

The 2x2 measurement adds a LOCO-kinship option to pseudo-QTN validation
(``_optimize_pseudo_qtns_mlm``).  Selection is passed ``chroms_num`` (numeric)
while the LOCO kernel dict is keyed by ``str(ch)`` from the string labels, so a
key miss would silently fall back to the global kinship -- and arm C would run
as arm A with no error and no warning, making the whole measurement worthless
while appearing to succeed.

The new path therefore looks the key up with the ``models.py:139`` precedent
(``.get(key, None)`` then an explicit ``is None`` check) and FAILS LOUDLY on a
miss.  These tests prove the guard fires, and that the correct-key path threads
the candidate's own-chromosome kernel (not a silent global fallback).
"""
import numpy as np
import pandas as pd
import pytest

from gwas import models


def _tiny_inputs():
    """20 samples x 6 SNPs across two string-labelled chromosomes."""
    rng = np.random.default_rng(0)
    n, m = 20, 6
    geno = rng.integers(0, 3, size=(n, m)).astype(float)
    y = rng.standard_normal(n)
    iid = np.array([[f"s{i}", f"s{i}"] for i in range(n)], dtype=str)
    sid = np.array([f"snp{j}" for j in range(m)], dtype=str)
    chroms_str = np.array(["1", "1", "1", "2", "2", "2"], dtype=str)
    chroms_num = chroms_str.astype(int)
    positions = np.array([100, 200, 300, 100, 200, 300], dtype=int)
    return geno, y, iid, sid, chroms_str, chroms_num, positions


def test_loco_selection_raises_on_mis_keyed_kernel(monkeypatch):
    """A chromosome label absent from ``K_by_chr`` must raise -- never a
    silent fall-back to global kinship."""
    geno, y, iid, sid, chroms_str, chroms_num, positions = _tiny_inputs()
    # single_snp must not be reached: the guard fires first.
    monkeypatch.setattr(
        models, "single_snp",
        lambda *a, **k: pytest.fail("guard did not fire; single_snp reached"),
    )
    # candidate 3 is on chr "2", which is absent from the kernel dict.
    K_by_chr = {"1": object()}
    with pytest.raises(KeyError, match="no kernel for chromosome"):
        models._optimize_pseudo_qtns_mlm(
            [3], geno, y, iid, sid, chroms_num, positions,
            None, object(), 0.05,
            chroms_str=chroms_str, K_by_chr=K_by_chr,
        )


def test_loco_selection_uses_the_chromosome_kernel(monkeypatch):
    """With correct keys, the candidate's OWN-chromosome LOCO kernel reaches
    single_snp -- not the global K0."""
    geno, y, iid, sid, chroms_str, chroms_num, positions = _tiny_inputs()
    K0 = object()
    k1, k2 = object(), object()
    K_by_chr = {"1": k1, "2": k2}
    seen = {}

    def _stub(*a, test_snps=None, pheno=None, K0=None, covar=None, **k):
        seen["K0"] = K0
        return pd.DataFrame({"PValue": [1.0]})  # >= threshold -> not accepted

    monkeypatch.setattr(models, "single_snp", _stub)
    models._optimize_pseudo_qtns_mlm(
        [3], geno, y, iid, sid, chroms_num, positions,
        None, K0, 0.05,
        chroms_str=chroms_str, K_by_chr=K_by_chr,
    )
    assert seen["K0"] is k2  # chr "2" kernel, not the global K0


def test_global_selection_uses_k0(monkeypatch):
    """Default (``K_by_chr=None``) passes the global K0 -- the current,
    byte-preserving behaviour for arms A/B."""
    geno, y, iid, sid, chroms_str, chroms_num, positions = _tiny_inputs()
    K0 = object()
    seen = {}

    def _stub(*a, test_snps=None, pheno=None, K0=None, covar=None, **k):
        seen["K0"] = K0
        return pd.DataFrame({"PValue": [1.0]})

    monkeypatch.setattr(models, "single_snp", _stub)
    models._optimize_pseudo_qtns_mlm(
        [3], geno, y, iid, sid, chroms_num, positions,
        None, K0, 0.05,
        chroms_str=chroms_str, K_by_chr=None,
    )
    assert seen["K0"] is K0
