"""Maximal-clique Compact Letter Display (gwas/cld.py).

The load-bearing property is the CLD invariant: two groups share at least one letter
IFF they are not significantly different. The 4a case (three mutually non-significant
groups) is the counterexample that the replaced greedy+absorption algorithm got wrong
(it produced ``a, ab, ac`` instead of ``a, a, a``).
"""
import numpy as np
import pandas as pd
import pytest

from gwas.cld import compact_letter_display


def _sig_matrix(groups, sig_pairs):
    m = pd.DataFrame(False, index=groups, columns=groups)
    for a, b in sig_pairs:
        m.loc[a, b] = True
        m.loc[b, a] = True
    return m


def _share(cld, a, b):
    return bool(set(cld[a]) & set(cld[b]))


def test_all_nonsignificant_share_one_letter():
    # 4a counterexample: three mutually non-significant groups -> a, a, a.
    groups = ["a", "b", "c"]
    cld = compact_letter_display(groups, _sig_matrix(groups, []),
                                 means={"a": 1.0, "b": 2.0, "c": 3.0})
    assert cld["a"] == cld["b"] == cld["c"]
    assert all(len(v) == 1 for v in cld.values())


def test_chain_sharing_relation():
    # A~B, B~C, A!=C (only A-C significant): assert the SHARING relation, not literals.
    groups = ["A", "B", "C"]
    cld = compact_letter_display(groups, _sig_matrix(groups, [("A", "C")]),
                                 means={"A": 1.0, "B": 2.0, "C": 3.0})
    assert not _share(cld, "A", "C")   # significantly different -> share nothing
    assert _share(cld, "A", "B")       # non-significant -> share
    assert _share(cld, "B", "C")       # non-significant -> share


def test_all_significant_distinct_letters():
    groups = ["a", "b", "c"]
    cld = compact_letter_display(
        groups, _sig_matrix(groups, [("a", "b"), ("a", "c"), ("b", "c")]),
        means={"a": 1.0, "b": 2.0, "c": 3.0})
    assert len({cld["a"], cld["b"], cld["c"]}) == 3
    assert all(len(v) == 1 for v in cld.values())


@pytest.mark.parametrize("groups,sig_pairs", [
    (["a", "b", "c"], []),
    (["A", "B", "C"], [("A", "C")]),
    (["a", "b", "c"], [("a", "b"), ("a", "c"), ("b", "c")]),
    (["p", "q", "r", "s"], [("p", "s"), ("q", "s"), ("p", "r")]),
    (["x"], []),
])
def test_share_iff_not_significant_invariant(groups, sig_pairs):
    means = {g: float(i) for i, g in enumerate(groups)}
    cld = compact_letter_display(groups, _sig_matrix(groups, sig_pairs), means=means)
    sig = {frozenset(p) for p in sig_pairs}
    for i, g1 in enumerate(groups):
        for g2 in groups[i + 1:]:
            shares = _share(cld, g1, g2)
            is_sig = frozenset((g1, g2)) in sig
            assert shares == (not is_sig), (g1, g2, cld[g1], cld[g2], is_sig)


def test_random_graphs_preserve_invariant():
    rng = np.random.default_rng(0)
    for _ in range(500):
        n = int(rng.integers(2, 9))
        groups = [f"g{i}" for i in range(n)]
        pairs = [(groups[i], groups[j])
                 for i in range(n) for j in range(i + 1, n) if rng.random() < 0.5]
        means = {g: float(rng.random()) for g in groups}
        cld = compact_letter_display(groups, _sig_matrix(groups, pairs), means=means)
        sig = {frozenset(p) for p in pairs}
        for i in range(n):
            for j in range(i + 1, n):
                a, b = groups[i], groups[j]
                assert _share(cld, a, b) == (frozenset((a, b)) not in sig)


def test_empty_and_singleton():
    assert compact_letter_display([], pd.DataFrame()) == {}
    single = compact_letter_display(["only"], _sig_matrix(["only"], []), means={"only": 0.0})
    assert single == {"only": "a"}
