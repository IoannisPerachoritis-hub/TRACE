"""Compact Letter Display (CLD) via maximal-clique enumeration.

A CLD assigns letters to groups so that two groups **share at least one letter iff
they are NOT significantly different**. The correct construction enumerates the
MAXIMAL CLIQUES of the *non-significance* graph (an edge joins two groups that are
not significantly different): each maximal clique -- a maximal set of mutually
non-different groups -- gets one letter, and every group carries the letters of all
cliques it belongs to.

This replaces a greedy insert-then-absorb variant that could leave private seed
letters unmerged (three mutually non-significant groups -> ``a, ab, ac`` where
``a, a, a`` is correct). Bron-Kerbosch with pivoting is dependency-free (group
counts are small); no networkx / scipy.
"""
from __future__ import annotations

import numpy as np

_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _bron_kerbosch(R, P, X, adj, out):
    """Emit every maximal clique of the graph ``adj`` (Bron-Kerbosch with pivoting)."""
    if not P and not X:
        out.append(set(R))
        return
    pivot = max(P | X, key=lambda z: len(P & adj[z]))
    for v in list(P - adj[pivot]):
        _bron_kerbosch(R | {v}, P & adj[v], X & adj[v], adj, out)
        P = P - {v}
        X = X | {v}


def compact_letter_display(groups, sig_matrix, means=None):
    """Return ``{group: letters}`` such that two groups share a letter iff not significant.

    ``groups``     : iterable of group labels.
    ``sig_matrix`` : symmetric boolean frame indexed by label; ``.loc[g1, g2]`` True
                     means the pair is significantly different.
    ``means``      : optional ``{group: mean}``; letters are ordered by clique mean
                     (ascending) then first appearance, so ``a`` is the lowest-mean
                     clique and the labelling is stable across runs.

    The letter assignment is a defect fix (a group that shares a letter with two
    groups that differ from each other used to keep a private seed letter); it moves
    only the DISPLAYED letters, never any p-value / F / eta2.
    """
    groups = list(groups)
    if not groups:
        return {}
    gidx = {g: i for i, g in enumerate(groups)}

    def _nonsig(a, b):
        try:
            return not bool(sig_matrix.loc[a, b])
        except Exception:
            return False

    # Non-significance graph: an edge means the two groups can share a letter.
    adj = {g: {h for h in groups if h != g and _nonsig(g, h)} for g in groups}

    cliques = []
    _bron_kerbosch(set(), set(groups), set(), adj, cliques)

    def _clique_key(clique):
        idxs = [gidx[g] for g in clique]
        first = min(idxs)
        if means is not None:
            vals = [float(means[g]) for g in clique
                    if g in means and means[g] is not None and np.isfinite(float(means[g]))]
            m = float(np.mean(vals)) if vals else float("inf")
        else:
            m = 0.0
        return (m, first)

    letters = {g: set() for g in groups}
    for ci, clique in enumerate(sorted(cliques, key=_clique_key)):
        ch = _LETTERS[ci] if ci < len(_LETTERS) else f"L{ci}"
        for g in clique:
            letters[g].add(ch)

    return {g: "".join(sorted(letters[g])) for g in groups}


def tukey_cld(values, groups, *, alpha=0.05):
    """Tukey HSD of ``values`` between ``groups``, and its compact letter display.

    Returns ``(tukey_df, cld_df)``: statsmodels' pairwise table (``reject`` as bool), and
    one row per group (MLG_label, CLD) with the groups sorted by name. Two groups share a
    letter iff Tukey HSD does not separate them (``compact_letter_display`` over the
    reject matrix, letters ordered by the group means). The haplotype panel's tables, the
    letters on its box plot and its plotted-data CSV all come from this one call. Raises
    what ``pairwise_tukeyhsd`` raises.
    """
    import pandas as pd
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    y = pd.Series(np.asarray(values, dtype=float))
    g = pd.Series(np.asarray(groups).astype(str))
    tukey = pairwise_tukeyhsd(endog=y, groups=g, alpha=alpha)
    table = tukey.summary().data
    tukey_df = pd.DataFrame(table[1:], columns=table[0])
    tukey_df["reject"] = tukey_df["reject"].astype(bool)

    names = sorted(g.unique())
    sig_matrix = pd.DataFrame(False, index=names, columns=names)
    for _, row in tukey_df.iterrows():
        g1, g2, reject = str(row["group1"]), str(row["group2"]), bool(row["reject"])
        sig_matrix.loc[g1, g2] = reject
        sig_matrix.loc[g2, g1] = reject
    letters = compact_letter_display(names, sig_matrix, means=y.groupby(g).mean().to_dict())
    return tukey_df, pd.DataFrame({"MLG_label": names, "CLD": [letters[n] for n in names]})
