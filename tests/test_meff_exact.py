"""M_eff (Li & Ji 2005) as the whole number it is in exact arithmetic (gwas/plotting.py).

The eigenvalues of the SNPs' correlation matrix sum to m' (the SNPs that are not constant), so
M_eff = sum of I(lam >= 1) + (lam mod 1) = m' + #(lam >= 1) - sum of floor(lam), a whole number.
The ceiling of the floating-point sum turned "a few units in the last place above it" into one
too many; now eigenvalues within 1e-9 of a whole number are set to it and the result is
ceil(sum - 1e-6). These tests hold the identity, its stability under blocking and memory
layout, and tables built so that the eigenvalues themselves are whole numbers.
"""
import numpy as np
import pytest

from gwas.genostore import LazyColumns, write_store
from gwas.plotting import (MEFF_BLOCK, _compute_meff_li_ji_impl, meff_eigenvalues,
                           meff_from_eigenvalues)

BLOCKS = (1, 7, 64, 997, MEFF_BLOCK)


def _panel(seed, n=None, m=None):
    rng = np.random.default_rng(seed)
    n = n or int(rng.integers(30, 120))
    m = m or int(rng.integers(150, 1500))
    p = rng.uniform(0.05, 0.5, size=m)
    G = ((rng.random((n, m)) < p).astype(np.float32) + (rng.random((n, m)) < p).astype(np.float32))
    return np.asfortranarray(G)


def _identity(eigs, G):
    """m' + #(lam >= 1) - sum floor(lam) on the snapped eigenvalues."""
    sd = np.asarray(G, dtype=np.float64).std(axis=0)
    m_nonconst = int((sd >= 1e-8).sum())
    e = np.asarray(eigs, dtype=np.float64)
    near = np.round(e)
    e = np.where(np.abs(e - near) < 1e-9, near, e)
    return m_nonconst + int((e >= 1.0).sum()) - int(np.floor(e[e >= 1.0]).sum())


@pytest.mark.parametrize("seed", range(12))
def test_meff_is_the_whole_number_of_the_identity(seed):
    G = _panel(seed)
    eigs = meff_eigenvalues(G, block=None)
    assert meff_from_eigenvalues(eigs) == _identity(eigs, G)


def _oa3(k):
    """The 3**k-run orthogonal array: the (3**k - 1) / 2 columns (a . x) mod 3, pairwise
    uncorrelated exactly (orthogonal contrasts)."""
    rows = np.array(np.meshgrid(*[np.arange(3)] * k, indexing="ij")).reshape(k, -1).T
    cols = []
    for a in np.ndindex(*([3] * k)):
        nz = [v for v in a if v]
        if nz and nz[0] == 1:
            cols.append((rows @ np.array(a)) % 3)
    return np.array(cols).T.astype(np.float32)


def _copies_table(sizes):
    """Exactly uncorrelated SNPs (columns of an orthogonal array), each repeated sizes[i]
    times: the correlation matrix is block diagonal with all-ones blocks, so its eigenvalues are
    the sizes (whole numbers) and zeros, and M_eff is the number of distinct SNPs exactly."""
    k = 3
    while (3 ** k - 1) // 2 < len(sizes):
        k += 1
    base = _oa3(k)[:, :len(sizes)]
    return np.asfortranarray(np.repeat(base, sizes, axis=1))


def _orthogonal_table(k=3, copies=(1, 3, 7)):
    """The orthogonal array with copies of each column (copies cycling through ``copies``)."""
    base = _oa3(k)
    reps = np.resize(np.array(copies), base.shape[1])
    return np.asfortranarray(np.repeat(base, reps, axis=1)), int(base.shape[1])


@pytest.mark.parametrize("sizes", [(1,) * 40, (3, 5, 1, 9, 2) * 6, (50, 50, 50), (2,) * 120])
def test_whole_number_eigenvalues_give_the_exact_count(sizes):
    G = _copies_table(sizes)
    for block in (None,) + BLOCKS:
        meff, eigs = _compute_meff_li_ji_impl(G, block=block)
        assert meff == len(sizes), (block, meff)


@pytest.mark.parametrize("k", [3, 4])
def test_orthogonal_design_with_copies(k):
    G, n_distinct = _orthogonal_table(k)
    results = {b: _compute_meff_li_ji_impl(G, block=b)[0] for b in (None,) + BLOCKS}
    assert set(results.values()) == {n_distinct}, results


@pytest.mark.parametrize("seed", range(10))
def test_blocked_equals_one_piece(seed):
    G = _panel(seed + 100)
    one = meff_from_eigenvalues(meff_eigenvalues(G, block=None))
    for b in BLOCKS:
        assert meff_from_eigenvalues(meff_eigenvalues(G, block=b)) == one, b


@pytest.mark.parametrize("seed", range(10))
def test_c_and_f_order_give_the_same_meff(seed):
    G = _panel(seed + 200)
    Gc = np.ascontiguousarray(G)
    for b in (None, 64, MEFF_BLOCK):
        assert (meff_from_eigenvalues(meff_eigenvalues(Gc, block=b))
                == meff_from_eigenvalues(meff_eigenvalues(G, block=b)))


def test_more_samples_than_snps(seed=7):
    G = _panel(seed, n=150, m=60)
    one = meff_from_eigenvalues(meff_eigenvalues(G, block=None))
    assert one == _identity(meff_eigenvalues(G, block=None), G)
    for b in (1, 7, 64):
        assert meff_from_eigenvalues(meff_eigenvalues(G, block=b)) == one


def test_the_table_on_disk_gives_the_same_meff(tmp_path):
    G = _panel(3, n=40, m=900)
    st = write_store(tmp_path / "s", G)
    try:
        L = LazyColumns(st, np.arange(G.shape[1]), fill=np.zeros(G.shape[1], np.float32))
        for b in (None, 64, MEFF_BLOCK):
            mem = _compute_meff_li_ji_impl(G, block=b)
            disk = _compute_meff_li_ji_impl(L, block=b)
            assert disk[0] == mem[0] and disk[1].tobytes() == mem[1].tobytes()
    finally:
        st.close()


@pytest.mark.parametrize("eigs,new,old", [
    ([0.4 + 3e-13, 0.6], 1, 2),        # the sum a hair above the whole number: the old ceiling gave one more
    ([0.4 - 3e-13, 0.6], 1, 1),        # a hair below it
    ([2 - 1e-12, 0.5, 0.5], 2, 3),     # an eigenvalue a hair below a whole number: 1 + 0.999... before the snap
    ([2 + 1e-12, 0.5, 0.5], 2, 3),     # a hair above it
    ([3.5, 1.2, 0.3], 3, 3),           # nothing near a whole number: 1.5 + 1.2 + 0.3 = 3
])
def test_near_whole_numbers(eigs, new, old):
    from gwas.ld import li_ji_sum

    assert meff_from_eigenvalues(np.array(eigs)) == new
    raw = sum((1.0 + (l % 1.0)) if l >= 1.0 else l for l in eigs)
    assert int(np.ceil(raw)) == old                         # today's rule, for the record
    assert abs(li_ji_sum(np.array(eigs)) - new) < 1e-6
