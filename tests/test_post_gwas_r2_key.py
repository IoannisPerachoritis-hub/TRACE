"""The Local LD tab's r2 cache key names the genotype table and the window's SNPs (reviewer
point 2-INT.23, the Post-GWAS page's part).

At ff56e5c the tab keyed its r2 matrix by the window's bounds, the lead SNP and the SNP count
only ("r2::local::<chr>:<start>-<end>::<lead>::<n>snps"), so a second table with the same
window and SNP count (another QC or imputation setting of the same trait) was shown the first
table's r2, and its r2 downloads. The key now names the table's identity, its rows and the
window's SNPs (gwas.guipost.r2_key).

The tab's own render function runs here outside a Streamlit page (bare mode), with the real
region extraction and a cache that, like the page's get_r2_cached, answers by key. The file
uses only names that ff56e5c has too: run against an export of ff56e5c it fails
(batch13/track_b/evidence/stale_r2_key_*.log).
"""
import numpy as np
import pandas as pd
import pytest

N, M = 60, 30


@pytest.fixture
def session():
    import streamlit as st

    before = set(st.session_state.keys())
    st.session_state.pop("r2_cache", None)
    st.session_state.pop("_local_ld_cache", None)
    yield st.session_state
    for k in set(st.session_state.keys()) - before:
        del st.session_state[k]


def test_two_tables_with_one_window_get_their_own_r2(session):
    import pages._ld_tabs.tab_local_ld as tll
    from gwas.ld import pairwise_r2
    from pages._ld_tabs import LDContext
    from pages._ld_tabs._window import RegionWindow

    positions = (1_000_000 + np.arange(M) * 10_000).astype(float)
    chroms = np.array(["2"] * M)
    sid = np.array([f"SL25ch02p{int(p)}" for p in positions])
    rng = np.random.default_rng(11)
    G_a = rng.binomial(2, 0.4, (N, M)).astype(float)
    G_b = G_a.copy()
    for j in range(0, M, 2):                      # the same SNPs, other LD: some columns shuffled
        G_b[:, j] = rng.permutation(G_b[:, j])
    pheno = pd.DataFrame({"t": rng.normal(0, 1, N)})
    gwas_df = pd.DataFrame({"SNP": sid, "Chr": chroms, "Pos": positions.astype(int),
                            "PValue": rng.uniform(1e-9, 1.0, M)})
    lead = str(sid[5])
    window = RegionWindow(lead_snp=lead, chr="2", start_bp=int(positions.min()),
                          end_bp=int(positions.max()), core_start=None, core_end=None,
                          use_block=False, label="window", lead_pos=int(positions[5]),
                          buffer_kb=100, block_snp_ids="")

    cache, calls = {}, []

    def get_r2_cached(region_geno, region_pos, min_pair_n=20, cache_key=None):
        # as the page's get_r2_cached: computed once per key, then answered from the cache
        if cache_key not in cache:
            cache[cache_key] = np.clip(pairwise_r2(np.asarray(region_geno, float),
                                                   min_pair_n=int(min_pair_n)), 0.0, 1.0)
        # the window as the tab passed it (its layout kept: r2's last bits depend on it)
        calls.append((cache_key, np.array(region_geno, dtype=float, copy=True, order="K"),
                      cache[cache_key]))
        return cache[cache_key]

    for name, G in (("A", G_a), ("B", G_b)):
        ctx = LDContext(
            geno_ld=G, geno_hard=G, chroms=chroms, positions=positions, sid=sid,
            geno_df=pd.DataFrame(G), pheno_df=pheno, trait_col="t", ph_aligned=pheno,
            y_aligned=pheno["t"].to_numpy(), keep_mask=np.ones(N, dtype=bool),
            geno_ld_aligned=G, gwas_df=gwas_df, haplo_df_auto=pd.DataFrame(),
            ld_decay_kb=100.0, adj_r2_min_global=0.2, ld_trait="t")
        ctx.table_identity = f"table {name}"     # the SessionTable's identity, on the page
        tll.render(ctx, get_r2_cached, window)

    assert len(calls) == 2
    (_, win_a, r2_a), (_, win_b, r2_b) = calls
    assert win_a.shape == win_b.shape and not np.array_equal(win_a, win_b)
    want_b = np.clip(pairwise_r2(win_b, min_pair_n=20), 0.0, 1.0)
    assert not np.array_equal(r2_a, want_b, equal_nan=True), "the two windows' r2 do not differ"
    assert np.array_equal(r2_b, want_b, equal_nan=True), \
        "the second table was shown the first table's r2 (one cache key for both)"
