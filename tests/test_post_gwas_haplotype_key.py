"""The haplotype test's session key names the genotype table, not its shape (reviewer point
2-INT.23, the Post-GWAS page's part).

The LD Blocks & Haplotypes tab keeps each haplotype test in the session under a key. At
ff56e5c the key named the table only by its shape (tab_genome_wide.py,
``geno_sig = f"{geno_ld.shape[0]}x{geno_ld.shape[1]}"``), so a second table of the same shape,
with the same blocks, settings and phenotype, was served the first table's result. The key now
names the table's identity (gwas.guipost.haplotype_key, from the context's table_identity).

The tab's own function (_render_haplotype_gwas) runs here outside a Streamlit page: in bare
mode the session state persists between calls and every widget returns its default. The
haplotype test is the real one (99 permutations). Table B holds the same block genotypes as
table A, but on other samples, so its test differs; both have one shape. The file uses only
names that ff56e5c has too: run against an export of ff56e5c it fails, showing the stale
result there.
"""
import numpy as np
import pandas as pd
import pytest

N, M = 60, 30
_KEYS_TO_CLEAR = ("geno_row_ids", "GWAS_CTX", "pheno_used_for_hap", "pheno_aligned", "pheno",
                  "n_perm_hap", "hap_gwas_df", "hap_gwas_df_enriched", "hap_tables", "pcs_full")


def _table(groups, seed=7):
    rng = np.random.default_rng(seed)
    G = rng.binomial(2, 0.3, (N, M)).astype(float)
    patterns = np.array([[0, 0, 1, 1, 2], [2, 1, 0, 2, 0], [1, 2, 2, 0, 1]], dtype=float)
    G[:, :5] = patterns[groups]
    return G


@pytest.fixture
def session():
    import streamlit as st

    before = set(st.session_state.keys())
    for k in list(st.session_state.keys()):
        if k in _KEYS_TO_CLEAR or (isinstance(k, str) and k.startswith(("HAPGWAS::", "LDDETECT::"))):
            del st.session_state[k]
    yield st.session_state
    for k in set(st.session_state.keys()) - before:
        del st.session_state[k]


def test_two_tables_of_one_shape_get_their_own_haplotype_tests(session, monkeypatch):
    import pages._ld_tabs.tab_genome_wide as tgw
    from gwas.haplotype import run_haplotype_block_gwas
    from pages._ld_tabs import LDContext

    ids = np.array([f"S{i:03d}" for i in range(N)])
    positions = (1_000_000 + np.arange(M) * 10_000).astype(float)
    chroms = np.array(["2"] * M)
    sid = np.array([f"SL25ch02p{int(p)}" for p in positions])
    grp_a = np.arange(N) % 3
    grp_b = np.random.default_rng(1).permutation(grp_a)
    tables = {"A": _table(grp_a), "B": _table(grp_b)}
    assert tables["A"].shape == tables["B"].shape and not np.array_equal(tables["A"], tables["B"])
    rng = np.random.default_rng(5)
    pheno = pd.DataFrame({"t": 10.0 + grp_a * 2.0 + rng.normal(0.0, 1.0, N)},
                         index=pd.Index(ids, dtype=object))
    pval = np.full(M, 0.5)
    pval[:5] = [1e-9, 2e-9, 3e-9, 4e-9, 5e-9]
    gwas_df = pd.DataFrame({"SNP": sid, "Chr": chroms, "Pos": positions.astype(int), "PValue": pval})
    blocks = pd.DataFrame({"Chr": ["2"], "Start (bp)": [1_000_000], "End (bp)": [1_040_000],
                           "lead_snp": [sid[0]], "SNP_IDs": [",".join(sid[:5])], "n_snps": [5]})

    session["pheno_used_for_hap"] = pheno.copy()
    session["n_perm_hap"] = 99
    monkeypatch.setattr(tgw, "_render_block_visualization", lambda **kw: None)
    monkeypatch.setattr(tgw, "_render_lead_snp_gallery", lambda *a, **kw: None)

    ran = []

    def _hap(blocks_, chroms_, positions_, geno_, sid_, geno_df_, pheno_, trait_, pcs_,
             min_hap_count, min_group_size, n_perm, geno_encoding, n_pcs_used, **kw):
        ran.append(np.array(geno_, copy=True))
        return run_haplotype_block_gwas(
            haplo_df=blocks_, chroms=chroms_, positions=positions_, geno_imputed=geno_,
            sid=sid_, geno_df=geno_df_, pheno_df=pheno_, trait_col=trait_, pcs=pcs_,
            min_hap_count=min_hap_count, min_group_size=min_group_size, n_perm=n_perm,
            geno_encoding=geno_encoding, n_pcs_used=n_pcs_used, **kw)

    shown = {}
    for name, G in tables.items():
        ctx = LDContext(
            geno_ld=G, geno_hard=G, chroms=chroms, positions=positions, sid=sid,
            geno_df=pd.DataFrame(G, index=pd.Index(ids, dtype=object)), pheno_df=pheno,
            trait_col="t", ph_aligned=pheno, y_aligned=pheno["t"].to_numpy(),
            keep_mask=np.ones(N, dtype=bool), geno_ld_aligned=G, gwas_df=gwas_df,
            haplo_df_auto=blocks, ld_decay_kb=100.0, adj_r2_min_global=0.2, ld_trait="t")
        ctx.table_identity = f"table {name}"     # the SessionTable's identity, on the page
        tgw._render_haplotype_gwas(
            ctx=ctx, haplo_df_auto=blocks, cached_extract_block_geno=None,
            compute_block_qc_effects=lambda hap_gwas_df=None, **k: hap_gwas_df,
            run_haplotype_block_gwas_cached_fn=_hap, settings=None)
        shown[name] = session["hap_gwas_df_enriched"].copy()

    # what each table's test gives
    want = {}
    for name, G in tables.items():
        res, _ = run_haplotype_block_gwas(
            haplo_df=blocks, chroms=chroms, positions=positions, geno_imputed=G, sid=sid,
            geno_df=pd.DataFrame(G, index=pd.Index(ids, dtype=object)), pheno_df=pheno,
            trait_col="t", pcs=None, min_hap_count=5, min_group_size=3, n_perm=99,
            geno_encoding="dosage012", n_pcs_used=None)
        want[name] = res
    cols = ["Chr", "Start", "End", "n_samples_tested", "n_tested_haplotypes", "F_perm", "P_perm", "eta2"]
    assert not want["A"][cols].equals(want["B"][cols]), "the two tables' tests do not differ"
    pd.testing.assert_frame_equal(shown["A"][cols], want["A"][cols])
    assert len(ran) == 2, f"the second table was served the first table's result ({len(ran)} test run)"
    assert np.array_equal(ran[1], tables["B"])
    pd.testing.assert_frame_equal(shown["B"][cols], want["B"][cols])
