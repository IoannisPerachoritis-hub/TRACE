"""The Post-GWAS page takes the GWAS page's results, its M_eff and its sample IDs only for the
table they belong to, and records which trait and table its final LD blocks are of (the
contract between the two pages, batch 13).

- The GWAS results: the GWAS page records, with gwas_results, the identity of its Run's table
  (gwas_run_identity), and rebuilds the session's table (geno_table) on every rerun. After a
  settings change without a new Run, the page computed with the last Run's p-values on the new
  table. It now says so plainly and stops when the identities differ, or when results come
  without an identity.
- M_eff: meff_val is used only when meff_identity names this table; otherwise M_eff is absent
  (the rule's path without M_eff), as when it was never computed.
- The final LD blocks: whenever the page or the LD Blocks tab writes ld_blocks_final, it also
  writes ld_blocks_final_key = (trait, the table's identity), which the GWAS page's block
  stability checks before it aggregates those blocks.
- The sample IDs: the table's own (geno_table.iid). The GWAS page writes geno_df before the
  table, so a run of it that stopped between the two left a geno_df beside an older table.

The page and the tab run here outside a Streamlit page (bare mode), on a small table in memory:
st.stop raises StopException as in a real run, and the run ends where the page builds the tabs'
context (LDContext), which is recorded.
"""
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gwas import guipost as gp

ROOT = Path(__file__).resolve().parents[1]
PAGE = str(ROOT / "pages" / "Post_GWAS_Analysis.py")
# the plain message of a table whose settings changed since the last Run GWAS (cc59/FIXES.md)
MESSAGE = ("The genotype table's settings changed since the last Run GWAS on the GWAS page: "
           "run GWAS again (or set the settings back) to use this page.")
N, M = 40, 30
TRAIT = "t"


class _Reached(Exception):
    """The page reached the tabs' context: the run stops there."""


def _blocks(sid, positions):
    return pd.DataFrame({"Chr": ["2"], "Start (bp)": [int(positions[3])],
                         "End (bp)": [int(positions[6])], "lead_snp": [str(sid[4])],
                         "SNP_IDs": [",".join(sid[3:7])], "n_snps": [4]})


@pytest.fixture
def run_page(monkeypatch):
    """Run the page with a session holding a memory table of identity "T1" and the GWAS
    results of one trait; ``extra`` keys are set (None: removed). Returns (what the page
    showed, the tabs' context or None, the session)."""
    import streamlit as st
    from streamlit.runtime.scriptrunner import StopException

    import gwas.ld as gld
    import pages._ld_tabs as tabs
    from gwas.guitable import SessionTable

    rng = np.random.default_rng(3)
    G = np.asfortranarray(rng.binomial(2, 0.4, (N, M)).astype(np.float32))
    ids = np.array([f"S{i:02d}" for i in range(N)])
    positions = (1_000_000 + np.arange(M) * 10_000).astype(np.int64)
    chroms = np.array(["2"] * M)
    sid = np.array([f"SL25ch02p{int(p)}" for p in positions])
    table = SessionTable(identity="T1", file_identity="F", mode="memory", view=G, raw=G,
                         sid=sid, chroms=chroms, positions=positions,
                         iid=np.column_stack([ids, ids]))
    p = rng.uniform(0.01, 1.0, M)
    p[4] = 1e-9
    gwas_df = pd.DataFrame({"SNP": sid, "Chr": chroms, "Pos": positions, "PValue": p})
    pheno = pd.DataFrame({TRAIT: rng.normal(0, 1, N)}, index=ids)
    blocks = _blocks(sid, positions)

    shown, ctxs = [], []
    for name in ("info", "error", "warning"):
        monkeypatch.setattr(st, name, lambda body="", *a, _n=name, **k: shown.append((_n, str(body))))

    def stop():
        raise StopException()

    def context(**kw):
        ctxs.append(kw)
        raise _Reached()

    monkeypatch.setattr(st, "stop", stop)
    monkeypatch.setattr(tabs, "LDContext", context)
    monkeypatch.setattr(gld, "find_ld_clusters_genomewide", lambda **kw: blocks.copy())
    before = set(st.session_state.keys())

    def run(**extra):
        shown.clear()
        ctxs.clear()
        for k in list(st.session_state.keys()):
            if k not in before:
                del st.session_state[k]
        session = {"geno_table": table, "pheno_used_for_hap": pheno, "active_trait": TRAIT,
                   "gwas_results": {TRAIT: gwas_df}, "meff_val": 72,
                   "meff_identity": "T1", "gwas_run_identity": "T1"}
        session.update(extra)
        for k, v in session.items():
            if v is not None:
                st.session_state[k] = v
        try:
            runpy.run_path(PAGE, run_name="__trace_page__")
        except (StopException, _Reached):
            pass
        return list(shown), (ctxs[0] if ctxs else None), st.session_state

    yield run, blocks
    for k in list(st.session_state.keys()):
        if k not in before:
            del st.session_state[k]


def test_results_of_another_table_stop_the_page_plainly(run_page):
    run, _ = run_page
    shown, ctx, _ = run(gwas_run_identity="T0")              # the settings changed since
    assert ctx is None, "the page went on with another table's results"
    assert ("warning", MESSAGE) in shown
    shown, ctx, _ = run(gwas_run_identity=None)              # results without an identity
    assert ctx is None and ("warning", MESSAGE) in shown
    shown, ctx, _ = run(gwas_run_identity="T1")              # this table's results
    assert ctx is not None and not [m for kind, m in shown if kind == "warning"], shown


def test_no_results_and_no_identity_keep_the_old_message(run_page):
    run, _ = run_page
    shown, ctx, _ = run(gwas_run_identity=None, gwas_results=None)
    assert ctx is None and not [m for kind, m in shown if kind == "warning"]
    assert any(kind == "error" and m.startswith("GWAS results not found") for kind, m in shown)


def test_meff_is_taken_only_for_its_own_table(run_page):
    run, _ = run_page
    assert run(meff_identity="T1")[1]["meff_val"] == 72
    assert run(meff_identity="T0")[1]["meff_val"] is None    # computed on another table
    assert run(meff_identity=None)[1]["meff_val"] is None    # table unknown
    assert run(meff_identity="T1", meff_val=None)[1]["meff_val"] is None


def test_the_samples_are_the_tables_own(run_page):
    """A geno_df of the same samples in another order (another trait's run that stopped before
    it wrote the table): the page still labels the table's rows with the table's samples."""
    run, _ = run_page
    ids = [f"S{i:02d}" for i in range(N)]
    stale = pd.DataFrame(index=pd.Index(ids[::-1], dtype=object))
    _, ctx, _ = run(geno_df=stale)
    assert ctx is not None
    assert list(ctx["geno_df"].index) == ids, "the rows were labelled with another frame's samples"
    assert list(run()[1]["geno_df"].index) == ids


def test_the_pages_final_blocks_name_their_trait_and_table(run_page):
    run, blocks = run_page
    _, ctx, state = run()
    assert ctx is not None
    pd.testing.assert_frame_equal(state["ld_blocks_final"], blocks)
    assert state["ld_blocks_final_key"] == (TRAIT, "T1")


def test_the_contract_helpers():
    from types import SimpleNamespace

    t1 = SimpleNamespace(identity="T1")
    assert not gp.run_table_changed({"gwas_results": {}, gp.RUN_IDENTITY: "T1"}, t1)
    assert gp.run_table_changed({"gwas_results": {}, gp.RUN_IDENTITY: "T0"}, t1)
    assert gp.run_table_changed({"gwas_results": {}}, t1)
    assert not gp.run_table_changed({}, t1)
    assert gp.meff_for_table({"meff_val": 5, gp.MEFF_IDENTITY: "T1"}, t1) == 5
    assert gp.meff_for_table({"meff_val": 5, gp.MEFF_IDENTITY: "T0"}, t1) is None
    assert gp.meff_for_table({"meff_val": 5}, t1) is None
    # the exact keys both pages rely on and the message (cc59/FIXES.md, the contract)
    assert (gp.RUN_IDENTITY, gp.MEFF_IDENTITY, gp.BLOCKS_FINAL_KEY) == (
        "gwas_run_identity", "meff_identity", "ld_blocks_final_key")
    assert gp.RUN_CHANGED_MESSAGE == MESSAGE
    assert gp.BLOCKS_FINAL_KEY in gp.POST_GWAS_SESSION_KEYS


def test_every_write_of_the_final_blocks_names_them():
    """Every place that writes ld_blocks_final also writes ld_blocks_final_key next to it."""
    import re

    files = [ROOT / "pages" / "Post_GWAS_Analysis.py"] + sorted((ROOT / "pages" / "_ld_tabs").glob("*.py"))
    writes = 0
    for f in files:
        lines = f.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if re.search(r"""st\.session_state\[["']ld_blocks_final["']\]\s*=(?!=)""", line):
                writes += 1
                near = "\n".join(lines[i:i + 4])
                assert re.search(r"""st\.session_state\[["']ld_blocks_final_key["']\]\s*=\s*\(trait_col, """, near), \
                    (f.name, i + 1)
    assert writes == 2                       # the page's automatic blocks and the LD Blocks tab


def test_the_ld_blocks_tabs_final_blocks_name_their_trait_and_table(monkeypatch):
    """The LD Blocks tab's own detection (a cache miss) writes the key with the blocks."""
    import streamlit as st

    import pages._ld_tabs.tab_genome_wide as tgw
    from pages._ld_tabs import LDContext

    rng = np.random.default_rng(5)
    G = rng.binomial(2, 0.4, (N, M)).astype(float)
    ids = [f"S{i:02d}" for i in range(N)]
    positions = (1_000_000 + np.arange(M) * 10_000).astype(np.int64)
    chroms = np.array(["2"] * M)
    sid = np.array([f"SL25ch02p{int(p)}" for p in positions])
    blocks = _blocks(sid, positions)
    pheno = pd.DataFrame({TRAIT: rng.normal(0, 1, N)}, index=ids)
    gwas_df = pd.DataFrame({"SNP": sid, "Chr": chroms, "Pos": positions,
                            "PValue": rng.uniform(1e-9, 1.0, M)})
    monkeypatch.setattr(tgw, "find_ld_clusters_genomewide", lambda **kw: blocks.copy())
    before = set(st.session_state.keys())
    st.session_state["pheno_used_for_hap"] = pheno
    try:
        ctx = LDContext(geno_ld=G, geno_hard=G, chroms=chroms, positions=positions, sid=sid,
                        geno_df=pd.DataFrame(index=ids), pheno_df=pheno, trait_col=TRAIT,
                        ph_aligned=pheno, y_aligned=pheno[TRAIT].to_numpy(),
                        keep_mask=np.ones(N, dtype=bool), geno_ld_aligned=G, gwas_df=gwas_df,
                        haplo_df_auto=pd.DataFrame(), ld_decay_kb=None, adj_r2_min_global=0.2,
                        ld_trait=TRAIT, geno_dosage_raw=G, table_identity="T1")
        tgw.render(ctx, get_r2_cached=None, cached_extract_block_geno=None,
                   compute_block_qc_effects=None,
                   run_haplotype_block_gwas_cached_fn=lambda *a, **k: (pd.DataFrame(), {}))
        pd.testing.assert_frame_equal(st.session_state["ld_blocks_final"], blocks)
        assert st.session_state["ld_blocks_final_key"] == (TRAIT, "T1")
    finally:
        for k in list(st.session_state.keys()):
            if k not in before:
                del st.session_state[k]
