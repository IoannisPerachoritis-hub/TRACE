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
- The PCs: those GWAS_CTX holds, which the GWAS page writes with the table. It writes the
  session's own "pcs" and "pcs_full" earlier, so a run of it for another sample set that stopped
  before the table (at its sample overlap warning) left that set's PCs beside the earlier
  table: the haplotype test, eta^2 and the PCA panel took them, and a smaller set gave a
  "PC matrix does not match" traceback. The LD Blocks tab now also stops plainly when the PCs
  are not of the table's samples.

The page and the tab run here outside a Streamlit page (bare mode), on a small table in memory:
st.stop raises StopException as in a real run, and the run ends where the page builds the tabs'
context (LDContext), which is recorded.
"""
import ast
import runpy
import sys
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
        # with the page's own names (its compute_block_qc_effects), from the caller's frame
        ctxs.append(dict(kw, _page=sys._getframe(1).f_globals))
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


# the MLM frame the GWAS page writes as gwas_df during a Run, before gwas_results
_FRAME = pd.DataFrame({"SNP": ["SL25ch02p1040000"], "PValue": [1e-9], "Chr": ["2"],
                       "Pos": [1_040_000]})


def test_a_frame_without_an_identity_counts_as_results_without_one(run_page):
    """The re-check's rc3 case: a first Run GWAS that did not finish (a page switch during
    MLMM) leaves only its MLM frame (gwas_df), without an identity; the settings then change.
    The page took that frame, the earlier table's p-values, on the new table. It now stops with
    the run-again message, before it reads any result."""
    run, _ = run_page
    shown, ctx, _ = run(gwas_results=None, gwas_run_identity=None, gwas_df=_FRAME)
    assert ctx is None, "the page went on with a frame of another table"
    assert ("warning", MESSAGE) in shown


def test_a_frame_beside_another_traits_results_is_not_taken(run_page):
    """A Run for this trait that did not finish locked it (trait_col_locked) and left a frame
    (gwas_df), beside the last finished Run's results of another trait, whose table the session
    holds again. The page took that frame, which is not this trait's results; it now stops with
    the run-again message."""
    run, _ = run_page
    shown, ctx, _ = run(gwas_results={"u": _FRAME}, active_trait="u", trait_col_locked=TRAIT,
                        gwas_df=_FRAME)
    assert ctx is None, "the page went on with a frame that is not this trait's results"
    assert ("warning", MESSAGE) in shown


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
    assert gp.run_table_changed({"gwas_df": _FRAME}, t1)                # a frame, no identity
    assert not gp.run_table_changed({"gwas_df": None}, t1)
    assert not gp.run_table_changed({"gwas_df": _FRAME, gp.RUN_IDENTITY: "T1"}, t1)
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


def test_a_detection_served_from_the_tabs_cache_is_the_final_one(monkeypatch):
    """The re-check's rc3 case: the LD Blocks tab at r2 0.5, then back at 0.6, whose detection
    the tab serves from its cache. ld_blocks_final stayed the 0.5 detection under the same key
    (trait, table), so the GWAS page's block stability aggregated blocks no longer on screen.
    On a cache hit the tab now writes the blocks on screen, and their key, too."""
    import streamlit as st

    import pages._ld_tabs.tab_genome_wide as tgw
    from pages._ld_tabs import LDContext

    rng = np.random.default_rng(6)
    G = rng.binomial(2, 0.4, (N, M)).astype(float)
    ids = [f"S{i:02d}" for i in range(N)]
    positions = (1_000_000 + np.arange(M) * 10_000).astype(np.int64)
    chroms = np.array(["2"] * M)
    sid = np.array([f"SL25ch02p{int(p)}" for p in positions])
    at = {0.6: _blocks(sid, positions),
          0.5: pd.DataFrame({"Chr": ["2"], "Start (bp)": [int(positions[9])],
                             "End (bp)": [int(positions[12])], "lead_snp": [str(sid[10])],
                             "SNP_IDs": [",".join(sid[9:13])], "n_snps": [4]})}
    pheno = pd.DataFrame({TRAIT: rng.normal(0, 1, N)}, index=ids)
    gwas_df = pd.DataFrame({"SNP": sid, "Chr": chroms, "Pos": positions,
                            "PValue": rng.uniform(1e-9, 1.0, M)})
    detected = []

    def find(**kw):
        detected.append(kw["ld_threshold"])
        return at[kw["ld_threshold"]].copy()

    r2 = {"value": 0.6}
    real_input = st.number_input

    def number_input(label, *a, **k):            # the tab's r2 setting, as a user sets it
        if label == "r² threshold for cluster definition":
            return r2["value"]
        return real_input(label, *a, **k)

    monkeypatch.setattr(tgw, "find_ld_clusters_genomewide", find)
    monkeypatch.setattr(st, "number_input", number_input)
    before = set(st.session_state.keys())
    st.session_state["pheno_used_for_hap"] = pheno
    try:
        ctx = LDContext(geno_ld=G, geno_hard=G, chroms=chroms, positions=positions, sid=sid,
                        geno_df=pd.DataFrame(index=ids), pheno_df=pheno, trait_col=TRAIT,
                        ph_aligned=pheno, y_aligned=pheno[TRAIT].to_numpy(),
                        keep_mask=np.ones(N, dtype=bool), geno_ld_aligned=G, gwas_df=gwas_df,
                        haplo_df_auto=pd.DataFrame(), ld_decay_kb=None, adj_r2_min_global=0.2,
                        ld_trait=TRAIT, geno_dosage_raw=G, table_identity="T1")
        for value in (0.6, 0.5, 0.6):
            r2["value"] = value
            tgw.render(ctx, get_r2_cached=None, cached_extract_block_geno=None,
                       compute_block_qc_effects=None,
                       run_haplotype_block_gwas_cached_fn=lambda *a, **k: (pd.DataFrame(), {}))
        assert detected == [0.6, 0.5], "the last render was not served from the tab's cache"
        pd.testing.assert_frame_equal(st.session_state["ld_blocks_final"], at[0.6])
        assert st.session_state["ld_blocks_final_key"] == (TRAIT, "T1")
    finally:
        for k in list(st.session_state.keys()):
            if k not in before:
                del st.session_state[k]


# ------------------------------------------------------------------------------- the PCs
def _pcs(rng, n):
    """(pcs, pcs_full) of n samples: the model's 3 PCs and the run's 20."""
    full = rng.normal(size=(n, 20))
    return full[:, :3].copy(), full


def test_the_pcs_are_those_written_with_the_table(run_page):
    """The re-check's window: the session's "pcs" and "pcs_full" are those of an interrupted
    GWAS page run for another sample set, of the same size (rc3b) or smaller (rc3d); GWAS_CTX
    holds the table's. The tabs get GWAS_CTX's; without GWAS_CTX, none: the session's keys are
    never taken."""
    run, _ = run_page
    rng = np.random.default_rng(11)
    ids = np.array([f"S{i:02d}" for i in range(N)])
    pcs, pcs_full = _pcs(rng, N)
    gwas_ctx = {"pcs": pcs, "pcs_full": pcs_full, "geno_index": ids}
    for n_other in (N, N - 10):                          # rc3b, rc3d
        other, other_full = _pcs(rng, n_other)
        _, ctx, _ = run(GWAS_CTX=gwas_ctx, pcs=other, pcs_full=other_full)
        assert ctx is not None
        assert ctx["pcs"] is pcs, "the tabs got the session's PCs of another sample set"
        assert ctx.get("pcs_full") is pcs_full, "the PCA panel would draw the session's PCs"
    _, ctx, _ = run(pcs=other, pcs_full=other_full)
    assert ctx["pcs"] is None and ctx.get("pcs_full") is None


def test_the_block_effects_take_the_pcs_written_with_the_table(run_page, monkeypatch):
    """eta^2 of the LD Blocks tab (the page's compute_block_qc_effects) is adjusted for
    GWAS_CTX's PCs, as the haplotype test is, not for the session's "pcs"."""
    run, _ = run_page
    rng = np.random.default_rng(12)
    pcs, pcs_full = _pcs(rng, N)
    other, other_full = _pcs(rng, N)
    _, ctx, _ = run(GWAS_CTX={"pcs": pcs, "pcs_full": pcs_full}, pcs=other, pcs_full=other_full)
    seen = {}
    monkeypatch.setattr(gp, "block_qc_effects", lambda **kw: seen.update(kw))
    ctx["_page"]["compute_block_qc_effects"](hap_gwas_df=None)
    assert seen["pcs"] is pcs, "eta^2 was adjusted for the session's PCs of another sample set"


def test_the_pca_panel_draws_the_pcs_it_is_given(monkeypatch):
    """The haplotype PCA panel draws the PCs the page gives it (GWAS_CTX's), not the
    session's "pcs_full" (here another run's, of fewer samples)."""
    import streamlit as st

    import pages._ld_tabs.tab_genome_wide as tgw

    rng = np.random.default_rng(13)
    ids = np.array([f"S{i:02d}" for i in range(N)])
    _, pcs_full = _pcs(rng, N)
    drawn = []
    monkeypatch.setattr(st, "fragment", lambda f=None, **k: f if f is not None else (lambda g: g))
    monkeypatch.setattr(tgw, "show_plot", lambda fig, size="", **k: drawn.append(k.get("data")))
    assignments = pd.DataFrame({"Sample": ids, "Haplotype": ["H1", "H2"] * (N // 2),
                                "in_haplotype_test": True})
    before = set(st.session_state.keys())
    st.session_state["pcs_full"] = _pcs(rng, N - 10)[1]
    try:
        tgw._render_haplotype_pca(assignments, ids, "2", 1, 2, pcs_full=pcs_full)
    finally:
        for k in set(st.session_state.keys()) - before:
            del st.session_state[k]
    points = [d for d in drawn if isinstance(d, pd.DataFrame)]
    assert len(points) == 1
    np.testing.assert_array_equal(points[0][["PC1", "PC2"]].to_numpy(), pcs_full[:, :2])
    assert list(points[0]["Sample"]) == list(ids)


class _Tested(Exception):
    """The haplotype test was reached: the run stops there."""


def test_pcs_of_other_samples_stop_the_ld_blocks_tab_plainly(monkeypatch):
    """PCs that are not of the table's samples (fewer rows, as rc3d left them; or GWAS_CTX's
    samples another set, as rc3b) stop the LD Blocks tab with a plain message before the
    haplotype test, which raised "PC matrix does not match" or paired the rows wrongly. The
    table's own PCs, and no PCs, reach the test."""
    import streamlit as st
    from streamlit.runtime.scriptrunner import StopException

    import pages._ld_tabs.tab_genome_wide as tgw
    from pages._ld_tabs import LDContext

    rng = np.random.default_rng(14)
    G = rng.binomial(2, 0.4, (N, M)).astype(float)
    ids = np.array([f"S{i:02d}" for i in range(N)])
    positions = (1_000_000 + np.arange(M) * 10_000).astype(np.int64)
    chroms = np.array(["2"] * M)
    sid = np.array([f"SL25ch02p{int(p)}" for p in positions])
    blocks = _blocks(sid, positions)
    pheno = pd.DataFrame({TRAIT: rng.normal(0, 1, N)}, index=ids)
    gwas_df = pd.DataFrame({"SNP": sid, "Chr": chroms, "Pos": positions,
                            "PValue": rng.uniform(1e-9, 1.0, M)})
    shown = []
    monkeypatch.setattr(st, "error", lambda body="", *a, **k: shown.append(str(body)))

    def stop():
        raise StopException()

    def tests(*a, **k):
        raise _Tested()

    monkeypatch.setattr(st, "stop", stop)
    before = set(st.session_state.keys())

    def run(pcs, pcs_full=None, geno_index=ids):
        shown.clear()
        st.session_state["GWAS_CTX"] = {"pcs": pcs, "pcs_full": pcs_full, "geno_index": geno_index}
        st.session_state["pheno_used_for_hap"] = pheno
        given = {"pcs_full": pcs_full} if pcs_full is not None else {}
        ctx = LDContext(geno_ld=G, geno_hard=G, chroms=chroms, positions=positions, sid=sid,
                        geno_df=pd.DataFrame(index=ids), pheno_df=pheno, trait_col=TRAIT,
                        ph_aligned=pheno, y_aligned=pheno[TRAIT].to_numpy(),
                        keep_mask=np.ones(N, dtype=bool), geno_ld_aligned=G, gwas_df=gwas_df,
                        haplo_df_auto=blocks, ld_decay_kb=None, adj_r2_min_global=0.2,
                        ld_trait=TRAIT, pcs=pcs, geno_dosage_raw=G, table_identity="T1",
                        **given)
        tgw._render_haplotype_gwas(ctx=ctx, haplo_df_auto=blocks, cached_extract_block_geno=None,
                                   compute_block_qc_effects=None,
                                   run_haplotype_block_gwas_cached_fn=tests, settings=None)

    pcs, pcs_full = _pcs(rng, N)
    try:
        for bad in (dict(pcs=pcs[:-10]), dict(pcs=pcs, pcs_full=pcs_full[:-10]),
                    dict(pcs=pcs, pcs_full=pcs_full, geno_index=ids[::-1])):
            with pytest.raises(StopException):
                run(**bad)
            assert len(shown) == 1 and shown[0].startswith(
                "The principal components on this page are not those of the genotype table's "
                f"{N} samples"), shown
        for good in (dict(pcs=pcs, pcs_full=pcs_full), dict(pcs=None)):
            with pytest.raises(_Tested):
                run(**good)
            assert not shown, shown
    finally:
        for k in set(st.session_state.keys()) - before:
            del st.session_state[k]


def test_the_page_and_its_tabs_never_read_the_sessions_pcs():
    """The PCs are read through gwas.guipost.run_pcs (GWAS_CTX), never from the session's
    "pcs" or "pcs_full" keys, which the GWAS page writes before the table."""
    import re

    files = [ROOT / "pages" / "Post_GWAS_Analysis.py"] + sorted((ROOT / "pages" / "_ld_tabs").glob("*.py"))
    reads = [(f.name, key) for f in files for key in re.findall(
        r"""session_state(?:\.get\(\s*|\[\s*)["'](pcs|pcs_full)["']""", f.read_text(encoding="utf-8"))]
    assert not reads, reads


def test_the_gwas_page_writes_both_pcs_with_the_table():
    """The GWAS page's GWAS_CTX holds both PC matrices and the sample IDs that name their rows,
    and is the statement right before the table's (geno_table): no Streamlit call between the
    two can end a run with one written and not the other."""
    tree = ast.parse((ROOT / "pages" / "GWAS_analysis.py").read_text(encoding="utf-8"))

    def key(stmt):
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Subscript):
            s = stmt.targets[0].slice
            return s.value if isinstance(s, ast.Constant) else None
        return None

    found = 0
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        for i, stmt in enumerate(body if isinstance(body, list) else []):
            if key(stmt) == "GWAS_CTX":
                found += 1
                names = {k.value for k in stmt.value.keys if isinstance(k, ast.Constant)}
                assert {"pcs", "pcs_full", "geno_index"} <= names, names
                assert key(body[i + 1]) == "geno_table"
    assert found == 1
