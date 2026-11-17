"""A table file closed while a Post-GWAS tab reads it (Clear session data in any session, or the
server's registry making room for another dataset) is said plainly, and never gives a table that
looks right.

- The Regional Plot and the LD-block coherence (gwas.ld.compute_block_ld_quality) took every
  ValueError of the r2-to-lead read for "the lead SNP is not typed". A closed table raises a
  ValueError too, so the regional CSV and the coherence row came out with no r2 and no word of
  the release. They now test that the lead is typed, as compute_r2_to_lead does, and let any
  read error through: on the page it reaches released_guard, which shows RELEASED_MESSAGE.
- The Significant SNPs tab and the LD Blocks tab's coherence catch every error themselves and
  showed it, with the table file's path. With the session's table released they show
  RELEASED_MESSAGE and stop instead; any other error is shown as before.

The tabs run here outside a Streamlit page (bare mode), on a small table written to a store;
st.stop raises StopException, as in a real run.
"""
import io

import numpy as np
import pandas as pd
import pytest

N, M = 60, 40
LEAD = 12


@pytest.fixture
def tables(tmp_path):
    """The raw and the mean-imputed table of one store, their labels, a GWAS table, one LD
    block around LEAD, and the session's table object (disk)."""
    from types import SimpleNamespace

    from gwas.genostore import LazyColumns, write_store
    from gwas.guitable import SessionTable

    rng = np.random.default_rng(4)
    base = rng.binomial(2, 0.4, N).astype(float)
    G = np.empty((N, M))
    for j in range(M):                          # neighbours of LEAD in LD with it
        G[:, j] = base if abs(j - LEAD) <= 3 else rng.binomial(2, 0.3, N)
        flip = rng.random(N) < 0.1
        G[flip, j] = rng.binomial(2, 0.5, int(flip.sum()))
    G[rng.random((N, M)) < 0.03] = np.nan
    store = write_store(tmp_path / "store", G)
    fill = np.nan_to_num(np.nanmean(G, axis=0)).astype(np.float32)
    view = LazyColumns(store, np.arange(M), fill=fill)
    raw = LazyColumns(store, np.arange(M))
    positions = (1_000_000 + np.arange(M) * 10_000).astype(np.int64)
    chroms = np.array(["2"] * M)
    sid = np.array([f"SL25ch02p{int(p)}" for p in positions])
    p = rng.uniform(0.05, 1.0, M)
    p[LEAD] = 1e-9
    gwas_df = pd.DataFrame({"SNP": sid, "Chr": chroms, "Pos": positions, "PValue": p})
    members = sid[LEAD - 3:LEAD + 4]
    blocks = pd.DataFrame({"Chr": ["2"], "Start (bp)": [int(positions[LEAD - 3])],
                           "End (bp)": [int(positions[LEAD + 3])], "lead_snp": [sid[LEAD]],
                           "SNP_IDs": [",".join(members)]})
    table = SessionTable(identity="T", file_identity="F", mode="disk",
                         parsed=SimpleNamespace(store=store), view=view, raw=raw,
                         sid=sid, chroms=chroms, positions=positions,
                         iid=np.array([[f"S{i}", f"S{i}"] for i in range(N)]))
    yield SimpleNamespace(store=store, view=view, raw=raw, G=G, positions=positions,
                          chroms=chroms, sid=sid, gwas_df=gwas_df, blocks=blocks, table=table)
    store.close()


def _closed_copy(t, tmp_path):
    """The raw table of a second store holding the same calls, closed."""
    from gwas.genostore import LazyColumns, write_store

    other = write_store(tmp_path / "closed", t.G)
    other.close()
    return LazyColumns(other, np.arange(M))


# ------------------------------------------------------------ the LD-block coherence (shared)
def test_the_block_coherence_does_not_take_a_closed_table_for_an_untyped_lead(tables, tmp_path):
    from gwas.ld import compute_block_ld_quality

    t = tables
    imputed = np.asarray(t.view)                    # the members' reads succeed
    args = (t.blocks, t.chroms, t.positions, t.sid, imputed, t.gwas_df)
    open_row = compute_block_ld_quality(*args, geno_dosage_raw=t.raw).iloc[0]
    assert bool(open_row["ldq_lead_in_block"]) and np.isfinite(open_row["ldq_r2_lead_median"])
    # the raw table's file closed before the lead's read: the read error, not "lead absent"
    with pytest.raises(ValueError, match="is closed"):
        compute_block_ld_quality(*args, geno_dosage_raw=_closed_copy(t, tmp_path))
    # a lead that is not typed is still "not in the block", with no r2 (as before)
    untyped = t.blocks.assign(lead_snp="not_typed")
    row = compute_block_ld_quality(untyped, *args[1:], geno_dosage_raw=t.raw).iloc[0]
    assert row["ldq_lead_snp"] == "not_typed" and not bool(row["ldq_lead_in_block"])
    assert np.isnan(row["ldq_r2_lead_median"]) and np.isnan(row["ldq_frac_members_r2_lead_ge"])


def test_the_block_coherence_of_a_typed_lead_is_unchanged(tables):
    """Pin the row of a typed lead against the expression the function always evaluated."""
    from gwas.ld import compute_block_ld_quality
    from gwas.plotting import compute_r2_to_lead

    t = tables
    imputed = np.asarray(t.view)
    row = compute_block_ld_quality(t.blocks, t.chroms, t.positions, t.sid, imputed, t.gwas_df,
                                   geno_dosage_raw=t.raw).iloc[0]
    mask = np.zeros(M, dtype=bool)
    mask[LEAD - 3:LEAD + 4] = True
    r2, _ = compute_r2_to_lead(t.raw, t.sid, t.sid[LEAD], mask, 20)
    r2 = r2[np.isfinite(r2)]
    assert row["ldq_r2_lead_median"] == float(np.median(r2))
    assert row["ldq_r2_lead_min"] == float(np.min(r2))
    assert row["ldq_frac_members_r2_lead_ge"] == float(np.mean(r2 >= 0.6))


# ------------------------------------------------------------------- the Regional Plot tab
def _ctx(t, raw, **over):
    from pages._ld_tabs import LDContext

    base = dict(geno_ld=raw, geno_hard=raw, chroms=t.chroms, positions=t.positions, sid=t.sid,
                geno_df=pd.DataFrame(index=[f"S{i}" for i in range(N)]), pheno_df=pd.DataFrame(),
                trait_col="t", ph_aligned=pd.DataFrame(), y_aligned=np.zeros(N),
                keep_mask=np.ones(N, dtype=bool), geno_ld_aligned=raw, gwas_df=t.gwas_df,
                haplo_df_auto=pd.DataFrame(), ld_decay_kb=100.0, adj_r2_min_global=0.2,
                geno_dosage_raw=raw, meff_val=None, sig_rule_label="Bonferroni (α = 0.05)",
                table_identity="T")
    base.update(over)
    return LDContext(**base)


def _window(t, lead):
    from pages._ld_tabs._window import RegionWindow

    return RegionWindow(lead_snp=lead, chr="2", start_bp=int(t.positions.min()),
                        end_bp=int(t.positions.max()), core_start=None, core_end=None,
                        use_block=False, label="window", lead_pos=int(t.positions[LEAD]),
                        buffer_kb=200, block_snp_ids="")


@pytest.fixture
def regional(monkeypatch):
    """tab_regional.render, recording its regional CSV (figures and their files skipped)."""
    import streamlit as st

    import pages._ld_tabs.tab_regional as tr

    out = {}
    monkeypatch.setattr(st, "download_button",
                        lambda label, data, *a, **k: out.__setitem__(label, data) or False)
    monkeypatch.setattr(tr, "show_plot", lambda *a, **k: None)
    monkeypatch.setattr(tr, "export_matplotlib", lambda *a, **k: None)

    def render(ctx, window):
        out.clear()
        tr.render(ctx, window)
        return pd.read_csv(io.BytesIO(out["Download regional data (CSV)"]))
    return render


def test_the_regional_plot_does_not_take_a_closed_table_for_an_untyped_lead(tables, tmp_path, regional):
    from gwas import guipost as gp

    t = tables
    lead = str(t.sid[LEAD])
    csv = regional(_ctx(t, t.raw), _window(t, lead))
    assert np.isfinite(csv["r2_to_lead"]).sum() > 3          # the open table: r2 to the lead
    closed = _closed_copy(t, tmp_path)
    with pytest.raises(ValueError, match="is closed"):
        regional(_ctx(t, closed), _window(t, lead))
    # on the page, the read error reaches released_guard: the plain message, not a CSV

    class Released(Exception):
        pass

    def on_released():
        raise Released()

    t.store.close()
    with pytest.raises(Released):
        with gp.released_guard(t.table, on_released):
            regional(_ctx(t, t.raw), _window(t, lead))


def test_an_untyped_lead_still_gives_grey_points(tables, regional):
    t = tables
    csv = regional(_ctx(t, t.raw), _window(t, "not_typed"))
    assert len(csv) == M and csv["r2_to_lead"].isna().all()


# ------------------------------------------------- tabs that catch the read error themselves
@pytest.fixture
def page(monkeypatch):
    """The session as the page leaves it for the tabs, with st.info / st.error / st.caption
    recorded and st.stop raising StopException; the keys the tabs write are removed after."""
    import streamlit as st
    from streamlit.runtime.scriptrunner import StopException

    before = set(st.session_state.keys())
    shown = []
    for name in ("info", "error", "caption", "warning"):
        monkeypatch.setattr(st, name, lambda body="", *a, _n=name, **k: shown.append((_n, str(body))))

    def stop():
        raise StopException()

    monkeypatch.setattr(st, "stop", stop)
    st.session_state.pop("_sigtable_cache", None)
    yield st.session_state, shown
    for k in set(st.session_state.keys()) - before:
        del st.session_state[k]


def test_the_significant_snps_tab_says_the_table_was_released(tables, tmp_path, page, monkeypatch):
    from streamlit.runtime.scriptrunner import StopException

    import gwas.sigtable as sigtable
    from gwas import guipost as gp
    from pages._ld_tabs import tab_significant_snps as T

    state, shown = page
    t = tables
    state["geno_table"] = t.table
    # an error while the table is open (a malformed frame): shown as before
    real = sigtable.build_significant_snp_table

    def boom(*a, **k):
        raise ValueError("boom")

    monkeypatch.setattr(sigtable, "build_significant_snp_table", boom)
    T.render(_ctx(t, t.raw))
    assert ("error", "Could not build the significant-SNP table: boom") in shown
    # the table file closed during the build: the plain message and a stop, no file path
    monkeypatch.setattr(sigtable, "build_significant_snp_table", real)
    shown.clear()
    t.store.close()
    with pytest.raises(StopException):
        T.render(_ctx(t, t.raw))
    assert ("info", gp.RELEASED_MESSAGE) in shown
    assert not [m for kind, m in shown if kind == "error"], shown
    assert not any("calls.i8" in m for _, m in shown), shown


def test_the_ld_blocks_coherence_says_the_table_was_released(tables, page):
    from streamlit.runtime.scriptrunner import StopException

    from gwas import guipost as gp
    from pages._ld_tabs import tab_genome_wide as tgw

    state, shown = page
    t = tables
    pheno = pd.DataFrame({"t": np.random.default_rng(2).normal(0, 1, N)},
                         index=[f"S{i}" for i in range(N)])
    state["geno_table"] = t.table
    state["pheno_used_for_hap"] = pheno

    def no_tests(*a, **k):                      # the haplotype test is not what this checks
        return pd.DataFrame(), {}

    def run():
        ctx = _ctx(t, t.view, geno_dosage_raw=t.raw, pheno_df=pheno, ph_aligned=pheno,
                   ld_trait="t", haplo_df_auto=t.blocks)
        tgw._render_haplotype_gwas(ctx=ctx, haplo_df_auto=t.blocks, cached_extract_block_geno=None,
                                   compute_block_qc_effects=None,
                                   run_haplotype_block_gwas_cached_fn=no_tests,
                                   settings={"hap_test": "pcs"})

    run()                                       # the table open: no coherence message
    assert not any(m.startswith("LD-block coherence unavailable") for _, m in shown), shown
    shown.clear()
    t.store.close()
    with pytest.raises(StopException):
        run()
    assert ("info", gp.RELEASED_MESSAGE) in shown
    assert not any("calls.i8" in m for _, m in shown), shown


def test_a_significant_snp_table_built_on_a_closed_table_is_neither_kept_nor_shown(tables, page):
    """The build leaves the per-SNP MAF empty when that read fails (gwas/sigtable.py). With every
    significant SNP below the seed threshold, no other read follows, so a closed table gave a
    complete-looking table without MAF, which the tab showed and kept for later runs. With the
    session's table released, the tab now shows the plain message and keeps nothing."""
    from streamlit.runtime.scriptrunner import StopException

    from gwas import guipost as gp
    from pages._ld_tabs import tab_significant_snps as T

    state, shown = page
    t = tables
    state["geno_table"] = t.table
    gwas_df = t.gwas_df.copy()
    gwas_df.loc[LEAD, "PValue"] = 1e-4     # significant (0.05 / 40 = 1.25e-3), not a seed (1e-5)
    t.store.close()
    with pytest.raises(StopException):
        T.render(_ctx(t, t.raw, gwas_df=gwas_df))
    assert ("info", gp.RELEASED_MESSAGE) in shown
    assert not state.get("_sigtable_cache"), "a table built on a closed file was kept"
