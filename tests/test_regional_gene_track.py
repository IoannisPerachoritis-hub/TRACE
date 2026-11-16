"""The Regional Plot's gene track is the gene model chosen now in the Gene Annotation tab.

The page renders the Regional Plot before the Gene Annotation tab, and switching tabs does not
rerun the script. The Regional Plot read the gene model the Gene Annotation tab had stored in
the session on the run before (st.session_state["genes_df"]): after a change of species, build
or gene file, its figure and its PNG / SVG / PDF showed the previous model, and on a first visit
none. It now takes the model chosen now (the Gene Annotation tab's widget values, as Streamlit
holds them at the start of the run, or their defaults), loaded by that tab's cached loader. With
the model unchanged the figure is the same.

The tab runs here outside a Streamlit page (bare mode); the genes it draws are recorded. The
file uses only names that 7eb09ea has too.
"""
import numpy as np
import pandas as pd
import pytest

TOMATO = "Solanum lycopersicum (tomato)"
CHR, START, END = "2", 30_000_000, 30_400_000
M = 30


@pytest.fixture
def regional(monkeypatch):
    """Render the Regional Plot tab on a window of tomato chromosome 2; returns the genes its
    static figure draws (Gene_IDs, or None)."""
    import streamlit as st

    import pages._ld_tabs.tab_regional as tr
    from pages._ld_tabs import LDContext
    from pages._ld_tabs._window import RegionWindow

    positions = np.linspace(START + 5_000, END - 5_000, M).astype(np.int64)
    chroms = np.array([CHR] * M)
    sid = np.array([f"SL25ch02p{int(p)}" for p in positions])
    rng = np.random.default_rng(1)
    G = rng.binomial(2, 0.4, (40, M)).astype(float)
    p = rng.uniform(0.01, 1.0, M)
    p[10] = 1e-9
    gwas_df = pd.DataFrame({"SNP": sid, "Chr": chroms, "Pos": positions, "PValue": p})
    ctx = LDContext(geno_ld=G, geno_hard=G, chroms=chroms, positions=positions, sid=sid,
                    geno_df=pd.DataFrame(index=[f"S{i}" for i in range(40)]), pheno_df=pd.DataFrame(),
                    trait_col="t", ph_aligned=pd.DataFrame(), y_aligned=np.zeros(40),
                    keep_mask=np.ones(40, dtype=bool), geno_ld_aligned=G, gwas_df=gwas_df,
                    haplo_df_auto=pd.DataFrame(), ld_decay_kb=100.0, adj_r2_min_global=0.2,
                    geno_dosage_raw=G, sig_rule_label="Bonferroni (α = 0.05)", has_annotation=True)
    window = RegionWindow(lead_snp=str(sid[10]), chr=CHR, start_bp=START, end_bp=END,
                          core_start=None, core_end=None, use_block=False, label="window",
                          lead_pos=int(positions[10]), buffer_kb=200, block_snp_ids="")
    drawn = []
    real = tr.plot_regional_association_static

    def capture(*a, **kw):
        g = kw.get("genes")
        drawn.append(None if g is None else tuple(g["Gene_ID"].astype(str)))
        return real(*a, **kw)

    monkeypatch.setattr(tr, "plot_regional_association_static", capture)
    monkeypatch.setattr(tr, "show_plot", lambda *a, **k: None)
    monkeypatch.setattr(tr, "export_matplotlib", lambda *a, **k: None)
    monkeypatch.setattr(st, "download_button", lambda *a, **k: False)
    keys = ("ld_gene_species", "ld_genome_build", "gene_model_upload", "gene_desc_upload",
            "genes_df", "regional_gene_labels")
    saved = {k: st.session_state[k] for k in keys if k in st.session_state}

    def render(**session):
        for k in keys:
            st.session_state.pop(k, None)
        for k, v in session.items():
            st.session_state[k] = v
        drawn.clear()
        tr.render(ctx, window)
        assert len(drawn) == 1
        return drawn[0]

    yield render
    for k in keys:
        st.session_state.pop(k, None)
    st.session_state.update(saved)


def _model(build):
    """The bundled tomato gene model of ``build``, as the Gene Annotation tab loads it."""
    from pages._ld_tabs.tab_gene_annotation import _bytes, _load_genes
    from utils.species_files import SPECIES_FILES

    files = SPECIES_FILES[TOMATO]
    return _load_genes(_bytes(files[f"gene_model_{build}"]), _bytes(files[f"gene_desc_{build}"]))


def _in_window(model):
    from pages._ld_tabs.tab_regional import _filter_genes

    sub = _filter_genes(model, CHR, START, END)
    return None if sub is None else tuple(sub["Gene_ID"].astype(str))


def test_the_regional_plot_draws_the_gene_model_chosen_now(regional):
    sl3, sl4 = _model("SL3"), _model("SL4")
    want3, want4 = _in_window(sl3), _in_window(sl4)
    assert want3 and want4 and want3 != want4, "the window must show the two builds apart"
    # the build changed to SL4; the session still holds the SL3 model the Gene Annotation tab
    # loaded on the run before
    assert regional(ld_gene_species=TOMATO, ld_genome_build="SL4", genes_df=sl3) == want4
    # a first visit: the Gene Annotation tab's defaults (tomato, SL3), loaded or not yet
    assert regional() == want3
    # the model unchanged: the same genes as the stored model gave
    assert regional(ld_gene_species=TOMATO, ld_genome_build="SL3", genes_df=sl3) == want3


def test_no_gene_model_draws_no_track(regional):
    assert regional(ld_gene_species="Other (upload files)") is None
