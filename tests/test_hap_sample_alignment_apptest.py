"""AppTest regression guard for the haplotype tab's sample axis (WO-HAPMASK-01).

Reproduces the reported crash: running a GWAS on a trait that is not the phenotype
file's first column, visiting Post-GWAS Analysis, returning to the GWAS page and
coming back made the LD/haplotype tab die with

    ValueError: Sample mask mismatch in extract_block_geno_for_paper():
    G has 140 samples, but sample_keep_mask has 130.

because the genotype matrix came from ``st.session_state["ld_geno_hard"]`` while the
sample-ID axis (and therefore the phenotype mask) came from
``st.session_state["geno_row_ids"]`` -- two keys written at different gates, so they
can disagree. The same desync hit ``_render_lead_snp_gallery`` one call earlier as a
bare numpy broadcast error.

Two blind spots in ``test_block_tables_apptest`` let this ship, and this harness
removes both: it does NOT stub ``_render_block_visualization`` (except in the one case
that isolates the gallery), and its extractor forwards every argument to the REAL
``gwas.ld.extract_block_geno_for_paper``, so ld.py's mask-length contract is genuinely
exercised. Only upstream compute is stubbed (LD detection, haplotype GWAS, eta2
enrichment) -- no real GWAS runs and no core statistical module is re-implemented.

The cases are parameterised by the four counts that can disagree:
(ctx rows / ctx ids / ``ld_geno_hard`` rows / ``geno_row_ids`` length).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

import pages._ld_tabs.tab_genome_wide as _tgw  # noqa: E402

# AppTest scripts run in THIS process, so a script that monkeypatches a module
# attribute leaks into every later test. Stash the real renderer at collection time
# (before any script has run) so each case can set the attribute explicitly.
if not hasattr(_tgw, "_pristine_render_block_visualization"):
    _tgw._pristine_render_block_visualization = _tgw._render_block_visualization
if not hasattr(_tgw, "_pristine_render_lead_snp_gallery"):
    _tgw._pristine_render_lead_snp_gallery = _tgw._render_lead_snp_gallery

_SCRIPT = '''
import sys
_ROOT = r"__ROOT__"
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import numpy as np
import pandas as pd
import streamlit as st
from gwas.ld import extract_block_geno_for_paper
from pages._ld_tabs import LDContext
import pages._ld_tabs.tab_genome_wide as tgw

N_CTX, N_HARD, N_IDS = __N_CTX__, __N_HARD__, __N_IDS__
STUB_BLOCKVIS = __STUB_BLOCKVIS__
STUB_GALLERY = __STUB_GALLERY__

N_FULL, m = 140, 30
rng = np.random.default_rng(7)
ids_full = np.array(["S%03d" % i for i in range(N_FULL)])
positions = (1_000_000 + np.arange(m) * 10_000).astype(float)
chroms = np.array(["2"] * m)
sid = np.array(["SL25ch02p%d" % int(p) for p in positions])

# Three multi-locus genotypes over the first five markers -> three MLG groups of
# ~47 samples each: every group clears min_hap_count=5 and min_group_size=3, so the
# MLG table, Tukey HSD and the forest plot all complete.
PATTERNS = np.array([[0, 0, 1, 1, 2], [2, 1, 0, 2, 0], [1, 2, 2, 0, 1]], dtype=float)
grp = np.arange(N_FULL) % 3
geno_full = rng.binomial(2, 0.3, (N_FULL, m)).astype(float)
geno_full[:, :5] = PATTERNS[grp]

pheno_full = pd.DataFrame(
    {"t": 10.0 + grp * 2.0 + rng.normal(0.0, 1.0, N_FULL)},
    index=pd.Index(ids_full, dtype=object),
)
# a handful of samples with no value for the analysed trait, so the phenotype mask is
# doing real work: the panel's sample count must be the analysed trait's non-missing
# count on the ctx axis, not simply "every row of some matrix".
MISSING_AT = [5, 15, 25, 35]
pheno_full.iloc[MISSING_AT, 0] = np.nan

pval = np.full(m, 0.5)
pval[:5] = np.array([1e-9, 2e-9, 3e-9, 4e-9, 5e-9])
gwas_df = pd.DataFrame(
    {"SNP": sid, "Chr": chroms, "Pos": positions.astype(int), "PValue": pval})

blocks = pd.DataFrame({
    "Chr": ["2"],
    "Start (bp)": [1_000_000],
    "End (bp)": [1_040_000],
    "lead_snp": [sid[0]],
    "SNP_IDs": [",".join(sid[:5])],
    "n_snps": [5],
})

# --- the two sample axes, sized independently: this IS the defect's shape ---
geno_ctx = geno_full[:N_CTX]
ids_ctx = ids_full[:N_CTX]
st.session_state["ld_geno_hard"] = geno_full[:N_HARD]
st.session_state["ld_chroms"] = chroms
st.session_state["ld_positions"] = positions
st.session_state["ld_sid"] = sid
st.session_state["geno_row_ids"] = ids_full[:N_IDS].tolist()
st.session_state["pheno_used_for_hap"] = pheno_full.copy()
# the two trait names the drift message must be able to name
st.session_state["gwas_run_summary"] = {"trait": "fruit_number"}
st.session_state["selected_traits_multiselect"] = ["fruit_weight"]

ctx = LDContext(
    geno_ld=geno_ctx, geno_hard=geno_ctx, chroms=chroms, positions=positions, sid=sid,
    geno_df=pd.DataFrame(geno_ctx, index=pd.Index(ids_ctx, dtype=object)),
    pheno_df=pheno_full.iloc[:N_CTX], trait_col="t",
    ph_aligned=pheno_full.iloc[:N_CTX],
    y_aligned=pheno_full["t"].to_numpy()[:N_CTX],
    keep_mask=np.ones(N_CTX, dtype=bool), geno_ld_aligned=geno_ctx,
    gwas_df=gwas_df, haplo_df_auto=blocks,
    ld_decay_kb=100.0, adj_r2_min_global=0.2, ld_trait="t",
)

_fake_hap = pd.DataFrame({
    "Chr": ["2"], "Start": [1_000_000], "End": [1_040_000],
    "lead_snp": [sid[0]], "n_snps": [5], "PValue": [0.001],
})
def _hap_stub(*a, **k):
    return _fake_hap.copy(), {}
def _qc_stub(hap_gwas_df=None, **k):
    return hap_gwas_df
def _r2_stub(*a, **k):
    return np.zeros((5, 5))

# The REAL extractor, every argument forwarded -> ld.py's sample-mask contract is
# genuinely exercised (the stub in test_block_tables_apptest ignored its arguments).
def _extract_real(geno_hard, chroms_, positions_, sid_,
                  block_chr, block_start, block_end,
                  sample_keep_mask=None, maf_threshold=0.01,
                  cache_key="", snp_ids=None):
    return extract_block_geno_for_paper(
        geno_hard, chroms_, positions_, sid_,
        block_chr, block_start, block_end,
        sample_keep_mask=sample_keep_mask,
        maf_threshold=maf_threshold,
        snp_ids=snp_ids,
    )

tgw.find_ld_clusters_genomewide = lambda **kw: blocks.copy()
# always set explicitly (never leave a previous script's stub in place). The gallery
# runs BEFORE the block panel, so each site is stubbed out in turn to prove the other
# one independently -- otherwise the gallery's failure masks the block panel's.
tgw._render_block_visualization = (
    (lambda **kw: None) if STUB_BLOCKVIS
    else tgw._pristine_render_block_visualization
)
tgw._render_lead_snp_gallery = (
    (lambda *a, **kw: None) if STUB_GALLERY
    else tgw._pristine_render_lead_snp_gallery
)

tgw.render(
    ctx,
    get_r2_cached=_r2_stub,
    cached_extract_block_geno=_extract_real,
    compute_block_qc_effects=_qc_stub,
    run_haplotype_block_gwas_cached_fn=_hap_stub,
)
'''

_MLG_PANEL = "Haplotype (MLG) allele sequences"
# phenotype values withheld at these positions (see MISSING_AT in the script)
_MISSING_AT = [5, 15, 25, 35]


def _expected_tested(n_ctx):
    """Non-missing phenotype count on the ctx sample axis -- what the panel must use."""
    return n_ctx - sum(1 for i in _MISSING_AT if i < n_ctx)


def _run(n_ctx, n_hard, n_ids, stub_blockvis=False, stub_gallery=False):
    script = (
        _SCRIPT.replace("__ROOT__", str(_ROOT).replace("\\", "/"))
        .replace("__N_CTX__", str(n_ctx))
        .replace("__N_HARD__", str(n_hard))
        .replace("__N_IDS__", str(n_ids))
        .replace("__STUB_BLOCKVIS__", "True" if stub_blockvis else "False")
        .replace("__STUB_GALLERY__", "True" if stub_gallery else "False")
    )
    at = AppTest.from_string(script, default_timeout=180)
    at.run()
    return at


def _only_harness_error(at):
    return all("url_pathname" in str(e.value) for e in at.exception)


def _assert_clean(at):
    assert not at.exception or _only_harness_error(at), \
        "unexpected exception: " + "; ".join(str(e.value) for e in at.exception)


# --- control: every axis agrees, as it does right after a completed GWAS run ---
def test_happy_path_renders():
    at = _run(130, 130, 130)
    _assert_clean(at)
    assert any(_MLG_PANEL in m.value for m in at.markdown), \
        "MLG panel missing on the happy path"
    assert len(at.session_state["sample_vis"]) == _expected_tested(130)


# --- (a): the tab takes its sample axis and its matrix from ONE source (ctx), so a
# stale session mirror can no longer mis-pair rows or blow up the extractor. ---
def test_block_panel_ignores_stale_session_mirrors():
    # the gallery is stubbed out so this exercises the block panel -- i.e. the REAL
    # extract_block_geno_for_paper, whose size-only mask guard produced the reported
    # "G has 140 samples, but sample_keep_mask has 130" ValueError.
    at = _run(130, 130 + 10, 130, stub_gallery=True)
    _assert_clean(at)
    assert any(_MLG_PANEL in m.value for m in at.markdown), \
        "block panel did not render past the extractor"
    # and it used the analysed trait's samples -- not the 140-row session mirror
    assert len(at.session_state["sample_vis"]) == _expected_tested(130)


def test_lead_snp_gallery_ignores_stale_session_mirrors():
    # the gallery runs BEFORE the block panel, so it is stubbed out here to isolate it
    at = _run(130, 130 + 10, 130, stub_blockvis=True)
    _assert_clean(at)
    assert any("genotype effect" in m.value for m in at.markdown), \
        "lead-SNP gallery did not render a SNP panel"


# --- the reported click-path: geno_row_ids belongs to the last completed run, the
# genotype objects were rebuilt for a different trait. Refuse, actionably. ---
def test_drifted_state_refuses_with_an_actionable_message():
    at = _run(140, 140, 130)
    _assert_clean(at)
    errs = " ".join(e.value for e in at.error)
    assert "fruit_number" in errs, "the analysed trait is not named"
    assert "fruit_weight" in errs, "the loaded trait is not named"
    assert "140" in errs and "130" in errs, "both sample counts are not given"
    assert "Re-run the GWAS on" in errs, "no actionable instruction"
    assert not any(_MLG_PANEL in m.value for m in at.markdown), \
        "MLG panel rendered on a drifted sample axis"


# --- and the guard must not over-fire once the user follows that instruction ---
def test_after_rerun_renders():
    at = _run(140, 140, 140)
    _assert_clean(at)
    assert any(_MLG_PANEL in m.value for m in at.markdown), \
        "MLG panel missing after the re-run state"
    assert len(at.session_state["sample_vis"]) == _expected_tested(140)
    assert not at.error, "guard fired on a coherent sample axis"
