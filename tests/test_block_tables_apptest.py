"""AppTest coverage for the two block-table labels (GUI brief, Task 2).

Renders ``tab_genome_wide.render`` through a real Streamlit script context with a
seeded LDContext + a small non-empty block inventory + stubbed haplotype compute,
and asserts the block-table sections render:

- Detection: the "LD blocks" section subheader.
- Results: the one block table, "LD blocks and haplotype tests" (it replaced the
  block inventory, the per-block results and the coherence table), whose "?" tooltip
  holds the η² and LD-coherence guidance, with one row per detected block.

The heavy per-block visualization (`_render_block_visualization`, MLG boxplots) is
monkeypatched to a no-op so the render reaches the labels without needing real
genotype extraction. The haplotype compute + η² enrichment are passed as stub
callables — no real GWAS runs, no core statistical module is exercised. The
results table is behind an unkeyed checkbox, so a second ``run()`` checks it.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

_SCRIPT = '''
import sys
_ROOT = r"__ROOT__"
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import numpy as np
import pandas as pd
import streamlit as st
from pages._ld_tabs import LDContext
import pages._ld_tabs.tab_genome_wide as tgw

# no-op the heavy per-block visualization so the render reaches the labels cleanly
tgw._render_block_visualization = lambda **kw: None

rng = np.random.default_rng(0)
n, m = 40, 30
positions = np.sort(rng.integers(1_000_000, 2_000_000, m)).astype(float)
chroms = np.array(["2"] * m)
sid = np.array(["SL25ch02p" + str(int(p)) for p in positions])
geno = rng.binomial(2, 0.3, (n, m)).astype(float)
pval = rng.uniform(1e-9, 1.0, m)
gwas_df = pd.DataFrame({"SNP": sid, "Chr": chroms, "Pos": positions.astype(int), "PValue": pval})

# a small non-empty block inventory -> the inventory expander + caption render
blocks = pd.DataFrame({
    "Chr": ["2", "2"],
    "Start (bp)": [1_000_000, 1_500_000],
    "End (bp)": [1_200_000, 1_700_000],
    "SNP_IDs": [",".join(sid[:5]), ",".join(sid[5:10])],
    "n_snps": [5, 5],
})

pheno = pd.DataFrame({"t": rng.normal(10, 2, n)})
st.session_state["pheno"] = pheno
# the PCs-only haplotype test: this script records no Run kernels (the kinship-aware
# test, the default, needs them; gwas.guipost.run_kernels)
st.session_state["hap_test_choice"] = "PCs and covariates only (the earlier test)"

ctx = LDContext(
    geno_ld=geno, geno_hard=geno, chroms=chroms, positions=positions, sid=sid,
    geno_df=pd.DataFrame(geno), pheno_df=pheno, trait_col="t",
    ph_aligned=pheno, y_aligned=pheno["t"].to_numpy(), keep_mask=np.ones(n, dtype=bool),
    geno_ld_aligned=geno, gwas_df=gwas_df, haplo_df_auto=blocks,
    ld_decay_kb=100.0, adj_r2_min_global=0.2, ld_trait="t",
)

# stub the haplotype compute + enrichment so no real GWAS / core module runs
_fake_hap = pd.DataFrame({
    "Chr": ["2", "2"], "Start": [1_000_000, 1_500_000], "End": [1_200_000, 1_700_000],
    "lead_snp": [sid[0], sid[5]], "n_snps": [5, 5], "PValue": [0.01, 0.2],
})
def _hap_stub(*a, **k):
    return _fake_hap.copy(), {}
def _qc_stub(hap_gwas_df=None, **k):
    return hap_gwas_df
def _r2_stub(*a, **k):
    return np.zeros((2, 2))
def _extract_stub(*a, **k):
    return np.zeros((n, 2)), positions[:2], sid[:2]

# detection now runs unconditionally (the run/refresh checkbox was removed); mock it
# to return the fixture inventory so the always-on path uses these blocks.
tgw.find_ld_clusters_genomewide = lambda **kw: blocks.copy()

tgw.render(
    ctx,
    get_r2_cached=_r2_stub,
    cached_extract_block_geno=_extract_stub,
    compute_block_qc_effects=_qc_stub,
    run_haplotype_block_gwas_cached_fn=_hap_stub,
)
'''


def _only_harness_error(at):
    return all("url_pathname" in str(e.value) for e in at.exception)


def test_block_tables_render_with_new_labels():
    script = _SCRIPT.replace("__ROOT__", str(_ROOT).replace("\\", "/"))
    at = AppTest.from_string(script, default_timeout=120)
    at.run()
    assert not at.exception or _only_harness_error(at)

    # block-inventory section renders (whenever haplo_df_auto is non-empty). The
    # descriptive captions were removed in the UI-text pass, so assert on the surviving
    # "LD blocks" section subheader rather than the deleted caption.
    assert any("LD blocks" in s.value for s in at.subheader), \
        "LD blocks subheader missing"

    # Detection + haplotype analysis now run unconditionally (the run/refresh and
    # 'Run haplotype' checkboxes were removed), so the results table renders on the
    # first run -- there is no checkbox to tick.
    assert not any("Run haplotype" in c.label for c in at.checkbox), \
        "'Run haplotype' checkbox should be gone (analysis is unconditional)"

    # the one block table's subheader
    assert any("LD blocks and haplotype tests" in s.value for s in at.subheader), \
        "block table subheader missing"
    # one row per detected block (2), the tested ones marked
    _tables = [d.value for d in at.dataframe if "Tested" in d.value.columns]
    assert len(_tables) == 1 and len(_tables[0]) == 2, "block table missing"
    assert _tables[0]["Tested"].tolist() == ["yes", "yes"]
    assert any(b.label == "Download block table (CSV)" for b in at.get("download_button"))

    # the effect-size (η²) and LD-coherence guidance is the subheader's "?" tooltip
    _help = " ".join(s.proto.help for s in at.subheader
                     if "LD blocks and haplotype tests" in s.value)
    assert "Variance explained" in _help, "effect-size (η²) guidance missing"
    assert "not adjusted for kinship" in _help, "η² kinship caveat missing"
    assert "member-pair" in _help, "LD-block coherence guidance missing"
    assert "Interpreting Results" in _help, "pointer to Help missing"


def test_no_testable_haplotype_block_shows_the_info_message():
    """A haplotype table with no rows (no block had enough variation) shows the tab's
    message; the block-level FDR step used to raise on it and crash the page."""
    script = _SCRIPT.replace("__ROOT__", str(_ROOT).replace("\\", "/")).replace(
        "return _fake_hap.copy(), {}", "return _fake_hap.iloc[0:0].copy(), {}")
    assert "_fake_hap.iloc[0:0]" in script
    at = AppTest.from_string(script, default_timeout=120)
    at.run()
    assert not at.exception or _only_harness_error(at), [e.value for e in at.exception]
    assert any("No haplotype blocks had enough variation" in i.value for i in at.info)
