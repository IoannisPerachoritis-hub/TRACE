"""AppTest coverage for the two block-table labels (GUI brief, Task 2).

Renders ``tab_genome_wide.render`` through a real Streamlit script context with a
seeded LDContext + a small non-empty block inventory + stubbed haplotype compute,
and asserts the two new labels render:

- Task 2a inventory: the "Block inventory — coordinates and SNP counts" expander
  and its "Where the blocks are. No statistics yet." caption.
- Task 2a results: the "Association results per block" subheader + its caption.

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
    "Lead SNP": [sid[0], sid[5]], "n_snps": [5, 5], "PValue": [0.01, 0.2],
})
def _hap_stub(*a, **k):
    return _fake_hap.copy(), {}
def _qc_stub(hap_gwas_df=None, **k):
    return hap_gwas_df
def _r2_stub(*a, **k):
    return np.zeros((2, 2))
def _extract_stub(*a, **k):
    return np.zeros((n, 2)), positions[:2], sid[:2]

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

    # Task 2a — block-inventory label (renders whenever haplo_df_auto is non-empty)
    assert any("Where the blocks are" in c.value for c in at.caption), \
        "block-inventory caption missing"

    # the results table is behind an unkeyed 'Run haplotype…' checkbox -> check + rerun
    hap_cb = [c for c in at.checkbox if "Run haplotype" in c.label]
    assert hap_cb, "'Run haplotype' checkbox not found"
    hap_cb[0].check().run()
    assert not at.exception or _only_harness_error(at)

    # Task 2a — results-table label + caption
    assert any("Association results per block" in s.value for s in at.subheader), \
        "results-table subheader missing"
    assert any("differ for your trait" in c.value for c in at.caption), \
        "results-table caption missing"
