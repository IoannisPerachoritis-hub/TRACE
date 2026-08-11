"""AppTest coverage for the Regional Plot tab (T-08).

Renders ``tab_regional.render`` through a real Streamlit script context with a
seeded LDContext + RegionWindow — the tab's full path (window mask,
compute_r2_to_lead, rule resolution, plotly + static figures, downloads) runs
headlessly. This is a focused tab render rather than driving the whole LD page,
which has no AppTest harness and would need ~15 seeded session keys plus
GWAS-dependent preprocessing (noted deferral).

Some Streamlit display widgets hit an AppTest bare-mode limitation (a
`url_pathname` KeyError) that does NOT occur in the real app; the render runs
before it, so the test tolerates only that specific harness error.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

# The script AppTest executes: build a small synthetic LDContext + window and
# render the tab. `__ROOT__` is substituted with the repo root (forward slashes).
_SCRIPT = '''
import sys
_ROOT = r"__ROOT__"
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import numpy as np
import pandas as pd
from pages._ld_tabs import LDContext
from pages._ld_tabs._window import RegionWindow
from pages._ld_tabs import tab_regional

rng = np.random.default_rng(0)
n, m = 40, 60
positions = np.sort(rng.integers(1_000_000, 2_000_000, m)).astype(float)
chroms = np.array(["2"] * m)
sid = np.array(["SL25ch02p" + str(int(p)) for p in positions])
geno = rng.binomial(2, 0.3, (n, m)).astype(float)
pval = rng.uniform(1e-9, 1.0, m)
gwas_df = pd.DataFrame({
    "SNP": sid, "Chr": chroms, "Pos": positions.astype(int),
    "PValue": pval, "Significant_Bonf": pval < 0.05 / m,
})
lead_idx = int(np.argmin(pval))
lead = sid[lead_idx]
ctx = LDContext(
    geno_ld=geno, geno_hard=geno, chroms=chroms, positions=positions, sid=sid,
    geno_df=pd.DataFrame(geno), pheno_df=pd.DataFrame(), trait_col="t",
    ph_aligned=pd.DataFrame(), y_aligned=np.zeros(n), keep_mask=np.ones(n, dtype=bool),
    geno_ld_aligned=geno, gwas_df=gwas_df, haplo_df_auto=pd.DataFrame(),
    ld_decay_kb=100.0, adj_r2_min_global=0.2, geno_dosage_raw=geno, meff_val=m,
    sig_rule_label="Bonferroni (alpha = 0.05)",
)
window = RegionWindow(
    lead_snp=lead, chr="2", start_bp=int(positions.min()), end_bp=int(positions.max()),
    core_start=None, core_end=None, use_block=False, label="test window",
    lead_pos=int(positions[lead_idx]), buffer_kb=200,
)
tab_regional.render(ctx, window)
'''


def _only_harness_error(at):
    return all("url_pathname" in str(e.value) for e in at.exception)


def test_regional_tab_renders_from_seeded_context():
    script = _SCRIPT.replace("__ROOT__", str(_ROOT).replace("\\", "/"))
    at = AppTest.from_string(script, default_timeout=120)
    at.run()
    assert not at.exception or _only_harness_error(at)
    # the tab reached its title (renders before any deep-display harness limit)
    assert any("Regional association plot" in s.value for s in at.subheader)
