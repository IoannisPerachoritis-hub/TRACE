"""AppTest coverage for the Local LD "Show lead-SNP star" toggle.

Renders ``tab_local_ld.render`` through a real Streamlit script context with a seeded
LDContext + a RegionWindow whose lead SNP sits inside the plotted region, and a stub
``get_r2_cached``. The heavy region extraction (``extract_block_geno_for_paper``) is
monkeypatched to a fixed region that includes the lead, and the r² matrix is stubbed,
so the render reaches the heatmap + caption without real genotype extraction / LD compute.

The gold-star matplotlib figure is closed after ``savefig``, so the toggle's effect is
asserted on the OBSERVABLE caption text (``at.caption``), not on figure artists:
- default (checkbox on): a caption states the gold star marks the lead SNP;
- after unchecking "Show lead-SNP star": no caption mentions the gold star.

It also checks that flipping the toggle produces a DISTINCT byte-cache entry (star-on vs
star-off render different image bytes), i.e. the toggle is threaded into ``local_cache_key``.

``render``'s signature is intentionally left unchanged (the toggle is an in-body
``st.checkbox``), so the ``TestTabLocalLDImport`` signature lock stays green.
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
from pages._ld_tabs._window import RegionWindow
import pages._ld_tabs.tab_local_ld as tll

rng = np.random.default_rng(0)
n, m = 30, 12
positions = np.arange(1_000_000, 1_000_000 + m * 1000, 1000).astype(float)
sid = np.array(["SL25ch02p" + str(int(p)) for p in positions])
chroms = np.array(["2"] * m)
geno = rng.binomial(2, 0.3, (n, m)).astype(float)
lead = str(sid[5])

# fixed region extraction (includes the lead) so the star path is exercised
tll.extract_block_geno_for_paper = lambda **kw: (geno, positions, sid)

def _r2_stub(region_geno, region_pos, min_pair_n=20, cache_key=None):
    return np.eye(int(region_geno.shape[1]))

pheno = pd.DataFrame({"t": rng.normal(0, 1, n)})
gwas_df = pd.DataFrame({"SNP": sid, "Chr": chroms, "Pos": positions.astype(int),
                        "PValue": rng.uniform(1e-9, 1.0, m)})
ctx = LDContext(
    geno_ld=geno, geno_hard=geno, chroms=chroms, positions=positions, sid=sid,
    geno_df=pd.DataFrame(geno), pheno_df=pheno, trait_col="t",
    ph_aligned=pheno, y_aligned=pheno["t"].to_numpy(), keep_mask=np.ones(n, dtype=bool),
    geno_ld_aligned=geno, gwas_df=gwas_df, haplo_df_auto=pd.DataFrame(),
    ld_decay_kb=100.0, adj_r2_min_global=0.2, ld_trait="t",
)
window = RegionWindow(
    lead_snp=lead, chr="2",
    start_bp=int(positions.min()), end_bp=int(positions.max()),
    core_start=None, core_end=None, use_block=False,
    label="test window", lead_pos=int(positions[5]), buffer_kb=100, block_snp_ids="",
)
tll.render(ctx, _r2_stub, window)
'''


def _only_harness_error(at):
    return all("url_pathname" in str(e.value) for e in at.exception)


def _star_captions(at):
    return [c for c in at.caption if "gold star" in c.value]


def test_star_toggle_hides_and_shows_lead_star():
    script = _SCRIPT.replace("__ROOT__", str(_ROOT).replace("\\", "/"))
    at = AppTest.from_string(script, default_timeout=120)
    at.run()
    assert not at.exception or _only_harness_error(at)

    # default (checkbox on): the star is drawn + the caption says so
    assert _star_captions(at), "gold-star caption missing when the toggle is on"

    # uncheck "Show lead-SNP star" -> the star + its caption note disappear
    star_cb = [c for c in at.checkbox if "Show lead-SNP star" in c.label]
    assert star_cb, "'Show lead-SNP star' checkbox not found"
    star_cb[0].uncheck().run()
    assert not at.exception or _only_harness_error(at)
    assert not _star_captions(at), "gold-star caption still present when the toggle is off"

    # the toggle is part of the byte-cache key: star-on and star-off produced distinct
    # cached renders (so flipping it never serves a stale image).
    cache = at.session_state["_local_ld_cache"]
    assert len(cache) >= 2, "star toggle not reflected in the local-LD byte-cache key"
    displays = {v["display"] for v in cache.values()}
    assert len(displays) >= 2, "cached heatmap image bytes identical for star on/off"
