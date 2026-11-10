"""gwas.windows.ld_window_kb: the LD window by the CLI's rule, int(2 x the median LD decay), or
300 kb without an estimate. Every default window, in the CLI and the GUI, uses it."""
import json
import math
from pathlib import Path

import pytest

from gwas.isolated import edge_flank_bp_from_decay
from gwas.windows import ld_window_kb

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tests" / "golden" / "tomato_locule" / "run_manifest.json"


@pytest.mark.parametrize("decay,window", [
    (40.6, 81),              # under 100 kb: the GUI used to hold it at 200
    (72.17, 144),            # the Varitome locule run
    (150.4, 300),            # in range
    (1500.9, 3001),          # over 1000 kb: the GUI used to hold it at 2000
    (None, 300),             # no estimate
    (float("nan"), 300),
])
def test_the_window_is_the_clis_rule(decay, window):
    assert ld_window_kb(decay) == window
    if decay is not None and math.isfinite(decay):
        assert ld_window_kb(decay) == int(2 * decay)      # the CLI's own expression


def test_the_goldens_decay_gives_its_flank():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert ld_window_kb(m["ld_decay_kb"]) == m["flank_kb"] == 144


def test_the_edge_flank_follows_the_window():
    assert edge_flank_bp_from_decay(72.17) == 144_000
    assert edge_flank_bp_from_decay(None) == 300_000


def _src(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_every_default_window_uses_the_rule():
    """The CLI's two branches, the GWAS page's run, Post-GWAS's auto-built blocks, the
    Settings default and the Regional Plot's window all call ld_window_kb; no limits
    or assumed decay remain."""
    pipe = _src("gwas/pipeline.py")
    assert "ld_flank_kb = ld_window_kb(ld_decay_kb)" in pipe
    assert "ld_flank_kb = ld_window_kb(None)" in pipe
    assert "flank_kb = ld_window_kb(ld_decay_kb)" in _src("pages/GWAS_analysis.py")
    post = _src("pages/Post_GWAS_Analysis.py")
    assert "flank_kb=ld_window_kb(ld_decay_kb)" in post
    assert "ld_decay_kb * 2" not in post and '"ld_decay_kb", 200.0' not in post
    tab = _src("pages/_ld_tabs/tab_genome_wide.py")
    assert "flank_kb_default = ld_window_kb(ctx.ld_decay_kb)" in tab
    assert "np.clip(float(ld_decay_kb) * 2, 200, 2000)" not in tab
    assert "ld_window_kb(ctx.ld_decay_kb)" in _src("pages/_ld_tabs/_window.py")
