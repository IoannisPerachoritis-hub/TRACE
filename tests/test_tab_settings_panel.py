"""The LD Blocks & Haplotypes tab keeps every setting in one "Settings" panel (static).

The tab's former settings (detection, the haplotype test's permutations and minimum
sample counts, the lead-SNP gallery's rule) were spread over the tab. Each must now
appear once in the module, inside ``_render_settings`` (a collapsed "Settings"
expander), with its old key and default, so session state, caches and the CLI's
defaults are unchanged. Plus the summary line above the panel.
"""
import ast
from pathlib import Path

import numpy as np

from pages._ld_tabs.tab_genome_wide import (_rule_short_name, ld_decay_line,
                                            ld_window_default_kb, settings_summary)

_SRC = (Path(__file__).resolve().parent.parent / "pages" / "_ld_tabs"
        / "tab_genome_wide.py").read_text(encoding="utf-8")

# label -> (widget, key, default): the defaults are the CLI's (cli.py)
_WIDGETS = {
    "Seed threshold for LD blocks (P <)": ("number_input", None, 1e-5),
    "Also include top N SNPs by P-value": ("number_input", None, 10),
    "LD window around each lead SNP (kb)": ("number_input", None, None),   # 2x LD decay
    "r² threshold for cluster definition": ("number_input", None, 0.6),
    "Haplotype permutations": ("number_input", "n_perm_hap", 1000),
    "Minimum total samples per haplotype (collapse rarer → Other)": ("number_input", None, 5),
    "Minimum samples per haplotype group (for testing)": ("number_input", None, 3),
    "Significance rule:": ("selectbox", "lead_gallery_rule", None),
    "Custom p-value": ("number_input", "lead_gallery_custom", None),
}


def _widget_calls(tree):
    """(enclosing function, widget, label, key, value) for every st widget call."""
    out = []

    def visit(node, func):
        for child in ast.iter_child_nodes(node):
            f = child.name if isinstance(child, ast.FunctionDef) else func
            if (isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                    and isinstance(child.func.value, ast.Name) and child.func.value.id == "st"
                    and child.func.attr in ("number_input", "selectbox", "slider", "radio")
                    and child.args and isinstance(child.args[0], ast.Constant)):
                kw = {k.arg: k.value for k in child.keywords}
                key = kw["key"].value if isinstance(kw.get("key"), ast.Constant) else None
                val = kw["value"].value if isinstance(kw.get("value"), ast.Constant) else None
                out.append((func, child.func.attr, child.args[0].value, key, val))
            visit(child, f)

    visit(tree, None)
    return out


def test_every_former_setting_is_in_the_panel_once_with_its_key_and_default():
    calls = _widget_calls(ast.parse(_SRC))
    for label, (widget, key, default) in _WIDGETS.items():
        found = [c for c in calls if c[2] == label]
        assert len(found) == 1, (label, found)
        func, w, _label, k, v = found[0]
        assert func == "_render_settings", (label, func)
        assert (w, k) == (widget, key), label
        if default is not None:
            assert v == default, (label, v)
    assert "Significance threshold (P <)" not in _SRC          # renamed
    assert "#### Haplotype analysis parameters" not in _SRC     # its inputs moved


def test_the_panel_is_one_collapsed_expander_named_settings():
    fn = next(n for n in ast.walk(ast.parse(_SRC))
              if isinstance(n, ast.FunctionDef) and n.name == "_render_settings")
    exp = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
           and isinstance(n.func, ast.Attribute) and n.func.attr == "expander"]
    assert len(exp) == 1 and exp[0].args[0].value == "Settings"
    assert {k.arg: k.value.value for k in exp[0].keywords}["expanded"] is False
    seed = next(n for n in ast.walk(fn) if isinstance(n, ast.Call) and n.args
                and isinstance(n.args[0], ast.Constant)
                and n.args[0].value == "Seed threshold for LD blocks (P <)")
    help_txt = ast.literal_eval({k.arg: k.value for k in seed.keywords}["help"])
    assert "--ld-seed-p" in help_txt and "not the genome-wide significance rule" in help_txt


def test_summary_line():
    assert settings_summary(1e-5, 10, 200.0, 0.6, 1000, 5, 3, "Bonferroni (α = 0.05)") == (
        "Seeds P < 1e-05 plus top 10 · window 200 kb · r² ≥ 0.6 · 1,000 permutations · "
        "haplotypes ≥ 5 samples · groups ≥ 3 · gallery: Bonferroni")
    s = settings_summary(5e-8, 0, 144.5, 0.1, 100, 3, 6, "Custom p-value", 1e-6)
    assert s.startswith("Seeds P < 5e-08 · window 144.5 kb · r² ≥ 0.1 · 100 permutations")
    assert s.endswith("gallery: custom P < 1.0e-06")
    assert _rule_short_name("M_eff: Li & Ji (LD-aware Bonferroni)") == "M_eff"
    assert _rule_short_name("FDR (q < 0.05)") == "FDR"


def test_the_window_default_is_unchanged():
    """2 x the GWAS page's LD decay within 200-2000 kb, the expression the panel had."""
    for x in (5.0, 50.0, 72.17, 99.9, 100.0, 150.0, 999.9, 1000.0, 1500.0):
        assert ld_window_default_kb(x) == float(np.clip(x * 2, 200, 2000)), x
    fn = next(n for n in ast.walk(ast.parse(_SRC))
              if isinstance(n, ast.FunctionDef) and n.name == "_render_settings")
    assert "flank_kb_default = ld_window_default_kb(ctx.ld_decay_kb)" in ast.unparse(fn)
    assert "value=float(flank_kb_default)" in ast.unparse(fn)


def test_the_ld_decay_line():
    assert ld_decay_line(150.4, True) == (
        "LD decay (median across chromosomes): 150.4 kb, so the window is 300.8 kb.")
    assert ld_decay_line(250, True) == (
        "LD decay (median across chromosomes): 250 kb, so the window is 500 kb.")
    assert ld_decay_line(72.17, True) == (
        "LD decay (median across chromosomes): 72.2 kb, so the window is 200 kb "
        "(2 × 72.2 kb, held at the 200 kb minimum).")
    assert ld_decay_line(1500, True).endswith("(2 × 1500 kb, held at the 2000 kb maximum).")
    assert ld_decay_line(200.0, False) == (
        "LD decay: no estimate from the GWAS page in this session, so the default 200 kb "
        "is used and the window is 400 kb.")
