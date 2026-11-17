"""The LD Blocks & Haplotypes tab keeps every setting in one "Settings" panel (static).

The tab's former settings (detection, the haplotype test's permutations and minimum
sample counts, the lead-SNP gallery's rule) were spread over the tab. Each must now
appear once in the module, inside ``_render_settings`` (a collapsed "Settings"
expander), with its old key and default, so session state, caches and the CLI's
defaults are unchanged. Plus the summary line above the panel.
"""
import ast
from pathlib import Path

from pages._ld_tabs._decay import decay_run_label
from pages._ld_tabs.tab_genome_wide import _rule_short_name, ld_decay_line, settings_summary

_SRC = (Path(__file__).resolve().parent.parent / "pages" / "_ld_tabs"
        / "tab_genome_wide.py").read_text(encoding="utf-8")

# label -> (widget, key, default): the defaults are the CLI's (cli.py)
_WIDGETS = {
    "Seed threshold for LD blocks (P <)": ("number_input", None, 1e-5),
    "Also include top N SNPs by P-value": ("number_input", None, 10),
    "LD window around each lead SNP (kb)": ("number_input", None, None),   # 2x LD decay
    "r² threshold for cluster definition": ("number_input", None, 0.6),
    "Haplotype permutations": ("number_input", "n_perm_hap", 1000),
    "Test model": ("radio", "hap_test_choice", None),      # index 0: the kinship-aware test
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


def _call_with(src, label):
    """The one st.<widget>(label, ...) call of ``src`` (a module's source)."""
    found = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and n.args
             and isinstance(n.args[0], ast.Constant) and n.args[0].value == label]
    assert len(found) == 1, (label, len(found))
    return found[0]


def test_the_default_haplotype_test_is_the_kinship_aware_one():
    """The GUI's two haplotype test widgets (the tab's "Test model" radio and One-Click's
    "Haplotype test" selectbox) list gwas.guipost.HAP_TEST_CHOICES and select its first entry,
    the kinship-aware test (the CLI's --hap-test default)."""
    from gwas import guipost

    assert list(guipost.HAP_TEST_CHOICES.values())[0] == "kinship"
    one_click = (Path(__file__).resolve().parent.parent / "pages"
                 / "GWAS_analysis.py").read_text(encoding="utf-8")
    for src, label, key, options in [
            (_SRC, "Test model", "hap_test_choice", "list(guipost.HAP_TEST_CHOICES)"),
            (one_click, "Haplotype test", "pipe_hap_test", "list(_HAP_TESTS)")]:
        call = _call_with(src, label)
        kw = {k.arg: k.value for k in call.keywords}
        assert ast.unparse(call.args[1]) == options, label
        assert isinstance(kw.get("index"), ast.Constant) and kw["index"].value == 0, label
        assert kw["key"].value == key, label
    assert "HAP_TEST_CHOICES as _HAP_TESTS" in one_click


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
        "Seeds P < 1e-05 plus top 10 · window 200 kb · r² ≥ 0.6 · kinship-aware test · "
        "1,000 permutations · haplotypes ≥ 5 samples · groups ≥ 3 · gallery: Bonferroni")
    s = settings_summary(5e-8, 0, 144.5, 0.1, 100, 3, 6, "Custom p-value", 1e-6, hap_test="pcs")
    assert s.startswith("Seeds P < 5e-08 · window 144.5 kb · r² ≥ 0.1 · PCs-only test · "
                        "100 permutations")
    assert s.endswith("gallery: custom P < 1.0e-06")
    assert _rule_short_name("M_eff: Li & Ji (LD-aware Bonferroni)") == "M_eff"
    assert _rule_short_name("FDR (q < 0.05)") == "FDR"


def test_the_window_default_is_the_clis_rule():
    """The Settings default is gwas.windows.ld_window_kb of the run's decay (tests/test_ld_window
    pins the rule), and the input's range always holds it: no 200 / 2000 kb limits."""
    fn = next(n for n in ast.walk(ast.parse(_SRC))
              if isinstance(n, ast.FunctionDef) and n.name == "_render_settings")
    src = ast.unparse(fn)
    assert "flank_kb_default = ld_window_kb(ctx.ld_decay_kb)" in src
    assert "value=float(flank_kb_default)" in src
    assert "min_value=0.0" in src and "max_value=max(10000.0, float(flank_kb_default))" in src
    assert "np.clip" not in src


def test_the_ld_decay_line_names_its_run():
    run = decay_run_label({"trait": "locule_number", "timestamp": "2026-11-10 10:30"})
    assert run == "the GWAS page's Run GWAS for `locule_number` (2026-11-10 10:30)"
    assert decay_run_label(None) == "the GWAS page's Run GWAS"
    assert ld_decay_line(72.17, run) == (
        "LD decay (median across chromosomes, from the GWAS page's Run GWAS for "
        "`locule_number` (2026-11-10 10:30)): 72.2 kb, so the window is 144 kb.")
    assert ld_decay_line(1500.9, run).endswith("1500.9 kb, so the window is 3001 kb.")
    assert ld_decay_line(None, run) == (
        "LD decay: no estimate from the GWAS page's Run GWAS for `locule_number` "
        "(2026-11-10 10:30), so the window is 300 kb, as in the CLI, and blocks are "
        "detected without a decay value.")
