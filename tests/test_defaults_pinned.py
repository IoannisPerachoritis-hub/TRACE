"""T-72 — Default-pinning tests (Batch A regression harness).

Pins every load-bearing default in the block-detection / haplotype / annotation
stack, by introspection (``inspect``) or AST literal extraction, so that an edit
to a default is a test failure even when no behaviour visibly moves. This is the
constant-level complement to the behavioural goldens (T-71 / T-94): a behavioural
golden can be masked by a compensating change; a pinned constant cannot.

Every assertion names the ``file:line`` it protects, so a failure tells the
reader exactly where to look. The line numbers are against the
``Solanaceae-gwas`` tree at the time of writing (branch ``revision/round1``).
They are advisory — each assertion checks the *value*, never the line — but are
kept current so the failure message is useful. If a line has drifted, correct
the message; if a *value* has changed, that is the regression this file exists
to catch.

Spec: ``docs/revision/specs/integration_and_testing.md`` §1.3.
"""
import ast
import inspect
from pathlib import Path

import pytest

import annotation
import cli
from gwas import haplotype, impute, ld, models, plotting, reports

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# introspection / AST helpers
# ---------------------------------------------------------------------------
def _sig_defaults(fn):
    """Mapping of parameter name -> default value for callables reachable by
    ``inspect`` (module-level functions)."""
    return {
        k: v.default
        for k, v in inspect.signature(fn).parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def _load(relpath):
    """Return ``(source_text, ast_tree)`` for a repo-relative ``.py`` file.

    GUI page modules are read and parsed here rather than imported, because
    importing a Streamlit page executes its UI at import time
    (``pages/Post_GWAS_Analysis.py`` calls ``ld_analysis_page()`` at module level).
    """
    path = REPO / relpath
    src = path.read_text(encoding="utf-8")
    return src, ast.parse(src)


def _find_funcdef(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found in tree")


def _call_name(call):
    """Callable name for an ``ast.Call`` whether ``foo(...)`` or ``a.b.foo(...)``."""
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return getattr(call.func, "id", None)


def _module_const(tree, name):
    """Value of the single ``name = <constant>`` assignment anywhere in the tree
    (walks into function bodies — ``MAX_REGION_SNPS`` etc. are in-function)."""
    found = [
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    ]
    assert found, f"no `{name} = <constant>` assignment found"
    assert len(set(found)) == 1, f"`{name}` assigned inconsistent constants: {found}"
    return found[0]


def _nested_arg_default(tree, funcname, argname):
    """Default of ``argname`` on a (possibly nested) ``def funcname(...)`` —
    reachable by AST but not by ``inspect`` when the def is nested."""
    fn = _find_funcdef(tree, funcname)
    args = fn.args
    positional = list(args.posonlyargs) + list(args.args)
    defaults = list(args.defaults)
    if defaults:
        aligned = dict(zip([a.arg for a in positional[len(positional) - len(defaults):]], defaults))
        if argname in aligned and isinstance(aligned[argname], ast.Constant):
            return aligned[argname].value
    for a, d in zip(args.kwonlyargs, args.kw_defaults):
        if a.arg == argname and isinstance(d, ast.Constant):
            return d.value
    raise AssertionError(f"{funcname}({argname}=?) constant default not found")


def _call_kwarg_src(src, tree, called_func, kwarg, within=None):
    """Source text of the ``kwarg`` value in a call to ``called_func``.

    ``within`` restricts the search to the body of that enclosing function so a
    same-named call elsewhere in the module cannot be matched by accident.
    """
    scope = _find_funcdef(tree, within) if within else tree
    for node in ast.walk(scope):
        if isinstance(node, ast.Call) and _call_name(node) == called_func:
            for kw in node.keywords:
                if kw.arg == kwarg:
                    return ast.get_source_segment(src, kw.value)
    raise AssertionError(
        f"call {called_func}(..., {kwarg}=?) not found"
        + (f" inside {within}()" if within else "")
    )


def _widget_value_by_target(tree, target_name):
    """``value=`` literal of a ``target_name = st.<widget>(...)`` assignment."""
    widgets = {"number_input", "slider", "selectbox", "checkbox", "radio", "text_input"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if _call_name(node.value) in widgets and any(
                isinstance(t, ast.Name) and t.id == target_name for t in node.targets
            ):
                for kw in node.value.keywords:
                    if kw.arg == "value" and isinstance(kw.value, ast.Constant):
                        return kw.value.value
    raise AssertionError(f"widget assigned to {target_name!r} with constant value= not found")


def _widget_value_by_label(tree, label_substr):
    """``value=`` literal of the ``st.<widget>`` whose label contains ``label_substr``."""
    widgets = {"number_input", "slider", "selectbox", "checkbox", "radio", "text_input"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) in widgets and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str) and label_substr in first.value:
                for kw in node.keywords:
                    if kw.arg == "value" and isinstance(kw.value, ast.Constant):
                        return kw.value.value
    raise AssertionError(f"widget with label containing {label_substr!r} and constant value= not found")


def _min_first_const(tree, funcname):
    """First positional constant of a ``min(<const>, ...)`` call inside ``funcname``."""
    fn = _find_funcdef(tree, funcname)
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "min" and node.args:
            if isinstance(node.args[0], ast.Constant):
                return node.args[0].value
    raise AssertionError(f"min(<const>, ...) not found in {funcname}()")


def _compare_const(tree, funcname, left_name, op_type):
    """Constant compared against ``left_name`` under ``op_type`` inside ``funcname``."""
    fn = _find_funcdef(tree, funcname)
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == left_name
            and len(node.ops) == 1
            and isinstance(node.ops[0], op_type)
            and isinstance(node.comparators[0], ast.Constant)
        ):
            return node.comparators[0].value
    raise AssertionError(f"`{left_name} {op_type.__name__} <const>` not found in {funcname}()")


def _ifexp_consts_for_target(tree, target_name):
    """``(body, orelse)`` constants of ``target_name = A if cond else B``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.IfExp):
            if any(isinstance(t, ast.Name) and t.id == target_name for t in node.targets):
                body, orelse = node.value.body, node.value.orelse
                if isinstance(body, ast.Constant) and isinstance(orelse, ast.Constant):
                    return body.value, orelse.value
    raise AssertionError(f"`{target_name} = <const> if ... else <const>` not found")


# ===========================================================================
# 1. signature defaults (28) — inspect
# ===========================================================================
# (dotted label, callable, param, expected, "file:line")
SIGNATURE_DEFAULTS = [
    ("ld.find_ld_clusters_genomewide", ld.find_ld_clusters_genomewide, "ld_threshold", 0.6, "gwas/ld.py:818"),
    ("ld.find_ld_clusters_genomewide", ld.find_ld_clusters_genomewide, "flank_kb", 300, "gwas/ld.py:819"),
    ("ld.find_ld_clusters_genomewide", ld.find_ld_clusters_genomewide, "min_snps", 3, "gwas/ld.py:820"),
    ("ld.find_ld_clusters_genomewide", ld.find_ld_clusters_genomewide, "top_n", 0, "gwas/ld.py:821"),
    ("ld.find_ld_clusters_genomewide", ld.find_ld_clusters_genomewide, "sig_thresh", 1e-5, "gwas/ld.py:822"),
    ("ld.find_ld_clusters_genomewide", ld.find_ld_clusters_genomewide, "adj_r2_min", 0.2, "gwas/ld.py:825"),
    ("ld.find_ld_clusters_genomewide", ld.find_ld_clusters_genomewide, "min_pair_n", 20, "gwas/ld.py:826"),
    ("ld.find_ld_clusters_genomewide", ld.find_ld_clusters_genomewide, "merge_iou", 0.3, "gwas/ld.py:827"),
    ("ld.find_ld_clusters_genomewide", ld.find_ld_clusters_genomewide, "gap_factor", 10.0, "gwas/ld.py:828"),
    ("ld.find_ld_clusters_genomewide", ld.find_ld_clusters_genomewide, "ld_merge_mode", "occupancy", "gwas/ld.py (--ld-merge-mode)"),
    ("ld.find_ld_clusters_genomewide", ld.find_ld_clusters_genomewide, "ld_merge_r2", 0.5, "gwas/ld.py (--ld-merge-r2)"),
    ("ld.find_ld_blocks_graph", ld.find_ld_blocks_graph, "min_snps", 3, "gwas/ld.py:690"),
    ("ld.find_ld_blocks_graph", ld.find_ld_blocks_graph, "adj_r2_min", 0.3, "gwas/ld.py:691"),
    ("ld.contiguous_segments_by_adjacent", ld.contiguous_segments_by_adjacent, "adj_r2_min", 0.3, "gwas/ld.py:445"),
    ("ld._adaptive_adj_threshold", ld._adaptive_adj_threshold, "base", 0.3, "gwas/ld.py:39"),
    ("ld._adaptive_adj_threshold", ld._adaptive_adj_threshold, "frac", 0.5, "gwas/ld.py:39"),
    ("ld.find_ld_blocks_from_genotypes", ld.find_ld_blocks_from_genotypes, "adj_r2_min", 0.2, "gwas/ld.py:1030"),
    ("ld.find_ld_blocks_from_genotypes", ld.find_ld_blocks_from_genotypes, "min_snps", 3, "gwas/ld.py:1029"),
    ("ld.filter_contained_blocks", ld.filter_contained_blocks, "min_contained", 2, "gwas/ld.py:1284"),
    ("ld.filter_contained_blocks", ld.filter_contained_blocks, "size_ratio_threshold", 3.0, "gwas/ld.py:1285"),
    ("ld.filter_contained_blocks", ld.filter_contained_blocks, "mode", "remove", "gwas/ld.py:1286"),
    ("ld.pairwise_r2", ld.pairwise_r2, "min_pair_n", 20, "gwas/ld.py:64"),
    ("haplotype.run_haplotype_block_gwas", haplotype.run_haplotype_block_gwas, "min_hap_count", 5, "gwas/haplotype.py:24"),
    ("haplotype.run_haplotype_block_gwas", haplotype.run_haplotype_block_gwas, "min_group_size", 3, "gwas/haplotype.py:25"),
    ("haplotype.run_haplotype_block_gwas", haplotype.run_haplotype_block_gwas, "n_perm", 1000, "gwas/haplotype.py:26"),
    ("annotation._find_flanking_genes", annotation._find_flanking_genes, "n_flank", 2, "annotation.py:143"),
    ("annotation._find_flanking_genes", annotation._find_flanking_genes, "max_dist_bp", 500_000, "annotation.py:143"),
    ("annotation.annotate_ld_blocks", annotation.annotate_ld_blocks, "n_flank", 2, "annotation.py:160"),
    ("annotation.annotate_ld_blocks", annotation.annotate_ld_blocks, "max_flank_dist_bp", 500_000, "annotation.py:161"),
    ("plotting.compute_r2_to_lead", plotting.compute_r2_to_lead, "min_pair_n", 15, "plotting.py:409"),
    # D-102: the shipped FarmCPU final scan is the classical OLS fixed-effect test.
    ("models.run_farmcpu", models.run_farmcpu, "final_scan", "ols", "gwas/models.py:1183"),
    # LD-kNNi (Money et al. 2015) opt-in imputer -- the published k/l/c defaults.
    ("impute.ld_knni", impute.ld_knni, "k", 5, "gwas/impute.py:153"),
    ("impute.ld_knni", impute.ld_knni, "l", 20, "gwas/impute.py:153"),
    ("impute.ld_knni", impute.ld_knni, "c", 1.0, "gwas/impute.py:153"),
]


@pytest.mark.parametrize(
    "label,fn,param,expected,loc",
    SIGNATURE_DEFAULTS,
    ids=[f"{lab.split('.')[-1]}.{p}" for lab, _fn, p, _e, _l in SIGNATURE_DEFAULTS],
)
def test_signature_default(label, fn, param, expected, loc):
    defaults = _sig_defaults(fn)
    assert param in defaults, f"{label}({param}=…) default removed — was pinned at {loc}"
    assert defaults[param] == expected, (
        f"{label}({param}) default changed: {defaults[param]!r} != {expected!r} — {loc}"
    )


def test_signature_default_count_is_stable():
    """The pinned set is the signature defaults enumerated by T-72, plus D-102's
    FarmCPU final_scan. A new load-bearing default is added here deliberately."""
    assert len(SIGNATURE_DEFAULTS) == 34


# ===========================================================================
# 2. in-code constants (9) — AST literal extraction
# ===========================================================================
def test_const_pairwise_r2_snp_ceiling():
    _src, tree = _load("gwas/ld.py")
    assert _compare_const(tree, "pairwise_r2", "m", ast.Gt) == 4000, "gwas/ld.py:80 (m > 4000 guard)"


def test_const_adaptive_adj_threshold_cap():
    _src, tree = _load("gwas/ld.py")
    assert _min_first_const(tree, "_adaptive_adj_threshold") == 0.5, "gwas/ld.py:60 (min(0.5, …) cap)"


def test_const_max_region_snps():
    _src, tree = _load("gwas/ld.py")
    assert _module_const(tree, "MAX_REGION_SNPS") == 1500, "gwas/ld.py:912"


def test_const_dist_floor_bp():
    _src, tree = _load("gwas/ld.py")
    assert _module_const(tree, "DIST_FLOOR_BP") == 200_000, "gwas/ld.py:954"


def test_const_thin_block_snps_r2():
    _src, tree = _load("gwas/haplotype.py")
    assert _nested_arg_default(tree, "_thin_block_snps", "r2_thresh") == 0.9, (
        "gwas/haplotype.py:182 (applied :218)"
    )


def test_const_retention_warn_floor():
    _src, tree = _load("gwas/haplotype.py")
    assert _compare_const(tree, "run_haplotype_block_gwas", "frac_retained", ast.Lt) == 0.75, (
        "gwas/haplotype.py:305 (rare-haplotype retention warn floor)"
    )


def test_const_df_to_html_max_rows():
    # signature default, reachable by inspect
    assert _sig_defaults(reports._df_to_html)["max_rows"] == 30, "gwas/reports.py:34"


def test_const_zip_figure_dpi_split():
    _src, tree = _load("gwas/plotting.py")
    assert _ifexp_consts_for_target(tree, "_dpi") == (150, 600), "gwas/plotting.py:363 (LD_heatmap 150 else 600)"


def test_const_gui_r2_cache_capacity():
    _src, tree = _load("pages/Post_GWAS_Analysis.py")
    assert _module_const(tree, "MAX_R2_CACHE") == 10, "pages/Post_GWAS_Analysis.py:107"


# ===========================================================================
# 3. CLI defaults (4) — parser introspection
# ===========================================================================
CLI_DEFAULTS = [
    ("sig_thresh", "bonferroni", "cli.py --sig-thresh"),
    ("species", "tomato", "cli.py --species"),
    ("genome_build", "SL3", "cli.py --genome-build"),
    ("seed", 42, "cli.py:104 --seed"),
    # guard against an accidental default flip of imputation to LD-kNNi (type c).
    ("impute", "mean", "cli.py --impute"),
]


@pytest.mark.parametrize("attr,expected,loc", CLI_DEFAULTS, ids=[c[0] for c in CLI_DEFAULTS])
def test_cli_default(attr, expected, loc):
    args = cli._build_parser().parse_args([])
    assert getattr(args, attr) == expected, f"CLI default {attr}={getattr(args, attr)!r} != {expected!r} — {loc}"


def test_hardcoded_pipeline_determinism_seed():
    """Tree note: unlike the dev tree (which hard-codes ``seed=42``), the TRACE
    paper repo exposes a real ``--seed`` flag (``cli.py:104``, default 42, pinned
    in ``CLI_DEFAULTS`` above). The subsampling stage forwards it, so the pin here
    is that the determinism seed still reaches that stage as ``seed=args.seed``
    rather than being replaced by a divergent literal."""
    src, tree = _load("cli.py")
    val = _call_kwarg_src(src, tree, "subsample_gwas_resampling", "seed")
    assert val == "args.seed", (
        f"subsampling stage no longer forwards the --seed flag (got {val!r}); "
        f"determinism reproducibility at risk — cli.py:104"
    )


# ===========================================================================
# 4. GUI widget defaults (5) — AST over the Streamlit page source
# ===========================================================================
def test_gui_widget_ld_threshold():
    _src, tree = _load("pages/_ld_tabs/tab_genome_wide.py")
    assert _widget_value_by_target(tree, "ld_threshold_auto") == 0.6, "tab_genome_wide.py:412"


def test_gui_widget_adj_r2_min():
    _src, tree = _load("pages/_ld_tabs/tab_genome_wide.py")
    assert _widget_value_by_target(tree, "adj_r2_min_auto") == 0.2, "tab_genome_wide.py:425"


def test_gui_widget_min_snps_block():
    _src, tree = _load("pages/_ld_tabs/tab_genome_wide.py")
    assert _widget_value_by_target(tree, "min_snps_block_auto") == 3, "tab_genome_wide.py:480"


def test_gui_widget_top_n():
    _src, tree = _load("pages/_ld_tabs/tab_genome_wide.py")
    assert _widget_value_by_target(tree, "top_n") == 10, "tab_genome_wide.py:488"


def test_gui_ld_analysis_adj_r2_min_fallback():
    """The LD page's session fallback for adj_r2_min, `st.session_state.get(
    "adj_r2_min", 0.2)` — the 0.2 must match the detector default, not the
    unused 0.3 signature default."""
    _src, tree = _load("pages/Post_GWAS_Analysis.py")
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and _call_name(node) == "get"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "adj_r2_min"
            and isinstance(node.args[1], ast.Constant)
        ):
            assert node.args[1].value == 0.2, "pages/Post_GWAS_Analysis.py:590"
            return
    raise AssertionError('`session_state.get("adj_r2_min", 0.2)` not found — pages/Post_GWAS_Analysis.py:590')


# ===========================================================================
# 5. call-site literal pinning — the documented-vs-actual adj_r2_min drift
# ===========================================================================
def test_adj_r2_min_call_site_literals():
    """0.3 is reachable only as an unused signature default; every executing
    path supplies 0.2. The Supplementary Methods must document 0.2 (T-62).

    This encodes *why two different numbers are both correct*: the 0.3 lives on
    ``find_ld_blocks_graph`` / ``_adaptive_adj_threshold`` signatures, but
    ``find_ld_clusters_genomewide`` forwards ``float(adj_r2_min)`` from its own
    0.2 default (``gwas/ld.py:974``), so 0.3 is never the value that runs.
    """
    assert _sig_defaults(ld.find_ld_blocks_graph)["adj_r2_min"] == 0.3, "gwas/ld.py:691"
    assert _sig_defaults(ld._adaptive_adj_threshold)["base"] == 0.3, "gwas/ld.py:39"
    assert _sig_defaults(ld.find_ld_clusters_genomewide)["adj_r2_min"] == 0.2, "gwas/ld.py:825"
    assert _sig_defaults(ld.find_ld_blocks_from_genotypes)["adj_r2_min"] == 0.2, "gwas/ld.py:1030"

    src, tree = _load("gwas/ld.py")
    forwarded = _call_kwarg_src(src, tree, "find_ld_blocks_graph", "adj_r2_min",
                                within="find_ld_clusters_genomewide")
    assert forwarded == "float(adj_r2_min)", (
        f"forwarding expression changed to {forwarded!r} — gwas/ld.py:974"
    )

    _gsrc, gtree = _load("pages/_ld_tabs/tab_genome_wide.py")
    assert _widget_value_by_label(gtree, "Adjacent coherence split threshold") == 0.2, (
        "tab_genome_wide.py:425"
    )


def test_cli_hardcoded_min_snps_at_block_detection():
    """The CLI exposes no ``--ld-min-snps`` flag, so this hard-coded literal is
    the only thing standing between a published block table and a re-partition."""
    src, tree = _load("cli.py")
    val = _call_kwarg_src(src, tree, "find_ld_clusters_genomewide", "min_snps")
    assert val == "3", f"CLI block-detection min_snps literal changed to {val!r} — cli.py:1070"
