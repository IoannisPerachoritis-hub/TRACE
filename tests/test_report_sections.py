"""T-79/T-80 — HTML report post-GWAS sections + the no-orphan guard."""
import ast
import pathlib
import re

import pandas as pd

from gwas.reports import generate_gwas_report

_GWAS = pd.DataFrame({"SNP": ["s1", "s2"], "Chr": ["1", "1"], "Pos": [100, 200],
                      "PValue": [1e-8, 1e-3], "Beta_OLS": [0.5, 0.1], "SE_OLS": [0.1, 0.1],
                      "FDR": [1e-6, 0.4]})
_QC = {"n_snps_pass": 2}
_REPO = pathlib.Path(__file__).resolve().parents[1]


def _render(**kw):
    return generate_gwas_report("TestTrait", _QC, _GWAS, **kw)


def test_significant_snp_section_renders():
    sig = pd.DataFrame({"SNP": ["s1"], "Chr": ["1"], "Pos": [100], "PValue": [1e-8],
                        "Block_Status": ["unblocked_not_seeded"]})
    html = _render(significant_snps_df=sig)
    assert "Significant SNPs" in html
    assert "unblocked_not_seeded" in html


def test_haplotype_section_renders_the_previously_orphaned_var():
    """This is the test that fails on the pre-fix template: haplotype_gwas_df was
    computed and passed but never rendered (defect 0.1)."""
    hap = pd.DataFrame({"Chr": ["2"], "Start": [1000], "End": [2000], "PValue": [0.001],
                        "eta2": [0.35]})
    html = _render(haplotype_gwas_df=hap)
    assert "Haplotype block GWAS" in html
    assert "0.35" in html


def test_snp_boxplot_section_renders():
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    from gwas.snpplots import render_snp_boxplot
    fig = render_snp_boxplot("s0", np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 2], dtype=float),
                             np.arange(10.0), beta_mlm=0.4, se_mlm=0.1, beta_ols=0.5)
    html = _render(snp_boxplots=[("s0", fig, "s0 · in_block · concordant")])
    assert "Per-SNP effect plots" in html
    assert "data:image/png;base64," in html
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_isolated_section_renders():
    iso = pd.DataFrame({"Chr": ["2"], "Start (bp)": [100], "End (bp)": [200],
                        "interval_bp": [100], "n_typed_markers_interior": [0]})
    html = _render(isolated_intervals_df=iso)
    assert "Significant SNPs with no LD block" in html


def test_absent_sections_do_not_render():
    html = _render()  # no post-GWAS / sig / isolated frames
    # check rendered <h2> headers (not the harmless section comments)
    for header in ("<h2>Significant SNPs</h2>", "<h2>Significant SNPs with no LD block</h2>",
                   "<h2>Post-GWAS: LD Blocks", "<h2>Per-model Post-GWAS</h2>",
                   "<h2>Multi-model Summaries</h2>"):
        assert header not in html, f"section {header!r} rendered without its data"
    # the pre-change report body is intact
    assert "Top Hits" in html and "Run Metadata" in html


def test_every_render_kwarg_is_referenced_in_template():
    """Permanent guard against re-introducing orphaned template variables
    (generalises the fix for defect 0.1). Every kwarg passed to template.render
    must appear in report.html.j2."""
    src = (_REPO / "gwas" / "reports.py").read_text(encoding="utf-8")
    tmpl = (_REPO / "gwas" / "templates" / "report.html.j2").read_text(encoding="utf-8")
    tree = ast.parse(src)
    render_kwargs = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "render"):
            render_kwargs = [kw.arg for kw in node.keywords if kw.arg]
            break
    assert render_kwargs, "template.render call not found"
    orphans = [k for k in render_kwargs if not re.search(rf"\b{re.escape(k)}\b", tmpl)]
    assert not orphans, f"template.render kwargs never referenced in report.html.j2: {orphans}"


def test_cli_passes_post_gwas_frames_to_report():
    """T-80 wiring guard: the CLI's generate_gwas_report call must forward the
    post-GWAS + significant-SNP frames (defect 0.2 was that it omitted them)."""
    src = (_REPO / "cli.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    call_kwargs = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "generate_gwas_report"):
            call_kwargs = {kw.arg for kw in node.keywords if kw.arg}
            break
    assert call_kwargs is not None, "generate_gwas_report call not found in cli.py"
    required = {"ld_blocks_df", "ld_blocks_annotated_df", "haplotype_gwas_df",
                "per_model_post_gwas", "significant_snps_df", "unblocked_snps_df",
                "isolated_intervals_df", "sig_label", "n_significant_override"}
    missing = required - call_kwargs
    assert not missing, f"cli.py generate_gwas_report omits post-GWAS frames: {missing}"
