"""Every on-screen plot goes through utils.pub_theme.show_plot (reviewer point 2-INT.8).

A static scan of the app's sources: no st.pyplot, st.image or st.plotly_chart outside
show_plot itself (on any receiver: st, st.sidebar, a column, an alias, a getattr or a
`from streamlit import`). The dead pages/_ld_tabs/tab_block_heatmaps.py is left alone.
Each show_plot call in pages/ has a literal size class, pinned below, so a plot cannot
change class unnoticed. And each offers the data it draws: it passes ``data=`` (a
"Download data (CSV)" button under it), or its data already has a download, named in
_DATA_ELSEWHERE with that button's label, which must appear in the same file.
"""
import ast
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_APIS = {"pyplot", "image", "plotly_chart"}
_ALLOWED = {"pages/_ld_tabs/tab_block_heatmaps.py"}   # dead code, out of scope


def _bypasses(source, name="<probe>"):
    """(line, what) for every display call that does not go through show_plot."""
    tree = ast.parse(source, filename=name)
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "show_plot":
            skip.update(range(node.lineno, node.end_lineno + 1))
    hits = []
    for node in ast.walk(tree):
        if getattr(node, "lineno", None) in skip:
            continue
        if isinstance(node, ast.Attribute) and node.attr in _APIS:
            base = node.value
            if isinstance(base, ast.Name) and base.id in ("matplotlib", "mpl"):
                continue            # import matplotlib.pyplot
            hits.append((node.lineno, f".{node.attr}"))
        elif isinstance(node, ast.ImportFrom) and node.module == "streamlit":
            hits += [(node.lineno, f"import {a.name}") for a in node.names if a.name in _APIS]
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == "getattr" and len(node.args) >= 2
              and isinstance(node.args[1], ast.Constant) and node.args[1].value in _APIS):
            hits.append((node.lineno, f"getattr {node.args[1].value}"))
    return hits


def _sources():
    files = [*sorted((_ROOT / "pages").rglob("*.py")), *sorted((_ROOT / "utils").rglob("*.py")),
             *sorted((_ROOT / "gwas").rglob("*.py")), _ROOT / "app.py"]
    for f in files:
        rel = f.relative_to(_ROOT).as_posix()
        if rel not in _ALLOWED:
            yield rel, f.read_text(encoding="utf-8")


def test_no_plot_bypasses_show_plot():
    probe = '''
import streamlit as st
import matplotlib.pyplot as plt
from streamlit import image
s = st
s.pyplot(fig)
col.plotly_chart(fig)
st.sidebar.image(b"")
getattr(st, "pyplot")(fig)
show = st.image
plt.plot([1]); fig.to_image(); matplotlib.pyplot.close()
def show_plot(obj, size):
    st.pyplot(obj); st.image(obj); st.plotly_chart(obj)
'''
    assert len(_bypasses(probe)) == 6       # the probe's own six bypasses, nothing else
    found = {rel: _bypasses(src, rel) for rel, src in _sources()}
    found = {rel: hits for rel, hits in found.items() if hits}
    assert found == {}, f"plots shown without show_plot: {found}"


def test_show_plot_is_the_renderer():
    src = (_ROOT / "utils" / "pub_theme.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "show_plot")
    used = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name) and n.value.id == "st"}
    assert used == {"pyplot", "image", "plotly_chart", "container", "download_button"}


# (file, plotted expression, size class, has a key), one entry per call
_PINNED = Counter({
    ("pages/GWAS_analysis.py", "_obj", "small", False): 1,          # restored QQ plots
    ("pages/GWAS_analysis.py", "_obj", "wide", False): 1,           # restored Manhattan, scree
    ("pages/GWAS_analysis.py", "_fig_maf", "small", False): 1,
    ("pages/GWAS_analysis.py", "_qc_fig", "wide", False): 1,        # the QC report strip
    ("pages/GWAS_analysis.py", "_figd", "small", False): 1,         # PC eigenvalue spectrum
    ("pages/GWAS_analysis.py", "fig_pc", "medium", False): 1,       # PCA
    ("pages/GWAS_analysis.py", "_fig_scree", "small", False): 1,
    ("pages/GWAS_analysis.py", "fig_boot_hist", "medium", False): 1,
    ("pages/GWAS_analysis.py", "fig_boot_manh", "wide", False): 1,
    ("pages/GWAS_analysis.py", "fig_int", "wide", False): 1,        # MLM Manhattan
    ("pages/GWAS_analysis.py", "fig_qq", "small", False): 1,
    ("pages/GWAS_analysis.py", "fig_hist", "medium", False): 1,     # p-value histogram
    ("pages/GWAS_analysis.py", "fig_plotly_mlmm", "wide", False): 1,
    ("pages/GWAS_analysis.py", "fig_qq_mlmm", "small", False): 1,
    ("pages/GWAS_analysis.py", "fig_plotly_fc", "wide", False): 1,
    ("pages/GWAS_analysis.py", "fig_qq_fc", "small", False): 1,
    ("pages/GWAS_analysis.py", "fig_overlay", "wide", False): 1,
    ("pages/_ld_tabs/tab_genome_wide.py", "cached['display']", "medium", False): 2,
    ("pages/_ld_tabs/tab_genome_wide.py", "_build_interactive_pca_fig()", "medium", True): 1,
    ("pages/_ld_tabs/tab_genome_wide.py", "_fig", "small", False): 1,   # lead-SNP box plot
    ("pages/_ld_tabs/tab_genome_wide.py", "_pfig", "small", True): 1,   # ... interactive
    ("pages/_ld_tabs/tab_genome_wide.py", "_build_interactive_raincloud_fig()", "medium", True): 1,
    ("pages/_ld_tabs/tab_genome_wide.py", "_fig_forest", "small", False): 1,
    ("pages/_ld_tabs/tab_local_ld.py", "local_cached['display']", "medium", False): 1,
    ("pages/_ld_tabs/tab_regional.py", "fig_i", "wide", True): 1,
    ("pages/_ld_tabs/tab_regional.py", "buf.getvalue()", "wide", False): 1,
    ("pages/_pheno_qc/diagnostics.py", "hist", "small", False): 1,
    ("pages/_pheno_qc/diagnostics.py", "qq", "small", False): 1,
})


def test_every_plot_has_its_size_class():
    calls, keys = Counter(), []
    for rel, src in _sources():
        if not rel.startswith("pages/"):
            continue
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "show_plot"):
                size = (node.args[1] if len(node.args) > 1 else
                        next(k.value for k in node.keywords if k.arg == "size"))
                assert isinstance(size, ast.Constant), (rel, node.lineno)
                key = next((k.value for k in node.keywords if k.arg == "key"), None)
                calls[(rel, ast.unparse(node.args[0]), size.value, key is not None)] += 1
                if rel.endswith("tab_regional.py") and key is not None:
                    keys.append(ast.literal_eval(key))
    assert calls == _PINNED
    assert sum(calls.values()) == 29
    assert keys == ["regional_chart"]       # a constant key: a varying one forced reruns


# figures whose data already has a download: (file, plotted expression) -> its label
_DATA_ELSEWHERE = {
    ("pages/GWAS_analysis.py", "_obj"): "Download GWAS results (ZIP)",   # restored images
    ("pages/GWAS_analysis.py", "fig_pc"): "Download PCA scores (CSV)",
    ("pages/GWAS_analysis.py", "fig_boot_manh"): "Download per-SNP subsampling results (CSV)",
    ("pages/GWAS_analysis.py", "fig_int"): "Download GWAS results (ZIP)",      # genome-wide
    ("pages/GWAS_analysis.py", "fig_qq"): "Download GWAS results (ZIP)",
    ("pages/GWAS_analysis.py", "fig_plotly_mlmm"): "Download GWAS results (ZIP)",
    ("pages/GWAS_analysis.py", "fig_qq_mlmm"): "Download GWAS results (ZIP)",
    ("pages/GWAS_analysis.py", "fig_plotly_fc"): "Download GWAS results (ZIP)",
    ("pages/GWAS_analysis.py", "fig_qq_fc"): "Download GWAS results (ZIP)",
    ("pages/GWAS_analysis.py", "fig_overlay"): "Download cross-model results (CSV)",
    # the haplotype box/raincloud: the block's assignments, under the plot
    ("pages/_ld_tabs/tab_genome_wide.py", "cached['display']"):
        "Download sample-to-haplotype assignments (CSV)",
    ("pages/_ld_tabs/tab_genome_wide.py", "_build_interactive_raincloud_fig()"):
        "Download sample-to-haplotype assignments (CSV)",
    ("pages/_ld_tabs/tab_genome_wide.py", "_fig"): "Download plotted data (CSV)",   # lead SNP
    ("pages/_ld_tabs/tab_genome_wide.py", "_pfig"): "Download plotted data (CSV)",
    ("pages/_ld_tabs/tab_genome_wide.py", "_fig_forest"): "Download haplotype effects (CSV)",
    ("pages/_ld_tabs/tab_local_ld.py", "local_cached['display']"): "Download r² (long",
    ("pages/_ld_tabs/tab_regional.py", "fig_i"): "Download regional data (CSV)",
    ("pages/_ld_tabs/tab_regional.py", "buf.getvalue()"): "Download regional data (CSV)",
}


def test_every_plot_offers_its_data():
    missing, sources = [], dict(_sources())
    for rel, src in sources.items():
        if not rel.startswith("pages/"):
            continue
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "show_plot"):
                kw = {k.arg for k in node.keywords}
                if "data" in kw:
                    assert "data_name" in kw, (rel, node.lineno)
                    continue
                label = _DATA_ELSEWHERE.get((rel, ast.unparse(node.args[0])))
                if label is None:
                    missing.append((rel, node.lineno, ast.unparse(node.args[0])))
                else:
                    assert label in src, (rel, label)      # the download it points to exists
    assert missing == [], f"figures without their data: {missing}"
