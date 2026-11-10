"""Every interactive HTML the GUI exports opens offline: plotly.js is embedded through
one helper (utils.pub_theme.plotly_html_bytes), never fetched from a CDN."""
import ast
import re
from pathlib import Path

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs_version

from utils.pub_theme import plotly_html_bytes

ROOT = Path(__file__).resolve().parent.parent


def test_the_html_embeds_plotly_js_and_loads_no_script():
    fig = go.Figure(go.Scattergl(x=[1, 2, 3], y=[3, 1, 2], mode="markers"))
    html = plotly_html_bytes(fig).decode("utf-8")
    assert f"plotly.js v{get_plotlyjs_version()}" in html          # the bundle itself
    assert not re.search(r"<script[^>]*\bsrc\s*=", html, re.I)     # no external script
    # the bundle's own text names cdn.plot.ly (a map-data default), so only script tags
    # are checked; the CDN form of the same figure does load plotly.js from there
    cdn = fig.to_html(full_html=True, include_plotlyjs="cdn")
    assert re.search(r'<script[^>]*\bsrc\s*=\s*"https://cdn\.plot\.ly/', cdn, re.I)


def test_every_export_uses_the_helper():
    """No Plotly export with include_plotlyjs other than the helper's; the GWAS page's
    four interactive figures (PCA, MLM, MLMM, FarmCPU) and the per-figure download
    (export_plotly) go through plotly_html_bytes."""
    for path in [*ROOT.glob("pages/**/*.py"), *ROOT.glob("utils/*.py"), *ROOT.glob("gwas/*.py"),
                 ROOT / "app.py"]:
        src = path.read_text(encoding="utf-8")
        if path.name == "pub_theme.py":
            assert src.count("include_plotlyjs=") == 1 and "include_plotlyjs=True" in src
        else:
            assert "include_plotlyjs" not in src, path
    page = (ROOT / "pages" / "GWAS_analysis.py").read_text(encoding="utf-8")
    assert page.count("plotly_html_bytes(") == 4
    tree = ast.parse((ROOT / "utils" / "pub_theme.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "export_plotly")
    assert "plotly_html_bytes(fig)" in ast.unparse(fn)
