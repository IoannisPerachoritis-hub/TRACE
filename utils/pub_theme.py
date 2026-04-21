"""Plot theme for TRACE.

Colorblind-safe palette (Wong 2011), shared matplotlib rcParams,
Plotly template, and export helpers (PNG/SVG/HTML).
"""

from io import BytesIO

import matplotlib as mpl
import plotly.graph_objects as go
import plotly.io as pio
try:
    import streamlit as st
except ImportError:
    st = None

# ── Colorblind-safe palette (Wong 2011) ──────────────────────
PALETTE = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "cyan":   "#56B4E9",
    "yellow": "#F0E442",
    "black":  "#000000",
    "grey":   "#999999",
}

PALETTE_CYCLE = [
    PALETTE["blue"], PALETTE["orange"], PALETTE["green"],
    PALETTE["red"], PALETTE["purple"], PALETTE["cyan"],
]

# Domain-specific color assignments
MANHATTAN_COLORS = [PALETTE["blue"], PALETTE["cyan"]]
SIG_LINE_COLOR = PALETTE["red"]

# Colormaps
LD_HEATMAP_CMAP = "YlOrBr"           # r² heatmaps (genetics convention)
PERFORMANCE_CMAP = "RdBu_r"          # bidirectional R² [-1, 1]
ENRICHMENT_CMAP = "YlOrRd"           # unidirectional -log10(FDR)

# ── Standard figure sizes (width, height) in inches ──────────
FIGSIZE = {
    "manhattan":      (10, 4.5),
    "qq":             (4.5, 4.5),
    "heatmap":        (7, 6),
    "heatmap_small":  (6, 5),
    "bar_horizontal": (7, None),       # height = max(3, n_items * 0.4)
    "histogram":      (6, 3.5),
    "boxplot":        (8, 5),
    "forest":         (7, None),       # height = max(3, n_items * 0.7)
    "scatter":        (6, 5),
    "line":           (8, 4),
}


# ── Matplotlib theme ─────────────────────────────────────────

_APPLIED = False


def apply_matplotlib_theme():
    """Apply publication-quality rcParams. Safe to call multiple times."""
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    params = {
        # Font
        "font.family":         "sans-serif",
        "font.sans-serif":     ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":           11,

        # Axes
        "axes.titlesize":      13,
        "axes.titleweight":    "bold",
        "axes.labelsize":      12,
        "axes.labelweight":    "normal",
        "axes.linewidth":      0.8,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.prop_cycle":     mpl.cycler(color=PALETTE_CYCLE),

        # Ticks
        "xtick.labelsize":     10,
        "ytick.labelsize":     10,
        "xtick.major.width":   0.8,
        "ytick.major.width":   0.8,
        "xtick.direction":     "out",
        "ytick.direction":     "out",

        # Legend
        "legend.fontsize":     9,
        "legend.frameon":      False,

        # Figure
        "figure.dpi":          150,
        "savefig.dpi":         600,
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.1,

        # Lines & scatter
        "lines.linewidth":     1.5,
        "scatter.edgecolors":  "none",
    }
    mpl.rcParams.update(params)


# ── Plotly template ──────────────────────────────────────────

_PLOTLY_REGISTERED = False
PLOTLY_TEMPLATE = "solanaceae_pub"


def build_plotly_template():
    """Build and register the publication-quality Plotly template."""
    global _PLOTLY_REGISTERED
    if _PLOTLY_REGISTERED:
        return
    _PLOTLY_REGISTERED = True

    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family="Arial, Helvetica, sans-serif", size=12, color="#333333"),
        title=dict(font=dict(size=14, color="#111111")),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            showgrid=False,
            linecolor="#333333",
            linewidth=0.8,
            ticks="outside",
            tickfont=dict(size=10),
            title_font=dict(size=12),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#E5E5E5",
            gridwidth=0.5,
            linecolor="#333333",
            linewidth=0.8,
            ticks="outside",
            tickfont=dict(size=10),
            title_font=dict(size=12),
        ),
        colorway=PALETTE_CYCLE,
        legend=dict(
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
            borderwidth=0,
        ),
        margin=dict(l=60, r=40, t=60, b=50),
    )
    pio.templates[PLOTLY_TEMPLATE] = template
    pio.templates.default = PLOTLY_TEMPLATE


# ── Export helpers ────────────────────────────────────────────

def export_matplotlib(fig, filename_stem, label_prefix="Download"):
    """Offer PNG + SVG + PDF download buttons for a matplotlib figure."""
    col_png, col_svg, col_pdf = st.columns(3)

    buf_png = BytesIO()
    fig.savefig(buf_png, format="png", dpi=600, bbox_inches="tight")
    col_png.download_button(
        f"📥 {label_prefix} PNG",
        buf_png.getvalue(),
        file_name=f"{filename_stem}.png",
        mime="image/png",
        key=f"dl_png_{filename_stem}",
    )

    buf_svg = BytesIO()
    fig.savefig(buf_svg, format="svg", bbox_inches="tight")
    col_svg.download_button(
        f"📥 {label_prefix} SVG",
        buf_svg.getvalue(),
        file_name=f"{filename_stem}.svg",
        mime="image/svg+xml",
        key=f"dl_svg_{filename_stem}",
    )

    buf_pdf = BytesIO()
    fig.savefig(buf_pdf, format="pdf", bbox_inches="tight")
    col_pdf.download_button(
        f"📥 {label_prefix} PDF",
        buf_pdf.getvalue(),
        file_name=f"{filename_stem}.pdf",
        mime="application/pdf",
        key=f"dl_pdf_{filename_stem}",
    )


def export_plotly(fig, filename_stem, label_prefix="Download"):
    """Offer HTML + SVG download buttons for a Plotly figure."""
    html_bytes = fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")

    try:
        svg_bytes = fig.to_image(format="svg")
        col_html, col_svg = st.columns(2)
        col_html.download_button(
            f"📥 {label_prefix} HTML",
            html_bytes,
            file_name=f"{filename_stem}.html",
            mime="text/html",
            key=f"dl_html_{filename_stem}",
        )
        col_svg.download_button(
            f"📥 {label_prefix} SVG",
            svg_bytes,
            file_name=f"{filename_stem}.svg",
            mime="image/svg+xml",
            key=f"dl_svg_{filename_stem}",
        )
    except Exception:
        st.download_button(
            f"📥 {label_prefix} HTML",
            html_bytes,
            file_name=f"{filename_stem}.html",
            mime="text/html",
            key=f"dl_html_{filename_stem}",
        )


def fig_to_png_bytes(fig, dpi=600):
    """Convert a matplotlib figure to PNG bytes (for ZIP builders)."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()
