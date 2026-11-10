"""Plot theme for TRACE.

Colorblind-safe palette (Wong 2011), shared matplotlib rcParams,
Plotly template, and export helpers (PNG/SVG/HTML).
"""

import re
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

# Haplotype (MLG) colours, shared by the block-level plots and the lead-SNP box plot.
HAPLOTYPE_OTHER_COLOR = "#bdbdbd"
_HAP_LABEL = re.compile(r"^H(\d+)$")


def hap_sort_key(label):
    """Numeric sort for H1, H2, ..., H10, H11; non-H labels sort last."""
    m = _HAP_LABEL.match(str(label))
    return (0, int(m.group(1))) if m else (1, str(label))


def haplotype_color_map(labels):
    """{label: colour} for haplotype labels, the same in every plot.

    H<n> takes PALETTE_CYCLE[(n - 1) % 6] from its own number, so a plot that shows
    H1, H3 and H7 colours H3 as every other plot does; "Other" is always grey
    (HAPLOTYPE_OTHER_COLOR) and always in the map. A label that is not H<n> takes the
    next colours after the H labels, in hap_sort_key order.
    """
    present = sorted({str(h) for h in labels if str(h) != "Other"}, key=hap_sort_key)
    n_h = sum(1 for h in present if _HAP_LABEL.match(h))
    cmap, j = {}, 0
    for h in present:
        m = _HAP_LABEL.match(h)
        if m:
            cmap[h] = PALETTE_CYCLE[(int(m.group(1)) - 1) % len(PALETTE_CYCLE)]
        else:
            cmap[h] = PALETTE_CYCLE[(n_h + j) % len(PALETTE_CYCLE)]
            j += 1
    cmap["Other"] = HAPLOTYPE_OTHER_COLOR
    return cmap

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

# ── On-screen plot sizes ─────────────────────────────────────
# Display only: downloads, ZIPs and reports never read these.
PLOT_WIDTH_MEDIUM = 700
PLOT_WIDTH_SMALL = 450
PLOT_WIDTHS = {"wide": "stretch", "medium": PLOT_WIDTH_MEDIUM, "small": PLOT_WIDTH_SMALL}
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def show_plot(obj, size, *, key=None, data=None, data_name=None):
    """Show a plot on screen in one of three size classes.

    ``obj`` is a matplotlib Figure, PNG bytes or a Plotly figure. ``size`` is "wide"
    (the container width), "medium" (at most PLOT_WIDTH_MEDIUM px) or "small" (at most
    PLOT_WIDTH_SMALL px). ``key`` is for Plotly figures only. The object is not
    modified, so the export helpers write the same files whether or not it was shown.
    A static figure (matplotlib or PNG) takes its width directly, with no box around it.
    A medium or small Plotly figure sits in a fixed-width box, keyed ``<key>_box``.

    ``data`` (a table, or CSV bytes) is the data the figure draws: "Download data (CSV)"
    sits directly under it as ``<data_name>.csv``. ``data_name`` must be unique on the
    page (it keys the button).
    """
    if size not in PLOT_WIDTHS:
        raise ValueError(f"size must be one of {sorted(PLOT_WIDTHS)}, got {size!r}")
    if data is not None and not data_name:
        raise ValueError("show_plot: data needs a data_name")
    from contextlib import nullcontext

    from matplotlib.figure import Figure

    if isinstance(obj, (bytes, bytearray)):
        if bytes(obj[:8]) != _PNG_SIGNATURE:
            raise TypeError("show_plot: bytes must be a PNG image")
        kind = "png"
    elif isinstance(obj, go.Figure):
        kind = "plotly"
    elif isinstance(obj, Figure):
        kind = "mpl"
    else:
        raise TypeError(f"show_plot: unsupported plot type {type(obj).__name__}")
    if key is not None and kind != "plotly":
        raise ValueError("show_plot: key applies to Plotly figures only")
    width = PLOT_WIDTHS[size]
    if kind == "plotly":
        # Streamlit 1.50's plotly_chart has no pixel width: medium and small sit in a
        # fixed-width box, keyed after the figure so that switching a plot between its
        # static and interactive views mounts a new element; wide fills the container
        box = (nullcontext() if width == "stretch" else
               st.container(width=width, key=None if key is None else f"{key}_box"))
    else:
        # a static figure takes its width directly (450 / 700 px, or the container's).
        # Stretched inside a fixed-width box, it could be drawn before the browser had
        # measured the box (after a style switch, or in a collapsed expander): tiny.
        box = nullcontext()
    with box:
        if kind == "plotly":
            st.plotly_chart(obj, use_container_width=True, key=key)
        elif kind == "png":
            st.image(bytes(obj), width=width)
        else:
            st.pyplot(obj, width=width)
        if data is not None:
            from utils.plot_data import csv_bytes

            st.download_button(
                "Download data (CSV)", csv_bytes(data), file_name=f"{data_name}.csv",
                mime="text/csv", key=f"dl_plot_data_{data_name}",
            )


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


def plotly_html_bytes(fig):
    """A Plotly figure as a self-contained HTML file: plotly.js is embedded (not fetched
    from a CDN), so the file opens offline and behind a firewall. Every interactive HTML
    the GUI exports is written by this function."""
    return fig.to_html(full_html=True, include_plotlyjs=True).encode("utf-8")


def export_plotly(fig, filename_stem, label_prefix="Download", html_bytes=None):
    """Offer HTML + SVG download buttons for a Plotly figure. ``html_bytes`` is the
    figure's HTML when the caller already built it (plotly_html_bytes), so it is built
    once and the download equals the ZIP's copy."""
    if html_bytes is None:
        html_bytes = plotly_html_bytes(fig)

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
