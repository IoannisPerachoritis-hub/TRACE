"""utils.pub_theme.show_plot: three on-screen sizes, display only (reviewer point 2-INT.8).

"wide" fills the container; "medium" and "small" are 700 or 450 px. A static figure
(matplotlib Figure or PNG bytes) takes that width directly, with no box around it: a
stretched image in a fixed-width box could be drawn before the browser measured the box,
tiny, after a switch from the interactive view or in a collapsed expander. A medium or
small Plotly figure sits in a fixed-width box (Streamlit 1.50's plotly_chart has no pixel
width), keyed after the figure. show_plot never changes the object, so exported files are
the same whether or not it was shown. Run through AppTest for every type and size, and
with streamlit mocked for the widths and the nesting.
"""
import io
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

from utils.pub_theme import PLOT_WIDTHS, show_plot  # noqa: E402


def _app(kind, size, root):
    import sys

    if root not in sys.path:
        sys.path.insert(0, root)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import streamlit as st

    from utils.pub_theme import fig_to_png_bytes, show_plot

    if kind == "plotly":
        obj = go.Figure(go.Scatter(x=[1, 2, 3], y=[3, 1, 2]))
        before = obj.to_json()
    else:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([1, 2, 3], [3, 1, 2])
        obj = fig if kind == "mpl" else fig_to_png_bytes(fig, dpi=100)
        before = (fig_to_png_bytes(fig, dpi=600), tuple(fig.get_size_inches()), fig.dpi)
    show_plot(obj, size, key="k" if kind == "plotly" else None)
    if kind == "plotly":
        after = obj.to_json()
        st.session_state["unchanged"] = after == before and obj.layout.width is None
    else:
        after = (fig_to_png_bytes(fig, dpi=600), tuple(fig.get_size_inches()), fig.dpi)
        st.session_state["unchanged"] = after == before


_TYPES = {"mpl": "imgs", "png": "imgs", "plotly": "plotly_chart"}


@pytest.mark.parametrize("size", ["wide", "medium", "small"])
@pytest.mark.parametrize("kind", ["mpl", "png", "plotly"])
def test_each_type_in_each_size(kind, size):
    at = AppTest.from_function(_app, args=(kind, size, str(_ROOT)), default_timeout=60)
    at.run()
    assert not at.exception and not at.warning
    assert at.session_state["unchanged"]        # the object (and its export) is unchanged
    main = at.main
    if kind != "plotly" or size == "wide":
        # a static figure sits in the page itself (its width is its own: see the mocked
        # test below, as AppTest does not expose an image's width); wide fills the page
        assert [c.type for c in main.children.values()] == [_TYPES[kind]]
    else:
        (box,) = main.children.values()          # one fixed-width box holding the chart
        assert box.proto.width_config.pixel_width == PLOT_WIDTHS[size]
        assert box.proto.id.endswith("-k_box")   # keyed after the figure's key
        assert [c.type for c in box.children.values()] == [_TYPES[kind]]
    if kind == "plotly":
        el = next(iter(main.children.values()))
        el = el if size == "wide" else next(iter(el.children.values()))
        assert el.proto.use_container_width


def test_sizes():
    assert PLOT_WIDTHS == {"wide": "stretch", "medium": 700, "small": 450}


class _FakeSt:
    """Records each streamlit call show_plot makes: (name, kwargs, depth in containers)."""

    def __init__(self):
        self.calls, self.depth = [], 0

    def container(self, **kw):
        self.calls.append(("container", kw, self.depth))
        fake = self

        class _Box:
            def __enter__(self):
                fake.depth += 1

            def __exit__(self, *exc):
                fake.depth -= 1

        return _Box()

    def _record(self, name, kw):
        self.calls.append((name, kw, self.depth))

    def image(self, data, **kw):
        self._record("image", kw)

    def pyplot(self, fig, **kw):
        self._record("pyplot", kw)

    def plotly_chart(self, fig, **kw):
        self._record("plotly_chart", kw)

    def download_button(self, *args, **kw):
        self._record("download_button", {"key": kw.get("key")})


def _shown(monkeypatch, kind, size, key):
    """What show_plot asks streamlit for, for one figure of this kind (with its data)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    import utils.pub_theme as pt

    fake = _FakeSt()
    monkeypatch.setattr(pt, "st", fake)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([1, 2, 3], [3, 1, 2])
    try:
        obj = {"mpl": fig, "png": pt.fig_to_png_bytes(fig, dpi=50),
               "plotly": go.Figure(go.Scatter(x=[1, 2], y=[2, 1]))}[kind]
        pt.show_plot(obj, size, key=key, data=pd.DataFrame({"x": [1, 2]}), data_name="d")
    finally:
        plt.close(fig)
    return fake.calls


@pytest.mark.parametrize("size", ["wide", "medium", "small"])
@pytest.mark.parametrize("kind", ["mpl", "png"])
def test_a_static_figure_takes_its_width_with_no_box(monkeypatch, kind, size):
    calls = _shown(monkeypatch, kind, size, None)
    name = "image" if kind == "png" else "pyplot"
    # 450 / 700 px, or "stretch" when wide; its data button right under it, in the page
    assert calls == [(name, {"width": PLOT_WIDTHS[size]}, 0),
                     ("download_button", {"key": "dl_plot_data_d"}, 0)]
    assert isinstance(PLOT_WIDTHS[size], int) == (size != "wide")


@pytest.mark.parametrize("size", ["wide", "medium", "small"])
def test_a_plotly_figure_sits_in_a_keyed_box(monkeypatch, size):
    calls = _shown(monkeypatch, "plotly", size, "k")
    chart = ("plotly_chart", {"use_container_width": True, "key": "k"})
    if size == "wide":                          # fills the container: no box
        assert calls == [chart + (0,), ("download_button", {"key": "dl_plot_data_d"}, 0)]
    else:
        assert calls == [("container", {"width": PLOT_WIDTHS[size], "key": "k_box"}, 0),
                         chart + (1,), ("download_button", {"key": "dl_plot_data_d"}, 1)]


def test_an_unkeyed_plotly_figure_keeps_an_unkeyed_box(monkeypatch):
    # the GWAS page's PCA and eigenvalue plots: no key, and no style switch to remount
    calls = _shown(monkeypatch, "plotly", "small", None)
    assert calls[0] == ("container", {"width": PLOT_WIDTHS["small"], "key": None}, 0)
    assert calls[1] == ("plotly_chart", {"use_container_width": True, "key": None}, 1)


def test_bad_input_is_refused_before_anything_is_shown():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    try:
        with pytest.raises(ValueError, match="size"):
            show_plot(fig, "large")
        with pytest.raises(ValueError, match="key"):
            show_plot(fig, "small", key="x")
    finally:
        plt.close(fig)
    for bad in (b"<html>", "plot.png", None, io.BytesIO()):
        with pytest.raises(TypeError):
            show_plot(bad, "wide")
