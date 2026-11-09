"""utils.pub_theme.show_plot: three on-screen sizes, display only (reviewer point 2-INT.8).

"wide" puts the plot straight in the page (container width); "medium" and "small" put it
in a fixed-width box of 700 or 450 px. It takes a matplotlib Figure, PNG bytes or a Plotly
figure and never changes the object, so exported files are the same whether or not it was
shown. Run through AppTest for every type and size.
"""
import io
import sys
from pathlib import Path

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
    if size == "wide":
        assert [c.type for c in main.children.values()] == [_TYPES[kind]]
    else:
        (box,) = main.children.values()          # one fixed-width box holding the plot
        assert box.proto.width_config.pixel_width == PLOT_WIDTHS[size]
        assert [c.type for c in box.children.values()] == [_TYPES[kind]]
    if kind == "plotly":
        el = next(iter(main.children.values()))
        el = el if size == "wide" else next(iter(el.children.values()))
        assert el.proto.use_container_width


def test_sizes():
    assert PLOT_WIDTHS == {"wide": "stretch", "medium": 700, "small": 450}


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
