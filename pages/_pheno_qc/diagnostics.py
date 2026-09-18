"""Phase 3: per-trait diagnostics (histogram, Q-Q) + transform before/after."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from utils.pub_theme import FIGSIZE, PALETTE, export_matplotlib, export_plotly

from . import PhenoQCContext
from .stats import (
    TRANSFORM_INT,
    TRANSFORM_LABELS,
    TRANSFORM_LOG10,
    TRANSFORM_NONE,
    apply_transform,
    run_normality_tests,
)


def _finite(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    return x[np.isfinite(x)]


def _hist_fig(vals, title):
    """Interactive density histogram with a fitted-normal overlay."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=vals, histnorm="probability density", name="data",
        marker_color=PALETTE["blue"], opacity=0.75,
    ))
    mu, sd = float(np.mean(vals)), float(np.std(vals))
    if sd > 0:
        xs = np.linspace(float(vals.min()), float(vals.max()), 200)
        fig.add_trace(go.Scatter(
            x=xs, y=stats.norm.pdf(xs, mu, sd), mode="lines",
            name="normal fit", line=dict(color=PALETTE["red"]),
        ))
    fig.update_layout(
        title=title, xaxis_title="value", yaxis_title="density",
        bargap=0.02, showlegend=True,
    )
    return fig


def _qq_fig(vals, title):
    """Normal Q-Q plot via scipy.stats.probplot on a themed matplotlib axis."""
    fig, ax = plt.subplots(figsize=FIGSIZE["qq"])
    stats.probplot(vals, dist="norm", plot=ax)
    ax.set_title(title)
    # probplot colours points/line with defaults; nudge toward the house palette.
    if ax.lines:
        ax.lines[0].set_color(PALETTE["blue"])
        ax.lines[0].set_markerfacecolor(PALETTE["blue"])
        if len(ax.lines) > 1:
            ax.lines[1].set_color(PALETTE["red"])
    return fig


def _metrics_block(vals, label):
    """Render a compact normality-metrics block for a vector."""
    r = run_normality_tests(vals)
    st.markdown(f"**{label}**, verdict: `{r['verdict']}`")
    m1, m2, m3 = st.columns(3)
    sp = r["shapiro_p"]
    m1.metric("Shapiro p", "n/a" if not np.isfinite(sp) else f"{sp:.2e}")
    m2.metric("Skew", "n/a" if not np.isfinite(r["skew"]) else f"{r['skew']:.2f}")
    m3.metric("Excess kurtosis",
              "n/a" if not np.isfinite(r["kurtosis"]) else f"{r['kurtosis']:.2f}")
    if r["note"]:
        st.caption(r["note"])


def _panel(vals, trait, tag):
    """One before/after column: histogram + Q-Q + metrics for a vector."""
    finite = _finite(vals)
    if finite.size < 3:
        st.warning(f"{tag}: fewer than 3 finite values; cannot plot.")
        return
    hist = _hist_fig(finite, f"{trait}: {tag}")
    st.plotly_chart(hist, use_container_width=True)
    export_plotly(hist, f"hist_{trait}_{tag}".replace(" ", "_"),
                  label_prefix=f"Download {tag} histogram")
    qq = _qq_fig(finite, f"{trait}: {tag}")
    st.pyplot(qq)
    export_matplotlib(qq, f"qq_{trait}_{tag}".replace(" ", "_"),
                      label_prefix=f"Download {tag} Q-Q")
    _metrics_block(finite, tag)


def render(ctx: PhenoQCContext, *, embedded=False, key_prefix="pheno_qc"):
    """Per-trait raw-vs-transformed diagnostics (histogram + Q-Q + normality
    metrics on each side). ``embedded=True`` drops the header and the per-trait
    CSV download (the standalone page provides it), states that the choice feeds
    the GWAS run, and RETURNS ``(trait, method, transformed, transform_ok)`` so the
    caller can apply it. ``key_prefix`` isolates the two selectbox keys from the
    standalone page's session state."""
    if not embedded:
        st.header("3 · Per-trait diagnostics & transforms")

    trait = st.selectbox("Trait", ctx.numeric_cols, key=f"{key_prefix}_trait")
    raw = ctx.pheno_df[trait].to_numpy(dtype=float)

    method = st.selectbox(
        "Transform", TRANSFORM_LABELS,
        index=TRANSFORM_LABELS.index(TRANSFORM_INT),
        key=f"{key_prefix}_transform",
        help=("Pick a method to see its effect on this trait; the right panel "
              "re-renders on every change. Log10 is refused (not shifted) when the "
              "trait has non-positive values."),
    )

    try:
        transformed = apply_transform(raw, method)
        transform_ok = True
    except ValueError as err:
        st.warning(f"Transform not applicable: {err}; showing raw values instead.")
        transformed = raw
        transform_ok = False

    col_raw, col_tr = st.columns(2)
    with col_raw:
        _panel(raw, trait, "raw")
    with col_tr:
        label = (f"transformed ({method})"
                 if transform_ok and method != TRANSFORM_NONE else "raw (no transform)")
        _panel(transformed, trait, label)

    if transform_ok and method == TRANSFORM_LOG10:
        st.caption("log10 scale; multiply effect sizes by 3.32 for per-doubling units.")

    if embedded:
        if transform_ok and method != TRANSFORM_NONE:
            st.info(f"This transform will be applied to **{trait}** for this GWAS run.")
        return trait, method, transformed, transform_ok

    # ── Standalone: per-trait download (preview only) ──────────
    st.divider()
    out = pd.DataFrame(
        {trait: raw, f"{trait}__{method}": transformed}, index=ctx.pheno_df.index
    )
    out.index.name = ctx.pheno_df.index.name or "ID"
    st.download_button(
        label=f"📥 Download '{trait}' (raw + transformed) CSV",
        data=out.to_csv().encode("utf-8"),
        file_name=f"phenotype_{trait}_transformed.csv".replace(" ", "_"),
        mime="text/csv",
        key=f"dl_{key_prefix}_trait",
    )
    return trait, method, transformed, transform_ok
