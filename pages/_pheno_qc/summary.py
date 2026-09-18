"""Phase 2: all-traits normality summary table."""

import streamlit as st

from . import PhenoQCContext
from .stats import (
    NORMALITY_ALPHA,
    TRANSFORM_INT,
    TRANSFORM_LABELS,
    normality_summary_table,
    recommend_uniform_method,
    transform_whole_table,
)


def _highlight_verdict(row):
    """Row styler: tint non-normal / degenerate traits."""
    verdict = str(row.get("Verdict", ""))
    if verdict == "Normal":
        color = "background-color: rgba(0, 158, 115, 0.15)"      # green
    elif verdict == "Non-normal":
        color = "background-color: rgba(213, 94, 0, 0.15)"       # orange
    else:  # Constant / Insufficient
        color = "background-color: rgba(153, 153, 153, 0.20)"    # grey
    return [color] * len(row)


def render(ctx: PhenoQCContext):
    st.header("2 · Normality summary")

    table = normality_summary_table(ctx.pheno_df, ctx.numeric_cols)

    n_nonnormal = int((table["Verdict"] == "Non-normal").sum())
    n_normal = int((table["Verdict"] == "Normal").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Traits tested", len(table))
    c2.metric("Normal", n_normal)
    c3.metric("Non-normal", n_nonnormal)

    styled = table.style.apply(_highlight_verdict, axis=1).format(
        {
            "Shapiro W": "{:.3f}",
            "Shapiro p": "{:.2e}",
            "D'Agostino K² p": "{:.2e}",
            "Skew": "{:.2f}",
            "Excess kurtosis": "{:.2f}",
        },
        na_rep="n/a",
    )
    st.dataframe(styled, use_container_width=True)

    st.download_button(
        label="📥 Download normality summary (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="phenotype_normality_summary.csv",
        mime="text/csv",
        key="dl_pheno_normality_summary",
    )

    # ── Transform ALL traits with ONE method & download ─────────
    st.divider()
    st.subheader("Transform all traits & download")
    st.markdown(
        "Apply **one** transform to **every** trait and download the result as a "
        "GWAS-ready phenotype CSV. For a metabolite panel this is the correct "
        "approach: a single uniform method keeps effect sizes and mQTL signals "
        "comparable across metabolites and gives a clean, defensible Methods "
        "statement. **rank-INT** (the default) is the metabolomics-GWAS standard: "
        "it maps each trait onto normal scores and is robust to outliers. Note "
        "that traits with heavy ties (many zeros / below-detection-limit values) "
        "can retain mild residual non-normality even after rank-INT; those may "
        "need presence/absence or two-part handling instead."
    )
    st.info(
        "Feed the downloaded file to the GWAS page with normalization set to "
        "**None**; otherwise the trait would be transformed twice.",
        icon="⚠️",
    )

    method = st.selectbox(
        "Transform to apply to all traits",
        TRANSFORM_LABELS,
        index=TRANSFORM_LABELS.index(TRANSFORM_INT),
        key="pheno_qc_uniform_method",
        help="log10 auto-shifts (pseudocount) so zeros/negatives survive; "
             "Yeo-Johnson fits one λ per trait; rank-INT forces an exact normal.",
    )

    transformed, report = transform_whole_table(ctx.pheno_df, ctx.numeric_cols, method)
    n_applied = int((~report["Applied"].str.contains("skipped")).sum())
    st.caption(f"{method} applied to {n_applied} of {len(report)} trait(s).")
    st.dataframe(report, use_container_width=True)

    st.download_button(
        label="📥 Download transformed phenotype (CSV)",
        data=transformed.to_csv().encode("utf-8"),
        file_name="phenotype_transformed.csv",
        mime="text/csv",
        key="dl_pheno_qc_transformed",
    )

    with st.expander("Compare how each method normalizes the panel"):
        counts = recommend_uniform_method(ctx.pheno_df, ctx.numeric_cols)
        n = len(ctx.numeric_cols)
        st.write(
            {m: f"{c}/{n} traits Normal" for m, c in counts.items()}
        )
