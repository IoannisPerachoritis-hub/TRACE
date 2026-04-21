"""GWAS HTML report generator.

Self-contained HTML with inline CSS and base64-encoded figures.
"""
import base64
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader


_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _fig_to_b64(fig_or_bytes):
    """Convert a matplotlib Figure or raw PNG bytes to base64 string."""
    if fig_or_bytes is None:
        return None
    if isinstance(fig_or_bytes, (bytes, bytearray)):
        return base64.b64encode(fig_or_bytes).decode("ascii")
    # Assume matplotlib Figure
    buf = BytesIO()
    fig_or_bytes.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _df_to_html(df, max_rows=30):
    """Convert DataFrame to HTML table string."""
    if df is None or df.empty:
        return None
    df_display = df.head(max_rows).copy()
    # Format float columns
    for col in df_display.select_dtypes(include=[np.floating]).columns:
        df_display[col] = df_display[col].map(
            lambda x: f"{x:.4g}" if pd.notna(x) else ""
        )
    return df_display.to_html(index=False, classes="", border=0)


def generate_gwas_report(
    trait_col,
    qc_snp,
    gwas_df,
    figures=None,
    metadata=None,
    mlmm_df=None,
    farmcpu_df=None,
    ld_blocks_df=None,
    lambda_gc=None,
    n_samples=None,
    n_snps=None,
    info_field=None,
    pc_selection_df=None,
    ld_blocks_annotated_df=None,
    haplotype_gwas_df=None,
    per_model_post_gwas=None,
    sig_label=None,
    n_significant_override=None,
):
    """
    Generate a self-contained HTML report for GWAS results.

    Parameters
    ----------
    trait_col    : str, trait name
    qc_snp       : dict, QC summary counts
    gwas_df      : DataFrame with SNP, PValue, Chr, Pos columns
    figures      : dict of {name: matplotlib Figure or PNG bytes} (optional)
    metadata     : dict of run parameters (optional)
    mlmm_df      : DataFrame, MLMM cofactor table (optional)
    farmcpu_df   : DataFrame, FarmCPU pseudo-QTN table (optional)
    ld_blocks_df : DataFrame, LD block summary (optional)
    lambda_gc    : float (optional)
    n_samples    : int (optional)
    n_snps       : int (optional)
    info_field   : str or None, name of INFO quality field detected

    Returns
    -------
    html : str, complete HTML report
    """
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=True,
    )
    template = env.get_template("report.html.j2")

    # Encode figures
    fig_b64 = {}
    if figures:
        for name, fig in figures.items():
            b64 = _fig_to_b64(fig)
            if b64:
                fig_b64[name] = b64

    # Top hits table
    n_significant = 0
    top_hits_html = None
    if n_significant_override is not None:
        n_significant = n_significant_override
    if gwas_df is not None and not gwas_df.empty:
        if n_significant_override is None:
            # Fallback: compute from DataFrame
            if "FDR" in gwas_df.columns:
                n_significant = int((gwas_df["FDR"] < 0.05).sum())
            elif "Significant_FDR" in gwas_df.columns:
                n_significant = int(gwas_df["Significant_FDR"].sum())

        # Show top 20 by p-value
        top = gwas_df.nsmallest(20, "PValue")
        display_cols = [c for c in ["SNP", "Chr", "Pos", "PValue", "Beta_OLS",
                                     "SE_OLS", "FDR"] if c in top.columns]
        top_hits_html = _df_to_html(top[display_cols])

    # Significance label for report template
    _sig_label = sig_label or "FDR<0.05"

    # Multi-model summaries
    mlmm_summary = _df_to_html(mlmm_df)
    farmcpu_summary = _df_to_html(farmcpu_df)
    ld_blocks_html = _df_to_html(ld_blocks_df)

    # PC selection
    pc_selection_html = _df_to_html(pc_selection_df)

    # Annotated LD blocks and haplotype testing
    ld_blocks_annotated_html = _df_to_html(ld_blocks_annotated_df, max_rows=50)
    haplotype_gwas_html = _df_to_html(haplotype_gwas_df, max_rows=50)

    # Per-model post-GWAS sections
    model_sections = []
    if per_model_post_gwas:
        for model_name, model_data in per_model_post_gwas.items():
            section = {"name": model_name}
            section["ld_annotated_html"] = _df_to_html(
                model_data.get("ld_blocks_annotated_df"), max_rows=50
            )
            section["haplotype_html"] = _df_to_html(
                model_data.get("haplotype_gwas_df"), max_rows=50
            )
            # Include section only if it has at least one non-empty table
            if any(section.get(k) for k in [
                "ld_annotated_html", "haplotype_html",
            ]):
                model_sections.append(section)

    # Metadata JSON
    metadata_json = json.dumps(metadata or {}, indent=2, default=str)

    html = template.render(
        trait_col=trait_col,
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_samples=n_samples or "?",
        n_snps=n_snps or "?",
        lambda_gc=lambda_gc,
        n_significant=n_significant,
        qc_snp=qc_snp or {},
        info_field=info_field,
        figures=fig_b64,
        top_hits_html=top_hits_html,
        pc_selection_html=pc_selection_html,
        mlmm_summary=mlmm_summary,
        farmcpu_summary=farmcpu_summary,
        ld_blocks_html=ld_blocks_html,
        ld_blocks_annotated_html=ld_blocks_annotated_html,
        haplotype_gwas_html=haplotype_gwas_html,
        model_sections=model_sections,
        metadata_json=metadata_json,
        sig_label=_sig_label,
    )

    return html
