"""T-21 / T-84–T-89 — per-SNP effect-plot rendering + the plot budget.

Renders the significant-SNP genotype-class boxplots for the HTML report from the
pure summaries in ``gwas/snpview.py``. Uses the Agg backend so it is
headless-testable and never opens a window. Like ``snpview``, this module never
calls the block detector — it only draws what ``genotype_class_summary`` and
``effect_flag`` already computed.

Rendering budget (T-84–T-86): the significant-SNP TABLE is always complete; only
the *plots* are capped. ``select_snps_for_plotting`` chooses which rows get a
figure; ``collapse_snps_for_plotting`` (in ``snpview``) may thin near-redundant
SNPs to a representative for plotting **only**.

Spec: post_gwas_visibility_design.md §2.2-2.4; integration_and_testing.md §5.2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gwas.snpview import effect_flag, genotype_class_summary

SNP_PLOT_MODES = ("none", "capped", "all")


def select_snps_for_plotting(sig_table, *, mode="capped", max_plots=24):
    """Which significant SNPs get a boxplot, ranked by PValue asc (stable).

    ``none`` → nothing; ``capped`` → the top ``max_plots``; ``all`` → every row.
    The significant-SNP table itself is ALWAYS complete regardless of this budget
    (T-86) — this only bounds how many figures the report embeds.
    """
    if mode not in SNP_PLOT_MODES:
        raise ValueError(f"mode must be one of {SNP_PLOT_MODES}, got {mode!r}")
    if sig_table is None or len(sig_table) == 0 or mode == "none":
        return []
    ranked = sig_table.sort_values("PValue", kind="stable")
    ids = list(ranked["SNP"].astype(str))
    return ids if mode == "all" else ids[: int(max_plots)]


def render_snp_boxplot(snp_id, geno_col, y, *, beta_mlm=None, se_mlm=None,
                       beta_ols=None, dpi=110):
    """One genotype-class boxplot for a SNP → a matplotlib ``Figure`` (Agg).

    A box is drawn only for classes with n ≥ 5 (else jittered points only); every
    class 0/1/2 is shown including n=0; the structure-adjusted β_MLM ± SE, the
    raw-vs-adjusted effect flag, and the per-SNP η² are annotated. When β_MLM is
    absent the annotation says "structure-unadjusted (p-values unaffected)" — the
    effect axis is suppressed, never faked (integration §5.2).
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    s = genotype_class_summary(geno_col, y)
    flag = effect_flag(beta_mlm, beta_ols)
    g = np.asarray(geno_col, dtype=float)
    yy = np.asarray(y, dtype=float)
    valid = np.isfinite(g) & np.isfinite(yy)
    rng = np.random.default_rng(0)                 # deterministic jitter

    fig, ax = plt.subplots(figsize=(3.2, 3.0), dpi=dpi)
    xticklabels = []
    for cls in (0, 1, 2):
        m = valid & (np.rint(g) == cls)
        yc = yy[m]
        info = s["classes"][cls]
        # Draw a box for every non-empty genotype class (n>=1). The per-class n in the
        # tick label below is the only remaining signal of how many points a box rests
        # on. (draw_box, the n>=5 flag from genotype_class_summary, is left intact but
        # no longer gates the box -- markers below the MAF floor are removed by QC.)
        if len(yc):
            ax.boxplot([yc], positions=[cls], widths=0.5, showfliers=False)
        if len(yc):
            jitter = cls + (rng.random(len(yc)) - 0.5) * 0.18
            ax.plot(jitter, yc, "o", ms=3, alpha=0.45, color="#0072B2")
        xticklabels.append(f"{cls}\n(n={info['n']})")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(-0.6, 2.6)
    ax.set_xlabel("Genotype class (alt dosage)")
    ax.set_ylabel("Phenotype")

    if beta_mlm is not None and np.isfinite(float(beta_mlm)):
        se_txt = (f" ± {float(se_mlm):.3g}"
                  if se_mlm is not None and np.isfinite(float(se_mlm)) else "")
        beta_txt = f"β_MLM = {float(beta_mlm):.3g}{se_txt}"
    else:
        beta_txt = "structure-unadjusted (p-values unaffected)"
    if s["n_missing"]:
        beta_txt += f" | {s['n_missing']} missing"
    ax.set_title(f"{snp_id}\n{beta_txt}\n{flag} | η² = {s['eta2']:.2f}",
                 fontsize=8)
    fig.tight_layout()
    return fig


def build_snp_view_index(sig_table, plotted_ids, collapse_df=None):
    """One row per significant SNP recording whether it was plotted and, if the
    plot budget collapsed near-redundant SNPs, its plotting representative.

    Columns: ``SNP, Plotted, Representative_SNP, r2_to_representative,
    Representative_Of_Count``. The table stays complete; this is a plotting ledger,
    never a filter.
    """
    plotted = {str(s) for s in plotted_ids}
    out = pd.DataFrame({"SNP": sig_table["SNP"].astype(str).values})
    out["Plotted"] = out["SNP"].isin(plotted)
    if collapse_df is not None and len(collapse_df):
        cm = collapse_df.set_index(collapse_df["SNP"].astype(str))
        out["Representative_SNP"] = out["SNP"].map(cm["Representative_SNP"]).fillna(out["SNP"])
        out["r2_to_representative"] = out["SNP"].map(cm["r2_to_representative"]).fillna(1.0)
        out["Representative_Of_Count"] = (
            out["SNP"].map(cm["Representative_Of_Count"]).fillna(1).astype(int))
    else:
        out["Representative_SNP"] = out["SNP"]
        out["r2_to_representative"] = 1.0
        out["Representative_Of_Count"] = 1
    return out
