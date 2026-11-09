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

# The box plot's note when its dots keep the single colour (no haplotype labels)
HAP_NOTE_NO_BLOCK = "not in an LD block: dots not coloured by haplotype"
HAP_NOTE_NO_GROUPS = "LD block has no haplotype groups: dots not coloured by haplotype"
# Names of the --norm slugs on a phenotype axis (the GUI's panel uses the same names)
NORM_AXIS_NAMES = {"int": "rank-INT", "log": "log10", "yeojohnson": "Yeo-Johnson",
                   "zscore": "z-score"}


def phenotype_axis_label(trait, norm=None):
    """The phenotype axis of a box plot: '<trait>' for raw values, '<trait> (<method>)'
    when the GWAS normalised it (e.g. 'locule_number (rank-INT)')."""
    if norm in (None, "", "none"):
        return str(trait)
    return f"{trait} ({NORM_AXIS_NAMES.get(str(norm), str(norm))})"


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


def _title_lines(snp_id, summary, flag, beta_mlm, se_mlm):
    """The box plot's three title lines: the SNP, beta_MLM (+- SE) or the unadjusted note
    (+ missing count), and the effect flag with the per-SNP eta2."""
    if beta_mlm is not None and np.isfinite(float(beta_mlm)):
        se_txt = (f" ± {float(se_mlm):.3g}"
                  if se_mlm is not None and np.isfinite(float(se_mlm)) else "")
        beta_txt = f"β_MLM = {float(beta_mlm):.3g}{se_txt}"
    else:
        beta_txt = "structure-unadjusted (p-values unaffected)"
    if summary["n_missing"]:
        beta_txt += f" | {summary['n_missing']} missing"
    return [str(snp_id), beta_txt, f"{flag} | η² = {summary['eta2']:.2f}"]


def snp_boxplot_title(snp_id, geno_col, y, *, beta_mlm=None, se_mlm=None, beta_ols=None):
    """The title lines of ``render_snp_boxplot`` for these inputs (the MLM effect and the
    per-SNP eta2), for the interactive plot."""
    return _title_lines(snp_id, genotype_class_summary(geno_col, y),
                        effect_flag(beta_mlm, beta_ols), beta_mlm, se_mlm)


def phenotype_scale_name(norm=None):
    """'raw', or the name of the GWAS's normalisation (e.g. 'rank-INT')."""
    if norm in (None, "", "none"):
        return "raw"
    return NORM_AXIS_NAMES.get(str(norm), str(norm))


SNP_BOXPLOT_DATA_COLUMNS = ["Sample", "Genotype_dosage", "Genotype_alleles", "Phenotype",
                            "Phenotype_scale", "Haplotype", "In_haplotype_test"]


def snp_boxplot_data(geno_col, y, sample_ids, *, haplotypes=None, in_test=None,
                     alleles=None, norm=None):
    """The samples a SNP's box plot draws, one row each, in input order.

    A sample is drawn when its genotype and phenotype are finite and its rounded dosage
    is 0, 1 or 2, as in ``render_snp_boxplot``. ``Phenotype`` is the value plotted (the
    GWAS's y; ``norm`` names its scale). ``haplotypes`` / ``in_test`` (aligned with
    ``sample_ids``) are the sample's label and test flag in the SNP's tested block
    (blank / False when the sample has none or the SNP is in no tested block).
    ``alleles`` is (REF, ALT): the genotype then reads e.g. "A/G" for dosage 1.
    """
    g = np.asarray(geno_col, dtype=float)
    yy = np.asarray(y, dtype=float)
    ids = np.asarray(sample_ids, dtype=str)
    if not (g.shape[0] == yy.shape[0] == ids.shape[0]):
        raise ValueError(f"{g.shape[0]} genotypes, {yy.shape[0]} phenotypes and "
                         f"{ids.shape[0]} sample IDs")
    drawn = (np.isfinite(g) & np.isfinite(yy)
             & np.isin(np.rint(np.where(np.isfinite(g), g, -1)), (0, 1, 2)))
    dose = np.rint(g[drawn]).astype(int)
    if alleles is not None and all(a not in (None, "", ".") for a in alleles):
        ref, alt = (str(a) for a in alleles)
        names = {0: f"{ref}/{ref}", 1: f"{ref}/{alt}", 2: f"{alt}/{alt}"}
        gt = [names[d] for d in dose]
    else:
        gt = [""] * int(drawn.sum())
    hap = (np.asarray(["" if h is None or (isinstance(h, float) and np.isnan(h)) else str(h)
                       for h in haplotypes], dtype=object)[drawn]
           if haplotypes is not None else np.full(int(drawn.sum()), "", dtype=object))
    flag = (np.asarray(in_test, dtype=bool)[drawn] if in_test is not None
            else np.zeros(int(drawn.sum()), dtype=bool))
    return pd.DataFrame({
        "Sample": ids[drawn], "Genotype_dosage": dose, "Genotype_alleles": gt,
        "Phenotype": yy[drawn], "Phenotype_scale": phenotype_scale_name(norm),
        "Haplotype": hap, "In_haplotype_test": flag,
    })[SNP_BOXPLOT_DATA_COLUMNS]


def plotly_snp_boxplot(data, title_lines, *, y_label=None, hap_color_map=None):
    """The interactive form of ``render_snp_boxplot`` from ``snp_boxplot_data`` rows.

    A box per genotype class and the same jittered points (same seed, same draws per
    class), coloured by haplotype as the static plot (a blank label counts as "Other";
    no label at all: one colour). Hover shows the sample, its genotype, the phenotype
    plotted and its haplotype. ``title_lines`` are ``snp_boxplot_title``'s.
    """
    import plotly.graph_objects as go

    from utils.pub_theme import hap_sort_key, haplotype_color_map

    rng = np.random.default_rng(0)                 # the static plot's jitter
    labelled = bool((data["Haplotype"].astype(str) != "").any())
    hap = data["Haplotype"].astype(str).where(data["Haplotype"].astype(str) != "", "Other")
    cmap = haplotype_color_map(sorted(set(hap), key=hap_sort_key)) if labelled else {}
    cmap.update(hap_color_map or {})
    fig = go.Figure()
    ticks, x_all = [], np.full(len(data), np.nan)
    for cls in (0, 1, 2):
        m = (data["Genotype_dosage"] == cls).to_numpy()
        n = int(m.sum())
        alle = data.loc[m, "Genotype_alleles"].iloc[0] if n and data.loc[m, "Genotype_alleles"].iloc[0] else ""
        ticks.append(f"{cls}{f' ({alle})' if alle else ''}<br>(n={n})")
        if not n:
            continue
        fig.add_trace(go.Box(x=np.full(n, cls), y=data.loc[m, "Phenotype"], width=0.5,
                             boxpoints=False, fillcolor="rgba(0,0,0,0)",
                             line=dict(color="#333333", width=1), hoverinfo="skip",
                             showlegend=False))
        x_all[m] = cls + (rng.random(n) - 0.5) * 0.18
    hover = ("<b>%{customdata[0]}</b><br>Genotype: %{customdata[1]}<br>"
             "Phenotype: %{y:.4g}<br>Haplotype: %{customdata[2]}<extra></extra>")
    gt_txt = [f"{d} ({a})" if a else f"{d}" for d, a in
              zip(data["Genotype_dosage"], data["Genotype_alleles"])]
    groups = ((["Other"] if "Other" in set(hap) else [])
              + sorted({h for h in hap if h != "Other"}, key=hap_sort_key)) if labelled else [None]
    for h in groups:
        k = (hap == h).to_numpy() if h is not None else np.ones(len(data), dtype=bool)
        fig.add_trace(go.Scatter(
            x=x_all[k], y=data.loc[k, "Phenotype"], mode="markers",
            marker=dict(size=6, opacity=0.6, color=cmap.get(h, "#0072B2") if h else "#0072B2"),
            name=h or "samples", showlegend=labelled,
            customdata=np.column_stack([data.loc[k, "Sample"].astype(str),
                                        np.asarray(gt_txt, dtype=object)[k],
                                        data.loc[k, "Haplotype"].astype(str).replace("", "none")]),
            hovertemplate=hover))
    fig.update_layout(
        title=dict(text="<br>".join(title_lines), font=dict(size=11)),
        xaxis=dict(title="Genotype class (alt dosage)", tickvals=[0, 1, 2], ticktext=ticks,
                   range=[-0.6, 2.6]),
        yaxis=dict(title=y_label or "Phenotype"),
        legend=dict(title="Haplotype"), height=460, margin=dict(t=90))
    return fig


def render_snp_boxplot(snp_id, geno_col, y, *, beta_mlm=None, se_mlm=None,
                       beta_ols=None, dpi=110, hap_labels=None, hap_color_map=None,
                       hap_note=None, y_label=None):
    """One genotype-class boxplot for a SNP → a matplotlib ``Figure`` (Agg).

    A box is drawn only for classes with n ≥ 5 (else jittered points only); every
    class 0/1/2 is shown including n=0; the structure-adjusted β_MLM ± SE, the
    raw-vs-adjusted effect flag, and the per-SNP η² are annotated. When β_MLM is
    absent the annotation says "structure-unadjusted (p-values unaffected)" — the
    effect axis is suppressed, never faked (integration §5.2).

    ``hap_labels`` (one label per sample, aligned with ``geno_col`` and ``y``; a
    missing label counts as "Other") colours each dot by its haplotype, with
    utils.pub_theme.haplotype_color_map (``hap_color_map`` overrides colours), and
    adds a legend of the labels plotted outside the axes. The jitter is drawn exactly
    as without labels, so no dot moves. Without labels, ``hap_note`` is written under
    the plot (why the dots keep the single colour). ``y_label`` names the phenotype
    axis ("Phenotype" when not given).
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

    labels, present, cmap = None, [], {}
    if hap_labels is not None:
        from utils.pub_theme import hap_sort_key, haplotype_color_map
        labels = np.asarray(["Other" if (h is None or (isinstance(h, float) and np.isnan(h))
                                         or str(h) == "") else str(h) for h in hap_labels],
                            dtype=object)
        if labels.shape[0] != g.shape[0]:
            raise ValueError(f"hap_labels has {labels.shape[0]} labels for {g.shape[0]} samples")
        drawn = valid & np.isin(np.rint(np.where(np.isfinite(g), g, -1)), (0, 1, 2))
        present = sorted(set(labels[drawn].tolist()), key=hap_sort_key)
        cmap = haplotype_color_map(present)
        cmap.update(hap_color_map or {})

    fig, ax = plt.subplots(figsize=(3.8, 3.0) if labels is not None else (3.2, 3.0), dpi=dpi)
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
            if labels is None:
                ax.plot(jitter, yc, "o", ms=3, alpha=0.45, color="#0072B2")
            else:
                # the class's dots split by haplotype: the same positions, one colour each
                hc = labels[m]
                order = (["Other"] if "Other" in set(hc.tolist()) else []) + sorted(
                    {h for h in hc.tolist() if h != "Other"}, key=hap_sort_key)
                for h in order:
                    k = hc == h
                    ax.plot(jitter[k], yc[k], "o", ms=3, alpha=0.45, color=cmap[h])
        xticklabels.append(f"{cls}\n(n={info['n']})")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(-0.6, 2.6)
    ax.set_xlabel("Genotype class (alt dosage)")
    ax.set_ylabel(y_label if y_label else "Phenotype")

    ax.set_title("\n".join(_title_lines(snp_id, s, flag, beta_mlm, se_mlm)), fontsize=8)
    if labels is not None and present:
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], ls="", marker="o", ms=4, color=cmap[h], label=h)
                   for h in present]
        ax.legend(handles=handles, title="Haplotype", loc="upper left",
                  bbox_to_anchor=(1.02, 1.0), borderaxespad=0, frameon=False, fontsize=6,
                  title_fontsize=6, handletextpad=0.2, labelspacing=0.3)
        fig.tight_layout()
    elif labels is None and hap_note:
        fig.text(0.5, 0.01, hap_note, ha="center", va="bottom", fontsize=6, color="#555555")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    else:
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
