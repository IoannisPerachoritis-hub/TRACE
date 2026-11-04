
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from io import BytesIO
import zipfile
from datetime import datetime
import numpy as np
try:
    import streamlit as st
except ImportError:
    st = None
from scipy.stats import chi2, beta as beta_dist
from utils.pub_theme import (
    PALETTE, PALETTE_CYCLE, MANHATTAN_COLORS, SIG_LINE_COLOR, FIGSIZE,  # noqa: F401 (PALETTE_CYCLE re-exported)
    LD_HEATMAP_CMAP, export_matplotlib, export_plotly,
)
from annotation import canon_chr

def compute_cumulative_positions(df, chr_col="Chr", pos_col="Pos"):
    """
    Computes cumulative genomic positions and returns:
    df (with CumPos), tick_positions, tick_labels.
    """
    import re as _re
    df = df.copy()
    df[chr_col] = df[chr_col].astype(str)
    # Force int64 — on Windows `astype(int)` may default to int32, which
    # silently wraps when the cumulative offset grows past ~2.1 Gbp
    # (~10–12 typical plant chromosomes worth). Wrapping turns later
    # chromosomes' CumPos negative, which then renders them at the LEFT
    # of the plot before chromosomes 1-9.
    df[pos_col] = df[pos_col].astype("int64")

    def chr_key(x):
        c = canon_chr(x)
        try:
            return (0, int(c))
        except ValueError:
            pass
        # Fallback: extract leading digits from the canonical form so
        # names like "1_subA", "1.0", or anything else that retains a
        # numeric prefix still sort numerically.
        m = _re.match(r"^(\d+)", c)
        if m:
            return (0, int(m.group(1)))
        return (1, c)

    chroms_unique = sorted(df[chr_col].unique(), key=chr_key)

    gap = 1_000_000
    offset = 0  # plain Python int — has unlimited precision
    tick_positions = []
    tick_labels = []

    df["CumPos"] = np.nan

    for ch in chroms_unique:
        mask = df[chr_col] == ch
        if not mask.any():
            continue

        # int(...) forces Python int so the running offset never demotes
        # to a fixed-width numpy dtype (which is what triggered the
        # overflow warning on Windows).
        min_pos = int(df.loc[mask, pos_col].min())
        max_pos = int(df.loc[mask, pos_col].max())

        df.loc[mask, "CumPos"] = df.loc[mask, pos_col].astype("int64") + offset
        tick_positions.append(offset + (min_pos + max_pos) / 2.0)
        tick_labels.append(ch)

        offset += (max_pos - min_pos) + gap

    return df, tick_positions, tick_labels

def plot_manhattan_static(
    df,
    active_lod,
    active_label,
    title,
    chr_col="Chr",
    secondary_lod=None,
    secondary_label=None,
):
    """Static matplotlib Manhattan plot with signal amplification.

    Optional secondary threshold line (`secondary_lod` / `secondary_label`)
    is drawn as a dotted dark-grey line, useful for showing both
    Bonferroni and M_eff thresholds when a peak lies between them. The
    secondary line is informational only — it does NOT trigger SNP
    amplification (only the primary `active_lod` does).
    """
    fig = plt.figure(figsize=FIGSIZE["manhattan"])

    # chromosome ordering
    def chr_key(x):
        c = canon_chr(x)
        try:
            return (0, int(c))
        except ValueError:
            return (1, c)

    chroms_unique = sorted(df[chr_col].unique(), key=chr_key)

    for i, ch in enumerate(chroms_unique):
        mask = df[chr_col] == ch
        plt.scatter(df.loc[mask, "CumPos"], df.loc[mask, "-log10p"],
                    s=8, color=MANHATTAN_COLORS[i % 2], zorder=1)

    # Secondary threshold line — drawn FIRST so the primary's red line
    # sits visually on top if they happen to coincide. No SNP
    # amplification (informational only).
    has_secondary = (
        secondary_lod is not None
        and not (isinstance(secondary_lod, float) and np.isnan(secondary_lod))
        and secondary_label is not None
        and not str(secondary_label).startswith("FDR")
    )
    if has_secondary:
        plt.axhline(
            secondary_lod,
            linestyle=":",
            label=secondary_label,
            color=PALETTE.get("grey", "#7F7F7F"),
            linewidth=1.0,
            zorder=2,
        )

    # Primary significance line + significant-SNP amplification
    has_threshold = (active_lod is not None
                     and not np.isnan(active_lod)
                     and not active_label.startswith("FDR"))
    if has_threshold:
        plt.axhline(active_lod, linestyle="--", label=active_label,
                     color=SIG_LINE_COLOR, linewidth=1.0, zorder=3)

        # Signal amplification: enlarge + recolor significant SNPs
        sig_mask = df["-log10p"] >= active_lod
        if sig_mask.any():
            plt.scatter(
                df.loc[sig_mask, "CumPos"], df.loc[sig_mask, "-log10p"],
                s=18, color=SIG_LINE_COLOR, edgecolors="white",
                linewidths=0.3, zorder=4, label=f"Significant ({int(sig_mask.sum())})",
            )

    plt.xlabel("Chromosome")
    plt.ylabel(r"$-\log_{10}(p)$")
    plt.title(title)

    if has_threshold or has_secondary:
        _leg = plt.legend(loc="upper right", fontsize=9)
        # dashed frame so the legend's marker glyphs read as a key, not data points
        _leg.get_frame().set_linestyle("--")
        _leg.get_frame().set_edgecolor("0.35")
        _leg.get_frame().set_linewidth(0.8)
        _leg.get_frame().set_alpha(0.9)

    plt.tight_layout()
    return fig


def plot_manhattan_interactive(
    df,
    active_lod,
    active_label,
    title,
    secondary_lod=None,
    secondary_label=None,
):
    """Plotly Manhattan plot with alternating chromosome colors.

    Optional secondary threshold line (`secondary_lod` / `secondary_label`)
    is drawn as a dotted dark-grey line with annotation in the top-right
    (so it does not collide with the top-left primary annotation). Useful
    for showing both Bonferroni and M_eff thresholds simultaneously.
    """
    # Build chromosome color map (alternating blue/cyan like static version)
    def _chr_key(x):
        c = canon_chr(x)
        try:
            return (0, int(c))
        except ValueError:
            return (1, c)

    chroms_unique = sorted(df["Chr"].astype(str).unique(), key=_chr_key)
    chr_color_map = {
        ch: MANHATTAN_COLORS[i % 2] for i, ch in enumerate(chroms_unique)
    }

    df = df.copy()
    df["Chr"] = df["Chr"].astype(str)

    fig = px.scatter(
        df,
        x="CumPos",
        y="-log10p",
        color="Chr",
        color_discrete_map=chr_color_map,
        hover_data=[c for c in ["SNP", "Chr", "Pos", "PValue", "Beta_OLS", "SE_OLS", "Tstat_OLS"] if c in df.columns],
        title=title,
        render_mode="webgl",
    )
    fig.update_traces(marker_size=4)
    fig.update_layout(showlegend=False)

    # Secondary threshold line (informational, drawn first so primary
    # red line sits visually on top if they overlap)
    if (
        secondary_lod is not None
        and secondary_label is not None
        and not str(secondary_label).startswith("FDR")
    ):
        fig.add_hline(
            y=secondary_lod,
            line_dash="dot",
            annotation_text=secondary_label,
            annotation_position="top right",
            line_color=PALETTE.get("grey", "#7F7F7F"),
        )

    # Primary threshold line
    if active_lod is not None and not active_label.startswith("FDR"):
        fig.add_hline(
            y=active_lod,
            line_dash="dash",
            annotation_text=active_label,
            annotation_position="top left",
            line_color=SIG_LINE_COLOR,
        )

    return fig


def plot_qq(p_values, lambda_gc_used=None):
    """Reusable QQ plot with 95% confidence band."""
    p_vals = np.clip(p_values.astype(float), 1e-300, 1.0)
    n = len(p_vals)
    p_sorted = np.sort(p_vals)
    obs = -np.log10(p_sorted)
    exp = -np.log10((np.arange(1, n + 1) - 0.5) / n)

    fig = plt.figure(figsize=FIGSIZE["qq"])

    # 95% confidence band under the null (beta distribution order statistics)
    ranks = np.arange(1, n + 1)
    ci_lo = -np.log10(beta_dist.ppf(0.975, ranks, n - ranks + 1))
    ci_hi = -np.log10(beta_dist.ppf(0.025, ranks, n - ranks + 1))
    plt.fill_between(
        exp, ci_lo, ci_hi,
        color=PALETTE["grey"], alpha=0.20, zorder=0, label="95% CI",
    )

    # Diagonal reference line
    lim = max(exp.max(), obs.max()) * 1.02
    plt.plot([0, lim], [0, lim], linestyle="--", color=PALETTE["grey"], zorder=1)

    # Observed p-values
    plt.scatter(exp, obs, s=4, alpha=0.45, color=PALETTE["blue"], zorder=2)

    if lambda_gc_used is not None:
        plt.title(f"QQ Plot: \u03BBGC = {lambda_gc_used:.2f}")
    else:
        plt.title("QQ Plot")

    plt.xlabel(r"Expected $-\log_{10}(p)$")
    plt.ylabel(r"Observed $-\log_{10}(p)$")
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()

    return fig

def compute_lambda_gc(pvals, trim=True):
    """
    Genomic inflation factor λGC.
    Robust to small SNP counts.

    Parameters
    ----------
    trim : bool
        If True (default), exclude the 5% smallest and 5% largest p-values
        before computing the median chi-squared (bulk estimator; Yang et al.,
        2011).  Set to False for auto-PC selection so the scan detects tail
        inflation from residual population structure.
    """

    p = np.asarray(pvals, dtype=float)
    p = p[np.isfinite(p)]
    p = p[(p > 0) & (p <= 1)]

    if p.size < 50:
        return np.nan

    p = np.clip(p, 1e-300, 1.0)

    # Trim the 5% smallest and 5% largest p-values before computing lambda_GC.
    # This "bulk" estimator approach (cf. Yang et al., 2011, Nat Genet) excludes
    # the lower tail (true associations / LD artifacts) and upper tail (potential
    # deflation from population structure correction) for a more robust median
    # chi-squared estimate. Skipped when fewer than 500 SNPs to preserve precision.
    _TRIM_LO, _TRIM_HI = 0.05, 0.95
    if trim and p.size >= 500:
        q_low, q_high = np.quantile(p, [_TRIM_LO, _TRIM_HI])
        mask_bulk = (p >= q_low) & (p <= q_high)

        if mask_bulk.sum() < 50:
            mask_bulk = np.ones_like(p, dtype=bool)
    else:
        mask_bulk = np.ones_like(p, dtype=bool)

    chisq = chi2.isf(p[mask_bulk], df=1)
    med = np.median(chisq)

    return med / 0.454936423119572

# ---- Download helpers (delegate to pub_theme for PNG+SVG / HTML+SVG) ----
def download_matplotlib_fig(fig, filename="plot.png", label="Download"):
    stem = filename.replace(".png", "").replace(".svg", "")
    export_matplotlib(fig, stem, label_prefix=label)


def download_plotly_fig(fig, filename="interactive_plot.html", label="Download"):
    stem = filename.replace(".html", "")
    export_plotly(fig, stem, label_prefix=label)

def _build_gwas_results_zip(
    trait_col,
    gwas_df,
    figures_dict=None,
    ld_blocks_df=None,
    hap_gwas_df=None,
    pheno_label=None,
    extra_model_dfs=None,
    extra_tables=None,
    report_html=None,
    metadata=None,
):
    """
    Build a ZIP archive with GWAS results for one trait.

    Parameters
    ----------
    figures_dict : dict, {filename.png: matplotlib Figure or bytes}
    extra_model_dfs : dict, {model_name: DataFrame} — additional model CSVs
    extra_tables : dict, {filename.csv: DataFrame} — arbitrary extra tables
        (e.g. PC_selection_lambda.csv, Subsampling_stability.csv, Enrichment_keywords.csv)
    report_html : str or None — full HTML report to include in the ZIP
    metadata : dict or None — run provenance; when given, written as
        run_metadata.json (the last member). None (default) adds nothing, so the
        archive is byte-identical for callers that don't pass it.
    """
    buf = BytesIO()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    label = pheno_label or "unknown_pheno"

    zip_name = (
        f"GWAS_{label}__{trait_col}__{timestamp}.zip"
    )

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(
            f"tables/GWAS_{trait_col}.csv",
            gwas_df.to_csv(index=False),
        )

        if extra_model_dfs:
            for model_name, model_df in extra_model_dfs.items():
                zf.writestr(
                    f"tables/GWAS_{model_name}_{trait_col}.csv",
                    model_df.to_csv(index=False),
                )

        if extra_tables:
            for fname, tbl_df in extra_tables.items():
                if tbl_df is not None and not tbl_df.empty:
                    zf.writestr(f"tables/{fname}", tbl_df.to_csv(index=False))

        if figures_dict:
            for fname, fig in figures_dict.items():
                if fig is None:
                    continue
                if hasattr(fig, "savefig"):
                    fig_buf = BytesIO()
                    _dpi = 150 if fname.startswith("LD_heatmap_") else 600
                    # LD heatmaps are laid out on a fixed canvas (plot_ld_block_heatmap):
                    # save that canvas whole, so every LD_heatmap PNG has the same pixel
                    # size whatever its labels and title. Other figures keep the tight crop.
                    _bbox = (Bbox([[0, 0], fig.get_size_inches()])
                             if fname.startswith("LD_heatmap_") else "tight")
                    fig.savefig(fig_buf, format="png", dpi=_dpi, bbox_inches=_bbox)
                    zf.writestr(f"figures/{fname}", fig_buf.getvalue())
                    del fig_buf
                    plt.close(fig)
                elif isinstance(fig, (bytes, bytearray)):
                    zf.writestr(f"figures/{fname}", fig)

        if report_html:
            zf.writestr("report.html", report_html)

        # run_metadata.json written LAST so every other member's bytes are
        # unchanged vs a metadata=None build (additive provenance).
        if metadata is not None:
            import json
            zf.writestr(
                "run_metadata.json",
                json.dumps(metadata, indent=2, default=str),
            )

    buf.seek(0)
    return zip_name, buf


def _combine_gwas_zips(per_trait_zips, pheno_label=None, extra_members=None):
    """
    Merge per-trait ZIPs (bytes) into a single archive with `{trait}/` prefixes.

    Parameters
    ----------
    per_trait_zips : dict {trait_name: bytes}
    pheno_label    : str or None
    extra_members  : dict {file name: str or bytes} or None
        Written at the archive root after the traits (e.g. a multi-trait run's
        ``failed_traits.txt``). None leaves the archive exactly as before.
    """
    buf = BytesIO()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    label = pheno_label or "unknown_pheno"
    zip_name = (
        f"GWAS_{label}__MultiTrait_{len(per_trait_zips)}traits__{timestamp}.zip"
    )
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as out_zf:
        for trait_name, zip_bytes in per_trait_zips.items():
            src = BytesIO(zip_bytes)
            with zipfile.ZipFile(src, "r") as in_zf:
                for info in in_zf.infolist():
                    out_zf.writestr(
                        f"{trait_name}/{info.filename}",
                        in_zf.read(info.filename),
                    )
        for member_name, content in (extra_members or {}).items():
            out_zf.writestr(member_name, content)
    buf.seek(0)
    return zip_name, buf


# ============================================================
# LD UTILITIES
# ============================================================

def compute_r2_to_lead(geno_dosage_raw, sid, lead_snp_id, snp_mask, min_pair_n=15):
    """
    Compute r² of every SNP in snp_mask to the lead SNP.
    Uses pairwise-complete Pearson r — handles missing genotypes correctly.

    Parameters
    ----------
    geno_dosage_raw : ndarray (n_samples, n_snps)  — may contain NaNs
    sid             : array-like (n_snps,)          — SNP identifiers
    lead_snp_id     : str                           — ID of the lead SNP
    snp_mask        : bool ndarray (n_snps,)        — SNPs in the window
    min_pair_n      : int                           — min pairwise observations

    Returns
    -------
    r2 : ndarray (sum(snp_mask),) — r² values; NaN where insufficient data
    lead_idx_global : int — column index of lead SNP in geno_dosage_raw
    """
    sid = np.asarray(sid).astype(str)
    matches = np.where(sid == str(lead_snp_id))[0]
    if len(matches) == 0:
        raise ValueError(f"Lead SNP '{lead_snp_id}' not found in sid array.")
    lead_idx_global = int(matches[0])

    G = np.asarray(geno_dosage_raw, dtype=float)
    g_lead = G[:, lead_idx_global]

    win_indices = np.where(snp_mask)[0]
    n_win = len(win_indices)
    r2 = np.full(n_win, np.nan)

    for k, j in enumerate(win_indices):
        g_j = G[:, j]
        valid = np.isfinite(g_lead) & np.isfinite(g_j)
        n_valid = int(valid.sum())
        if n_valid < min_pair_n:
            continue
        gl = g_lead[valid]
        gj = g_j[valid]
        sd_l = gl.std()
        sd_j = gj.std()
        if sd_l < 1e-8 or sd_j < 1e-8:
            continue
        r = float(np.corrcoef(gl, gj)[0, 1])
        r2[k] = min(1.0, r * r)

    return r2, lead_idx_global


# ============================================================
# M_EFF — LI & JI (2005) EFFECTIVE NUMBER OF INDEPENDENT TESTS
# ============================================================

def compute_meff_li_ji(geno_imputed):
    """
    Estimate the effective number of independent SNP tests (Li & Ji 2005).

    Algorithm
    ---------
    1. Standardise each SNP column (zero mean, unit variance).
    2. Eigendecompose the dual Gram matrix D = Z Z.T / n  (n×n).
       The non-zero eigenvalues of D equal those of the m×m correlation
       matrix C = Z.T Z / n, but D is much smaller when n << m
       (e.g., 165×165 vs 44 000×44 000).  The remaining m − n eigenvalues
       of C are exactly 0 and contribute nothing to M_eff.
    3. M_eff = Σ_i  [ I(λ_i ≥ 1) + (λ_i mod 1) ]  for λ_i > 0
       (Negative eigenvalues from floating-point errors are clipped to 0.)

    Parameters
    ----------
    geno_imputed : ndarray (n_samples, m_snps) — mean-imputed, no NaNs

    Returns
    -------
    meff       : int   — effective number of independent tests
    eigenvalues: ndarray — eigenvalues sorted descending (clipped ≥ 0)
    """
    G = np.asarray(geno_imputed, dtype=np.float64)
    n, m = G.shape

    # Standardise columns (zero mean, unit variance → correlation structure)
    mu = G.mean(axis=0)
    sd = G.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Z = (G - mu) / sd

    # Dual-matrix trick: eigendecompose the n×n matrix instead of m×m.
    # Non-zero eigenvalues of Z.T@Z/n == non-zero eigenvalues of Z@Z.T/n.
    # When n << m this is orders of magnitude faster and gives the exact answer.
    if n < m:
        D = (Z @ Z.T) / float(n)
        eigs = np.linalg.eigvalsh(D)
    else:
        C = (Z.T @ Z) / float(n)
        np.fill_diagonal(C, 1.0)
        eigs = np.linalg.eigvalsh(C)

    eigs = np.clip(eigs, 0.0, None)[::-1]   # descending, clip negatives

    # Li & Ji formula
    meff = 0.0
    for lam in eigs:
        if lam >= 1.0:
            meff += 1.0 + (lam % 1.0)
        elif lam > 0:
            meff += lam
    # Remaining m - n eigenvalues are exactly 0 → contribute nothing

    return int(np.ceil(meff)), eigs

if st is not None:
    compute_meff_li_ji = st.cache_data(
        show_spinner=False, max_entries=3
    )(compute_meff_li_ji)


def _append_metadata_to_zip(zip_bytes, metadata_dict):
    import io, json, zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zin:
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                zout.writestr(item, zin.read(item.filename))

            # Add metadata JSON
            zout.writestr(
                "run_metadata.json",
                json.dumps(metadata_dict, indent=2, default=str),
            )

    buf.seek(0)
    return buf.read()


def plot_pca_scatter(pcs, y=None, n_show=2, title="PCA: Population Structure",
                     eigenvalues=None):
    """PCA scatter plot of first n_show principal components.

    Parameters
    ----------
    pcs : ndarray (n_samples, n_pcs)
        Principal component scores.
    y : ndarray or None
        Phenotype values for colour-coding. If None, uniform colour.
    n_show : int
        Number of PCs to display (2 → single scatter, 3 → pair grid).
    title : str
        Plot title.
    eigenvalues : ndarray or None
        Eigenvalues from PCA decomposition. If provided, axis labels show
        "PCx (XX.X%)" with the percentage of variance explained.

    Returns
    -------
    matplotlib Figure
    """
    pcs = np.asarray(pcs, dtype=float)
    n_show = min(n_show, pcs.shape[1])

    # Build PC axis labels with optional % variance explained
    def _pc_label(idx):
        label = f"PC{idx + 1}"
        if eigenvalues is not None:
            eig = np.asarray(eigenvalues, dtype=float)
            total = eig.sum()
            if total > 0 and idx < len(eig):
                pct = eig[idx] / total * 100
                label = f"PC{idx + 1} ({pct:.1f}%)"
        return label

    if n_show < 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "Need >= 2 PCs for scatter", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    if n_show == 2:
        fig, ax = plt.subplots(figsize=FIGSIZE.get("scatter", (6, 5)))
        sc = ax.scatter(
            pcs[:, 0], pcs[:, 1],
            c=y if y is not None else PALETTE["blue"],
            cmap="viridis" if y is not None else None,
            s=30, alpha=0.7, edgecolors="white", linewidths=0.3,
        )
        ax.set_xlabel(_pc_label(0))
        ax.set_ylabel(_pc_label(1))
        ax.set_title(title)
        if y is not None:
            fig.colorbar(sc, ax=ax, label="Phenotype")
        fig.tight_layout()
        return fig

    # n_show >= 3: pairwise grid of first 3 PCs
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    pairs = [(0, 1), (0, 2), (1, 2)]
    for ax, (i, j) in zip(axes, pairs):
        sc = ax.scatter(
            pcs[:, i], pcs[:, j],
            c=y if y is not None else PALETTE["blue"],
            cmap="viridis" if y is not None else None,
            s=25, alpha=0.7, edgecolors="white", linewidths=0.3,
        )
        ax.set_xlabel(_pc_label(i))
        ax.set_ylabel(_pc_label(j))
    axes[1].set_title(title)
    if y is not None:
        fig.colorbar(sc, ax=axes.tolist(), label="Phenotype", shrink=0.8)
    fig.tight_layout()
    return fig


def plot_pca_scree(eigenvalues, n_pcs_used=None, title="PCA Scree Plot"):
    """Bar chart of variance explained per PC with cumulative line overlay."""
    eig = np.asarray(eigenvalues, dtype=float)
    total = eig.sum()
    if total <= 0:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.text(0.5, 0.5, "No eigenvalue data", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    pct = eig / total * 100
    n = min(len(pct), 20)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(range(1, n + 1), pct[:n], color=PALETTE["blue"], alpha=0.7,
           edgecolor="white", linewidth=0.5)

    # Cumulative variance line on secondary axis
    cum_pct = np.cumsum(pct[:n])
    ax2 = ax.twinx()
    ax2.plot(range(1, n + 1), cum_pct, color=PALETTE["red"],
             marker="o", markersize=4, linewidth=1.5)
    ax2.set_ylabel("Cumulative variance (%)")
    ax2.set_ylim(0, 105)
    ax2.spines["top"].set_visible(False)

    # Mark selected PC count
    if n_pcs_used is not None and 1 <= n_pcs_used <= n:
        ax.axvline(n_pcs_used + 0.5, color=PALETTE["grey"],
                   linestyle="--", alpha=0.7)
        ax.text(n_pcs_used + 0.6, ax.get_ylim()[1] * 0.55,
                f"{n_pcs_used} PCs", color=PALETTE["grey"],
                fontsize=8, ha="left", va="center", fontstyle="italic")

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_xticks(range(1, n + 1))
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_maf_histogram(maf_values, maf_threshold=0.05,
                       title="Minor Allele Frequency Distribution (post-QC)"):
    """Post-QC MAF spectrum histogram."""
    maf = np.asarray(maf_values, dtype=float)
    maf = maf[np.isfinite(maf)]

    fig, ax = plt.subplots(figsize=FIGSIZE["histogram"])
    ax.hist(maf, bins=50, color=PALETTE["blue"], edgecolor="white", linewidth=0.5)

    if maf_threshold > 0:
        ax.axvline(maf_threshold, color=SIG_LINE_COLOR, linestyle="--",
                   linewidth=1.2, label=f"MAF threshold = {maf_threshold}")
        ax.legend()

    ax.set_xlabel("Minor Allele Frequency")
    ax.set_ylabel("Number of SNPs")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ============================================================
# REGIONAL ASSOCIATION (locus-zoom-style) — T-08
# ============================================================

# r² bins (LocusZoom convention), ordered low -> high LD to the lead SNP.
_R2_BIN_COLORS = ["#2C5AA0", "#56B4E9", "#009E73", "#E69F00", "#D55E00"]
_R2_BIN_LABELS = ["r²<0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "r²≥0.8"]
_R2_NA_COLOR = "#B0B0B0"
MAX_REGIONAL_SNPS = 4000  # matches the pairwise_r2 ceiling used elsewhere


def _r2_bin_index(r2):
    return np.digitize(np.asarray(r2, dtype=float), [0.2, 0.4, 0.6, 0.8]).astype(int)


def _r2_point_colors(r2):
    r2 = np.asarray(r2, dtype=float)
    idx = _r2_bin_index(r2)
    cols = [_R2_BIN_COLORS[i] for i in idx]
    for k in range(len(cols)):
        if not np.isfinite(r2[k]):
            cols[k] = _R2_NA_COLOR
    return cols


def _draw_gene_track(ax, genes, label=True):
    """Greedy-stacked gene bars with a strand arrowhead on the bar; gene-name
    labels are drawn iff ``label`` (dense windows overprint, so the tab turns
    them off above a threshold — the arrowhead keeps strand visible either way)."""
    g = genes.reset_index(drop=True)
    rows_end = []
    for _, gene in g.iterrows():
        s = float(gene["Start"]) / 1e6
        e = float(gene["End"]) / 1e6
        strand = str(gene.get("Strand", "."))
        yr = None
        for ri, rend in enumerate(rows_end):
            if s > rend + 0.002:
                yr = ri
                rows_end[ri] = e
                break
        if yr is None:
            yr = len(rows_end)
            rows_end.append(e)
        y = -yr
        ax.plot([s, e], [y, y], lw=3, color=PALETTE["green"], solid_capstyle="butt", zorder=2)
        # strand arrowhead on the bar itself — kept even when name labels are hidden
        if strand == "+":
            ax.plot([e], [y], marker=">", color=PALETTE["green"], markersize=5, zorder=3)
        elif strand == "-":
            ax.plot([s], [y], marker="<", color=PALETTE["green"], markersize=5, zorder=3)
        if label:
            ax.text((s + e) / 2.0, y + 0.18, str(gene.get("Gene_ID", "")),
                    fontsize=6, ha="center", va="bottom", clip_on=True)
    n = max(1, len(rows_end))
    ax.set_ylim(-n - 0.3, 0.9)
    ax.set_yticks([])
    ax.set_ylabel("genes", fontsize=8)


def _clamp_block_span(b0_mb, b1_mb, xlo, xhi, edge_frac=0.9):
    """Clamp a block span (in Mb) to the plotted x-range and decide how to draw it.

    Returns ``(bs, be, mode)`` where ``mode`` is ``'none'`` (span falls outside the
    view), ``'edges'`` (the clamped span exceeds ``edge_frac`` of the axis, so drawing
    a fill would swamp the panel — draw the two edges instead), or ``'fill'``."""
    bs = max(float(b0_mb), float(xlo))
    be = min(float(b1_mb), float(xhi))
    if be <= bs:
        return bs, be, "none"
    width = xhi - xlo
    if width > 0 and (be - bs) / width > edge_frac:
        return bs, be, "edges"
    return bs, be, "fill"


def plot_regional_association_static(
    window_df, r2_to_lead, lead_snp, sig_threshold,
    block_interval=None, block_members=None, genes=None, seed_threshold=None,
    gene_labels=True,
):
    """Regional (locus-zoom-style) association plot for one window. Pure; no Streamlit.

    x = position (Mb), y = -log10(p); points coloured by r² to the lead SNP (bins
    >=0.8 / 0.6-0.8 / 0.4-0.6 / 0.2-0.4 / <0.2; grey where r² is not computable). The
    lead SNP is a labelled diamond; a dashed line marks ``sig_threshold`` (None -> no
    line, e.g. FDR); ``block_interval`` (start_bp, end_bp) is shaded and
    ``block_members`` outlined; SNPs significant by the reporting rule but with
    p >= ``seed_threshold`` get a square marker (the seed-vs-member / omission set).
    A gene track (if ``genes`` given) shares the x-axis. Returns a matplotlib Figure.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    df = window_df.reset_index(drop=True)
    pos_mb = df["Pos"].to_numpy(dtype=float) / 1e6
    pv = np.clip(df["PValue"].to_numpy(dtype=float), 1e-300, 1.0)
    logp = -np.log10(pv)
    r2 = np.asarray(r2_to_lead, dtype=float)
    snp_ids = df["SNP"].astype(str).to_numpy()
    colors = _r2_point_colors(r2)

    has_genes = genes is not None and len(genes) > 0
    if has_genes:
        fig, (ax, axg) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                      gridspec_kw={"height_ratios": [4, 1]})
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        axg = None

    marker_handles = []
    if block_interval is not None:
        _bs, _be, _mode = _clamp_block_span(block_interval[0] / 1e6, block_interval[1] / 1e6,
                                            float(pos_mb.min()), float(pos_mb.max()))
        if _mode == "fill":
            ax.axvspan(_bs, _be, color=PALETTE["cyan"], alpha=0.25, zorder=0)
            marker_handles.append(Patch(facecolor=PALETTE["cyan"], alpha=0.25, label="LD block"))
        elif _mode == "edges":
            for _x in (_bs, _be):
                ax.axvline(_x, color=PALETTE["cyan"], ls=":", lw=1.2, zorder=1)
            marker_handles.append(Line2D([0], [0], color=PALETTE["cyan"], ls=":", label="LD block edges"))

    is_lead = snp_ids == str(lead_snp)
    nl = ~is_lead
    ax.scatter(pos_mb[nl], logp[nl], c=[colors[k] for k in range(len(colors)) if nl[k]],
               s=28, edgecolor="none", zorder=3)

    if block_members:
        mem = np.array([s in set(block_members) for s in snp_ids]) & nl
        if mem.any():
            ax.scatter(pos_mb[mem], logp[mem], facecolors="none", edgecolors="black",
                       s=64, linewidths=0.8, zorder=4)
            marker_handles.append(Line2D([0], [0], marker="o", ls="", mfc="none",
                                         mec="black", label="block member"))

    if sig_threshold is not None and seed_threshold is not None:
        below = (pv < float(sig_threshold)) & (pv >= float(seed_threshold)) & nl
        if below.any():
            ax.scatter(pos_mb[below], logp[below], marker="s", facecolors="none",
                       edgecolors=PALETTE["red"], s=95, linewidths=1.2, zorder=5)
            marker_handles.append(Line2D([0], [0], marker="s", ls="", mfc="none",
                                         mec=PALETTE["red"], label="sig, below seed p"))

    if is_lead.any():
        ax.scatter(pos_mb[is_lead], logp[is_lead], marker="D", c=PALETTE["red"],
                   s=95, edgecolor="black", linewidths=0.6, zorder=6)
        ax.annotate(str(lead_snp), (pos_mb[is_lead][0], logp[is_lead][0]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8, fontweight="bold")
        marker_handles.append(Line2D([0], [0], marker="D", ls="", mfc=PALETTE["red"],
                                     mec="black", label="lead SNP"))

    if sig_threshold is not None:
        ax.axhline(-np.log10(float(sig_threshold)), ls="--", color=SIG_LINE_COLOR, lw=1.2, zorder=2)
        marker_handles.append(Line2D([0], [0], ls="--", color=SIG_LINE_COLOR, label="significance"))

    r2_handles = [Line2D([0], [0], marker="o", ls="", mfc=_R2_BIN_COLORS[i], mec="none",
                         label=_R2_BIN_LABELS[i]) for i in range(5)]
    leg1 = ax.legend(handles=r2_handles, title="r² to lead", fontsize=7,
                     title_fontsize=7, loc="upper right", framealpha=0.9)
    ax.add_artist(leg1)
    if marker_handles:
        ax.legend(handles=marker_handles, fontsize=7, loc="upper left", framealpha=0.9)

    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_title(f"Regional association: lead {lead_snp}")

    if has_genes:
        _draw_gene_track(axg, genes, label=gene_labels)
        axg.set_xlabel("Position (Mb)")
    else:
        ax.set_xlabel("Position (Mb)")

    fig.tight_layout()
    return fig


def plot_regional_association_interactive(
    window_df, r2_to_lead, lead_snp, sig_threshold,
    block_interval=None, block_members=None, seed_threshold=None, uirevision=None,
):
    """Plotly regional association plot (hover: SNP id, p, r², block membership).

    ``uirevision`` should be a stable per-window signature: Plotly resets the
    axes when it changes (a new window) and preserves the user's zoom/pan while
    it stays the same (reruns within one window)."""
    import plotly.graph_objects as go

    df = window_df.reset_index(drop=True)
    pos_mb = df["Pos"].to_numpy(dtype=float) / 1e6
    pv = np.clip(df["PValue"].to_numpy(dtype=float), 1e-300, 1.0)
    logp = -np.log10(pv)
    r2 = np.asarray(r2_to_lead, dtype=float)
    snp_ids = df["SNP"].astype(str).to_numpy()
    colors = _r2_point_colors(r2)
    members = set(block_members) if block_members else set()
    is_lead = snp_ids == str(lead_snp)

    def _hover(k):
        parts = [snp_ids[k], f"p={pv[k]:.2e}",
                 ("r²=n/a" if not np.isfinite(r2[k]) else f"r²={r2[k]:.2f}")]
        if snp_ids[k] in members:
            parts.append("block member")
        return "<br>".join(parts)

    fig = go.Figure()
    if block_interval is not None:
        _bs, _be, _mode = _clamp_block_span(block_interval[0] / 1e6, block_interval[1] / 1e6,
                                            float(pos_mb.min()), float(pos_mb.max()))
        if _mode == "fill":
            fig.add_vrect(x0=_bs, x1=_be, fillcolor=PALETTE["cyan"], opacity=0.25,
                          line_width=0, layer="below")
        elif _mode == "edges":
            fig.add_vline(x=_bs, line_color=PALETTE["cyan"], line_dash="dot")
            fig.add_vline(x=_be, line_color=PALETTE["cyan"], line_dash="dot")
    nl = ~is_lead
    fig.add_trace(go.Scatter(
        x=pos_mb[nl], y=logp[nl], mode="markers",
        marker=dict(color=[colors[k] for k in range(len(colors)) if nl[k]], size=8,
                    line=dict(width=0)),
        text=[_hover(k) for k in range(len(snp_ids)) if nl[k]], hoverinfo="text", name="SNP"))
    if is_lead.any():
        _li = int(np.where(is_lead)[0][0])
        fig.add_trace(go.Scatter(
            x=pos_mb[is_lead], y=logp[is_lead], mode="markers+text",
            marker=dict(color=PALETTE["red"], size=14, symbol="diamond",
                        line=dict(color="black", width=1)),
            text=[str(lead_snp)], textposition="top center",
            hovertext=[_hover(_li)], hoverinfo="text", name="lead"))
    if sig_threshold is not None:
        fig.add_hline(y=-np.log10(float(sig_threshold)), line_dash="dash",
                      line_color=SIG_LINE_COLOR)
    fig.update_layout(title=f"Regional association: lead {lead_snp}",
                      xaxis_title="Position (Mb)", yaxis_title="-log10(p)",
                      height=480, showlegend=False, uirevision=uirevision)
    return fig


def ld_pairs_long(matrix, labels):
    """Tidy long-format of a square pairwise-LD matrix: one row per unordered pair.

    Returns a DataFrame ``[SNP_A, SNP_B, r2]`` over the upper triangle (i<j).
    Pure — the numeric export behind the LD heatmaps (serialisation, not
    computation)."""
    import pandas as pd
    M = np.asarray(matrix, dtype=float)
    labs = np.asarray(labels).astype(str)
    iu, ju = np.triu_indices(M.shape[0], k=1)
    return pd.DataFrame({"SNP_A": labs[iu], "SNP_B": labs[ju], "r2": M[iu, ju]})


# ---------------------------------------------------------------------------
# LD-block heatmap with neighbouring SNPs (reviewer point 2-INT.3)
# ---------------------------------------------------------------------------
# One drawing for every LD-block heatmap (CLI ZIP, One-Click ZIP, and the
# Post-GWAS Local LD tab in "Detected block" mode), so a reader can see whether
# LD drops at the block edges. Window rule, per block:
#   1. B = SNPs on the block chromosome with start <= pos <= end, by position.
#   2. Target N = max(ceil(|B| / LD_HEATMAP_BLOCK_MAX_FRAC), LD_HEATMAP_MIN_WINDOW).
#      Flank budget F = N - |B|: F // 2 SNPs immediately left of B, the rest
#      immediately right, on the same chromosome. A side that runs out gives its
#      shortfall to the other side; if both run out, everything available is
#      shown and the title states the block's share.
#   3. Above LD_HEATMAP_MAX_DISPLAY SNPs the window is thinned evenly at one rate
#      across block and flanks, always keeping the first and last block SNP, the
#      lead SNP and the nearest flank SNP on each side.
#   4. r2 on the shown SNPs with gwas.ld.pairwise_r2 (min_pair_n=20, as before).
# The constants are deliberately not GUI or CLI options.
LD_HEATMAP_BLOCK_MAX_FRAC = 0.5
LD_HEATMAP_MIN_WINDOW = 20
LD_HEATMAP_MAX_DISPLAY = 200
LD_HEATMAP_MAX_LABELS = 10


def _even_pick(n, k):
    """``k`` evenly spaced indices into ``range(n)``, ascending, including both ends
    when ``k >= 2``; all of them when ``k >= n``."""
    if k >= n:
        return np.arange(n)
    if k <= 0:
        return np.empty(0, dtype=int)
    if k == 1:
        return np.array([0])
    return np.unique(np.floor(np.linspace(0, n - 1, k) + 0.5).astype(int))


def _pick_keeping(n, k, protect, extra=None):
    """``k`` evenly spaced indices into ``range(n)`` that always include the indices
    in ``protect`` and ``extra``. Each one missing from the even picks replaces the
    nearest unprotected pick, so the count stays ``k`` (or the number protected)."""
    must = {int(i) for i in protect if 0 <= int(i) < n}
    if extra is not None and 0 <= int(extra) < n:
        must.add(int(extra))
    if k >= n:
        return np.arange(n)
    picks = {int(i) for i in _even_pick(n, max(int(k), len(must)))}
    for m in sorted(must):
        if m in picks:
            continue
        free = [p for p in picks if p not in must]
        if free:
            picks.discard(min(free, key=lambda p: (abs(p - m), p)))
        picks.add(m)
    return np.array(sorted(picks), dtype=int)


def _thinning_quota(sizes, k, need):
    """Split ``k`` picks across the segments ``sizes`` (left flank, block, right
    flank) at one rate: floor of each segment's exact share, then the remainder
    to the flanks first, so thinning never raises the block's share. Each segment
    gets at least ``need[i]`` and at most ``sizes[i]``."""
    total = float(sum(sizes))
    exact = [s * k / total for s in sizes]
    quota = [min(s, max(nd, int(np.floor(e)))) for s, nd, e in zip(sizes, need, exact)]
    order = sorted(range(len(sizes)), key=lambda i: (i == 1, -(exact[i] - np.floor(exact[i])), i))
    while sum(quota) < k and any(quota[i] < sizes[i] for i in range(len(sizes))):
        for i in order:
            if sum(quota) < k and quota[i] < sizes[i]:
                quota[i] += 1
    while sum(quota) > k:  # only when the minimums alone exceed k
        over = [i for i in range(len(sizes)) if quota[i] > need[i]]
        if not over:
            break
        quota[max(over, key=lambda i: quota[i])] -= 1
    return quota


def select_ld_heatmap_window(chroms, positions, block_chr, block_start, block_end,
                             lead_pos=None,
                             block_max_frac=LD_HEATMAP_BLOCK_MAX_FRAC,
                             min_window=LD_HEATMAP_MIN_WINDOW,
                             max_display=LD_HEATMAP_MAX_DISPLAY):
    """Which SNPs an LD-block heatmap shows (rule in the section comment above).

    ``chroms``/``positions`` are per-SNP arrays (any order; chromosome labels are
    canonicalised). Returns a dict: ``indices`` (into the caller's arrays, in
    position order) and ``in_block`` (per shown SNP); the pre-thinning counts
    ``n_block``, ``n_left``, ``n_right``, ``n_window`` and the target ``n_target``;
    what is drawn, ``n_shown``, ``n_block_shown``, ``block_share``; the flags
    ``thinned`` and ``exhausted`` (both sides ran out before the target);
    ``window_start``/``window_end`` (bp); and ``lead_index``, the lead SNP's index
    when it is shown, else None.
    """
    import math

    # canonicalise each distinct label once (a genome-wide array has few of them)
    uniq, inv = np.unique(np.asarray(chroms).astype(str), return_inverse=True)
    chroms_c = np.array([canon_chr(u) for u in uniq], dtype=object)[inv.ravel()]
    pos = np.asarray(positions).astype(np.int64)
    bchr = canon_chr(str(block_chr))
    block_start, block_end = int(block_start), int(block_end)

    on_chr = np.flatnonzero(chroms_c == bchr)
    order = on_chr[np.argsort(pos[on_chr], kind="stable")]
    p = pos[order]
    i0 = int(np.searchsorted(p, block_start, side="left"))
    i1 = int(np.searchsorted(p, block_end, side="right"))
    n_block = max(0, i1 - i0)
    avail_left, avail_right = i0, int(order.size) - i1

    n_target = max(int(math.ceil(round(n_block / float(block_max_frac), 9))), int(min_window))
    budget = max(0, n_target - n_block) if n_block else 0
    n_left = min(budget // 2, avail_left)
    n_right = min(budget - budget // 2, avail_right)
    spare = budget - n_left - n_right
    give = min(spare, avail_left - n_left)
    n_left, spare = n_left + give, spare - give
    give = min(spare, avail_right - n_right)
    n_right, spare = n_right + give, spare - give

    window = order[i0 - n_left:i1 + n_right] if n_block else order[:0]
    n_window = int(window.size)

    lead_local = None
    if lead_pos is not None and n_window:
        hit = np.flatnonzero(pos[window] == int(lead_pos))
        if hit.size:
            lead_local = int(hit[0])

    thinned = n_window > int(max_display)
    if thinned:
        sizes = [n_left, n_block, n_right]
        starts = [0, n_left, n_left + n_block]
        protect = [[n_left - 1], [0, n_block - 1], [0]]   # nearest flank SNPs, block ends
        need = [min(sizes[s], len(protect[s])) if sizes[s] else 0 for s in range(3)]
        lead_seg = None
        if lead_local is not None:
            lead_seg = int(np.searchsorted(starts, lead_local, side="right") - 1)
            if lead_local - starts[lead_seg] not in protect[lead_seg]:
                need[lead_seg] = min(sizes[lead_seg], need[lead_seg] + 1)
        quota = _thinning_quota(sizes, int(max_display), need)
        parts = []
        for s in range(3):
            if sizes[s] and quota[s]:
                extra = lead_local - starts[s] if lead_seg == s else None
                parts.append(starts[s] + _pick_keeping(sizes[s], quota[s], protect[s], extra))
        local = np.concatenate(parts) if parts else np.empty(0, dtype=int)
    else:
        local = np.arange(n_window)

    shown = window[local]
    in_block = (local >= n_left) & (local < n_left + n_block)
    n_shown, n_block_shown = int(shown.size), int(in_block.sum())
    lead_index = None
    if lead_local is not None and np.any(local == lead_local):
        lead_index = int(window[lead_local])
    return {
        "indices": shown,
        "in_block": in_block,
        "positions": pos[shown],
        "block_chr": bchr,
        "block_start": block_start,
        "block_end": block_end,
        "n_block": int(n_block),
        "n_left": int(n_left),
        "n_right": int(n_right),
        "n_window": n_window,
        "n_target": int(n_target),
        "n_shown": n_shown,
        "n_block_shown": n_block_shown,
        "block_share": (n_block_shown / n_shown) if n_shown else float("nan"),
        "thinned": bool(thinned),
        "exhausted": bool(n_block and spare > 0),
        "window_start": int(pos[shown[0]]) if n_shown else None,
        "window_end": int(pos[shown[-1]]) if n_shown else None,
        "lead_pos": None if lead_pos is None else int(lead_pos),
        "lead_index": lead_index,
    }


def lead_snp_position(lead_snp, sid, chroms, positions, block_chr):
    """Position (bp) of SNP ``lead_snp`` on ``block_chr``, or None if it is absent."""
    lead = str(lead_snp or "").strip()
    if not lead or sid is None:
        return None
    hits = np.flatnonzero(np.asarray(sid).astype(str) == lead)
    bchr = canon_chr(str(block_chr))
    chroms = np.asarray(chroms)
    for h in hits:
        if canon_chr(str(chroms[h])) == bchr:
            return int(np.asarray(positions)[h])
    return None


def _mb_labels(pos_bp):
    """Positions in Mb with just enough decimals (3 to 6) to tell them apart."""
    for d in range(3, 7):
        labels = [f"{p / 1e6:.{d}f}" for p in pos_bp]
        if len(set(labels)) == len(labels):
            break
    return labels


def plot_ld_block_heatmap(geno, chroms, positions, block_chr, block_start, block_end,
                          lead_pos=None,
                          block_max_frac=LD_HEATMAP_BLOCK_MAX_FRAC,
                          min_window=LD_HEATMAP_MIN_WINDOW,
                          max_display=LD_HEATMAP_MAX_DISPLAY,
                          label_mode="position", sid=None, min_pair_n=20, lead_star=False,
                          models=None):
    """r2 heatmap of an LD block and its neighbouring SNPs; returns ``(fig, info)``.

    ``geno`` is samples x SNPs (the imputed dosages the block finder used), its
    columns aligned with ``chroms``, ``positions`` and, for ``label_mode="snp_id"``,
    ``sid``. The SNPs shown follow select_ld_heatmap_window(); ``info`` is that
    dict plus ``r2`` (the matrix drawn) and ``title``. The figure is a fixed
    FIGSIZE["heatmap"] canvas with a fixed layout (no tight_layout), so every LD
    heatmap has the same size; _build_gwas_results_zip saves LD_heatmap_ files on
    that canvas. Lower triangle; the block's span is outlined, the lead SNP's tick
    is red, ticks are positions in Mb (or SNP IDs), at most LD_HEATMAP_MAX_LABELS
    of them. ``lead_star`` adds the Local LD tab's gold star on the lead's cell.
    ``models`` (e.g. ``["MLM", "MLMM"]``) adds "significant in MLM, MLMM" to the
    title. The pipeline passes the models that report this block or in which the
    marked lead SNP is significant, unless that is MLM alone.
    """
    import textwrap

    from matplotlib.patches import Polygon

    from gwas.ld import pairwise_r2

    info = select_ld_heatmap_window(
        chroms, positions, block_chr, block_start, block_end, lead_pos=lead_pos,
        block_max_frac=block_max_frac, min_window=min_window, max_display=max_display)
    idx = info["indices"]
    if idx.size < 2:
        raise ValueError(
            f"LD heatmap needs at least 2 SNPs; Chr{info['block_chr']} "
            f"{info['block_start']}-{info['block_end']} has {idx.size}")
    r2 = pairwise_r2(np.asarray(geno)[:, idx], min_pair_n=min_pair_n)
    info["r2"] = r2
    m = int(idx.size)

    lead_i = None
    if info["lead_index"] is not None:
        lead_i = int(np.flatnonzero(idx == info["lead_index"])[0])
    ticks = _pick_keeping(m, min(LD_HEATMAP_MAX_LABELS, m), [0, m - 1], extra=lead_i)
    if lead_i is not None:  # keep the lead's label clear of its neighbours' labels
        gap = max(1, m // 25)
        ticks = np.array([t for t in ticks if t == lead_i or abs(t - lead_i) >= gap], dtype=int)

    use_ids = label_mode == "snp_id" and sid is not None
    if use_ids:
        labels = [s if len(s) <= 28 else s[:27] + "…" for s in (str(v) for v in np.asarray(sid)[idx[ticks]])]
    else:
        labels = _mb_labels(info["positions"][ticks])
    fs = 6 if use_ids else 7
    label_in = 0.6 * fs / 72 * max(len(s) for s in labels)   # longest tick label, inches

    fig = plt.figure(figsize=FIGSIZE["heatmap"])
    W, H = (float(v) for v in fig.get_size_inches())
    top_in, right_in = 0.8, 1.0
    left_in, bottom_in = 0.35 + label_in, 0.66 + label_in
    side = min(W - left_in - right_in, H - bottom_in - top_in)
    ax = fig.add_axes([left_in / W, bottom_in / H, side / W, side / H])
    cax = fig.add_axes([(left_in + side + 0.15) / W, bottom_in / H, 0.18 / W, side / H])

    shown = np.tril(np.ones((m, m), dtype=bool), k=-1)
    data = np.ma.masked_where(~shown | ~np.isfinite(r2), r2)
    im = ax.imshow(data, cmap=LD_HEATMAP_CMAP, vmin=0.0, vmax=1.0,
                   interpolation="nearest", aspect="equal")
    ax.set_xlim(-0.5, m - 0.5)
    ax.set_ylim(m - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("r²", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    blk = np.flatnonzero(info["in_block"])
    if blk.size >= 2:
        b0, b1 = int(blk[0]), int(blk[-1])
        verts = [(b0 - 0.5, b0 + 0.5)]
        for i in range(b0 + 1, b1 + 1):
            verts += [(i - 0.5, i - 0.5), (i - 0.5, i + 0.5)]
        verts.append((b0 - 0.5, b1 + 0.5))
        ax.add_patch(Polygon(verts, closed=True, fill=False, edgecolor="black",
                             linewidth=1.4 if m <= 60 else (1.0 if m <= 120 else 0.8),
                             joinstyle="miter", zorder=4, clip_on=False))
    if lead_star and lead_i is not None:
        ax.scatter(lead_i, lead_i, marker="*", s=220, color="#F0E442",
                   edgecolor="#333333", linewidth=0.6, zorder=6, clip_on=False)

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=fs)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=fs)
    ax.tick_params(length=3, width=0.6)
    if lead_i is not None:
        k = int(np.flatnonzero(ticks == lead_i)[0])
        for axis, tls in ((ax.xaxis, ax.get_xticklabels()), (ax.yaxis, ax.get_yticklabels())):
            tls[k].set_color("#C00000")
            tls[k].set_fontweight("bold")
            tick = axis.get_major_ticks()[k]
            tick.tick1line.set_color("#C00000")
            tick.tick1line.set_markersize(6)
            tick.tick1line.set_markeredgewidth(1.4)
    ax.set_xlabel("SNP" if use_ids else f"Position on Chr{info['block_chr']} (Mb)", fontsize=9)

    c = info["block_chr"]
    head = (f"LD heatmap: Chr{c} block {info['block_start']:,}-{info['block_end']:,} "
            f"(block = {info['n_block_shown']} of {info['n_shown']} SNPs shown)")
    span = f"window {info['window_start']:,}-{info['window_end']:,}"
    if info["exhausted"]:
        span += f", all of Chr{c} ({info['n_window']} SNPs)"
    tail = [span]
    if info["thinned"]:
        tail.append(f"thinned to {info['n_shown']} of {info['n_window']}")
    if info["exhausted"]:
        tail.append(f"block share {info['block_share']:.0%}")
    if models:
        tail.append("significant in " + ", ".join(str(m) for m in models))
    title = "\n".join(textwrap.wrap(head, 90) + textwrap.wrap("; ".join(tail), 90))
    info["title"] = title
    fig.text(0.5, 1 - 0.12 / H, title, ha="center", va="top", fontsize=9.5)
    note = ("Black outline: the block's span (start to end)."
            + (" Red tick: lead SNP." if lead_i is not None else ""))
    fig.text(0.1 / W, 0.08 / H, note, ha="left", va="bottom", fontsize=7, color="#444444")
    return fig, info