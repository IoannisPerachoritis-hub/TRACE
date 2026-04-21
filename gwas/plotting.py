
import plotly.express as px
import matplotlib.pyplot as plt
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
    export_matplotlib, export_plotly,
)

def compute_cumulative_positions(df, chr_col="Chr", pos_col="Pos"):
    """
    Computes cumulative genomic positions and returns:
    df (with CumPos), tick_positions, tick_labels.
    """
    df = df.copy()
    df[chr_col] = df[chr_col].astype(str)
    df[pos_col] = df[pos_col].astype(int)

    def chr_key(x):
        x = str(x).replace("chr", "").replace("Chr", "")
        try:
            return (0, int(x))
        except ValueError:
            return (1, x)

    chroms_unique = sorted(df[chr_col].unique(), key=chr_key)

    gap = 1_000_000
    offset = 0
    tick_positions = []
    tick_labels = []

    df["CumPos"] = np.nan

    for ch in chroms_unique:
        mask = df[chr_col] == ch
        if not mask.any():
            continue

        min_pos = df.loc[mask, pos_col].min()
        max_pos = df.loc[mask, pos_col].max()

        df.loc[mask, "CumPos"] = df.loc[mask, pos_col] + offset
        tick_positions.append(offset + (min_pos + max_pos) / 2.0)
        tick_labels.append(ch)

        offset += (max_pos - min_pos) + gap

    return df, tick_positions, tick_labels

def plot_manhattan_static(df, active_lod, active_label, title, chr_col="Chr"):
    """Static matplotlib Manhattan plot with signal amplification."""
    fig = plt.figure(figsize=FIGSIZE["manhattan"])

    # chromosome ordering
    def chr_key(x):
        x = str(x).replace("chr", "").replace("Chr", "")
        try:
            return (0, int(x))
        except ValueError:
            return (1, x)

    chroms_unique = sorted(df[chr_col].unique(), key=chr_key)

    for i, ch in enumerate(chroms_unique):
        mask = df[chr_col] == ch
        plt.scatter(df.loc[mask, "CumPos"], df.loc[mask, "-log10p"],
                    s=8, color=MANHATTAN_COLORS[i % 2], zorder=1)

    # significance line
    has_threshold = (active_lod is not None
                     and not np.isnan(active_lod)
                     and not active_label.startswith("FDR"))
    if has_threshold:
        plt.axhline(active_lod, linestyle="--", label=active_label,
                     color=SIG_LINE_COLOR, linewidth=1.0, zorder=2)

        # Signal amplification: enlarge + recolor significant SNPs
        sig_mask = df["-log10p"] >= active_lod
        if sig_mask.any():
            plt.scatter(
                df.loc[sig_mask, "CumPos"], df.loc[sig_mask, "-log10p"],
                s=18, color=SIG_LINE_COLOR, edgecolors="white",
                linewidths=0.3, zorder=3, label=f"Significant ({int(sig_mask.sum())})",
            )

    plt.xlabel("Chromosome")
    plt.ylabel(r"$-\log_{10}(p)$")
    plt.title(title)

    if has_threshold:
        plt.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    return fig


def plot_manhattan_interactive(df, active_lod, active_label, title):
    """Plotly Manhattan plot with alternating chromosome colors."""
    # Build chromosome color map (alternating blue/cyan like static version)
    def _chr_key(x):
        x = str(x).replace("chr", "").replace("Chr", "")
        try:
            return (0, int(x))
        except ValueError:
            return (1, x)

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
        plt.title(f"QQ Plot \u2014 \u03BBGC = {lambda_gc_used:.2f}")
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
                    fig.savefig(fig_buf, format="png", dpi=_dpi, bbox_inches="tight")
                    zf.writestr(f"figures/{fname}", fig_buf.getvalue())
                    del fig_buf
                    plt.close(fig)
                elif isinstance(fig, (bytes, bytearray)):
                    zf.writestr(f"figures/{fname}", fig)

        if report_html:
            zf.writestr("report.html", report_html)

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