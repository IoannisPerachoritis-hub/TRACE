"""
Pure statistics for phenotype normality checking — no Streamlit imports.

Kept UI-free so it is unit-testable headlessly. Normality tests, skew/kurtosis,
transform application and transform suggestion all live here. Transform
application DELEGATES to ``gwas.qc.normalise_phenotype`` so this panel and the
GWAS run cannot disagree (they were two implementations that drifted on log).
"""

import numpy as np
import pandas as pd
from scipy import stats

# Significance threshold for normality tests; matches the house convention
# (pages/Pre GWAS-QTL analysis.py uses Shapiro p > 0.05 as "normal").
NORMALITY_ALPHA = 0.05

# Transform menu; apply_transform delegates to gwas.qc.normalise_phenotype.
TRANSFORM_NONE = "None (raw values)"
TRANSFORM_LOG10 = "Log10 (refused if any value <= 0)"
TRANSFORM_YEOJOHNSON = "Yeo-Johnson (robust Box-Cox)"
TRANSFORM_INT = "Rank-based inverse normal (INT)"
TRANSFORM_LABELS = [
    TRANSFORM_NONE,
    TRANSFORM_LOG10,
    TRANSFORM_YEOJOHNSON,
    TRANSFORM_INT,
]


def _finite(x):
    """Return the finite (non-NaN, non-inf) values of x as a 1-D float array."""
    x = np.asarray(x, dtype=float).reshape(-1)
    return x[np.isfinite(x)]


def _is_nonnormal(vals, alpha=NORMALITY_ALPHA):
    """Quick normality check on already-finite values (True if non-normal)."""
    n = vals.size
    if n < 3 or float(np.std(vals)) < 1e-12:
        return False
    if n <= 5000:
        p = float(stats.shapiro(vals)[1])
    elif n >= 8:
        p = float(stats.normaltest(vals)[1])
    else:
        return False
    return p < alpha


def run_normality_tests(x, alpha=NORMALITY_ALPHA):
    """Run a battery of normality tests on a single trait vector.

    Returns a dict of metrics. NaN/inf values are dropped first. Guards:
    n < 3 → insufficient; zero variance → constant; Shapiro skipped for
    n > 5000; D'Agostino K² skipped for n < 8.
    """
    clean = _finite(x)
    n = int(clean.size)
    res = {
        "n": n,
        "shapiro_W": np.nan, "shapiro_p": np.nan,
        "k2_stat": np.nan, "k2_p": np.nan,
        "ad_stat": np.nan, "ad_crit_5pct": np.nan, "ad_normal": None,
        "skew": np.nan, "kurtosis": np.nan,
        "verdict": "", "note": "",
    }

    if n < 3:
        res["verdict"] = "Insufficient (n<3)"
        res["note"] = "Need at least 3 non-missing values."
        return res

    if float(np.std(clean)) < 1e-12:
        res["verdict"] = "Constant (no variance)"
        res["note"] = "All values identical — normality undefined."
        res["skew"] = 0.0
        res["kurtosis"] = 0.0
        return res

    res["skew"] = float(stats.skew(clean))
    res["kurtosis"] = float(stats.kurtosis(clean, fisher=True))

    notes = []
    # Shapiro-Wilk — most powerful for small/moderate n; unreliable above 5000.
    if n <= 5000:
        W, p = stats.shapiro(clean)
        res["shapiro_W"], res["shapiro_p"] = float(W), float(p)
    else:
        notes.append("Shapiro-Wilk skipped (n>5000; unreliable)")

    # D'Agostino-Pearson K² — needs n >= 8 for the skew component.
    if n >= 8:
        k2, kp = stats.normaltest(clean)
        res["k2_stat"], res["k2_p"] = float(k2), float(kp)
    else:
        notes.append("D'Agostino K² skipped (n<8)")

    # Anderson-Darling — significance_level array is [15,10,5,2.5,1]; idx 2 = 5%.
    ad = stats.anderson(clean, dist="norm")
    res["ad_stat"] = float(ad.statistic)
    crit5 = float(ad.critical_values[2])
    res["ad_crit_5pct"] = crit5
    res["ad_normal"] = bool(ad.statistic < crit5)

    # Overall verdict: prefer Shapiro, else K², else Anderson-Darling.
    primary_p = res["shapiro_p"]
    if not np.isfinite(primary_p):
        primary_p = res["k2_p"]
    if np.isfinite(primary_p):
        res["verdict"] = "Normal" if primary_p >= alpha else "Non-normal"
    else:
        res["verdict"] = "Normal" if res["ad_normal"] else "Non-normal"

    res["note"] = "; ".join(notes)
    return res


def normality_summary_table(df, cols):
    """Build a one-row-per-trait normality summary DataFrame."""
    rows = []
    for c in cols:
        r = run_normality_tests(df[c].to_numpy())
        if r["ad_normal"] is None:
            ad_label = "—"
        else:
            ad_label = "Normal" if r["ad_normal"] else "Non-normal"
        rows.append({
            "Trait": c,
            "N": r["n"],
            "Shapiro W": r["shapiro_W"],
            "Shapiro p": r["shapiro_p"],
            "D'Agostino K² p": r["k2_p"],
            "Anderson-Darling": ad_label,
            "Skew": r["skew"],
            "Excess kurtosis": r["kurtosis"],
            "Verdict": r["verdict"],
            "Note": r["note"],
        })
    return pd.DataFrame(rows)


def apply_transform(x, method):
    """Apply a named transform to a trait vector, DELEGATING to the pipeline's
    ``gwas.qc.normalise_phenotype`` so this panel and the GWAS run cannot disagree
    (they were two implementations that had drifted on the log convention).

    Preserves NaN. Raises ValueError for an unknown method, a degenerate column
    Yeo-Johnson genuinely can't handle (constant / <3 finite), or a transform the
    pipeline REFUSES (log10 on non-positive values) -- so callers surface a clear
    message and fall back to raw.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if method == TRANSFORM_NONE:
        return x.copy()
    slug = {
        TRANSFORM_LOG10: "log",
        TRANSFORM_YEOJOHNSON: "yeojohnson",
        TRANSFORM_INT: "int",
    }.get(method)
    if slug is None:
        raise ValueError(f"Unknown transform method: {method!r}")
    if slug == "yeojohnson":
        vals = _finite(x)
        if vals.size < 3 or float(np.std(vals)) < 1e-12:
            raise ValueError("Yeo-Johnson requires ≥3 finite values with non-zero variance.")
    from gwas.qc import normalise_phenotype
    out, note = normalise_phenotype(x, slug)
    if note is not None:
        raise ValueError(note)
    return out


def transform_whole_table(df, cols, method):
    """Apply ONE transform uniformly to every numeric trait in a table.

    This is the correct choice for a metabolite panel destined for mGWAS: a
    single method keeps effect sizes / heritabilities comparable across traits
    and gives a clean, defensible Methods statement (rank-INT is the field
    default). Returns ``(transformed_df, report_df)``. NaN positions and the
    DataFrame index (sample/accession IDs) are preserved. A degenerate column a
    transform can't handle is left raw with an explanatory Note.

    ``report_df`` columns: ``Trait``, ``N``, ``Applied``, ``Note``.
    """
    out = df.copy()
    rows = []
    for c in cols:
        raw = df[c].to_numpy(dtype=float)
        vals = _finite(raw)
        note = ""
        try:
            out[c] = apply_transform(raw, method)
            applied = method
        except ValueError as err:
            applied = f"{TRANSFORM_NONE} (skipped)"
            note = str(err)
        rows.append({"Trait": c, "N": int(vals.size), "Applied": applied, "Note": note})
    return out, pd.DataFrame(rows)


def recommend_uniform_method(df, cols, methods=None):
    """Count how many traits each candidate method renders Normal (α = 0.05).

    Decision support for picking THE one method to apply to the whole panel.
    Note: rank-INT maps ranks onto normal scores, so it usually maxes this count
    (though heavy ties can leave a few non-normal). Callers should present it
    alongside the interpretability trade-off, not as a blind winner. Returns
    ``{method: n_normalised}``.
    """
    if methods is None:
        methods = [TRANSFORM_LOG10, TRANSFORM_YEOJOHNSON, TRANSFORM_INT]
    counts = {}
    for m in methods:
        n_norm = 0
        for c in cols:
            raw = df[c].to_numpy(dtype=float)
            try:
                tr = apply_transform(raw, m)
            except ValueError:
                continue
            if not _is_nonnormal(_finite(tr)):
                n_norm += 1
        counts[m] = n_norm
    return counts
