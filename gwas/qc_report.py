"""Report-only QC statistics (packages P1-P5 of the QC overhaul).

Everything here is *report-only*: it computes and summarises QC statistics but
filters nothing -- no sample, marker, or trait is removed on the strength of any
value below.  The functions take the post-QC genotype DataFrame (samples x
markers, NaN for missing) that both orchestration paths already build, so no
existing pipeline function changes signature.

  P1  per_sample_heterozygosity  -- het fraction + |z|>3 outlier flag
  P2  duplicate_pairs            -- pairwise genotype concordance, near-dup list
  P3  per_variant_fis            -- per-marker Ho and F_IS = 1 - Ho/He
  P4  trait_summary              -- trait distribution summary
  P5  compute_qc_report / render_qc_report_markdown / qc_report_figures

Wired into cli.py (written into the run ZIP) and pages/GWAS_analysis.py
(rendered inline).  DEV tree only for now.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── P1 ─────────────────────────────────────────────────────────────────────
def per_sample_heterozygosity(geno_df: pd.DataFrame, z_thresh: float = 3.0) -> dict:
    """Per-sample heterozygosity = fraction of *called* genotypes that are
    heterozygous (dosage == 1).  Flags samples with |z| > ``z_thresh`` against
    the panel mean/SD (+-3 SD is the Anderson 2010 / Marees 2018 convention).
    Report only -- flagged samples are NOT excluded.
    """
    G = geno_df.to_numpy(dtype="float64")
    called = np.isfinite(G)
    n_called = called.sum(axis=1)
    n_het = (G == 1).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        het = np.where(n_called > 0, n_het / n_called, np.nan)
    mean = float(np.nanmean(het))
    sd = float(np.nanstd(het, ddof=1))
    z = (het - mean) / sd if sd > 0 else np.zeros_like(het)
    flagged = np.abs(z) > z_thresh
    samples = geno_df.index.to_numpy().astype(str)
    per_sample = pd.DataFrame(
        {"sample": samples, "heterozygosity": het, "z": z, "flagged": flagged}
    )
    flag_idx = [i for i in np.argsort(-np.abs(z)) if flagged[i]]
    return {
        "mean": mean,
        "sd": sd,
        "median": float(np.nanmedian(het)),
        "z_thresh": z_thresh,
        "n_flagged": int(flagged.sum()),
        "per_sample": per_sample,
        "flagged_samples": [
            {"sample": samples[i], "heterozygosity": float(het[i]), "z": float(z[i])}
            for i in flag_idx
        ],
    }


# ── P2 ─────────────────────────────────────────────────────────────────────
def duplicate_pairs(
    geno_df: pd.DataFrame,
    conc_thresh: float = 0.99,
    sample_cap: int = 2000,
    marker_subsample: int | None = None,
    rng_seed: int = 42,
) -> dict:
    """Pairwise raw genotype concordance across all sample pairs; report pairs
    with concordance >= ``conc_thresh`` (near-duplicates / high relatedness).
    Concordance = fraction of jointly-called markers with equal dosage.

    O(n^2 * m).  Above ``sample_cap`` samples the marker set is subsampled (a
    runtime guard, disclosed in the result) so the scan never dominates the run.
    Report only -- no sample is excluded.
    """
    G = geno_df.to_numpy(dtype="float32")
    n, m = G.shape
    samples = geno_df.index.to_numpy().astype(str)
    subsampled = False
    if n > sample_cap and marker_subsample is None:
        marker_subsample = min(m, 5000)
    if marker_subsample is not None and marker_subsample < m:
        rng = np.random.default_rng(rng_seed)
        cols = np.sort(rng.choice(m, size=marker_subsample, replace=False))
        G = G[:, cols]
        subsampled = True
    used_m = G.shape[1]

    # Vectorised concordance via BLAS: for each dosage d, (G==d) is an n x m
    # 0/1 matrix and (G==d) @ (G==d).T counts markers where both samples carry
    # d; summing over d=0,1,2 gives jointly-equal counts.  NaN==d is False, so
    # missing genotypes never count as equal.  n_both = called @ called.T.
    called = np.isfinite(G).astype(np.float32)
    n_both = called @ called.T
    n_equal = np.zeros((n, n), dtype=np.float32)
    for d in (0.0, 1.0, 2.0):
        ind = (G == d).astype(np.float32)
        n_equal += ind @ ind.T
    with np.errstate(invalid="ignore", divide="ignore"):
        conc = np.where(n_both > 0, n_equal / n_both, 0.0)
    iu, ju = np.triu_indices(n, k=1)
    keep = (conc[iu, ju] >= conc_thresh) & (n_both[iu, ju] > 0)
    pairs = []
    for i, j in zip(iu[keep], ju[keep]):
        nb = int(round(float(n_both[i, j])))
        n_diff = nb - int(round(float(n_equal[i, j])))
        pairs.append(
            {
                "sample_a": samples[i],
                "sample_b": samples[j],
                "concordance": round(float(conc[i, j]), 4),
                "n_markers_compared": nb,
                "n_markers_differ": n_diff,
            }
        )
    pairs.sort(key=lambda p: (-p["concordance"], p["n_markers_differ"]))
    return {
        "conc_thresh": conc_thresh,
        "n_pairs_scanned": n * (n - 1) // 2,
        "n_markers_used": int(used_m),
        "marker_subsampled": subsampled,
        "n_pairs_flagged": len(pairs),
        "pairs": pairs,
    }


# ── P3 ─────────────────────────────────────────────────────────────────────
def per_variant_fis(geno_df: pd.DataFrame, he_min: float = 0.05) -> dict:
    """Per-marker observed heterozygosity Ho and F_IS = 1 - Ho/He, He = 2pq
    (guarded at He == 0).  Summary (median, 5-95th pct) is over markers with
    He > ``he_min``.  On a single unstructured panel this quantity is strictly
    F_IS; the crop-QC literature calls it F_IT and the arithmetic is identical.
    Report only.
    """
    G = geno_df.to_numpy(dtype="float64")
    called = np.isfinite(G)
    n_called = called.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ho = np.where(n_called > 0, (G == 1).sum(axis=0) / n_called, np.nan)
        ac = np.nansum(G, axis=0)
        an = 2.0 * n_called
        p = np.where(an > 0, ac / an, np.nan)
        he = 2.0 * p * (1.0 - p)
        fis = np.where(he > 0, 1.0 - ho / he, np.nan)
    mask = (he > he_min) & np.isfinite(fis)
    sub = fis[mask]
    return {
        "he_min": he_min,
        "n_markers": int(len(ho)),
        "n_he_gt_min": int(mask.sum()),
        "median_fis": float(np.median(sub)) if sub.size else float("nan"),
        "pct5_fis": float(np.percentile(sub, 5)) if sub.size else float("nan"),
        "pct95_fis": float(np.percentile(sub, 95)) if sub.size else float("nan"),
        "ho": ho,
        "fis": fis,
    }


# ── P4 ─────────────────────────────────────────────────────────────────────
def trait_summary(trait_values, trait_col: str, n_original: int | None = None) -> dict:
    """Trait distribution: n, mean, SD, min, max, skewness, % missing, and the
    count of distinct values.  Report only.
    """
    v = pd.to_numeric(pd.Series(np.asarray(trait_values).ravel()), errors="coerce")
    finite = v.dropna()
    denom = n_original if n_original is not None else int(len(v))
    try:
        from scipy.stats import skew as _skew
        sk = float(_skew(finite.to_numpy())) if finite.size > 2 else float("nan")
    except Exception:
        sk = float("nan")
    return {
        "trait": trait_col,
        "n": int(finite.size),
        "mean": float(finite.mean()) if finite.size else float("nan"),
        "sd": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
        "min": float(finite.min()) if finite.size else float("nan"),
        "max": float(finite.max()) if finite.size else float("nan"),
        "skew": sk,
        "pct_missing": float(100.0 * (denom - finite.size) / denom) if denom else 0.0,
        "n_distinct": int(finite.nunique()),
        "_values": finite.to_numpy(),
    }


# ── P5 ─────────────────────────────────────────────────────────────────────
def compute_qc_report(
    geno_df: pd.DataFrame,
    trait_values,
    trait_col: str,
    qc_snp: dict | None = None,
    n_original_samples: int | None = None,
    het_z_thresh: float = 3.0,
    dup_conc_thresh: float = 0.99,
    dup_sample_cap: int = 2000,
) -> dict:
    """Assemble P1-P4 into one report dict.  Called from both orchestration
    paths right after the post-QC dosage matrix is built (before any LD-dedup
    marker subset), so P2 sees the full retained marker set.
    """
    return {
        "sample_het": per_sample_heterozygosity(geno_df, z_thresh=het_z_thresh),
        "dup_pairs": duplicate_pairs(
            geno_df, conc_thresh=dup_conc_thresh, sample_cap=dup_sample_cap
        ),
        "variant_fis": per_variant_fis(geno_df),
        "trait": trait_summary(trait_values, trait_col, n_original=n_original_samples),
        "qc_snp": dict(qc_snp) if qc_snp else {},
        "n_samples": int(geno_df.shape[0]),
        "n_markers": int(geno_df.shape[1]),
    }


def render_qc_report_markdown(report: dict) -> str:
    """One markdown table set summarising the QC report.  No figures here."""
    h = report["sample_het"]
    d = report["dup_pairs"]
    f = report["variant_fis"]
    t = report["trait"]
    lines = []
    lines += [
        "## Per-sample heterozygosity",
        "",
        f"- Samples: {report['n_samples']} (none excluded)",
        f"- Panel mean {h['mean']:.4f}, median {h['median']:.4f}, SD {h['sd']:.4f}",
        f"- Flagged |z| > {h['z_thresh']:g}: **{h['n_flagged']}** (report only, not excluded)",
        "",
    ]
    if h["flagged_samples"]:
        lines += ["| sample | heterozygosity | z |", "| --- | --- | --- |"]
        lines += [
            f"| {s['sample']} | {s['heterozygosity']:.4f} | {s['z']:+.2f} |"
            for s in h["flagged_samples"]
        ]
        lines += [""]
    lines += [
        "## Duplicate / high-relatedness pairs",
        "",
        f"- Pairs scanned: {d['n_pairs_scanned']} on {d['n_markers_used']} markers"
        + (" (marker-subsampled runtime guard)" if d["marker_subsampled"] else ""),
        f"- Pairs at concordance >= {d['conc_thresh']:g}: **{d['n_pairs_flagged']}** (report only)",
        "",
    ]
    if d["pairs"]:
        lines += ["| sample A | sample B | concordance | markers differ |",
                  "| --- | --- | --- | --- |"]
        lines += [
            f"| {p['sample_a']} | {p['sample_b']} | {p['concordance']:.4f} | "
            f"{p['n_markers_differ']} of {p['n_markers_compared']} |"
            for p in d["pairs"]
        ]
        lines += [""]
    lines += [
        "## Per-variant F_IS",
        "",
        "On a single unstructured panel this is strictly F_IS; the crop-QC literature "
        "(Glaubitz 2014; Pavan 2020) calls it F_IT and the arithmetic is identical.",
        "",
        f"- Median F_IS over markers with He > {f['he_min']:g}: **{f['median_fis']:.3f}** "
        f"(5-95th pct {f['pct5_fis']:.3f}-{f['pct95_fis']:.3f}; n = {f['n_he_gt_min']})",
        "",
        "## Trait distribution",
        "",
        f"- {t['trait']}: n = {t['n']}, mean {t['mean']:.4g}, SD {t['sd']:.4g}, "
        f"range [{t['min']:.4g}, {t['max']:.4g}], skew {t['skew']:.3f}, "
        f"missing {t['pct_missing']:.1f}%, distinct values {t['n_distinct']}",
        "",
    ]
    return "\n".join(lines)


def qc_report_dataframes(report: dict) -> dict:
    """Tidy CSV-able frames for the run ZIP (report-only)."""
    h, d, f, t = (report["sample_het"], report["dup_pairs"],
                  report["variant_fis"], report["trait"])
    summary = pd.DataFrame(
        [
            ("samples", report["n_samples"]),
            ("markers", report["n_markers"]),
            ("het_mean", round(h["mean"], 4)),
            ("het_median", round(h["median"], 4)),
            ("het_outliers_z>%g" % h["z_thresh"], h["n_flagged"]),
            ("relatedness_pairs_conc>=%g" % d["conc_thresh"], d["n_pairs_flagged"]),
            ("median_F_IS(He>%g)" % f["he_min"], round(f["median_fis"], 3)),
            ("F_IS_5th_pct", round(f["pct5_fis"], 3)),
            ("F_IS_95th_pct", round(f["pct95_fis"], 3)),
            ("trait", t["trait"]),
            ("trait_n", t["n"]),
            ("trait_mean", t["mean"]),
            ("trait_sd", t["sd"]),
            ("trait_skew", round(t["skew"], 3)),
            ("trait_pct_missing", t["pct_missing"]),
            ("trait_n_distinct", t["n_distinct"]),
        ],
        columns=["statistic", "value"],
    )
    out = {
        "QC_report_summary.csv": summary,
        "QC_sample_heterozygosity.csv": h["per_sample"],
    }
    if d["pairs"]:
        out["QC_relatedness_pairs.csv"] = pd.DataFrame(d["pairs"])
    return out


def qc_report_figures(report: dict):
    """Small figure set (P5): heterozygosity, F_IS, and trait histograms.
    Returns {filename.png: matplotlib Figure}.  Import matplotlib lazily so the
    module stays importable in headless/no-mpl contexts.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}
    het = report["sample_het"]["per_sample"]["heterozygosity"].to_numpy()
    fis = report["variant_fis"]["fis"]
    fis = fis[np.isfinite(fis)]
    tvals = report["trait"].get("_values")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    axes[0].hist(het[np.isfinite(het)], bins=40, color="#4C72B0")
    axes[0].set_title("Per-sample heterozygosity"); axes[0].set_xlabel("het fraction")
    axes[1].hist(fis, bins=40, color="#55A868")
    axes[1].set_title("Per-variant F_IS"); axes[1].set_xlabel("F_IS")
    if tvals is not None and len(tvals):
        axes[2].hist(np.asarray(tvals), bins=40, color="#C44E52")
    axes[2].set_title(f"Trait: {report['trait']['trait']}"); axes[2].set_xlabel("value")
    fig.tight_layout()
    return {"QC_report_distributions.png": fig}
