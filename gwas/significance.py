"""T-40 — SignificanceRule: the reporting-threshold carrier.

The block-*seeding* threshold (`--ld-seed-p`, default 1e-5) and the *reporting*
threshold (`--sig-thresh`, Meff / Bonferroni / FDR) are different numbers, and
that gap is one of the two ways a genome-wide-significant SNP silently
disappears from the post-GWAS view. Every consumer that needs "which SNPs are
significant under the analysis threshold" — the isolated-SNP rescue, the
significant-SNP table, the LD page — must use the **reporting** rule, resolved
once per run, never re-derived. This dataclass carries it.

The resolution mirrors, exactly, the two places the rule is computed today: the
per-trait pipeline (``gwas/pipeline.py``, ``_run_trait``: the threshold block,
``_hm_sig_col`` and ``_sig_col_final``) and ``pages/GWAS_analysis.py`` (the
``_active_sig_col`` block).
Dependency-light on purpose (dataclasses / numpy / pandas only;
``add_significance_columns`` imports statsmodels when called) so it is shared by
pure code and by Streamlit without pulling either in.

``add_significance_columns`` writes the ``Significant_*`` columns themselves, with one
rule for every model's table (MLM, MLMM, FarmCPU) in the pipeline and in the GUI's
single-trait run.

Spec: ``docs/revision/specs/isolated_snp_spec.md`` §T-40.
"""
from __future__ import annotations

import dataclasses

import numpy as np

REPORTING_RULES = ("meff", "bonferroni", "fdr")

# rule -> the Significant_* boolean column add_significance_columns writes for it.
_BOOLEAN_COLUMN = {
    "meff": "Significant_Meff",
    "bonferroni": "Significant_Bonf",
    "fdr": "Significant_FDR",
    "custom": "Significant_Custom",
}


@dataclasses.dataclass(frozen=True)
class SignificanceRule:
    """The analysis reporting rule for one GWAS run.

    Fields
    ------
    rule            : one of ``REPORTING_RULES``.
    p_threshold     : the numeric p-value threshold; ``None`` **iff** ``rule == "fdr"``
                      (FDR is a per-SNP q-value decision, not a fixed threshold).
    boolean_column  : the ``Significant_*`` column the pipeline writes for this rule.
    n_tests         : M_eff for ``meff``; number of scanned SNPs for ``bonferroni``;
                      ``None`` for ``fdr``.
    label           : human label, e.g. ``"M_eff (M=242)"``.
    """

    rule: str
    p_threshold: float | None
    boolean_column: str
    n_tests: int | None
    label: str

    def __post_init__(self):
        if self.rule not in _BOOLEAN_COLUMN:
            raise ValueError(f"unknown reporting rule {self.rule!r}; expected one of {tuple(_BOOLEAN_COLUMN)}")
        if (self.p_threshold is None) != (self.rule == "fdr"):
            raise ValueError("p_threshold must be None iff rule == 'fdr'")
        if self.boolean_column != _BOOLEAN_COLUMN[self.rule]:
            raise ValueError(
                f"boolean_column {self.boolean_column!r} does not match rule {self.rule!r} "
                f"(expected {_BOOLEAN_COLUMN[self.rule]!r})"
            )

    def significant_mask(self, gwas_df) -> np.ndarray:
        """Boolean mask over ``gwas_df`` rows selecting the reporting-significant set.

        Strict resolution order (never falls back to another rule, never to 1e-5):

        1. the ``Significant_*`` boolean column if present (preferred for ALL rules,
           so a mask can never disagree with the column the pipeline published);
        2. else ``PValue < p_threshold`` (non-FDR rules only);
        3. else raise, naming both the missing column and the rule.
        """
        if self.boolean_column in gwas_df.columns:
            return gwas_df[self.boolean_column].fillna(False).to_numpy(dtype=bool)
        if self.p_threshold is not None:
            return gwas_df["PValue"].to_numpy(dtype=float) < float(self.p_threshold)
        raise ValueError(
            f"cannot resolve significance for rule {self.rule!r}: column "
            f"{self.boolean_column!r} is absent from gwas_df and this rule has no "
            f"p-value threshold (FDR requires the {self.boolean_column!r} column)"
        )


def _build(rule: str, n_snps: int, meff_val) -> SignificanceRule:
    n_snps = int(n_snps)
    if rule == "fdr":
        return SignificanceRule("fdr", None, "Significant_FDR", None, "FDR<0.05")
    if rule == "bonferroni":
        thr = 0.05 / n_snps
        return SignificanceRule("bonferroni", thr, "Significant_Bonf", n_snps, f"Bonferroni (m={n_snps})")
    # meff (default). M_eff-failure fallback mirrors cli.py:837-840: when M_eff
    # cannot be computed the pipeline uses naive Bonferroni over n_snps, but the
    # rule stays "meff" and the column stays Significant_Meff.
    m = int(meff_val) if meff_val is not None else n_snps
    thr = 0.05 / m
    return SignificanceRule("meff", thr, "Significant_Meff", m, f"M_eff (M={m})")


def _build_custom(thr: float) -> SignificanceRule:
    """A user-specified fixed p-value threshold (e.g. ``--sig-thresh 5e-8``).

    ``n_tests`` is ``None`` — the threshold is fixed by the user, not derived from a
    multiple-testing count. ``significant_mask`` resolves it via ``PValue < thr``.
    """
    thr = float(thr)
    return SignificanceRule("custom", thr, "Significant_Custom", None, f"p < {thr:.1e}")


def rule_from_cli_args(args, n_snps: int, meff_val) -> SignificanceRule:
    """Resolve the rule from a CLI ``args`` namespace — mirrors ``cli.py`` exactly.

    ``meff_val`` is the ``compute_meff_li_ji`` result (cli.py:834); pass ``None`` if
    the M_eff computation raised (cli.py:837-840), and the fallback threshold
    ``0.05 / n_snps`` is used. A numeric ``sig_thresh`` (e.g. ``5e-8``) resolves to a
    ``custom`` rule.
    """
    rule = getattr(args, "sig_thresh", "meff")
    if isinstance(rule, (int, float)) and not isinstance(rule, bool):
        return _build_custom(float(rule))
    if rule not in REPORTING_RULES:
        rule = "meff"
    return _build(rule, n_snps, meff_val)


def rule_from_streamlit(sig_rule_label, n_snps, meff_val, custom_thresh=None) -> SignificanceRule:
    """Resolve the rule from the GUI's reporting-rule label (the ``startswith``
    tests in ``pages/GWAS_analysis.py``). A ``"Custom…"`` label with a numeric
    ``custom_thresh`` resolves to the ``custom`` rule — parity with the CLI numeric
    ``--sig-thresh``."""
    s = str(sig_rule_label)
    if s.startswith("Custom") and custom_thresh is not None:
        return _build_custom(float(custom_thresh))
    if s.startswith("FDR"):
        rule = "fdr"
    elif s.startswith("M_eff"):
        rule = "meff"
    else:
        rule = "bonferroni"
    return _build(rule, n_snps, meff_val)


def add_significance_columns(df, *, bonf_thresh, meff_thresh=None, custom_thresh=None,
                             fdr_alpha=0.05):
    """Return a copy of a GWAS results table with its significance columns added.

    One rule for every model's table (MLM, MLMM, FarmCPU), so each model's columns
    mean the same thing. The p-value thresholds are MLM's (pass the same values for
    every model); FDR is computed within the table's own p-values:

    ``FDR``                Benjamini-Hochberg q-value (statsmodels ``multipletests``,
                           ``fdr_bh``) over this table's p-values.
    ``Significant_FDR``    the Benjamini-Hochberg decision at ``fdr_alpha``.
    ``Significant_Bonf``   ``PValue < bonf_thresh`` (0.05 / number of tested SNPs).
    ``Significant_Meff``   ``PValue < meff_thresh``; written only when it is given.
    ``Significant_Custom`` ``PValue < custom_thresh``; written only when it is given.

    A missing p-value (NaN, or NA in a nullable column) is never significant: its
    ``FDR`` is NaN, its ``Significant_*`` values are False, and Benjamini-Hochberg
    counts only the non-missing p-values. A p-value equal to a threshold is not
    below it. With no missing p-value the columns equal those the pipeline wrote for
    MLM before this helper existed. Columns are appended in the order above; an
    existing column of the same name is overwritten in place. The input frame is not
    modified.
    """
    from statsmodels.stats.multitest import multipletests

    out = df.copy()
    p = out["PValue"].to_numpy(dtype=float, na_value=np.nan)
    ok = ~np.isnan(p)
    if p.size and ok.all():
        reject, q, _, _ = multipletests(p, alpha=fdr_alpha, method="fdr_bh")
    else:
        reject = np.zeros(p.shape, dtype=bool)
        q = np.full(p.shape, np.nan)
        if ok.any():
            reject[ok], q[ok], _, _ = multipletests(p[ok], alpha=fdr_alpha, method="fdr_bh")
    out["FDR"] = q
    out["Significant_FDR"] = reject
    out["Significant_Bonf"] = p < bonf_thresh
    if meff_thresh is not None:
        out["Significant_Meff"] = p < meff_thresh
    if custom_thresh is not None:
        out["Significant_Custom"] = p < float(custom_thresh)
    return out


def frame_for_rule(df, rule):
    """The table to judge ``rule`` on when its threshold was chosen here, not by the run
    that wrote the table's columns (a Post-GWAS tab's own custom p-value).

    ``significant_mask`` prefers a stored ``Significant_*`` column. A stored
    ``Significant_Custom`` was written with the run's custom threshold, so for a custom
    rule it is dropped and the p-values are judged against the rule's own threshold.
    Other rules keep their column (their thresholds do not depend on the tab). The
    input frame is not modified.
    """
    if rule.rule == "custom" and rule.boolean_column in df.columns:
        return df.drop(columns=[rule.boolean_column])
    return df
