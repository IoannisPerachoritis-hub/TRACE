"""T-40 — SignificanceRule: the reporting-threshold carrier.

The block-*seeding* threshold (`--ld-seed-p`, default 1e-5) and the *reporting*
threshold (`--sig-thresh`, Meff / Bonferroni / FDR) are different numbers, and
that gap is one of the two ways a genome-wide-significant SNP silently
disappears from the post-GWAS view. Every consumer that needs "which SNPs are
significant under the analysis threshold" — the isolated-SNP rescue, the
significant-SNP table, the LD page — must use the **reporting** rule, resolved
once per run, never re-derived. This dataclass carries it.

The resolution mirrors, exactly, the two places the rule is computed today:
`cli.py` (threshold block near :829-854, boolean columns :870-872, `_sig_col_name`
:1091-1094) and `pages/GWAS_analysis.py` (:3462-3473). Dependency-light on
purpose (dataclasses / numpy / pandas only) so it is shared by pure code and by
Streamlit without pulling either in.

Spec: ``docs/revision/specs/isolated_snp_spec.md`` §T-40.
"""
from __future__ import annotations

import dataclasses

import numpy as np

REPORTING_RULES = ("meff", "bonferroni", "fdr")

# rule -> the Significant_* boolean column the pipeline materialises (cli.py:870-872).
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


def rule_from_streamlit(sig_rule_label: str, n_snps: int, meff_val) -> SignificanceRule:
    """Resolve the rule from the GUI's label string — mirrors
    ``pages/GWAS_analysis.py:3462-3473`` (the ``startswith`` tests)."""
    s = str(sig_rule_label)
    if s.startswith("FDR"):
        rule = "fdr"
    elif s.startswith("M_eff"):
        rule = "meff"
    else:
        rule = "bonferroni"
    return _build(rule, n_snps, meff_val)
