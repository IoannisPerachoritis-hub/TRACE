"""LD-quality triage — Layer 3 routing + η² comparability (T-32 / T-35).

Pure module: numpy + pandas only, no Streamlit, no logging, no I/O — usable from
cli.py and from tests without a Streamlit runtime. T-35 (η² comparability) lands
first; T-32 (TriageThresholds + triage_locus_view + triage_blocks) extends this
module. See docs/revision/specs/ld_triage_spec.md.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

ETA2_COMPARABILITY_COLUMNS = ["eta2_null_expected", "eta2_adj", "eta2_F_rank_delta"]

# ---- Layer 3 routing (T-32) ---------------------------------------------------
TRIAGE_COLUMNS = ["triage_view", "triage_primary", "triage_reason_code",
                  "triage_reason", "triage_axis_ld", "triage_flag_lead_outside_block",
                  "triage_thresholds_json"]

# reason codes (§6.5) + the §14.6 not-tested code
OK_BOTH = "OK_BOTH"
MLG_SAMPLE_LOSS = "MLG_SAMPLE_LOSS"
MLG_CONCENTRATED = "MLG_CONCENTRATED"
BLOCK_INCOHERENT = "BLOCK_INCOHERENT"
BLOCK_INCOHERENT_AND_FRAGMENTED = "BLOCK_INCOHERENT_AND_FRAGMENTED"
LEAD_UNUSABLE = "LEAD_UNUSABLE"
NO_INTERPRETABLE_VIEW = "NO_INTERPRETABLE_VIEW"
LD_UNDETERMINED = "LD_UNDETERMINED"
INSUFFICIENT_METRICS = "INSUFFICIENT_METRICS"
HAPLOTYPE_NOT_TESTED = "HAPLOTYPE_NOT_TESTED"
LEAD_OUTSIDE_BLOCK = "LEAD_OUTSIDE_BLOCK"


@dataclass(frozen=True)
class TriageThresholds:
    """Routing thresholds (§6.6). Five are bound to existing constants/flags at
    call time; ``lead_r2_frac`` is the one admitted CONVENTION (T-33: not
    load-bearing on the published set). ``r2_coherent`` has no default — the
    caller binds it to the run's ``--ld-r2`` so triage is a self-consistency
    check on the detector, not a new opinion."""
    r2_coherent: float                 # bound to --ld-r2
    lead_r2_frac: float = 0.50         # CONVENTION (exposed as --triage-lead-r2-frac)
    frac_retained_min: float = 0.75    # = gwas/haplotype.py:305 comparison
    mlg_eff_min: float = 2.0           # the F-test's >=2-group requirement, in effective units
    min_group_n: int = 3               # bound to --hap-min-group-size
    lead_maf_min: float = 0.01         # bound to maf_ld
    enabled: bool = True


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def triage_thresholds_json(thr: TriageThresholds) -> str:
    return json.dumps(asdict(thr), sort_keys=True)


def _null_triage(thr: TriageThresholds) -> dict:
    """--no-triage / disabled: stable schema, all routing fields empty."""
    return {"triage_view": "", "triage_primary": "", "triage_reason_code": "",
            "triage_reason": "", "triage_axis_ld": "", "triage_flag_lead_outside_block": False,
            "triage_thresholds_json": triage_thresholds_json(thr)}


def triage_locus_view(row, thr: TriageThresholds) -> dict:
    """Pure routing decision for one merged locus (Layer 3, §6.5). Consumes only
    the ldq_* / mlg_* / lead-usability fields of the row (no genotype access, no
    I/O). Returns exactly the triage_* keys of §9.4. Missing/NaN inputs degrade to
    both / lead_snp / INSUFFICIENT_METRICS — never to a suppression."""
    if not thr.enabled:
        return _null_triage(thr)

    def g(k):
        return row.get(k, np.nan) if hasattr(row, "get") else np.nan

    r2_mean = _f(g("ldq_r2_mean"))
    frac_lead = _f(g("ldq_frac_members_r2_lead_ge"))
    pairs_fin = _f(g("ldq_r2_pairs_finite"))
    pairs_tot = _f(g("ldq_r2_pairs_total"))
    frac_ret = _f(g("mlg_frac_retained"))
    inv_simp = _f(g("mlg_eff_inv_simpson"))
    lead_classes = _f(g("n_lead_classes_ge"))
    lead_maf = _f(g("lead_maf"))
    lead_in_raw = g("ldq_lead_in_block")
    lead_outside = (lead_in_raw is False) or (str(lead_in_raw).lower() == "false")
    n_members = _f(g("ldq_n_members"))
    n_tested = _f(g("n_samples_tested"))
    n_block = _f(g("n_samples_block"))
    top1 = _f(g("mlg_top1_freq"))
    n_hap = _f(g("n_tested_haplotypes"))
    n_other = _f(g("mlg_n_other"))
    lead_snp = g("ldq_lead_snp")

    # --- axis A (LD coherence), with the undetermined guard evaluated first ---
    ld_undet = (np.isfinite(pairs_fin) and pairs_fin < 3) or (
        np.isfinite(pairs_fin) and np.isfinite(pairs_tot) and pairs_tot > 0
        and pairs_fin / pairs_tot < 0.5)
    if ld_undet:
        A = "undetermined"
    elif np.isnan(r2_mean) or np.isnan(frac_lead):
        A = "nan"
    else:
        A = "pass" if (r2_mean >= thr.r2_coherent and frac_lead >= thr.lead_r2_frac) else "fail"

    # --- axis B (MLG usability); absent Layer-2 metrics => not tested ---
    if np.isnan(frac_ret) and np.isnan(inv_simp):
        B, B1_fail = "not_tested", False
    elif np.isnan(frac_ret) or np.isnan(inv_simp):
        B, B1_fail = "nan", False
    else:
        B1 = frac_ret >= thr.frac_retained_min
        B2 = inv_simp >= thr.mlg_eff_min
        B, B1_fail = ("pass" if (B1 and B2) else "fail"), (not B1)

    # --- axis L (lead usability) ---
    if np.isnan(lead_classes) or np.isnan(lead_maf):
        L = "nan"
    else:
        L = "pass" if (lead_classes >= 2 and lead_maf >= thr.lead_maf_min) else "fail"

    axis_ld = A
    view, primary, code = "both", "lead_snp", INSUFFICIENT_METRICS

    if B == "not_tested":
        view, primary, code = "both", "lead_snp", HAPLOTYPE_NOT_TESTED
    elif A == "nan" or B == "nan" or L == "nan":
        view, primary, code = "both", "lead_snp", INSUFFICIENT_METRICS
    elif A == "undetermined":
        if L == "pass":
            view, primary, code = "both", "lead_snp", LD_UNDETERMINED
        elif L == "fail" and B == "fail":
            view, primary, code = "neither", "none", NO_INTERPRETABLE_VIEW
        else:
            view, primary, code = "both", "lead_snp", LD_UNDETERMINED
    else:
        # A, B, L all in {pass, fail}: walk §6.5 rows 1-7
        if A == "pass" and B == "pass" and L == "pass":
            view, primary, code = "both", "haplotype", OK_BOTH
        elif A == "pass" and B == "fail" and L == "pass":
            code = MLG_SAMPLE_LOSS if B1_fail else MLG_CONCENTRATED
            view, primary = "both", "lead_snp"
        elif A == "fail" and B == "pass" and L == "pass":
            view, primary, code = "both", "lead_snp", BLOCK_INCOHERENT
        elif A == "fail" and B == "fail" and L == "pass":
            view, primary, code = "lead_snp", "lead_snp", BLOCK_INCOHERENT_AND_FRAGMENTED
        elif A == "pass" and B == "pass" and L == "fail":
            view, primary, code = "haplotype", "haplotype", LEAD_UNUSABLE
        else:  # L fail and (A fail or B fail)
            view, primary, code = "neither", "none", NO_INTERPRETABLE_VIEW

    # lead-outside-block override (§6.5): force lead_snp primary where a lead view
    # exists, and always record the flag + append the code token.
    if lead_outside:
        if view in ("both", "lead_snp"):
            primary = "lead_snp"
        code = f"{code};{LEAD_OUTSIDE_BLOCK}"

    reason = _reason_string(code, thr, dict(
        n_members=n_members, r2_mean=r2_mean, frac_lead=frac_lead, lead_snp=lead_snp,
        n_tested=n_tested, n_block=n_block, frac_ret=frac_ret, n_other=n_other,
        inv_simp=inv_simp, n_hap=n_hap, top1=top1, pairs_fin=pairs_fin, pairs_tot=pairs_tot))

    return {"triage_view": view, "triage_primary": primary, "triage_reason_code": code,
            "triage_reason": reason, "triage_axis_ld": axis_ld,
            "triage_flag_lead_outside_block": bool(lead_outside),
            "triage_thresholds_json": triage_thresholds_json(thr)}


def _pct(x):
    return f"{x:.0%}" if np.isfinite(x) else "?"


def _reason_string(code, thr, v) -> str:
    """Prose reason (§6.7): quote the numbers, name the threshold, name the figure
    to check. Never asserts causality or that a view is 'wrong'."""
    codes = code.split(";")
    base = codes[0]
    parts = []
    if base == OK_BOTH:
        parts.append("Both the haplotype and lead-SNP views are interpretable on their own sample sets.")
    elif base == BLOCK_INCOHERENT:
        parts.append(
            f"Mean pairwise r2 among the {int(v['n_members']) if np.isfinite(v['n_members']) else '?'} "
            f"block members is {v['r2_mean']:.2f}, below the block-forming threshold of "
            f"{thr.r2_coherent:.2f}. Check the block heatmap: a trustworthy haplotype view should show a "
            f"mostly dark triangle.")
    elif base == MLG_SAMPLE_LOSS:
        parts.append(
            f"Multi-locus genotypes retained {int(v['n_tested']) if np.isfinite(v['n_tested']) else '?'}/"
            f"{int(v['n_block']) if np.isfinite(v['n_block']) else '?'} samples ({_pct(v['frac_ret'])}); "
            f"{int(v['n_other']) if np.isfinite(v['n_other']) else '?'} samples fell into rare genotypes "
            f"and were excluded from the haplotype test.")
    elif base == MLG_CONCENTRATED:
        parts.append(
            f"Effective number of multi-locus genotypes is {v['inv_simp']:.1f} across "
            f"{int(v['n_hap']) if np.isfinite(v['n_hap']) else '?'} tested groups; the most common genotype "
            f"holds {_pct(v['top1'])} of samples. The haplotype comparison is close to one common genotype "
            f"versus a scatter.")
    elif base == BLOCK_INCOHERENT_AND_FRAGMENTED:
        parts.append(
            f"Block LD is incoherent (mean r2 {v['r2_mean']:.2f} < {thr.r2_coherent:.2f}) and the MLG "
            f"partition is fragmented (effective {v['inv_simp']:.1f}); prefer the lead-SNP view.")
    elif base == LEAD_UNUSABLE:
        parts.append("The lead-SNP genotype has too few usable classes or too low MAF; the haplotype view "
                     "is the interpretable panel here.")
    elif base == NO_INTERPRETABLE_VIEW:
        parts.append("No per-sample genotype-by-phenotype panel is interpretable for this block; the block "
                     "interval, annotation, heatmap, F, permutation p and eta2 are still reported.")
    elif base == LD_UNDETERMINED:
        parts.append(
            f"Only {int(v['pairs_fin']) if np.isfinite(v['pairs_fin']) else '?'}/"
            f"{int(v['pairs_tot']) if np.isfinite(v['pairs_tot']) else '?'} member SNP pairs had enough "
            f"overlapping calls to estimate r2; block-level LD quality is undetermined, not low.")
    elif base == HAPLOTYPE_NOT_TESTED:
        parts.append("This block was not haplotype-tested (fewer than two valid MLG groups after "
                     "exclusions); only the lead-SNP view is available.")
    else:
        parts.append("Insufficient metrics to route this locus; both views are shown, lead-SNP first.")
    if LEAD_OUTSIDE_BLOCK in codes:
        parts.append(
            f"Lead SNP {v['lead_snp']} is the block's seeding SNP and is not among its member SNPs; "
            f"the block was seeded from it but does not contain it.")
    return " ".join(parts)


def triage_blocks(merged: pd.DataFrame, thr: TriageThresholds) -> pd.DataFrame:
    """Row-wise application of triage_locus_view. Returns a copy with the triage_*
    columns appended in §9.4 order. Row count and row order are invariant."""
    out = merged.copy()
    n0 = len(out)
    if n0 == 0:
        for c in TRIAGE_COLUMNS:
            out[c] = pd.Series(dtype=object)
        return out
    recs = [triage_locus_view(out.iloc[i], thr) for i in range(n0)]
    for c in TRIAGE_COLUMNS:
        out[c] = [r[c] for r in recs]
    assert len(out) == n0, "triage_blocks must not change the row count"
    return out


def add_eta2_comparability(df: pd.DataFrame) -> pd.DataFrame:
    """T-35 — append η²-comparability columns. Pure arithmetic on the EXISTING
    ``eta2`` / ``df1`` / ``df2`` / ``F_perm`` columns; ``eta2`` is never recomputed
    and never modified, so any pre-existing value is bit-identical after this call.

    - ``eta2_null_expected`` = df1 / (df1 + df2): the η² a null (no group effect)
      partition produces by chance with these degrees of freedom. η² must be read
      against this floor, not against 0.
    - ``eta2_adj``           = 1 − (1 − eta2)·(df1 + df2)/df2: the bias-adjusted
      (Ω²-style) effect. Negatives are STORED, not clipped — a negative adjusted
      effect is the honest statement that η² did not clear its null expectation.
    - ``eta2_F_rank_delta``  = rank(eta2) − rank(F_perm) across the blocks in
      ``df`` (average ranks, ascending): 0 where η² and F order a block the same,
      signed where they disagree. Cross-block, so it is defined on the full table.
    """
    out = df.copy()
    if len(out) == 0:
        for c in ETA2_COMPARABILITY_COLUMNS:
            out[c] = pd.Series(dtype=float)
        return out

    df1 = pd.to_numeric(out["df1"], errors="coerce")
    df2 = pd.to_numeric(out["df2"], errors="coerce")
    eta2 = pd.to_numeric(out["eta2"], errors="coerce")

    out["eta2_null_expected"] = df1 / (df1 + df2)
    out["eta2_adj"] = 1.0 - (1.0 - eta2) * (df1 + df2) / df2      # negatives kept

    fcol = "F_perm" if "F_perm" in out.columns else ("F_param" if "F_param" in out.columns else None)
    if fcol is not None:
        r_eta = eta2.rank(method="average")
        r_f = pd.to_numeric(out[fcol], errors="coerce").rank(method="average")
        out["eta2_F_rank_delta"] = r_eta - r_f
    else:
        out["eta2_F_rank_delta"] = np.nan
    return out
