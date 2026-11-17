"""PC counts (k) of the TRACE MLM runs behind the concordance tables, one source of truth.

PUBLISHED_K: the number of principal components used as fixed-effect covariates in the
published TRACE mixed-model GWAS for each concordance panel (Tables S4 and S11). They came from
the strict band rule behind the paper's PC counts (smallest k with 0.95 <= lambda_GC <= 1.05;
commit 03095b7), with its deflation guard. That rule has since been removed from TRACE, which
takes a fixed --n-pcs; these values are kept as the provenance of the published results.

  tomato_locule_number = 0, tomato_weight_g = 0
      The deflation guard forces k = 0: lambda_GC at k = 0 is below 0.95, so the kinship is
      already over-correcting and adding PCs cannot repair it.
  pepper_BX = 0
      lambda_GC is in the band at k = 0.
  pepper_FWe = 2
      The rule selects k = 2. Replicating the published run at k = 2 reproduces its Platform-LOCO
      and Platform-Global p-values at Spearman rho = 1.000; k = 0 does not.

BAND_RULE_K: the same strict band rule run again on the corrected QC data (band_rule_pc_counts.py,
its scan in band_rule_pc_counts.csv): FWe 3, BX 3, weight 0, locule 0. The pepper rows at these
counts are the side rows of Tables S4 and S11 (--k-set band).

GAPIT_FARMCPU_K_AS_RUN: GAPIT FarmCPU's PC counts as run for Table S10 (its 12 Mar 2026
timing.csv files): FWe 3, BX 0, weight 0, locule 2. The second set of Table S10 uses PUBLISHED_K.

Used by rerun_trace_mlm.py, rerun_global_gwas.py and prune_ablation.py through resolve_k, so the
counts are written once and never defaulted silently.
"""

# panel run-name -> published PC count (k)
PUBLISHED_K = {
    "tomato_locule_number": 0,
    "tomato_weight_g": 0,
    "pepper_BX": 0,
    "pepper_FWe": 2,
}

# The strict band rule on the corrected QC data (band_rule_pc_counts.py): the k of the side rows
BAND_RULE_K = {
    "tomato_locule_number": 0,
    "tomato_weight_g": 0,
    "pepper_BX": 3,
    "pepper_FWe": 3,
}

# GAPIT FarmCPU's PC counts as run for Table S10 (the 12 Mar 2026 timing.csv files)
GAPIT_FARMCPU_K_AS_RUN = {
    "tomato_locule_number": 2,
    "tomato_weight_g": 0,
    "pepper_BX": 0,
    "pepper_FWe": 3,
}


def resolve_k(panel, k_set="published"):
    """Return the PC count for ``panel``: the published one, or with k_set="band" the strict band
    rule's on the corrected data (BAND_RULE_K).

    Raises ValueError on any panel not recorded here -- a global/LOCO rerun must never
    fall back to a silent default k (that is the bug this module exists to prevent).
    """
    table = {"published": PUBLISHED_K, "band": BAND_RULE_K}[k_set]
    if panel not in table:
        raise ValueError(
            f"No published PC count recorded for panel {panel!r}; "
            f"known panels: {sorted(PUBLISHED_K)}. "
            f"Refusing to default silently -- add the panel's band-selected k to "
            f"benchmarks/published_pc_counts.PUBLISHED_K first."
        )
    return table[panel]
