"""Per-panel PC count (k) of the PUBLISHED TRACE MLM runs -- single source of truth.

These are the number of principal components used as fixed-effect covariates in the
published TRACE mixed-model GWAS for each concordance panel. They were selected by the
shipped band auto-PC strategy (with the directional deflation guard); see the "Auto-PC
Strategy" section of CLAUDE.md and gwas/models.py::auto_select_pcs.

Provenance of each value (why it is what it is, not a magic number):

  tomato_locule_number = 0
  tomato_weight_g      = 0
      Band auto-PC's directional deflation guard forces k=0: lambda_GC at k=0 is below the
      band lower bound (0.95), so the kinship is already over-correcting on its own and
      adding PCs cannot repair it. The guard surfaces the deflation instead of burying it.

  pepper_BX            = 0
      Band auto-PC: lambda_GC is in-band at k=0, so no PCs are added.

  pepper_FWe           = 2
      Band auto-PC selects k=2. EMPIRICALLY CONFIRMED by the pruning ablation: replicating
      the published run at k=2 reproduces the on-disk Platform-LOCO AND Platform-Global
      p-values at Spearman rho = 1.000, whereas k=0 does not (it gives rho ~0.63 vs GAPIT3
      instead of the published 0.9145). This is the BAND value, NOT the elbow-strategy k=3
      recorded (as an old value) in benchmarks/check_pc_threshold.py.

Used by benchmarks/prune_ablation.py (the R1.15 pruning ablation, both LOCO and Global
conditions) and benchmarks/rerun_global_gwas.py (the Platform-Global regeneration). Both
import resolve_k so the four numbers are documented once and never defaulted silently.
"""

# panel run-name -> published PC count (k)
PUBLISHED_K = {
    "tomato_locule_number": 0,
    "tomato_weight_g": 0,
    "pepper_BX": 0,
    "pepper_FWe": 2,
}


def resolve_k(panel):
    """Return the published PC count for ``panel``.

    Raises ValueError on any panel not recorded here -- a global/LOCO rerun must never
    fall back to a silent default k (that is the bug this module exists to prevent).
    """
    if panel not in PUBLISHED_K:
        raise ValueError(
            f"No published PC count recorded for panel {panel!r}; "
            f"known panels: {sorted(PUBLISHED_K)}. "
            f"Refusing to default silently -- add the panel's band-selected k to "
            f"benchmarks/published_pc_counts.PUBLISHED_K first."
        )
    return PUBLISHED_K[panel]
