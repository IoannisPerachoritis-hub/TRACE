"""Recovered PC-selection helpers -- benchmark REPRODUCIBILITY ONLY (not shipped).

`auto_select_pcs` (lambda-GC band scan + parsimony + directional deflation guard) and
`select_best_pc_from_lambdas` (its pure-arithmetic FarmCPU picker) were REMOVED from the
shipped tool (deleted from gwas/models.py in commit 30f84fe). TRACE no longer
auto-selects PCs -- it uses a fixed --n-pcs (default 0) plus the report-only diagnostics in
gwas/pc_diagnostics.py (which never decide a PC count).

These two functions are recovered here VERBATIM from the pre-removal commit (30f84fe~1) so the
historical benchmark reproducers -- rerun_trace_farmcpu (Table S10 FarmCPU concordance),
check_farmcpu_pc_guard, check_pc_threshold, runtime_benchmark -- remain runnable and reproduce
the published lambda-GC scan / band-vs-elbow numbers from a clean clone. This module is NOT
imported by cli.py, pages/, or any shipped code path; it does not re-introduce the selector.
"""
import logging
import numpy as np
import pandas as pd
from gwas.models import _run_gwas_impl, run_mlmm_research_grade_fast, run_farmcpu
from gwas.kinship import _build_loco_kernels_impl
from gwas.utils import PhenoData, CovarData


def auto_select_pcs(
    geno_imputed, y, sid, chroms, chroms_num, positions,
    iid, Z_grm, chroms_grm, K_base, pcs_full,
    max_pcs=10, progress_callback=None, strategy="band",
    use_loco=True, band_lo=0.95, band_hi=1.05,
    parsimony_tolerance=0.02,
    early_stop=True,
    model="mlm",
    farmcpu_p_threshold=0.01,
    farmcpu_max_iterations=10,
    farmcpu_max_pseudo_qtns=15,
    farmcpu_final_scan="ols",
    mlmm_p_enter=1e-4,
    mlmm_max_cof=10,
):
    """
    Run a GWAS model at each PC count 0..max_pcs and compute λGC.

    Returns a DataFrame with columns:
        n_pcs, lambda_gc, delta_from_1, recommended

    Parameters
    ----------
    model : {'mlm', 'mlmm', 'farmcpu'}, default 'mlm'
        Which association model to scan. All three share the same band /
        closest_to_1 selection logic and the directional deflation guard.

        - ``'mlm'``: LOCO MLM via :func:`_run_gwas_impl`. The ``use_loco``
          flag controls whether per-chromosome kinships or the global GRM
          are used for the final scan.
        - ``'mlmm'``: MLMM via :func:`run_mlmm_research_grade_fast` with
          the global kinship (``K0``). ``use_loco`` is ignored since MLMM
          uses ``K0`` directly.
        - ``'farmcpu'``: FarmCPU via :func:`run_farmcpu`. ``use_loco``
          controls LOCO kinship in FarmCPU's MLM final scan.
    strategy : str, 'band' or 'closest_to_1'
        'band' (default): edge-tolerant band selection. Recommend the
            smallest k whose delta_from_1 is within
            (band_edge_delta + parsimony_tolerance) of zero, where
            band_edge_delta = max(band_hi - 1.0, 1.0 - band_lo). With
            the defaults this is 0.05 + 0.02 = 0.07, so any k with
            lambda_GC in [0.93, 1.07] is acceptable and the smallest
            such k wins.

            This unified rule applies parsimony uniformly across the
            band edge — differences in delta smaller than the
            measurement noise (~0.02 on small panels) do not justify
            additional PCs. The earlier strict-band rule discarded
            parsimony as soon as one k crossed the band edge, which
            could pick k=8 over k=1 to chase a 0.01 reduction in
            lambda_GC.

            If no k is within (band_edge_delta + parsimony_tolerance)
            of 1.0, falls back to closest-to-1.0 with adaptive
            tolerance: smallest k whose delta is within
            max(parsimony_tolerance, best_delta * 0.15) of the best
            delta. This handles severe inflation where PCs cannot
            fully control structure.
        'closest_to_1': pick the PC count with λGC nearest to 1.0.
    band_lo : float
        Lower bound of the acceptable λGC band (default 0.95).
    band_hi : float
        Upper bound of the acceptable λGC band (default 1.05).
    parsimony_tolerance : float
        Tolerance for treating delta-from-1 differences as noise
        (default 0.02). Applied to both the band edge (acceptable
        if delta <= band_edge_delta + parsimony_tolerance) and the
        fallback (when nothing is acceptable, pick smallest k within
        max(parsimony_tolerance, best_delta * 0.15) of best).
    early_stop : bool
        If True (default), the band scan stops as soon as it finds the
        smallest acceptable k or the deflation guard fires. The
        returned DataFrame still has max_pcs + 1 rows; un-scanned k
        get lambda_gc=NaN and status='skipped'. Set False to force a
        full scan for diagnostic purposes (full λ-vs-k curve).
    farmcpu_p_threshold, farmcpu_max_iterations, farmcpu_max_pseudo_qtns,
    farmcpu_final_scan : FarmCPU tuning parameters used only when
        ``model='farmcpu'``. Defaults mirror :func:`run_farmcpu`.
    mlmm_p_enter, mlmm_max_cof : MLMM tuning parameters used only when
        ``model='mlmm'``. Defaults mirror :func:`run_mlmm_research_grade_fast`.
    """
    from gwas.plotting import compute_lambda_gc
    from gwas.kinship import _build_loco_kernels_impl

    model = str(model).lower()
    if model not in {"mlm", "mlmm", "farmcpu"}:
        raise ValueError(
            f"auto_select_pcs: model must be one of 'mlm', 'mlmm', "
            f"'farmcpu'; got {model!r}."
        )

    max_pcs = min(max_pcs, pcs_full.shape[1] if pcs_full is not None else 0)
    pheno_reader = PhenoData(iid=iid, val=y)

    # Edge-tolerant band: a k is "acceptable" if its delta from 1.0
    # falls within (band_edge_delta + parsimony_tolerance). With the
    # defaults [0.95, 1.05] and parsimony=0.02 this is 0.05 + 0.02 = 0.07,
    # so any lambda_GC in [0.93, 1.07] counts as acceptable.
    band_edge_delta = max(band_hi - 1.0, 1.0 - band_lo)
    acceptable_delta = band_edge_delta + parsimony_tolerance

    rows = []
    K0 = None
    K_by_chr = None
    early_stopped = False
    for k in range(0, max_pcs + 1):
        if progress_callback:
            progress_callback(k, max_pcs + 1)

        if early_stopped:
            rows.append({
                "n_pcs": k,
                "lambda_gc": np.nan,
                "delta_from_1": np.nan,
                "status": "skipped",
            })
            continue

        # Build LOCO kernels (reused across PCs since kinship doesn't change)
        if k == 0:
            K0, K_by_chr, _ = _build_loco_kernels_impl(
                iid=iid, Z_grm=Z_grm, chroms_grm=chroms_grm, K_base=K_base,
            )
            if not use_loco:
                K_by_chr = {ch: K0 for ch in K_by_chr}

        # Build per-k PC covariates once (MLMM and FarmCPU expect a
        # CovarData, or None when k == 0).
        if k > 0 and pcs_full is not None:
            _pcs_k = pcs_full[:, :k]
            _covar_k = CovarData(iid=iid, val=_pcs_k)
        else:
            _covar_k = None

        try:
            if model == "mlm":
                gwas_df = _run_gwas_impl(
                    geno_imputed=geno_imputed, y=y, pcs_full=pcs_full,
                    n_pcs=k, sid=sid, positions=positions, chroms=chroms,
                    chroms_num=chroms_num, iid=iid,
                    _K0=K0, _K_by_chr=K_by_chr, pheno_reader=pheno_reader,
                )
                lam = compute_lambda_gc(gwas_df["PValue"].values, trim=False)
            elif model == "mlmm":
                mlmm_df, _ = run_mlmm_research_grade_fast(
                    geno_imputed=geno_imputed, sid=sid, chroms=chroms,
                    chroms_num=chroms_num, positions=positions, iid=iid,
                    pheno_reader=pheno_reader, K0=K0,
                    covar_reader=_covar_k,
                    p_enter=float(mlmm_p_enter),
                    max_cof=int(mlmm_max_cof),
                    verbose=False,
                )
                lam = compute_lambda_gc(mlmm_df["PValue"].values, trim=False)
            else:  # farmcpu
                fc_df, _, _ = run_farmcpu(
                    geno_imputed=geno_imputed, sid=sid, chroms=chroms,
                    chroms_num=chroms_num, positions=positions, iid=iid,
                    pheno_reader=pheno_reader, K0=K0,
                    covar_reader=_covar_k,
                    p_threshold=float(farmcpu_p_threshold),
                    max_iterations=int(farmcpu_max_iterations),
                    max_pseudo_qtns=int(farmcpu_max_pseudo_qtns),
                    final_scan=str(farmcpu_final_scan),
                    verbose=False,
                    use_loco=bool(use_loco),
                )
                lam = compute_lambda_gc(fc_df["PValue"].values, trim=False)
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            logging.exception("%s GWAS failed at k=%d PCs", model.upper(), k)
            lam = np.nan

        rows.append({
            "n_pcs": k,
            "lambda_gc": round(lam, 4) if np.isfinite(lam) else np.nan,
            "delta_from_1": round(abs(lam - 1.0), 4) if np.isfinite(lam) else np.nan,
            "status": "evaluated",
        })

        # Early-stop decision (band strategy only; finite lambda only).
        # A failed GWAS at k=0 (lam = NaN) must NOT fire the deflation
        # guard — the np.isfinite check below handles that.
        if early_stop and strategy == "band" and np.isfinite(lam):
            if k == 0 and lam < band_lo:
                early_stopped = True   # deflation guard
            elif abs(lam - 1.0) <= acceptable_delta:
                early_stopped = True   # smallest acceptable k

    df = pd.DataFrame(rows)

    # Select best PC count based on strategy
    df["recommended"] = ""
    valid = df["lambda_gc"].notna()
    if valid.any():
        if strategy == "band":
            # Edge-tolerant band: pick smallest k whose delta is within
            # acceptable_delta of 0. Parsimony tolerance is applied
            # uniformly across the band edge, so e.g. delta=0.06 and
            # delta=0.05 are treated as noise-equivalent (both
            # acceptable when acceptable_delta=0.07).
            acceptable = df[valid & (df["delta_from_1"] <= acceptable_delta)]
            if not acceptable.empty:
                best_idx = acceptable["n_pcs"].idxmin()
            else:
                # Fallback: closest to 1.0 with adaptive parsimony tolerance.
                # Only fires when every k has delta > acceptable_delta
                # (e.g. severe inflation that PCs can't fully control).
                best_delta = df.loc[valid, "delta_from_1"].min()
                tol = max(parsimony_tolerance, best_delta * 0.15)
                near_best = df[valid & (df["delta_from_1"]
                                        <= best_delta + tol)]
                best_idx = near_best["n_pcs"].idxmin()
            df.loc[best_idx, "recommended"] = "★"

            # --- Directional guard: deflated baseline ---
            # If λ(0) < band_lo the kinship is already over-correcting
            # on its own. Adding PCs cannot fix that — any "recovery"
            # back into band at k > 0 is either coincidence on an
            # oscillating λ curve or artificial inflation from
            # discarding signal. Force k=0 so the user sees (and
            # reports) the deflation instead of burying it under
            # spurious PCs.
            lam0 = df.loc[df["n_pcs"] == 0, "lambda_gc"].iloc[0]
            if pd.notna(lam0) and lam0 < 0.80:
                logging.warning(
                    "Severe lambda_GC deflation (%.3f at k=0). The "
                    "kinship model may be absorbing substantial trait "
                    "signal. Results are valid but interpret cautiously.",
                    lam0,
                )
            if pd.notna(lam0) and lam0 < band_lo:
                df["recommended"] = ""
                df.loc[df["n_pcs"] == 0, "recommended"] = "★"
                logging.warning(
                    "Deflated baseline lambda_GC=%.3f (< %.2f): forced "
                    "recommendation to k=0 (kinship is over-correcting; "
                    "adding PCs cannot repair it).",
                    lam0, band_lo,
                )
        else:
            # closest_to_1: pick PC count with λGC nearest to 1.0
            best_idx = df["delta_from_1"].idxmin()
            df.loc[best_idx, "recommended"] = "★"

    return df


def select_best_pc_from_lambdas(
    lambdas, strategy="band", band_lo=0.95, band_hi=1.05,
    parsimony_tolerance=0.02,
):
    """Pick best PC count from a list of lambda_GC values.

    Applies the same edge-tolerant band rule as :func:`auto_select_pcs`,
    plus the directional deflation guard: if ``lambdas[0] < band_lo``
    the kinship is already over-correcting and adding PCs cannot repair
    it; selection is forced to ``k=0`` so the deflation is surfaced
    rather than buried under spurious PCs that artificially pull
    lambda_GC back into band. This keeps FarmCPU's auto-PC behaviour
    consistent with MLM's on deflated datasets.

    Parameters
    ----------
    lambdas : list[float]
        Lambda GC value for each k = 0, 1, ..., len(lambdas)-1. NaN
        entries (e.g. from skipped k under early-stop, or from failed
        GWAS calls) are treated as invalid and excluded from selection.
    strategy : str
        'band' (edge-tolerant) or 'closest_to_1'.

    Returns
    -------
    int
        Best PC count (index into *lambdas*).
    """
    n = len(lambdas)
    deltas = [abs(v - 1.0) if np.isfinite(v) else np.inf for v in lambdas]
    valid = [np.isfinite(v) for v in lambdas]

    if not any(valid):
        return 0

    if strategy != "band":
        # closest_to_1: pick PC count with lambda nearest to 1.0
        # (guard does not apply, matching auto_select_pcs)
        return int(np.argmin(deltas))

    # Edge-tolerant band: same rule as auto_select_pcs. Acceptable when
    # delta_k <= band_edge_delta + parsimony_tolerance.
    band_edge_delta = max(band_hi - 1.0, 1.0 - band_lo)
    acceptable_delta = band_edge_delta + parsimony_tolerance

    def _pick(indices):
        """Acceptable-delta then fallback within a set of candidate indices."""
        acceptable = [
            i for i in indices
            if valid[i] and deltas[i] <= acceptable_delta
        ]
        if acceptable:
            return min(acceptable)
        valid_in = [i for i in indices if valid[i]]
        if not valid_in:
            return indices[0] if indices else 0
        best_delta = min(deltas[i] for i in valid_in)
        tol = max(parsimony_tolerance, best_delta * 0.15)
        near = [i for i in valid_in if deltas[i] <= best_delta + tol]
        return min(near) if near else min(valid_in, key=lambda i: deltas[i])

    best = _pick(list(range(n)))

    # --- Directional deflation guard (mirrors auto_select_pcs) ---
    # If lambda(0) < band_lo, the kinship is already over-correcting.
    # Any in-band recovery at k > 0 on an oscillating lambda curve is
    # noise, not a genuine fix. Force k=0 unconditionally so deflation
    # is surfaced rather than buried under spurious PCs.
    if valid[0] and lambdas[0] < band_lo:
        if best != 0:
            best = 0
            logging.warning(
                "FarmCPU auto-PC: deflated baseline lambda_GC=%.3f "
                "(< %.2f); forced k=0 (adding PCs cannot repair an "
                "over-correcting kinship).",
                lambdas[0], band_lo,
            )

    return int(best)
