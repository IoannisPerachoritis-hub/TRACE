"""Reproducibility pins for the LOCO / GRM-pruning cross-tool concordance (D-76).

Two tiers:
  * data-FREE guards (resolve_k, the k=0-bug guard, the committed CSV values) run everywhere,
    including CI and the TRACE-release tree.
  * data-DEPENDENT pins (recompute rho / reproduce the pruner from on-disk p-values) SKIP where
    the upstream inputs are absent -- qc_data/, results/{gapit,rmvp}/, and loco_sensitivity/*.zip
    are git-ignored, so these skip on a fresh clone / in CI / in TRACE-release, and run only in a
    working tree that has the benchmark data (DEV). They are marked ``golden`` per the repo
    convention and additionally guarded inline so they SKIP (never pass vacuously) when data is
    absent -- TRACE-release CI runs all tests with no ``-m 'not golden'`` deselection.
"""
import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LOCO = REPO / "benchmarks" / "loco_sensitivity"
CSV = LOCO / "loco_comparison.csv"
QC = REPO / "benchmarks" / "qc_data"
GAPIT = REPO / "benchmarks" / "results" / "gapit"
RMVP = REPO / "benchmarks" / "results" / "rmvp"

# The 8 published baselines the LOCO / GRM-pruning validation gate named (Platform-{LOCO,Global} x
# {GAPIT3,rMVP} on the two validation panels). From loco_comparison.csv rows 3-6 / 15-18.
EXPECTED_BASELINES = {
    ("tomato_locule_number", "Platform-LOCO", "GAPIT3"): 0.6335,
    ("tomato_locule_number", "Platform-LOCO", "rMVP"): 0.7173,
    ("tomato_locule_number", "Platform-Global", "GAPIT3"): 0.7525,
    ("tomato_locule_number", "Platform-Global", "rMVP"): 0.8534,
    ("pepper_FWe", "Platform-LOCO", "GAPIT3"): 0.8750,
    ("pepper_FWe", "Platform-LOCO", "rMVP"): 0.8287,
    ("pepper_FWe", "Platform-Global", "GAPIT3"): 0.9145,
    ("pepper_FWe", "Platform-Global", "rMVP"): 0.8395,
}

# The 16 appended ablation rows, from decisions.md D-76 (Platform-{LOCO,Global}-unpruned).
EXPECTED_ABLATION = {
    ("tomato_locule_number", "Platform-LOCO-unpruned", "GAPIT3"): 0.6622,
    ("tomato_locule_number", "Platform-LOCO-unpruned", "rMVP"): 0.7213,
    ("tomato_locule_number", "Platform-Global-unpruned", "GAPIT3"): 0.8501,
    ("tomato_locule_number", "Platform-Global-unpruned", "rMVP"): 0.9532,
    ("tomato_weight_g", "Platform-LOCO-unpruned", "GAPIT3"): 0.6229,
    ("tomato_weight_g", "Platform-LOCO-unpruned", "rMVP"): 0.6545,
    ("tomato_weight_g", "Platform-Global-unpruned", "GAPIT3"): 0.8152,
    ("tomato_weight_g", "Platform-Global-unpruned", "rMVP"): 0.9462,
    ("pepper_FWe", "Platform-LOCO-unpruned", "GAPIT3"): 0.9156,
    ("pepper_FWe", "Platform-LOCO-unpruned", "rMVP"): 0.8599,
    ("pepper_FWe", "Platform-Global-unpruned", "GAPIT3"): 0.9626,
    ("pepper_FWe", "Platform-Global-unpruned", "rMVP"): 0.8730,
    ("pepper_BX", "Platform-LOCO-unpruned", "GAPIT3"): 0.8937,
    ("pepper_BX", "Platform-LOCO-unpruned", "rMVP"): 0.8955,
    ("pepper_BX", "Platform-Global-unpruned", "GAPIT3"): 0.9621,
    ("pepper_BX", "Platform-Global-unpruned", "rMVP"): 0.9642,
}


def _data_ready():
    """All four upstream p-value sources + the CSV present (else the recompute tests SKIP)."""
    return (CSV.exists() and QC.exists() and GAPIT.exists() and RMVP.exists()
            and (LOCO / "tomato_locule_number").exists())


def _load_module(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------
# Data-free guards (run everywhere -- CI, fresh clone, TRACE-release)
# ----------------------------------------------------------------------------

def test_resolve_k_published_values():
    from benchmarks.published_pc_counts import PUBLISHED_K, resolve_k
    assert PUBLISHED_K == {
        "tomato_locule_number": 0, "tomato_weight_g": 0,
        "pepper_BX": 0, "pepper_FWe": 2,
    }
    assert resolve_k("pepper_FWe") == 2       # band k=2, the case that broke first
    assert resolve_k("tomato_weight_g") == 0


def test_resolve_k_raises_on_unknown_panel():
    from benchmarks.published_pc_counts import resolve_k
    with pytest.raises(ValueError):
        resolve_k("some_unlisted_panel")


def test_rerun_global_gwas_has_no_silent_default_k():
    """The k=0 bug fix: k is resolved per panel, no hardcoded module-global default."""
    src = (REPO / "benchmarks" / "rerun_global_gwas.py").read_text(encoding="utf-8")
    assert "resolve_k(" in src, "rerun_global_gwas must resolve k per panel"
    assert re.search(r"^\s*N_PCS\s*=", src, re.M) is None, "hardcoded N_PCS default must be gone"


def test_committed_csv_pins_all_rows():
    """The committed loco_comparison.csv carries the expected baseline + ablation rho (value pin)."""
    if not CSV.exists():
        pytest.skip("loco_comparison.csv absent")
    import pandas as pd
    df = pd.read_csv(CSV)
    idx = {(r.dataset, r.tool_1, r.tool_2): r.spearman_rho for r in df.itertuples(index=False)}
    for key, val in {**EXPECTED_BASELINES, **EXPECTED_ABLATION}.items():
        assert key in idx, f"loco_comparison.csv missing row {key}"
        assert abs(idx[key] - val) < 1e-4, f"{key}: {idx[key]} != {val}"


# ----------------------------------------------------------------------------
# Data-dependent pins (SKIP where inputs are gitignored/absent -- e.g. TRACE-release)
# ----------------------------------------------------------------------------

@pytest.mark.golden
def test_baselines_reproduce_from_ondisk_pvalues():
    """The rho pipeline reproduces the 8 published baselines from the on-disk p-values."""
    if not _data_ready():
        pytest.skip("loco input p-values absent (gitignored: qc_data/results/loco_sensitivity)")
    from benchmarks.loco_sensitivity.compare_loco import (
        compare_tools, load_platform_loco_results, load_platform_global_results,
        load_gapit_results, load_rmvp_results,
    )
    recomputed = {}
    for ds in ["tomato_locule_number", "tomato_weight_g", "pepper_FWe", "pepper_BX"]:
        pairs = compare_tools(
            ds, load_platform_loco_results(ds), load_platform_global_results(ds),
            load_gapit_results(ds), load_rmvp_results(ds),
        ) or []
        for p in pairs:
            recomputed[(p["dataset"], p["tool_1"], p["tool_2"])] = p["spearman_rho"]
    for key, val in EXPECTED_BASELINES.items():
        assert key in recomputed, f"rho pipeline did not recompute {key}"
        assert abs(recomputed[key] - val) < 1e-4, f"rho drift {key}: {recomputed[key]} != {val}"


@pytest.mark.golden
def test_pruned_replication_reproduces_published():
    """The replicated pruned LOCO run reproduces the on-disk published p-values (rho -> 1.000)."""
    if not _data_ready():
        pytest.skip("loco input data absent (gitignored)")
    pa = _load_module("prune_ablation", "benchmarks/prune_ablation.py")
    from benchmarks.loco_sensitivity.compare_loco import load_platform_loco_results
    from benchmarks.published_pc_counts import resolve_k
    mine, _ = pa.run_condition(
        "tomato_locule_number", r2=0.2, loco=True, n_pcs=resolve_k("tomato_locule_number"),
    )
    rho, _ = pa._rho(mine, pa._sp(load_platform_loco_results("tomato_locule_number")))
    assert rho is not None and rho >= 0.999, f"pruner drift: rho={rho}"
