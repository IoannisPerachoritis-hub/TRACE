"""T-31 — Layer 2 (MLG fragmentation + sample retention) acceptance.

The pure unit tests always run; the invariant checks over the real MLG partition
are golden-marked (need benchmarks/qc_data/, gitignored) and skip cleanly.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gwas.haplotype import _inv_simpson_from_counts

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "golden" / "tomato_locule"
QC = REPO / "benchmarks" / "qc_data" / "tomato_locule_number"

MLG_COLS = ["mlg_n_observed", "mlg_top1_freq", "mlg_eff_inv_simpson", "mlg_n_other",
            "mlg_frac_retained", "mlg_min_group_n", "n_samples_tested"]


# ---- pure unit: inverse Simpson (acceptance T-31.4) ----
def test_inv_simpson_single_uniform_empty():
    assert abs(_inv_simpson_from_counts(pd.Series([120])) - 1.0) < 1e-10
    for k in (2, 3, 5):
        assert abs(_inv_simpson_from_counts(pd.Series([10] * k)) - k) < 1e-10
    assert np.isnan(_inv_simpson_from_counts(pd.Series([], dtype=float)))
    assert np.isnan(_inv_simpson_from_counts(pd.Series(dtype=float)))


# ---- integration on the golden qc_data ----
def _ready():
    return GOLDEN.exists() and QC.exists() and (GOLDEN / "run_manifest.json").exists()


@pytest.mark.golden
@pytest.mark.skipif(not _ready(), reason="benchmarks/qc_data absent")
def test_layer2_columns_and_invariants():
    manifest = json.loads((GOLDEN / "run_manifest.json").read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("capture_golden", REPO / "benchmarks" / "capture_golden.py")
    cap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cap)

    geno, geno_df, pheno, sid, chroms, positions = cap._load_qc(QC)
    gwas_df = pd.read_csv(QC / "platform_GWAS_locule_number.csv")
    blocks = cap.ld.find_ld_clusters_genomewide(
        gwas_df=gwas_df, chroms=chroms, positions=positions, geno_imputed=geno.astype(float), sid=sid,
        ld_threshold=0.6, flank_kb=manifest["flank_kb"], ld_decay_kb=manifest.get("ld_decay_kb"),
        min_snps=3, top_n=10, sig_thresh=1e-5, adj_r2_min=0.2, merge_iou=0.3, gap_factor=10.0)
    blocks, _ = cap.ld.filter_contained_blocks(blocks, min_contained=2)
    haplo_df = cap._rename_start_end(blocks.copy())
    df_res, _ = cap.run_haplotype_block_gwas(
        haplo_df=haplo_df, chroms=chroms, positions=positions, geno_imputed=geno.astype(float), sid=sid,
        geno_df=geno_df, pheno_df=pheno, trait_col=manifest.get("trait_col", "locule_number"),
        pcs=None, n_perm=manifest.get("n_perm", 1000), n_pcs_used=0, min_hap_count=5, min_group_size=3)

    assert not df_res.empty
    for c in MLG_COLS:
        assert c in df_res.columns, f"Layer-2 column missing: {c}"

    # T-31.2 — mlg_frac_retained == n_samples_tested / n_samples_block, exactly (k=0 PCs, no PC drop)
    assert np.allclose(df_res["mlg_frac_retained"] * df_res["n_samples_block"], df_res["n_samples_tested"])
    # N7 / T-31.5 — every tested group at/above min_group_size; at least two tested groups
    assert (df_res["mlg_min_group_n"] >= 3).all()
    assert (df_res["n_tested_haplotypes"] >= 2).all()
    # inverse Simpson is a finite diversity >= 1
    assert np.isfinite(df_res["mlg_eff_inv_simpson"]).all()
    assert (df_res["mlg_eff_inv_simpson"] >= 1.0 - 1e-9).all()
    # mlg_n_observed matches the pre-collapse haplotype count already reported
    assert (df_res["mlg_n_observed"] == df_res["n_haplotypes"]).all()
