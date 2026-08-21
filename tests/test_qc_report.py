"""Tests for gwas/qc_report.py — report-only QC statistics P1-P5.

Every function must be report-only: it may compute and summarise, but it must
never drop a sample, marker, or trait.  These tests enforce that invariant and
check the arithmetic against brute-force references.
"""
import numpy as np
import pandas as pd
import pytest

from gwas import qc_report as qr


@pytest.fixture
def toy_geno():
    rng = np.random.default_rng(0)
    n, m = 20, 200
    G = rng.integers(0, 3, size=(n, m)).astype(float)
    G[1] = G[0].copy()               # S1 = near-duplicate of S0 ...
    G[1, :2] = (G[0, :2] + 1) % 3    # ... differing at exactly 2 markers (guaranteed)
    G[2, :] = 1.0               # S2 = all-heterozygous outlier
    G[3, :10] = np.nan          # S3 has some missing calls
    return pd.DataFrame(G, index=[f"S{i}" for i in range(n)])


def test_p1_flags_het_outlier_without_removing(toy_geno):
    h = qr.per_sample_heterozygosity(toy_geno)
    assert "S2" in [s["sample"] for s in h["flagged_samples"]]  # all-het flagged
    assert h["per_sample"].shape[0] == 20                       # nothing removed
    assert h["n_flagged"] == len(h["flagged_samples"])


def test_p2_finds_the_duplicate_pair(toy_geno):
    d = qr.duplicate_pairs(toy_geno, conc_thresh=0.9)
    pairs = {frozenset((p["sample_a"], p["sample_b"])) for p in d["pairs"]}
    assert frozenset(("S0", "S1")) in pairs
    s01 = [p for p in d["pairs"] if {p["sample_a"], p["sample_b"]} == {"S0", "S1"}][0]
    assert s01["n_markers_differ"] == 2
    assert d["n_pairs_scanned"] == 20 * 19 // 2


def test_p2_concordance_matches_bruteforce(toy_geno):
    G = toy_geno.to_numpy()
    both = np.isfinite(G[0]) & np.isfinite(G[1])
    manual = 1.0 - (G[0][both] != G[1][both]).sum() / both.sum()
    d = qr.duplicate_pairs(toy_geno, conc_thresh=0.9)
    s01 = [p for p in d["pairs"] if {p["sample_a"], p["sample_b"]} == {"S0", "S1"}][0]
    assert abs(s01["concordance"] - manual) < 1e-4


def test_p3_fis_arithmetic(toy_geno):
    f = qr.per_variant_fis(toy_geno)
    assert f["n_markers"] == 200                    # nothing removed
    assert np.isfinite(f["median_fis"])
    # F_IS = 1 - Ho/He, guarded at He==0 (no inf/nan leakage into the summary)
    assert np.isfinite([f["pct5_fis"], f["pct95_fis"]]).all()


def test_p4_trait_summary_missing_and_distinct():
    t = qr.trait_summary([1.0, 2.0, 3.0, np.nan, 5.0], "yield")
    assert t["n"] == 4
    assert t["n_distinct"] == 4
    assert abs(t["pct_missing"] - 20.0) < 1e-6


def test_p5_report_structure_and_render(toy_geno):
    rep = qr.compute_qc_report(toy_geno, [1.0] * 20, "trait", qc_snp={"Total SNPs": 200})
    assert {"sample_het", "dup_pairs", "variant_fis", "trait", "qc_snp"} <= set(rep)
    dfs = qr.qc_report_dataframes(rep)
    assert "QC_report_summary.csv" in dfs
    assert "QC_sample_heterozygosity.csv" in dfs
    md = qr.render_qc_report_markdown(rep)
    assert "QC Report" in md and "F_IS" in md


def test_report_only_never_filters(toy_geno):
    rep = qr.compute_qc_report(toy_geno, [1.0] * 20, "t")
    assert rep["n_samples"] == 20 and rep["n_markers"] == 200
