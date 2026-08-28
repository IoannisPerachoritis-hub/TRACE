"""Correlation-based LD-block merge criterion (--ld-merge-mode correlation).

The default (iou) path is byte-identical (covered by the golden suite); these
tests cover the opt-in correlation path: the `merge_coherence` numerics, the new
`merge_r2` column, and the emitted-block invariant `Mean r2 >= ld_merge_r2`.
"""
import numpy as np
import pandas as pd
import pytest

from gwas import ld


# --------------------------------------------------------------------------- #
# merge_coherence: cross seam vs union coherence
# --------------------------------------------------------------------------- #
def _two_cluster_panel(seed=0):
    """Two internally-correlated clusters (A, B) that are ~independent of each other."""
    rng = np.random.default_rng(seed)
    n = 200
    base_a = rng.normal(size=n)
    base_b = rng.normal(size=n)
    cols = [base_a + 0.05 * rng.normal(size=n) for _ in range(3)]
    cols += [base_b + 0.05 * rng.normal(size=n) for _ in range(3)]
    geno = np.column_stack(cols)
    sid = np.array(["a0", "a1", "a2", "b0", "b1", "b2"])
    chroms = np.array(["1"] * 6)
    positions = np.array([1000, 2000, 3000, 4000, 5000, 6000])
    return geno, sid, chroms, positions


def test_merge_coherence_cross_below_union():
    geno, sid, chroms, positions = _two_cluster_panel()
    A = {"Chr": "1", "Start (bp)": 1000, "End (bp)": 3000, "SNP_IDs": "a0,a1,a2"}
    B = {"Chr": "1", "Start (bp)": 4000, "End (bp)": 6000, "SNP_IDs": "b0,b1,b2"}
    cross, union = ld.merge_coherence(A, B, chroms, positions, sid, geno)
    assert np.isfinite(cross) and np.isfinite(union)
    # cross pairs are the weak A-B links; the union average is pulled up by the
    # strong within-cluster pairs, so cross < union.
    assert cross < union
    assert cross < 0.2 and union > 0.4


def test_merge_coherence_shared_member_still_measurable():
    """A fully-contained block must yield a measurable seam, not an empty cross set."""
    geno, sid, chroms, positions = _two_cluster_panel()
    A = {"Chr": "1", "Start (bp)": 1000, "End (bp)": 4000, "SNP_IDs": "a0,a1,a2,b0"}
    B = {"Chr": "1", "Start (bp)": 4000, "End (bp)": 6000, "SNP_IDs": "b0,b1,b2"}
    cross, union = ld.merge_coherence(A, B, chroms, positions, sid, geno)
    assert np.isfinite(cross)   # symmetric mask -> non-empty even with a shared member
    assert np.isfinite(union)


def test_merge_coherence_degenerate_union_is_nan():
    geno, sid, chroms, positions = _two_cluster_panel()
    A = {"Chr": "1", "Start (bp)": 1000, "End (bp)": 1000, "SNP_IDs": "a0"}
    B = {"Chr": "1", "Start (bp)": 1000, "End (bp)": 1000, "SNP_IDs": "a0"}
    cross, union = ld.merge_coherence(A, B, chroms, positions, sid, geno)
    assert np.isnan(cross) and np.isnan(union)   # |union| < 2 -> (nan, nan), no raise


def test_merge_coherence_monomorphic_dropped():
    geno, sid, chroms, positions = _two_cluster_panel()
    geno = np.column_stack([geno, np.ones(geno.shape[0])])   # a constant marker
    sid = np.append(sid, "mono")
    chroms = np.append(chroms, "1")
    positions = np.append(positions, 3500)
    A = {"Chr": "1", "Start (bp)": 1000, "End (bp)": 3500, "SNP_IDs": "a0,a1,a2,mono"}
    B = {"Chr": "1", "Start (bp)": 4000, "End (bp)": 6000, "SNP_IDs": "b0,b1,b2"}
    cross, union = ld.merge_coherence(A, B, chroms, positions, sid, geno)
    assert np.isfinite(cross) and np.isfinite(union)   # monomorphic marker dropped, no raise


# --------------------------------------------------------------------------- #
# find_ld_clusters_genomewide: merge_r2 column + invariant on a single coherent block
# --------------------------------------------------------------------------- #
def _one_block_inputs(seed=1):
    rng = np.random.default_rng(seed)
    n = 200
    base = rng.normal(size=n)
    geno = np.column_stack([base + 0.05 * rng.normal(size=n) for _ in range(3)])
    sid = np.array(["s0", "s1", "s2"])
    chroms = np.array(["1"] * 3)
    positions = np.array([1000, 1100, 1200])
    gwas_df = pd.DataFrame({
        "SNP": sid, "Chr": ["1"] * 3, "Pos": positions,
        "PValue": [1e-8, 0.5, 0.5],
    })
    return gwas_df, chroms, positions, geno, sid


def _detect(mode="iou", r2=0.5):
    gwas_df, chroms, positions, geno, sid = _one_block_inputs()
    return ld.find_ld_clusters_genomewide(
        gwas_df=gwas_df, chroms=chroms, positions=positions,
        geno_imputed=geno.astype(float), sid=sid,
        ld_threshold=0.6, flank_kb=50, min_snps=3, top_n=0, sig_thresh=1e-5,
        adj_r2_min=0.2, merge_iou=0.3, gap_factor=10.0,
        ld_merge_mode=mode, ld_merge_r2=r2)


def test_merge_r2_column_present_and_nan_when_no_merge():
    """B1: the merge_r2 column exists even when no merge fires (else KeyError)."""
    out = _detect(mode="iou")
    assert "merge_r2" in out.columns and "Mean r2" in out.columns
    assert len(out) == 1
    assert np.isnan(float(out["merge_r2"].iloc[0]))   # single block, never merged


def test_default_mode_is_iou():
    gwas_df, chroms, positions, geno, sid = _one_block_inputs()
    default = ld.find_ld_clusters_genomewide(
        gwas_df=gwas_df, chroms=chroms, positions=positions,
        geno_imputed=geno.astype(float), sid=sid,
        ld_threshold=0.6, flank_kb=50, min_snps=3, top_n=0, sig_thresh=1e-5)
    explicit = _detect(mode="iou")
    pd.testing.assert_frame_equal(default.reset_index(drop=True),
                                  explicit.reset_index(drop=True))


def test_correlation_mode_runs_and_invariant_synth():
    """Correlation mode does not crash and every emitted block is coherent."""
    for r2 in (0.5, 0.6, 0.7):
        out = _detect(mode="correlation", r2=r2)
        mr = out["Mean r2"].astype(float)
        assert (mr[mr.notna()] >= r2 - 1e-9).all()


# --------------------------------------------------------------------------- #
# Real-data invariant + split (opt-in; skips without the committed tomato QC)
# --------------------------------------------------------------------------- #
def _load_tomato_qc():
    from pathlib import Path
    from annotation import canon_chr
    qc = Path(__file__).resolve().parents[1] / "benchmarks" / "qc_data" / "tomato_locule_number"
    if not (qc / "QC_genotype_matrix.csv").exists():
        pytest.skip("tomato QC data not present")
    geno_df = pd.read_csv(qc / "QC_genotype_matrix.csv", index_col=0)
    snp_map = pd.read_csv(qc / "QC_snp_map.csv")
    gwas_df = pd.read_csv(qc / "platform_GWAS_locule_number.csv")
    geno = geno_df.values.astype(np.float64)
    cm = np.nanmean(geno, axis=0); nm = np.isnan(geno)
    if nm.any():
        geno[nm] = np.take(cm, np.where(nm)[1])
    sid = snp_map["SNP_ID"].values
    chroms = np.array([str(canon_chr(c)) for c in snp_map["Chr"].values])
    positions = snp_map["Pos"].values.astype(int)
    return gwas_df, chroms, positions, geno, sid


@pytest.mark.golden
def test_correlation_invariant_and_splits_bridged_block_tomato():
    gwas_df, chroms, positions, geno, sid = _load_tomato_qc()

    def detect(mode, r2=0.5):
        b = ld.find_ld_clusters_genomewide(
            gwas_df=gwas_df, chroms=chroms, positions=positions,
            geno_imputed=geno.astype(float), sid=sid,
            ld_threshold=0.6, flank_kb=144, ld_decay_kb=72.17, min_snps=3,
            top_n=10, sig_thresh=1e-5, adj_r2_min=0.2, merge_iou=0.3, gap_factor=10.0,
            ld_merge_mode=mode, ld_merge_r2=r2)
        b, _ = ld.filter_contained_blocks(b, min_contained=2)
        return b

    iou = detect("iou")
    # The published Table S8 lead block (mean r2 = 0.427, an LD-bridged union).
    lead = (iou["Start (bp)"].astype(int) == 47301921) & (iou["End (bp)"].astype(int) == 47657766)
    assert lead.any(), "iou mode must reproduce the published bridged lead block"

    for r2 in (0.5, 0.6, 0.7):
        corr = detect("correlation", r2)
        mr = corr["Mean r2"].astype(float)
        assert (mr[mr.notna()] >= r2 - 1e-9).all(), f"invariant violated at r2={r2}"
        # the 0.427 bridged block must be gone (split) once coherence is required
        still = (corr["Start (bp)"].astype(int) == 47301921) & (corr["End (bp)"].astype(int) == 47657766)
        assert not still.any(), f"bridged lead block survived at r2={r2}"


# --------------------------------------------------------------------------- #
# discard counter (segments dropped below min_snps under correlation merging)
# --------------------------------------------------------------------------- #
def test_discard_counter_zero_when_no_drop():
    """The counter is 0 in iou mode and when correlation splits nothing."""
    iou = _detect(mode="iou")
    assert int(iou.attrs.get("n_fragments_discarded", 0)) == 0
    corr = _detect(mode="correlation", r2=0.5)   # single coherent block, no split
    assert int(corr.attrs.get("n_fragments_discarded", 0)) == 0


@pytest.mark.golden
def test_discard_counter_and_log_tomato_r07(caplog):
    import logging
    gwas_df, chroms, positions, geno, sid = _load_tomato_qc()
    with caplog.at_level(logging.INFO, logger="gwas.ld"):
        out = ld.find_ld_clusters_genomewide(
            gwas_df=gwas_df, chroms=chroms, positions=positions,
            geno_imputed=geno.astype(float), sid=sid, ld_threshold=0.6,
            flank_kb=144, ld_decay_kb=72.17, min_snps=3, top_n=10, sig_thresh=1e-5,
            adj_r2_min=0.2, merge_iou=0.3, gap_factor=10.0,
            ld_merge_mode="correlation", ld_merge_r2=0.7)
    # 40 sub-min_snps fragments dropped at r2=0.7 (measured; the usable-range warning)
    assert int(out.attrs["n_fragments_discarded"]) == 40
    assert any("dropped 40 fragment" in r.message for r in caplog.records)
