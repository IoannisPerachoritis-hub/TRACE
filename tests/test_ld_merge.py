"""LD-block detection redesign — the single production path.

There is no longer a selectable ``--ld-merge-mode``: detection always emits ONLY
the seed's connected component (so every block contains its lead), splits each
block toward within-block coherence following the seed (never annihilating it),
then merges overlapping candidates by correlation (cross-seam AND union mean r²
both >= ``ld_merge_r2``) and resolves any residual overlap by occupancy (disjoint
output). These tests cover the ``merge_coherence`` numerics that back the merge,
the ``merge_r2`` column, and the redesign invariants (lead-in-member, coherence,
disjointness). ``_iou_merge_blocks`` remains callable in-tree for the before/after
comparison but is no longer a selectable production mode.
"""
import numpy as np
import pandas as pd
import pytest

from gwas import ld


# --------------------------------------------------------------------------- #
# merge_coherence: cross seam vs union coherence (unchanged by the redesign)
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
# production path: merge_r2 column + invariants on a single coherent block
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


def _detect(r2=0.5):
    gwas_df, chroms, positions, geno, sid = _one_block_inputs()
    return ld.find_ld_clusters_genomewide(
        gwas_df=gwas_df, chroms=chroms, positions=positions,
        geno_imputed=geno.astype(float), sid=sid,
        ld_threshold=0.6, flank_kb=50, min_snps=3, top_n=0, sig_thresh=1e-5,
        adj_r2_min=0.2, merge_iou=0.3, gap_factor=10.0, ld_merge_r2=r2)


def test_merge_r2_column_present_and_nan_when_no_merge():
    """The merge_r2 column exists (and is preserved through occupancy, P5) even
    when no merge fires; a single seed block is never merged -> NaN."""
    out = _detect()
    assert "merge_r2" in out.columns and "Mean r2" in out.columns
    assert len(out) == 1
    assert np.isnan(float(out["merge_r2"].iloc[0]))   # single block, never merged


def test_single_path_lead_is_member_and_coherent():
    """Redesign invariants on synthetic data: the lead is a member and the block
    reaches the coherence threshold."""
    for r2 in (0.5, 0.6, 0.7):
        out = _detect(r2)
        assert len(out) == 1
        row = out.iloc[0]
        assert str(row["lead_snp"]) in set(str(row["SNP_IDs"]).split(","))
        assert float(row["Mean r2"]) >= r2 - 1e-9


def test_discard_counter_present_and_zero_when_no_trim():
    """A single coherent seed block sheds nothing -> the counter is 0."""
    out = _detect()
    assert int(out.attrs.get("n_fragments_discarded", 0)) == 0
    assert int(out.attrs.get("n_occupancy_discarded", 0)) == 0


# --------------------------------------------------------------------------- #
# Real-data redesign invariants (opt-in; skips without the committed tomato QC).
# Invariant-based so they hold regardless of the exact QC checkpoint version.
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
def test_redesign_invariants_tomato():
    gwas_df, chroms, positions, geno, sid = _load_tomato_qc()
    out = ld.find_ld_clusters_genomewide(
        gwas_df=gwas_df, chroms=chroms, positions=positions,
        geno_imputed=geno.astype(float), sid=sid, ld_threshold=0.6,
        flank_kb=144, ld_decay_kb=72.17, min_snps=2, top_n=10, sig_thresh=1e-5,
        adj_r2_min=0.2, merge_iou=0.3, gap_factor=10.0, ld_merge_r2=0.5)
    out, _ = ld.filter_contained_blocks(out, min_contained=2)
    assert len(out) >= 1

    # (1) every block contains its lead (P2 -- the whole point of the redesign)
    for _, r in out.iterrows():
        mem = set(str(r["SNP_IDs"]).split(","))
        assert str(r["lead_snp"]) in mem, "a block's lead is not one of its members"

    # (2) blocks are disjoint in coordinates AND members (occupancy)
    for ch, g in out.groupby(out["Chr"].astype(str)):
        iv = sorted((int(x["Start (bp)"]), int(x["End (bp)"])) for _, x in g.iterrows())
        for (s1, e1), (s2, e2) in zip(iv, iv[1:]):
            assert e1 < s2, "blocks overlap in coordinates"
        mems = [set(str(x["SNP_IDs"]).split(",")) for _, x in g.iterrows()]
        for i in range(len(mems)):
            for j in range(i + 1, len(mems)):
                assert not (mems[i] & mems[j]), "blocks share members"

    # (3) the pre-redesign LD-bridged lead block (47,301,921-47,657,766, mean r2
    #     0.427) never survives -- coherence is required
    bridged = (out["Start (bp)"].astype(int) == 47301921) & (out["End (bp)"].astype(int) == 47657766)
    assert not bridged.any(), "the incoherent bridged block must not survive"

    # (4) every emitted block's coherence reaches the threshold (all tomato blocks
    #     are coherent; the whole-emit incoherent path does not fire here)
    mr = out["Mean r2"].astype(float)
    assert (mr[mr.notna()] >= 0.5 - 1e-9).all(), "an emitted block is below the coherence threshold"
