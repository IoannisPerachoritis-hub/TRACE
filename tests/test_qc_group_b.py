"""Group B QC-overhaul tests: P6 non-autosome disclosure, P8 allopolyploid
chromosome naming, P9 fail-loud on polyploid dosage, P12 dead-branch removal."""
import numpy as np
import pandas as pd
import pytest

from gwas.io import _clean_chr_series
from gwas.qc import allele_freq_from_called_dosage, _pipeline_snp_qc


# ── P8 ──────────────────────────────────────────────────────────────────────
def test_p8_wheat_subgenomes_are_distinct_groups():
    lab, num, uniq, _ = _clean_chr_series([f"{n}{L}" for n in range(1, 8) for L in "ABD"])
    nonalt = [u for u in uniq if u != "ALT"]
    assert len(nonalt) == 21                              # 7 x {A,B,D}
    assert set(lab) == {f"{n}{L}" for n in range(1, 8) for L in "ABD"}
    assert len(set(num)) == 21                            # distinct codes -> distinct LOCO


def test_p8_chr_prefix_subgenome():
    lab, _, _, _ = _clean_chr_series(["chr1A", "1B", "1D"])
    assert list(lab) == ["1A", "1B", "1D"]


def test_p8_barley_and_normal_panels_unchanged():
    lab, _, _, _ = _clean_chr_series(["1H", "2H", "7H"])
    assert list(lab) == ["1", "2", "7"]                  # barley trailing-H preserved
    lab2, num2, _, _ = _clean_chr_series(["1", "2", "10"])
    assert list(lab2) == ["1", "2", "10"] and list(num2) == [1, 2, 10]


# ── P9 ──────────────────────────────────────────────────────────────────────
def test_p9_polyploid_dosage_raises():
    with pytest.raises(ValueError, match="dosage > 2"):
        allele_freq_from_called_dosage(np.array([[3.0], [1.0], [2.0]]))


def test_p9_float_noise_near_two_is_clipped_not_raised():
    p = allele_freq_from_called_dosage(np.array([[2.0001], [-0.0001], [1.0]]))
    assert 0.0 <= p[0] <= 1.0


# ── P6 / P12 (via the SNP-QC pipeline) ──────────────────────────────────────
def _snp_inputs(chroms, n=20):
    m = len(chroms)
    rng = np.random.default_rng(0)
    G = rng.integers(0, 3, size=(n, m)).astype(float)
    geno_df = pd.DataFrame(G, index=[f"S{i}" for i in range(n)],
                           columns=[f"snp{j}" for j in range(m)])
    positions = (np.arange(1, m + 1) * 1000)
    sid = np.array([f"snp{j}" for j in range(m)])
    return geno_df, np.asarray(chroms), positions, sid


def test_p6_nonautosome_and_scaffold_counts():
    geno_df, chroms, pos, sid = _snp_inputs(["1", "1", "chrX", "0", "0"])
    (_g, _chr, _cn, _pos, _sid, _n, qc_snp, _info) = _pipeline_snp_qc(
        geno_df, chroms, pos, sid, maf_thresh=0.0, miss_thresh=1.0,
        mac_thresh=0, drop_alt=False)
    assert qc_snp["Non-autosomal markers"] == 1      # chrX
    assert qc_snp["Unplaced/scaffold markers"] == 2  # two chr-0 scaffolds
    assert qc_snp["ALT chromosomes"] == 3            # existing count unchanged in kind


def test_p12_all_alt_still_raises_outer_guard():
    geno_df, chroms, pos, sid = _snp_inputs(["scaff1", "scaff2", "contigA"])
    with pytest.raises(ValueError, match="recognized"):
        _pipeline_snp_qc(geno_df, chroms, pos, sid, maf_thresh=0.0,
                         miss_thresh=1.0, mac_thresh=0, drop_alt=True)
