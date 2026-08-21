"""Group D QC-overhaul tests: P13 heterozygosity screen -- one-sided heterozygote
EXCESS only, report-always, filter only on --max-het / --het-excess-p (both off by
default). Heterozygote DEFICIT (expected in selfers) is never screened."""
import numpy as np
import pandas as pd

from gwas.qc import _pipeline_snp_qc


def _het_inputs():
    """3 variants, all MAF 0.5: an all-heterozygous variant (Ho=1.0, extreme
    excess), a homozygous-split variant (Ho=0.0, extreme deficit), and a
    moderate one (Ho=0.4)."""
    n = 20
    het = np.ones(n)                                 # Ho = 1.0
    hom = np.array([0.0] * 10 + [2.0] * 10)          # Ho = 0.0 (deficit)
    mid = np.array([1.0] * 8 + [0.0] * 6 + [2.0] * 6)  # Ho = 0.4
    G = np.column_stack([het, hom, mid])
    geno_df = pd.DataFrame(G, index=[f"S{i}" for i in range(n)],
                           columns=["snp0_het", "snp1_hom", "snp2_mid"])
    chroms = np.array(["1", "1", "1"])
    positions = np.array([1000, 2000, 3000])
    sid = np.array(["snp0_het", "snp1_hom", "snp2_mid"])
    return geno_df, chroms, positions, sid


def _run(geno_df, chroms, pos, sid, **kw):
    (g, _chr, _cn, _pos, _sid, _n, qc, _info) = _pipeline_snp_qc(
        geno_df.copy(), chroms.copy(), pos.copy(), sid.copy(),
        maf_thresh=0.0, miss_thresh=1.0, mac_thresh=0, drop_alt=False, **kw)
    return g, qc


def test_p13_off_reports_median_het_but_filters_nothing():
    geno_df, chroms, pos, sid = _het_inputs()
    g, qc = _run(geno_df, chroms, pos, sid)          # both flags off (default)
    assert "Median het" in qc and abs(qc["Median het"] - 0.4) < 1e-9
    assert qc["Fail HetExcess"] == 0
    assert g.shape[1] == 3                            # nothing removed


def test_p13_max_het_removes_the_excess_variant_only():
    geno_df, chroms, pos, sid = _het_inputs()
    g, qc = _run(geno_df, chroms, pos, sid, max_het=0.5)
    assert qc["Fail HetExcess"] == 1
    assert "snp0_het" not in g.columns               # Ho=1.0 removed
    assert "snp1_hom" in g.columns and "snp2_mid" in g.columns


def test_p13_deficit_is_never_flagged():
    """The homozygous-split variant (Ho=0, strong heterozygote DEFICIT) survives
    even an aggressive excess screen -- the screen is strictly one-sided."""
    geno_df, chroms, pos, sid = _het_inputs()
    g, _qc = _run(geno_df, chroms, pos, sid, max_het=0.5, het_excess_p=0.5)
    assert "snp1_hom" in g.columns


def test_p13_het_excess_p_flags_clear_excess():
    geno_df, chroms, pos, sid = _het_inputs()
    g_off, _ = _run(geno_df, chroms, pos, sid, het_excess_p=None)
    assert "snp0_het" in g_off.columns
    g_on, qc = _run(geno_df, chroms, pos, sid, het_excess_p=0.01)
    assert "snp0_het" not in g_on.columns            # Ho=1.0 vs He=0.5 -> p<<0.01
    assert qc["Fail HetExcess"] == 1


def test_p13_cli_flags_default_off():
    from cli import _build_parser
    args = _build_parser().parse_args(
        ["--vcf", "x", "--pheno", "p", "--trait", "Y", "--output", "o"])
    assert args.max_het is None and args.het_excess_p is None
    args2 = _build_parser().parse_args(
        ["--vcf", "x", "--pheno", "p", "--trait", "Y", "--output", "o",
         "--max-het", "0.20", "--het-excess-p", "1e-4"])
    assert args2.max_het == 0.20 and args2.het_excess_p == 1e-4
