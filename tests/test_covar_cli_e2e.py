"""End-to-end CLI test for --covar: full run_pipeline with user covariates.

Self-contained (own synthetic builders) so it does not depend on other test
modules. Proves the CLI wiring: covariate load, sample alignment, drop of
covar-missing samples, and threading into the MLM run.
"""
import numpy as np
import pandas as pd

from cli import _build_parser, run_pipeline


def _vcf(tmp_path, n=50, n_per_chr=40, chroms=("1", "2", "3"), seed=42):
    rng = np.random.default_rng(seed)
    samples = [f"S{i:03d}" for i in range(n)]
    lines = [
        "##fileformat=VCFv4.2",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples),
    ]
    for ch in chroms:
        for j in range(n_per_chr):
            pos = (j + 1) * 1000
            gts = rng.binomial(2, 0.3, size=n)
            gt = ["0/0" if g == 0 else "0/1" if g == 1 else "1/1" for g in gts]
            lines.append(f"{ch}\t{pos}\tchr{ch}_{pos}\tA\tT\t.\tPASS\t.\tGT\t" + "\t".join(gt))
    p = tmp_path / "g.vcf"
    p.write_text("\n".join(lines) + "\n")
    return p


def _pheno(tmp_path, n=50, trait="TestTrait", seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"SampleID": [f"S{i:03d}" for i in range(n)],
                       trait: rng.normal(10, 2, size=n)})
    p = tmp_path / "p.csv"
    df.to_csv(p, index=False)
    return p


def _covar(tmp_path, n=50, cols=("age", "batch"), seed=7):
    rng = np.random.default_rng(seed)
    data = {"SampleID": [f"S{i:03d}" for i in range(n)]}
    for c in cols:
        data[c] = rng.normal(0, 1, size=n)
    p = tmp_path / "cov.csv"
    pd.DataFrame(data).to_csv(p, index=False)
    return p


def _args(tmp_path, vcf, pheno, extra):
    return _build_parser().parse_args([
        "--vcf", str(vcf), "--pheno", str(pheno), "--trait", "TestTrait",
        "--output", str(tmp_path / "res"), "--model", "mlm",
        "--no-report", "--no-plots", "--maf", "0.01", "--mac", "1", "--n-pcs", "2",
    ] + extra)


def test_covar_pipeline_runs_and_keeps_all_samples(tmp_path):
    vcf, pheno, cov = _vcf(tmp_path), _pheno(tmp_path), _covar(tmp_path)
    ctx = run_pipeline(_args(tmp_path, vcf, pheno, ["--covar", str(cov)]))
    assert ctx is not None and "gwas_df" in ctx
    g = ctx["gwas_df"]
    assert (g["PValue"] >= 0).all() and (g["PValue"] <= 1).all()
    assert ctx["geno_imputed"].shape[0] == 50    # covar complete for all samples


def test_covar_missing_samples_are_dropped(tmp_path):
    vcf, pheno = _vcf(tmp_path), _pheno(tmp_path)
    cov = _covar(tmp_path, n=40)                  # covariates for 40 of 50 samples
    ctx = run_pipeline(_args(tmp_path, vcf, pheno,
                             ["--covar", str(cov), "--covar-cols", "age"]))
    assert ctx is not None
    assert ctx["geno_imputed"].shape[0] == 40    # 10 covar-missing samples dropped


def test_numeric_sig_thresh_materialises_custom_column(tmp_path):
    vcf, pheno = _vcf(tmp_path), _pheno(tmp_path)
    ctx = run_pipeline(_args(tmp_path, vcf, pheno, ["--sig-thresh", "5e-8"]))
    assert ctx is not None and "gwas_df" in ctx
    assert "Significant_Custom" in ctx["gwas_df"].columns
