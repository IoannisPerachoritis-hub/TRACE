"""One rule for missing covariate values, the CLI's: the samples without one are dropped
after phenotype QC, before SNP QC, imputation, kinship and the PCs, by one helper
(gwas.covariates.drop_covariate_missing), in the CLI, One-Click and the GWAS page."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gwas.pipeline as pl
import gwas.qc as qc
from gwas.covariates import (CovariateDropError, covariate_drop_message,
                             drop_covariate_missing, load_covariate_frame,
                             select_covariate_columns)
from gwas.io import load_vcf_cached

EX = Path(__file__).resolve().parent / "data" / "example_v1"   # the first example
TRAIT = "Trait1"
MISSING = ["SAMPLE_003", "SAMPLE_017", "SAMPLE_042"]   # no covariate value


def test_the_message_names_the_first_ten():
    assert covariate_drop_message(["a", "b"], 50) == (
        "2 of 50 analysis samples have no covariate value and were dropped: a, b.")
    ids = [f"s{i}" for i in range(13)]
    assert covariate_drop_message(ids, 100) == (
        "13 of 100 analysis samples have no covariate value and were dropped: "
        "s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, and 3 more.")


def test_the_helper_keeps_the_complete_samples_in_order():
    idx = pd.Index(["a", "b", "c", "d"])
    geno = pd.DataFrame({"m1": [0, 1, 2, 1]}, index=idx)
    pheno = pd.DataFrame({"t": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    y = np.array([1.0, 2.0, 3.0, 4.0])
    cov = pd.DataFrame({"x": [0.5, np.nan, 1.5]}, index=["a", "b", "d"])  # c absent, b blank
    g, p, yy, dropped = drop_covariate_missing(cov, geno, pheno, y)
    assert dropped == ["b", "c"]
    assert list(g.index) == ["a", "d"] and list(p.index) == ["a", "d"]
    assert yy.tolist() == [1.0, 4.0]
    complete = pd.DataFrame({"x": [0.1, 0.2, 0.3, 0.4]}, index=idx)
    g2, p2, y2, d2 = drop_covariate_missing(complete, geno, pheno, y)
    assert d2 == [] and g2 is geno and p2 is pheno and y2 is y   # nothing dropped: unchanged


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    vcf_bytes = (EX / "example.vcf.gz").read_bytes()
    pheno = pd.read_csv(EX / "example_pheno.csv").set_index("SampleID")
    pheno.index = pheno.index.astype(str)
    rng = np.random.default_rng(3)
    cov = pd.DataFrame({"SampleID": list(pheno.index),
                        "Batch": rng.normal(0, 1, len(pheno))})
    cov.loc[cov["SampleID"].isin(MISSING), "Batch"] = np.nan
    path = tmp_path_factory.mktemp("covar") / "covariates.csv"
    cov.to_csv(path, index=False)
    return {"vcf_bytes": vcf_bytes, "pheno": pheno, "covar_path": path,
            "covar_frame": select_covariate_columns(load_covariate_frame(path)),
            "parsed": pl.ParsedGenotypes.from_callset(load_vcf_cached(vcf_bytes))}


def _opts(**kw):
    return pl.cli_default_options(vcf="example.vcf.gz", pheno="example_pheno.csv",
                                  no_plots=True, no_report=True, **kw)


def test_the_gwas_page_and_the_cli_pass_the_same_samples(example, monkeypatch):
    """Kinship, the PCA input (kinship's Z) and the scan get the same samples from the
    GWAS page's preprocessing and from the CLI's pipeline."""
    seen = []
    real = qc._pipeline_build_kinship

    def recording(geno_df, geno_imputed, chroms, positions):
        out = real(geno_df, geno_imputed, chroms, positions)
        seen.append({"samples": list(geno_df.index.astype(str)), "pca_rows": int(out[1].shape[0])})
        return out

    monkeypatch.setattr(qc, "_pipeline_build_kinship", recording)
    o = _opts(covar=str(example["covar_path"]))
    gui = qc._gwas_pipeline_impl(
        example["vcf_bytes"], example["pheno"].copy(), TRAIT, o.maf, o.miss, o.norm, True,
        o.ind_miss, o.mac, info_thresh=o.info_thresh, impute_method=o.impute,
        impute_k=o.impute_k, impute_l=o.impute_l, covar_df=example["covar_frame"],
        vcf_hash="test_covariate_drop")
    cli = pl.run_trait_pipeline(example["parsed"], example["pheno"].copy(), TRAIT, o)
    assert len(seen) == 2
    gui_seen, cli_seen = seen
    assert gui_seen == cli_seen
    assert len(gui_seen["samples"]) == len(example["pheno"]) - len(MISSING)
    assert not set(MISSING) & set(gui_seen["samples"])
    assert gui["covar_dropped"] == MISSING and cli.summary["covariate_dropped"] == MISSING
    assert gui["covar_n_before"] == cli.summary["covariate_n_before"] == len(example["pheno"])
    # the scan: the same samples, in the same order, with the same trait values
    assert list(np.asarray(gui["iid"])[:, 0]) == gui_seen["samples"]
    np.testing.assert_array_equal(np.asarray(gui["y"], float).ravel(),
                                  np.asarray(cli.context["y"], float).ravel())


def test_one_click_and_the_cli_give_the_same_mlm_table(example):
    """One-Click passes the uploaded table (run_traits with covar_frame); the CLI reads
    the file (--covar). Same samples dropped, byte-identical MLM table."""
    by_file = pl.run_trait_pipeline(example["parsed"], example["pheno"].copy(), TRAIT,
                                    _opts(covar=str(example["covar_path"])))
    [(trait, res, err)] = list(pl.run_traits(
        example["parsed"], example["pheno"].copy(), [TRAIT],
        _opts(covar="covariates", covar_frame=example["covar_frame"])))
    assert err is None and trait == TRAIT
    table = f"GWAS_{TRAIT}.csv"
    assert res.tables[table].to_csv(index=False) == by_file.tables[table].to_csv(index=False)
    assert res.summary["covariate_dropped"] == by_file.summary["covariate_dropped"] == MISSING


def test_no_sample_left_stops_with_the_message(example):
    """A covariate file whose IDs match no VCF sample: the GWAS page's preprocessing
    stops with the message (the page shows it), not with an unrelated error."""
    o = _opts()
    wrong_ids = example["covar_frame"].copy()
    wrong_ids.index = "X_" + wrong_ids.index
    with pytest.raises(CovariateDropError, match=r"^50 of 50 analysis samples have no covariate"):
        qc._gwas_pipeline_impl(
            example["vcf_bytes"], example["pheno"].copy(), TRAIT, o.maf, o.miss, o.norm, True,
            o.ind_miss, o.mac, info_thresh=o.info_thresh, impute_method=o.impute,
            impute_k=o.impute_k, impute_l=o.impute_l, covar_df=wrong_ids,
            vcf_hash="test_covariate_drop")
