"""Covariates taken from columns of the phenotype file on the GWAS page ("Use columns of this
file as covariates").

- The table the page builds from the uploaded file equals what the CLI's --covar <the file>
  --covar-cols <columns> builds: load_covariate_frame and select_covariate_columns, with the
  page's sample ID column.
- The chosen columns leave the trait list.
- Changing the chosen columns changes the results' identity (the covariate table's hash is part
  of the table identity and the trait identity).
- The page builds both from these helpers (its source, read with ast).
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gwas import guigwas, guitable
from gwas.covariates import covariates_from_phenotype, load_covariate_frame, select_covariate_columns
from gwas.utils import hash_df

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "pages" / "GWAS_analysis.py"


def _pheno_file(tmp_path, first="Batch", id_name="Accession"):
    """A phenotype file whose ID column is not the first one, with padded IDs and a missing value."""
    rng = np.random.default_rng(3)
    n = 40
    df = pd.DataFrame({
        first: rng.integers(0, 3, n),
        id_name: [f" S{i:03d} " if i % 3 == 0 else f"S{i:03d}" for i in range(n)],
        "weight_g": rng.normal(100, 10, n).round(3),
        "locule_number": rng.integers(2, 8, n).astype(float),
        "height": rng.normal(1.5, 0.2, n),
        "note": ["a"] * n,
    })
    df.loc[5, "weight_g"] = np.nan
    path = tmp_path / "pheno.csv"
    df.to_csv(path, index=False)
    return path


def _as_the_page_reads(path):
    """The page's read of an uploaded phenotype file, then its sample ID index."""
    raw = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    return guigwas.index_phenotype(raw)


@pytest.mark.parametrize("cols", [["weight_g"], ["weight_g", "height"], ["Batch", "locule_number"]])
def test_the_table_from_the_phenotype_file_is_the_clis(tmp_path, cols):
    path = _pheno_file(tmp_path)
    pheno, id_col, from_first = _as_the_page_reads(path)
    assert id_col == "Accession" and not from_first
    gui = covariates_from_phenotype(pheno, cols)
    cli = select_covariate_columns(load_covariate_frame(path, id_col=id_col), cols)
    pd.testing.assert_frame_equal(gui, cli)
    assert gui.index[0] == "S000" and gui.index.name == "Accession"     # the IDs stripped of blanks
    if "weight_g" in cols:
        assert np.isnan(gui.loc["S005", "weight_g"])                    # a missing value stays missing


def test_the_page_takes_the_first_column_when_no_column_has_an_id_name(tmp_path):
    rng = np.random.default_rng(4)
    path = tmp_path / "coded.tsv"
    pd.DataFrame({"Code": [f"L{i:02d} " for i in range(30)], "weight_g": rng.normal(size=30),
                  "brix": rng.normal(size=30)}).to_csv(path, sep="\t", index=False)
    pheno, id_col, from_first = _as_the_page_reads(path)
    assert id_col == "Code" and from_first
    cli = select_covariate_columns(load_covariate_frame(path, id_col=id_col), ["weight_g"])
    pd.testing.assert_frame_equal(covariates_from_phenotype(pheno, ["weight_g"]), cli)
    assert guigwas.trait_columns(pheno, ["weight_g"]) == ["brix"]


def test_a_non_numeric_column_or_no_column_is_refused(tmp_path):
    pheno, _, _ = _as_the_page_reads(_pheno_file(tmp_path))
    with pytest.raises(ValueError, match="non-numeric"):
        covariates_from_phenotype(pheno, ["note"])
    with pytest.raises(ValueError, match="No covariate columns selected"):
        covariates_from_phenotype(pheno, [])


def test_the_chosen_columns_leave_the_trait_list(tmp_path):
    pheno, _, _ = _as_the_page_reads(_pheno_file(tmp_path))
    assert guigwas.trait_columns(pheno, []) == ["Batch", "weight_g", "locule_number", "height"]
    assert guigwas.trait_columns(pheno, ["weight_g", "Batch"]) == ["locule_number", "height"]


def test_changing_the_chosen_columns_changes_the_results_identity(tmp_path):
    pheno, _, _ = _as_the_page_reads(_pheno_file(tmp_path))
    choices = (["weight_g"], ["weight_g", "height"], ["height"], ["height", "weight_g"])
    hashes = [hash_df(covariates_from_phenotype(pheno, c)) for c in choices]
    assert len(set(hashes)) == len(choices)
    settings = dict(pheno="p", trait="locule_number", norm="none", maf=0.05, miss=0.1, ind_miss=0.2,
                    mac=5, info=0.0, drop_alt=1, impute="mean", impute_k=5, impute_l=20, canonical=None)
    trait_ids = {guigwas.trait_identity("vcf", covar=h, samples="s", **settings) for h in hashes + [None]}
    table_ids = {guitable.table_identity("vcf", mode="memory", covar=h, **settings) for h in hashes + [None]}
    assert len(trait_ids) == len(table_ids) == len(choices) + 1


def test_the_page_builds_the_covariates_and_the_traits_from_these_helpers():
    page = ast.parse(PAGE.read_text(encoding="utf-8"))
    calls = {ast.unparse(n) for n in ast.walk(page) if isinstance(n, ast.Call)}
    assert "_gg.index_phenotype(pheno)" in calls
    assert "_covar_from_pheno(pheno, pheno_covar_cols)" in calls
    assert "_gg.trait_columns(pheno, pheno_covar_cols)" in calls
    (assign,) = [n for n in ast.walk(page) if isinstance(n, ast.Assign)
                 and [ast.unparse(t) for t in n.targets] == ["numeric_traits"]]
    assert ast.unparse(assign.value) == "_gg.trait_columns(pheno, pheno_covar_cols)"
    # one source at a time: a covariate file in use (uploaded now, or kept from an earlier visit)
    # with chosen columns stops the page with the plain message
    (stop,) = [n for n in ast.walk(page) if isinstance(n, ast.If)
               and ast.unparse(n.test) == "pheno_covar_cols and _covar_decision in ('upload', 'session')"]
    assert ast.unparse(stop.body[0]) == "_rehydrate_or_stop(_gg.PHENO_COVAR_TWO_SOURCES)"
    # Clear session data drops the chosen columns and their kept copy
    state = {guigwas.PHENO_COVAR_KEY: ["weight_g"], guigwas.PHENO_COVAR_KEPT: ["weight_g"]}
    guitable.clear_inputs(state)
    assert guigwas.PHENO_COVAR_KEY not in state
    assert guigwas.PHENO_COVAR_KEPT in guigwas.SESSION_KEYS
