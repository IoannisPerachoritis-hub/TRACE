"""MLMM's cofactor window is the run's LD window (author's decision, 10 Nov):
--ld-flank-kb when given, else int(2 x the median LD decay), else 300 kb, in the CLI and
One-Click (the shared pipeline) as on the GWAS page."""
import json

import numpy as np
import pandas as pd
import pytest

import gwas.ld as ld
import gwas.models as models
import gwas.pipeline as pl
from gwas.io import load_vcf_cached
from tests import _multitrait_data as mt


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    vcf, pheno_csv, _info = mt.write(tmp_path_factory.mktemp("mlmm_window"))
    return {"vcf": vcf, "pheno_csv": pheno_csv,
            "parsed": pl.ParsedGenotypes.from_callset(load_vcf_cached(vcf.read_bytes())),
            "pheno": pd.read_csv(pheno_csv, index_col=0)}


def _run(planted, monkeypatch, **kw):
    seen = []
    real = models.run_mlmm_research_grade_fast

    def recording(*a, **k):
        seen.append(k.get("window_kb"))
        return real(*a, **k)

    monkeypatch.setattr(models, "run_mlmm_research_grade_fast", recording)
    opts = pl.cli_default_options(vcf=str(planted["vcf"]), pheno=str(planted["pheno_csv"]),
                                  model=["mlm", "mlmm"], no_plots=True, no_report=True, **kw)
    res = pl.run_trait_pipeline(planted["parsed"], planted["pheno"], "T_chr1", opts)
    import io
    import zipfile
    with zipfile.ZipFile(io.BytesIO(res.zip_bytes)) as zf:
        meta = json.loads(zf.read("run_metadata.json"))
    return seen, meta


def test_the_estimated_decay_gives_the_window(planted, monkeypatch):
    seen, meta = _run(planted, monkeypatch)
    assert isinstance(meta["LD decay (kb)"], float)
    assert seen == [meta["LD flank (kb)"]] == [meta["MLMM cofactor window (kb)"]]
    # int(2 x decay); the metadata rounds the decay to 0.1 kb
    assert abs(meta["LD flank (kb)"] - 2 * meta["LD decay (kb)"]) <= 1
    keys = list(meta)
    assert keys.index("MLMM cofactor window (kb)") == keys.index("LD flank (kb)") + 1


def test_a_given_flank_is_the_window(planted, monkeypatch):
    seen, meta = _run(planted, monkeypatch, ld_flank_kb=200)
    assert seen == [200] and meta["MLMM cofactor window (kb)"] == 200


def test_no_estimate_gives_300_kb(planted, monkeypatch):
    monkeypatch.setattr(ld, "ld_decay", lambda *a, **k: (np.nan, None, pd.DataFrame()))
    seen, meta = _run(planted, monkeypatch)
    assert meta["LD decay (kb)"] == "N/A"
    assert seen == [300] and meta["MLMM cofactor window (kb)"] == 300


def test_an_mlm_only_run_records_no_mlmm_window(planted, monkeypatch):
    opts = pl.cli_default_options(vcf=str(planted["vcf"]), pheno=str(planted["pheno_csv"]),
                                  no_plots=True, no_report=True)
    res = pl.run_trait_pipeline(planted["parsed"], planted["pheno"], "T_chr1", opts)
    import io
    import zipfile
    with zipfile.ZipFile(io.BytesIO(res.zip_bytes)) as zf:
        assert "MLMM cofactor window (kb)" not in json.loads(zf.read("run_metadata.json"))
