"""MLMM and FarmCPU tables carry MLM's significance columns, and their significant LD
blocks get heatmaps (reviewer point 2-INT.6a).

Planted data (tests/_multitrait_data), all three models, through the per-trait pipeline
the CLI and One-Click share (gwas.pipeline.run_trait_pipeline):
A. the MLMM and FarmCPU tables carry the same significance columns as MLM, for every
   --sig-thresh rule; Bonferroni, M_eff and custom equal PValue below MLM's thresholds,
   and FDR is Benjamini-Hochberg within each model's own p-values;
B. a block whose lead SNP is significant in MLMM or FarmCPU gets a heatmap; a block that
   several models report (same bounds) is drawn once, one PNG per block; its title names
   every model whose own block with these bounds has a significant lead, or in which the
   marked lead SNP is significant, unless that is MLM alone, whose heatmaps stay as they
   were (also inside a three-model run);
E. the report's significant-SNP count is the rule's column, also when the
   significant-SNP table is not built (it used to fall back to the FDR count), and with
   several models the report gives one count per model.
Under --sig-thresh fdr the cross-model consensus judges each model by its own FDR.
"""
import io
import re
import zipfile

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

import gwas.pipeline as pl
import gwas.plotting as gp
from annotation import canon_chr
from gwas.io import load_vcf_cached
from tests import _multitrait_data as mt

MODELS = ["mlm", "mlmm", "farmcpu"]
SIG_COLS = ["FDR", "Significant_FDR", "Significant_Bonf", "Significant_Meff"]


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    base = tmp_path_factory.mktemp("model_significance")
    vcf, pheno_csv, _info = mt.write(base)
    parsed = pl.ParsedGenotypes.from_callset(load_vcf_cached(vcf.read_bytes()))
    return {"vcf": vcf, "pheno_csv": pheno_csv, "parsed": parsed,
            "pheno": pd.read_csv(pheno_csv, index_col=0)}


def _run(p, trait="T_chr1", model=MODELS, **kw):
    opts = pl.cli_default_options(vcf=str(p["vcf"]), pheno=str(p["pheno_csv"]),
                                  model=list(model), **kw)
    return pl.run_trait_pipeline(p["parsed"], p["pheno"], trait, opts)


def _tables(res, trait="T_chr1"):
    return {"MLM": res.tables[f"GWAS_{trait}.csv"],
            "MLMM": res.tables[f"GWAS_MLMM_{trait}.csv"],
            "FarmCPU": res.tables[f"GWAS_FarmCPU_{trait}.csv"]}


def _sig_cols(df):
    return [c for c in df.columns if c == "FDR" or c.startswith("Significant_")]


def _small_meff(monkeypatch):
    """M_eff = 5 (threshold 0.01): on the planted data the real M_eff picks the same SNPs
    as Bonferroni, so a check of Significant_Meff could not tell the two apart."""
    monkeypatch.setattr(gp, "compute_meff_li_ji", lambda geno: (5, np.array([])))


@pytest.mark.parametrize("rule", ["bonferroni", "meff", "fdr", 2e-2])
def test_mlmm_and_farmcpu_carry_mlms_significance_columns(planted, rule, monkeypatch):
    _small_meff(monkeypatch)
    res = _run(planted, sig_thresh=rule, no_plots=True)
    assert res.summary["meff"] == 5
    tables = _tables(res)
    custom = isinstance(rule, float)
    expected = SIG_COLS + (["Significant_Custom"] if custom else [])
    bonf = 0.05 / res.summary["n_snps"]              # MLM's thresholds, for every model
    meff = 0.05 / res.summary["meff"]
    for model, df in tables.items():
        assert _sig_cols(df) == expected, model
        p = df["PValue"].to_numpy(dtype=float)
        ok = ~np.isnan(p)
        assert np.array_equal(df["Significant_Bonf"].to_numpy(), p < bonf), model
        assert np.array_equal(df["Significant_Meff"].to_numpy(), p < meff), model
        if custom:
            assert np.array_equal(df["Significant_Custom"].to_numpy(), p < rule), model
        reject, q, _, _ = multipletests(p[ok], method="fdr_bh")   # within this model
        assert np.array_equal(df["Significant_FDR"].to_numpy()[ok], reject), model
        assert not df["Significant_FDR"].to_numpy()[~ok].any(), model
        np.testing.assert_array_equal(df["FDR"].to_numpy()[ok], q)
        # the thresholds select different SNPs here, so a column on the wrong one fails
        assert df["Significant_Bonf"].any(), f"{model}: nothing significant, the check is vacuous"
        assert (df["Significant_Meff"] != df["Significant_Bonf"]).any(), model
        if custom:
            assert (df["Significant_Custom"] != df["Significant_Meff"]).any(), model


def _heatmap_name(chrom, start, end):
    return f"figures/LD_heatmap_Chr{canon_chr(str(chrom))}_{int(start)}_{int(end)}.png"


def _record_heatmaps(monkeypatch):
    real = gp.plot_ld_block_heatmap
    calls = []

    def _record(*a, **k):
        fig, info = real(*a, **k)
        calls.append({"name": _heatmap_name(a[3], a[4], a[5]), "models": k.get("models"),
                      "title": " ".join(info["title"].splitlines())})
        return fig, info

    monkeypatch.setattr(gp, "plot_ld_block_heatmap", _record)
    return calls


def _zip_members(res):
    with zipfile.ZipFile(io.BytesIO(res.zip_bytes)) as zf:
        return {n: zf.read(n) for n in zf.namelist()}


def test_blocks_significant_in_mlmm_or_farmcpu_get_one_heatmap_each(planted, monkeypatch):
    calls = _record_heatmaps(monkeypatch)
    res = _run(planted, sig_thresh="bonferroni")
    tables = _tables(res)
    with zipfile.ZipFile(io.BytesIO(res.zip_bytes)) as zf:
        names = zf.namelist()
        blocks = {m: pd.read_csv(io.BytesIO(zf.read(f"tables/LD_blocks_annotated_{m}.csv")))
                  for m in tables}
    sig = {m: set(t.loc[t["Significant_Bonf"], "SNP"].astype(str)) for m, t in tables.items()}
    # every block whose lead SNP is significant in its own model, keyed by file name:
    # the models reporting it, and the lead of the first one (the SNP the heatmap marks)
    finders, lead = {}, {}
    for model, b in blocks.items():
        for _, r in b.iterrows():
            if str(r["lead_snp"]) in sig[model]:
                name = _heatmap_name(r["Chr"], r["Start (bp)"], r["End (bp)"])
                finders.setdefault(name, []).append(model)
                lead.setdefault(name, str(r["lead_snp"]))
    expected = {n: [m for m in tables if m in finders[n] or lead[n] in sig[m]] for n in finders}
    heatmaps = sorted(n for n in names if "LD_heatmap_" in n)
    assert len(names) == len(set(names)), "duplicate ZIP member names"
    assert heatmaps == sorted(finders)
    assert any(len(ms) > 1 for ms in finders.values()), "no block reported by two models"
    assert any("MLM" not in ms for ms in finders.values()), "no block outside MLM's"
    assert any(expected[n] != finders[n] for n in finders), "no lead called by a model with other bounds"
    assert sorted(c["name"] for c in calls) == heatmaps, "each block is drawn exactly once"
    for c in calls:
        models = expected[c["name"]]
        assert models != ["MLM"]                              # every planted block is shared
        assert c["models"] == models
        assert c["title"].endswith("significant in " + ", ".join(models))


def test_mlm_alone_keeps_its_heatmaps_byte_identical_in_a_three_model_run(planted, monkeypatch):
    """With MLMM and FarmCPU calling nothing (their p-values set to 0.5), every heatmap of
    a three-model run is MLM's alone: drawn without a model list, and byte-identical to
    the MLM-only run's."""
    import gwas.models as gm
    alone = _run(planted, model=["mlm"])
    calls = _record_heatmaps(monkeypatch)

    def _calling_nothing(fn):
        def _wrapped(*a, **k):
            out = fn(*a, **k)
            table = out[0].copy()
            table["PValue"] = 0.5
            return (table, *out[1:])
        return _wrapped

    monkeypatch.setattr(gm, "run_mlmm_research_grade_fast", _calling_nothing(gm.run_mlmm_research_grade_fast))
    monkeypatch.setattr(gm, "run_farmcpu", _calling_nothing(gm.run_farmcpu))
    three = _run(planted)
    assert not _tables(three)["MLMM"]["Significant_Bonf"].any()
    a, b = _zip_members(alone), _zip_members(three)
    heat_a = sorted(n for n in a if "LD_heatmap_" in n)
    assert heat_a and heat_a == sorted(n for n in b if "LD_heatmap_" in n)
    assert all(a[n] == b[n] for n in heat_a), "an MLM-only heatmap changed in a three-model run"
    assert calls and all(c["models"] is None and "significant in" not in c["title"] for c in calls)


def _report_count(zip_bytes):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        html = zf.read("report.html").decode("utf-8")
    m = re.search(r'<div class="value">([^<]*)</div>\s*<div class="label">Significant \(([^<]*)\)</div>',
                  html)
    assert m, "summary card not found in report.html"
    return int(m.group(1)), m.group(2)


@pytest.mark.parametrize("rule,column", [("bonferroni", "Significant_Bonf"),
                                         ("meff", "Significant_Meff")])
def test_report_count_is_the_rule_column_with_or_without_the_significant_snp_table(
        planted, rule, column):
    gw = None
    for no_rescue in (False, True):
        res = _run(planted, model=["mlm"], sig_thresh=rule, no_plots=True,
                   no_isolated_rescue=no_rescue)
        gw = res.tables["GWAS_T_chr1.csv"]
        n, _label = _report_count(res.zip_bytes)
        assert n == int(gw[column].sum()), (rule, no_rescue)
        if no_rescue:
            assert "Significant_SNPs_MLM.csv" not in res.tables
        else:
            assert n == len(res.tables["Significant_SNPs_MLM.csv"])
    # the count the report used to fall back to differs here, so the check bites
    assert int((gw["FDR"] < 0.05).sum()) != int(gw[column].sum())


def test_report_gives_one_count_per_model_in_a_multi_model_run(planted, monkeypatch):
    _small_meff(monkeypatch)
    res = _run(planted, sig_thresh="meff", no_plots=True)
    with zipfile.ZipFile(io.BytesIO(res.zip_bytes)) as zf:
        html = zf.read("report.html").decode("utf-8")
    cards = re.findall(r'<div class="value">([^<]*)</div>\s*<div class="label">Significant in (\w+) \(', html)
    assert [m for _n, m in cards] == ["MLM", "MLMM", "FarmCPU"]
    for n, model in cards:
        assert int(n) == int(_tables(res)[model]["Significant_Meff"].sum()), model
    assert '<div class="label">Significant (' not in html
    assert "<h2>Significant SNPs (MLM)</h2>" in html
    alone = _run(planted, model=["mlm"], sig_thresh="meff", no_plots=True)
    with zipfile.ZipFile(io.BytesIO(alone.zip_bytes)) as zf:
        html1 = zf.read("report.html").decode("utf-8")
    assert "Significant in " not in html1 and "<h2>Significant SNPs</h2>" in html1
    # an MLM-only report keeps the card exactly as 7062830's template rendered it
    n1 = int(alone.tables["GWAS_T_chr1.csv"]["Significant_Meff"].sum())
    card = "\n".join([
        "  </div>",
        "  ",
        '  <div class="summary-card">',
        f'    <div class="value">{n1}</div>',
        f'    <div class="label">Significant ({alone.summary["sig_label"]})</div>',
        "  </div>",
        "</div>",
        "",
    ])
    assert card in html1


def test_fdr_rule_judges_each_model_by_its_own_fdr_in_the_consensus(planted):
    res = _run(planted, trait="T_chr2", sig_thresh="fdr", no_plots=True)
    consensus = res.tables["CrossModel_Consensus.csv"]
    detected = consensus["Detected_by"].str.split(",")
    for model, df in _tables(res, "T_chr2").items():
        own = set(df.loc[df["FDR"] < 0.05, "SNP"].astype(str))
        got = set(consensus.loc[detected.apply(lambda ms, m=model: m in ms), "SNP"].astype(str))
        assert got == own, model
    # MLMM and FarmCPU now also get their significant-SNP tables under the FDR rule
    assert {"Significant_SNPs_MLMM.csv", "Significant_SNPs_FarmCPU.csv"} <= set(res.tables)
