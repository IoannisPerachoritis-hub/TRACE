"""A numeric --sig-thresh with plots on (reviewer point 2-INT.11, bug 1).

The bare number used to reach plot code that needs a string label, so every such run
stopped at the MLM Manhattan plot ('float' object has no attribute 'startswith') and
wrote no ZIP; the MLMM and FarmCPU Manhattan calls had the same fault. Now the line is
labelled with the custom rule's own label ("p < 1.0e-02"), as in the report and the GUI.

Bundled example, all three models, CC-8's LD settings, plots on, --sig-thresh 1e-2
(1e-5 selects the same SNPs as Bonferroni on this data, so it could not show which
column is used): the CLI exits 0 and writes every table and figure, and each place the
rule applies uses Significant_Custom: the significant-SNP tables, the isolated-SNP
rescue, the consensus, the heatmaps and the report. Plus the brief's own example, 1e-5,
MLM only.
"""
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import gwas.plotting as gp  # noqa: E402
from annotation import canon_chr  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
_EX = _ROOT / "tests" / "data" / "example_v1"   # the first example
MODELS = ("MLM", "MLMM", "FarmCPU")


def _main(argv, out, monkeypatch):
    """Run the CLI as a user would (cli.main with these arguments); record every static
    Manhattan call; return the ZIP's members and the recorded calls."""
    import cli
    calls = []
    real = gp.plot_manhattan_static

    def _record(df, active_lod, active_label, title, *a, **k):
        calls.append({"title": title, "lod": active_lod, "label": active_label})
        return real(df, active_lod, active_label, title, *a, **k)

    monkeypatch.setattr(gp, "plot_manhattan_static", _record)
    monkeypatch.setattr(sys, "argv", ["cli.py", *argv, "--output", str(out)])
    try:
        cli.main()
    except SystemExit as e:
        assert e.code in (0, None), f"the CLI exited with {e.code!r}"
    zips = sorted(out.glob("*.zip"))
    assert len(zips) == 1, [p.name for p in out.iterdir()]
    with zipfile.ZipFile(zips[0]) as zf:
        return {n: zf.read(n) for n in zf.namelist()}, calls


@pytest.fixture(scope="module")
def three_models(tmp_path_factory):
    out = tmp_path_factory.mktemp("numeric_thresh")
    with pytest.MonkeyPatch.context() as mp:
        members, calls = _main(
            ["--vcf", str(_EX / "example.vcf.gz"), "--pheno", str(_EX / "example_pheno.csv"),
             "--trait", "Trait1", "--model", "mlm", "mlmm", "farmcpu", "--sig-thresh", "1e-2",
             "--ld-r2", "0.1", "--ld-flank-kb", "200"], out, mp)
    tables = {m: pd.read_csv(io.BytesIO(members[f"tables/{name}.csv"]))
              for m, name in zip(MODELS, ("GWAS_Trait1", "GWAS_MLMM_Trait1", "GWAS_FarmCPU_Trait1"))}
    return {"members": members, "calls": calls, "tables": tables}


def test_every_table_and_figure_is_written(three_models):
    per_model = ("GWAS_{t}", "LD_blocks_annotated_{m}", "LD_triage_{m}", "Significant_SNPs_{m}",
                 "Unblocked_SNPs_{m}", "Isolated_SNP_intervals_{m}", "Isolated_SNP_candidate_genes_{m}")
    tables = {f"tables/{p.format(m=m, t=name)}.csv"
              for m, name in zip(MODELS, ("Trait1", "MLMM_Trait1", "FarmCPU_Trait1")) for p in per_model}
    tables |= {f"tables/{n}.csv" for n in ("CrossModel_Consensus", "PC_diagnostics_spectrum",
                                            "QC_report_summary", "QC_sample_heterozygosity",
                                            "SNP_view_index_MLM", "Haplotype_assignments_Trait1",
                                            # each other model's tested blocks (CC-30)
                                            "Haplotype_assignments_Trait1_MLMM",
                                            "Haplotype_assignments_Trait1_FarmCPU")}
    figures = {f"figures/{k}_{m}.png" for k in ("Manhattan", "QQ") for m in MODELS}
    figures |= {"figures/PCA_scree.png", "figures/QC_report_distributions.png"}
    names = set(three_models["members"])
    heatmaps = {n for n in names if n.startswith("figures/LD_heatmap_")}   # see the heatmap test
    assert names - heatmaps == tables | figures | {"report.html", "run_metadata.json"}
    for name in tables:
        assert len(pd.read_csv(io.BytesIO(three_models["members"][name]))) > 0, name


def test_every_models_table_carries_significant_custom(three_models):
    for model, t in three_models["tables"].items():
        p = t["PValue"].to_numpy(dtype=float)
        assert np.array_equal(t["Significant_Custom"].to_numpy(bool), p < 1e-2), model
        # the threshold selects other SNPs than Bonferroni and M_eff, so the check bites
        assert (t["Significant_Custom"] != t["Significant_Bonf"]).any(), model
        assert (t["Significant_Custom"] != t["Significant_Meff"]).any(), model


def test_all_figures_are_written_with_the_custom_label(three_models):
    names = set(three_models["members"])
    for model in MODELS:
        assert f"figures/Manhattan_{model}.png" in names, model
        assert f"figures/QQ_{model}.png" in names, model
    assert {"figures/PCA_scree.png", "figures/QC_report_distributions.png"} <= names
    calls = three_models["calls"]
    assert [c["title"] for c in calls] == [f"Manhattan ({m}): Trait1" for m in MODELS]
    for c in calls:
        assert c["label"] == "p < 1.0e-02"
        assert c["lod"] == pytest.approx(-np.log10(1e-2))


def test_heatmaps_are_the_blocks_whose_lead_passes_the_custom_threshold(three_models):
    members, tables = three_models["members"], three_models["tables"]
    expected = set()
    for model in MODELS:
        blocks = pd.read_csv(io.BytesIO(members[f"tables/LD_blocks_annotated_{model}.csv"]))
        sig = set(tables[model].loc[tables[model]["Significant_Custom"], "SNP"].astype(str))
        for _, r in blocks.iterrows():
            if str(r["lead_snp"]) in sig:
                expected.add(f"figures/LD_heatmap_Chr{canon_chr(str(r['Chr']))}_"
                             f"{int(r['Start (bp)'])}_{int(r['End (bp)'])}.png")
    heatmaps = {n for n in members if "LD_heatmap_" in n}
    assert len(expected) > 1 and heatmaps == expected


def test_significant_snp_tables_and_consensus_use_the_custom_column(three_models):
    members, tables = three_models["members"], three_models["tables"]
    consensus = pd.read_csv(io.BytesIO(members["tables/CrossModel_Consensus.csv"]))
    detected = consensus["Detected_by"].str.split(",")
    for model in MODELS:
        own = set(tables[model].loc[tables[model]["Significant_Custom"], "SNP"].astype(str))
        sig = pd.read_csv(io.BytesIO(members[f"tables/Significant_SNPs_{model}.csv"]))
        assert set(sig["SNP"].astype(str)) == own, model
        assert set(sig["Sig_Rule"]) == {"custom"} and set(sig["Sig_Threshold"]) == {1e-2}, model
        got = set(consensus.loc[detected.apply(lambda ms, m=model: m in ms), "SNP"].astype(str))
        assert got == own, model
        # the isolated-SNP rescue takes the rule's SNPs that are in no LD block
        unblocked = pd.read_csv(io.BytesIO(members[f"tables/Unblocked_SNPs_{model}.csv"]))
        assert set(unblocked["Sig_Rule"]) == {"custom"} and set(unblocked["Sig_Threshold"]) == {1e-2}, model
        assert (set(unblocked["SNP"].astype(str))
                == set(sig.loc[sig["Block_Status"] != "in_block", "SNP"].astype(str))), model


def test_report_counts_the_custom_column_and_metadata_keeps_the_number(three_models):
    members, tables = three_models["members"], three_models["tables"]
    html = members["report.html"].decode("utf-8")
    cards = re.findall(r'<div class="value">([^<]*)</div>\s*<div class="label">Significant in (\w+) '
                       r'\(p &lt; 1\.0e-02\)</div>', html)
    assert [m for _n, m in cards] == list(MODELS)
    for n, model in cards:
        assert int(n) == int(tables[model]["Significant_Custom"].sum()), model
    assert json.loads(members["run_metadata.json"])["Significance"] == 0.01


def test_the_briefs_example_1e5_mlm_only(tmp_path, monkeypatch):
    members, calls = _main(
        ["--vcf", str(_EX / "example.vcf.gz"), "--pheno", str(_EX / "example_pheno.csv"),
         "--trait", "Trait1", "--sig-thresh", "1e-5"], tmp_path / "out", monkeypatch)
    assert {"figures/Manhattan_MLM.png", "figures/QQ_MLM.png"} <= set(members)
    assert [c["label"] for c in calls] == ["p < 1.0e-05"]
    gw = pd.read_csv(io.BytesIO(members["tables/GWAS_Trait1.csv"]))
    assert np.array_equal(gw["Significant_Custom"].to_numpy(bool), gw["PValue"].to_numpy(float) < 1e-5)
