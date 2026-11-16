"""scripts/compare_result_zips.py: the default comparison and --one-click (a GUI One-Click ZIP
against the CLI's ZIP of the same analysis)."""
import importlib.util
import io
import json
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("compare_result_zips",
                                               REPO / "scripts" / "compare_result_zips.py")
crz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(crz)


def _zip(path, meta, report, table=b"SNP,PValue\nA,0.5\n"):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("run_metadata.json", json.dumps(meta))
        zf.writestr("report.html", report)
        zf.writestr("GWAS_T.csv", table)
    return path


def _report(meta, when):
    return (b"<html><strong>Generated:</strong> " + when + b"\n<div class=\"metadata\">"
            + json.dumps(meta).encode() + b"</div><p>results</p></html>")


def _pair(tmp_path, table_b=b"SNP,PValue\nA,0.5\n"):
    cli = {"VCF": "/data/x.vcf.gz", "Phenotype": "/data/p.csv", "Genotype table": "memory", "MAF": 0.05}
    gui = {"VCF": "x.vcf.gz", "Phenotype": "p.csv", "Pipeline": "One-Click Full Analysis",
           "Genotype table": "disk", "MAF": 0.05}
    a = _zip(tmp_path / "gui.zip", gui, _report(gui, b"10:00"))
    b = _zip(tmp_path / "cli.zip", cli, _report(cli, b"11:00"), table_b)
    return a, b


def test_default_comparison_names_the_run_information_that_differs(tmp_path):
    a, b = _pair(tmp_path)
    out = io.StringIO()
    assert crz.compare(a, b, out=out) == 2
    assert "differs: run_metadata.json" in out.getvalue() and "differs: report.html" in out.getvalue()


def test_one_click_leaves_out_the_input_names_and_the_pipeline_only(tmp_path):
    a, b = _pair(tmp_path)
    out = io.StringIO()
    assert crz.compare(a, b, out=out, one_click=True) == 0
    assert "all identical" in out.getvalue()
    assert crz.main(["--one-click", str(a), str(b)]) == 0
    second = tmp_path / "second"
    second.mkdir()
    a2, b2 = _pair(second, table_b=b"SNP,PValue\nA,0.4\n")
    out = io.StringIO()
    assert crz.compare(a2, b2, out=out, one_click=True) == 1          # a result still counts
    assert "differs: GWAS_T.csv" in out.getvalue()


def test_one_click_still_compares_every_other_run_information_key(tmp_path):
    gui = {"VCF": "x", "Pipeline": "One-Click Full Analysis", "MAF": 0.05}
    cli = {"VCF": "y", "MAF": 0.10}
    a = _zip(tmp_path / "a.zip", gui, _report(gui, b"1"))
    b = _zip(tmp_path / "b.zip", cli, _report(cli, b"2"))
    out = io.StringIO()
    assert crz.compare(a, b, out=out, one_click=True) == 1
    assert "differs: run_metadata.json" in out.getvalue()
