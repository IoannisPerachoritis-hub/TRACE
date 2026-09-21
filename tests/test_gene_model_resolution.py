"""Gene-model resolution: one answer, shared by every consumer, honest when absent.

The defect this guards: the gene models were not in the wheel, so an installed
TRACE silently produced no candidate-gene tables while ``run_metadata.json``
still named a gene model. Two halves are pinned here — the resolver itself, and
the end-to-end behaviour when the bundled tables are missing (a warning that
names the path searched, a manifest that records the absence, and no
candidate-gene table).

The CLI cases drive the shipped ``examples/`` data because it carries a planted
signal: the isolated-SNP rescue fires, so the presence/absence of
``Isolated_SNP_candidate_genes_*.csv`` is a real discriminator rather than a
vacuous assertion on a run that would emit nothing either way.
"""
import contextlib
import json
import logging
import pathlib
import zipfile

import pytest

import annotation
from annotation import resolve_gene_model
from cli import _build_parser, run_pipeline

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_EXAMPLES = _REPO_ROOT / "examples"


class _Collector(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def capture_cli_log():
    """Collect records from the "cli" logger.

    pytest's ``caplog`` cannot be used here: pysnptools' ``Local.__init__``
    (util/mapreduce1/runner/local.py) removes EVERY handler from the root logger
    and installs its own, and it runs on every FaST-LMM scan. Anything the CLI
    logs after the first scan -- the whole post-GWAS phase, including the
    gene-model warning -- is therefore invisible to a root-attached handler.
    A handler on the "cli" logger itself is dispatched regardless of what the
    root logger holds, so it survives.
    """
    logger = logging.getLogger("cli")
    handler = _Collector()
    prev_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        yield handler.messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)


# ── the resolver ─────────────────────────────────────────────


def _fake_data_dir(tmp_path, names=("Sol_genes_SL3.csv", "SL3.1_descriptions.txt",
                                    "Sol_genes.csv", "ITAG4.0_annotation.txt")):
    d = tmp_path / "data"
    d.mkdir()
    for n in names:
        (d / n).write_text("stub\n")
    return d


def test_tomato_sl3_resolves_to_the_sl3_pair(tmp_path):
    d = _fake_data_dir(tmp_path)
    res = resolve_gene_model("tomato", genome_build="SL3", data_dir=d)
    assert res.found
    assert res.gene_model.name == "Sol_genes_SL3.csv"
    assert res.descriptions.name == "SL3.1_descriptions.txt"
    assert res.status.startswith("used: ")


def test_tomato_sl4_resolves_to_the_itag4_pair(tmp_path):
    d = _fake_data_dir(tmp_path)
    res = resolve_gene_model("tomato", genome_build="SL4", data_dir=d)
    assert res.found
    assert res.gene_model.name == "Sol_genes.csv"
    assert res.descriptions.name == "ITAG4.0_annotation.txt"


def test_override_takes_precedence_and_keeps_species_descriptions(tmp_path):
    d = _fake_data_dir(tmp_path)
    mine = tmp_path / "mine.csv"
    mine.write_text("stub\n")
    res = resolve_gene_model("tomato", genome_build="SL3", override=mine, data_dir=d)
    assert res.found
    assert res.gene_model == mine
    # descriptions still come from the species defaults -- the shipped behaviour
    assert res.descriptions.name == "SL3.1_descriptions.txt"


def test_species_without_a_bundled_table_and_no_override(tmp_path):
    d = _fake_data_dir(tmp_path)
    res = resolve_gene_model("custom", genome_build="SL3", data_dir=d)
    assert not res.found
    assert res.gene_model is None
    assert res.searched is None
    assert "custom" in res.status and "no --gene-model" in res.status


def test_missing_data_dir_reports_the_path_it_searched(tmp_path):
    missing = tmp_path / "nowhere"
    res = resolve_gene_model("tomato", genome_build="SL3", data_dir=missing)
    assert not res.found
    assert res.gene_model is None
    assert res.searched is not None
    assert "not found: searched" in res.status
    assert "Sol_genes_SL3.csv" in res.status


def test_descriptions_are_none_when_only_the_gene_table_exists(tmp_path):
    d = _fake_data_dir(tmp_path, names=("Sol_genes_SL3.csv",))
    res = resolve_gene_model("tomato", genome_build="SL3", data_dir=d)
    assert res.found
    assert res.descriptions is None


# ── end to end: the bundled tables present vs absent ─────────


def _run_example(out_dir):
    args = _build_parser().parse_args([
        "--vcf", str(_EXAMPLES / "example.vcf.gz"),
        "--pheno", str(_EXAMPLES / "example_pheno.csv"),
        "--trait", "Trait1",
        "--model", "mlm",
        "--output", str(out_dir),
        "--no-report", "--no-plots",
    ])
    run_pipeline(args)
    zips = list(out_dir.glob("*.zip"))
    assert len(zips) == 1, f"expected one ZIP, got {zips}"
    with zipfile.ZipFile(zips[0]) as zf:
        names = sorted(zf.namelist())
        meta = json.loads(zf.read("run_metadata.json"))
    return names, meta


@pytest.mark.skipif(not (_EXAMPLES / "example.vcf.gz").exists(),
                    reason="examples/example.vcf.gz not present")
def test_bundled_gene_model_is_used_and_recorded(tmp_path):
    """Positive control: the gene model resolves, so the candidate-gene table is
    written and the manifest names the file that produced it."""
    with capture_cli_log() as messages:
        names, meta = _run_example(tmp_path / "out_ok")

    assert any("Isolated_SNP_candidate_genes" in n for n in names), names
    assert meta["Gene model"] == "Sol_genes_SL3.csv"
    assert meta["Gene model status"].startswith("used: ")
    assert not any("Gene model unavailable" in m for m in messages), messages


@pytest.mark.skipif(not (_EXAMPLES / "example.vcf.gz").exists(),
                    reason="examples/example.vcf.gz not present")
def test_missing_gene_model_warns_and_is_recorded_as_absent(tmp_path, monkeypatch):
    """The regression: with the bundled tables gone, the run must say so — in the
    log, in the manifest, and by the candidate-gene table simply not being there.

    Before the fix this path was silent and the manifest still named a gene model.
    """
    empty = tmp_path / "empty_data"
    empty.mkdir()
    monkeypatch.setattr(annotation, "_BUNDLED_DATA_DIR", empty)

    with capture_cli_log() as messages:
        names, meta = _run_example(tmp_path / "out_missing")

    # (i) the warning fires and names the path that was searched
    warnings = [m for m in messages if "Gene model unavailable" in m]
    assert warnings, f"no gene-model warning in: {messages}"
    assert "not found: searched" in warnings[0]
    assert str(empty / "Sol_genes_SL3.csv") in warnings[0]

    # (ii) the manifest records the real state, not the configured filename
    assert meta["Gene model"] is None
    assert meta["Gene model status"].startswith("not found: searched")
    assert "Sol_genes_SL3.csv" in meta["Gene model status"]

    # (iii) and no candidate-gene table is claimed
    assert not any("Isolated_SNP_candidate_genes" in n for n in names), names
