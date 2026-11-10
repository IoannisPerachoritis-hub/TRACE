"""Gene files: the loader, its "?" format help, the upload checks, the annotation table and
the CLI's --gene-desc.

annotation.GENE_FILE_HELP / DESCRIPTIONS_FILE_HELP are the "?" of every gene-file uploader
(the Gene Annotation tab and the GWAS page's One-Click overrides). Every example in them is
run here against the loader, so the texts cannot drift from what the loader does. The
Gene Annotation tab lists the intervals the CLI annotates (blocks and unblocked SNPs).
"""
import ast
import io
import json
import pathlib
import re
import zipfile

import pandas as pd
import pytest

from annotation import (
    ANNOTATION_TABLE_COLUMNS,
    DESCRIPTIONS_FILE_HELP,
    GENE_FILE_HELP,
    annotate_candidate_intervals,
    canon_chr,
    chromosomes_without_genes,
    load_gene_annotation,
    load_gene_model,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def _gene_example():
    """The gene file example of GENE_FILE_HELP (its fenced block)."""
    return re.search(r"```\n(.*?)\n```", GENE_FILE_HELP, re.S).group(1)


def _desc_example():
    """(gene ID, description) of DESCRIPTIONS_FILE_HELP's example."""
    tail = DESCRIPTIONS_FILE_HELP.split("Example", 1)[1]
    gid, desc = re.findall(r"`([^`]+)`", tail)[:2]
    return gid, desc


def _write(tmp_path, name, text, bom=False):
    p = tmp_path / name
    p.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))
    return p


_ROW = [["1", 30726, 43941, "-", "Solyc01g005020.3"]]
_COLS = ["Chr", "Start", "End", "Strand", "Gene_ID"]


def test_the_gene_file_example_loads_and_is_the_bundled_row(tmp_path):
    g = load_gene_model(_write(tmp_path, "genes.csv", _gene_example() + "\n"))
    assert g[_COLS].values.tolist() == _ROW
    # data/Sol_genes_SL3.csv line 10, with its index column dropped
    line10 = (ROOT / "data" / "Sol_genes_SL3.csv").read_text(encoding="utf-8").splitlines()[9]
    assert line10.split(",", 1)[1] == _gene_example().splitlines()[1]


def test_semicolon_and_tab_separated_files_and_an_excel_bom_load(tmp_path):
    text = _gene_example()
    for name, body, bom in (("semi.csv", text.replace(",", ";"), False),
                            ("tab.tsv", text.replace(",", "\t"), False),
                            ("excel.csv", text, True)):
        g = load_gene_model(_write(tmp_path, name, body + "\n", bom=bom))
        assert g[_COLS].values.tolist() == _ROW, name
    assert "semicolon" in GENE_FILE_HELP and "CSV UTF-8" in GENE_FILE_HELP


def test_the_chromosome_spellings_in_the_help_are_one_chromosome():
    para = GENE_FILE_HELP.split("Chromosome names", 1)[1].split("\n\n", 1)[0]
    names = re.findall(r"`([^`]+)`", para)
    assert names == ["chr01", "Chr1", "SL2.50ch01", "1"]
    assert {canon_chr(n) for n in names} == {"1"}


def test_the_description_example_attaches_with_and_without_its_version(tmp_path):
    gid, desc = _desc_example()
    assert (gid, desc) == ("Solyc01g005020.3", "Salt Overly Sensitive 1")
    # data/SL3.1_descriptions.txt line 616
    line616 = (ROOT / "data" / "SL3.1_descriptions.txt").read_text(encoding="utf-8").splitlines()[615]
    assert line616 == f"{gid}\t{desc}"
    base = re.sub(r"(\.\d+)+$", "", gid)
    assert f"`{gid}` matches `{base}`" in DESCRIPTIONS_FILE_HELP
    cases = ((gid, f"{gid}\t{desc}"),      # the example as it stands
             (base, f"{gid}\t{desc}"),     # a gene file without the version ending
             (gid, f"{base}\t{desc}"))     # descriptions without it
    for gene_id, desc_line in cases:
        genes = _write(tmp_path, "g.csv", f"Chr,Start,End,Gene_ID\n1,30726,43941,{gene_id}\n")
        descs = _write(tmp_path, "d.txt", f"# a comment line\n{desc_line}\n")
        assert load_gene_annotation(genes, descs)["Description"].tolist() == [desc], gene_id


def test_a_comma_separated_descriptions_file_adds_no_descriptions(tmp_path):
    gid, desc = _desc_example()
    genes = _write(tmp_path, "g.csv", f"Chr,Start,End,Gene_ID\n1,30726,43941,{gid}\n")
    descs = _write(tmp_path, "d.csv", f"{gid},{desc}\n")
    assert load_gene_annotation(genes, descs)["Description"].tolist() == [""]


@pytest.mark.parametrize("header,message", [
    ("Chr,Start,End,Gene_ID,Name", "Use one column for the gene ID; this file has Gene_ID and Name."),
    ("Chr,Start,Begin,End,Gene_ID", "Use one column for the start; this file has Start and Begin."),
])
def test_two_columns_for_one_field_are_refused_by_name(tmp_path, header, message):
    p = _write(tmp_path, "two.csv", header + "\n" + ",".join(["1"] * len(header.split(","))) + "\n")
    with pytest.raises(ValueError) as e:
        load_gene_model(p)
    assert str(e.value) == message


def test_the_bundled_tables_have_one_column_per_field():
    for name in ("Sol_genes_SL3.csv", "Sol_genes.csv"):
        assert len(load_gene_model(ROOT / "data" / name)) > 1000, name


def test_chromosomes_without_genes_are_named():
    genes = pd.DataFrame({"Chr": ["1", "chr03"], "Start": [1, 1], "End": [2, 2], "Gene_ID": ["a", "b"]})
    geno = ["1", "2", "chr03", "SL2.50ch04", "2", "10"]
    assert chromosomes_without_genes(geno, genes) == ["2", "4", "10"]
    assert chromosomes_without_genes(["chr1", "Chr3"], genes) == []


def test_the_annotation_table_lists_each_interval_with_its_genes():
    intervals = pd.DataFrame({
        "Type": ["LD block", "unblocked SNP"], "Chr": ["1", "2"],
        "Interval_start": [100, 5000], "Interval_end": [900, 6000], "Lead_SNP": ["s1", "s2"],
    })
    genes = pd.DataFrame({"Chr": ["1", "1", "2"], "Start": [50, 950, 10], "End": [150, 990, 20],
                          "Gene_ID": ["G1", "G2", "G3"], "Description": ["kinase", "", "x"]})
    table, line = annotate_candidate_intervals(intervals, genes)
    assert list(table.columns) == ANNOTATION_TABLE_COLUMNS
    assert table[["Type", "Chr", "Interval start", "Interval end", "Lead SNP",
                  "Number of genes", "Genes"]].values.tolist() == [
        ["LD block", "1", 100, 900, "s1", 1, "G1: kinase"],
        ["unblocked SNP", "2", 5000, 6000, "s2", 0, ""]]
    assert line == "2 intervals, 1 gene"


def test_both_uploader_places_use_the_shared_texts():
    for rel in ("pages/_ld_tabs/tab_gene_annotation.py", "pages/GWAS_analysis.py"):
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        helps = [kw.value.id for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "attr", None) == "file_uploader"
                 for kw in n.keywords if kw.arg == "help" and isinstance(kw.value, ast.Name)]
        assert sorted(helps) == ["DESCRIPTIONS_FILE_HELP", "GENE_FILE_HELP"], rel


# ── end to end on the bundled example ──
_needs_examples = pytest.mark.skipif(not (EXAMPLES / "example.vcf.gz").exists(),
                                     reason="examples/example.vcf.gz not present")


# the example forms LD blocks at r² 0.1 (and then no unblocked SNP), and with the
# defaults one unblocked SNP and no block: the two settings cover both interval types
_LD = ["--ld-r2", "0.1", "--ld-flank-kb", "200"]


def _run(out_dir, extra=()):
    from cli import _build_parser, run_pipeline
    args = _build_parser().parse_args([
        "--vcf", str(EXAMPLES / "example.vcf.gz"),
        "--pheno", str(EXAMPLES / "example_pheno.csv"),
        "--trait", "Trait1", "--model", "mlm", "--output", str(out_dir),
        "--no-report", "--no-plots", *extra])
    run_pipeline(args)
    return zipfile.ZipFile(next(pathlib.Path(out_dir).glob("*.zip")))


def _table(zf, name, cols=None):
    if name not in zf.namelist():
        return pd.DataFrame(columns=cols or [])
    return pd.read_csv(io.BytesIO(zf.read(name)), dtype={"Chr": str})


@_needs_examples
def test_a_custom_gene_model_with_gene_desc_gives_descriptions(tmp_path):
    # one gene spanning each chromosome, so every interval holds its chromosome's gene
    genes = _write(tmp_path, "genes.csv", "Chr,Start,End,Gene_ID\n"
                   + "".join(f"{c},1,900000000,G{c}\n" for c in range(1, 13)))
    descs = _write(tmp_path, "descs.txt", "".join(f"G{c}\tgene of chromosome {c}\n"
                                                    for c in range(1, 13)))
    zf = _run(tmp_path / "out", [*_LD, "--species", "custom", "--gene-model", str(genes),
                                 "--gene-desc", str(descs)])
    blocks = _table(zf, "tables/LD_blocks_annotated_MLM.csv")
    assert len(blocks)
    for _, b in blocks.iterrows():
        assert b["overlapping_descriptions"] == f"G{b['Chr']}: gene of chromosome {b['Chr']}"
    meta = json.loads(zf.read("run_metadata.json"))
    assert meta["Gene descriptions"] == str(descs)
    keys = list(meta)
    assert keys.index("Gene descriptions") == keys.index("Gene model status") + 1


@_needs_examples
def test_without_gene_desc_the_metadata_has_no_descriptions_key(tmp_path):
    meta = json.loads(_run(tmp_path / "out").read("run_metadata.json"))
    assert "Gene descriptions" not in meta


@_needs_examples
@pytest.mark.parametrize("extra,kind", [(_LD, "LD block"), ([], "unblocked SNP")])
def test_the_gui_lists_the_intervals_the_cli_annotates(tmp_path, extra, kind):
    from gwas.isolated import default_edge_flank_bp
    from pages._ld_tabs._intervals import _candidate_intervals

    zf = _run(tmp_path / "out", extra)
    gwas = _table(zf, "tables/GWAS_Trait1.csv")
    blocks = _table(zf, "tables/LD_blocks_annotated_MLM.csv",
                    ["Chr", "Start (bp)", "End (bp)", "lead_snp", "SNP_IDs",
                     "Interval_start", "Interval_end"])
    iso = _table(zf, "tables/Isolated_SNP_intervals_MLM.csv",
                 ["Chr", "Start (bp)", "End (bp)", "lead_snp"])
    meta = json.loads(zf.read("run_metadata.json"))
    # the GUI's list (pages/_ld_tabs/_intervals.py), from the run's blocks, results and rule
    gui = _candidate_intervals(
        blocks[["Chr", "Start (bp)", "End (bp)", "lead_snp", "SNP_IDs"]], gwas,
        "Bonferroni (α = 0.05)", None, None,
        gwas["Chr"].astype(str).to_numpy(), gwas["Pos"].to_numpy(), gwas["SNP"].astype(str).to_numpy(),
        default_edge_flank_bp(None, int(meta["LD flank (kb)"])))
    cli = sorted([("LD block", str(r["Chr"]), int(r["Interval_start"]), int(r["Interval_end"]),
                   str(r["lead_snp"])) for _, r in blocks.iterrows()]
                 + [("unblocked SNP", str(r["Chr"]), int(r["Start (bp)"]), int(r["End (bp)"]),
                     str(r["lead_snp"])) for _, r in iso.iterrows()])
    got = sorted((str(r["Type"]), str(r["Chr"]), int(r["Interval_start"]), int(r["Interval_end"]),
                  str(r["Lead_SNP"])) for _, r in gui.iterrows())
    assert got == cli and any(k == kind for k, *_ in got)
