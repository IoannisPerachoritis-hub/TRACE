"""GFF3 gene files: read directly, plain or gzip, recognised by their content (not their
name), with the gene rows' IDs and descriptions; a descriptions file overrides those. The
fixtures are handwritten in the styles of Ensembl, NCBI RefSeq and SGN/ITAG."""
import ast
import gzip
import io
import pathlib
import re
import zipfile

import pandas as pd
import pytest

from annotation import GENE_FILE_HELP, load_gene_annotation, load_gene_model

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "tests" / "data" / "example_v1"   # the first example, frozen
T = "\t"


def _gff(*rows, header="##gff-version 3\n"):
    return header + "".join(T.join(map(str, r)) + "\n" for r in rows)


ENSEMBL = _gff(
    ("1", "ITAG", "gene", 30726, 43941, ".", "-", ".",
     "ID=gene:Solyc01g005020.3;Name=SOS1;biotype=protein_coding;"
     "description=Salt Overly Sensitive 1 [Source:UniProtKB/TrEMBL]"),
    ("1", "ITAG", "mRNA", 30726, 43941, ".", "-", ".",
     "ID=transcript:Solyc01g005020.3.1;Parent=gene:Solyc01g005020.3"),
    ("1", "ITAG", "exon", 30726, 31000, ".", "-", ".", "Parent=transcript:Solyc01g005020.3.1"),
    ("2", "ITAG", "gene", 1000, 2500, ".", "+", ".",
     "ID=gene:Solyc02g000100.1;biotype=protein_coding"),
    header="##gff-version 3\n##sequence-region 1 1 98543444\n")

NCBI = _gff(
    ("NC_015438.3", "RefSeq", "region", 1, 98543444, ".", "+", ".",
     "ID=NC_015438.3:1..98543444;Dbxref=taxon:4081;chromosome=1"),
    ("NC_015438.3", "Gnomon", "gene", 5000, 9000, ".", "+", ".",
     "ID=gene-LOC101245629;Dbxref=GeneID:101245629;Name=LOC101245629;gbkey=Gene;"
     "gene_biotype=protein_coding;description=uncharacterized LOC101245629"),
    ("NC_015438.3", "RefSeq", "gene", 12000, 13000, ".", "-", ".",
     "ID=gene-rbcL;Name=rbcL;Note=plastid copy;product=ribulose bisphosphate carboxylase"),
    ("NC_015438.3", "RefSeq", "gene", 20000, 21000, ".", ".", ".",
     "ID=gene-orf1;Name=orf1;product=hypothetical protein"),
)

# SGN / ITAG style: no ##gff-version line (recognised by its first feature line), the
# description in Note, with GFF3's percent-encoding
ITAG = _gff(
    ("SL4.0ch01", "maker_ITAG", "gene", 279, 1437, ".", "+", ".",
     "ID=gene:Solyc01g004000.1;Alias=Solyc01g004000;Name=Solyc01g004000.1;"
     "Note=Protein kinase%2C putative%3B 100%25 identity (AHRD V3.3 *-* A0A0K9QB06)"),
    ("SL4.0ch01", "maker_ITAG", "mRNA", 279, 1437, ".", "+", ".",
     "ID=mRNA:Solyc01g004000.1.1;Parent=gene:Solyc01g004000.1"),
    ("SL4.0ch02", "maker_ITAG", "gene", 500, 900, ".", "-", ".",
     "ID=gene:Solyc02g004010.1;Name=Solyc02g004010.1"),
    header="")

TABLE = "Chr,Start,End,Strand,Gene_ID\n1,30726,43941,-,Solyc01g005020.3\n"


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_bytes(text.encode("utf-8"))
    return p


def _rows(df):
    return df[["Chr", "Start", "End", "Strand", "Gene_ID"]].values.tolist()


def test_an_ensembl_gff3_gives_its_gene_rows(tmp_path):
    # named .csv on purpose: the GUI saves every upload as gene_model.csv
    p = _write(tmp_path, "gene_model.csv", ENSEMBL)
    g = load_gene_model(p)
    assert list(g.columns) == ["Chr", "Start", "End", "Strand", "Gene_ID"]
    assert _rows(g) == [["1", 30726, 43941, "-", "Solyc01g005020.3"],
                        ["2", 1000, 2500, "+", "Solyc02g000100.1"]]
    d = load_gene_annotation(p)
    assert d["Description"].tolist() == ["Salt Overly Sensitive 1 [Source:UniProtKB/TrEMBL]", ""]


def test_an_ncbi_gff3_strips_gene_dash_and_takes_the_first_description(tmp_path):
    g = load_gene_annotation(_write(tmp_path, "GCF_000188115.gff", NCBI))
    assert g["Gene_ID"].tolist() == ["LOC101245629", "rbcL", "orf1"]   # no region row
    assert g["Chr"].tolist() == ["NC_015438.3"] * 3
    assert g["Strand"].tolist() == ["+", "-", "."]
    # the first of description, Note and product
    assert g["Description"].tolist() == ["uncharacterized LOC101245629", "plastid copy",
                                         "hypothetical protein"]


def test_an_itag_gff3_without_a_version_line_is_recognised_and_decoded(tmp_path):
    g = load_gene_annotation(_write(tmp_path, "ITAG4.0_gene_models.gff", ITAG))
    assert _rows(g) == [["1", 279, 1437, "+", "Solyc01g004000.1"],
                        ["2", 500, 900, "-", "Solyc02g004010.1"]]
    assert g["Description"].iloc[0] == ("Protein kinase, putative; 100% identity "
                                        "(AHRD V3.3 *-* A0A0K9QB06)")


def test_gzip_is_recognised_by_its_first_bytes_whatever_the_name(tmp_path):
    plain = load_gene_annotation(_write(tmp_path, "a.gff3", ENSEMBL))
    gz = tmp_path / "upload.csv"
    gz.write_bytes(gzip.compress(ENSEMBL.encode("utf-8")))
    pd.testing.assert_frame_equal(load_gene_annotation(gz), plain)
    tgz = tmp_path / "genes.csv.gz"                     # a gzipped table loads too
    tgz.write_bytes(gzip.compress(TABLE.encode("utf-8")))
    table = load_gene_model(_write(tmp_path, "genes.csv", TABLE))
    pd.testing.assert_frame_equal(load_gene_model(tgz), table)
    # an open file loads as its path does
    with open(gz, "rb") as fh:
        pd.testing.assert_frame_equal(load_gene_annotation(fh), plain)
    with open(tmp_path / "genes.csv", "rb") as fh:
        pd.testing.assert_frame_equal(load_gene_model(fh), table)


def test_a_table_named_gff3_is_read_as_a_table(tmp_path):
    assert _rows(load_gene_model(_write(tmp_path, "genes.gff3", TABLE))) == [
        ["1", 30726, 43941, "-", "Solyc01g005020.3"]]


def test_a_gff3_without_gene_rows_names_its_feature_types(tmp_path):
    p = _write(tmp_path, "transcripts.gff3", _gff(
        ("1", "x", "mRNA", 1, 100, ".", "+", ".", "ID=mRNA:a"),
        ("1", "x", "exon", 1, 50, ".", "+", ".", "Parent=mRNA:a"),
        ("1", "x", "exon", 60, 100, ".", "+", ".", "Parent=mRNA:a")))
    with pytest.raises(ValueError, match=re.escape("feature types are: exon (2), mRNA (1).")):
        load_gene_model(p)


def test_a_descriptions_file_overrides_the_gff3_for_the_genes_it_lists(tmp_path):
    descs = _write(tmp_path, "d.txt", "rbcL\tRuBisCO large subunit\n")
    g = load_gene_annotation(_write(tmp_path, "g.gff3", NCBI), descs)
    assert g["Description"].tolist() == ["uncharacterized LOC101245629",
                                         "RuBisCO large subunit", "hypothetical protein"]


def test_the_docs_script_writes_the_table_a_direct_read_gives(tmp_path, monkeypatch):
    doc = (ROOT / "docs" / "gene_model_upload.md").read_text(encoding="utf-8")
    section = doc.split("## Deriving a gene model from a GFF3", 1)[1]
    script = re.search(r"```python\n(.*?)\n```", section, re.S).group(1)
    for name, text in (("ensembl", ENSEMBL), ("ncbi", NCBI), ("itag", ITAG)):
        d = tmp_path / name
        d.mkdir()
        _write(d, "your_genome.gff3", text)
        monkeypatch.chdir(d)
        exec(compile(script, "gene_model_upload.md", "exec"), {})
        pd.testing.assert_frame_equal(load_gene_model(d / "gene_model.csv"),
                                      load_gene_model(d / "your_genome.gff3"))


def test_the_gff3_example_in_the_help_loads(tmp_path):
    blocks = re.findall(r"```\n(.*?)\n```", GENE_FILE_HELP, re.S)
    assert len(blocks) == 2                         # the table example, then the GFF3 one
    g = load_gene_annotation(_write(tmp_path, "example.gff3", blocks[1] + "\n"))
    assert _rows(g) == [["1", 30726, 43941, "-", "Solyc01g005020.3"]]
    assert g["Description"].tolist() == ["Salt Overly Sensitive 1"]


def test_both_gene_uploaders_accept_gff3():
    want = ["csv", "tsv", "txt", "gff3", "gff", "gff3.gz", "gff.gz"]
    for rel in ("pages/_ld_tabs/tab_gene_annotation.py", "pages/GWAS_analysis.py"):
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        types = [ast.literal_eval(kw.value) for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "file_uploader"
                 and any(k.arg == "help" and getattr(k.value, "id", "") == "GENE_FILE_HELP"
                         for k in n.keywords)
                 for kw in n.keywords if kw.arg == "type"]
        assert types == [want], rel


@pytest.mark.skipif(not (EXAMPLES / "example.vcf.gz").exists(),
                    reason="tests/data/example_v1/example.vcf.gz not present")
def test_the_cli_annotates_with_a_gzipped_gff3(tmp_path):
    from cli import _build_parser, run_pipeline
    # one gene spanning each chromosome, so every block holds its chromosome's gene
    gff = _gff(*[(f"chr{c:02d}", "x", "gene", 1, 900000000, ".", "+", ".",
                  f"ID=gene:G{c};description=gene of chromosome {c}") for c in range(1, 13)])
    gm = tmp_path / "genes.gff3.gz"
    gm.write_bytes(gzip.compress(gff.encode("utf-8")))
    run_pipeline(_build_parser().parse_args([
        "--vcf", str(EXAMPLES / "example.vcf.gz"), "--pheno", str(EXAMPLES / "example_pheno.csv"),
        "--trait", "Trait1", "--model", "mlm", "--output", str(tmp_path / "out"),
        "--no-report", "--no-plots", "--ld-r2", "0.1", "--ld-flank-kb", "200",
        "--species", "custom", "--gene-model", str(gm)]))
    zf = zipfile.ZipFile(next((tmp_path / "out").glob("*.zip")))
    blocks = pd.read_csv(io.BytesIO(zf.read("tables/LD_blocks_annotated_MLM.csv")),
                         dtype={"Chr": str})
    assert len(blocks)
    for _, b in blocks.iterrows():
        assert b["overlapping_descriptions"] == f"G{b['Chr']}: gene of chromosome {b['Chr']}"
