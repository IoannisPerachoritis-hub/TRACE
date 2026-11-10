"""The bundled example forms an LD block at TRACE's defaults. The CLI at its defaults
(MLM, Bonferroni, r2 0.6, the LD window from the estimated decay) finds the QTL that
examples/simulate_example.py places on chromosome 2: one LD block around it, the block's
haplotype test, its LD heatmap and its sample-to-haplotype file."""
import io
import json
import pathlib
import zipfile

import pandas as pd
import pytest

from cli import _build_parser, run_pipeline

ROOT = pathlib.Path(__file__).resolve().parents[1]
EX = ROOT / "examples"
QTL_CHR, QTL_POS = "2", 1_010_000          # simulate_example.py: QTL_CHROM, QTL_IDX

pytestmark = [pytest.mark.slow_gui,
              pytest.mark.skipif(not (EX / "example.vcf.gz").exists(),
                                 reason="examples/example.vcf.gz not present")]


def test_the_example_gives_a_block_a_haplotype_test_and_a_heatmap_at_the_defaults(tmp_path):
    run_pipeline(_build_parser().parse_args([
        "--vcf", str(EX / "example.vcf.gz"), "--pheno", str(EX / "example_pheno.csv"),
        "--trait", "Trait1", "--output", str(tmp_path)]))
    zf = zipfile.ZipFile(next(tmp_path.glob("*.zip")))
    names = zf.namelist()
    gwas = pd.read_csv(io.BytesIO(zf.read("tables/GWAS_Trait1.csv")), dtype={"Chr": str})
    hits = gwas[gwas["Significant_Bonf"]]
    assert len(hits) and (hits["Chr"].str.lstrip("0") == QTL_CHR).all()
    assert ((hits["Pos"] - QTL_POS).abs() <= 100_000).all()        # all at the QTL

    blocks = pd.read_csv(io.BytesIO(zf.read("tables/LD_blocks_annotated_MLM.csv")),
                         dtype={"Chr": str})
    at_qtl = blocks[(blocks["Chr"] == QTL_CHR) & (blocks["Start (bp)"] <= QTL_POS)
                    & (blocks["End (bp)"] >= QTL_POS)]
    assert len(at_qtl) == 1, blocks
    block = at_qtl.iloc[0]
    assert block["Hap_PValue"] < 0.05 and block["Hap_n_tested"] >= 2      # its haplotype test
    assert (f"figures/LD_heatmap_Chr{QTL_CHR}_{block['Start (bp)']}_{block['End (bp)']}.png"
            in names)
    assert "tables/Haplotype_assignments_Trait1.csv" in names
    meta = json.loads(zf.read("run_metadata.json"))
    assert meta["LD flank (kb)"] != 300                     # the window came from a decay
