"""Every file of the results ZIP is documented. The CLI runs on the example with three
models, a short subsampling and --export-qc, for two traits (so a combined ZIP too); each
member name is matched to a file pattern, and every pattern is listed both in
docs/outputs.md and on the Help page (pages/z_Help.py)."""
import pathlib
import re
import zipfile

import pytest

from cli import _build_parser, run_pipeline

ROOT = pathlib.Path(__file__).resolve().parents[1]
EX = ROOT / "examples"
M = r"(?:MLM|MLMM|FarmCPU)"
# (regex for a member name, the pattern as the docs write it); the first match wins
PATTERNS = [
    (rf"tables/GWAS_(?:MLMM|FarmCPU)_[^/]+\.csv", "tables/GWAS_<model>_<trait>.csv"),
    (r"tables/GWAS_[^/]+\.csv", "tables/GWAS_<trait>.csv"),
    (rf"tables/Significant_SNPs_{M}\.csv", "tables/Significant_SNPs_<model>.csv"),
    (rf"tables/Unblocked_SNPs_{M}\.csv", "tables/Unblocked_SNPs_<model>.csv"),
    (rf"tables/Isolated_SNP_intervals_{M}\.csv", "tables/Isolated_SNP_intervals_<model>.csv"),
    (rf"tables/Isolated_SNP_candidate_genes_{M}\.csv", "tables/Isolated_SNP_candidate_genes_<model>.csv"),
    (rf"tables/LD_blocks_annotated_{M}\.csv", "tables/LD_blocks_annotated_<model>.csv"),
    (rf"tables/LD_triage_{M}\.csv", "tables/LD_triage_<model>.csv"),
    (r"tables/Haplotype_assignments_[^/]+_(?:MLMM|FarmCPU)\.csv", "tables/Haplotype_assignments_<trait>_<model>.csv"),
    (r"tables/Haplotype_assignments_[^/]+\.csv", "tables/Haplotype_assignments_<trait>.csv"),
    (rf"tables/SNP_view_index_{M}\.csv", "tables/SNP_view_index_<model>.csv"),
    (r"tables/(?:CrossModel_Consensus|PC_diagnostics_spectrum|QC_report_summary|QC_sample_heterozygosity"
     r"|QC_genotype_matrix|QC_genotype_raw|QC_snp_map|QC_phenotype|Subsampling_SNP_stability"
     r"|Subsampling_block_stability|Subsampling_rep_metadata)\.csv", None),     # the name itself
    (rf"figures/Manhattan_{M}\.png", "figures/Manhattan_<model>.png"),
    (rf"figures/QQ_{M}\.png", "figures/QQ_<model>.png"),
    (r"figures/LD_heatmap_Chr[^_/]+_\d+_\d+\.png", "figures/LD_heatmap_Chr<c>_<start>_<end>.png"),
    (r"figures/(?:PCA_scree|PCA_scatter|QC_report_distributions)\.png", None),
    (r"figures/Subsampling_discovery_freq_[^/]+\.png", "figures/Subsampling_discovery_freq_<trait>.png"),
    (r"(?:report\.html|run_metadata\.json|failed_traits\.txt)", None),
]


def _pattern(member):
    for rx, pattern in PATTERNS:
        if re.fullmatch(rx, member):
            return pattern or member
    return None


@pytest.mark.slow_gui
@pytest.mark.skipif(not (EX / "example.vcf.gz").exists(), reason="examples/example.vcf.gz not present")
def test_every_member_of_the_zip_is_documented(tmp_path):
    run_pipeline(_build_parser().parse_args([
        "--vcf", str(EX / "example.vcf.gz"), "--pheno", str(EX / "example_pheno.csv"),
        "--trait", "Trait1", "Trait2", "--model", "mlm", "mlmm", "farmcpu",
        "--subsampling", "--boot-reps", "3", "--export-qc", "--output", str(tmp_path)]))
    zips = sorted(tmp_path.glob("*.zip"))
    assert len(zips) == 3                                   # two traits and the combined one
    seen, folders = set(), set()
    for z in zips:
        for name in zipfile.ZipFile(z).namelist():
            head, _, rest = name.partition("/")
            if head in ("Trait1", "Trait2"):                # a combined ZIP's trait folder
                folders.add(head)
                name = rest
            pattern = _pattern(name)
            assert pattern is not None, f"{z.name}: {name} matches no documented file"
            seen.add(pattern)
    assert folders == {"Trait1", "Trait2"}
    assert "figures/LD_heatmap_Chr<c>_<start>_<end>.png" in seen   # the example forms a block
    docs = (ROOT / "docs" / "outputs.md").read_text(encoding="utf-8")
    helps = (ROOT / "pages" / "z_Help.py").read_text(encoding="utf-8")
    for pattern in sorted(seen | {"<trait>/", "failed_traits.txt"}):
        assert f"`{pattern}`" in docs, f"docs/outputs.md does not list {pattern}"
        assert f"`{pattern}`" in helps, f"the Help page does not list {pattern}"
