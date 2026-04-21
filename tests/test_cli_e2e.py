"""End-to-end integration tests for cli.py — runs the full GWAS pipeline
on synthetic data to verify correct outputs are produced."""
import io
import textwrap
import zipfile

import numpy as np
import pandas as pd

from cli import _build_parser, run_pipeline


# ── Synthetic data builders ──────────────────────────────────


def _build_synthetic_vcf(
    n_samples=50, n_snps_per_chr=40, chroms=("1", "2", "3"), rng_seed=42
):
    """Build a minimal valid VCF string with random biallelic genotypes.

    Produces ``n_snps_per_chr * len(chroms)`` SNPs across the given
    chromosomes, with MAF ~0.3 (genotypes drawn from Binomial(2, 0.3)).
    """
    rng = np.random.default_rng(rng_seed)
    sample_names = [f"S{i:03d}" for i in range(n_samples)]

    header = textwrap.dedent("""\
        ##fileformat=VCFv4.2
        ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
    """).rstrip()
    col_line = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(sample_names)
    lines = [header, col_line]

    for chrom in chroms:
        for j in range(n_snps_per_chr):
            pos = (j + 1) * 1000
            snp_id = f"chr{chrom}_{pos}"
            gts = rng.binomial(2, 0.3, size=n_samples)
            gt_strs = []
            for g in gts:
                if g == 0:
                    gt_strs.append("0/0")
                elif g == 1:
                    gt_strs.append("0/1")
                else:
                    gt_strs.append("1/1")
            line = f"{chrom}\t{pos}\t{snp_id}\tA\tT\t.\tPASS\t.\tGT\t" + "\t".join(gt_strs)
            lines.append(line)

    return "\n".join(lines) + "\n"


def _build_synthetic_pheno(n_samples=50, trait_name="TestTrait", rng_seed=42):
    """Build a phenotype DataFrame matching the synthetic VCF samples."""
    rng = np.random.default_rng(rng_seed)
    sample_names = [f"S{i:03d}" for i in range(n_samples)]
    return pd.DataFrame({
        "SampleID": sample_names,
        trait_name: rng.normal(10, 2, size=n_samples),
    })


def _write_test_data(tmp_path, n_samples=50, trait_name="TestTrait"):
    """Write synthetic VCF + phenotype to tmp_path, return file paths."""
    vcf_text = _build_synthetic_vcf(n_samples=n_samples)
    vcf_path = tmp_path / "test.vcf"
    vcf_path.write_text(vcf_text)

    pheno_df = _build_synthetic_pheno(n_samples=n_samples, trait_name=trait_name)
    pheno_path = tmp_path / "pheno.csv"
    pheno_df.to_csv(pheno_path, index=False)

    return vcf_path, pheno_path


# ── Tests ────────────────────────────────────────────────────


class TestGWASPipelineE2E:
    """End-to-end test of the GWAS-only pipeline."""

    def test_gwas_pipeline_e2e(self, tmp_path):
        vcf_path, pheno_path = _write_test_data(tmp_path)
        output_dir = tmp_path / "results"

        parser = _build_parser()
        args = parser.parse_args([
            "--vcf", str(vcf_path),
            "--pheno", str(pheno_path),
            "--trait", "TestTrait",
            "--output", str(output_dir),
            "--model", "mlm",
            "--no-report",
            "--no-plots",
            "--maf", "0.01",
            "--mac", "1",
            "--n-pcs", "2",
        ])

        ctx = run_pipeline(args)

        # ── Verify returned context ──────────────────────
        assert ctx is not None
        assert "gwas_df" in ctx
        assert "geno_imputed" in ctx
        assert "y" in ctx
        assert "meff_val" in ctx

        # ── Verify ZIP-only output ─────────────────────────
        zips = list(output_dir.glob("*.zip"))
        assert len(zips) == 1, f"Expected exactly 1 ZIP, got {len(zips)}"

        # No loose CSVs or PNGs outside the ZIP
        loose_csvs = list(output_dir.glob("*.csv"))
        assert len(loose_csvs) == 0, f"Loose CSVs found outside ZIP: {loose_csvs}"

        # Read GWAS CSV from inside the ZIP
        with zipfile.ZipFile(zips[0]) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            gwas_csv_name = [n for n in csv_names if "GWAS_TestTrait" in n]
            assert len(gwas_csv_name) >= 1, f"No GWAS CSV in ZIP. Contents: {zf.namelist()}"
            with zf.open(gwas_csv_name[0]) as f:
                gwas_df = pd.read_csv(io.TextIOWrapper(f))

        expected_cols = {"SNP", "Chr", "Pos", "PValue"}
        assert expected_cols.issubset(set(gwas_df.columns)), (
            f"Missing columns: {expected_cols - set(gwas_df.columns)}"
        )
        assert len(gwas_df) > 0, "GWAS CSV is empty"
        assert (gwas_df["PValue"] >= 0).all(), "Negative p-values found"
        assert (gwas_df["PValue"] <= 1).all(), "p-values > 1 found"


    def test_export_qc_files_in_zip(self, tmp_path):
        vcf_path, pheno_path = _write_test_data(tmp_path)
        output_dir = tmp_path / "results_qc"

        parser = _build_parser()
        args = parser.parse_args([
            "--vcf", str(vcf_path),
            "--pheno", str(pheno_path),
            "--trait", "TestTrait",
            "--output", str(output_dir),
            "--model", "mlm",
            "--no-report",
            "--no-plots",
            "--maf", "0.01",
            "--mac", "1",
            "--n-pcs", "2",
            "--export-qc",
        ])

        ctx = run_pipeline(args)
        zips = list(output_dir.glob("*.zip"))
        assert len(zips) == 1

        with zipfile.ZipFile(zips[0]) as zf:
            names = zf.namelist()
            # Check QC export files exist
            geno_csvs = [n for n in names if "QC_genotype_matrix" in n]
            map_csvs = [n for n in names if "QC_snp_map" in n]
            pheno_csvs = [n for n in names if "QC_phenotype" in n]
            assert len(geno_csvs) == 1, f"Expected QC_genotype_matrix.csv in ZIP: {names}"
            assert len(map_csvs) == 1, f"Expected QC_snp_map.csv in ZIP: {names}"
            assert len(pheno_csvs) == 1, f"Expected QC_phenotype.csv in ZIP: {names}"

            # Validate genotype matrix shape
            with zf.open(geno_csvs[0]) as f:
                geno_df = pd.read_csv(io.TextIOWrapper(f))
            assert "SampleID" in geno_df.columns
            n_snps = geno_df.shape[1] - 1  # minus SampleID column
            assert n_snps == ctx["geno_imputed"].shape[1]
            assert geno_df.shape[0] == ctx["geno_imputed"].shape[0]

            # Validate SNP map
            with zf.open(map_csvs[0]) as f:
                snp_map = pd.read_csv(io.TextIOWrapper(f))
            assert set(snp_map.columns) == {"SNP_ID", "Chr", "Pos", "Ref", "Alt"}
            assert len(snp_map) == n_snps

            # Validate phenotype
            with zf.open(pheno_csvs[0]) as f:
                pheno_df = pd.read_csv(io.TextIOWrapper(f))
            assert "SampleID" in pheno_df.columns
            assert "TestTrait" in pheno_df.columns
            assert len(pheno_df) == ctx["geno_imputed"].shape[0]


