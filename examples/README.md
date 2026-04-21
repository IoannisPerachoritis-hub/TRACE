# Quick-Start Tutorial

This directory contains a small synthetic example to verify your TRACE installation end-to-end in under 30 seconds.

## One-shot reproducible run

```bash
bash examples/run_example.sh
```

That script:

1. Runs `simulate_example.py`, which writes `example.vcf.gz` (~50 samples, ~510 SNPs across 3 chromosomes; one embedded QTL on chromosome 2 with h^2 ~ 0.4) and `example_pheno.csv` (`Trait1` carries the QTL signal; `Trait2` is pure noise).
2. Runs the TRACE MLM pipeline on `Trait1` and writes a results ZIP to `examples/output/`.

## Manual CLI invocation

If you want to call the CLI directly (e.g., to try a different model):

```bash
trace-gwas \
  --vcf  examples/example.vcf.gz \
  --pheno examples/example_pheno.csv \
  --trait Trait1 \
  --output examples/output/ \
  --model mlm
```

## Interactive (Streamlit)

```bash
streamlit run app.py
```

Navigate to **GWAS Analysis**, upload `example.vcf.gz` and `example_pheno.csv`, select `Trait1`, and click **Run One-Click Pipeline**.

## Key CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--vcf` | Input VCF file (bgzipped or uncompressed) | Required |
| `--pheno` | Phenotype CSV (SampleID + trait columns) | Required |
| `--trait` | Trait column name | Required |
| `--output` | Output directory | Required |
| `--model` | GWAS models to run (`mlm`, `mlmm`, `farmcpu`) | `mlm` |
| `--n-pcs` | Number of principal components | 4 |
| `--auto-pcs` | Automatic PC selection via lambda-GC band | Off |
| `--maf` | Minor allele frequency filter | 0.05 |
| `--miss` | Maximum SNP missingness | 0.10 |
| `--seed` | RNG seed for subsampling / permutation | 42 |
| `--sig-thresh` | Significance threshold (`meff`, `bonferroni`, `fdr`) | `meff` |
| `--subsampling` | Enable subsampling stability analysis | Off |
| `--export-qc` | Export QC'd matrices for external tools | Off |
| `--no-annotation` | Skip gene annotation | Off |

## Output Files

The pipeline produces a ZIP file containing:

- `tables/GWAS_<trait>.csv` — Full GWAS results table
- `tables/CrossModel_Consensus.csv` — Per-SNP detection across MLM/MLMM/FarmCPU (when multiple models are run)
- `tables/LD_blocks_annotated_<model>.csv` — LD blocks with gene annotations and haplotype statistics
- `figures/` — Manhattan plots, QQ plots, LD heatmaps
- `report.html` — Self-contained HTML analysis report

## Real Datasets

TRACE has been validated on:

1. **Tomato Varitome panel** — 165 accessions, ~44K SNPs (Pereira et al., 2021)
2. **Pepper G2P-SOL panel** — 350 accessions, ~5K SNPs (Tripodi et al., 2021)

## Docker

```bash
docker build -t trace-gwas .
docker run -p 8501:8501 -v $(pwd)/data:/app/data trace-gwas
```
