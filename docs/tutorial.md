# TRACE Quick-Start Tutorial

This tutorial takes you end-to-end on a small synthetic example — first from the
**command line**, then through the **web app** — so you can verify your TRACE
installation and learn both interfaces in a few minutes.

## The example dataset

The example ships in the repository:

- `examples/example.vcf.gz` — ~50 samples, ~510 SNPs across 3 chromosomes, with
  one embedded QTL on chromosome 2 (h² ≈ 0.4).
- `examples/example_pheno.csv` — two traits: `Trait1` carries the QTL signal,
  `Trait2` is pure noise.

It is byte-reproducible: `examples/simulate_example.py` regenerates it
deterministically (seed 42), so a fresh clone and a re-run produce identical
files. The whole example runs through the full pipeline in under 30 seconds.

---

## Part A — Command line

### 1. One-shot reproducible run

```bash
bash examples/run_example.sh
```

That script (1) runs `simulate_example.py` to write `example.vcf.gz` and
`example_pheno.csv`, then (2) runs the TRACE MLM pipeline on `Trait1` and writes
a results ZIP to `examples/output/`.

### 2. Manual CLI invocation

Call the CLI directly to try a different model or options:

```bash
trace-gwas \
  --vcf   examples/example.vcf.gz \
  --pheno examples/example_pheno.csv \
  --trait Trait1 \
  --output examples/output/ \
  --model mlm
```

If you have not installed the package (`pip install -e .`), use
`python cli.py` in place of `trace-gwas`. Run `trace-gwas --help` — or see
[`docs/cli_reference.md`](cli_reference.md) — for the complete list of options.

### Key CLI options

| Flag | Description | Default |
|------|-------------|---------|
| `--vcf` | Input VCF (bgzipped or plain) | Required |
| `--pheno` | Phenotype CSV (SampleID + trait columns) | Required |
| `--trait` | Trait column name | Required |
| `--output` | Output directory | Required |
| `--model` | Model(s) to run, space-separated (`mlm`, `mlmm`, `farmcpu`) | `mlm` |
| `--n-pcs` | Number of principal components (fixed; the run report includes PC-selection diagnostics) | 0 |
| `--maf` | Minor-allele-frequency filter | 0.05 |
| `--miss` | Maximum per-SNP missingness | 0.10 |
| `--sig-thresh` | Significance rule: `meff`, `bonferroni`, `fdr`, or a numeric p-value (e.g. `5e-8`) | `bonferroni` |
| `--seed` | RNG seed for subsampling / permutation | 42 |
| `--subsampling` | Enable subsampling stability analysis | Off |
| `--export-qc` | Export QC'd matrices for external tools | Off |
| `--no-plots` | Skip plot generation | Off |
| `--no-annotation` | Skip gene annotation | Off |

### Output files

The pipeline writes a ZIP containing:

- `tables/GWAS_<trait>.csv` — full GWAS results table
- `tables/CrossModel_Consensus.csv` — per-SNP detection across MLM/MLMM/FarmCPU (when multiple models are run)
- `tables/LD_blocks_annotated_<model>.csv` — LD blocks with gene annotations and haplotype statistics (when significant LD blocks are detected)
- `figures/` — Manhattan plots, QQ plots, LD heatmaps
- `report.html` — self-contained HTML analysis report

The columns of these tables are described in the CSV Column Glossary on the app's **Help & Reference** page. See [`docs/outputs.md`](outputs.md) for the additional post-GWAS CSVs written by the LD, haplotype and regional-plot tabs.

---

## Part B — Web app (Streamlit)

The web app runs the same engine as the CLI, with an interactive, guided
workflow. Launch it:

```bash
streamlit run app.py
```

Your browser opens the TRACE landing page. The left sidebar lists the analysis
pages; you will visit them in the order below. (The landing page also links the
first three directly.)

### Step 1 — GWAS Analysis

Open **GWAS Analysis** ("TRACE GWAS") in the sidebar.

1. **Upload data.** Upload the VCF (`examples/example.vcf.gz`) and the phenotype
   CSV (`examples/example_pheno.csv`).
2. **Review quality control.** Accept the default QC thresholds (MAF, per-SNP
   missingness, INFO) and imputation, or adjust them. Set the number of principal
   components used as covariates (a fixed count; TRACE does not auto-select).
3. **Select the trait.** Pick `Trait1`.
4. **Run.** For the standard screen, open the **One-Click Full Analysis**
   expander and click **Run Full Analysis** — this runs MLM GWAS →
   multi-model → LD blocks → haplotype testing → gene annotation → report. To run
   a single model on the selected trait instead, use **Run GWAS**.

When the run finishes you will see the Manhattan and QQ plots, the genomic
inflation factor (λ_GC), and the significant-SNP table, plus a button to
download the full results ZIP — the same archive the CLI produces.

### Step 2 — Post-GWAS Analysis

Open **Post-GWAS Analysis**. It reuses the QC'd genotype matrix from your GWAS
run (run Step 1 first). Pick a region / lead SNP in the selector at the top, then
explore the tabs:

- **Significant SNPs** — the complete, never-truncated significant-SNP table, with each SNP marked in-block or unblocked.
- **Regional Plot** — a locus-zoom view: −log₁₀(p) vs position, coloured by LD (r²) to the lead SNP, with a gene track.
- **Local LD** — the pairwise r² heatmap for the selected window.
- **Gene Annotation** — genes overlapping or flanking each LD block.
- **LD Decay** — genome-wide LD decay, which sets the default block window.
- **LD Blocks & Haplotypes** — detected blocks, haplotype-based association tests, and per-block MLG summaries.

### Step 3 — Help & Reference

Open **Help & Reference** for the output-file glossary, a column glossary, and
guidance on interpreting results for breeding decisions.

---

## Real datasets

TRACE has been validated on:

1. **Tomato Varitome panel** — 165 accessions, ~44K SNPs (Pereira et al., 2021)
2. **Pepper G2P-SOL panel** — 350 accessions, ~5K SNPs (Tripodi et al., 2021)

## Docker

```bash
docker build -t trace-gwas .
docker run -p 8501:8501 -v $(pwd)/data:/app/data trace-gwas
```
