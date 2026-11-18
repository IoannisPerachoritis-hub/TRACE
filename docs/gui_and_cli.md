# The GUI and the CLI

Every analysis of the GUI's One-Click Full Analysis has a CLI option, so a run on a server or in a
batch script writes the same results as the GUI. The table lists where each one is.

| What | GUI page | CLI option |
|---|---|---|
| MLM with LOCO kinship | GWAS Analysis | `--model mlm` (the default), `--no-loco` for one kinship for the whole genome |
| Several traits in one run | GWAS Analysis, One-Click Full Analysis | `--trait A B C` (one ZIP per trait and a combined ZIP) |
| MLMM | GWAS Analysis | `--model mlm mlmm` |
| TRACE-FarmCPU | GWAS Analysis | `--model mlm farmcpu` |
| The models compared | GWAS Analysis | made when two or more models run (`CrossModel_Consensus.csv`) |
| Covariates | GWAS Analysis (a covariate file, or columns of the phenotype file) | `--covar FILE`, with `--covar-cols` for some of its columns |
| PCs as covariates, and their diagnostics | GWAS Analysis | `--n-pcs N` (a fixed count, 0 by default). The report shows the eigenvalue spectrum. |
| SNP and sample QC | GWAS Analysis | `--maf`, `--mac`, `--miss`, `--ind-miss`, `--info-thresh` |
| The significance threshold | GWAS Analysis | `--sig-thresh` (`bonferroni` by default, `meff`, `fdr` or a p-value) |
| Where the genotype table is kept | GWAS Analysis, "Large datasets" | `--geno-store` (`auto` by default, `memory` or `disk`) |
| LD blocks and the haplotype test | Post-GWAS Analysis, LD Blocks & Haplotypes | `--ld-r2`, `--ld-seed-p`, `--ld-top-n`, `--hap-perms`, `--hap-test` |
| Gene annotation | Post-GWAS Analysis, Gene Annotation | `--gene-model`, `--genome-build`, `--no-annotation` to skip it |
| Subsampling stability | GWAS Analysis | `--subsampling`, `--boot-reps`, `--boot-frac` |
| The random seed of the subsampling | GWAS Analysis, the sidebar's "Subsampling" | `--seed N` (42 by default) |
| The HTML report | GWAS Analysis | made by default, `--no-report` to skip it |
| Plots | GWAS Analysis, Post-GWAS Analysis | made by default, `--no-plots` to skip them |
| The QC data for other tools | (not in the GUI) | `--export-qc` |

Some views are in the GUI only: the Local LD heatmap for any window, the LD decay curves and the
regional plot, on the Post-GWAS Analysis page. The CLI writes their tables, so the same plots can
be drawn elsewhere. The LD block heatmaps are the exception. The CLI's ZIP and One-Click's ZIP draw
them for the blocks whose lead SNP is significant in the MLM, MLMM or FarmCPU scan, and the Local
LD tab draws any detected block when the region selector is set to "Detected block". One function
draws them all, each block with its neighbouring SNPs.

[CLI reference](cli_reference.md) lists every option.
