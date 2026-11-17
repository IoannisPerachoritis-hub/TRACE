# Post-GWAS output CSVs

This page documents the tabular outputs added to TRACE **after the v1.0.1
release**: the complete significant-SNP tables, the sample-to-haplotype assignments
of the haplotype test, the haplotype test's own columns (which test ran, its kernel and its
null model's heritability), the LD r² exports behind the Local LD heatmap, and the
per-SNP data behind the regional plot. It starts with
the significance columns of the per-model GWAS tables: MLM's shipped in v1.0.1
(`Significant_Custom` in 1.1.0, with the numeric `--sig-thresh` described at the
end), and MLMM and FarmCPU now carry the same ones under the same rule. The GWAS,
LD-block, haplotype, subsampling and consensus CSVs that shipped in v1.0.1 are
documented in the in-app **Help & Reference** page ("CSV Column Glossary").

Three related command-line options are covered at the end: `--covar` and `--hap-test`
(new) and the numeric branch of `--sig-thresh`. First, the list of every file the results ZIP
holds.

---

## Every file of the CLI and One-Click ZIP

The CLI and One-Click Full Analysis write the same ZIP for a trait. `<trait>` is the trait
column, `<model>` is `MLM`, `MLMM` or `FarmCPU`, and `<c>`, `<start>`, `<end>` are an LD
block's chromosome and bounds. A table with no rows is not written. `--no-isolated-rescue`
drops the significant-SNP, unblocked-SNP and isolated-SNP tables and the SNP view index;
`--no-annotation` leaves the genes out.

| File | What it holds | Written when |
|---|---|---|
| `tables/GWAS_<trait>.csv` | MLM's scan: every tested SNP's p-value, effect and significance columns | always |
| `tables/GWAS_<model>_<trait>.csv` | MLMM's or FarmCPU's scan, with the same significance columns | that model runs |
| `tables/Significant_SNPs_<model>.csv` | every SNP significant in that model, in an LD block or unblocked | the model has a significant SNP |
| `tables/Unblocked_SNPs_<model>.csv` | the significant SNPs outside every LD block | the model has one |
| `tables/Isolated_SNP_intervals_<model>.csv` | the candidate interval of each unblocked significant SNP, or run of them | the model has one |
| `tables/Isolated_SNP_candidate_genes_<model>.csv` | the genes inside those intervals | a gene lies in one (gene annotation on) |
| `tables/LD_blocks_annotated_<model>.csv` | the model's LD blocks: bounds, candidate interval, genes and haplotype test | the model has a block |
| `tables/LD_triage_<model>.csv` | each block's LD-quality triage and the view it recommends | the model has a block, unless `--no-triage` |
| `tables/Haplotype_assignments_<trait>.csv` | which haplotype each sample carries in each of MLM's tested blocks | MLM has a tested block |
| `tables/Haplotype_assignments_<trait>_<model>.csv` | the same for MLMM's or FarmCPU's tested blocks | that model has a tested block |
| `tables/SNP_view_index_<model>.csv` | for each significant SNP of the primary model (MLM), whether the report plots it, and its plotting representative | MLM has a significant SNP |
| `tables/CrossModel_Consensus.csv` | per significant SNP, the models that call it | two or more models run |
| `tables/PC_diagnostics_spectrum.csv` | the genotype PCA's eigenvalue spectrum (rank, eigenvalue, share of the trace) | always |
| `tables/QC_report_summary.csv` | the QC report: imputation, heterozygosity, relatedness and F_IS statistics | always |
| `tables/QC_sample_heterozygosity.csv` | each sample's heterozygosity, its z-score and outlier flag | always |
| `tables/QC_genotype_matrix.csv` | the QC'd, imputed genotypes (samples × SNPs) | `--export-qc` |
| `tables/QC_genotype_raw.csv` | the same genotypes before imputation, missing calls kept | `--export-qc` |
| `tables/QC_snp_map.csv` | the QC'd SNPs: ID, chromosome, position, REF and ALT | `--export-qc` |
| `tables/QC_phenotype.csv` | the analysed trait values | `--export-qc` |
| `tables/Subsampling_SNP_stability.csv` | each SNP's discovery frequency and p-values across the subsamples | `--subsampling` |
| `tables/Subsampling_block_stability.csv` | each MLM LD block's discovery frequency across the subsamples | `--subsampling`, with blocks |
| `tables/Subsampling_rep_metadata.csv` | one row per subsample: its size, status, λGC and hits | `--subsampling` |
| `figures/Manhattan_<model>.png` | each model's Manhattan plot | unless `--no-plots` |
| `figures/QQ_<model>.png` | each model's QQ plot | unless `--no-plots` |
| `figures/LD_heatmap_Chr<c>_<start>_<end>.png` | the r² heatmap of each LD block whose lead is significant in its model | unless `--no-plots`, or One-Click's "Generate LD block heatmap PNGs" is off |
| `figures/PCA_scree.png` | the PCA scree plot | unless `--no-plots` |
| `figures/PCA_scatter.png` | PC1 against PC2 | two or more PCs, unless `--no-plots` |
| `figures/QC_report_distributions.png` | the QC report's distributions | unless `--no-plots` |
| `figures/Subsampling_discovery_freq_<trait>.png` | the subsampling discovery frequencies | `--subsampling`, unless `--no-plots` |
| `report.html` | the self-contained HTML report | unless `--no-report` |
| `run_metadata.json` | every setting of the run, for reproducibility; it also records the run's environment and where the genotype table was kept (the keys below) | always |
| `<trait>/` | in a combined ZIP, one folder per trait holding that trait's files | several traits |
| `failed_traits.txt` | in a combined ZIP, each trait's status and the error of any that failed | a trait failed |

**`run_metadata.json`: the environment and the genotype table.** Besides the settings, these keys
(the report's provenance table lists the first six too):

- `BLAS library`: the linear-algebra libraries and their versions (numpy's and scipy's, numpy's first)
- `BLAS threads`: their thread count
- `CPU cores (logical)`: the logical cores
- `CPU cores (physical)`: the physical cores
- `Subsampling workers (--boot-jobs)`: the subsampling workers (N/A without subsampling)
- `Subsampling BLAS threads per worker`: the threads each worker's library uses
- `Genotype table`: where the table was kept, `memory` or `disk` (docs/large_datasets.md)
- `Genotype table choice`: what was asked, `auto` or `--geno-store memory|disk`
- `Genotype table reason`: why it was kept there, in words with no number
- `Memory estimate, in-memory run (GB)`: TRACE's estimate for an in-memory run of this analysis
- `Genotype calls`: samples × SNPs as read

The GWAS page's single-trait download is a ZIP of that page's own tables and figures,
interactive HTML included; the files of the Post-GWAS tabs are their own downloads (below).

---

## Significance columns of `GWAS_<trait>.csv`, `GWAS_MLMM_<trait>.csv` and `GWAS_FarmCPU_<trait>.csv`

**What they contain.** The MLM table and, when those models run, the MLMM and
FarmCPU tables carry the same significance columns. One function writes them
(`gwas.significance.add_significance_columns`), in the CLI, in One-Click Full
Analysis and in the GWAS page's single-trait run. The CLI and One-Click write
`Significant_Meff` for every model; the single-trait run writes it, for every
model, when the M_eff rule is chosen.

| column | type | meaning |
|--------|------|---------|
| `FDR` | float | Benjamini–Hochberg q-value, computed within this model's own p-values. |
| `Significant_FDR` | bool | The Benjamini–Hochberg decision at 0.05 within this model: a p-value at or below its step (rank/m × 0.05), which is `FDR` ≤ 0.05 up to rounding. |
| `Significant_Bonf` | bool | `PValue < 0.05 / m`, where m is the number of SNPs tested (MLM's Bonferroni threshold). |
| `Significant_Meff` | bool | `PValue < 0.05 / M_eff` (MLM's M_eff threshold, Li & Ji). |
| `Significant_Custom` | bool | `PValue < t` for a numeric threshold t (`--sig-thresh 5e-8`, or the single-trait run's Custom p-value); written only when one is set. |

**The rule.** The p-value thresholds are MLM's, the same numbers for every model,
so a SNP with a given p-value is judged the same whichever model produced it.
FDR is the one per-model quantity: Benjamini–Hochberg counts the tests of that
model only. A SNP without a p-value is never significant (its `FDR` is blank),
and Benjamini–Hochberg counts only the SNPs that have one.

**What uses them.** The column for the analysis rule (`--sig-thresh`) selects
the SNPs of each model's `Significant_SNPs_<model>.csv`, the report's per-model
count of significant SNPs, the cross-model consensus (`CrossModel_Consensus.csv`),
and the blocks that get an LD heatmap.

**LD heatmaps.** A heatmap (`figures/LD_heatmap_Chr<c>_<start>_<end>.png`) is
drawn for each LD block whose lead SNP is significant in its model. A block is
its chromosome, start and end: a block that several models report with the same
bounds is drawn once, marking the lead SNP of the first model (in the order MLM,
MLMM, FarmCPU) whose block with these bounds has a significant lead. The
title names every model whose own block with these bounds has a significant
lead, or in which the marked lead SNP is significant (for example "significant
in MLM, FarmCPU"); a model that lists the block with a lead that is not
significant is not named. When the list is MLM alone, the title has none, as
before. Models often delimit one signal with slightly different bounds; those
are separate blocks, each with its own heatmap and its own list.

**Effect columns.** The MLM table (`GWAS_<trait>.csv`) carries `Beta_MLM` and
`SE_MLM`, the MLM's effect and its standard error per copy of the ALT allele
(FaST-LMM's estimates on the standardised SNP, divided by the SNP's dosage standard
deviation over the tested samples), and `Beta_OLS`, `SE_OLS` (per ALT copy too) and
`Tstat_OLS` from the OLS fit with the same PC covariates and no kinship. `EffectSize`,
`Mixing` and `Nullh2` are FaST-LMM's own columns, unchanged; `EffectSize` is a fraction
of variance explained. The MLMM and FarmCPU tables carry no MLM effect. A significant
SNP's box plot flags the two effects: `sign-flip` when their signs differ,
`structure-sensitive` when |`Beta_MLM`| is under half of |`Beta_OLS`|, `concordant`
otherwise, and `undetermined` when either is missing or `Beta_OLS` is near 0.

**MLMM's p-values.** `GWAS_MLMM_<trait>.csv` holds MLMM's final scan, with the selected
cofactors as covariates. A cofactor would be tested with itself there, so its p-value
comes instead from a test with the other cofactors as covariates, in MLMM's own model
(the same kinship and base covariates). Every other SNP keeps the final scan's p-value.
The report's MLMM cofactor table gives each cofactor's entry p-value (`PValue_entry`,
from the forward step at which it entered) and that p-value (`PValue_final`).

---

## `Significant_SNPs.csv`

**What it contains.** Every SNP that passes the **reporting** threshold
(`--sig-thresh` / the rule chosen on the GWAS page), one row each. The table is
always complete, never truncated, and it uses the *reporting* threshold, not the
lower block-seeding threshold. Each SNP is marked as belonging to an LD block or
as *unblocked*, with its candidate interval, gene count, and gene-evidence label.

**One row per** reporting-significant SNP.

**When it's emitted.**
- GUI: **Post-GWAS Analysis → Significant SNPs** tab, "Download Significant_SNPs.csv".
- CLI: in the results ZIP as `Significant_SNPs_<model>.csv` (one per model, e.g.
  `Significant_SNPs_MLM.csv`), unless `--no-isolated-rescue` is set.

**Columns** (38, in this order):

| # | column | type | meaning |
|---|--------|------|---------|
| 1 | `SNP` | str | SNP identifier (VCF ID column). |
| 2 | `Chr` | str | Chromosome, canonicalised (`canon_chr`). |
| 3 | `Pos` | int | Position in base pairs (1-based, from the VCF). |
| 4 | `PValue` | float | Association p-value from the model's final scan. |
| 5 | `-log10p` | float | −log₁₀(`PValue`). |
| 6 | `Beta_MLM` | float | Effect per copy of the ALT allele from the structure-adjusted MLM (kinship and the PC covariates): FaST-LMM estimates it on the standardised SNP, and TRACE divides it by the SNP's dosage standard deviation over the tested samples, so it shares `Beta_OLS`'s unit; blank if unavailable. |
| 7 | `SE_MLM` | float | Standard error of `Beta_MLM`, per copy of the ALT allele. |
| 8 | `Beta_OLS` | float | Effect per copy of the ALT allele from an OLS fit with the same PC covariates and no kinship; blank if unavailable. |
| 9 | `SE_OLS` | float | Standard error of `Beta_OLS`. |
| 10 | `Effect_Source` | str | Which effect estimates are present: `both` / `MLM_only` / `OLS_only` / `unavailable`. |
| 11 | `MAF` | float | Minor allele frequency (0–0.5) from the QC'd genotypes. |
| 12 | `ImputationRate` | float | Fraction of genotypes imputed at this SNP (0–1); blank if not tracked. |
| 13 | `Sig_Rule` | str | Reporting rule applied: `meff` / `bonferroni` / `fdr` / `custom`. |
| 14 | `Sig_Threshold` | float | The p-value cutoff for the rule; blank for `fdr` (a q-value decision, not a fixed p). |
| 15 | `Seed_Threshold` | float | The block-**seeding** p-value (`--ld-seed-p`, default 1e-5). |
| 16 | `Passes_Seed_Threshold` | bool | `PValue < Seed_Threshold`, i.e. this SNP was eligible to seed a block. |
| 17 | `Passes_Meff` | bool | Significant under M_eff (blank if that column was not computed). |
| 18 | `Passes_Bonf` | bool | Significant under Bonferroni. |
| 19 | `Passes_FDR` | bool | Significant under FDR (the model's `Significant_FDR`). |
| 20 | `FDR` | float | Benjamini–Hochberg q-value; blank if not computed. |
| 21 | `Block_ID` | str | `chr:start-end` of the containing LD block, or empty when unblocked. |
| 22 | `Block_Status` | str | One of five values (see below). |
| 23 | `r2_to_block_lead` | float | r² of this SNP to the block's lead SNP. Populated in the GUI display layer; **blank in the CLI export**. |
| 24 | `N_SNPs_in_Block` | int | Member SNPs in the containing block; blank (`<NA>`) when unblocked. |
| 25 | `Max_r2_in_Window` | float | Max r² to any neighbour in the window (GUI display layer; blank in the CLI export). |
| 26 | `Nearest_Typed_Upstream_SNP` | str | Nearest typed marker upstream of this SNP. |
| 27 | `Nearest_Typed_Upstream_bp` | int | Distance to it, base pairs. |
| 28 | `Nearest_Typed_Downstream_SNP` | str | Nearest typed marker downstream. |
| 29 | `Nearest_Typed_Downstream_bp` | int | Distance to it, base pairs. |
| 30 | `Interval_Start_bp` | int | Candidate-interval start. The interval runs from the nearest post-QC marker before the first SNP (the block's first SNP when `in_block`, else the SNP or run of unblocked SNPs) to the nearest marker after the last; at a chromosome edge it reaches a fixed flank past the SNP, and wider than 5 Mb it is clamped around the lead SNP. Genes are searched here. Blank if none. |
| 31 | `Interval_End_bp` | int | Candidate-interval end. |
| 32 | `Interval_Bounded_By` | str | `typed_snps` / `chromosome_start` / `chromosome_end` / `chromosome_both`. |
| 33 | `N_Genes_Interval` | int | Genes in the interval; blank (`<NA>`) when no gene annotation is loaded, **not `0`**. |
| 34 | `Candidate_Genes` | str | `;`-joined IDs of the genes that overlap the candidate interval; nothing outside it is listed. |
| 35 | `Gene_Evidence` | str | `no_annotation_loaded` / `no_gene_within_interval` / `overlapping_interval` / `custom_gene_model`. |
| 36 | `Genome_Build` | str | Gene-model build used for annotation (e.g. `SL3`), or `custom`. |
| 37 | `Plot_Rendered` | bool | Whether a per-SNP effect plot was rendered for this SNP (subject to the plot budget). |
| 38 | `Withheld_Reason` | str | Why a plot was withheld; empty when one was rendered. |

**`Block_Status` values.**

- `in_block`: positionally inside an emitted LD block (`Start ≤ Pos ≤ End`).
- `unblocked_not_seeded`: significant by the reporting rule but `PValue ≥ Seed_Threshold`,
  so the block detector never even considered it as a seed. **These are the SNPs a
  block-only view silently drops** (see the `--ld-seed-p` note under
  `Unblocked_SNPs.csv`).
- `unblocked_isolated`: seeded, but no typed neighbour within the flank window.
- `unblocked_monomorphic_window`: seeded, but the neighbours have no genotype variance.
- `unblocked_low_ld`: seeded with polymorphic neighbours, but none reached the LD
  threshold to form a block. This is an *inference* from the emitted block table; the
  detector is never re-run.

---

## `Unblocked_SNPs.csv`

**What it contains.** Exactly the rows of `Significant_SNPs.csv` where
`Block_Status ≠ in_block`, the significant SNPs that formed no LD block, restricted
to the 22 columns relevant to an unblocked SNP:

`SNP, Chr, Pos, PValue, -log10p, Sig_Rule, Sig_Threshold, Seed_Threshold,
Passes_Seed_Threshold, Block_Status, Max_r2_in_Window, Nearest_Typed_Upstream_SNP,
Nearest_Typed_Upstream_bp, Nearest_Typed_Downstream_SNP, Nearest_Typed_Downstream_bp,
Interval_Start_bp, Interval_End_bp, Interval_Bounded_By, N_Genes_Interval,
Candidate_Genes, Gene_Evidence, Genome_Build`.

**One row per** unblocked significant SNP. It is a strict projection of
`Significant_SNPs.csv`. No SNP appears here that is not also in that table.

**When it's emitted.** GUI Significant SNPs tab ("Download Unblocked_SNPs.csv"); CLI
ZIP as `Unblocked_SNPs_<model>.csv`.

**Why a significant SNP can be "unblocked": the reporting vs seeding threshold.**
TRACE uses **two different p-value thresholds**. The *reporting* threshold
(`--sig-thresh`) decides which SNPs are significant. The lower, separate *block-seeding*
threshold (`--ld-seed-p`, default 1e-5) decides which SNPs the LD-block detector starts
from. A SNP can be significant by the reporting rule yet have a p-value **weaker than
`--ld-seed-p`**, so it is never used as a block seed and forms no block
(`Block_Status = unblocked_not_seeded`). That is expected behaviour, not a failure.
This table exists precisely so those SNPs are visible as data rather than dropped by a
block-only view. The `Interval_Start_bp`/`Interval_End_bp` columns give each such SNP a
flanking-marker candidate interval whose width reflects marker density, not association
strength.

---

## `Haplotype_assignments_<trait>.csv`

**What it contains.** Which haplotype each sample carries in each tested LD block, and
whether the block's haplotype test took it: one function
(`gwas.haplotype.haplotype_assignments`) builds it from the test's own per-sample
tables, so the labels are the ones the test compared and the Post-GWAS MLG plots show.

**One row per** sample of the panel **per tested block** of one model, blocks in the
order of that model's `LD_blocks_annotated_<model>.csv`. The panel is the trait's analysis
samples, in genotype order: genotyped, with a finite trait value, after `--ind-miss`
and any covariate drops. So a sample without a trait value, or one removed by sample
QC, is not listed.

**When it's emitted.** CLI and One-Click ZIP:
- `tables/Haplotype_assignments_<trait>.csv` for the primary model (MLM), the file the
  report's per-SNP box plots read;
- `tables/Haplotype_assignments_<trait>_<model>.csv` for each other model with tested
  blocks (MLMM, FarmCPU), named as `LD_triage_<model>.csv`, with the same columns.

Each covers every tested block of its model. There is no file for a model with no tested
block, while `LD_blocks_annotated_<model>.csv` is still written for the detected blocks.

GUI **Post-GWAS Analysis → LD Blocks & Haplotypes**, next to the
block selector: "Download sample-to-haplotype assignments (CSV)" gives the selected
block's rows as `haplotype_assignments_block_<chr>_<start>_<end>_<trait>.csv`, and
"Download assignments for all blocks (CSV)" gives every tested block as
`Haplotype_assignments_<trait>.csv`. For the same inputs the all-blocks file and the ZIP
member are byte-identical, and a block's file is its rows of the all-blocks file. The GUI
files follow the model chosen for downstream analysis on the GWAS page; for MLMM or
FarmCPU the all-blocks file is named `Haplotype_assignments_<trait>_<model>.csv`.

**Columns** (12, in this order; a 13th only as described below):

| Column | Type | Meaning |
|---|---|---|
| `Trait` | str | The trait analysed. |
| `Chr` | str | The block's chromosome. |
| `Block_start` | int | The block's start (bp). |
| `Block_end` | int | The block's end (bp). |
| `Lead_SNP` | str | The block's lead SNP. |
| `Sample` | str | The sample ID. |
| `Haplotype` | str | `H1`..`Hk` (by descending count) or `Other` (a haplotype seen in fewer than `--hap-min-count` samples); empty for a sample without a haplotype (missing genotype or phenotype; does not occur in TRACE's own files). |
| `Allele_sequence` | str | The sample's alleles at the block's markers, `\|`-separated alternate-allele counts (0, 1, 2) in `SNP_order`: the rounded dosages the test used. TRACE passes the run's imputed genotypes, so a missing call carries its imputed dosage (`--impute`: the per-SNP mean or the LD-kNNi value), rounded; empty with `Haplotype`. |
| `SNP_order` | str | The markers of `Allele_sequence`, `\|`-separated: the block's SNPs after the call-rate filter, dropping markers constant among the block's samples and the r² ≥ 0.9 thinning; at least two are kept (when thinning leaves one, the block's first two markers are used). `n_snps` of the haplotype results; can be fewer than the block's SNP_IDs. |
| `in_haplotype_test` | bool | The test took this sample (its group is one of the tested haplotype groups). |
| `exclusion_reason` | str | Empty when in the test; else `missing genotype` (over 30% of the block's markers missing), `missing phenotype` (no finite trait value; also when both are missing), `rare haplotype (Other)`, `group below minimum size` (a haplotype with fewer than `--hap-min-group-size` samples) or `missing covariate` (a missing PC value; does not occur with genotype PCs). `missing genotype` and `missing phenotype` do not occur in TRACE's own files either: the genotypes are imputed and the panel holds only the trait's analysis samples. |
| `Phenotype_used` | float | The trait value the haplotype test takes for this sample (blank when missing). Only rows with `in_haplotype_test` enter the test; the PCs are covariates of the F-test, not subtracted from this value (the GUI's MLG plots show PC-adjusted values). |
| `Phenotype_raw` | float | Only when the test's phenotype differs from the uploaded values: the uploaded value. The haplotype test uses the uploaded values (a phenotype QC transformation applies to the GWAS scan, as `--norm` does), so the column does not appear in current outputs. |

**Counts.** A block's number of rows with `in_haplotype_test` equals its
`n_samples_tested` (in `LD_triage_MLM.csv` and the GUI's haplotype results table), which
is not `Hap_n_samples` (the samples with a usable block genotype, before samples without
a phenotype, rare haplotypes and small groups are left out).

---

## `LD_r2_long_<stem>.csv`

`<stem>` = `Chr<chr>_<start>_<end>_<n>snps` (e.g. `Chr2_47000000_47600000_180snps`),
so exports from different windows are distinguishable by filename.

**What it contains.** The pairwise LD (r²) numbers behind the **Local LD** heatmap, in
tidy long format: one row per unordered SNP pair (the upper triangle). In **Detected block**
mode it covers the block's member SNPs; the heatmap there also shows the block's
neighbouring SNPs.

**One row per** SNP pair (i < j) in the Local LD window (for a detected block, among its
member SNPs).

| column | type | meaning |
|--------|------|---------|
| `SNP_A` | str | First SNP of the pair. |
| `SNP_B` | str | Second SNP of the pair. |
| `r2` | float | Linkage disequilibrium r² (0–1), pairwise-complete Pearson, from the imputed dosages. |

**When it's emitted.** GUI **Post-GWAS Analysis → Local LD** tab. (No CLI counterpart:
the CLI emits the block-level LD tables instead.)

**Relation to neighbours.** The long form of `LD_r2_matrix_<stem>.csv`, the same r²
values, one pair per row instead of a square matrix.

---

## `LD_r2_matrix_<stem>.csv`

Same `<stem>` convention as the long file.

**What it contains.** The full square r² matrix for the Local LD window (for a detected
block, its member SNPs): the first (unnamed) column and the header row are both the SNP
IDs, and each cell is the r² between its row and column SNP (diagonal = 1).

**One row per** SNP in the window, or block member (and one column per SNP).

**When it's emitted.** GUI Local LD tab.

**Relation to neighbours.** The square form of `LD_r2_long_<stem>.csv`.

---

## `Regional_data_Chr<c>_<s>_<e>_<lead>.csv`

Filename carries the chromosome, window start/end, and lead SNP.

**What it contains.** The per-SNP numbers behind the **Regional Plot**, position,
p-value, r² to the lead SNP, and LD-block membership, so the plotted values can go into
a table instead of being read off a figure.

**One row per** scanned SNP (with a p-value) inside the regional window.

| column | type | meaning |
|--------|------|---------|
| `SNP` | str | SNP identifier. |
| `Chr` | str | Chromosome. |
| `Pos` | int | Position, base pairs. |
| `PValue` | float | Association p-value (unchanged, the regional plot never recomputes p-values). |
| `r2_to_lead` | float | r² to the lead SNP (0–1); blank where it is not computable (e.g. a single typed marker). |
| `block_member` | bool | Whether the SNP is a member (`SNP_IDs`) of the shaded LD block. |

**When it's emitted.** GUI **Post-GWAS Analysis → Regional Plot** tab.

**`block_member` and the lead SNP.** A block's reported **lead is the block's *seed*
SNP**, the significant SNP the detector started from, and it need **not** be a member
of the block's LD-connected `SNP_IDs`. So `block_member = False` for the lead SNP is
**expected behaviour, not a bug**: the seed frequently sits just outside the LD-connected
span it nucleated. (This is the same seed-vs-member distinction visible on the plot,
where the labelled lead can fall outside the shaded block.)

---

## The haplotype test's results

**Which test.** Each LD block's haplotype groups are tested by one of two tests, chosen with
`--hap-test` (in the GUI: **Post-GWAS Analysis → LD Blocks & Haplotypes → Settings →
Haplotype test → Test model**, and One-Click's **Haplotype test**):

- `kinship` (the default): a mixed model y = Xβ + Hγ + g + ε, with X the intercept, the PCs
  and any covariates, H the tested haplotype groups (one column per group but the first), and
  g ~ N(0, σg² K), where K is the GWAS scan's own kinship kernel for the block's chromosome
  (its LOCO kernel, or the whole-genome kernel with `--no-loco`), restricted to the block's
  tested samples. The null model y ~ X is fitted by REML (FaST-LMM's likelihood and search,
  on the phenotype's residual on X); the nested F test and the Freedman-Lane permutations are
  made on the data whitened by it. When the REML estimate is h² = 0 the whitening is the
  identity, and the block is tested exactly as by `pcs`.
- `pcs`: the earlier test, a nested F test and Freedman-Lane permutations with the PCs and
  covariates alone (no kinship).

`run_metadata.json` and the report's metadata block record the choice as
`Hap_test (--hap-test)`.

**Where the results are.** The test's full results table is in the report's haplotype tables
(`report.html`, one per model) and in the GUI's block table ("Download block table (CSV)",
`LD_blocks_and_haplotype_tests_<trait>.csv`). `LD_blocks_annotated_<model>.csv` carries its
main columns (`Hap_*`), and `LD_triage_<model>.csv` its F, η² and degrees of freedom.

**A block the kinship-aware test cannot test** keeps its row, with the reason in
`kinship_model` and no statistics; it is never tested another way. The reasons: no kernel for
its chromosome, a tested sample missing from the kernel, a failed REML fit, haplotype groups
that the intercept, PCs and covariates already span on its tested samples (a covariate that
defines them), an F test with no degrees of freedom, or a full model that leaves no residual
variance. `FDR_BH` is then Benjamini-Hochberg over the blocks with a p-value, and the GUI's
block table says in **Tested** why such a block was not tested.

**The columns about the test** (in both tests; the GUI's name in brackets):

| Column | Type | Meaning |
|---|---|---|
| `permutation_type` | str | `Freedman-Lane_residual_permutation_kinship_whitened` (kinship) or `Freedman-Lane_residual_permutation_with_PCs` (pcs). |
| `kinship_model` | str | The kernel the block was tested with: `LOCO kernel`, or `global kernel` (with `--no-loco`, or a chromosome whose LOCO kernel is the whole-genome one); `none` for the pcs test; `failed: <reason>` for a block the kinship-aware test could not test. [Kinship model] |
| `h2_null` | float | The null model's REML heritability, σg² / (σg² + σε²), on the block's tested samples; empty for the pcs test and for a failed block. [h² (null model)] |
| `eta2` | float | The test's η²: df1·F / (df1·F + df2) of the kinship-aware test's F; for the pcs test, the share of variance the haplotype groups explain in the phenotype after the PCs and covariates are removed (equal to df1·F / (df1·F + df2) without PCs and covariates). [η² (haplotype test)] |
| `eta2_pcs_only` | float | The pcs test's η² for the same block and samples, for comparison; equal to `eta2` with the pcs test. [η² (PCs-only test)] |

`F_param`, `PValue_param`, `F_perm` (= `F_param`), `P_perm` (= `PValue`), `eta2_ci_low`,
`eta2_ci_high`, `omega2` and `lead_snp_r2` are the chosen test's. With the kinship-aware test
they are those of the whitened model: `lead_snp_r2` is the r² of the whitened phenotype and the
whitened lead SNP after the whitened intercept, PCs and covariates are projected out.
`hap_stats_json` (each group's size, mean and interval) is descriptive, on the phenotype after
the PCs and covariates are removed, in both tests.

**The GUI's other effect value.** The block table's CSV also has **Window R² (all SNPs in the
block window)**: the share of the trait's variance, after the PCs, that the haplotype groups of
every SNP in the block's coordinate window explain over all phenotyped samples (it was named
`EtaSq`). It is not the test's η², and the tab does not show it. Every value the tab labels η²
(the block table, the box plot's title, the forest plot) is the test's `eta2`.

---

## Related CLI options (new since v1.0.1)

The full, generated option list is in [`cli_reference.md`](cli_reference.md); these three
are called out because they are new capabilities.

### `--hap-test {kinship,pcs}`

The haplotype test of every LD block (the section above): `kinship`, the default, the
kinship-aware test with the scan's own kernels; `pcs`, the earlier test with the PCs and
covariates alone. With `pcs` the haplotype results are the earlier test's; the report's
haplotype tables also carry the three columns above (`kinship_model` `none`, an empty
`h2_null`, `eta2_pcs_only` equal to `eta2`), and the run information records
`Hap_test (--hap-test)`.

### `--covar` / `--covar-cols`

Supply user covariates (batch, environment, a structure axis, …) as fixed effects
**in addition to** the principal components, for every model (MLM / MLMM / FarmCPU).

- `--covar` takes a CSV/TSV whose **first column is the sample ID** (matching the VCF)
  and whose remaining columns are **numeric** covariates. Encode categoricals as numeric
  indicators (e.g. one-hot 0/1) beforehand.
- `--covar-cols` selects a comma-separated subset of covariate columns (default: all).
- Samples missing a covariate value are **dropped** after phenotype QC, before SNP QC,
  kinship and the PCs. The CLI, One-Click and the GWAS page's covariate upload all
  follow this rule, through one helper (`gwas.covariates.drop_covariate_missing`). The
  CLI logs it, and the GUI shows it as a warning: "N of M analysis samples have no
  covariate value and were dropped: <the first 10 IDs>, and K more."
  `run_metadata.json` records the count (`Covariate samples dropped`).
- Effect: the covariates are concatenated onto the PC block, so a covariate perfectly
  correlated with the phenotype removes the signal. **Output is byte-identical when
  `--covar` is not given.**

### `--sig-thresh <p-value>` (numeric branch)

`--sig-thresh` accepted named rules (`meff` / `bonferroni` / `fdr`) at v1.0.1; it now
also accepts a **numeric p-value** such as `5e-8` (any value in `(0, 1]`). A numeric
threshold reports significance in a **`Significant_Custom`** boolean column of the GWAS
results, and appears in `Significant_SNPs.csv` as `Sig_Rule = custom` with
`Sig_Threshold` set to the value you passed. The default remains `bonferroni`.
