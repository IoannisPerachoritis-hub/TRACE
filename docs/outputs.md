# Post-GWAS output CSVs

This page documents the tabular outputs added to TRACE **after the v1.0.1
release**: the complete significant-SNP tables, the LD r² exports behind the
Local LD heatmap, and the per-SNP data behind the regional plot. The GWAS,
LD-block, haplotype, subsampling and consensus CSVs that shipped in v1.0.1 are
documented in the in-app **Help & Reference** page ("CSV Column Glossary").

Two related command-line options are covered at the end: `--covar` (new) and the
numeric branch of `--sig-thresh`.

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
| 6 | `Beta_MLM` | float | Allele-substitution effect from the structure-adjusted MLM; blank if unavailable. |
| 7 | `SE_MLM` | float | Standard error of `Beta_MLM`. |
| 8 | `Beta_OLS` | float | Effect from the unadjusted OLS fit; blank if unavailable. |
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
| 19 | `Passes_FDR` | bool | Significant under FDR (q < 0.05). |
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
| 30 | `Interval_Start_bp` | int | Candidate-interval start: the block bounds when `in_block`, else the flanking-marker interval. Blank if none. |
| 31 | `Interval_End_bp` | int | Candidate-interval end. |
| 32 | `Interval_Bounded_By` | str | `typed_snps` / `chromosome_start` / `chromosome_end` / `chromosome_both`. |
| 33 | `N_Genes_Interval` | int | Genes in the interval; blank (`<NA>`) when no gene annotation is loaded, **not `0`**. |
| 34 | `Candidate_Genes` | str | `;`-joined gene IDs (overlapping + up to two flanking each side). |
| 35 | `Gene_Evidence` | str | `no_annotation_loaded` / `no_gene_within_interval` / `overlapping_interval` / `flanking_within_500kb` / `custom_gene_model`. |
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

## `LD_r2_long_<stem>.csv`

`<stem>` = `Chr<chr>_<start>_<end>_<n>snps` (e.g. `Chr2_47000000_47600000_180snps`),
so exports from different windows are distinguishable by filename.

**What it contains.** The pairwise LD (r²) numbers behind the **Local LD** heatmap, in
tidy long format: one row per unordered SNP pair (the upper triangle).

**One row per** SNP pair (i < j) in the Local LD window.

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

**What it contains.** The full square r² matrix for the Local LD window: the first
(unnamed) column and the header row are both the SNP IDs, and each cell is the r²
between its row and column SNP (diagonal = 1).

**One row per** SNP in the window (and one column per SNP).

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

## Related CLI options (new since v1.0.1)

The full, generated option list is in [`cli_reference.md`](cli_reference.md); these two
are called out because they are new capabilities.

### `--covar` / `--covar-cols`

Supply user covariates (batch, environment, a structure axis, …) as fixed effects
**in addition to** the principal components, for every model (MLM / MLMM / FarmCPU).

- `--covar` takes a CSV/TSV whose **first column is the sample ID** (matching the VCF)
  and whose remaining columns are **numeric** covariates. Encode categoricals as numeric
  indicators (e.g. one-hot 0/1) beforehand.
- `--covar-cols` selects a comma-separated subset of covariate columns (default: all).
- Samples missing a covariate value are **dropped** from the analysis (reported in the
  log).
- Effect: the covariates are concatenated onto the PC block, so a covariate perfectly
  correlated with the phenotype removes the signal. **Output is byte-identical when
  `--covar` is not given.**

### `--sig-thresh <p-value>` (numeric branch)

`--sig-thresh` accepted named rules (`meff` / `bonferroni` / `fdr`) at v1.0.1; it now
also accepts a **numeric p-value** such as `5e-8` (any value in `(0, 1]`). A numeric
threshold reports significance in a **`Significant_Custom`** boolean column of the GWAS
results, and appears in `Significant_SNPs.csv` as `Sig_Rule = custom` with
`Sig_Threshold` set to the value you passed. The default remains `bonferroni`.
