# Changelog

All notable changes to TRACE are documented in this file.

## [Unreleased]

### Added
- **Large datasets: the genotype table can be kept on disk** (`--geno-store`). The VCF is then
  read in pieces into a file of one byte per genotype call, and every step reads from that file
  only the columns it works on, so memory no longer grows with samples x SNPs. The results are
  byte-identical to a run with the table in memory (at the same thread count). By default
  (`--geno-store auto`) TRACE estimates, before reading the genotypes, how much memory an
  in-memory run would need; when that does not fit in the free memory it keeps the table on disk
  and says so in one log line and in `run_metadata.json`. `--geno-store memory` and
  `--geno-store disk` force a mode; `memory` stops before reading when it will not fit. The
  table file goes into a new folder inside `--output` (or `--geno-store-dir`), which is removed
  when the run ends. TRACE checks the free disk space first. Not available with the table on
  disk: `--impute ldknni`, and `--export-qc` above 100 million calls; the QC report is skipped
  with a warning at 16.8 million markers or more. See docs/large_datasets.md.
  `scripts/compare_result_zips.py` compares the results of two runs file by file.
- **The thread and core counts are recorded.** `run_metadata.json` and the report's provenance
  table list the linear-algebra libraries and their versions (numpy's and scipy's), the number of
  threads they use, the CPU cores and `--boot-jobs` with the threads each subsampling worker
  gets. The last digits of p-values depend on the thread count (by at most 5.1e-7 of the p-value
  between 1 and 12 threads in our tests), so results are byte-identical between runs at the same
  thread count; no default changes.
- **The haplotype test accounts for kinship, by default** (`--hap-test`). Each LD block's
  haplotype groups are tested by a mixed model with the GWAS scan's own kinship kernel for the
  block's chromosome (its LOCO kernel, or the whole-genome kernel with `--no-loco`), the PCs
  and any covariates: the null model's heritability is fitted by REML, and the F test and its
  Freedman-Lane permutations are made on the data whitened by that model. `--hap-test pcs`
  (in the GUI "PCs and covariates only", in the LD Blocks & Haplotypes tab's Settings and in
  One-Click) keeps the earlier test. The old test from version 1.1.0. It ignores relatedness and
  can give false positives in related panels. Use it only to reproduce old results. The haplotype
  results gain three columns in both tests: `kinship_model` (the kernel, `none` for the earlier
  test, or why the kinship-aware test could not test a block), `h2_null` (the null model's
  heritability) and `eta2_pcs_only` (the earlier test's η², for comparison). A block the
  kinship-aware test cannot test (for example one whose haplotype groups a covariate defines) is
  kept with the reason and no statistics (the tab's block table gives it under Tested), never
  tested another way, and the block FDR is over the blocks with a p-value. `run_metadata.json`
  records the choice as `Hap_test (--hap-test)`. `LD_blocks_annotated_*` and `LD_triage_*` keep
  their columns; the new ones are in the report's haplotype tables and the GUI's block table CSV.
  See docs/outputs.md.

### Fixed
- **A run with thousands of significant SNPs no longer stalls before the per-SNP box plots.** The
  step that groups near-identical significant SNPs for the plots built the set of all SNP IDs
  again for every SNP it looked at. With 3,744 significant SNPs among 185,618 (a rice run), it ran
  for hours. It now builds the set once, and the results are the same.
- **The haplotype test's parametric F, η² and lead SNP R² now use the user covariates** that
  its permutations already used, so every statistic of the test has one design (intercept,
  PCs, covariates). A tested sample without a covariate value stops the test instead of
  quietly dropping the covariates; the CLI and the GUI drop such samples before the scan, so
  their runs do not meet it. Without covariates nothing changes.
- **One η² per LD block on the haplotype tab.** The forest plot showed an η² of its own,
  computed from the plotted values; it now shows the haplotype test's, as the block table and
  the box plot do. The block table CSV's value over every SNP in the block's window, "η² (all
  SNPs in the block window)", is renamed "Window R² (all SNPs in the block window)", with a
  tooltip on "Download block table (CSV)": it is another quantity, not the test's η².
- **A haplotype result visited again keeps its block plots and downloads.** Switching the test
  (or the Run's LOCO choice) back to an earlier one served its statistics without its per-block
  tables, so the block plots and the assignments downloads were empty; every remembered result
  now keeps its own.
- **Quality fields declared per allele stopped the parse.** An imputation-quality field declared
  `Number=A` or `Number=R` in the VCF header (Beagle 5 declares DR2 so) stopped the run with an
  IndexError. It is now read as one value per record, the ALT allele's.
- **Phenotype IDs with blanks around them are matched.** The blanks were removed for the
  duplicate check only, so a padded ID never matched its VCF sample and that sample was left
  out without a message (25 of 50 samples in a test file). IDs are now matched without the blanks,
  and the log says how many had them.
- **An infinite trait value stops the run with a clear message.** A value such as -inf (from a
  log of zero taken before the file was written) crashed FaST-LMM. The run now stops before the
  scan, naming the count and the first ten samples. A missing value is still dropped as before.
- **The CLI shows the pipeline's warnings.** Messages such as the collapse of duplicate
  positions, partial sample overlap and non-numeric trait values went to a Streamlit call that
  prints nothing outside a page; without a page they now go to the log. The GUI is unchanged.
- **The collapse of records at one position keeps a copy with calls.** It could keep a copy
  without any call, which SNP QC then removed, losing the position although a called copy existed.
- **`--seed` says what it covers:** the subsampling draws. Haplotype permutations are seeded per
  block from the block's position and the trait, so their p-values do not depend on it (the help
  text said they did). No seeding changed.
- **M_eff could come out one too high because of rounding.** The Li & Ji M_eff is a whole number
  in exact arithmetic (the eigenvalues sum to the number of SNPs), but TRACE took the ceiling of
  the floating-point sum, so a sum a few units in the last place above the whole number gave one
  more (and several more when eigenvalues are themselves whole numbers, which happens with exact
  copies of markers). Eigenvalues within 1e-9 of a whole number are now set to it, as the
  per-block M_eff already did, and the sum gives that whole number. On the data shipped and used
  in the paper: the old example 73 to 72, pepper (BX, FWe) 503 to 502, tomato 248 unchanged, the
  new example 115 unchanged. Only M_eff and what uses it change (the M_eff threshold, the
  Significant_Meff columns and what follows from them); Bonferroni and FDR results do not. The
  M_eff matrix is now summed over blocks of 4,096 SNPs, in memory as on disk: the same M_eff
  (checked on 469 tables at block sizes from 1 to 4,096) with far less memory.

### Changed
- **LD-block detection keeps the r2 matrices of its last eight windows only.** It kept every
  window it had computed, which at a million SNPs grew past the size of the genotype table. A
  window that comes back is recomputed, with the same result; no output changes.
- **The Help page and docs/outputs.md list every file of the results ZIP**, one line
  each, with when it is written: the per-model tables, the LD heatmaps, the QC and
  subsampling exports, and the combined ZIP's folders and `failed_traits.txt`. The Help
  page links each input file (VCF, phenotypes, covariates, gene and descriptions files)
  to its docs. The Gene Annotation tab's download is now `Gene_annotation_<trait>.csv`,
  with one underscore like the other GUI files (it was `Gene_annotation__<trait>.csv`).
- **The example forms LD blocks at the defaults.** `examples/` now holds 100 inbred
  lines and 600 SNPs on 3 chromosomes, mosaics of four founder haplotypes, so LD decays
  with distance. Trait1's QTL (`chr02_1010000`) sits in an LD block: at the defaults
  the CLI finds that block, tests its haplotypes and draws its LD heatmap, and the
  GUI's defaults find the same block. The old example had independent SNPs and so no
  block at the defaults. It is kept byte for byte in `tests/data/example_v1/` for the
  tests that pin its numbers. `benchmarks/reproduce.sh` checks for a Bonferroni hit
  within 100 kb of the new QTL.
- **GFF3 gene files are read directly.** The gene file of the Gene Annotation tab,
  of One-Click's overrides and of `--gene-model` may be a GFF3, plain or gzip-compressed,
  recognised by its content rather than its name. Its `gene` rows give the chromosome,
  start, end, strand and gene ID (the `ID` without a leading `gene:` or `gene-`); the
  description is the first of `description`, `Note` and `product`, percent-decoded. A
  descriptions file overrides those descriptions for the genes it lists. The uploaders
  accept .gff3, .gff, .gff3.gz and .gff.gz. The conversion script in
  docs/gene_model_upload.md now strips the same ID prefixes, so its table equals a direct
  read. Gene tables load as before.
- **Interactive HTML files open offline.** Every interactive figure the GUI exports
  (the PCA plot, the MLM, MLMM and FarmCPU interactive Manhattans, and each per-figure
  HTML download) now embeds plotly.js, through one helper
  (`utils.pub_theme.plotly_html_bytes`). Before, they fetched plotly.js from a CDN when
  opened, so they stayed blank offline or behind a firewall. Every point is kept and
  WebGL is unchanged; each file grows by the size of plotly.js, about 4.8 MB. Nothing
  changes in the CLI ZIP.
- **Sample-to-haplotype files for every model.** Besides
  `Haplotype_assignments_<trait>.csv` (MLM, unchanged), the CLI and One-Click ZIP now
  hold `Haplotype_assignments_<trait>_<model>.csv` for MLMM and FarmCPU when they have
  tested blocks. Before, only the primary model's blocks were covered, so in the
  three-model example runs most MLMM and FarmCPU blocks had no assignments.
- **MLMM reports each cofactor's own p-value.** MLMM's final scan includes its
  cofactors as covariates, so each cofactor was tested with itself and got p = 1. Each
  cofactor is now re-tested with the other cofactors as covariates, in MLMM's own model
  (the same kinship and base covariates). A failed re-test is reported as NaN, with a
  warning. The cofactor table gives each cofactor's entry p-value and this one
  (`PValue_entry`, and `PValue_final` as in FarmCPU's pseudo-QTN table). Every other
  SNP's p-value is unchanged.
- **MLMM chooses its cofactors by SNP.** Before this commit, MLMM took cofactors by row
  position in single_snp's p-sorted output, so its first cofactor was the genome's
  first SNP. On the Varitome locule run that was SL25ch01p25864 (MLM p 0.47), not the
  top hit SL25ch02p47366168. Its later cofactors, and the p-values it compared with the
  entry threshold, were mismatched the same way. MLMM now puts single_snp's p-values in
  SNP order (by sid_index, else by SNP ID, never by row) before the exclusion window,
  the choice, the entry threshold and the BIC step. The final scan no longer falls back
  to row order. MLMM's results change; MLM and FarmCPU results do not.
- **MLMM's cofactor window is the LD window.** In the CLI and One-Click, MLMM used its
  250 kb default, while the GWAS page's single-trait run passed the LD window. Every
  path now passes the run's LD window: `--ld-flank-kb` (One-Click's manual flank) when
  given, else int(2 × the median LD decay), else 300 kb. The CLI estimates the decay
  before the models for this. `run_metadata.json` records "MLMM cofactor window
  (kb)" when MLMM runs. MLMM's results can change; MLM and FarmCPU results do not.
- **One LD-window rule, the CLI's, in the GUI.** Every default LD window is now
  int(2 × the median LD decay), or 300 kb when the decay cannot be estimated, the
  rule the CLI and One-Click use (`gwas.windows.ld_window_kb`). This covers the LD Blocks &
  Haplotypes Settings default, Post-GWAS's auto-built blocks, the candidate-interval
  edge flank and the Regional Plot's window. The Settings default had been held within
  200 to 2000 kb (200 kb on the Varitome data, where the CLI uses 144 kb), and the GUI
  had assumed a 200 kb decay without an estimate. Without an estimate the GUI now
  detects blocks without a decay value, as the CLI does. The decay line names the run
  it comes from, the GWAS page's Run GWAS; One-Click writes its results to its ZIP only
  and hands nothing to Post-GWAS.
- **The haplotype PCA inset uses the haplotype labels of the assignments CSV.** On the
  LD Blocks & Haplotypes tab, "Population structure (PC1 vs PC2, coloured by
  haplotype)" now colours each sample by its Haplotype in the block's
  sample-to-haplotype assignments. Before, every sample outside the plotted groups was
  "Other". A group below the minimum size now keeps its H label. The samples the
  haplotype test left out are hollow, with "not tested" in the legend, and the samples
  without a haplotype are grey, "no haplotype". The inset's data CSV has the same labels
  and an `in_haplotype_test` column.
- **Samples without a covariate value are dropped in the GUI too.** There is now one
  rule, the CLI's. The GWAS page's single-trait run and One-Click drop the analysis
  samples that lack a covariate value at the stage the CLI does, before SNP QC, kinship
  and the PCs. One-Click used to stop the trait, and the GWAS page stopped the run. A
  warning says how many samples were dropped: "N of M analysis samples have no
  covariate value and were dropped: …". The CLI's results are unchanged.
- **β_MLM and its SE are per ALT allele copy.** FaST-LMM fits each SNP standardised, so
  its effect and SE were per standard deviation of the SNP's dosage. TRACE now divides
  both by that SD (over the same samples), so `Beta_MLM` and `SE_MLM` share `Beta_OLS`'s
  unit in `GWAS_<trait>.csv`, `Significant_SNPs_<model>.csv`, the box-plot titles and
  the report. `GWAS_<trait>.csv`'s `SnpWeightSE` column is now `SE_MLM`, its one name.
  The effect flag keeps its rules and now compares like units: a low-frequency SNP had
  been flagged structure-sensitive only because its per-SD effect is small. P-values,
  blocks and every other result are unchanged. The GWAS page hands the four effect
  columns to Post-GWAS, so the lead-SNP titles there show β_MLM and the flag and the
  Significant SNPs tab fills its effect columns.
- **The LD Decay tab is gone; the estimate is shown where it is used.** The Post-GWAS
  page has five tabs. The LD Blocks & Haplotypes tab's Settings show, next to the LD
  window, "LD decay (median across chromosomes): X kb, so the window is 2X kb" and a
  collapsed "Show LD decay curve" with the per-chromosome summary and curve, both from
  the GWAS page's estimate (the CLI's function and settings), with no recomputation.
  The tab re-estimated the decay with its own settings and could make the GUI's window
  differ from the CLI's. The window's default is unchanged.
- **A simpler Gene Annotation tab, for LD blocks and unblocked SNPs.** Step 1 loads the
  gene models (the built-in one for the species and build, or your own gene file and,
  optionally, a descriptions file, with visible uploaders); Step 2 lists every LD block
  and every unblocked significant SNP with the genes inside its candidate interval, in
  one table with a summary line ("N intervals, M genes") and one CSV, by the CLI's
  rules. The "Use Haplotype GWAS blocks instead" checkbox, the flanking sliders and the
  summary and full tables are gone. Every gene-file uploader (this tab and the One-Click
  overrides) carries the same "?" format help, written next to the loader; a file with
  two columns for one field is refused with a clear message, a UTF-8 byte-order mark
  (Excel's "CSV UTF-8") is accepted, and a warning names the genotype data's chromosomes
  that have no gene in the file. The CLI gains `--gene-desc` (a custom species' gene
  descriptions), named in `run_metadata.json` when given; docs/gene_model_upload.md
  describes both files and both flags.
- **An LD block's genes are searched over its candidate interval; flanking genes are
  no longer reported.** The candidate interval runs from the nearest post-QC marker
  before the block's first SNP to the nearest after its last, by the rule unblocked
  SNPs already followed (the same chromosome-edge flank and 5 Mb clamp around the lead
  SNP): the haplotype's true boundary lies between the block's outer SNPs and those
  markers. The block's own Start/End are unchanged. `LD_blocks_annotated_<model>.csv`
  adds `Interval_start` / `Interval_end`, the significant-SNP table gives an in-block
  SNP its block's interval, and a gene counts only when it overlaps the interval. The
  flanking-gene columns (`upstream_/downstream_gene_*`), the `intergenic` status and the
  `flanking_within_500kb` gene evidence are gone; an interval without genes is
  `no_genes_nearby`. The Regional Plot shades the same interval for blocks and
  unblocked SNPs, the block table shows it, and the GUI uses the CLI's chromosome-edge
  flank (2 × LD decay).
- **The haplotype plot shows the values Tukey HSD uses.** On the LD Blocks &
  Haplotypes tab, when the GWAS used PCs, the block's plot now shows the PC-adjusted
  values that Tukey HSD, the compact letters and the forest plot use (the haplotype test
  adjusts for the same PCs), with the axis "<trait> (PC-adjusted)"; with no PCs it shows
  the uploaded values, as before. Each box carries its compact letters again, and
  "Download plotted data (CSV)" under the plot gives each sample's haplotype, allele
  sequence, uploaded and plotted values, their scale and its letters; the block's
  assignments downloads sit with the block selector. The block table shows one η² on
  screen, the haplotype test's; the window-based η² stays in its CSV. Its "?" no longer
  explains β, which the lead-SNP plot's "?" now does, as Help → Effect Sizes and
  docs/outputs.md do (β_MLM and β_OLS both per ALT allele copy: see the β entry above).
- **How-to text on the LD Blocks & Haplotypes tab moved into "?" tooltips.** The
  lead-SNP gallery's instructions are the "?" on "Lead SNP genotype effect"
  (they had named "two structure-aware tests"; the panel shows one rank test,
  adjusted for neither structure nor kinship, as its caption says), and how to
  read several compact letters is the "?" on "Compact Letter Display (CLD)".
  Caveats (the rank test's disclosure, the uploaded-values note, the excluded and
  tested sample counts), statuses and every warning stay visible. The "no LD
  blocks" message names the seed threshold and where to change it.
- **The block's haplotype plot is drawn like the lead-SNP plot.** On the LD
  Blocks & Haplotypes tab, the selected block's plot is a box per tested
  haplotype group with the samples as jittered points, coloured by haplotype:
  Interactive by default (hover: sample, haplotype, value, in the test) or
  Static (PNG/SVG/PDF), in the collapsed "Plot style". It shows the samples,
  labels and uploaded values the haplotype test compared (its rows of the
  assignments table), and its title names the block, P (permutation) and η².
  The raincloud, the interactive raincloud and the older seaborn box plot are
  gone; the compact letters are in the CLD table. The MLG allele sequences,
  Tukey HSD and compact letters now sit in one collapsed expander, "Haplotype
  details: allele sequences, Tukey HSD, compact letters", with their downloads.
  If Tukey HSD fails, its warning stays on the page above that expander, and the
  forest plot and the PCA still show (the tab used to stop there).
- **One block table on the LD Blocks & Haplotypes tab, with one CSV.** "LD
  blocks and haplotype tests" has one row per detected block, with its
  haplotype test and its LD coherence joined on the block's chromosome, start
  and end. A "Tested" column says "yes", or why the block was not tested (the
  haplotype test now records the reason, e.g. "fewer than 2 haplotype groups of
  at least 3 samples"; nothing it writes to a file changes). The default view
  shows the main columns, "Show all columns" the rest. A value that appeared
  under two names is shown once: P (permutation) (PValue is P_perm), FDR q, mean
  r². The test's η² and the η² over every SNP in the block's window differ, so
  both are kept, labelled. "Download block table (CSV)" gives every column and
  replaces the three CSV buttons (LD blocks, LD-block coherence, haplotype
  analysis results). The tab's ZIP bundle, which held the same two tables, is
  gone; the GWAS page's results ZIP is unchanged.
- **The LD Blocks & Haplotypes tab keeps its settings in one panel.** A collapsed
  "Settings" panel at the top of the tab holds the LD-block detection settings
  (seed threshold, top N, LD window, r² threshold), the haplotype test's
  (permutations and the two minimum sample counts) and the lead-SNP gallery's
  significance rule, with one line above it naming the values in use. The keys
  and defaults are unchanged. "Significance threshold (P <)" is now "Seed
  threshold for LD blocks (P <)": it is the CLI's `--ld-seed-p`, not the
  genome-wide significance rule. The gallery names the rule it uses.
- **"How to read" guidance moved into "?" tooltips.** On the haplotype tab, the
  "How to read effect sizes (η²)" and "How to read LD-block coherence (r²)"
  buttons are now one "?" on "Association results per block"; on the GWAS page,
  "How to read discovery frequency" is a "?" on "Per-SNP discovery frequency".
  The text is the same, with one addition: η² is not adjusted for kinship, so in
  structured panels it can be inflated (it comes from a one-way ANOVA on the
  haplotype groups). The pointer to Help → Interpreting Results stays.
- **Plots on screen have three sizes.** In wide mode the Manhattan, overlay,
  subsampling Manhattan and regional plots and the QC report strip fill the
  page; the LD heatmaps, PCA, haplotype box/raincloud plots, haplotype PCA and
  the p-value and subsampling histograms are at most 700 px wide; the QQ, scree,
  MAF, PC spectrum, lead-SNP box, forest and phenotype-QC plots at most 450 px.
  In Post-GWAS's LD Blocks & Haplotypes tab, the forest plot and the haplotype
  PCA sit in collapsed expanders, and both "Plot style" choices start collapsed.
  One helper shows every plot (`utils.pub_theme.show_plot`), and a test fails if
  a page shows a plot without it. Display only: downloads, ZIPs and reports are
  unchanged.
- **The lead-SNP box plots colour each dot by its sample's haplotype.** In the
  report's per-SNP plots (CLI and One-Click) and in Post-GWAS's "Lead SNP
  genotype effect" panel, each dot takes the colour of its sample's haplotype in
  the LD block that contains the SNP (by position, the block of the
  significant-SNP table's `Block_ID`), with the labels of
  `Haplotype_assignments_<trait>.csv` and a legend beside the plot. A label has
  the same colour in the box plot, the haplotype tab's block plots and the
  report (`utils.pub_theme.haplotype_color_map`: H<n> takes palette colour n,
  Other is grey); the block plots keep their colours. A SNP outside every block,
  or in a block with no haplotype groups, keeps the single colour, with a note
  under the plot. The dots do not move (same jitter) and the boxes are
  unchanged. The y axis names the trait, with its scale when the GWAS
  transformed it ("locule_number (rank-INT)"), instead of "Phenotype". In the
  ZIP only the report's per-SNP images change.
- **The phenotype QC panel's transformation is used only after Apply.** The
  panel's trait and method selectors now drive only its preview. Before, the
  GWAS page transformed the previewed trait in every run, even when the panel
  was never opened: by default the first numeric trait was rank-INT
  transformed, and choosing another trait in the preview moved the
  transformation to it. An Apply button ("Apply rank-INT to Trait1") records a
  method for a trait, the panel lists the applied methods with a Remove button
  each and Reset all, and a status line above the panel says what is used
  ("Phenotypes used: raw values for all traits"). Each applied method is that
  trait's normalisation, applied where the CLI applies `--norm` (to the trait's
  own analysis samples, after phenotype QC, by `gwas.qc.normalise_phenotype`),
  so with rank-INT applied One-Click gives that trait the CLI's `--norm int`
  results and every other trait the CLI's `--norm none` results. The phenotype
  table keeps the uploaded values, so the QC report and the haplotype tests use
  raw values, as in the CLI (Post-GWAS too, which says so for an applied trait);
  the subsampling and the lead-SNP box plots use the analysed phenotype. The
  GUI's Log10 is the CLI's `log` (both base 10). A new phenotype file clears the
  list. One-Click records each trait's normalisation (`none` or the CLI slug,
  such as `int`) as `Normalization` in its `run_metadata.json` and report; the
  single-trait view records it as `Normalization method` and in its run
  manifest, whose applied list and subsampling summary now describe that run
  only (they could carry an earlier run's or trait's entry). The single-trait
  view's subsampling results are now recomputed when the trait's normalisation,
  the phenotype file or another scan input changes (they were cached on the
  trait and the subsampling settings alone). One-Click's figures are now drawn
  with the CLI's matplotlib settings, so its ZIP equals the CLI's figure for
  figure: the GUI's publication theme (fonts, title weight, tight bounding box)
  reached them before, so every figure and the report's embedded images
  differed from the CLI's.
- **MLMM and FarmCPU results carry the same significance columns as MLM.**
  `GWAS_MLMM_<trait>.csv` and `GWAS_FarmCPU_<trait>.csv` now have `FDR`,
  `Significant_FDR`, `Significant_Bonf`, `Significant_Meff` and, for a numeric
  `--sig-thresh`, `Significant_Custom`, as the MLM table `GWAS_<trait>.csv`
  always had. One function writes them for every model
  (`gwas.significance.add_significance_columns`), in the CLI, One-Click and the
  single-trait run. The thresholds are MLM's (Bonferroni over the SNPs tested,
  M_eff, custom); FDR is Benjamini-Hochberg within each model's own p-values,
  and a missing p-value is never significant. Before, the CLI wrote none of these
  columns for MLMM and FarmCPU, 1.1.0's One-Click took their Bonferroni threshold
  from each table's row count, and the single-trait run wrote only FDR and
  Bonferroni. The single-trait run writes `Significant_Meff` when the M_eff rule
  is chosen and `Significant_Custom` under the Custom p-value rule, for every
  model as for MLM. Each model's `Significant_SNPs_<model>.csv` now fills
  `Passes_Meff`, `Passes_Bonf`, `Passes_FDR` and `FDR` for MLMM and FarmCPU, and
  under `--sig-thresh fdr` MLMM and FarmCPU now get their significant-SNP tables
  and isolated-SNP intervals. MLM's tables, LD blocks and haplotype tests are
  unchanged.
- **LD heatmaps cover the blocks whose lead SNP is significant in MLMM or
  FarmCPU**, in the CLI as in One-Click (1.1.0's One-Click drew them; the CLI
  drew MLM's only). A block that several models report with the same bounds is
  drawn once. The title names every model whose own block with these bounds has
  a significant lead, or in which the marked lead SNP is significant
  ("significant in MLM, FarmCPU"); when that is MLM alone the title stays as it
  was, so an MLM-only run's heatmaps are unchanged. Models often delimit one
  signal with slightly different bounds: those are separate blocks, each with
  its own heatmap. The HTML report embeds the new heatmaps and, when several
  models ran, gives one significant-SNP count per model.
- **Several traits in one run, in the CLI and in One-Click Full Analysis.**
  `--trait` now takes one or more names, and One-Click runs every selected
  trait. Each trait goes through the same per-trait code
  (`gwas.pipeline.run_trait_pipeline`) on its own phenotyped samples: QC,
  imputation, kinship, LOCO kernels, PCs, every model, LD blocks, haplotype
  tests, figures and the report are rebuilt for each trait, and only the parsed
  VCF and the gene models are shared. Each trait gets its own ZIP, and a combined
  ZIP holds them all. A trait that fails is reported (`failed_traits.txt` in the
  combined ZIP) and the others still run; the CLI then exits non-zero.
  Multi-trait support leaves a single-trait CLI run unchanged, byte for byte.
- **One-Click Full Analysis now runs the CLI's per-trait pipeline**, so its ZIP
  has the CLI's layout: file names such as `Manhattan_MLM.png`, the CLI's report
  and `run_metadata.json`, and the QC tables. The MLMM and FarmCPU significance
  columns follow the pipeline's rule (see above) instead of One-Click's own, which
  took Bonferroni from each table's row count, and the ZIP has no interactive
  Manhattan HTML (the single-trait detailed view still writes it). As in the
  detailed view, a trait whose samples lack covariate values stops with its own
  error rather than dropping them; each trait gets the transformation applied to
  it in the phenotype QC panel (see above), and its `run_metadata.json` records
  it. The detailed view below One-Click uses the first selected trait that passes
  phenotype QC.
- **LD-block heatmaps now show the block's neighbouring SNPs.** Each heatmap in
  the CLI and One-Click ZIPs, and the Post-GWAS Local LD view of a detected
  block, shows the block plus the SNPs immediately on each side, so a reader can
  see whether LD drops at the block edges. The block is at most half of the
  SNPs shown and at least 20 SNPs are shown (unless the chromosome has fewer),
  and windows above 200 SNPs are thinned evenly. The block's span is outlined,
  its lead SNP's tick is marked, and ticks are positions in Mb. One function
  draws all three (`gwas.plotting.plot_ld_block_heatmap`), and every LD heatmap
  PNG in the ZIPs has the same size (1050 x 900 px). This change leaves file
  names, the blocks that get a heatmap and every table as they were (for the
  MLMM and FarmCPU heatmaps and columns, see above); the Local LD r² downloads
  for a detected block still cover its member SNPs.

### Added
- **Every figure in the GUI offers the data it draws.** A "Download data (CSV)"
  button sits under each figure: the MAF spectrum, the QC report strip and the
  p-value and subsampling histograms give their bars (bin start, end and count;
  the strip names its three panels), the PC spectrum and scree plot their
  components, the phenotype QC panel its values and Q-Q points, and the haplotype
  PCA its samples (PC1, PC2, haplotype). Where a figure's data already had a
  download, that button is the figure's: the haplotype box/raincloud plot's is
  the block's "Download sample-to-haplotype assignments (CSV)", now under the
  plot; the forest plot's, the lead-SNP plot's, the LD heatmap's (r²), the
  regional plot's and the PCA's (scores) stay. The genome-wide Manhattan and QQ
  plots point to the results download instead of making a second copy. The
  figures, their image downloads and every export are unchanged.
- **The Post-GWAS lead-SNP box plot is interactive, with its data as CSV.** A
  "Plot style" choice (collapsed; Interactive by default) shows the "Lead SNP
  genotype effect" plot in Plotly: a box per genotype class with the same
  jittered points, coloured by haplotype as the static plot, and on hover each
  sample's ID, genotype (dosage and alleles, e.g. "1 (A/G)"), plotted value and
  haplotype. The values, title (MLM effect and η²) and y axis are the static
  plot's. "Download plotted data (CSV)" under the plot gives one row per plotted
  sample: Sample, Genotype_dosage, Genotype_alleles, Phenotype (the values
  plotted), Phenotype_scale (raw, or e.g. rank-INT), Haplotype and
  In_haplotype_test, as `lead_snp_<SNP>_<trait>.csv`. The static plot and every
  export are unchanged.
- **Sample-to-haplotype assignments.** `Haplotype_assignments_<trait>.csv` (CLI
  and One-Click ZIP, `tables/`) lists, for every tested LD block of the primary
  model (MLM), every analysis sample of the trait with its haplotype label
  (H1..Hk or Other), its allele sequence and the markers in that sequence
  (`SNP_order`), whether the haplotype test took it, why it was left out, and
  the phenotype value the test used. One function builds it from the test's
  own per-sample tables (`gwas.haplotype.haplotype_assignments`). In Post-GWAS
  the LD Blocks tab has "Download sample-to-haplotype assignments (CSV)" for the
  selected block and "Download assignments for all blocks (CSV)", next to the
  block selector; they replace the per-sample MLG ledger. The tab's MLG plots,
  tables and Tukey test now show the groups the haplotype test tested (its
  samples, labels and allele sequences): before, the tab re-derived the groups
  from the polymorphic SNPs in the block's coordinate window (without the test's
  call-rate filter and r² thinning), which for many blocks gave other groups than
  the test's. The haplotype results, the LD block tables and
  every other ZIP member are unchanged. See `docs/outputs.md`.

### Fixed
- **Static plots are no longer drawn tiny after a switch from Interactive.** A small or
  medium static figure (for example the lead-SNP and haplotype plots in Static, the
  forest plot and the haplotype PCA) is now drawn at its size class's width directly,
  450 or 700 px, instead of stretching inside a fixed-width box that the browser could
  draw before measuring (after a style switch, or in a collapsed expander). Interactive
  figures keep their fixed-width box, now keyed after the figure, so switching styles
  draws a new element. Display only: no exported figure changes.
- **The static Manhattan plots highlight only the SNPs that pass the
  threshold.** They enlarged every SNP with p at or below it, and counted those
  in the legend's "Significant (n)", while every table counts p below it. A
  p-value exactly on the threshold was highlighted but significant nowhere. The
  plots now highlight the rule's own column (CLI, One-Click and the single-trait
  view). No comparison run has a p-value on a threshold, so no figure changes
  there.
- **A numeric `--sig-thresh` no longer crashes the CLI when plots are on.** The
  bare number reached plot code that needs a text label, so every such run
  stopped at the MLM Manhattan plot and wrote no results (with several traits,
  every trait failed and the CLI exited non-zero); the MLMM and FarmCPU
  Manhattan plots had the same fault. The threshold line is now labelled with
  the threshold ("p < 1.0e-05"), as in the report and the GUI.
- **Post-GWAS works under the FDR rule with MLM as the downstream model.** The
  single-trait view hands MLM's table over without significance columns, so the
  Significant SNPs tab could not build its table and the lead-SNP gallery
  stopped the page, whether FDR was the GWAS page's rule or was chosen in the
  tab's or the gallery's own threshold selector. Both now apply
  Benjamini-Hochberg within that table, as the GWAS tables do; the GWAS page's
  own table and downloads are unchanged.
- **The cross-model consensus decides a tie at a threshold as the tables do.**
  Under `--sig-thresh fdr` it tested q < 0.05, while `Significant_FDR` is the
  Benjamini-Hochberg decision (a p-value at or below its step, rank/m × 0.05).
  Each model is now judged by its own significance column for the analysis
  rule. Only a SNP whose p-value sits exactly on its Benjamini-Hochberg step is
  affected (its q-value is then 0.05 up to rounding), and such a SNP can only be
  added to the consensus.
- **The CLI and One-Click report's significant-SNP count follows the analysis
  rule.** It counted the rows of the primary model's significant-SNP table, so
  when that table was empty (no SNP passes the rule) or not built
  (`--no-isolated-rescue`) the summary card fell back to the FDR < 0.05 count
  while still naming the rule (for example "Significant (Bonferroni
  (m=...))"). It now counts the rule's own significance column, in MLM-only runs
  too; this card is the one MLM output that changes, and only in those two
  cases. The single-trait run's report still falls back when the M_eff
  computation fails.
- **Under `--sig-thresh fdr`, `CrossModel_Consensus.csv` judged MLMM and FarmCPU
  SNPs by the M_eff threshold**, because their tables had no FDR column. Each
  model is now judged by its own FDR decision.
- **The single-trait run's results ZIP could hold an earlier run's tables.** It
  was cached under the trait, the models, the SNP count and λGC, so a re-run
  after changing the significance rule served the first run's tables and
  metadata. It is now rebuilt on every run.
- **One-Click multi-trait runs used the first trait's data for every trait.**
  The loop resolved genotypes and the phenotype reader through session keys
  registered once, from the first selected trait; 1.1.0 refused multi-trait runs
  instead. Each trait now runs on its own data (see above), and a trait whose
  samples differ from the others' no longer aborts the batch.
- **One-Click applied uploaded covariates only to the haplotype test.** They now
  enter every model, as in the CLI.
- **One-Click MLMM ignored a changed PC count within a session** (its cached
  covariate reader was keyed without the PC count). One-Click no longer caches
  per-trait results across runs.
- **Fresh installs could not run a GWAS: `six` is now a declared dependency.**
  pysnptools 0.5.14 imports `six` without declaring it, and python-dateutil
  4.0.0, which used to bring it in, no longer does. A new environment or
  container image therefore failed on `import fastlmm.association` before any
  analysis ran. TRACE now requires `six` directly (1.17.0 in `constraints.txt`).
- **The Docker CLI example now runs the container as your own user.** The
  README command passes `--user "$(id -u):$(id -g)"` to docker and mounts the
  working folder at `/data`, and the image sets `HOME=/tmp` so matplotlib can
  write its cache under any uid. The README used to say the results belonged to
  your user, which did not hold for a folder that uid 1000 could not write.
- **`trace-gwas` checks that the `--output` folder is writable before it
  starts.** An unwritable folder (typically a Docker mount owned by another uid)
  used to fail with a traceback when the results ZIP was written, after the
  whole analysis. The run now stops within seconds with a message that names
  the folder, the uid and the Docker fix.

## [1.1.0] - 2026-09-21

### Changed
- **Renamed the "LD Analysis" page to "Post-GWAS Analysis"** (sidebar shows
  "Post GWAS Analysis"), aligning the app with the manuscript's terminology.
  The page URL changed to `/Post_GWAS_Analysis`.
- **`--n-pcs` default changed 4 → 0** (fixed PC count; TRACE no longer auto-selects PCs).
- **Slimmed the shipped Docker image** (~3 GB → ~1.94 GB): removed the build-time
  toolchain (`build-essential`, `gcc`, `gfortran`, `libopenblas-dev`), every
  dependency installs from a wheel (numpy and scipy bundle their own OpenBLAS),
  and set file ownership during `COPY --chown` instead of a post-hoc `chown -R`.
- **The published container image now installs its dependencies against
  `constraints.txt`**, the pinned set used for the manuscript benchmarks.
  Earlier images resolved from PyPI at build time and could ship newer major
  versions of numpy, scipy and statsmodels than the tested configuration.
- **Pinned `fastlmmclib`, and corrected what `constraints.txt` claims to be.**
  fastlmm declares `fastlmmclib>=0.0.7`, which the constraints file did not pin,
  so CI and the published container each resolved whatever version PyPI offered
  on the day (0.0.8 since 2026-08-07). It is now fixed at 0.0.7, the version the
  benchmark environment has carried since 2026-02-23. The file's header no
  longer describes itself as a bit-for-bit reproducible environment: it is
  hand-maintained rather than generated, covers direct dependencies only, carries
  no hashes, and its pins were transcribed from the development environment
  rather than captured at run time.
- **Rewrote the README**: corrected the title and the archive DOI, merged
  the install paths into one "Getting started" with three routes, documented
  `--covar`, regional plots, LD-kNNi imputation and the PC spectrum diagnostics,
  and refreshed the screenshots (now five).

### Added
- **Regional Plot tab** (locus-zoom-style): −log₁₀(p) vs position around a lead
  SNP, coloured by r² to the lead, with the detected LD block shaded and a gene
  track; interactive (Plotly) + static (PNG/SVG/PDF) with numeric CSV export.
- **Local LD** tab gained numeric r² export (long `SNP_A,SNP_B,r2` + square matrix).
- **PC-selection diagnostics** (report-only): the run report + GUI now show the
  genotype-PCA eigenvalue spectrum (variance explained per PC + cumulative).
  These REPORT and never set the PC count.
- **README test-suite table is now generated** by `scripts/gen_test_table.py`
  from the collected pytest suite, grouped into thematic rows whose counts sum
  to the collected total (the generator aborts if a test file is unmapped), so the
  table can no longer drift from the suite; it also prints the line-coverage figure.

### Removed
- The per-block "Block Heatmaps" tab moved behind a "Legacy views" expander,
  superseded by the region selector's *Detected block* mode + the Local LD tab.
- **The λGC auto-PC selector** and its flags (`--auto-pcs`, `--pc-strategy`,
  `--max-pcs`, `--pc-band-lo/-hi`, `--pc-parsimony-tol`). TRACE now uses a fixed
  `--n-pcs` default plus the report-only PC-selection diagnostics above.

### Fixed
- **LD blocks could bridge uncorrelated SNP clusters.** The block merge fused
  overlapping intervals with no correlation check, and the adjacency split only
  tested consecutive pairs, so a long-range LD edge (or a chain of segment
  merges) could join distinct clusters into one low-coherence block. Block
  formation now enforces coherence unconditionally, controlled by
  `--ld-merge-r2`: a block is split (following its seed) until its members reach
  that mean r², and two overlapping blocks fuse only when their cross-seam AND
  union mean r² both reach it.
- **Haplotype visualisation could crash, or align the wrong samples, after
  returning to the GWAS page.** The LD/haplotype tab read its genotype matrix
  from a session copy refreshed on every Post-GWAS page load, but its sample-ID
  axis — and so the phenotype mask built from it — from a list written only by a
  completed GWAS run. Analysing a trait that is not the phenotype file's first
  column, then revisiting the GWAS page, rebuilt the first without the second and
  the tab died with a sample-mask mismatch; the lead-SNP gallery hit the same
  desync one step earlier. Both panels now take the genotype matrix, the SNP axis
  and the sample axis from the same source, so the mask always belongs to the
  analysed trait. Haplotype visualisation requires the genotype objects from the
  analysed trait's own run: where the loaded genotypes were rebuilt for a
  different trait, the tab now names both traits and both sample counts and asks
  for a re-run rather than reporting statistics that would not match that
  analysis.
- **One-Click Full Analysis returned the first trait's results for every other
  trait.** The per-trait loop resolved its genotype matrix and phenotype reader
  through session keys registered once above the loop from the first selected
  trait, and the scan regresses the resolved reader rather than the phenotype
  vector passed alongside it, so traits after the first were scanned against the
  first trait's data and reported under their own names — with no error, and with
  each trait's own OLS effect sizes beside another trait's p-values. The pipeline
  now runs **one trait per run** and refuses to start on a multi-trait selection,
  naming the CLI batch route (`cli.py --trait`); per-trait key registration is a
  follow-up.

## [1.0.0] - 2026-04-21

### Initial release

TRACE v1.0.0 is the paper companion release. Features lifted from the
parent repository:

- **Multi-model GWAS**: MLM, LOCO, MLMM, FarmCPU with lambda-based
  automatic PC selection
- **LD analysis**: Graph-based block detection with adaptive r² thresholds,
  gap-aware splitting, and LD decay curves
- **Haplotype mapping**: MLG-based Freedman-Lane permutation testing
- **Gene annotation**: Automatic annotation via gene models (SL3.1 / ITAG4.0 /
  pepper CDS) with flanking gene reporting
- **Cross-model consensus**: Multi-model agreement scoring across MLM,
  FarmCPU, and MLMM for high-confidence candidate identification
- **Subsampling stability**: Resampling GWAS for signal reproducibility
- **Publication-ready output**: 300 DPI static plots, interactive Plotly
  figures, colorblind-safe palette (Wong 2011), ZIP export with HTML report
- **CLI for HPC**: Full headless pipeline (`trace-gwas` command) with
  interactive wizard
- **Streamlit UI**: Two-page interface (GWAS Analysis + LD & Haplotype
  Analysis) with one-click full pipeline
- **367 automated tests** across 19 test files. Includes an end-to-end
  null-calibration gate: λGC on a permuted phenotype must fall in
  [0.85, 1.15] with zero genome-wide hits.
- **CI**: GitHub Actions on Python 3.11 and 3.12 with a 50% coverage floor
