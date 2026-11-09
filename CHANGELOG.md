# Changelog

All notable changes to TRACE are documented in this file.

## [Unreleased]

### Changed
- **The haplotype plot shows the values Tukey HSD uses.** On the LD Blocks &
  Haplotypes tab, when the GWAS used PCs, the block's plot now shows the PC-adjusted
  values that Tukey HSD, the compact letters and the forest plot use (the haplotype test
  adjusts for the same PCs), with the axis "<trait> (PC-adjusted)"; with no PCs it shows
  the uploaded values, as before. Each box carries its compact letters again, and
  "Download plotted data (CSV)" under the plot gives each sample's haplotype, allele
  sequence, uploaded and plotted values, their scale and its letters; the block's
  assignments downloads sit with the block selector. The block table shows one η² on
  screen, the haplotype test's; the window-based η² stays in its CSV. Its "?" no longer
  explains β, which the lead-SNP plot's "?" now does: β_MLM is per standard deviation of
  the SNP's ALT-allele dosage (FaST-LMM standardises each SNP), not per allele copy, and
  β_OLS is per ALT copy, as Help → Effect Sizes and docs/outputs.md now say too.
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
