# Changelog

All notable changes to TRACE are documented in this file.

## [Unreleased]

### Changed
- **Several traits in one run, in the CLI and in One-Click Full Analysis.**
  `--trait` now takes one or more names, and One-Click runs every selected
  trait. Each trait goes through the same per-trait code
  (`gwas.pipeline.run_trait_pipeline`) on its own phenotyped samples: QC,
  imputation, kinship, LOCO kernels, PCs, every model, LD blocks, haplotype
  tests, figures and the report are rebuilt for each trait, and only the parsed
  VCF and the gene models are shared. Each trait gets its own ZIP, and a combined
  ZIP holds them all. A trait that fails is reported (`failed_traits.txt` in the
  combined ZIP) and the others still run; the CLI then exits non-zero. A
  single-trait CLI run is unchanged, byte for byte.
- **One-Click Full Analysis now runs the CLI's per-trait pipeline**, so its ZIP
  has the CLI's layout: file names such as `Manhattan_MLM.png`, the CLI's report
  and `run_metadata.json`, and the QC tables. MLMM and FarmCPU tables no longer
  carry One-Click's own significance columns, so, as in the CLI, their blocks get
  no LD heatmap, and the ZIP has no interactive Manhattan HTML (the single-trait
  detailed view still writes it). As in the detailed view, a trait whose samples
  lack covariate values stops with its own error rather than dropping them; the
  phenotype QC panel's transform still applies to the one trait chosen there, and
  that trait's `run_metadata.json` now records it. The detailed view below
  One-Click uses the first selected trait that passes phenotype QC.
- **LD-block heatmaps now show the block's neighbouring SNPs.** Each heatmap in
  the CLI and One-Click ZIPs, and the Post-GWAS Local LD view of a detected
  block, shows the block plus the SNPs immediately on each side, so a reader can
  see whether LD drops at the block edges. The block is at most half of the
  SNPs shown and at least 20 SNPs are shown (unless the chromosome has fewer),
  and windows above 200 SNPs are thinned evenly. The block's span is outlined,
  its lead SNP's tick is marked, and ticks are positions in Mb. One function
  draws all three (`gwas.plotting.plot_ld_block_heatmap`), and every LD heatmap
  PNG in the ZIPs has the same size (1050 x 900 px). File names, the blocks that
  get a heatmap, and every table are unchanged; the Local LD r² downloads for a
  detected block still cover its member SNPs.

### Fixed
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
