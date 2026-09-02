# Changelog

All notable changes to TRACE are documented in this file.

## [Unreleased]

### Changed
- **Renamed the "LD Analysis" page to "Post-GWAS Analysis"** (sidebar shows
  "Post GWAS Analysis"), aligning the app with the manuscript's terminology.
  The page URL changed to `/Post_GWAS_Analysis`.
- **`--n-pcs` default changed 4 → 0** (fixed PC count; TRACE no longer auto-selects PCs).

### Added
- **Regional Plot tab** (locus-zoom-style): −log₁₀(p) vs position around a lead
  SNP, coloured by r² to the lead, with the detected LD block shaded and a gene
  track; interactive (Plotly) + static (PNG/SVG/PDF) with numeric CSV export.
- **Local LD** tab gained numeric r² export (long `SNP_A,SNP_B,r2` + square matrix).
- **PC-selection diagnostics** (report-only): the run report + GUI now show the
  genotype-PCA eigenvalue spectrum and a conventional-criteria table (Kaiser /
  cumulative-variance / Marchenko–Pastur edge / broken-stick; Horn's parallel
  analysis opt-in via `--pc-diagnostics-parallel`). These REPORT and never set the
  PC count.

### Removed
- The per-block "Block Heatmaps" tab moved behind a "Legacy views" expander —
  superseded by the region selector's *Detected block* mode + the Local LD tab.
- **The λGC auto-PC selector** and its flags (`--auto-pcs`, `--pc-strategy`,
  `--max-pcs`, `--pc-band-lo/-hi`, `--pc-parsimony-tol`). TRACE now uses a fixed
  `--n-pcs` default plus the report-only PC-selection diagnostics above (R1.4).

### Fixed
- **LD blocks could bridge uncorrelated SNP clusters.** The block merge fused
  overlapping intervals with no correlation check, and the adjacency split only
  tested consecutive pairs, so a long-range LD edge (or a chain of segment
  merges) could join distinct clusters into one low-coherence block. The opt-in
  `--ld-merge-mode correlation` gates both the within-segment split and the
  cross-boundary merge on mean r² (see Added).

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
