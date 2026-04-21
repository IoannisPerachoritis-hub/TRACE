# Changelog

All notable changes to TRACE are documented in this file.

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
