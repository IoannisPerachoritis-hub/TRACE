# Development

## Repository layout

```
TRACE/
├── app.py, cli.py                    # the GUI and the CLI
├── gwas/                             # QC, kinship, models, LD, haplotypes, subsampling, reports
├── pages/                            # the GUI's pages and the Post-GWAS tabs
├── annotation.py, data/              # gene annotation and the bundled tomato gene models
├── utils/                            # plot theme, species files
├── tests/                            # the test suite
├── examples/                         # the bundled example
├── benchmarks/                       # the scripts behind the paper's tables (benchmarks/README.md)
├── docs/                             # this documentation and the screenshots
├── launchers/                        # launch scripts for users who do not use a terminal
└── Dockerfile, pyproject.toml, requirements.txt, constraints.txt
```

## Tests

CI runs the suite on every push and pull request to `main`, on Python 3.11 and 3.12, with a
lowest line coverage below which the build fails.

```bash
python -m pytest tests/ -v              # the whole suite
python -m pytest tests/test_ld.py -v    # one file
```

Four markers sort the slower tests. `slow` marks the full CLI runs that show the genotype table in
memory and on disk give the same results. `slow_gui` marks the GUI tests that run real scans.
`golden` marks the tests that compare a run on the real tomato data with stored results, and
`manuscript` a test of the paper's block table. Both read the tomato QC data in
`benchmarks/qc_data/`, which [the benchmarks README](../benchmarks/README.md) shows how to make.
Leave the slow ones out of a quick run with `-m "not slow and not slow_gui and not golden"`.

The tests cover edge cases such as columns of missing values only, monomorphic SNPs, a single
haplotype group, too few samples, perfect LD (r² = 1) and SNPs between genes.

<!-- BEGIN GENERATED: test-table (scripts/gen_test_table.py -- do not edit by hand) -->

| Module | Tests | What it exercises |
|--------|-------|-------------------|
| Genotype I/O & parsing | 79 | VCF/dosage parsing, INFO scores, upload edge cases |
| Quality control | 55 | MAF/missingness/MAC/heterozygosity filters, per-group QC, QC report |
| Imputation | 18 | mean and LD-kNNi imputation and its cache |
| Phenotype QC & transforms | 59 | normality diagnostics, transformations, embedded QC panel, applied transformations |
| Kinship (GRM / LOCO) | 15 | VanRaden GRM and leave-one-chromosome-out kernels |
| Association models | 40 | MLM / MLMM / FarmCPU scans and FarmCPU pseudo-QTN selection |
| Significance & multiple testing | 82 | M_eff / Bonferroni / FDR / custom-threshold rules and every model's significance columns |
| Covariates & PC diagnostics | 32 | user covariates, covariates from the phenotype file and the PC eigenvalue-spectrum diagnostics |
| LD block detection | 56 | peak-centric LD block detection and merging |
| Haplotype testing | 135 | block haplotype effects, the kinship-aware test, effect sizes, compact letter display, sample-to-haplotype assignments |
| LD triage | 28 | coherence/haplotype triage layers, router and eta-squared comparability |
| Isolated-SNP rescue & significant-SNP table | 40 | unblocked-SNP intervals and the significant-SNP table |
| Regional & per-SNP visualisation | 112 | regional association plots, LD-block heatmaps, per-SNP boxplots and their haplotype colours, plotting stats, sample views |
| Gene annotation | 76 | LD-block gene annotation, gene-model summaries and resolution |
| Subsampling stability | 25 | bootstrap subsampling and stability metrics |
| HTML report | 24 | run-report assembly and section rendering |
| Command-line interface | 60 | CLI parsing, end-to-end and multi-trait runs, doc-to-parser flag parity |
| Web UI (Streamlit) | 390 | app-test coverage of the GWAS, Post-GWAS, help and landing pages |
| Large datasets (genotype table on disk) | 541 | the table file, its ingest and reads, the memory plan, the models and the Post-GWAS steps on disk, memory and disk giving the same results |
| Pipeline integration | 44 | end-to-end GWAS pipeline and stage wiring |
| Golden regression & pinned defaults | 105 | byte-stable golden fixtures, the golden lock, and pinned signatures/defaults |
| Calibration & reproducibility | 23 | null-phenotype calibration, LOCO reproducibility and the thread counts |
| Utilities | 20 | shared helpers and the data behind the GUI's figures |

_Total: 2059 tests. Line coverage: 87% (gwas + utils)._

<!-- END GENERATED: test-table -->

See [CONTRIBUTING.md](../CONTRIBUTING.md) for how to set up a development install and submit a
change.
