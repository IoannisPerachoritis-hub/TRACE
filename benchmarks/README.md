# Benchmarks

Scripts and data used for the TRACE manuscript: simulation studies,
cross-tool concordance, and the downstream case-study rerun.

## Quick reproduction

```bash
bash benchmarks/reproduce.sh
```

Runs the full pipeline on a simulated demo panel, then prints expected
vs observed λGC, M_eff count, and significant-SNP count with PASS/FAIL.
Takes under a minute. Use this as the first sanity check after cloning.

## Regenerating the QC matrices

The post-QC genotype matrices used by the R comparison scripts total
~106 MB and are not tracked in git. Regenerate them from the source
VCFs with:

```bash
bash benchmarks/make_qc_data.sh <dataset> <vcf> <pheno> <trait>
```

Example:

```bash
bash benchmarks/make_qc_data.sh tomato_locule_number \
  data/tomato/varitome_filtered.vcf \
  data/tomato/varitome_phenotypes.csv \
  locule_number
```

Output lands in `benchmarks/qc_data/<dataset>/` as three CSVs:
`QC_genotype_matrix.csv`, `QC_snp_map.csv`, and `QC_phenotype.csv`.
Source VCFs are from Pereira et al. (2021, tomato Varitome) and
Tripodi et al. (2021, pepper G2P-SOL).

## Cross-tool concordance (R)

- `Rscript install_r_packages.R` — one-off dependency install.
- `Rscript run_gapit.R`, `Rscript run_gapit_farmcpu.R`, `Rscript run_rmvp.R`
  run GAPIT3 and rMVP on the QC'd matrices. Requires `qc_data/` first.
- `python compare_results.py` computes Spearman ρ, top-10 overlap, and
  significant-SNP counts against TRACE. Output CSVs land in `results/`.

`run_all_benchmarks.R` is a convenience wrapper that chains the R
scripts across all four datasets.

## Simulation benchmarks

The `simulation/` subdirectory holds the 9-cell benchmark used for
Figure 1B and Table S2 (3 heritabilities × 3 QTN counts × 100 reps).
Entry point: `python simulation/run_all_simulations.py`. Each tool has
its own runner (`run_simulation_platform.py`, `run_simulation_gapit.R`,
`run_simulation_rmvp.R`). Summary CSVs land in `simulation/sim_summary/`.

## PLINK concordance

`plink_qc/compare_qc.py` compares TRACE's post-QC SNP set against
PLINK's (`--snps-only just-acgt --max-alleles 2`). Output lands in
`plink_qc/qc_concordance.csv` and `plink_qc/results/`.

## Directory layout

```
benchmarks/
├── reproduce.sh              # End-to-end reviewer check
├── make_qc_data.sh           # Regenerate qc_data/ from source VCFs
├── run_*.R, run_*.py         # Per-tool benchmark runners
├── compare_results.py        # Cross-tool concordance
├── qc_data/                  # Post-QC matrices (not tracked; regenerate)
├── results/                  # Concordance CSVs
├── plots/                    # Figure outputs
├── simulation/               # 9-cell simulation benchmark
└── plink_qc/                 # PLINK concordance scripts
```
