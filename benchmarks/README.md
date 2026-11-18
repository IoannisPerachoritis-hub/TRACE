# Benchmarks

The scripts and summary files behind the paper's benchmark tables and figures. Each entry below
names one table or figure: what it shows, the commands that rebuild it in order, the result
file, the input data and, for the steps that were timed, the measured run time. The times were
measured on 17 November 2026 with a development version before 1.2.0, on a desktop with 6 cores,
at 12 threads unless noted. The result files are committed, so you can compare a rebuild with them. They keep the exact bytes of the runs behind the paper,
which were written on Windows (CRLF line ends).

## Before you start

1. Install TRACE (see the main README) and run the commands below from the repository root.
2. Get the source data. The tomato panel is the Varitome VCF and its phenotype file (Pereira et
   al. 2021). The pepper panel is the G2P-SOL VCF and its averaged phenotypes (Tripodi et al.
   2021). The paper's data availability statement gives where to download them.
3. Make the QC data, one run per dataset (11 to 47 s each):

```bash
bash benchmarks/make_qc_data.sh tomato_locule_number varitome_filtered.vcf varitome_phenotypes.csv locule_number
bash benchmarks/make_qc_data.sh tomato_weight_g varitome_filtered.vcf varitome_phenotypes.csv weight_g
bash benchmarks/make_qc_data.sh pepper_FWe G2P-SOL_Pepper_CC_Geno_data_2023.vcf pepper_phenotypes_averaged.csv FWe
bash benchmarks/make_qc_data.sh pepper_BX G2P-SOL_Pepper_CC_Geno_data_2023.vcf pepper_phenotypes_averaged.csv BX
```

   Each run writes `benchmarks/qc_data/<run>/` with `QC_genotype_matrix.csv`, `QC_snp_map.csv` and
   `QC_phenotype.csv`. TRACE's QC keeps 164 x 43,749 for locule number, 165 x 43,734 for fruit
   weight and 350 x 4,183 for pepper (the 397 unplaced markers are dropped).
4. For the GAPIT3 and rMVP columns, install R and run `Rscript benchmarks/install_r_packages.R`. It
   installs GAPIT 3.5.0 at commit 7cb17f32, the build behind the tables. rMVP 1.4.6 made the rMVP
   columns.

The folder `benchmarks/qc_data` and the run outputs in `benchmarks/results` are not in the
repository. The R scripts read and write them under the folder they run from.

## The data root and the side analyses

The Python scripts take `--data-root DIR`: the folder that holds `benchmarks/qc_data` and
`benchmarks/results`. It is this repository by default. Three side analyses run in a folder of
their own, so their outputs do not replace the main ones:

- the pepper rows at the strict band rule's PC counts (FWe 3, BX 3): a folder with copies of
  `qc_data/pepper_FWe` and `qc_data/pepper_BX`, run with `--k-set band` and k = 3 in the R scripts.
- the second set of Table S10 (GAPIT FarmCPU at 2, 0, 0, 0 PCs): a folder with the four QC folders.
- the null calibration on the corrected template: a folder with copies of the QC data.

Copy each side analysis's result file into the repository under the name given below. The R
scripts work the same way: run them from the side analysis's folder.

## Table S4: TRACE, GAPIT3 and rMVP on the real data

lambda_GC, Bonferroni counts, Spearman rho and the overlap of the top 10 SNPs of the three tools'
MLM scans, at the published PC counts (FWe 2, the others 0).

1. `python benchmarks/rerun_trace_mlm.py` (24 to 25 s)
2. For each run, `Rscript benchmarks/run_gapit.R <run>` (12 to 36 s each) and
   `Rscript benchmarks/run_rmvp.R <run>` (1.9 to 6.2 s each)
3. `python benchmarks/compare_results.py`

Result: `benchmarks/table_s4_comparison.csv`. Each statistic has an `_unrounded` column beside it:
the table rounds once from it to three decimals.

The values the table gives in brackets come from rMVP with the PCs passed twice, as in the
published runs: run `Rscript benchmarks/run_rmvp.R pepper_FWe 2 pcs_twice` (and the same for the
other runs), then `python benchmarks/compare_results.py --rmvp-pcs-twice`. Result:
`benchmarks/table_s4_comparison_pcs_twice.csv`.

The side rows at the strict band rule's counts: in the side folder, steps 1 to 3 with
`--k-set band` and k = 3 for both pepper runs. Result, copied in:
`benchmarks/table_s4_comparison_band_k.csv` and `benchmarks/table_s4_comparison_band_k_pcs_twice.csv`.

## Table S8: the haplotype tests of the four tomato LD blocks

The four LD blocks near the lc locus on chromosome 2 and their kinship-aware haplotype tests.

1. `python benchmarks/table_s8.py` (10 s). Run it with OPENBLAS_NUM_THREADS=12 for the
   committed last digits.

Result: `benchmarks/table_s8.csv`. Input: `benchmarks/qc_data/tomato_locule_number`. The golden
files of the same run are `tests/golden/tomato_locule/`, and
`benchmarks/reconstruct_table_s8_params.py` checks the block settings.

## Table S10: TRACE-FarmCPU and GAPIT3 FarmCPU

lambda_GC, Bonferroni counts, Spearman rho and the overlap of the top 10 SNPs of the two FarmCPU
scans, and the tail recovery: the share of one tool's top 1% that falls in the other's top 5%.

1. `python benchmarks/rerun_trace_farmcpu.py` (22 s). TRACE-FarmCPU uses the PC counts of
   the published Table S10 run: FWe 4, BX 5, weight 10, locule 10.
2. For each run, `Rscript benchmarks/run_gapit_farmcpu.R <run>` (14 to 81 s each). Its default
   PC counts are the ones GAPIT FarmCPU ran with: FWe 3, BX 0, weight 0, locule 2.
3. `python benchmarks/run_farmcpu_benchmarks.py`
4. `python benchmarks/tail_recovery.py`

Results: `benchmarks/table_s10_farmcpu_comparison.csv` and `benchmarks/table_s10_tail_recovery.csv`.
In the tail recovery file, `variant` names the definition: `std` is GAPIT's standardized table
(its "(Kansas)" file, behind the rest of the table), `nyc` its "(NYC)" file. `gapit_to_trace` is the
share of GAPIT3's top 1% in TRACE's top 5%, `trace_to_gapit` the other way.

The PCs of step 1 come from an exact SVD, so a rebuild gives the same table. The paper's Table
S10 prints these exact SVD values. The first submission used a randomized SVD with no seed. That
draw, on the corrected data, is in `benchmarks/table_s10_farmcpu_comparison_randomized_pca.csv`
and `benchmarks/table_s10_tail_recovery_randomized_pca.csv`.

The second set: in its side folder, step 2 with k = 2, 0, 0, 0, then steps 1, 3 and 4. Results,
copied in: `benchmarks/table_s10_farmcpu_comparison_second_set.csv` and
`benchmarks/table_s10_tail_recovery_second_set.csv` (and `_second_set_randomized_pca` for the
first submission's draw).

## Table S11: LOCO against global kinship, and an unpruned GRM

How TRACE's LOCO and global runs, with the GRM of the LD pruned markers and with all markers,
agree with GAPIT3 and rMVP: Spearman rho on -log10 p and lambda_GC.

1. Steps 1 and 2 of Table S4
2. `python benchmarks/rerun_global_gwas.py` (29 s)
3. `python benchmarks/loco_sensitivity/compare_loco.py`
4. `python benchmarks/prune_ablation.py --write` (63 s)

Results: `benchmarks/loco_sensitivity/table_s11_loco_global.csv` (the LOCO and Global rows) and
`benchmarks/loco_sensitivity/table_s11_unpruned.csv` (the unpruned rows). With `--rmvp-pcs-twice`,
the `_pcs_twice` files. The side rows: `table_s11_loco_global_band_k.csv` and
`table_s11_unpruned_band_k.csv` (and `_pcs_twice`). The older `loco_comparison.csv` holds the
published values on the earlier QC data, which a test checks.

## Table S14: three added columns

The pass counts behind Table S14's rates, the precision with and without the subsampling filter,
and the share of true blocks kept, computed from the published table.

1. `python benchmarks/simulation/table_s14_columns.py` (0.6 s)

Result: `benchmarks/simulation/sim_summary/table_s14_columns.csv`.

## Figure S5: the p-value concordance of the three tools

1. Steps 1 and 2 of Table S4
2. `python benchmarks/plots/pvalue_scatter.py`

Result: `benchmarks/plots/figure_s5_pvalue_concordance.png` and `.pdf`. Its numbers are those of
`benchmarks/table_s4_comparison.csv`.

## Figure S6: the FarmCPU p-value concordance

Made by step 3 of Table S10. Result: `benchmarks/plots/figure_s6_farmcpu_concordance.png` and `.pdf`.
Its numbers are those of `benchmarks/table_s10_farmcpu_comparison.csv`.

## Figure S7: the three tools' Manhattan plots for tomato locule number

1. Steps 1 and 2 of Table S4 (tomato_locule_number)
2. `python benchmarks/plots/manhattan_comparison.py`

Result: `benchmarks/plots/figure_s7_manhattan_comparison.png` and `.pdf`.

## Figure S8: the subsampling of tomato locule number

1. Run the CLI with 100 resamples (9.5 minutes):

```bash
trace-gwas --vcf varitome_filtered.vcf --pheno varitome_phenotypes.csv --trait locule_number --model mlm --n-pcs 0 --sig-thresh meff --subsampling --boot-reps 100 --boot-frac 0.8 --boot-thresh 1e-4 --seed 42 --output subsampling_run
```

2. Take `tables/Subsampling_SNP_stability.csv` out of the results ZIP.
3. `python benchmarks/plots/subsampling_frequency.py --csv Subsampling_SNP_stability.csv`
4. `python benchmarks/plots/subsampling_summary.py --zip subsampling_run/<the results ZIP>`

Results: `benchmarks/plots/figure_s8_subsampling.png` and `.pdf`, and the figure's numbers in
`benchmarks/plots/figure_s8_subsampling_summary.json`.

## The haplotype tests' null calibration

How often the kinship-aware test and the earlier PCs-only test reject a true null on tomato and
rice, at h2 0, 0.3 and 0.6.

1. `python benchmarks/haplotype_null/null_calibration.py --dataset tomato --vcf varitome_filtered.vcf --pheno varitome_phenotypes.csv --out tomato_run` (14 minutes, at 8 threads)
2. `python benchmarks/haplotype_null/null_calibration.py --dataset rice --vcf rice_core_matched.vcf.gz --pheno rice_pheno.csv --out rice_run --band-k 0` (2.1 hours)

Results: `benchmarks/haplotype_null/tomato_rates.csv` and `rice_rates.csv` (the rejection rates
with Clopper-Pearson 95% intervals, and `interval_entirely_above_alpha` as the check), the blocks
tested (`tomato_blocks.csv`, `rice_blocks.csv`), the null phenotypes' seeds and causal SNPs
(`tomato_phenotypes.csv`, `rice_phenotypes.csv`) and the band rule's lambda_GC scans
(`tomato_band_rule.csv`, and `rice_band_rule.csv` from a first rice run whose result the run
above passes with `--band-k 0`). The rice inputs are the 3K Rice Genomes Core SNPs of the 2,261
accessions with grain length (SNP-Seek, Core v0.7) and the phenotypes of Wang et al. 2018
(Mendeley Data, doi 10.17632/sr8zzsrpcs.1).

## The PC counts of the strict band rule on the corrected data

The strict band rule behind the paper's PC counts (smallest k with 0.95 <= lambda_GC <= 1.05,
commit 03095b7), run on the corrected QC data. It gives FWe 3, BX 3, weight 0, locule 0.

1. `python benchmarks/band_rule_pc_counts.py` (2.5 minutes)

Result: `benchmarks/band_rule_pc_counts.csv`, the lambda_GC scan for k = 0 to 10 with each rule's
pick (`strict_pick`, and `edge_tolerant_pick` for the later variant with a wider band in
`_pc_band_select.py`).

## The null calibration on the corrected template, beside Table S6

Table S6 and Figure S1 stay as published (the earlier 165 x 43,974 template). These values repeat
the null calibration on the corrected template.

1. In the side folder: `python benchmarks/simulation/simulate_phenotypes.py --data-root DIR --mode template`,
   then `--mode null --n-perms 100 --seed 2026` (4.3 and 4.8 s)
2. `python benchmarks/simulation/run_simulation_platform.py --data-root DIR --mode null --n-pcs 2` (7.6 minutes)
3. `python benchmarks/simulation/evaluate_simulation.py --data-root DIR --mode null`

Results, copied in: `benchmarks/simulation/sim_summary/null_type1_error_corrected_template.csv` and
`null_lambda_gc_corrected_template.csv`.

## The other tables and figures

The simulation benchmark (Tables S2, S3, S5, S6, S7, S12 and S13, and Figures S1 to S4) and the QC
concordance with PLINK (Table S1) are as published. Their scripts are in `simulation/` and
`plink_qc/`, and their summary files in `simulation/sim_summary/` and `plink_qc/`. The runtime
table (Table S9) was measured with the CLI.

## Other scripts

- `reproduce.sh`: a short check of the whole pipeline on a simulated panel.
- `capture_golden.py`, `reconstruct_table_s8_params.py`: the golden files and the Table S8 block
  settings.
- `published_pc_counts.py`, `_pc_band_select.py`, `_options.py`: shared by the scripts above.
