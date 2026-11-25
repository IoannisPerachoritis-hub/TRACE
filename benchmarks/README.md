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

## The data root

The Python scripts take `--data-root DIR`: the folder that holds `benchmarks/qc_data` and
`benchmarks/results`. It is this repository by default.

## Table S4: TRACE, GAPIT3 and rMVP on the real data

lambda_GC, Bonferroni counts, Spearman rho and the overlap of the top 10 SNPs of the three tools'
MLM scans, at the published PC counts (FWe 2, the others 0).

1. `python benchmarks/rerun_trace_mlm.py` (24 to 25 s)
2. For each run, `Rscript benchmarks/run_gapit.R <run>` (12 to 36 s each) and
   `Rscript benchmarks/run_rmvp.R <run>` (1.9 to 6.2 s each)
3. `python benchmarks/compare_results.py`

Result: `benchmarks/table_s4_comparison.csv`. Each statistic has an `_unrounded` column beside it:
the table rounds once from it to three decimals.

## Table S8: the haplotype tests of the four tomato LD blocks

The four LD blocks near the lc locus on chromosome 2 and their kinship-aware haplotype tests.

1. `python benchmarks/table_s8.py` (10 s). Run it with OPENBLAS_NUM_THREADS=12 for the
   committed last digits.

Result: `benchmarks/table_s8.csv`. Input: `benchmarks/qc_data/tomato_locule_number`. The golden
files of the same run are `tests/golden/tomato_locule/`, and
`benchmarks/reconstruct_table_s8_params.py` checks the block settings.

## Table S9d: run time and memory on large panels

Run time is the wall time and memory the peak working set of the process tree, both measured
from outside the run, at 12 threads with nothing else running.

1. The simulated panel of the reviewer's size, command line (12 November 2026, a development
   version before 1.2.0; 2,284.6 s, 2,405.1 MiB). The panel has 996 samples and 1,070,795 SNPs on
   12 chromosomes of 65 Mb with mosaic LD (seed 20260923). It was made by the panel generator of
   the scaling work, which is not in the repository. The run used a test runner that kept the
   genotype table on disk before the CLI had `--geno-store`:

   ```bash
   OPENBLAS_NUM_THREADS=12 TRACE_STORE_FORBID_MATERIALIZE=1 TRACE_STORE_MAX_BLOCK_CELLS=20000000 \
       python store_runner.py --vcf p996_1070795_mosaic_gz.vcf.gz \
       --pheno p996_1070795_mosaic_gz_pheno.csv --trait SimTrait --output OUT
   ```

   With 1.2.0 the CLI keeps the table on disk itself: `trace-gwas --vcf p996_1070795_mosaic_gz.vcf.gz
   --pheno p996_1070795_mosaic_gz_pheno.csv --trait SimTrait --geno-store disk --output OUT`.
2. The same panel, the GWAS page's computations (17 November 2026, a development version before
   1.2.0; 2,193.3 s, 1,928.2 MiB): the page's steps one by one with the table on disk, MLM, MLMM
   and FarmCPU, without the post-GWAS steps. The measurement script of that work, which is not in
   the repository, was run as
   `python reviewer_size_run.py --vcf p996_1070795_mosaic_gz.vcf.gz --pheno p996_1070795_mosaic_gz_pheno.csv --trait SimTrait --work-dir WORK --models mlm mlmm farmcpu --out run.json`.
3. Rice grain length, MLM (18 November 2026; 4,825.4 s, 1,671.3 MiB), on the inputs of the rice
   section below:

   ```bash
   TRACE_STAGE_LOG=stagelog.csv trace-gwas --vcf rice_core_filtered_windows.vcf.gz \
       --pheno rice_pheno.csv --trait GRLT --species custom --gene-model locus.gff \
       --gene-desc rice_gene_desc.tsv --geno-store auto --geno-store-dir STORE --output p1_grlt
   ```

4. Rice grain length, MLM and FarmCPU (18 November 2026; 5,590.7 s, 1,740.6 MiB): the same
   command with `--model mlm farmcpu` and its own output folder.

Rows 3 and 4 ran the CLI through a small wrapper that checked which copy of TRACE was imported.

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
S10 prints these exact SVD values.

## Table S11: LOCO against global kinship, and an unpruned GRM

How TRACE's LOCO and global runs, with the GRM of the LD pruned markers and with all markers,
agree with GAPIT3 and rMVP: Spearman rho on -log10 p and lambda_GC.

1. Steps 1 and 2 of Table S4
2. `python benchmarks/rerun_global_gwas.py` (29 s)
3. `python benchmarks/loco_sensitivity/compare_loco.py`
4. `python benchmarks/prune_ablation.py --write` (63 s)

Results: `benchmarks/loco_sensitivity/table_s11_loco_global.csv` (the LOCO and Global rows) and
`benchmarks/loco_sensitivity/table_s11_unpruned.csv` (the unpruned rows). The older
`loco_comparison.csv` holds the published values on the earlier QC data, which a test checks.

## Table S14 and Figure S9: the subsampling filter on the simulated LD blocks

TRACE's LD blocks in four simulation scenarios (h2 0.5 with 5 QTNs; h2 0.8 with 5, 15 and 50
QTNs; 100 replicates each), each block's subsampling discovery frequency (50 resamples of 80%
without replacement), and whether a causal QTN lies within 500 kb of the block.

1. `python benchmarks/simulation/evaluate_fdr_reduction.py --scenario all --n-reps 100 --n-sub 50 --n-pcs 2 --n-jobs 2`
   (about 7.8 hours, on 8 and 9 September 2026, with a development version before 1.2.0)
2. `python benchmarks/simulation/evaluate_fdr_reduction.py --plot-only`
3. `python benchmarks/simulation/table_s14_columns.py` (0.6 s)

Results: step 1 writes one block file per scenario, `benchmarks/simulation/sim_summary/fdr_reduction_blocks_<scenario>.csv`;
the four files of the paper are committed. Table S14's TP and FP block counts, median frequencies,
pass rates and Mann-Whitney p are those of these files, pooled over the replicates. Step 2 draws
Figure S9 from them, as `figure_s7_fdr_reduction_combined.png` and `.pdf` (the name keeps the
earlier figure number). Step 3 writes `benchmarks/simulation/sim_summary/table_s14_columns.csv`:
the pass counts behind the rates, the precision with and without the filter, and the share of
true blocks kept, computed from the published table. Input: the simulation data in
`benchmarks/simulation/sim_data/` (not in the repository).

## Table S15: candidate regions around true signals, TRACE and PLINK

For each true signal (a causal SNP with a Bonferroni-significant SNP within 500 kb), the region
that each method's intervals highlight: TRACE's candidate intervals at 1.2.0's default window
(4 x the LD decay) and at the earlier one (2 x), PLINK 1.9's haplotype blocks (200 and 600 kb)
and PLINK's clumps (its defaults, and r2 0.6 with 300 kb), in the 900 simulated datasets and at
the rice *GS3* locus. Six scripts in `benchmarks/simulation/`, run in this order; each writes a work
folder (P2, P2B, P2C, P2D, P2E) that the later ones read.

```bash
S=benchmarks/simulation
# PLINK's haplotype blocks
python $S/ld_blocks_vs_plink.py vcf --vcf varitome_filtered.vcf --out P2
python $S/ld_blocks_vs_plink.py plink --plink plink --out P2
python $S/ld_blocks_vs_plink.py compare --jobs 4 --out P2
python $S/ld_blocks_vs_plink.py loci --plink plink --out P2 --vcf varitome_filtered.vcf \
    --tomato-zip TOMATO.zip --rice-zip RICE.zip --rice-bfile vcf2_matched_ref --rice-gene-model locus.gff
# PLINK's clumps
python $S/ld_blocks_vs_plink_clump.py prepare --vcf varitome_filtered.vcf --plink plink --out P2B
python $S/ld_blocks_vs_plink_clump.py sim --plink plink --out P2B --jobs 4
python $S/ld_blocks_vs_plink_clump.py loci --plink plink --out P2B --vcf varitome_filtered.vcf \
    --tomato-zip TOMATO.zip --rice-zip RICE.zip --rice-bfile vcf2_matched_ref --rice-gene-model locus.gff
# TRACE at the earlier default window (2 x the LD decay)
python $S/ld_blocks_vs_plink_trace_default.py sim --out P2C --jobs 4
python $S/ld_blocks_vs_plink_trace_default.py loci --out P2C --name tomato --vcf varitome_filtered.vcf \
    --pheno varitome_phenotypes.csv --trait locule_number --zip TOMATO.zip
python $S/ld_blocks_vs_plink_trace_default.py loci --out P2C --name rice --vcf rice_core_filtered_windows.vcf.gz \
    --pheno rice_pheno.csv --trait GRLT --zip RICE.zip --geno-store-dir P2C/store
# the per signal scoring
python $S/ld_blocks_vs_plink_per_signal.py --out P2D --p2 P2 --p2b P2B --p2c P2C \
    --tomato-zip TOMATO.zip --rice-zip RICE.zip --rice-gene-model locus.gff
# TRACE at 1.2.0's default window (4 x the LD decay), scored by the same script
python $S/ld_blocks_vs_plink_trace_wide.py sim --p2c P2C --out P2E --jobs 4
python $S/ld_blocks_vs_plink_trace_wide.py loci --p2c P2C --out P2E --name tomato --vcf varitome_filtered.vcf \
    --pheno varitome_phenotypes.csv --trait locule_number --zip TOMATO.zip
python $S/ld_blocks_vs_plink_trace_wide.py loci --p2c P2C --out P2E --name rice --vcf rice_core_filtered_windows.vcf.gz \
    --pheno rice_pheno.csv --trait GRLT --zip RICE.zip --geno-store-dir P2E/store
python $S/ld_blocks_vs_plink_per_signal.py --out P2E/score_p2d --p2 P2 --p2b P2B --p2c P2C \
    --tomato-zip TOMATO.zip --rice-zip RICE.zip --rice-gene-model locus.gff
python $S/ld_blocks_vs_plink_per_signal.py --out P2E/score_wide --p2 P2 --p2b P2B --p2c P2E \
    --tomato-zip TOMATO.zip --rice-zip RICE.zip --rice-gene-model locus.gff
python $S/ld_blocks_vs_plink_trace_wide.py tables --out P2E/tables --wide P2E/score_wide \
    --repro P2E/score_p2d --p2d-regions P2D/regions.csv
```

Results, in `benchmarks/simulation/sim_summary/`: the scoring step writes
`ld_blocks_vs_plink_per_signal.csv` (per scenario), `_5qtn.csv`, `_loci.csv` and `_main.csv`, with
TRACE at the earlier window. The tables step adds TRACE at 1.2.0's window and writes the four
`ld_blocks_vs_plink_per_signal_wide*.csv` files. Table S15a prints `_wide_main.csv` (all 9
scenarios) and `_wide_5qtn.csv` (the three 5-QTN scenarios), and Table S15b the rice_GS3 rows of
`_wide_loci.csv`. In these files `trace_wide_bonferroni` is TRACE 1.2.0's default window and
`trace_default_bonferroni` the earlier one.

Inputs: the simulation data in `benchmarks/simulation/sim_data/` (not in the repository); PLINK
1.9; the Varitome VCF and phenotypes; `TOMATO.zip`, the results ZIP of
`trace-gwas --vcf varitome_filtered.vcf --pheno varitome_phenotypes.csv --trait locule_number`;
`RICE.zip`, the results ZIP of the rice run in row 3 of Table S9d; the rice VCF, its PLINK files
`vcf2_matched_ref` and RAP-DB's `locus.gff`. The runs behind the committed files took, with a
development version before 1.2.0 (18 to 23 November 2026, 4 processes for the simulations):
the block comparison 106 min, the clumps 118 min, TRACE on the simulations 53 min (2 x) and
66 min (4 x), TRACE on rice 72 and 84 min, and each other step under 4 minutes.

## Table S16: agreement between MLM and TRACE-FarmCPU in the simulations

How often TRACE-LOCO's MLM (Table S2) and TRACE-FarmCPU (Table S12) agree in the 9 x 100
simulated datasets, and the precision of each group. Counting only: no GWAS is run.

1. `python benchmarks/simulation/consensus_precision.py --out DIR` (100.6 s, on 24 November 2026)

The rules: a SNP is significant at p < 0.05 / 43,974 in its model. Each model's significant SNPs
are grouped into signals: the most significant SNP not yet used leads a signal, and every unused
significant SNP on the same chromosome within 500 kb of it joins. A signal is true when a causal
QTN lies within 500 kb of its lead SNP. An MLM signal and a TRACE-FarmCPU signal agree when their
lead SNPs lie within 500 kb. The second analysis uses TRACE's own consensus rule: the same SNP
significant in both models.

Results: `benchmarks/simulation/sim_summary/consensus_signals.csv` (the signals, Table S16) and
`consensus_snps.csv` (the exact SNPs, not printed), with one row per scenario and one for all nine.
DIR also gets the counts per replicate and `consensus_tables.md`. Before it writes them, the script
checks MLM's power, FDR and number of significant SNPs in every replicate against
`sim_summary/bonferroni_results.csv`, and their summary against the Table S2 rows of
`sim_summary/power_fdr_summary_ci.csv`. Input: the simulation data in
`benchmarks/simulation/sim_data/` (not in the repository): for each replicate
`platform_loco/results.csv`, `farmcpu_2x2_E_F_nogate2_ols/results.csv` and `truth.json`, and
`snp_map.csv`.

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
tested (`tomato_blocks.csv`, `rice_blocks.csv`) and the band rule's lambda_GC scans
(`tomato_band_rule.csv`, and `rice_band_rule.csv` from a first rice run whose result the run
above passes with `--band-k 0`). The rice inputs are the 3K Rice Genomes Core SNPs of the 2,261
accessions with grain length (SNP-Seek, Core v0.7) and the phenotypes of Wang et al. 2018
(Mendeley Data, doi 10.17632/sr8zzsrpcs.1).

## Rice grain length (3K-RG)

The rice runs (rows 3 and 4 of Table S9d, Table S15b, Figure S10 and the null calibration) read
files made in November 2026 from these downloads, each gunzipped first:

- the 3K-RG Core SNP set v0.7 as PLINK files:
  https://s3.amazonaws.com/3kricegenome/snpseek-dl/3krg-base-filt-core-v0.7/core_v0.7.bed.gz (and
  `.bim.gz`, `.fam.gz`);
- the 3K-RG Filtered SNP set v0.7:
  https://s3.amazonaws.com/3kricegenome/snpseek-dl/3krg-base-filt-v0.7/base_filtered_v0.7.bed.gz (and
  `.bim.gz`, `.fam.gz`);
- the IRGSP-1.0 genome of Ensembl Plants release 63:
  https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-63/fasta/oryza_sativa/dna/Oryza_sativa.IRGSP-1.0.dna.toplevel.fa.gz;
- RAP-DB's representative gene models and annotation of 5 February 2026:
  https://rapdb.dna.naro.go.jp/download/archive/irgsp1/IRGSP-1.0_representative_2026-02-05.tar.gz
  (`locus.gff`) and
  https://rapdb.dna.naro.go.jp/download/archive/irgsp1/IRGSP-1.0_representative_annotation_2026-02-05.tsv.gz;
- grain length: Mendeley Data, doi 10.17632/sr8zzsrpcs.1 (`real_quantitative_traits.csv`), with the
  accession table of Wang et al. 2018 (Supplementary Data, `41586_2018_63_MOESM3_ESM.xlsx`).

Small scripts of that work, not in the repository, made three of the files:

- `matched_samples.txt` and `rice_pheno.csv`: the 2,261 accessions with grain length, matched to
  the 3K-RG sample IDs, with their GRLT and GRWD values;
- `vcf2_extract_chrpos.txt`: the 442,308 SNPs of the second VCF as CHROM:POS IDs, the Core SNPs
  outside three windows and the Filtered SNPs inside them; the windows are the gene bounds of
  *GS3*, *GW5* and *GL7* in `locus.gff`, plus and minus 500 kb;
- `rice_gene_desc.tsv`: the gene descriptions from the RAP-DB annotation (the first transcript of
  each gene; the RAP-DB symbol before the CGSNL one).

The plink2 commands as they were run, with the paths cut to file names:

```bash
plink2 --bfile core_v0.7 --keep matched_samples.txt --chr-set 12 --fa Oryza_sativa.IRGSP-1.0.dna.toplevel.fa \
    --ref-from-fa --set-all-var-ids @:# --export vcf bgz id-paste=iid --threads 6 --memory 4000 \
    --out rice_core_matched
plink2 --bfile base_filtered_v0.7 --keep matched_samples.txt --chr-set 12 --set-all-var-ids @:# \
    --extract vcf2_extract_chrpos.txt --fa Oryza_sativa.IRGSP-1.0.dna.toplevel.fa --ref-from-fa \
    --export vcf bgz id-paste=iid --threads 6 --memory 4000 --out rice_core_filtered_windows
plink2 --bfile base_filtered_v0.7 --keep matched_samples.txt --chr-set 12 --set-all-var-ids @:# \
    --extract vcf2_extract_chrpos.txt --fa Oryza_sativa.IRGSP-1.0.dna.toplevel.fa --ref-from-fa \
    --make-bed --threads 6 --memory 4000 --out vcf2_matched_ref
```

`rice_core_matched.vcf.gz` (404,388 SNPs) is the null calibration's input.
`rice_core_filtered_windows.vcf.gz` (442,308 SNPs) is read by the GWAS runs of Table S9d, Figure
S10 and the rice steps of Table S15. `vcf2_matched_ref` holds the same SNPs as PLINK files, for
PLINK at the rice locus in Table S15.

Figure S10, from the two rice runs of Table S9d (rows 3 and 4; 20 s):

```bash
python benchmarks/plots/rice_gs3_figure.py --mlm-zip P1.zip --farmcpu-zip FARMCPU.zip \
    --vcf rice_core_filtered_windows.vcf.gz --pheno rice_pheno.csv --gff locus.gff --out-dir DIR
```

Result: `rice_gs3_figure.png` and `.pdf` in DIR.

## PLINK

- Table S15 used PLINK v1.9.1 64-bit (11 Nov 2026), the "Windows 64-bit" build of
  https://www.cog-genomics.org/plink/1.9/ (`plink_win64_20261111.zip`, sha256
  03c83d4a6bb457f2b4f0bf02d2fde821faba06aaf38cff10577d9d5c23378f14). The page replaces its builds;
  the sha256 names this one.
- Table S1 used PLINK v2.00a5.13 64-bit (18 Aug 2024), through `plink_qc/compare_qc.py`.
- The rice VCFs were made with PLINK v2.0.0-b.1.4 AVX2 (7 Nov 2026), from
  https://www.cog-genomics.org/plink/2.0/.

## The other tables and figures

The simulation benchmark (Tables S2, S3, S5, S6, S7, S12 and S13, and Figures S1 to S4) and the QC
concordance with PLINK (Table S1) are as published. Their scripts are in `simulation/` and
`plink_qc/`, and their summary files in `simulation/sim_summary/` and `plink_qc/`.
`plink_qc/qc_concordance.csv` predates the corrected data of the printed Table S1. Tables S9a to
S9c were measured with the CLI.

## Other scripts

- `reproduce.sh`: a short check of the whole pipeline on a simulated panel.
- `capture_golden.py`, `reconstruct_table_s8_params.py`: the golden files and the Table S8 block
  settings.
- `published_pc_counts.py`, `_pc_band_select.py`, `_options.py`: shared by the scripts above.
