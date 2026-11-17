"""Help & Reference: TRACE.

Quick-start guide, output column glossary, and citation info.
"""

import streamlit as st

st.title("Help & Reference")

# ============================================================
# 0. Quick Start
# ============================================================
st.header("Quick Start")
st.markdown("""
The full step-by-step walkthrough (VCF/phenotype/gene-model formats with examples,
QC presets, and downloading results) lives in the project
[README](https://github.com/IoannisPerachoritis-hub/TRACE/blob/main/README.md). It is kept
there as the single source so it can't drift out of sync with the code. For command-line
use, run `trace-gwas --help`; it prints the full parameter list on every install route.
Each input file's format:

- **VCF** (genotypes): [README, Genotypes](https://github.com/IoannisPerachoritis-hub/TRACE/blob/main/README.md#genotypes)
- **Phenotype file**: [README, Phenotypes](https://github.com/IoannisPerachoritis-hub/TRACE/blob/main/README.md#phenotypes)
- **Covariate file** (optional): [docs/outputs.md, Related CLI options](https://github.com/IoannisPerachoritis-hub/TRACE/blob/main/docs/outputs.md#related-cli-options-new-since-v101)
- **Gene file** (a table, or a GFF3, plain or gzip) and **descriptions file** (optional):
  [docs/gene_model_upload.md](https://github.com/IoannisPerachoritis-hub/TRACE/blob/main/docs/gene_model_upload.md)

**In 30 seconds:** on **GWAS Analysis**, upload a VCF + a phenotype CSV (sample IDs in
the first column, matching the VCF), pick a trait, and click **Run GWAS** or **One-Click
Full Analysis**. Then open **Post-GWAS Analysis** for LD blocks and haplotype effects;
every page offers a ZIP download.

**Large datasets:**
- **Big VCFs.** For a VCF above the 2 GB upload limit, or to skip the upload, type the file's
  location on this computer in the box under the upload box. TRACE then reads the file where
  it is.
- **Where the genotype table is kept.** Before reading, the page checks the memory a run needs.
  Under **Large datasets**, Automatic keeps the table on disk when it would not fit; In memory and
  On disk force the choice. The results are the same either way. LD-kNNi imputation needs the
  table in memory.
- **Clear session data** frees this server's genotype tables, the uploads it saved and every
  cached result, on both pages, and empties the GWAS page's file boxes and trait selection.
- **The interactive Manhattan.** Above 500,000 SNPs it draws every SNP with p ≤ 0.01 and a
  sample of the others. The static Manhattan and the results table keep every SNP.

Details: [docs/large_datasets.md](https://github.com/IoannisPerachoritis-hub/TRACE/blob/main/docs/large_datasets.md#in-the-gui).
""")

# ============================================================
# 1. Output Format
# ============================================================
st.header("Output Format")
st.markdown("""
The One-Click Pipeline and the single-trait download both produce a **ZIP**. With several
traits, One-Click's download is one **combined ZIP** with a folder per trait (plus
`failed_traits.txt` when a trait failed); each trait runs on its own phenotyped samples,
through the same per-trait pipeline as `trace-gwas --trait A B C`.

Every file of the CLI and One-Click ZIP (`<trait>` is the trait, `<model>` is `MLM`,
`MLMM` or `FarmCPU`; a table with no rows is not written). The single-trait download's ZIP
holds that page's own tables and figures, interactive HTML included.

| File | What it holds | Written when |
|---|---|---|
| `tables/GWAS_<trait>.csv` | MLM's scan: every tested SNP's p-value, effect and significance columns | always |
| `tables/GWAS_<model>_<trait>.csv` | MLMM's or FarmCPU's scan, with the same significance columns | that model runs |
| `tables/Significant_SNPs_<model>.csv` | every SNP significant in that model, in an LD block or unblocked | the model has a significant SNP |
| `tables/Unblocked_SNPs_<model>.csv` | the significant SNPs outside every LD block | the model has one |
| `tables/Isolated_SNP_intervals_<model>.csv` | the candidate interval of each unblocked significant SNP, or run of them | the model has one |
| `tables/Isolated_SNP_candidate_genes_<model>.csv` | the genes inside those intervals | a gene lies in one (gene annotation on) |
| `tables/LD_blocks_annotated_<model>.csv` | the model's LD blocks: bounds, candidate interval, genes and haplotype test | the model has a block |
| `tables/LD_triage_<model>.csv` | each block's LD-quality triage and the view it recommends | the model has a block, unless `--no-triage` |
| `tables/Haplotype_assignments_<trait>.csv` | which haplotype each sample carries in each of MLM's tested blocks | MLM has a tested block |
| `tables/Haplotype_assignments_<trait>_<model>.csv` | the same for MLMM's or FarmCPU's tested blocks | that model has a tested block |
| `tables/SNP_view_index_<model>.csv` | for each significant SNP of the primary model (MLM), whether the report plots it, and its plotting representative | MLM has a significant SNP |
| `tables/CrossModel_Consensus.csv` | per significant SNP, the models that call it | two or more models run |
| `tables/PC_diagnostics_spectrum.csv` | the genotype PCA's eigenvalue spectrum (rank, eigenvalue, share of the trace) | always |
| `tables/QC_report_summary.csv` | the QC report: imputation, heterozygosity, relatedness and F_IS statistics | always |
| `tables/QC_sample_heterozygosity.csv` | each sample's heterozygosity, its z-score and outlier flag | always |
| `tables/QC_genotype_matrix.csv` | the QC'd, imputed genotypes (samples × SNPs) | `--export-qc` |
| `tables/QC_genotype_raw.csv` | the same genotypes before imputation, missing calls kept | `--export-qc` |
| `tables/QC_snp_map.csv` | the QC'd SNPs: ID, chromosome, position, REF and ALT | `--export-qc` |
| `tables/QC_phenotype.csv` | the analysed trait values | `--export-qc` |
| `tables/Subsampling_SNP_stability.csv` | each SNP's discovery frequency and p-values across the subsamples | `--subsampling` |
| `tables/Subsampling_block_stability.csv` | each MLM LD block's discovery frequency across the subsamples | `--subsampling`, with blocks |
| `tables/Subsampling_rep_metadata.csv` | one row per subsample: its size, status, λGC and hits | `--subsampling` |
| `figures/Manhattan_<model>.png` | each model's Manhattan plot | unless `--no-plots` |
| `figures/QQ_<model>.png` | each model's QQ plot | unless `--no-plots` |
| `figures/LD_heatmap_Chr<c>_<start>_<end>.png` | the r² heatmap of each LD block whose lead is significant in its model | unless `--no-plots`, or One-Click's "Generate LD block heatmap PNGs" is off |
| `figures/PCA_scree.png` | the PCA scree plot | unless `--no-plots` |
| `figures/PCA_scatter.png` | PC1 against PC2 | two or more PCs, unless `--no-plots` |
| `figures/QC_report_distributions.png` | the QC report's distributions | unless `--no-plots` |
| `figures/Subsampling_discovery_freq_<trait>.png` | the subsampling discovery frequencies | `--subsampling`, unless `--no-plots` |
| `report.html` | the self-contained HTML report | unless `--no-report` |
| `run_metadata.json` | every setting of the run, for reproducibility; also the run's environment and where the genotype table was kept (the keys below) | always |
| `<trait>/` | in a combined ZIP, one folder per trait holding that trait's files | several traits |
| `failed_traits.txt` | in a combined ZIP, each trait's status and the error of any that failed | a trait failed |
""")
st.markdown("""
**`run_metadata.json`: the environment and the genotype table.** Besides the settings:

- `BLAS library`: the linear-algebra libraries and their versions (numpy's and scipy's)
- `BLAS threads`: their thread count
- `CPU cores (logical)`: the logical cores
- `CPU cores (physical)`: the physical cores
- `Subsampling workers (--boot-jobs)`: the subsampling workers (N/A without subsampling)
- `Subsampling BLAS threads per worker`: the threads each worker's library uses
- `Genotype table`: where the table was kept, memory or disk
- `Genotype table choice`: what was asked: `auto`, `--geno-store memory` or `--geno-store disk` (the GUI's Automatic, In memory or On disk)
- `Genotype table reason`: why it was kept there
- `Memory estimate, in-memory run (GB)`: TRACE's estimate for an in-memory run of this analysis
- `Genotype calls`: samples × SNPs as read
""")

# ============================================================
# 2. CSV Column Glossary
# ============================================================
st.header("CSV Column Glossary")

st.subheader("GWAS Results (per-model)")
st.markdown("""
| Column | Meaning |
|--------|---------|
| **SNP** | SNP identifier (typically chromosome_position format) |
| **Chr** | Chromosome name |
| **Pos** | Physical position (bp) on the chromosome |
| **PValue** | Association p-value from the mixed linear model |
| **PValue (MLMM)** | MLMM's final scan, with its cofactors as covariates; each cofactor's own p-value comes from a test with the other cofactors as covariates |
| **-log10p** | Negative log10 of the p-value (higher = more significant) |
| **Beta_MLM** | Effect size estimate from the mixed linear model (kinship and any PC covariates), per copy of the ALT allele (FaST-LMM's estimate on the standardised SNP, divided by the SNP's dosage standard deviation). Positive means the ALT allele increases the trait value. |
| **SE_MLM** | Standard error of the MLM effect estimate, per copy of the ALT allele |
| **Beta_OLS** | Effect size per copy of the ALT allele from ordinary least squares with the same PCs (ignoring kinship) |
| **SE_OLS** | Standard error of the OLS effect estimate |
| **FDR** | False discovery rate-adjusted p-value (Benjamini-Hochberg), computed within this model's own p-values. Values ≤ 0.05 are significant after multiple testing correction. |
| **Significant_FDR** | True if the SNP passes Benjamini-Hochberg at 0.05 (FDR ≤ 0.05) |
| **Significant_Bonf** | True if p-value < Bonferroni threshold (0.05 / number of tested SNPs) |
| **Significant_Meff** | True if p-value < M_eff threshold (always in the CLI and One-Click; in the single-trait run when M_eff is selected) |
| **Significant_Custom** | True if p-value < the custom p-value threshold, when one is set |
| **Model** | Which GWAS model produced this result (MLM, MLMM, FarmCPU) |
| **Nullh2** | Null-model heritability estimate (internal parameter) |
| **Mixing** | Mixing parameter from FaST-LMM (internal) |
| **ImputationRate** | Fraction of non-missing genotypes at this SNP |

The MLM, MLMM and FarmCPU tables carry the same significance columns, judged against MLM's
thresholds (Bonferroni, M_eff, custom); FDR is computed within each model's own p-values. A SNP
without a p-value is never significant.
""")

st.subheader("LD Block Annotations")
st.markdown("""
| Column | Meaning |
|--------|---------|
| **Chr** | Chromosome |
| **Start (bp)** / **End (bp)** | The LD block's first and last SNP: its member span, which the LD and haplotype steps use |
| **Interval_start** / **Interval_end** | The block's candidate interval: from the nearest marker before its first SNP to the nearest after its last, since the haplotype's true boundary lies in between. Genes are searched here |
| **N_SNPs** | Number of SNPs in the block |
| **lead_snp** | The block's most significant SEEDING SNP (smallest GWAS p-value among the significant SNPs that formed the block). It may not be a member of the block's LD-connected SNPs. **lead_snp_pvalue** gives its p-value. |
| **overlapping_genes** | Genes that overlap the candidate interval; nothing outside it is listed |
| **annotation_status** | `overlapping`, or `no_genes_nearby` when no gene lies inside the interval |
""")

st.subheader("Haplotype GWAS Results")
st.markdown("""
Each LD block's haplotype groups are tested by a mixed model with the GWAS scan's own kinship
kernel for the block's chromosome (its LOCO kernel, or the whole-genome kernel with
`--no-loco`), the PCs and any covariates: the null model's heritability is fitted by REML,
and the F test and its Freedman-Lane permutations are made on the data whitened by that model.
`--hap-test pcs` (the GUI's "PCs and covariates only") gives the earlier test, with the PCs and
covariates and no kinship.

| Column | Meaning |
|--------|---------|
| **Hap_PValue** | Permutation-based p-value for haplotype effect (Freedman-Lane) |
| **Hap_FDR_BH** | FDR-adjusted haplotype p-value |
| **Hap_F_perm** | Observed F-statistic from the permutation test |
| **Hap_F_param** | Parametric F-statistic (for reference) |
| **Hap_eta2** | Effect size: the share of the phenotype's variance the haplotype groups explain in the test's model, df1·F / (df1·F + df2) of its F (with the kinship-aware test, of the whitened model; with `--hap-test pcs`, on the phenotype after the PCs and covariates). Multiply by 100 for percentage. Grows with the haplotype count and is not corrected for block selection. |
| **Hap_eta2_ci_low** / **Hap_eta2_ci_high** | 95% confidence interval for Hap_eta2 (noncentral-F inversion of the parametric F). |
| **Hap_omega2** | Omega-squared: a less upward-biased variance-explained estimate than eta2 (matters when there are many haplotype groups). Can be negative when the parametric F < 1. Reported alongside eta2, not as a replacement. |
| **Hap_lead_r2** | Additive R² of the block's lead SNP (1 df), in the test's model (with the kinship-aware test, of the whitened phenotype and lead SNP after the PCs and covariates; with `--hap-test pcs`, on the phenotype after the PCs and covariates). Comparable across blocks (fixed df), unlike eta2. The lead SNP may not be a block member. |
| **Hap_n_haplotypes** | Total number of distinct haplotypes observed in the block |
| **Hap_n_tested** | Number of haplotypes with sufficient sample size for testing |
| **Hap_n_samples** | Number of samples with non-missing genotypes in the block |
| **Hap_n_perms** | Number of permutations used |
""")

st.subheader("Subsampling Stability")
st.markdown("""
| Column | Meaning |
|--------|---------|
| **DiscoveryFreq** | Fraction of subsampling resamples in which this SNP was significant. Values > 0.8 indicate robust signals. |
| **MeanNegLog10P** | Average -log10(p) across resamples |
| **MedianNegLog10P** | Median -log10(p) across resamples |
""")

st.subheader("Cross-Model Consensus")
st.markdown("""
| Column | Meaning |
|--------|---------|
| **Detected_by** | Which models detected this SNP as significant |
| **N_models** | Number of models detecting significance (higher = more confidence) |
| **Best_PValue** | Best (lowest) p-value across all models |
""")

st.subheader("Post-GWAS CSVs (added after v1.0.1)")
st.markdown("""
The **Significant SNPs**, **Local LD**, and **Regional Plot** tabs export CSVs
documented in full (every column, type, and when it's emitted) in
[docs/outputs.md](https://github.com/IoannisPerachoritis-hub/TRACE/blob/main/docs/outputs.md):

- **`Significant_SNPs.csv`** / **`Unblocked_SNPs.csv`**: every reporting-significant SNP,
  marked in-block or *unblocked*. A SNP can be significant by the reporting rule yet fall
  **below the block-seeding threshold `--ld-seed-p`**, so it never seeds a block
  (`Block_Status = unblocked_not_seeded`), which is why it shows up here.
  (CLI runs emit these as `Significant_SNPs_<model>.csv` / `Unblocked_SNPs_<model>.csv`.)
- **`LD_r2_long_*.csv`** / **`LD_r2_matrix_*.csv`**: the r² numbers behind the Local LD
  heatmap (long-format pairs, and the square matrix). For a detected block they cover the
  block's member SNPs; the heatmap also shows the block's neighbouring SNPs.
- **`Haplotype_assignments_<trait>.csv`** (MLM) and **`Haplotype_assignments_<trait>_<model>.csv`**
  (MLMM, FarmCPU): which haplotype each sample carries in each tested LD block of that
  model, and whether the block's haplotype test took it (with the reason when not).
- **`Gene_annotation_<trait>.csv`**: the Gene Annotation tab's table: each LD block and
  unblocked significant SNP, with the genes in its candidate interval.
- **`Regional_data_*.csv`**: per-SNP p-value, r²-to-lead, and block membership behind the
  regional plot. A block's `lead_snp` is its **most significant seeding SNP** (the smallest
  GWAS p-value among the significant SNPs that formed the block); it may not be a member, so
  `block_member = False` for the lead is expected, not a bug.

**Covariates** (the GWAS page's covariate upload, or the CLI's `--covar`): a sample
without a value for a covariate is dropped before SNP QC, kinship and the PCs. A warning
names the dropped samples, and `run_metadata.json` records their count.

Run `trace-gwas --help` for the `--covar` covariate flag and the numeric
`--sig-thresh` p-value (e.g. `5e-8`); it prints the full CLI parameter list.
""")

# ============================================================
# 3. Interpreting Results
# ============================================================
st.header("Interpreting Results")

st.subheader("Significance Thresholds")
st.markdown("""
TRACE reports significance under one of four rules. **The default is Bonferroni.**

- **Bonferroni**: divides α by the number of SNPs, assuming they are independent.
  The most stringent option: when many SNPs are correlated it is conservative, and
  real signals in extended LD can fall short of it.
- **M_eff (Li & Ji)**: divides α by the *effective* number of independent tests
  rather than the raw SNP count, so it is **less stringent than Bonferroni**, by how
  much depends on how much LD is present. It does not assume the SNPs are independent.
- **FDR (Benjamini-Hochberg)**: controls the expected proportion of false positives
  among the SNPs called significant, instead of the family-wise error rate. The least
  stringent of these three; suited to exploratory screening.
- **Custom p-value**: a fixed threshold you set (e.g. 5e-8).

Which rule fits depends on whether you are prioritising **discovery** (more permissive)
or **control of false positives** (more stringent), a study-design choice, not a
property of the data.
""")

st.subheader("Lambda GC")
st.markdown("""
Lambda GC is the ratio of the median observed chi-square statistic to its expected value
under the null. It summarises whether the bulk of the test statistics is inflated
relative to the null. It does not tell you whether any individual p-value is correct.

There is no single correct value. What it should be depends on the genetic architecture
of the trait. Under a polygenic architecture, real signal at many loci raises the median
statistic, so an elevated lambda GC is an expected consequence of real association
rather than a defect. Under an oligogenic architecture most markers are null and a value
near 1 is the expected result.

TRACE reports lambda GC and does not act on it. A value far from 1 in either direction is
worth examining alongside the QQ plot and the principal-component diagnostics. Adding
covariates until lambda GC approaches 1 can remove real signal from a polygenic trait.
""")

st.subheader("Effect Sizes")
st.markdown("""
- **Effect (Beta)**: two estimates, both the change per copy of the ALT allele on the
  scale of the phenotype the GWAS used. **Beta_MLM** comes from the mixed model (kinship
  and any PC covariates): FaST-LMM fits the standardised SNP, and TRACE divides its
  estimate and SE by the SNP's dosage standard deviation. **Beta_OLS** comes from an
  ordinary least squares fit with the same PCs and no kinship. A positive value means the
  ALT allele increases the trait. A significant SNP's box plot compares the two:
  `sign-flip` (opposite signs) or `structure-sensitive` (Beta_MLM under half of Beta_OLS)
  means kinship changes the estimate; `concordant` means it does not.
- **eta-squared**: the share of the phenotype's variance that an LD block's haplotype
  groups explain in the haplotype test's model: by default the kinship-aware test, whose
  model holds the kinship, the PCs and any covariates (with the PCs-only test, the
  phenotype after the PCs and covariates are removed). The LD Blocks & Haplotypes tab
  shows this η² in its block table, its box plot and its forest plot; the block table's
  CSV also has the window R² of every SNP in the block's window, another quantity.

Three properties to keep in mind when comparing eta-squared across blocks. It grows with
the number of haplotype groups, so a block with more groups is not directly comparable
with one that has fewer. It is not corrected for the block having been selected because
it contains a significant marker, so it is biased upward. And it describes the panel
analysed, not a general population.
""")

st.subheader("Cross-Model Consensus")
st.markdown("""
The consensus table lists markers that pass the significance threshold in more than one
model. Agreement between models that make different assumptions is informative, but the
models are not independent: they share the genotype matrix, the quality control and, in
most configurations, the kinship. Agreement is therefore weaker evidence than the count
suggests. TRACE reports the overlap without weighting it.
""")

st.subheader("Subsampling Stability")
st.markdown("""
Subsampling repeats the full scan on random subsets of the panel, recomputing the kinship
matrix and the covariates inside each replicate, and reports how often each signal is
recovered. A signal recovered in most replicates does not depend on a few influential
samples; one recovered rarely may.

It is a guide rather than a correction. In TRACE's own simulations it separated true from
false signals in most but not all of the architectures tested; under high polygenicity
the separation disappeared.
""")

# ============================================================
# 4. How to Cite TRACE
# ============================================================
st.header("How to Cite TRACE")
st.markdown("""
If you use TRACE in your research, please cite:

> Perachoritis I., Vatov E., Alseekh S., Gechev T., Rai A. (2026).
> TRACE: An Automated, Extensive GWAS Framework for Crop Breeding.
> *Bioinformatics Advances* (Application Note). Submitted.

**Software:**
> TRACE. https://github.com/IoannisPerachoritis-hub/TRACE
> https://doi.org/10.5281/zenodo.19678860
""")

# ============================================================
# 5. Getting Help
# ============================================================
st.header("Getting Help")
st.markdown("""
- **Bug reports & feature requests**: [GitHub Issues](https://github.com/IoannisPerachoritis-hub/TRACE/issues)
- **Documentation**: [README.md](https://github.com/IoannisPerachoritis-hub/TRACE/blob/main/README.md)
- **Example data**: run `bash examples/run_example.sh`; it generates a small synthetic dataset (100 inbred lines, 600 SNPs, one QTL inside an LD block on chromosome 2) and runs the pipeline end-to-end in about 7 s. (A ready-made copy, `examples/example.vcf.gz`, is in the repository; it is not part of the installed package.)
""")
