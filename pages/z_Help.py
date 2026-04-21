"""Help & Reference — TRACE.

Quick-start guide, output column glossary, and citation info.
"""

import streamlit as st

st.title("Help & Reference")

# ============================================================
# 0. Quick Start
# ============================================================
st.header("Quick Start")
st.markdown("""
1. **Prepare your VCF file** -- Biallelic SNPs, numbered chromosomes. Compressed (`.vcf.gz`) supported. Max 1 GB.
2. **Prepare your phenotype file** -- CSV/TSV with sample IDs in the first column, numeric traits in subsequent columns. Sample IDs must match VCF sample names exactly.
3. **Navigate to GWAS Analysis** -- Upload both files, adjust QC parameters (or use a preset), select a trait. Use **Auto-select PCs** to find the optimal PC count (lambda_GC near 1.0), or set it manually. Click **Run GWAS** for step-by-step results, or **One-Click Full Analysis** for an automated pipeline with ZIP download.
4. **Explore LD structure** -- After GWAS, go to LD & Haplotype Analysis. Significant loci are pre-loaded from your GWAS results.
5. **Download results** -- Each page offers ZIP downloads with all tables, figures, and metadata.

**Data format examples:**

Phenotype CSV:
```
SampleID,Yield,Brix,Firmness
Sample_001,45.2,5.8,3.1
Sample_002,38.7,6.1,2.8
```

Gene model (for LD annotation):
```
CHROM,START,END,STRAND,GENE
1,100000,105000,+,Solyc01g005000
```
""")

# ============================================================
# 1. Output Format
# ============================================================
st.header("Output Format")
st.markdown("""
When you run the **One-Click Pipeline** or download results from the single-trait view,
you receive a **ZIP file** containing:

| Folder / File | Contents |
|--------------|----------|
| `tables/*.csv` | All result tables: GWAS hits, LD blocks, haplotype tests, gene annotations, subsampling stability, cross-model consensus |
| `figures/*.png` | Publication-quality plots: Manhattan, QQ, LD heatmaps |
| `figures/*.html` | Interactive Plotly plots (Manhattan, overlay) |
| `report.html` | Self-contained HTML report with executive summary, per-model sections, and all figures (works offline) |
| `MANIFEST_*.json` | Full analysis parameters for reproducibility |

All CSV files use standard comma-separated format and can be opened in Excel, R, or Python.
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
| **-log10p** | Negative log10 of the p-value (higher = more significant) |
| **Beta_MLM** | Effect size estimate from the mixed linear model. Positive means the minor allele increases the trait value. |
| **SE_MLM** | Standard error of the MLM effect estimate |
| **Beta_OLS** | Marginal effect size from ordinary least squares (ignoring kinship) |
| **SE_OLS** | Standard error of the OLS effect estimate |
| **FDR** | False discovery rate-adjusted p-value (Benjamini-Hochberg). Values < 0.05 are significant after multiple testing correction. |
| **Significant_FDR** | True if FDR < 0.05 |
| **Significant_Bonf** | True if p-value < Bonferroni threshold |
| **Significant_Meff** | True if p-value < M_eff threshold (when M_eff is selected) |
| **Model** | Which GWAS model produced this result (MLM, MLMM, FarmCPU) |
| **Nullh2** | Null-model heritability estimate (internal parameter) |
| **Mixing** | Mixing parameter from FaST-LMM (internal) |
| **ImputationRate** | Fraction of non-missing genotypes at this SNP |
""")

st.subheader("LD Block Annotations")
st.markdown("""
| Column | Meaning |
|--------|---------|
| **Chr** | Chromosome |
| **Start (bp)** / **End (bp)** | Physical boundaries of the LD block |
| **N_SNPs** | Number of SNPs in the block |
| **Lead SNP** | Most significant SNP(s) in the block |
| **overlapping_genes** | Genes whose coordinates overlap with the block |
| **upstream_gene_1** / **downstream_gene_1** | Nearest flanking genes outside the block |
""")

st.subheader("Haplotype GWAS Results")
st.markdown("""
| Column | Meaning |
|--------|---------|
| **Hap_PValue** | Permutation-based p-value for haplotype effect (Freedman-Lane) |
| **Hap_FDR_BH** | FDR-adjusted haplotype p-value |
| **Hap_F_perm** | Observed F-statistic from the permutation test |
| **Hap_F_param** | Parametric F-statistic (for reference) |
| **Hap_eta2** | Effect size: proportion of phenotypic variance explained by haplotype differences. Multiply by 100 for percentage. |
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

# ============================================================
# 3. Interpreting Results for Breeding
# ============================================================
st.header("Interpreting Results for Breeding")

st.subheader("Significance Thresholds")
st.markdown("""
- **M_eff (recommended)**: Accounts for linkage disequilibrium between SNPs.
  Less conservative than Bonferroni, appropriate for structured breeding panels.
- **Bonferroni**: Assumes all SNPs are independent. Very conservative
  — may miss real signals when many SNPs are correlated.
- **FDR**: Controls the expected proportion of false positives among significant results.
  Good for exploratory analysis when you expect many true associations.

For most crop breeding panels (100-500 accessions), M_eff strikes the best balance
between discovery and false positive control.
""")

st.subheader("Lambda GC (Genomic Control)")
st.markdown("""
Lambda GC measures how well the model accounts for population structure:

| Lambda GC | Interpretation |
|-----------|---------------|
| **0.9 - 1.1** | Well-calibrated. P-values are reliable. |
| **< 0.9** | Deflated. Common for oligogenic traits with large-effect loci, or when LOCO kinship removes most confounding signal. Not a problem. |
| **1.1 - 1.3** | Mildly inflated. Some residual population structure may remain. Results are usable. |
| **> 1.3** | Notably inflated. Interpret with caution. Consider adding more PCs or checking for batch effects. |
""")

st.subheader("Effect Sizes and Variance Explained")
st.markdown("""
- **Effect (Beta)**: The estimated change in trait value per copy of the minor allele.
  Positive = minor allele increases the trait. The magnitude depends on the trait's scale.
- **Variance Explained (eta-squared)**: The percentage of total phenotypic variance
  accounted for by haplotype differences at an LD block. This is the most interpretable
  measure of QTL importance for breeding decisions.
  - **> 10%**: Major QTL — strong candidate for marker-assisted selection
  - **5-10%**: Moderate QTL — useful in combination with other loci
  - **< 5%**: Minor QTL — contributes to polygenic background
""")

st.subheader("Cross-Model Consensus")
st.markdown("""
SNPs detected by multiple GWAS models (e.g., MLM + FarmCPU) are higher-confidence
candidates. The cross-model consensus table shows which SNPs pass the significance
threshold in 2 or more models. These are your strongest candidates for follow-up.
""")

st.subheader("Subsampling Stability")
st.markdown("""
Subsampling tests whether a GWAS signal is robust to sample composition.
Each iteration subsamples your panel, recomputes the kinship matrix, and reruns
the full MLM GWAS.

- **Discovery frequency > 80%**: Robust signal — detected in most resamples
- **Discovery frequency 50-80%**: Moderately stable — consider sample-size limitations
- **Discovery frequency < 50%**: Unstable — may be driven by a few influential samples
""")

# ============================================================
# 4. How to Cite TRACE
# ============================================================
st.header("How to Cite TRACE")
st.markdown("""
If you use TRACE in your research, please cite:

> Perachoritis I., Vatov E., Alseekh S., Gechev T., Rai A. (2026).
> TRACE: An Automated End-To-End GWAS Framework for Crop Breeding.
> *Bioinformatics* (Application Note). [In preparation]

**Software:**
> TRACE v1.0 — https://github.com/IoannisPerachoritis-hub/TRACE
""")

# ============================================================
# 5. Getting Help
# ============================================================
st.header("Getting Help")
st.markdown("""
- **Bug reports & feature requests**: [GitHub Issues](https://github.com/IoannisPerachoritis-hub/TRACE/issues)
- **Documentation**: See the README.md in the repository root
- **Example data**: Example VCF and phenotype files are in the `examples/` directory
""")
