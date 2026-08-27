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
The full step-by-step walkthrough — VCF/phenotype/gene-model formats with examples,
QC presets, and downloading results — lives in the project
[README](https://github.com/IoannisPerachoritis-hub/TRACE/blob/main/README.md). It is kept
there as the single source so it can't drift out of sync with the code. For command-line
use, run `trace-gwas --help` — it prints the full parameter list on every install route.

**In 30 seconds:** on **GWAS Analysis**, upload a VCF + a phenotype CSV (sample IDs in
the first column, matching the VCF), pick a trait, and click **Run GWAS** or **One-Click
Full Analysis**. Then open **Post-GWAS Analysis** for LD blocks and haplotype effects;
every page offers a ZIP download.
""")

# ============================================================
# 1. Output Format
# ============================================================
st.header("Output Format")
st.markdown("""
The One-Click Pipeline and the single-trait download both produce a **ZIP**:

| Folder / File | Contents |
|--------------|----------|
| `tables/*.csv` | GWAS hits, LD blocks, haplotype tests, gene annotations, subsampling stability, cross-model consensus |
| `figures/*.png` · `*.html` | Manhattan, QQ, LD heatmaps (static + interactive Plotly) |
| `report.html` | Self-contained offline report (summary, per-model sections, figures) |
| `MANIFEST_*.json` | Full analysis parameters, for reproducibility |
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
| **lead_snp** | The block's most significant SEEDING SNP (smallest GWAS p-value among the significant SNPs that formed the block). It may not be a member of the block's LD-connected SNPs. **lead_snp_pvalue** gives its p-value. |
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
| **Hap_eta2** | Effect size: proportion of phenotypic variance explained by haplotype differences. Multiply by 100 for percentage. Computed on the PC-residualised phenotype; grows with the haplotype count and is not corrected for block selection. |
| **Hap_eta2_ci_low** / **Hap_eta2_ci_high** | 95% confidence interval for Hap_eta2 (noncentral-F inversion of the parametric F). |
| **Hap_omega2** | Omega-squared: a less upward-biased variance-explained estimate than eta2 (matters when there are many haplotype groups). Can be negative when the parametric F < 1. Reported alongside eta2, not as a replacement. |
| **Hap_lead_r2** | Additive R² of the block's lead SNP (1 df), on the same PC-residualised phenotype. Comparable across blocks (fixed df), unlike eta2. The lead SNP may not be a block member. |
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
documented in full — every column, type, and when it's emitted — in
[docs/outputs.md](https://github.com/IoannisPerachoritis-hub/TRACE/blob/main/docs/outputs.md):

- **`Significant_SNPs.csv`** / **`Unblocked_SNPs.csv`** — every reporting-significant SNP,
  marked in-block or *unblocked*. A SNP can be significant by the reporting rule yet fall
  **below the block-seeding threshold `--ld-seed-p`**, so it never seeds a block
  (`Block_Status = unblocked_not_seeded`) — which is why it shows up here.
  (CLI runs emit these as `Significant_SNPs_<model>.csv` / `Unblocked_SNPs_<model>.csv`.)
- **`LD_r2_long_*.csv`** / **`LD_r2_matrix_*.csv`** — the r² numbers behind the Local LD
  heatmap (long-format pairs, and the square matrix).
- **`Regional_data_*.csv`** — per-SNP p-value, r²-to-lead, and block membership behind the
  regional plot. A block's `lead_snp` is its **most significant seeding SNP** (the smallest
  GWAS p-value among the significant SNPs that formed the block); it may not be a member, so
  `block_member = False` for the lead is expected, not a bug.

Run `trace-gwas --help` for the `--covar` covariate flag and the numeric
`--sig-thresh` p-value (e.g. `5e-8`) — it prints the full CLI parameter list.
""")

# ============================================================
# 3. Interpreting Results for Breeding
# ============================================================
st.header("Interpreting Results for Breeding")

st.subheader("Significance Thresholds")
st.markdown("""
TRACE reports significance under one of four rules. **The default is Bonferroni.**

- **Bonferroni**: divides α by the number of SNPs, assuming they are independent.
  The most stringent option — when many SNPs are correlated it is conservative, and
  real signals in extended LD can fall short of it.
- **M_eff (Li & Ji)**: divides α by the *effective* number of independent tests
  rather than the raw SNP count, so it is **less stringent than Bonferroni** — by how
  much depends on how much LD is present. It does not assume the SNPs are independent.
- **FDR (Benjamini–Hochberg)**: controls the expected proportion of false positives
  among the SNPs called significant, instead of the family-wise error rate. The least
  stringent of the three; suited to exploratory screening.
- **Custom p-value**: a fixed threshold you set (e.g. 5e-8).

Which rule fits depends on whether you are prioritising **discovery** (more permissive)
or **control of false positives** (more stringent) — a study-design choice, not a
property of the data.
""")

st.subheader("Lambda GC (Genomic Control)")
st.markdown("""
Lambda GC measures how well the model accounts for population structure:

| Lambda GC | Interpretation |
|-----------|---------------|
| **0.9 - 1.1** | Well-calibrated. P-values are reliable. |
| **< 0.9** | Deflated. Common for oligogenic traits with large-effect loci, or when LOCO kinship removes most confounding. Usually benign, but check that known loci are still detected. |
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
> TRACE: *[full title pending — under R1.8 revision]*.
> *Bioinformatics Advances* (Application Note). [In preparation]

**Software:**
> TRACE v1.0 — https://github.com/IoannisPerachoritis-hub/TRACE
""")

# ============================================================
# 5. Getting Help
# ============================================================
st.header("Getting Help")
st.markdown("""
- **Bug reports & feature requests**: [GitHub Issues](https://github.com/IoannisPerachoritis-hub/TRACE/issues)
- **Documentation**: [README.md](https://github.com/IoannisPerachoritis-hub/TRACE/blob/main/README.md)
- **Example data**: run `bash examples/run_example.sh` — it generates a small synthetic dataset (~50 samples, ~510 SNPs) and runs the pipeline end-to-end in under 30 s. (The example VCF is also shipped as `examples/example.vcf.gz`.)
""")
