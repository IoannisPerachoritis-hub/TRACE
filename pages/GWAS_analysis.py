
import os
import logging
import plotly.express as px
import matplotlib.pyplot as plt
from utils.pub_theme import apply_matplotlib_theme, build_plotly_template
apply_matplotlib_theme()
build_plotly_template()
from fastlmm.association import single_snp
from pysnptools.kernelreader import KernelData as PSKernelData
from pysnptools.snpreader import SnpData
from statsmodels.stats.multitest import multipletests
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import gwas.ld as ld
import json
from scipy import stats
from gwas.utils import (
    PhenoData, CovarData, hash_bytes, hash_df,
    put_array_in_session, put_kernel_in_session, put_object_in_session,
    stable_seed, bump_data_version,
)
from gwas.io import load_vcf_cached, _clean_chr_series
from gwas.qc import gwas_pipeline, allele_freq_from_called_dosage
from gwas.utils import _mean_impute_cols
from gwas.kinship import (
    build_loco_kernels_cached, compute_pcs_full_cached,
    _build_grm_from_Z, _standardize_geno_for_grm,
)
from gwas.models import (
    run_gwas_cached, run_mlmm_core_cached, run_farmcpu, run_farmcpu_cached,
    add_ols_effects_to_gwas, auto_select_pcs,
    bic_proxy_from_design,
    build_mlmm_snpdata, standardize_genotypes_for_mlmm_cached,
    subset_snps_for_mlmm, _ols_fit, _nested_f_test, _one_hot_drop_first,
)
from gwas.subsampling import subsample_gwas_resampling, aggregate_subsampling_to_ld_blocks
from gwas.haplotype import (
    run_haplotype_block_gwas, run_haplotype_block_gwas_cached,
    normalize_ld_blocks_schema, block_test_lm_with_pcs,
    freedman_lane_perm_pvalue,
)
from gwas.stability import _make_run_manifest
from gwas.plotting import (
    compute_cumulative_positions, plot_manhattan_static,
    plot_manhattan_interactive, plot_qq, compute_lambda_gc,
    download_matplotlib_fig, download_plotly_fig,
    _build_gwas_results_zip, _append_metadata_to_zip,
    compute_meff_li_ji,
    PALETTE, PALETTE_CYCLE, export_matplotlib,
)
from annotation import (
    load_gene_annotation, annotate_ld_blocks, consolidate_ld_block_table,
    canon_chr, compute_ld_decay_by_chromosome,
)

# -----------------
# Helper functions for user-friendly display
# -----------------
_MANHATTAN_CAPTION = (
    "Each point is a SNP. Points above the red dashed line are genome-wide "
    "significant. Clusters of significant SNPs on the same chromosome suggest "
    "a quantitative trait locus (QTL)."
)

# Column rename mapping: internal name → user-friendly display name
_COLUMN_RENAME = {
    "Beta_MLM": "Effect (MLM)",
    "SE_MLM": "SE (MLM)",
    "Beta_OLS": "Effect (OLS)",
    "SE_OLS": "SE (OLS)",
    "PValue": "P-value",
    "-log10p": "-log10(P)",
    "Significant_FDR": "Sig (FDR)",
    "Significant_Bonf": "Sig (Bonf)",
    "Significant_Meff": "Sig (M_eff)",
    "eta2": "Var Explained",
    "Hap_eta2": "Hap Var Explained",
}

# Internal columns to hide from user-facing tables
_COLUMNS_TO_HIDE = {
    "Nullh2", "Mixing", "SnpWeightSE", "sid_index", "GenDist",
    "ChrPos", "EffectSize", "ChrNum",
}


def _display_gwas_df(df, **kwargs):
    """Display a GWAS DataFrame with user-friendly column names."""
    display = df.copy()
    # Drop internal columns
    cols_to_drop = [c for c in _COLUMNS_TO_HIDE if c in display.columns]
    if cols_to_drop:
        display = display.drop(columns=cols_to_drop)
    # Convert eta2 to percentage
    for eta_col in ["eta2", "Hap_eta2"]:
        if eta_col in display.columns:
            display[eta_col] = (display[eta_col] * 100).round(1)
    # Rename columns
    rename_map = {k: v for k, v in _COLUMN_RENAME.items() if k in display.columns}
    display = display.rename(columns=rename_map)
    st.dataframe(display, **kwargs)


def _render_results_summary_card(
    lambda_gc, n_significant, sig_label,
    n_ld_blocks=None, auto_pc_k=None,
):
    """Render a results summary card with context-aware lambda_GC interpretation."""
    # Lambda interpretation
    if lambda_gc is None or not np.isfinite(lambda_gc):
        lgc_badge, lgc_note = "N/A", ""
    elif 0.9 <= lambda_gc <= 1.1:
        lgc_badge = "Well-calibrated"
        lgc_note = "P-values are well-calibrated under the null hypothesis."
    elif lambda_gc < 0.9:
        lgc_badge = "Deflated"
        lgc_note = (
            "Deflated genomic control — common for oligogenic traits "
            "with large-effect loci or when LOCO kinship removes most signal."
        )
    elif lambda_gc <= 1.3:
        lgc_badge = "Mildly inflated"
        lgc_note = (
            "Some residual population structure or cryptic relatedness "
            "may remain. Results are usable but interpret with awareness."
        )
    else:
        lgc_badge = "Notably inflated"
        lgc_note = (
            "Substantial inflation detected. Interpret results with caution "
            "— consider adding more PCs or checking for batch effects."
        )

    if auto_pc_k is not None:
        lgc_note += f" Auto-PC selected k = {auto_pc_k}."

    # Metrics row
    cols = st.columns(4 if n_ld_blocks is not None else 3)
    cols[0].metric("Lambda GC", f"{lambda_gc:.3f}" if lambda_gc and np.isfinite(lambda_gc) else "N/A")
    cols[1].metric(f"Significant SNPs ({sig_label})", f"{n_significant:,}")
    if n_ld_blocks is not None:
        cols[2].metric("LD Blocks", f"{n_ld_blocks:,}")
        cols[3].metric("Calibration", lgc_badge)
    else:
        cols[2].metric("Calibration", lgc_badge)

    if lgc_note:
        st.caption(lgc_note)


def _render_interpretation_panel():
    """Render an expandable panel explaining GWAS output for non-specialists."""
    with st.expander("Understanding Your Results", expanded=False):
        st.markdown("""
**Key columns in your results:**

| Column | Meaning |
|--------|---------|
| **P-value** | Probability of observing this association by chance. Smaller = stronger evidence. |
| **Effect (MLM)** | Estimated allele effect size from the mixed linear model. Positive = allele increases trait. |
| **Effect (OLS)** | Marginal allele effect size (ordinary least squares, ignoring kinship). |
| **SE** | Standard error of the effect estimate. Smaller = more precise. |
| **FDR** | False discovery rate-adjusted p-value. Controls the expected proportion of false positives. |
| **Var Explained** | Percentage of phenotypic variance explained by haplotype differences in an LD block. |

**How many significant SNPs is normal?**

This depends on genetic architecture, panel size, and trait heritability. For a typical
crop breeding panel (100-500 accessions):
- **0-5 significant SNPs**: Common for complex polygenic traits (e.g., yield).
- **5-50 significant SNPs**: Expected for moderately heritable traits with a few major loci.
- **50+ significant SNPs**: Typical for highly heritable traits or those controlled by a few large-effect genes with extensive LD.

**What to do next:**

1. **Check LD blocks**: Significant SNPs often cluster into LD blocks on the same chromosome — these represent the same underlying QTL.
2. **Look at candidate genes**: The gene annotation identifies genes overlapping or flanking significant LD blocks.
3. **Cross-model consensus**: SNPs detected by multiple models (MLM + FarmCPU) are higher confidence.
4. **Subsampling stability**: If available, SNPs with high discovery frequency (>80%) across subsampling resamples are robust.
5. **For breeding**: Focus on LD blocks with large effect sizes (Var Explained) and genes with known biological function relevant to your trait.
""")


@st.cache_data(show_spinner="Computing LD decay (per-chromosome)…", max_entries=8)
def _compute_ld_decay_for_gwas_page(geno_key: str, chroms_tuple: tuple,
                                    positions_tuple: tuple):
    """
    Accurate LD decay using compute_ld_decay_by_chromosome (same function
    used by pages/_ld_tabs/tab_decay.py). Returns (decay_df, summary_df,
    median_decay_kb). median_decay_kb may be None if no chromosome yielded
    a valid decay curve.

    Cached by geno_key + chroms + positions — stays hot across one-click
    and manual Run GWAS paths within a session, and across re-runs with
    the same VCF/QC result.
    """
    geno = st.session_state.get(geno_key)
    if geno is None:
        return None, None, None
    chroms_arr = np.asarray(chroms_tuple, dtype=str)
    positions_arr = np.asarray(positions_tuple, dtype=int)
    decay_df, summary_df = compute_ld_decay_by_chromosome(
        chroms=chroms_arr,
        positions=positions_arr,
        geno_imputed=geno,
        max_snps_per_chr=2000,
        max_dist_kb=5000.0,
        n_bins=50,
    )
    median_decay = None
    if summary_df is not None and not summary_df.empty \
            and "decay_kb_r2_0.2" in summary_df.columns:
        _m = summary_df["decay_kb_r2_0.2"].median()
        if pd.notna(_m) and np.isfinite(_m):
            median_decay = float(_m)
    return decay_df, summary_df, median_decay


# -----------------
# Streamlit App UI
# -----------------
st.title("TRACE GWAS")
# =========================
# 1. Data upload & basic QC
# =========================
with st.container():
    col_files, col_qc = st.columns([2, 3])

    with col_files:
        vcf_file = st.file_uploader("Upload Genotype (VCF)", type=["vcf", "vcf.gz"])
        if vcf_file is not None:
            _vcf_size_mb = vcf_file.size / (1024 * 1024)
            if _vcf_size_mb > 200:
                st.warning(
                    f"VCF file is {_vcf_size_mb:.0f} MB. Large files may cause slow "
                    "processing or memory issues. Consider pre-filtering to keep only "
                    "target chromosomes and MAF > 0.01 variants."
                )
        phe_file = st.file_uploader("Upload Phenotype file (.csv or .txt)", type=["csv", "txt", "tsv"])
        # ------------------------------------------------------------
        # Store phenotype file label for reproducible exports
        # ------------------------------------------------------------
        if phe_file is not None:
            pheno_file_label = os.path.splitext(os.path.basename(phe_file.name))[0]
            # sanitize: keep alphanum, dash, underscore only
            pheno_file_label = (
                pheno_file_label
                .replace(" ", "_")
                .replace("/", "_")
                .replace("\\", "_")
            )
            st.session_state["pheno_file_label"] = pheno_file_label

    # ---- Data format guidance (collapsible) ----
    with st.expander("Data format requirements", expanded=False):
        _fmt1, _fmt2 = st.columns(2)
        with _fmt1:
            st.markdown("""
**VCF file (genotypes)**
- Standard VCF format (v4.0+), `.vcf` or `.vcf.gz`
- Must contain `#CHROM`, `POS`, `REF`, `ALT`, and `GT` fields
- Biallelic SNPs only (multi-allelic sites are auto-filtered)
- Sample IDs in the VCF header must match the phenotype file
- Chromosome names: numeric (`1`-`N`), `chrN`, `SL4.0chN`, `CaN` all accepted (auto-detected)
- Imputed VCFs (Beagle, minimac4) are supported -- DR2/R2 scores auto-detected
""")
        with _fmt2:
            st.markdown("""
**Phenotype file (traits)**
- CSV, TSV, or TXT format (separator auto-detected)
- First row must be a header
- Must include an accession/sample ID column (named `Accession`, `Sample`, `IID`, or similar -- or the first column is used)
- Trait columns must contain numeric values
- Missing values: empty cells or `NA`/`NaN` are accepted
- Sample IDs must match VCF sample names exactly

Example:
```
Accession,Fruit_Weight,Brix
LA0716,45.2,5.1
LA1589,12.8,7.3
```
""")

    with col_qc:
        st.markdown("**Genotype QC thresholds**")

        qc_preset = st.selectbox(
            "QC preset (tomato/pepper panels)",
            ["Standard GWAS", "Rare-variant friendly", "Conservative (very clean)"],
            index=0,
            help="Preset sets sensible defaults; you can still override with sliders."
        )

        if qc_preset == "Standard GWAS":
            maf_default, miss_default, indmiss_default, mac_default, info_default = 0.05, 0.10, 0.20, 5, 0.0
        elif qc_preset == "Rare-variant friendly":
            maf_default, miss_default, indmiss_default, mac_default, info_default = 0.01, 0.15, 0.25, 3, 0.3
        else:  # Conservative
            maf_default, miss_default, indmiss_default, mac_default, info_default = 0.05, 0.05, 0.10, 10, 0.8

        col_qc1, col_qc2, col_qc3 = st.columns(3)
        with col_qc1:
            maf_thresh = st.slider("MAF threshold", 0.0, 0.2, float(maf_default), 0.005)
        with col_qc2:
            miss_thresh = st.slider("Missingness (per SNP) max", 0.0, 0.5, float(miss_default), 0.01)
        with col_qc3:
            ind_miss_thresh = st.slider(
                "Missingness (per individual) max",
                0.0, 0.5, float(indmiss_default), 0.01,
                help="Remove individuals missing more than this fraction of SNPs."
            )

        mac_thresh = st.slider(
            "Minor allele count (MAC) minimum",
            min_value=0,
            max_value=50,
            value=int(mac_default),
            step=1,
            help="Filter out extremely rare variants with too low allele support."
        )

        info_thresh = st.slider(
            "Imputation quality (INFO/DR2) minimum",
            min_value=0.0,
            max_value=1.0,
            value=float(info_default),
            step=0.05,
            help=(
                "Filter imputed SNPs below this quality score. "
                "0 = disabled (suitable for non-imputed VCFs). "
                "0.8+ recommended for imputed panels (Beagle DR2, minimac4 R2)."
            ),
        )

# -------------------------------
# Sidebar – GWAS configuration
# -------------------------------
st.sidebar.header("Model selection & mixed-model parameters")

advanced_mode = st.sidebar.toggle(
    "Show advanced settings",
    value=False,
    help="Toggle to show/hide advanced tuning parameters for MLMM and FarmCPU.",
)

use_loco = st.sidebar.checkbox(
    "Use LOCO kinship",
    value=True,
    help=(
        "Leave-One-Chromosome-Out kinship excludes the tested chromosome "
        "from the relationship matrix, avoiding proximal contamination. "
        "Disable for low-marker-density panels where per-chromosome "
        "marker counts may be insufficient."
    ),
)

# Available models
# MLM (FaST-MLM) is the foundational model and ALWAYS runs — its results
# are required by MLMM cofactor seeding, post-GWAS LD
# blocks, and the report generator. The multiselect below only chooses
# which *additional* multi-locus models to layer on top.
model_choices = st.sidebar.multiselect(
    "Additional GWAS models",
    ["MLMM (iterative cofactors)", "FarmCPU (multi-locus)"],
    default=["FarmCPU (multi-locus)"],
    help=(
        "MLM (FaST-MLM) is the foundational single-locus model and always "
        "runs. Optionally layer on multi-locus models:\n\n"
        "**MLMM** iteratively adds significant SNPs as cofactors.\n\n"
        "**FarmCPU** alternates fixed-effect (GLM) and random-effect (MLM) "
        "models to iteratively identify pseudo-QTNs."
    ),
)

# --- MLMM tuning parameters ---
if advanced_mode:
    st.sidebar.subheader("MLMM configuration")

    mlmm_p_enter = st.sidebar.number_input(
        "MLMM entry P-threshold",
        min_value=1e-12,
        max_value=0.2,
        value=1e-4,
        step=1e-1,
        format="%.1e",
        help="Lower = stricter inclusion of cofactors.",
    )

    mlmm_max_cof = st.sidebar.number_input(
        "Maximum number of cofactors",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        help=(
            "Number of loci MLMM can include before stopping. "
            "Note: MLMM uses a BIC proxy from OLS residuals that ignores "
            "kinship covariance, which may stop cofactor selection "
            "prematurely. Consider raising max cofactors or lowering the "
            "entry threshold if you suspect the model is under-fitting."
        ),
    )
else:
    mlmm_p_enter = 1e-4
    mlmm_max_cof = 10

# --- FarmCPU tuning parameters ---
if "FarmCPU (multi-locus)" in model_choices:
    if advanced_mode:
        st.sidebar.subheader("FarmCPU configuration")
        farmcpu_max_iter = st.sidebar.slider(
            "Max FarmCPU iterations", 5, 50, 10,
            help="Stop after this many fixed/random cycles.",
        )
        farmcpu_p_threshold = st.sidebar.number_input(
            "P-threshold for pseudo-QTN selection",
            min_value=1e-12, max_value=0.05, value=1e-2,
            step=1e-1, format="%.1e",
            help="SNPs with p < threshold are candidates for pseudo-QTNs.",
        )
        farmcpu_max_pqtn = st.sidebar.number_input(
            "Max pseudo-QTNs", min_value=5, max_value=50, value=15, step=5,
            help="Cap on pseudo-QTN count.",
        )
    else:
        farmcpu_max_iter = 10
        farmcpu_p_threshold = 1e-2
        farmcpu_max_pqtn = 15
    farmcpu_final_scan = st.sidebar.selectbox(
        "Final scan model",
        options=["mlm", "ols"],
        index=0,
        help="MLM = LOCO-corrected final scan (default, recommended). "
             "OLS = standard FarmCPU (Liu et al. 2016).",
    )

# --- Significance rule ---
st.sidebar.subheader("Significance threshold")
sig_rule = st.sidebar.selectbox(
    "Significance threshold",
    ["FDR (q < 0.05)", "Bonferroni (α = 0.05)", "M_eff — Li & Ji (LD-aware Bonferroni)"],
    index=2,
    help=(
        "**M_eff** (recommended): Corrects for correlated SNPs in LD "
        "— less conservative than Bonferroni. Best for breeding panels "
        "where many SNPs are in linkage disequilibrium.\n\n"
        "**Bonferroni**: Assumes all SNPs are independent — most conservative. "
        "May miss real signals in panels with extended LD.\n\n"
        "**FDR**: Controls the false discovery rate at 5% — least conservative. "
        "Good for exploratory screening when you expect many true associations."
    ),
)

# --- Reproducibility ---
st.sidebar.subheader("Reproducibility")
random_seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=2**31 - 1,
    value=42,
    step=1,
    help=(
        "Seed for stochastic steps (FarmCPU initialization, subsampling "
        "resampling, permutation tests). Set to a fixed value for "
        "reproducible results across runs."
    ),
)


# =========================
# 2. Phenotype handling & transforms
# =========================
if vcf_file and phe_file:

    # --- Load phenotype file (with encoding fallbacks) ---
    try:
        pheno = pd.read_csv(phe_file, sep=None, engine="python", encoding="utf-8-sig")
    except (UnicodeDecodeError, UnicodeError):
        phe_file.seek(0)
        try:
            pheno = pd.read_csv(phe_file, sep=None, engine="python", encoding="latin-1")
            st.warning(
                "Phenotype file is not UTF-8 encoded (detected Latin-1/Windows-1252). "
                "Consider re-saving as UTF-8 for best compatibility."
            )
        except Exception as _enc_err:
            st.error(f"Could not read phenotype file: {_enc_err}")
            st.stop()
    except Exception as _read_err:
        st.error(f"Could not parse phenotype file: {_read_err}")
        st.stop()
    with st.expander("Phenotype file preview", expanded=True):
        st.dataframe(pheno.head())

    # ---- Fix phenotype accession names ----
    pheno = pheno.copy()
    pheno.columns = pheno.columns.astype(str).str.strip()

    # Try to find an explicit accession column first (case-insensitive)
    _possible_cols_lower = {
        "accessions", "accession", "sample", "samples",
        "iid", "genotype", "id", "sample_id", "sampleid",
    }
    accession_col = None
    for col in pheno.columns:
        if col.lower().strip() in _possible_cols_lower:
            accession_col = col
            break

    # If not found, fall back to using the first column as accessions
    if accession_col is None:
        accession_col = pheno.columns[0]
        st.warning(
            f"No explicit accession column found; using first column '{accession_col}' "
            "as accession IDs."
        )

    # Standardize accession names & use as index
    pheno[accession_col] = pheno[accession_col].astype(str).str.strip()
    pheno = pheno.set_index(accession_col)
    pheno.index = pheno.index.astype(str).str.strip()
    # Store RAW phenotype immediately (before zero-handling or transformations)
    # Always refresh pheno_raw when a new phenotype file is loaded
    st.session_state["pheno_raw"] = pheno.copy()

    st.caption(f"Phenotype IDs (first 10): {pheno.index[:10].tolist()}")

    # Post-upload validation: numeric trait column check
    _n_numeric_cols = sum(
        1 for c in pheno.columns
        if pd.to_numeric(pheno[c], errors="coerce").notna().any()
    )
    _n_total_cols = len(pheno.columns)
    if _n_numeric_cols == 0:
        st.error(
            "No numeric trait columns detected. "
            "Check that your phenotype file contains numeric measurements."
        )
    elif _n_numeric_cols < _n_total_cols:
        st.caption(
            f"{_n_numeric_cols} numeric trait column(s) detected, "
            f"{_n_total_cols - _n_numeric_cols} non-numeric column(s) will be skipped."
        )

    # ------------------------------------------------
    # 2a. Phenotype column cleanup
    # ------------------------------------------------
    pheno = pheno.copy()
    pheno.columns = pheno.columns.str.strip()
    # Phenotype state documentation:
    #   pheno_raw       — original upload (for fallback/reset)
    #   pheno_aligned   — GWAS QC-aligned subset (primary for downstream analysis)
    #   pheno_used_for_hap — column-standardized copy for LD/haplotype pages

    # ------------------------------------------------
    # 2b. Optional phenotype transformations
    # ------------------------------------------------
    with st.expander("Optional: transform phenotype columns", expanded=False):
        num_cols = pheno.select_dtypes(include=[np.number]).columns.tolist()

        if not num_cols:
            st.write("No numeric phenotype columns available for transformation.")
        else:
            cols_to_norm = st.multiselect(
                "Select phenotype columns to transform:",
                num_cols,
                help="Transformations will overwrite the selected columns.",
            )

            norm_method = st.selectbox(
                "Transformation method:",
                [
                    "None (raw values)",
                    "Log10 (recommended for metabolites)",
                    "Rank-based inverse normal (INT)",
                ],
            )

            if cols_to_norm and norm_method != "None (raw values)":
                pheno_trans = pheno.copy()
                import scipy.stats as ss

                def safe_log10(x):
                    x = x.astype(float)
                    min_val = np.nanmin(x)
                    shift = abs(min_val) + 1e-6 if min_val <= 0 else 0
                    return np.log10(x + shift)

                def rank_int(x):
                    x = x.astype(float)
                    mask = np.isfinite(x)
                    if mask.sum() < 3:
                        return x
                    ranks = ss.rankdata(x[mask])
                    u = (ranks - 0.5) / mask.sum()
                    z = ss.norm.ppf(u)
                    out = np.full_like(x, np.nan)
                    out[mask] = z
                    return out

                for col in cols_to_norm:
                    x = pd.to_numeric(pheno_trans[col], errors="coerce").astype(float)

                    if norm_method == "Log10 (recommended for metabolites)":
                        pheno_trans[col] = safe_log10(x)
                    elif norm_method == "Rank-based inverse normal (INT)":
                        pheno_trans[col] = rank_int(x)

                st.write(f"Transformation '{norm_method}' applied.")
                st.dataframe(pheno_trans[cols_to_norm].head())

                # SAVE BACK transformed phenotype
                pheno = pheno_trans

                # DO NOT overwrite pheno_raw (preserves original upload)
                # --- Record phenotype preprocessing (transformations) for manifest reproducibility ---
                st.session_state["pheno_transformations"] = {
                    "transformation_method": str(norm_method),
                    "columns_transformed": list(cols_to_norm),
                }

                # Flag to prevent later sidebar normalization from applying again
                st.session_state["pheno_already_transformed"] = True

    # =========================
    # 3. Trait selection
    # =========================
    # After normalization, select only numeric columns as possible traits
    numeric_traits = pheno.select_dtypes(include=[np.number]).columns.tolist()

    st.markdown("### Select trait for GWAS")
    trait_col = st.selectbox(
        "Select a numeric trait for GWAS:",
        numeric_traits,
        help="Select one trait for GWAS analysis. Run separate analyses for additional traits.",
    )
    if not trait_col:
        st.info("Select a numeric trait to proceed.")
        st.stop()

    # =========================
    # 4. GWAS preprocessing (VCF, QC, kinship, PCs)
    # =========================
    drop_alt = st.sidebar.checkbox(
        "Drop non-numeric chromosomes (scaffolds, MT, etc.)",
        value=True,
        help="Keep only numbered chromosomes (auto-detected from VCF); "
             "drops scaffolds, mitochondrial, and other non-chromosomal sequences.",
    )

    # Load VCF bytes
    vcf_bytes = vcf_file.read()
    is_gz = vcf_file.name.endswith(".gz")

    # Phenotype normalization: handled exclusively in Section 2b (Log10/INT).
    # No additional sidebar normalization to avoid double-transformation.
    norm_option = "None"

    n_pcs = st.sidebar.slider(
        "Number of PCs (MLM)",
        min_value=0,
        max_value=20,
        value=4,
        step=1,
        key="n_pcs_mlm_slider",
        help=(
            "Number of principal components used as fixed-effect covariates "
            "in the MLM (FaST-MLM) model. MLMM and FarmCPU default to this "
            "value unless overridden in the expander below."
        ),
    )

    # --- Per-model PC overrides ---
    # Each selected multi-locus model can use a different PC count from MLM.
    # Widgets only render when the model is in `model_choices`; otherwise the
    # override falls back to the MLM value above.
    n_pcs_mlmm = n_pcs
    n_pcs_farmcpu = n_pcs
    with st.sidebar.expander("Per-model PC overrides", expanded=False):
        if "MLMM (iterative cofactors)" in model_choices:
            n_pcs_mlmm = st.number_input(
                "MLMM PCs",
                min_value=0, max_value=20,
                value=int(n_pcs), step=1, key="n_pcs_mlmm_override",
                help="PC count used for MLMM covariates. Defaults to the MLM value.",
            )
        if "FarmCPU (multi-locus)" in model_choices:
            n_pcs_farmcpu = st.number_input(
                "FarmCPU PCs",
                min_value=0, max_value=20,
                value=int(n_pcs), step=1, key="n_pcs_farmcpu_override",
                help="PC count used for FarmCPU covariates. Defaults to the MLM value.",
            )
        if not model_choices:
            st.caption(
                "Overrides appear when MLMM or FarmCPU is selected in "
                "'Additional GWAS models' above."
            )

    st.markdown("### Preparing genotype & kinship matrices for GWAS…")

    # --- Build stable cache keys ---
    vcf_hash = hash_bytes(vcf_bytes)
    pheno_hash = hash_df(pheno)

    # --- Store heavy inputs in session_state under those keys ---
    st.session_state[f"VCF_BYTES::{vcf_hash}"] = vcf_bytes
    st.session_state[f"PHENO_DF::{pheno_hash}"] = pheno.copy()

    # --- Call cached preprocessing using hashes only ---
    results = gwas_pipeline(
        vcf_hash=vcf_hash,
        pheno_hash=pheno_hash,
        trait_col=trait_col,
        maf_thresh=maf_thresh,
        miss_thresh=miss_thresh,
        norm_option=norm_option,
        drop_alt=drop_alt,
        ind_miss_thresh=ind_miss_thresh,
        mac_thresh=mac_thresh,
        info_thresh=info_thresh,
    )

    # Free VCF bytes from session state — no longer needed after parsing
    st.session_state.pop(f"VCF_BYTES::{vcf_hash}", None)
    st.session_state["Z_grm"] = results["Z_for_pca"]
    # Force numpy dtype — chroms_grm may be ArrowStringArray on newer pandas,
    # which Streamlit @st.cache_data cannot hash (build_loco_kernels_cached fails).
    st.session_state["chroms_grm"] = np.asarray(results["chroms_grm"], dtype=str)
    # ---- PCA computed separately, cached, then sliced ----
    pcs_full, pca_eigenvalues = compute_pcs_full_cached(results["Z_for_pca"], max_pcs=20)
    # --- SAFE PCA alignment fix ---
    if pcs_full is not None:
        if pcs_full.shape[0] != results["geno_df"].shape[0]:
            raise RuntimeError(
                f"PCA rows ({pcs_full.shape[0]}) != samples ({results['geno_df'].shape[0]}). "
                "Cache key mismatch; refresh the page or change a parameter to invalidate cache."
            )
    st.session_state["pcs_full"] = pcs_full

    if n_pcs > 0 and pcs_full is not None:
        pcs = pcs_full[:, :n_pcs]
    else:
        pcs = None

    results["pcs"] = pcs  # so the rest of your code works unchanged
    # ============================================================
    # Rebuild FastLMM readers (NOT cached — required)
    # ============================================================

    iid = results["iid"]
    K = results["K"]
    geno_df = results["geno_df"]
    pcs = results["pcs"]
    y = results["y"]
    y_key = put_array_in_session(
        np.asarray(y),
        "Y_VEC",
        vcf_hash,
        pheno_hash,
        trait_col,
    )

    # Kernel reader
    K0 = PSKernelData(iid=iid, val=K)

    # ============================================================
    # LOCO kernels (CACHED)
    # ============================================================
    geno_imputed = results["geno_imputed"]
    chroms = results["chroms"]
    positions = results["positions"]
    iid = results["iid"]
    chroms_num = results["chroms_num"]

    # ------------------------------------------------------------
    # Canonical, cache-safe storage for large arrays (genotypes)
    # ------------------------------------------------------------
    geno_key = put_array_in_session(
        geno_imputed,
        "GENO_IMPUTED",
        vcf_hash, pheno_hash,
        maf_thresh, miss_thresh, ind_miss_thresh, mac_thresh,
        int(drop_alt)
    )

    # (Optional but recommended) also store raw dosage if you pass it around a lot
    dos_key = put_array_in_session(
        results["geno_dosage_raw"],
        "GENO_DOSAGE_RAW",
        vcf_hash, pheno_hash,
        maf_thresh, miss_thresh, ind_miss_thresh, mac_thresh,
        int(drop_alt)
    )

    # put core objects into session_state before building LOCO cache
    # Force numpy dtype on chroms — ArrowStringArray (newer pandas) cannot be
    # hashed by Streamlit @st.cache_data, breaking run_gwas_cached / FarmCPU /
    # MLMM / haplotype cached wrappers downstream.
    st.session_state["chroms"] = np.asarray(chroms, dtype=str)
    st.session_state["positions"] = positions
    st.session_state["iid"] = iid
    st.session_state["K"] = results["K"]

    K0, K_by_chr, loco_diag = build_loco_kernels_cached(
        iid=iid,
        Z_grm=st.session_state["Z_grm"],
        chroms_grm=st.session_state["chroms_grm"],
        K_base=results["K"],
    )
    # Override LOCO kernels with global K0 if user disabled LOCO
    if not use_loco:
        K_by_chr = {ch: K0 for ch in K_by_chr}
    # Warn user if LOCO fell back to the whole-genome kernel for any chromosome
    if use_loco:
        _loco_fallbacks = {
            ch: d for ch, d in loco_diag.items() if d.get("status", "ok") != "ok"
        }
        if _loco_fallbacks:
            _fallback_lines = [
                f"  • Chr {ch}: {d['status']} (off-chr SNPs: {d.get('m_grm_off_chr', '?')})"
                for ch, d in sorted(_loco_fallbacks.items())
            ]
            st.warning(
                "**LOCO kernel fallback:** The whole-genome kinship K₀ was used instead of a "
                "leave-one-chromosome-out kernel for the following chromosomes. This means the "
                "LOCO correction is not applied there, which may slightly inflate statistics "
                "for SNPs on those chromosomes.\n\n" + "\n".join(_fallback_lines)
            )

    # Store final kinship kernel for MLMM caching
    K0_key = put_kernel_in_session(K0, "K0", vcf_hash, pheno_hash)
    st.session_state["K0"] = K0
    st.session_state["K_by_chr"] = K_by_chr
    st.session_state["loco_diagnostics_by_chr"] = loco_diag

    # Phenotype reader
    pheno_reader = PhenoData(iid=iid, val=y)
    pheno_reader_key = put_object_in_session(
        pheno_reader,
        "PHENO_READER",
        vcf_hash,
        pheno_hash,
        trait_col
    )

    # Covariates
    if pcs is not None:
        covar_reader = CovarData(
            iid=iid,
            val=pcs,
            names=[f"PC{i + 1}" for i in range(pcs.shape[1])]
        )
    else:
        covar_reader = None
    # ============================================================
    # SAVE FastLMM objects to session_state (CRITICAL)
    # ============================================================
    st.session_state["K0"] = K0
    st.session_state["K_by_chr"] = K_by_chr
    st.session_state["pheno_reader"] = pheno_reader
    st.session_state["covar_reader"] = covar_reader

    if "qc_snp" in results:
        _qc = results["qc_snp"]
        _n_total = _qc.get("Total SNPs", 0)
        _n_pass = _qc.get("Pass ALL", _n_total)
        _n_samples = results["geno_df"].shape[0]

        st.markdown("#### Data Summary")
        _c1, _c2, _c3 = st.columns(3)
        _c1.metric("Samples", f"{_n_samples:,}")
        _c2.metric("SNPs input", f"{_n_total:,}")
        _c3.metric("SNPs passing QC", f"{_n_pass:,}",
                    delta=f"{_n_pass/_n_total*100:.1f}%" if _n_total > 0 else None)

        # Step-by-step QC filtering table
        _qc_rows = []
        for _filt, _key in [
            ("MAF", "Fail MAF"),
            ("Missingness", "Fail Missingness"),
            ("MAC", "Fail MAC"),
            ("INFO", "Fail INFO"),
        ]:
            _cnt = _qc.get(_key, 0)
            if _cnt > 0 or _key in ("Fail MAF", "Fail Missingness"):
                _pct = _cnt / _n_total * 100 if _n_total > 0 else 0
                _qc_rows.append(f"| {_filt} | {_cnt:,} | {_pct:.1f}% |")

        if _qc_rows:
            _qc_table = "| Filter | SNPs removed | % of input |\n|--------|-------------|------------|\n"
            _qc_table += "\n".join(_qc_rows)
            st.markdown(_qc_table)

        if results.get("info_field"):
            st.info(f"Imputation quality field detected: **{results['info_field']}**")

        # MAF distribution + per-chromosome breakdown (collapsible)
        with st.expander("Post-QC diagnostics", expanded=False):
            from gwas.plotting import plot_maf_histogram
            _geno_imp = results["geno_imputed"]
            _freq = _geno_imp.mean(axis=0) / 2.0
            _maf_vals = np.minimum(_freq, 1.0 - _freq)
            _fig_maf = plot_maf_histogram(_maf_vals, maf_threshold=maf_thresh)
            st.pyplot(_fig_maf)
            plt.close(_fig_maf)

            # Per-chromosome SNP summary
            _chr_arr = results["chroms"]
            _pos_arr = results["positions"].astype(int)
            _chr_summary = []
            for _ch in sorted(pd.Series(_chr_arr).unique(), key=lambda x: (0, int(str(x).replace("chr","").replace("Chr",""))) if str(x).replace("chr","").replace("Chr","").isdigit() else (1, str(x))):
                _m = _chr_arr == _ch
                _n_snps_ch = int(_m.sum())
                _span_mb = (_pos_arr[_m].max() - _pos_arr[_m].min()) / 1e6 if _n_snps_ch > 1 else 0
                _density = _n_snps_ch / max(_span_mb, 0.001)
                _chr_summary.append({
                    "Chr": _ch,
                    "SNPs": _n_snps_ch,
                    "Coverage (Mb)": round(_span_mb, 1),
                    "Density (SNPs/Mb)": round(_density, 1),
                })
            st.dataframe(pd.DataFrame(_chr_summary), hide_index=True, use_container_width=True)

    # --- REQUIRED: Save everything to session_state for cached GWAS ---
    # ------------------------------------------------------------
    # Store SERIALIZABLE GWAS results in session_state
    # ------------------------------------------------------------
    for key in [
        "geno_imputed",
        "geno_dosage_raw",
        "chroms",
        "chroms_num",
        "positions",
        "sid",
        "iid",
        "K",
        "pcs",
        "geno_df",
        "pheno",
        "y",
        "n_snps_raw",
        "qc_snp",
    ]:
        st.session_state[key] = results[key]

    # GWAS-aligned phenotype (post-QC)
    st.session_state["pheno_aligned"] = results["pheno"]

    # --- Sample overlap reporting ---
    _n_geno = results["geno_df"].shape[0]
    _n_pheno = len(pheno)
    if _n_geno < 10:
        st.error(
            f"Only {_n_geno} samples overlap between genotype and phenotype "
            f"({_n_pheno} phenotype samples). Check that sample IDs match between files."
        )
        st.stop()
    elif _n_geno < _n_pheno * 0.5:
        st.warning(
            f"Only {_n_geno}/{_n_pheno} phenotype samples found in genotype data. "
            f"{_n_pheno - _n_geno} samples dropped during alignment."
        )

    # ------------------------------------------------------------
    # CRITICAL: LD / haplotype pages must use the SAME phenotype table
    # that was aligned & filtered for GWAS samples.
    # ------------------------------------------------------------
    pheno_for_downstream = results["pheno"].copy()
    pheno_for_downstream.columns = pheno_for_downstream.columns.astype(str).str.strip()

    st.session_state["pheno_used_for_hap"] = pheno_for_downstream
    bump_data_version()  # notify downstream pages (LD) that data changed

    # --- Use canonical objects from session_state ---
    geno_imputed = st.session_state["geno_imputed"]
    chroms = st.session_state["chroms"]
    chroms_num = st.session_state["chroms_num"]
    positions = st.session_state["positions"]
    sid = st.session_state["sid"]
    iid = st.session_state["iid"]
    K = st.session_state["K"]
    K_by_chr = st.session_state.get("K_by_chr")
    pcs = st.session_state["pcs"]
    covar_reader = st.session_state["covar_reader"]
    geno_df = st.session_state["geno_df"]
    pheno_clean = st.session_state["pheno_aligned"]
    y = st.session_state["y"]
    pheno_reader = st.session_state["pheno_reader"]
    # ============================================================
    # CANONICAL GWAS CONTEXT (used by LD / haplotype pages)
    # ============================================================

    st.session_state["GWAS_CTX"] = {
        "geno_imputed": st.session_state["geno_imputed"],
        "geno_dosage_raw": st.session_state.get("geno_dosage_raw"),
        "geno_index": st.session_state["geno_df"].index.astype(str).to_numpy(),
        "sid": st.session_state["sid"],
        "chroms": st.session_state["chroms"],
        "chroms_num": st.session_state["chroms_num"],
        "positions": st.session_state["positions"],
        "pheno_aligned": st.session_state["pheno_aligned"].copy(),
        "trait": trait_col,
        "iid": st.session_state["iid"],
        "y": st.session_state["y"],
        "pcs": st.session_state["pcs"],
        "covar_reader": st.session_state["covar_reader"],
    }

    # ================================================================
    #       AUTO PC SELECTION (lambda-based)
    # ================================================================

    with st.sidebar.expander("Auto-select PCs (lambda-based)", expanded=False):
        st.caption(
            "Scan the selected association model at different PC counts "
            "and pick the one with λGC closest to 1.0 (well-calibrated "
            "population structure correction)."
        )
        _scan_model = st.selectbox(
            "Model to scan",
            ["MLM", "MLMM", "FarmCPU"],
            key="scan_pcs_model",
            help=(
                "Run the λGC-vs-PCs scan for the selected association "
                "model. MLMM and FarmCPU are slower than MLM."
            ),
        )
        max_pc_scan = st.slider("Max PCs to scan", 2, 15, 10, key="max_pc_scan")
        if st.button("Scan PCs", key="scan_pcs_btn"):
            pcs_full_arr = st.session_state["pcs_full"]
            prog = st.progress(0, text="Scanning PC counts...")
            def _pc_progress(k, total):
                prog.progress(
                    k / total,
                    text=f"Running {_scan_model} with {k} PCs… ({k+1}/{total})",
                )
            pc_scan_df = auto_select_pcs(
                geno_imputed=geno_imputed,
                y=y,
                sid=sid,
                chroms=chroms,
                chroms_num=chroms_num,
                positions=positions,
                iid=iid,
                Z_grm=st.session_state["Z_grm"],
                chroms_grm=st.session_state["chroms_grm"],
                K_base=K,
                pcs_full=pcs_full_arr,
                max_pcs=max_pc_scan,
                progress_callback=_pc_progress,
                use_loco=use_loco,
                model=_scan_model.lower(),
            )
            prog.empty()
            st.session_state["pc_scan_df"] = pc_scan_df
            st.session_state["pc_scan_model"] = _scan_model

        if "pc_scan_df" in st.session_state:
            pc_df = st.session_state["pc_scan_df"]
            _scan_model_shown = st.session_state.get("pc_scan_model", "MLM")
            st.caption(f"Last scan: **{_scan_model_shown}**")
            st.dataframe(
                pc_df.style.apply(
                    lambda row: ["background-color: #d4edda" if row["recommended"] == "★" else "" for _ in row],
                    axis=1,
                ),
                use_container_width=True,
                hide_index=True,
            )
            # Lambda vs PCs chart
            import plotly.graph_objects as go
            fig_pc = go.Figure()
            fig_pc.add_trace(go.Scatter(
                x=pc_df["n_pcs"], y=pc_df["lambda_gc"],
                mode="lines+markers", name=f"λGC ({_scan_model_shown})",
            ))
            fig_pc.add_hline(y=1.0, line_dash="dash", line_color="red",
                             annotation_text="λ = 1.0")
            fig_pc.update_layout(
                xaxis_title="Number of PCs",
                yaxis_title="λGC",
                height=300,
                margin=dict(t=30, b=30),
            )
            st.plotly_chart(fig_pc, use_container_width=True)
            download_plotly_fig(
                fig_pc,
                filename=f"PC_lambda_curve_{_scan_model_shown}_{trait_col}.html",
                label="Download PC lambda curve",
            )

            # "Use recommended" button
            best_row = pc_df.loc[pc_df["recommended"] == "★"]
            if not best_row.empty:
                best_k = int(best_row.iloc[0]["n_pcs"])
                best_lam = best_row.iloc[0]["lambda_gc"]
                st.write(
                    f"Recommended ({_scan_model_shown}): **{best_k} PCs** "
                    f"(λGC = {best_lam})"
                )
                # Propagate the recommended PC count to every widget tied
                # to the scanned model via an on_click callback. Callbacks
                # fire BEFORE the next script run, so the target widgets
                # haven't been instantiated yet — writing to session_state
                # inside the button branch instead raises
                # StreamlitAPIException because the PC widgets already
                # rendered earlier in this run.
                def _apply_recommended_pcs(model: str, k: int) -> None:
                    if model == "MLM":
                        st.session_state["n_pcs_mlm_slider"] = k
                    elif model == "MLMM":
                        st.session_state["n_pcs_mlmm_override"] = k
                    elif model == "FarmCPU":
                        st.session_state["n_pcs_farmcpu_override"] = k
                    # One-click pipeline manual widget (same naming
                    # convention as the factory: pipe_pcs_<model_lower>)
                    st.session_state[f"pipe_pcs_{model.lower()}"] = k

                st.button(
                    f"Use {best_k} PCs",
                    key="use_recommended_pcs",
                    on_click=_apply_recommended_pcs,
                    args=(_scan_model_shown, best_k),
                )

    # ================================================================
    #       ONE-CLICK FULL ANALYSIS PIPELINE
    # ================================================================

    with st.expander("One-Click Full Analysis", expanded=False):
        st.caption(
            "Automatically chains: PC selection → MLM GWAS → multi-model → "
            "LD blocks → haplotype testing → gene annotation → "
            "LD heatmaps → (optional subsampling) → report → ZIP download."
        )

        # --- Configuration panel ---
        _cfg_col1, _cfg_col2 = st.columns(2)
        with _cfg_col1:
            # MLM (FaST-MLM) is the foundational model and always runs — see
            # the matching note on `model_choices` in the sidebar above.
            _pipe_models = st.multiselect(
                "Additional models to include",
                ["MLMM (iterative cofactors)", "FarmCPU (multi-locus)"],
                default=["FarmCPU (multi-locus)"],
                key="pipe_models",
                help=(
                    "MLM (FaST-MLM) is the foundational model and always "
                    "runs. Pick any additional multi-locus models to layer "
                    "on top."
                ),
            )
            _pipe_run_subsampling = st.checkbox(
                "Include subsampling stability screening",
                value=False,
                key="pipe_subsampling",
                help="Run subsampling GWAS resampling (MLM + GRM recomputation) "
                     "to assess signal stability across random subsamples.",
            )
            _pipe_use_loco = st.checkbox(
                "Use LOCO kinship",
                value=use_loco,
                key="pipe_use_loco",
                help=(
                    "Leave-One-Chromosome-Out kinship for MLM and FarmCPU's "
                    "final scan (default). Uncheck to use the whole-genome "
                    "GRM instead — useful when LOCO over-corrects on traits "
                    "with strong single-chromosome QTL. Overrides the "
                    "sidebar LOCO setting for this pipeline run only."
                ),
            )
            _pipe_pc_mode = st.radio(
                "PC selection mode",
                ["Auto (per-model λ scan)", "Manual (set per model)"],
                key="pipe_pc_mode",
                horizontal=True,
                help="Auto: scan λGC for each model independently and pick best k. "
                     "Manual: set PC count per model yourself.",
            )
            if _pipe_pc_mode.startswith("Auto"):
                _pipe_pc_strategy = st.radio(
                    "Auto strategy",
                    ["Band [0.95–1.05] (recommended)", "Closest to λ=1.0"],
                    key="pipe_pc_strategy",
                    help="Band: smallest PC count with λGC in [0.95, 1.05]; "
                         "falls back to closest-to-1.0 with parsimony tolerance. "
                         "Closest: pick the PC count with λGC nearest to 1.0.",
                )
            else:
                _pipe_pc_strategy = None
                _pc_manual_models = ["MLM"]
                if "MLMM (iterative cofactors)" in _pipe_models:
                    _pc_manual_models.append("MLMM")
                if "FarmCPU (multi-locus)" in _pipe_models:
                    _pc_manual_models.append("FarmCPU")
                _pc_man_cols = st.columns(len(_pc_manual_models))
                _pipe_manual_pcs = {}
                for _col, _mname in zip(_pc_man_cols, _pc_manual_models):
                    with _col:
                        _pipe_manual_pcs[_mname] = st.number_input(
                            f"{_mname} PCs", min_value=0, max_value=20,
                            value=4, step=1, key=f"pipe_pcs_{_mname.lower()}",
                        )
            _pipe_sig_rule = st.selectbox(
                "Significance threshold",
                ["M_eff — Li & Ji (LD-aware Bonferroni)", "Bonferroni (α = 0.05)", "FDR (q < 0.05)"],
                key="pipe_sig_rule",
                help=(
                    "**M_eff** (recommended): Corrects for correlated SNPs in LD "
                    "— less conservative than Bonferroni. Best for breeding panels.\n\n"
                    "**Bonferroni**: Assumes all SNPs are independent — most conservative.\n\n"
                    "**FDR**: Controls false discovery rate at 5% — least conservative, "
                    "good for exploratory screening."
                ),
            )
        with _cfg_col2:
            from utils.species_files import SPECIES_FILES as _PIPE_SP
            _pipe_species_opts = list(_PIPE_SP.keys()) + ["Other (upload files)"]
            _pipe_species = st.selectbox(
                "Species",
                _pipe_species_opts,
                key="pipe_species",
                help=(
                    "Pick a bundled species for auto-loaded gene annotation, "
                    "or 'Other (upload files)' to supply your own gene model "
                    "(any biallelic diploid species with a CSV-formatted gene "
                    "coordinate file)."
                ),
            )
            _sp_cfg = _PIPE_SP.get(_pipe_species, {})
            _has_build_choice = "gene_model_SL3" in _sp_cfg
            if _has_build_choice:
                _pipe_genome_build = st.selectbox(
                    "Genome build",
                    ["SL3", "SL4"],
                    key="pipe_genome_build",
                    help=(
                        "SL3 matches Varitome / SL2.5 SNP coordinates (recommended for most tomato panels). "
                        "SL4 uses ITAG4.0/SL4 coordinates."
                    ),
                )
            else:
                _pipe_genome_build = None

            # Override gene files — mandatory for "Other (upload files)",
            # optional override for bundled species.
            _is_other_species = _pipe_species == "Other (upload files)"
            with st.expander(
                "Override gene files"
                + (" (required for 'Other')" if _is_other_species else ""),
                expanded=_is_other_species,
            ):
                st.file_uploader(
                    "Gene coordinates CSV (columns: Chr, Start, End, [Strand,] Gene_ID)",
                    type=["csv", "tsv", "txt"],
                    key="pipe_gene_model_override",
                    help=(
                        "Overrides the bundled gene model if provided. "
                        "Column aliases accepted (CHROM/Chr/chromosome, "
                        "START/Start/start_pos, END/End/end_pos, "
                        "GENE/Gene_ID/name/gene_name). "
                        "For a new species, derive this from your GFF3 with "
                        "a short pandas script."
                    ),
                )
                st.file_uploader(
                    "Gene descriptions TSV (optional: gene_id \\t description)",
                    type=["txt", "tsv"],
                    key="pipe_gene_desc_override",
                    help=(
                        "Optional. Overrides bundled descriptions if provided. "
                        "Tab-separated; '#' comment lines allowed."
                    ),
                )
            _pipe_ld_r2 = st.number_input(
                "LD r² threshold", min_value=0.1, max_value=1.0,
                value=0.6, step=0.05, key="pipe_ld_r2",
            )
            _pipe_ld_seed = st.radio(
                "LD block seed SNPs",
                ["Suggestive (p-threshold + top N)", "Significant only (chosen threshold)"],
                key="pipe_ld_seed",
                horizontal=True,
                help="'Suggestive' finds LD blocks around top peaks (same as LD Analysis page). "
                     "'Significant only' restricts to SNPs passing your genome-wide threshold.",
            )
            if _pipe_ld_seed.startswith("Suggestive"):
                _c1, _c2 = st.columns(2)
                with _c1:
                    _pipe_ld_sig_p = st.number_input(
                        "Seed p-threshold", min_value=1e-12, max_value=0.1,
                        value=1e-5, format="%.1e", key="pipe_ld_sig_p",
                    )
                with _c2:
                    _pipe_ld_top_n = st.number_input(
                        "Also include top N", min_value=0, max_value=500,
                        value=10, step=1, key="pipe_ld_top_n",
                    )
            else:
                _pipe_ld_sig_p = None
                _pipe_ld_top_n = 0
            _pipe_ld_flank_mode = st.radio(
                "LD flank window",
                ["Auto (LD decay-based)", "Manual (fixed kb)"],
                key="pipe_ld_flank_mode",
                horizontal=True,
            )
            if _pipe_ld_flank_mode == "Manual (fixed kb)":
                _pipe_ld_flank = st.number_input(
                    "LD flank (kb)", min_value=50, max_value=1000,
                    value=300, step=50, key="pipe_ld_flank",
                )
            else:
                _pipe_ld_flank = None  # will be set during pipeline
            _pipe_hap_perms = st.number_input(
                "Haplotype permutations", min_value=100, max_value=5000,
                value=1000, step=100, key="pipe_hap_perms",
            )

        with st.expander("Advanced model settings", expanded=False):
            _adv1, _adv2 = st.columns(2)
            with _adv1:
                st.caption("MLMM")
                _pipe_mlmm_p = st.number_input(
                    "Entry P-threshold", min_value=1e-12, max_value=0.2,
                    value=1e-4, format="%.1e", key="pipe_mlmm_p",
                )
                _pipe_mlmm_max_cof = st.number_input(
                    "Max cofactors", min_value=1, max_value=50,
                    value=10, step=1, key="pipe_mlmm_max_cof",
                )
            with _adv2:
                st.caption("FarmCPU")
                _pipe_fc_p = st.number_input(
                    "P-threshold", min_value=1e-12, max_value=0.05,
                    value=1e-2, format="%.1e", key="pipe_fc_p",
                )
                _pipe_fc_max_iter = st.number_input(
                    "Max iterations", min_value=5, max_value=50,
                    value=10, step=5, key="pipe_fc_max_iter",
                )
                _pipe_fc_max_pqtn = st.number_input(
                    "Max pseudo-QTNs", min_value=5, max_value=50,
                    value=15, step=5, key="pipe_fc_max_pqtn",
                )
                _pipe_fc_final_scan = st.selectbox(
                    "Final scan", options=["mlm", "ols"], index=0,
                    key="pipe_fc_final_scan",
                    help="OLS = standard FarmCPU. MLM = LOCO-corrected.",
                )
            if _pipe_run_subsampling:
                st.caption("Subsampling GWAS")
                _boot_a1, _boot_a2, _boot_a3 = st.columns(3)
                with _boot_a1:
                    _pipe_boot_n_reps = st.number_input(
                        "Iterations", min_value=10, max_value=200,
                        value=50, step=10, key="pipe_boot_reps",
                    )
                with _boot_a2:
                    _pipe_boot_frac = st.number_input(
                        "Sample fraction", min_value=0.60, max_value=0.95,
                        value=0.80, step=0.05, key="pipe_boot_frac",
                    )
                with _boot_a3:
                    _pipe_boot_thresh = st.number_input(
                        "Discovery P threshold", min_value=1e-8, max_value=1e-2,
                        value=1e-4, format="%.1e", key="pipe_boot_thresh",
                    )
                _pipe_boot_loco = st.checkbox(
                    "Use LOCO kinship in subsampling",
                    value=False,
                    key="pipe_boot_loco",
                    help=(
                        "Rebuild per-chromosome LOCO kinship for each subsample. "
                        "More accurate but significantly slower (~12× per iteration)."
                    ),
                )
            else:
                _pipe_boot_n_reps = 50
                _pipe_boot_frac = 0.80
                _pipe_boot_thresh = 1e-4
                _pipe_boot_loco = False

        if st.button("Run Full Analysis", key="run_full_pipeline", type="primary"):
            from gwas.reports import generate_gwas_report as _gen_report

            _pipe_extra_tables = {}
            _pipe_extra_dfs = {}
            _pipe_figures = {}
            _pipe_pc_df = None
            _pipe_lam = 1.0  # safe default
            _pipe_boot_disc_df = None

            # Per-model PC counts — will be set by auto or manual mode
            _pipe_k_mlm = n_pcs  # fallback to sidebar slider
            _pipe_k_mlmm = n_pcs
            _pipe_k_fc = n_pcs

            with st.status("Running full analysis...", expanded=True) as _status:

                # --- Step 1: PC selection (auto or manual) ---
                _status.update(label="Step 1: Selecting PCs...")
                pcs_full_arr = st.session_state["pcs_full"]
                _max_avail = pcs_full_arr.shape[1] if pcs_full_arr is not None else 0

                if _pipe_pc_mode.startswith("Manual"):
                    # --- Manual per-model PCs ---
                    _pipe_k_mlm = min(_pipe_manual_pcs.get("MLM", n_pcs), _max_avail)
                    _pipe_k_mlmm = min(_pipe_manual_pcs.get("MLMM", _pipe_k_mlm), _max_avail)
                    _pipe_k_fc = min(_pipe_manual_pcs.get("FarmCPU", _pipe_k_mlm), _max_avail)
                    _pc_msg = f"Manual PCs — MLM: {_pipe_k_mlm}"
                    if "MLMM (iterative cofactors)" in _pipe_models:
                        _pc_msg += f", MLMM: {_pipe_k_mlmm}"
                    if "FarmCPU (multi-locus)" in _pipe_models:
                        _pc_msg += f", FarmCPU: {_pipe_k_fc}"
                    st.write(_pc_msg)

                elif pcs_full_arr is not None and _max_avail >= 2:
                    # --- Auto per-model PC selection ---
                    if "Band" in _pipe_pc_strategy:
                        _pc_strat = "band"
                    else:
                        _pc_strat = "closest_to_1"

                    # MLM lambda scan
                    _pipe_pc_df = auto_select_pcs(
                        geno_imputed=geno_imputed, y=y, sid=sid,
                        chroms=chroms, chroms_num=chroms_num, positions=positions,
                        iid=iid, Z_grm=st.session_state["Z_grm"],
                        chroms_grm=st.session_state["chroms_grm"],
                        K_base=K, pcs_full=pcs_full_arr, max_pcs=10,
                        strategy=_pc_strat,
                        use_loco=_pipe_use_loco,
                        model="mlm",
                    )
                    # MLM best k
                    _best_row_mlm = _pipe_pc_df.loc[_pipe_pc_df["recommended"] == "★"]
                    if not _best_row_mlm.empty:
                        _pipe_k_mlm = int(_best_row_mlm.iloc[0]["n_pcs"])
                    _pipe_pc_df.rename(
                        columns={"lambda_gc": "lambda_gc_MLM",
                                 "delta_from_1": "delta_MLM",
                                 "recommended": "recommended_MLM"},
                        inplace=True,
                    )

                    # MLMM independent lambda scan
                    if "MLMM (iterative cofactors)" in _pipe_models:
                        st.write("Scanning PCs for MLMM...")
                        _mlmm_scan_df = auto_select_pcs(
                            geno_imputed=geno_imputed, y=y, sid=sid,
                            chroms=chroms, chroms_num=chroms_num, positions=positions,
                            iid=iid, Z_grm=st.session_state["Z_grm"],
                            chroms_grm=st.session_state["chroms_grm"],
                            K_base=K, pcs_full=pcs_full_arr, max_pcs=10,
                            strategy=_pc_strat,
                            use_loco=_pipe_use_loco,
                            model="mlmm",
                            mlmm_p_enter=_pipe_mlmm_p,
                            mlmm_max_cof=_pipe_mlmm_max_cof,
                        )
                        _pipe_pc_df["lambda_gc_MLMM"] = _mlmm_scan_df["lambda_gc"].values
                        _pipe_pc_df["delta_MLMM"] = _mlmm_scan_df["delta_from_1"].values
                        _pipe_pc_df["recommended_MLMM"] = _mlmm_scan_df["recommended"].values
                        _best_row_mlmm = _mlmm_scan_df.loc[
                            _mlmm_scan_df["recommended"] == "★"
                        ]
                        if not _best_row_mlmm.empty:
                            _pipe_k_mlmm = int(_best_row_mlmm.iloc[0]["n_pcs"])

                    # FarmCPU independent lambda scan
                    if "FarmCPU (multi-locus)" in _pipe_models:
                        st.write("Scanning PCs for FarmCPU...")
                        _fc_scan_df = auto_select_pcs(
                            geno_imputed=geno_imputed, y=y, sid=sid,
                            chroms=chroms, chroms_num=chroms_num, positions=positions,
                            iid=iid, Z_grm=st.session_state["Z_grm"],
                            chroms_grm=st.session_state["chroms_grm"],
                            K_base=K, pcs_full=pcs_full_arr, max_pcs=10,
                            strategy=_pc_strat,
                            use_loco=_pipe_use_loco,
                            model="farmcpu",
                            farmcpu_p_threshold=_pipe_fc_p,
                            farmcpu_max_iterations=_pipe_fc_max_iter,
                            farmcpu_max_pseudo_qtns=_pipe_fc_max_pqtn,
                            farmcpu_final_scan=_pipe_fc_final_scan,
                        )
                        _pipe_pc_df["lambda_gc_FarmCPU"] = _fc_scan_df["lambda_gc"].values
                        _pipe_pc_df["delta_FarmCPU"] = _fc_scan_df["delta_from_1"].values
                        _pipe_pc_df["recommended_FarmCPU"] = _fc_scan_df["recommended"].values
                        _best_row_fc = _fc_scan_df.loc[
                            _fc_scan_df["recommended"] == "★"
                        ]
                        if not _best_row_fc.empty:
                            _pipe_k_fc = int(_best_row_fc.iloc[0]["n_pcs"])

                    _pipe_extra_tables["PC_selection_lambda.csv"] = _pipe_pc_df

                    # Display per-model PC summary
                    _pc_msg = f"Auto-selected PCs — MLM: **{_pipe_k_mlm}**"
                    if "MLMM (iterative cofactors)" in _pipe_models:
                        _pc_msg += f", MLMM: **{_pipe_k_mlmm}**"
                    if "FarmCPU (multi-locus)" in _pipe_models:
                        _pc_msg += f", FarmCPU: **{_pipe_k_fc}**"
                    st.write(_pc_msg)

                    # PC selection lambda curve figure
                    if _pipe_pc_df is not None and len(_pipe_pc_df) > 1:
                        try:
                            _fig_pc, _ax_pc = plt.subplots(figsize=(6, 3.5))
                            _ax_pc.plot(_pipe_pc_df["n_pcs"], _pipe_pc_df["lambda_gc_MLM"],
                                        "o-", color="#0072B2", label="MLM")
                            if "lambda_gc_MLMM" in _pipe_pc_df.columns:
                                _ax_pc.plot(_pipe_pc_df["n_pcs"], _pipe_pc_df["lambda_gc_MLMM"],
                                            "^-", color="#009E73", label="MLMM")
                            if "lambda_gc_FarmCPU" in _pipe_pc_df.columns:
                                _ax_pc.plot(_pipe_pc_df["n_pcs"], _pipe_pc_df["lambda_gc_FarmCPU"],
                                            "s-", color="#D55E00", label="FarmCPU")
                            _ax_pc.axhline(1.0, ls="--", color="#999", alpha=0.7)
                            # Mark per-model best PC counts
                            _ax_pc.axvline(_pipe_k_mlm, ls=":", color="#0072B2", alpha=0.6, lw=1.5,
                                           label=f"MLM best: {_pipe_k_mlm}")
                            _mlm_lam_at_best = _pipe_pc_df.loc[
                                _pipe_pc_df["n_pcs"] == _pipe_k_mlm, "lambda_gc_MLM"
                            ]
                            if not _mlm_lam_at_best.empty:
                                _ax_pc.plot(_pipe_k_mlm, float(_mlm_lam_at_best.iloc[0]),
                                            "*", color="#0072B2", ms=14, zorder=5)
                            if ("MLMM (iterative cofactors)" in _pipe_models
                                    and _pipe_k_mlmm != _pipe_k_mlm):
                                _ax_pc.axvline(_pipe_k_mlmm, ls=":", color="#009E73", alpha=0.6, lw=1.5,
                                               label=f"MLMM best: {_pipe_k_mlmm}")
                            if "lambda_gc_MLMM" in _pipe_pc_df.columns:
                                _mlmm_lam_at_best = _pipe_pc_df.loc[
                                    _pipe_pc_df["n_pcs"] == _pipe_k_mlmm, "lambda_gc_MLMM"
                                ]
                                if (not _mlmm_lam_at_best.empty
                                        and np.isfinite(float(_mlmm_lam_at_best.iloc[0]))):
                                    _ax_pc.plot(_pipe_k_mlmm, float(_mlmm_lam_at_best.iloc[0]),
                                                "*", color="#009E73", ms=14, zorder=5)
                            if "FarmCPU (multi-locus)" in _pipe_models and _pipe_k_fc != _pipe_k_mlm:
                                _ax_pc.axvline(_pipe_k_fc, ls=":", color="#D55E00", alpha=0.6, lw=1.5,
                                               label=f"FarmCPU best: {_pipe_k_fc}")
                            if "lambda_gc_FarmCPU" in _pipe_pc_df.columns:
                                _fc_lam_at_best = _pipe_pc_df.loc[
                                    _pipe_pc_df["n_pcs"] == _pipe_k_fc, "lambda_gc_FarmCPU"
                                ]
                                if not _fc_lam_at_best.empty and np.isfinite(float(_fc_lam_at_best.iloc[0])):
                                    _ax_pc.plot(_pipe_k_fc, float(_fc_lam_at_best.iloc[0]),
                                                "*", color="#D55E00", ms=14, zorder=5)
                            _ax_pc.legend(fontsize=8)
                            _ax_pc.set_xlabel("Number of PCs")
                            _ax_pc.set_ylabel("\u03bbGC")
                            _ax_pc.set_title("PC Selection: \u03bbGC by PC Count (per model)")
                            _fig_pc.tight_layout()
                            _pipe_figures["PC_selection_lambda.png"] = _fig_pc
                            plt.close(_fig_pc)
                        except Exception:
                            logging.exception("PC selection lambda figure failed")
                else:
                    st.write("Skipping PC scan (no PCs available).")

                # Build per-model PC matrices
                def _slice_pcs(k):
                    if pcs_full_arr is None or k <= 0:
                        return None
                    k = min(k, _max_avail)
                    return pcs_full_arr[:, :k]

                _pcs_for = {
                    "MLM": _slice_pcs(_pipe_k_mlm),
                    "MLMM": _slice_pcs(_pipe_k_mlmm),
                    "FarmCPU": _slice_pcs(_pipe_k_fc),
                }

                _step = 2

                # Build pipeline-specific kinship kernels respecting the
                # local LOCO toggle. K_by_chr from session state was built
                # using the sidebar `use_loco`, so we may need to adjust it.
                if _pipe_use_loco == use_loco:
                    _pipe_K_by_chr = K_by_chr
                elif _pipe_use_loco:
                    # Sidebar had LOCO off but pipeline wants it on:
                    # rebuild real LOCO kernels from scratch.
                    from gwas.kinship import _build_loco_kernels_impl
                    _, _pipe_K_by_chr, _ = _build_loco_kernels_impl(
                        iid=iid, Z_grm=st.session_state["Z_grm"],
                        chroms_grm=st.session_state["chroms_grm"], K_base=K,
                    )
                else:
                    # Sidebar had LOCO on but pipeline wants whole-genome:
                    # override per-chromosome kernels with K0.
                    _pipe_K_by_chr = {ch: K0 for ch in K_by_chr}

                # --- Step 2: Core MLM GWAS ---
                _status.update(label=f"Step {_step}: Running MLM GWAS...")
                try:
                    _pipe_gwas = run_gwas_cached(
                        geno_imputed=geno_key, y=y,
                        pcs_full=pcs_full_arr, n_pcs=_pipe_k_mlm,
                        sid=sid, positions=positions, chroms=chroms,
                        chroms_num=chroms_num, iid=iid,
                        _K0=K0, _K_by_chr=_pipe_K_by_chr,
                        _pheno_reader_key=pheno_reader_key,
                        trait_name=trait_col,
                    )
                    _pipe_lam = compute_lambda_gc(_pipe_gwas["PValue"].values)
                except np.linalg.LinAlgError:
                    logging.exception("Core MLM GWAS failed (singular matrix)")
                    st.error(
                        "MLM GWAS failed: kinship matrix is singular. "
                        "Try fewer PCs, relax QC thresholds to retain more SNPs, "
                        "or disable LOCO kinship."
                    )
                    st.stop()
                except ValueError as e:
                    logging.exception("Core MLM GWAS failed (value error)")
                    _emsg = str(e).lower()
                    if "overlap" in _emsg or "sample" in _emsg:
                        st.error(
                            "MLM GWAS failed: sample IDs in VCF and phenotype file "
                            "do not match. Check that accession naming conventions "
                            "are consistent between files."
                        )
                    else:
                        st.error(f"MLM GWAS failed: {e}")
                    st.stop()
                except Exception as e:
                    logging.exception("Core MLM GWAS failed")
                    st.error(f"MLM GWAS failed: {e}")
                    st.stop()

                # OLS marginal effects
                try:
                    _mlm_pcs = _pcs_for["MLM"]
                    _pipe_covar = CovarData(iid=iid, val=_mlm_pcs) if _mlm_pcs is not None else None
                    _pipe_gwas = add_ols_effects_to_gwas(
                        _pipe_gwas, geno_imputed, y, _pipe_covar, sid,
                    )
                    st.write("OLS effects added (Beta, SE, t-stat).")
                except Exception as e:
                    logging.exception("Pipeline OLS effects failed")
                    st.write(f"OLS effects skipped: {e}")

                _pipe_gwas["-log10p"] = -np.log10(_pipe_gwas["PValue"].clip(1e-300))

                # FDR
                _rej, _fdr, _, _ = multipletests(_pipe_gwas["PValue"].values, method="fdr_bh")
                _pipe_gwas["FDR"] = _fdr
                _pipe_gwas["Significant_FDR"] = _rej

                # M_eff (only when user selected LD-aware threshold)
                _pipe_meff = None
                _pipe_meff_thresh = None
                if _pipe_sig_rule.startswith("M_eff"):
                    try:
                        _pipe_meff, _ = compute_meff_li_ji(geno_imputed)
                        _pipe_meff_thresh = 0.05 / _pipe_meff
                        st.session_state["meff_val"] = _pipe_meff
                        st.write(
                            f"M_eff = {_pipe_meff:,} independent tests "
                            f"(of {len(_pipe_gwas):,} SNPs)"
                        )
                    except Exception:
                        logging.exception("M_eff computation failed in pipeline")
                        st.warning("M_eff computation failed — falling back to Bonferroni.")
                        # leave _pipe_meff = None, _pipe_meff_thresh = None

                # Derive threshold variables from user's significance rule
                _n_tested = len(_pipe_gwas)
                if _pipe_sig_rule.startswith("M_eff"):
                    if _pipe_meff_thresh is not None:
                        _pipe_sig_thresh = _pipe_meff_thresh
                        _pipe_sig_lod = -np.log10(_pipe_sig_thresh)
                        _pipe_sig_label = f"M_eff (M={_pipe_meff:,})"
                    else:
                        # M_eff failed — fall back to Bonferroni
                        _pipe_sig_thresh = 0.05 / max(_n_tested, 1)
                        _pipe_sig_lod = -np.log10(_pipe_sig_thresh)
                        _pipe_sig_label = "Bonferroni (M_eff unavailable)"
                elif _pipe_sig_rule.startswith("Bonferroni"):
                    _pipe_sig_thresh = 0.05 / max(_n_tested, 1)
                    _pipe_sig_lod = -np.log10(_pipe_sig_thresh)
                    _pipe_sig_label = "Bonferroni"
                else:  # FDR
                    _pipe_sig_thresh = 0.05 / max(_n_tested, 1)  # fallback when FDR column missing
                    _pipe_sig_lod = None
                    _pipe_sig_label = "FDR q<0.05"

                # Add significance columns for available threshold types
                _pipe_gwas["Significant_Bonf"] = _pipe_gwas["PValue"] < (0.05 / max(_n_tested, 1))
                if _pipe_meff_thresh is not None:
                    _pipe_gwas["Significant_Meff"] = _pipe_gwas["PValue"] < _pipe_meff_thresh

                # Column name for the user's chosen significance rule
                if _pipe_sig_rule.startswith("M_eff") and _pipe_meff_thresh is not None:
                    _sig_col = "Significant_Meff"
                elif _pipe_sig_rule.startswith("Bonferroni"):
                    _sig_col = "Significant_Bonf"
                else:
                    _sig_col = "Significant_FDR"

                # Display key metrics
                if _pipe_meff is not None:
                    _m1, _m2, _m3, _m4 = st.columns(4)
                    _m4.metric("M_eff", f"{_pipe_meff:,}")
                else:
                    _m1, _m2, _m3 = st.columns(3)
                _m1.metric("λGC", f"{_pipe_lam:.3f}")
                _m2.metric(f"Significant ({_pipe_sig_label})", int(_pipe_gwas[_sig_col].sum()))
                _m3.metric("SNPs tested", f"{len(_pipe_gwas):,}")

                # Manhattan + QQ figures
                try:
                    _cumpos_df, _tick_pos, _tick_lab = compute_cumulative_positions(_pipe_gwas)
                    _fig_man = plot_manhattan_static(
                        _cumpos_df, active_lod=_pipe_sig_lod,
                        active_label=_pipe_sig_label,
                        title=f"Manhattan — {trait_col} (MLM)",
                    )
                    plt.figure(_fig_man.number)
                    plt.xticks(_tick_pos, _tick_lab, fontsize=8)
                    _pipe_figures[f"Manhattan_{trait_col}_MLM.png"] = _fig_man
                    plt.close(_fig_man)

                    # Interactive Manhattan for ZIP export
                    try:
                        _fig_man_int = plot_manhattan_interactive(
                            _cumpos_df, active_lod=_pipe_sig_lod,
                            active_label=_pipe_sig_label,
                            title=f"Interactive Manhattan — {trait_col} (MLM)",
                        )
                        _fig_man_int.update_layout(
                            xaxis=dict(tickmode="array", tickvals=_tick_pos, ticktext=_tick_lab),
                            height=500,
                        )
                        _pipe_figures[f"Interactive_Manhattan_{trait_col}_MLM.html"] = (
                            _fig_man_int.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
                        )
                    except Exception:
                        logging.exception("Pipeline MLM interactive Manhattan failed")

                    _fig_qq = plot_qq(_pipe_gwas["PValue"].values, lambda_gc_used=_pipe_lam)
                    _pipe_figures[f"QQ_{trait_col}_MLM.png"] = _fig_qq
                    plt.close(_fig_qq)

                    # PCA scatter plot (population structure)
                    if pcs_full_arr is not None and _pipe_k_mlm >= 2:
                        from gwas.plotting import plot_pca_scatter, plot_pca_scree
                        _fig_pca = plot_pca_scatter(
                            pcs_full_arr[:, :_pipe_k_mlm],
                            y=y.ravel(),
                            title=f"PCA — {trait_col}",
                            eigenvalues=pca_eigenvalues,
                        )
                        _pipe_figures[f"PCA_{trait_col}.png"] = _fig_pca
                        plt.close(_fig_pca)
                        # Scree plot
                        if pca_eigenvalues is not None:
                            _fig_scree = plot_pca_scree(
                                pca_eigenvalues, n_pcs_used=_pipe_k_mlm,
                                title=f"PCA Scree — {trait_col}",
                            )
                            _pipe_figures[f"PCA_scree_{trait_col}.png"] = _fig_scree
                            plt.close(_fig_scree)
                except Exception:
                    logging.exception("Pipeline figure generation failed")
                _step += 1

                # --- Step N: Multi-model (if selected) ---
                if "MLMM (iterative cofactors)" in _pipe_models:
                    _status.update(label=f"Step {_step}: Running MLMM...")
                    try:
                        _pr_key = put_object_in_session(
                            PhenoData(iid=iid, val=y),
                            "PHENO_READER", vcf_hash, pheno_hash, trait_col,
                        )
                        _mlmm_pcs = _pcs_for["MLMM"]
                        _cr = CovarData(iid=iid, val=_mlmm_pcs) if _mlmm_pcs is not None else None
                        _cr_key = put_object_in_session(
                            _cr, "COVAR_READER", vcf_hash, pheno_hash,
                        ) if _cr is not None else None
                        _mlmm_df, _cof_tbl = run_mlmm_core_cached(
                            geno_key=geno_key, y_key=y_key,
                            iid=iid, sid=sid, chroms=chroms,
                            chroms_num=chroms_num, positions=positions,
                            K0_key=K0_key, pheno_reader_key=_pr_key,
                            covar_reader_key=_cr_key, gwas_df_mlm=_pipe_gwas,
                            p_enter=_pipe_mlmm_p, max_cof=_pipe_mlmm_max_cof, window_kb=250,
                        )
                        _mlmm_df["PValue"] = _mlmm_df["PValue"].astype(float)
                        _rej_m, _fdr_m, _, _ = multipletests(
                            _mlmm_df["PValue"].clip(lower=1e-300).values, method="fdr_bh"
                        )
                        _mlmm_df["-log10p"] = -np.log10(_mlmm_df["PValue"].clip(lower=1e-300))
                        _mlmm_df["FDR"] = _fdr_m
                        _mlmm_df["Significant_FDR"] = _rej_m
                        _mlmm_df["Significant_Bonf"] = _mlmm_df["PValue"] < (0.05 / max(len(_mlmm_df), 1))
                        if _pipe_meff_thresh is not None:
                            _mlmm_df["Significant_Meff"] = _mlmm_df["PValue"] < _pipe_meff_thresh
                        _pipe_extra_dfs["MLMM"] = _mlmm_df
                        st.write(f"MLMM complete — {int(_rej_m.sum())} significant SNPs")
                        # Manhattan + QQ for MLMM
                        try:
                            _cumpos_m, _tp_m, _tl_m = compute_cumulative_positions(_mlmm_df)
                            _lam_m = compute_lambda_gc(_mlmm_df["PValue"].values)
                            _fig_man_m = plot_manhattan_static(
                                _cumpos_m, active_lod=_pipe_sig_lod,
                                active_label=_pipe_sig_label, title=f"Manhattan — {trait_col} (MLMM)",
                            )
                            plt.figure(_fig_man_m.number)
                            plt.xticks(_tp_m, _tl_m, fontsize=8)
                            _pipe_figures[f"Manhattan_{trait_col}_MLMM.png"] = _fig_man_m
                            plt.close(_fig_man_m)
                            # Interactive Manhattan for ZIP export
                            try:
                                _fig_man_m_int = plot_manhattan_interactive(
                                    _cumpos_m, active_lod=_pipe_sig_lod,
                                    active_label=_pipe_sig_label,
                                    title=f"Interactive Manhattan — {trait_col} (MLMM)",
                                )
                                _fig_man_m_int.update_layout(
                                    xaxis=dict(tickmode="array", tickvals=_tp_m, ticktext=_tl_m),
                                    height=500,
                                )
                                _pipe_figures[f"Interactive_Manhattan_{trait_col}_MLMM.html"] = (
                                    _fig_man_m_int.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
                                )
                            except Exception:
                                logging.exception("Pipeline MLMM interactive Manhattan failed")
                            _fig_qq_m = plot_qq(_mlmm_df["PValue"].values, lambda_gc_used=_lam_m)
                            _pipe_figures[f"QQ_{trait_col}_MLMM.png"] = _fig_qq_m
                            plt.close(_fig_qq_m)
                        except Exception:
                            logging.exception("MLMM figure generation failed")
                    except Exception as e:
                        logging.exception("Pipeline MLMM failed")
                        st.write(f"MLMM skipped: {e}")
                    _step += 1

                if "FarmCPU (multi-locus)" in _pipe_models:
                    _status.update(label=f"Step {_step}: Running FarmCPU...")
                    try:
                        _fc_pcs = _pcs_for["FarmCPU"]
                        _fc_df, _pqtn_tbl, _conv = run_farmcpu(
                            geno_imputed=geno_imputed, sid=sid,
                            chroms=chroms, chroms_num=chroms_num,
                            positions=positions, iid=iid,
                            pheno_reader=PhenoData(iid=iid, val=y),
                            K0=K0,
                            covar_reader=CovarData(iid=iid, val=_fc_pcs) if _fc_pcs is not None else None,
                            p_threshold=_pipe_fc_p, max_iterations=_pipe_fc_max_iter,
                            max_pseudo_qtns=_pipe_fc_max_pqtn,
                            final_scan=_pipe_fc_final_scan,
                            use_loco=_pipe_use_loco,
                        )
                        _fc_df["PValue"] = _fc_df["PValue"].astype(float)
                        _rej_fc, _fdr_fc, _, _ = multipletests(
                            _fc_df["PValue"].clip(lower=1e-300).values, method="fdr_bh"
                        )
                        _fc_df["-log10p"] = -np.log10(_fc_df["PValue"].clip(lower=1e-300))
                        _fc_df["FDR"] = _fdr_fc
                        _fc_df["Significant_FDR"] = _rej_fc
                        _fc_df["Significant_Bonf"] = _fc_df["PValue"] < (0.05 / max(len(_fc_df), 1))
                        if _pipe_meff_thresh is not None:
                            _fc_df["Significant_Meff"] = _fc_df["PValue"] < _pipe_meff_thresh
                        _pipe_extra_dfs["FarmCPU"] = _fc_df
                        st.write(f"FarmCPU complete — {int(_rej_fc.sum())} significant SNPs")
                        # Manhattan + QQ for FarmCPU
                        try:
                            _cumpos_fc, _tp_fc, _tl_fc = compute_cumulative_positions(_fc_df)
                            _lam_fc = compute_lambda_gc(_fc_df["PValue"].values)
                            _fig_man_fc = plot_manhattan_static(
                                _cumpos_fc, active_lod=_pipe_sig_lod,
                                active_label=_pipe_sig_label, title=f"Manhattan — {trait_col} (FarmCPU)",
                            )
                            plt.figure(_fig_man_fc.number)
                            plt.xticks(_tp_fc, _tl_fc, fontsize=8)
                            _pipe_figures[f"Manhattan_{trait_col}_FarmCPU.png"] = _fig_man_fc
                            plt.close(_fig_man_fc)
                            # Interactive Manhattan for ZIP export
                            try:
                                _fig_man_fc_int = plot_manhattan_interactive(
                                    _cumpos_fc, active_lod=_pipe_sig_lod,
                                    active_label=_pipe_sig_label,
                                    title=f"Interactive Manhattan — {trait_col} (FarmCPU)",
                                )
                                _fig_man_fc_int.update_layout(
                                    xaxis=dict(tickmode="array", tickvals=_tp_fc, ticktext=_tl_fc),
                                    height=500,
                                )
                                _pipe_figures[f"Interactive_Manhattan_{trait_col}_FarmCPU.html"] = (
                                    _fig_man_fc_int.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
                                )
                            except Exception:
                                logging.exception("Pipeline FarmCPU interactive Manhattan failed")
                            _fig_qq_fc = plot_qq(_fc_df["PValue"].values, lambda_gc_used=_lam_fc)
                            _pipe_figures[f"QQ_{trait_col}_FarmCPU.png"] = _fig_qq_fc
                            plt.close(_fig_qq_fc)
                        except Exception:
                            logging.exception("FarmCPU figure generation failed")
                    except Exception as e:
                        logging.exception("Pipeline FarmCPU failed")
                        st.write(f"FarmCPU skipped: {e}")
                    _step += 1

                # --- Step N: Cross-model consensus ---
                _model_results = [("MLM", _pipe_gwas)]
                for _mname in ["MLMM", "FarmCPU"]:
                    if _mname in _pipe_extra_dfs:
                        _model_results.append((_mname, _pipe_extra_dfs[_mname]))

                if len(_model_results) >= 2:
                    _status.update(label=f"Step {_step}: Cross-model consensus...")
                    try:
                        _sig_by_model = {}
                        for _mname, _mdf in _model_results:
                            if _pipe_sig_rule.startswith("FDR") and "FDR" in _mdf.columns:
                                _sig_by_model[_mname] = set(
                                    _mdf.loc[_mdf["FDR"] < 0.05, "SNP"]
                                )
                            else:
                                _sig_by_model[_mname] = set(
                                    _mdf.loc[_mdf["PValue"] < _pipe_sig_thresh, "SNP"]
                                )
                        _all_sig = set().union(*_sig_by_model.values())
                        if _all_sig:
                            _all_res = pd.concat(
                                [_df.assign(Model=_m) for _m, _df in _model_results],
                                ignore_index=True,
                            )
                            _consensus_rows = []
                            for _snp in _all_sig:
                                _det = [m for m, s in _sig_by_model.items() if _snp in s]
                                _best = _all_res.loc[
                                    _all_res["SNP"] == _snp
                                ].nsmallest(1, "PValue").iloc[0]
                                _crow = {
                                    "SNP": _snp, "Chr": _best["Chr"],
                                    "Pos": _best["Pos"],
                                    "Best_PValue": _best["PValue"],
                                    "Best_FDR": _best.get("FDR", np.nan),
                                    "Significant_Bonf": bool(_best.get("Significant_Bonf", False)),
                                    "Significant_FDR": bool(_best.get("Significant_FDR", False)),
                                    "Detected_by": ", ".join(_det),
                                    "N_models": len(_det),
                                }
                                if _pipe_meff_thresh is not None:
                                    _crow["Significant_Meff"] = bool(_best.get("Significant_Meff", False))
                                _consensus_rows.append(_crow)
                            _consensus_df = pd.DataFrame(_consensus_rows).sort_values(
                                ["N_models", "Best_PValue"], ascending=[False, True],
                            )
                            _n_high = int((_consensus_df["N_models"] >= 2).sum())
                            st.write(
                                f"Cross-model consensus: {_n_high}/{len(_consensus_df)} "
                                f"SNPs detected by 2+ models ({_pipe_sig_label})"
                            )
                            _pipe_extra_tables["CrossModel_Consensus.csv"] = _consensus_df
                        else:
                            st.write(f"No SNPs reached significance ({_pipe_sig_label}) in any model.")
                    except Exception as e:
                        logging.exception("Pipeline consensus failed")
                        st.write(f"Cross-model consensus skipped: {e}")
                    _step += 1

                # --- Step N: LD decay (accurate per-chromosome computation) ---
                _pipe_ld_decay_kb = None

                if _pipe_ld_flank is None:  # Auto mode
                    _status.update(label=f"Step {_step}: Computing LD decay (per-chromosome)...")
                    try:
                        _decay_df, _summary_df, _median_decay = _compute_ld_decay_for_gwas_page(
                            geno_key=geno_key,
                            chroms_tuple=tuple(np.asarray(chroms, dtype=str).tolist()),
                            positions_tuple=tuple(int(p) for p in positions),
                        )
                        if _median_decay is not None:
                            _pipe_ld_decay_kb = float(_median_decay)
                            _pipe_ld_flank = int(2 * _pipe_ld_decay_kb)
                            # Publish to session_state so LD Analysis page shows
                            # the accurate value (and the "from Decay tab" label).
                            st.session_state["ld_decay_df"] = _decay_df
                            st.session_state["ld_decay_summary"] = _summary_df
                            st.session_state["ld_decay_kb"] = _pipe_ld_decay_kb
                            st.session_state["ld_decay_computed"] = True
                            st.write(
                                f"LD decay (r² ≤ 0.2) ≈ {_pipe_ld_decay_kb:.0f} kb "
                                f"→ flank = {_pipe_ld_flank} kb"
                            )
                        else:
                            _pipe_ld_flank = 300
                            st.write("Could not compute LD decay — using 300 kb flank")
                    except Exception as e:
                        logging.exception("Pipeline LD decay computation failed")
                        _pipe_ld_flank = 300
                        st.write(f"LD decay computation failed: {e} — using 300 kb flank")
                    _step += 1

                # --- Load gene model + KEGG mapping once (shared across models) ---
                _pipe_sp_auto = _PIPE_SP.get(_pipe_species, {})
                if _pipe_genome_build and f"gene_model_{_pipe_genome_build}" in _pipe_sp_auto:
                    _pipe_gm_path = _pipe_sp_auto[f"gene_model_{_pipe_genome_build}"]
                    _pipe_desc_path = _pipe_sp_auto.get(f"gene_desc_{_pipe_genome_build}")
                else:
                    _pipe_gm_path = _pipe_sp_auto.get("gene_model")
                    _pipe_desc_path = _pipe_sp_auto.get("gene_desc")

                # User-uploaded overrides (apply for any species, including "Other")
                import tempfile as _tempfile
                from pathlib import Path as _PipePath
                _pipe_gm_up = st.session_state.get("pipe_gene_model_override")
                _pipe_desc_up = st.session_state.get("pipe_gene_desc_override")
                if _pipe_gm_up is not None:
                    _gm_tmp = os.path.join(_tempfile.gettempdir(), "trace_pipe_gene_model.csv")
                    with open(_gm_tmp, "wb") as _f:
                        _f.write(_pipe_gm_up.getbuffer())
                    _pipe_gm_path = _PipePath(_gm_tmp)
                if _pipe_desc_up is not None:
                    _desc_tmp = os.path.join(_tempfile.gettempdir(), "trace_pipe_gene_desc.txt")
                    with open(_desc_tmp, "wb") as _f:
                        _f.write(_pipe_desc_up.getbuffer())
                    _pipe_desc_path = _PipePath(_desc_tmp)

                _pipe_genes_df = None
                if _pipe_gm_path and _pipe_gm_path.exists():
                    try:
                        _pipe_genes_df = load_gene_annotation(
                            str(_pipe_gm_path),
                            str(_pipe_desc_path) if _pipe_desc_path and _pipe_desc_path.exists() else None,
                        )
                        st.write(f"Loaded {len(_pipe_genes_df):,} gene records.")
                    except Exception as e:
                        logging.exception("Gene model loading failed")
                        st.write(f"Gene model loading failed: {e}")
                elif _pipe_species == "Other (upload files)":
                    st.info(
                        "No custom gene model uploaded — gene annotation will be skipped. "
                        "Upload a gene coordinates CSV under 'Override gene files' "
                        "in the pipeline config to enable annotation."
                    )

                # --- Per-model post-GWAS analysis ---
                _post_gwas_models = [("MLM", _pipe_gwas)]
                for _mname in ["MLMM", "FarmCPU"]:
                    if _mname in _pipe_extra_dfs:
                        _post_gwas_models.append((_mname, _pipe_extra_dfs[_mname]))

                # Track MLM results for report backward compat
                _pipe_ld_blocks_mlm = None
                _pipe_ld_annotated_mlm = None
                _pipe_hap_gwas_mlm = None
                _geno_float = np.asarray(geno_imputed, float)
                _chr_arr = np.array([canon_chr(str(c)) for c in chroms])
                _pos_arr = np.array(positions)
                _sid_arr = np.array(sid)
                _per_model_post = {}
                _heatmap_blocks_done = set()

                for _model_name, _model_df in _post_gwas_models:
                    _status.update(label=f"Step {_step}: Post-GWAS ({_model_name})...")
                    st.write(f"**Post-GWAS analysis: {_model_name}**")

                    # Guard: ensure FDR column exists (compute BH-FDR)
                    if "FDR" not in _model_df.columns:
                        _model_df = _model_df.copy()
                        _rej_fb, _fdr_fb, _, _ = multipletests(
                            _model_df["PValue"].astype(float).clip(lower=1e-300).values,
                            method="fdr_bh",
                        )
                        _model_df["FDR"] = _fdr_fb
                        if "Significant_FDR" not in _model_df.columns:
                            _model_df["Significant_FDR"] = _rej_fb
                        st.write(f"  {_model_name}: FDR computed (was missing).")

                    _m_ld_blocks = None
                    _m_ld_annotated = None
                    _m_hap_gwas = None
                    _m_consolidated = None
                    # --- LD block detection ---
                    try:
                        _ld_suggestive = _pipe_ld_seed.startswith("Suggestive")
                        if _ld_suggestive:
                            _m_has_seeds = (_model_df["PValue"] < _pipe_ld_sig_p).any()
                        else:
                            _m_has_seeds = _model_df[_sig_col].any()
                        if _m_has_seeds:
                            _m_ld_blocks = ld.find_ld_clusters_genomewide(
                                gwas_df=_model_df,
                                chroms=chroms,
                                positions=positions,
                                geno_imputed=_geno_float,
                                sid=sid,
                                ld_threshold=_pipe_ld_r2,
                                flank_kb=_pipe_ld_flank,
                                ld_decay_kb=_pipe_ld_decay_kb,
                                min_snps=3,
                                top_n=_pipe_ld_top_n if _ld_suggestive else 0,
                                sig_thresh=_pipe_ld_sig_p if _ld_suggestive else _pipe_sig_thresh,
                            )
                            _n_before_filter = len(_m_ld_blocks)
                            _m_ld_blocks, _ = ld.filter_contained_blocks(
                                _m_ld_blocks, min_contained=2,
                                size_ratio_threshold=3.0, mode="remove",
                            )
                            if _n_before_filter > 0 and _m_ld_blocks.empty:
                                st.write(
                                    f"  {_model_name}: All {_n_before_filter} LD blocks "
                                    f"removed by containment filter."
                                )
                            _n_seeds = int((_model_df["PValue"] < _pipe_ld_sig_p).sum()) if _ld_suggestive else int(_model_df[_sig_col].sum())
                            st.write(
                                f"  {_model_name}: {len(_m_ld_blocks)} LD blocks "
                                f"around {_n_seeds} seed peaks"
                            )
                        else:
                            if _ld_suggestive:
                                st.write(f"  {_model_name}: No SNPs with p < {_pipe_ld_sig_p:.1e} — skipping LD blocks.")
                            else:
                                st.write(f"  {_model_name}: No significant SNPs — skipping LD blocks.")
                    except Exception as e:
                        logging.exception("Pipeline LD detection failed for %s", _model_name)
                        st.write(f"  {_model_name} LD detection skipped: {e}")

                    # --- Haplotype testing ---
                    if _m_ld_blocks is not None and not _m_ld_blocks.empty:
                        try:
                            _m_hap_gwas, _ = run_haplotype_block_gwas(
                                haplo_df=_m_ld_blocks,
                                chroms=chroms,
                                positions=positions,
                                geno_imputed=_geno_float,
                                sid=sid,
                                geno_df=geno_df,
                                pheno_df=pheno_clean,
                                trait_col=trait_col,
                                pcs=_pcs_for["MLM"],
                                n_perm=_pipe_hap_perms,
                                n_pcs_used=_pipe_k_mlm,
                            )
                            if _m_hap_gwas is not None and not _m_hap_gwas.empty:
                                _n_sig_hap = int((_m_hap_gwas.get("FDR_BH", pd.Series(dtype=float)) < 0.05).sum())
                                st.write(
                                    f"  {_model_name} haplotype: {_n_sig_hap} of "
                                    f"{len(_m_hap_gwas)} blocks significant"
                                )
                            else:
                                st.write(f"  {_model_name} haplotype: no results.")
                        except Exception as e:
                            logging.exception("Haplotype testing failed for %s", _model_name)
                            st.write(f"  {_model_name} haplotype skipped: {e}")

                    # --- Gene annotation ---
                    if _m_ld_blocks is not None and not _m_ld_blocks.empty and _pipe_genes_df is not None:
                        try:
                            _m_ld_annotated = annotate_ld_blocks(
                                _m_ld_blocks, _pipe_genes_df,
                                n_flank=2, max_flank_dist_bp=500_000,
                            )
                        except Exception as e:
                            logging.exception("Gene annotation failed for %s", _model_name)
                            st.write(f"  {_model_name} annotation skipped: {e}")

                    # --- Consolidate LD blocks + annotation + haplotype into one table ---
                    if _m_ld_blocks is not None and not _m_ld_blocks.empty:
                        try:
                            _m_consolidated = consolidate_ld_block_table(
                                _m_ld_blocks, _m_hap_gwas, _m_ld_annotated,
                            )
                            if _m_consolidated is not None and not _m_consolidated.empty:
                                _pipe_extra_tables[f"LD_blocks_annotated_{_model_name}.csv"] = _m_consolidated

                                # --- LD block summary narrative ---
                                if _model_name == "MLM":
                                    try:
                                        _nar_n = len(_m_consolidated)
                                        _nar_chrs = sorted(_m_consolidated["Chr"].astype(str).unique())
                                        _nar_chr_str = ", ".join(_nar_chrs)
                                        _nar_parts = [
                                            f"**{_nar_n} LD block(s)** detected on chromosome(s) {_nar_chr_str}."
                                        ]
                                        if "Hap_eta2" in _m_consolidated.columns:
                                            _lead_idx = _m_consolidated["Hap_eta2"].idxmax()
                                            _lead = _m_consolidated.loc[_lead_idx]
                                            _l_start = int(_lead.get("Start (bp)", 0)) / 1e6
                                            _l_end = int(_lead.get("End (bp)", 0)) / 1e6
                                            _l_eta = float(_lead["Hap_eta2"]) * 100
                                            _l_chr = _lead["Chr"]
                                            _nar_parts.append(
                                                f"The lead block (Chr{_l_chr}: {_l_start:.2f}\u2013{_l_end:.2f} Mb) "
                                                f"explains {_l_eta:.1f}% of phenotypic variance."
                                            )
                                        if "overlapping_genes" in _m_consolidated.columns:
                                            _n_genes = sum(
                                                len([g for g in str(v).split(";") if g.strip() and g.strip() != "nan"])
                                                for v in _m_consolidated["overlapping_genes"]
                                                if pd.notna(v)
                                            )
                                            if _n_genes:
                                                _nar_parts.append(f"{_n_genes} annotated gene(s) overlap these blocks.")
                                        st.info(" ".join(_nar_parts))
                                    except Exception:
                                        pass  # narrative is best-effort
                        except Exception:
                            logging.exception("LD block consolidation failed for %s", _model_name)

                    # --- LD heatmaps for significant blocks (deduplicated across models) ---
                    if _m_ld_blocks is not None and not _m_ld_blocks.empty:
                        try:
                            import seaborn as sns
                            _m_sig_snps_set = set(_model_df.loc[_model_df[_sig_col], "SNP"].astype(str))
                            _sig_blocks = []
                            for _, _blk in _m_ld_blocks.iterrows():
                                _lead_snps = str(_blk.get("Lead SNP", "")).split(";")
                                if any(s.strip() in _m_sig_snps_set for s in _lead_snps):
                                    _sig_blocks.append(_blk)
                            _sig_blocks_df = pd.DataFrame(_sig_blocks) if _sig_blocks else pd.DataFrame()

                            if not _sig_blocks_df.empty:
                                _n_heatmaps = 0
                                for _bi, _brow in _sig_blocks_df.iterrows():
                                    _bchr = canon_chr(str(_brow["Chr"]))
                                    _bstart = int(_brow.get("Start (bp)", _brow.get("Start", 0)))
                                    _bend = int(_brow.get("End (bp)", _brow.get("End", 0)))
                                    _block_key = (_bchr, _bstart, _bend)
                                    if _block_key in _heatmap_blocks_done:
                                        continue
                                    _heatmap_blocks_done.add(_block_key)
                                    _in_block = (
                                        (_chr_arr == _bchr)
                                        & (_pos_arr >= _bstart)
                                        & (_pos_arr <= _bend)
                                    )
                                    _block_idx = np.where(_in_block)[0]
                                    if len(_block_idx) < 2:
                                        continue
                                    _block_geno = _geno_float[:, _block_idx]
                                    _block_r2 = ld.pairwise_r2(_block_geno)
                                    _block_snp_labels = [
                                        f"{_chr_arr[i]}_{_pos_arr[i]}" for i in _block_idx
                                    ]
                                    _fig_hm, _ax_hm = plt.subplots(
                                        figsize=(max(4, len(_block_idx) * 0.3 + 1),
                                                 max(3, len(_block_idx) * 0.25 + 1))
                                    )
                                    _mask_upper = np.triu(np.ones_like(_block_r2, dtype=bool))
                                    sns.heatmap(
                                        _block_r2, mask=_mask_upper,
                                        cmap="YlOrBr", vmin=0, vmax=1,
                                        square=True, linewidths=0.5,
                                        xticklabels=_block_snp_labels,
                                        yticklabels=_block_snp_labels,
                                        cbar_kws={"label": "r²"},
                                        ax=_ax_hm,
                                    )
                                    _ax_hm.set_title(
                                        f"LD heatmap: Chr{_bchr} {_bstart:,}–{_bend:,}",
                                        fontsize=10,
                                    )
                                    _ax_hm.tick_params(labelsize=8)
                                    _fig_hm.tight_layout()
                                    _pipe_figures[f"LD_heatmap_Chr{_bchr}_{_bstart}_{_bend}.png"] = _fig_hm
                                    _n_heatmaps += 1
                                    plt.close(_fig_hm)
                                if _n_heatmaps:
                                    st.write(f"  {_n_heatmaps} LD heatmaps generated.")
                        except Exception as e:
                            logging.exception("LD heatmap generation failed for %s", _model_name)
                            st.write(f"  LD heatmaps skipped: {e}")

                    # Track MLM results for report backward compat
                    if _model_name == "MLM":
                        _pipe_ld_blocks_mlm = _m_ld_blocks
                        _pipe_ld_annotated_mlm = _m_consolidated if _m_consolidated is not None else _m_ld_annotated
                        _pipe_hap_gwas_mlm = _m_hap_gwas

                    # Collect per-model post-GWAS for report
                    _per_model_post[_model_name] = {
                        "ld_blocks_annotated_df": _m_consolidated if _m_consolidated is not None else _m_ld_annotated,
                        "haplotype_gwas_df": _m_hap_gwas,
                    }

                _step += 1

                # Expose pipeline tables to session state (used by VNN LD-informed masks)
                st.session_state["_pipe_extra_tables"] = _pipe_extra_tables

                # --- Optional subsampling stability screening ---
                if _pipe_run_subsampling:
                    _status.update(label=f"Step {_step}: Subsampling GWAS resampling...")
                    st.write("**Subsampling GWAS stability screening (MLM + GRM recomputation)**")
                    try:
                        import os as _os_boot
                        from gwas.subsampling import (
                            subsample_gwas_resampling,
                            aggregate_subsampling_to_ld_blocks,
                        )

                        _boot_n_jobs = min(4, _os_boot.cpu_count() or 1)

                        _Z_grm_boot = st.session_state.get("Z_grm", None)
                        if _Z_grm_boot is None:
                            raise RuntimeError("Z_grm not in session state")

                        _Z_grm_boot = np.asarray(_Z_grm_boot, dtype=np.float32)
                        if _Z_grm_boot.shape[0] != geno_imputed.shape[0]:
                            raise RuntimeError(
                                f"Z_grm rows ({_Z_grm_boot.shape[0]}) != "
                                f"genotype rows ({geno_imputed.shape[0]})"
                            )

                        _boot_parallel = _boot_n_jobs > 1
                        if _boot_parallel:
                            st.write(
                                f"  Running {int(_pipe_boot_n_reps)} subsampling iterations "
                                f"on {_boot_n_jobs} cores..."
                            )
                            _boot_cb = None
                        else:
                            _boot_pbar = st.progress(0)
                            def _boot_cb(rep, total):
                                _boot_pbar.progress((rep + 1) / total)

                        _boot_disc_df, _boot_raw_pvals, _boot_meta = subsample_gwas_resampling(
                            geno_imputed=geno_imputed,
                            y=y,
                            sid=sid,
                            chroms=chroms,
                            chroms_num=chroms_num,
                            positions=positions,
                            iid=iid,
                            Z_for_grm=_Z_grm_boot,
                            pcs_full=pcs_full_arr,
                            n_pcs=_pipe_k_mlm,
                            n_reps=int(_pipe_boot_n_reps),
                            sample_frac=float(_pipe_boot_frac),
                            discovery_thresh=float(_pipe_boot_thresh),
                            seed=int(random_seed),
                            progress_callback=_boot_cb,
                            n_jobs=_boot_n_jobs,
                            use_loco=_pipe_boot_loco,
                            chroms_grm=st.session_state.get("chroms_grm") if _pipe_boot_loco else None,
                        )

                        if not _boot_parallel:
                            _boot_pbar.empty()

                        # Per-iteration diagnostics
                        _boot_meta_df = pd.DataFrame(_boot_meta)
                        _boot_n_ok = int((_boot_meta_df["status"] == "ok").sum())

                        st.write(
                            f"  Subsampling complete: {_boot_n_ok}/{int(_pipe_boot_n_reps)} "
                            f"successful iterations."
                        )

                        _boot_lam_vals = _boot_meta_df.loc[
                            _boot_meta_df["status"] == "ok", "lambda_gc"
                        ].dropna()
                        if _boot_lam_vals.size > 0:
                            st.write(
                                f"  λGC across reps: "
                                f"median={_boot_lam_vals.median():.3f}, "
                                f"SD={_boot_lam_vals.std():.3f}"
                            )

                        # Add to ZIP tables
                        _pipe_extra_tables["Subsampling_SNP_stability.csv"] = _boot_disc_df
                        _pipe_extra_tables["Subsampling_rep_metadata.csv"] = _boot_meta_df

                        # Discovery frequency histogram
                        try:
                            from utils.pub_theme import PALETTE, SIG_LINE_COLOR, FIGSIZE
                            _fig_bh, _ax_bh = plt.subplots(
                                figsize=FIGSIZE.get("histogram", (7, 4))
                            )
                            _freq_vals = _boot_disc_df["DiscoveryFreq"].values
                            _ax_bh.hist(
                                _freq_vals[_freq_vals > 0], bins=30,
                                edgecolor="white", linewidth=0.5,
                                color=PALETTE["blue"],
                            )
                            _ax_bh.set_xlabel("Discovery frequency")
                            _ax_bh.set_ylabel("Number of SNPs")
                            _ax_bh.set_title(
                                f"Subsampling GWAS: SNP discovery frequency — {trait_col}"
                            )
                            _ax_bh.axvline(
                                0.5, color=SIG_LINE_COLOR, linestyle="--",
                                label="50% threshold",
                            )
                            _ax_bh.legend()
                            plt.tight_layout()
                            _pipe_figures[
                                f"Subsampling_discovery_freq_{trait_col}.png"
                            ] = _fig_bh
                            plt.close(_fig_bh)
                        except Exception:
                            logging.exception("Subsampling histogram failed")

                        # LD block aggregation (MLM blocks only)
                        if _pipe_ld_blocks_mlm is not None and not _pipe_ld_blocks_mlm.empty:
                            try:
                                _boot_block_stab = aggregate_subsampling_to_ld_blocks(
                                    discovery_df=_boot_disc_df,
                                    ld_blocks_df=_pipe_ld_blocks_mlm,
                                    raw_pvals=_boot_raw_pvals,
                                    sid=sid,
                                    chroms=chroms,
                                    positions=positions,
                                    discovery_thresh=float(_pipe_boot_thresh),
                                )
                                if not _boot_block_stab.empty:
                                    _pipe_extra_tables[
                                        "Subsampling_block_stability.csv"
                                    ] = _boot_block_stab
                                    _n_stable = int(
                                        (_boot_block_stab["BlockDiscoveryFreq"] >= 0.5).sum()
                                    )
                                    st.write(
                                        f"  Block stability: {_n_stable}/"
                                        f"{len(_boot_block_stab)} blocks with "
                                        f"BlockDiscoveryFreq ≥ 0.5"
                                    )
                                    st.session_state[
                                        "subsampling_block_stability"
                                    ] = _boot_block_stab
                            except Exception:
                                logging.exception("Subsampling block aggregation failed")
                                st.write("  Block aggregation skipped (error).")
                        else:
                            st.write("  No MLM LD blocks — skipping block aggregation.")

                        # Store in session state for downstream use
                        st.session_state["subsampling_gwas_summary"] = {
                            "n_reps": int(_pipe_boot_n_reps),
                            "n_successful": _boot_n_ok,
                            "sample_frac": float(_pipe_boot_frac),
                            "discovery_thresh": float(_pipe_boot_thresh),
                            "seed": int(random_seed),
                            "n_snps_freq_gt_50pct": int(
                                (_boot_disc_df["DiscoveryFreq"] > 0.5).sum()
                            ),
                            "n_snps_freq_gt_80pct": int(
                                (_boot_disc_df["DiscoveryFreq"] > 0.8).sum()
                            ),
                            "lambda_gc_median": (
                                float(_boot_lam_vals.median())
                                if _boot_lam_vals.size > 0
                                else None
                            ),
                        }

                        # Store subsampling cache in session state
                        st.session_state["boot_gwas_cache_pipeline"] = {
                            "disc_df": _boot_disc_df,
                            "raw_pvals": _boot_raw_pvals,
                            "meta": _boot_meta,
                        }

                        _pipe_boot_disc_df = _boot_disc_df

                    except Exception as _boot_err:
                        logging.exception("Pipeline subsampling GWAS failed")
                        st.write(f"  Subsampling skipped: {_boot_err}")
                    _step += 1

                # --- Step N: Generate report ---
                _status.update(label=f"Step {_step}: Generating report...")
                _pipe_meta = {
                    "Run date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Pipeline": "One-Click Full Analysis",
                    "Trait": trait_col,
                    "PCs_MLM": _pipe_k_mlm,
                    "PCs_MLMM": _pipe_k_mlmm if "MLMM (iterative cofactors)" in _pipe_models else "N/A",
                    "PCs_FarmCPU": _pipe_k_fc if "FarmCPU (multi-locus)" in _pipe_models else "N/A",
                    "Lambda_GC": round(_pipe_lam, 3),
                    "Samples": int(geno_df.shape[0]),
                    "SNPs": int(geno_df.shape[1]),
                    "Models": ", ".join(["MLM (FaST-MLM)"] + list(_pipe_models)),
                    "Significance_rule": _pipe_sig_rule,
                    "M_eff": int(_pipe_meff) if _pipe_meff is not None else "N/A",
                    "M_eff_threshold": f"{_pipe_meff_thresh:.2e}" if _pipe_meff_thresh is not None else "N/A",
                    "LD_decay_kb": round(_pipe_ld_decay_kb, 1) if _pipe_ld_decay_kb else "N/A",
                    "LD_flank_kb": _pipe_ld_flank,
                    **{f"LD_blocks_{mn}": len(tb) if tb is not None else 0
                       for mn, tb in [(mn, _pipe_extra_tables.get(f"LD_blocks_annotated_{mn}.csv"))
                                      for mn, _ in _post_gwas_models]},
                    **({f"Subsampling_{k}": v
                        for k, v in st.session_state["subsampling_gwas_summary"].items()}
                       if "subsampling_gwas_summary" in st.session_state else {}),
                }
                try:
                    _pipe_report_html = _gen_report(
                        trait_col=trait_col,
                        qc_snp=results.get("qc_snp"),
                        gwas_df=_pipe_gwas,
                        figures={k: v for k, v in _pipe_figures.items() if not k.startswith("LD_heatmap")},
                        metadata=_pipe_meta,
                        mlmm_df=_pipe_extra_dfs.get("MLMM"),
                        farmcpu_df=_pipe_extra_dfs.get("FarmCPU"),
                        ld_blocks_df=_pipe_ld_blocks_mlm,
                        lambda_gc=_pipe_lam,
                        n_samples=int(geno_df.shape[0]),
                        n_snps=int(geno_df.shape[1]),
                        info_field=results.get("info_field"),
                        pc_selection_df=_pipe_pc_df,
                        ld_blocks_annotated_df=_pipe_ld_annotated_mlm,
                        haplotype_gwas_df=_pipe_hap_gwas_mlm,
                        per_model_post_gwas=_per_model_post,
                        sig_label=_pipe_sig_label,
                        n_significant_override=int(_pipe_gwas[_sig_col].sum()),
                    )
                except Exception as e:
                    logging.exception("Report generation failed")
                    st.warning(f"HTML report generation failed: {e}")
                    _pipe_report_html = None
                _step += 1

                # --- Step N+1: Build ZIP ---
                _status.update(label=f"Step {_step}: Building ZIP...")
                try:
                    _pipe_zip_name, _pipe_zip_buf = _build_gwas_results_zip(
                        trait_col=trait_col,
                        gwas_df=_pipe_gwas,
                        figures_dict=_pipe_figures,
                        pheno_label=st.session_state.get("pheno_file_label"),
                        extra_model_dfs=_pipe_extra_dfs or None,
                        extra_tables=_pipe_extra_tables or None,
                        report_html=_pipe_report_html,
                    )
                    _pipe_zip_bytes = _pipe_zip_buf.getvalue()
                    _pipe_zip_bytes = _append_metadata_to_zip(_pipe_zip_bytes, _pipe_meta)
                except Exception as e:
                    logging.exception("ZIP building failed")
                    st.error(f"Failed to build results ZIP: {e}")
                    st.stop()

                _status.update(label="Full analysis complete!", state="complete")

            # --- Results summary card ---
            _n_ld_blks = (
                len(_pipe_ld_blocks_mlm)
                if _pipe_ld_blocks_mlm is not None and not _pipe_ld_blocks_mlm.empty
                else None
            )
            _render_results_summary_card(
                lambda_gc=_pipe_lam,
                n_significant=int(_pipe_gwas[_sig_col].sum()),
                sig_label=_pipe_sig_label,
                n_ld_blocks=_n_ld_blks,
                auto_pc_k=_pipe_k_mlm if _pipe_pc_mode.startswith("Auto") else None,
            )

            # --- Interpretation panel ---
            _render_interpretation_panel()

            # Single download button (HTML report is inside the ZIP)
            st.download_button(
                label="Download Full Results (ZIP)",
                data=_pipe_zip_bytes,
                file_name=_pipe_zip_name,
                mime="application/zip",
                key="pipe_zip_dl",
            )

            # ── Stop here — don't run the detailed single-trait view below ──
            st.stop()

    # ================================================================
    #                 SINGLE-TRAIT ANALYSIS (ORDERED)
    # ================================================================

    st.header("Single-trait GWAS (detailed view)")

    # --- Run GWAS button (prevents auto-trigger on every sidebar change) ---
    # Reset trigger when trait changes so user must re-click
    if st.session_state.get("_last_gwas_trait") != trait_col:
        st.session_state["gwas_triggered"] = False
    if st.button("Run GWAS", type="primary", help="Click to start the GWAS analysis with current parameters."):
        st.session_state["gwas_triggered"] = True
    if not st.session_state.get("gwas_triggered", False):
        st.stop()
    st.session_state["_last_gwas_trait"] = trait_col

    # ------------------------------------------------------------
    # Collect all figures for ZIP export (per trait)
    # ------------------------------------------------------------
    st.session_state["gwas_figures"] = {}

    # ============================================================
    # Core GWAS (needed BEFORE λGC and QQ)
    # ============================================================
    try:
        gwas_df = run_gwas_cached(
            geno_imputed=geno_key,
            y=y,
            pcs_full=pcs_full,
            n_pcs=n_pcs,
            sid=sid,
            positions=positions,
            chroms=chroms,
            chroms_num=chroms_num,
            iid=iid,
            _K0=K0,
            _K_by_chr=K_by_chr,
            _pheno_reader_key=pheno_reader_key,
            trait_name=trait_col,
        )
    except np.linalg.LinAlgError:
        logging.exception("MLM GWAS failed (singular matrix)")
        st.error(
            "MLM GWAS failed: kinship matrix is singular. "
            "Try fewer PCs, relax QC thresholds to retain more SNPs, "
            "or disable LOCO kinship."
        )
        st.stop()
    except ValueError as e:
        logging.exception("MLM GWAS failed (value error)")
        _emsg = str(e).lower()
        if "overlap" in _emsg or "sample" in _emsg:
            st.error(
                "MLM GWAS failed: sample IDs in VCF and phenotype file "
                "do not match. Check that accession naming conventions "
                "are consistent between files."
            )
        else:
            st.error(f"MLM GWAS failed: {e}")
        st.stop()
    except Exception as e:
        logging.exception("MLM GWAS failed")
        st.error(f"MLM GWAS failed: {e}")
        st.stop()
    # REQUIRED FOR LD PAGE TRAIT LOCKING
    st.session_state["trait_col_locked"] = trait_col

    # ============================================================
    # EXPORT GWAS RESULTS TO SESSION STATE (FOR LD PAGE)
    # ============================================================
    # QC
    unsorted = (gwas_df.sort_values(["ChrNum", "Pos"]).index != gwas_df.index).sum()
    if unsorted > 0:
        gwas_df = gwas_df.sort_values(["ChrNum", "Pos"]).reset_index(drop=True)

    gwas_df = gwas_df.dropna(subset=["Chr", "Pos", "PValue"])
    gwas_df["PValue"] = pd.to_numeric(
        gwas_df["PValue"],
        errors="coerce"
    ).astype(float)
    gwas_df["PValue"] = np.clip(gwas_df["PValue"].astype(float).values, 1e-300, 1.0)
    gwas_df["-log10p"] = -np.log10(gwas_df["PValue"])

    # Add OLS marginal effects (conditional on PCs only, no kinship)
    # These complement Beta_MLM from the mixed model
    try:
        gwas_df = add_ols_effects_to_gwas(
            gwas_df=gwas_df,
            geno_imputed=geno_imputed,
            y=y,
            covar_reader=covar_reader,
            sid=sid,
        )
    except Exception as _ols_err:
        logging.exception("OLS marginal effects computation failed")
        st.warning(f"OLS marginal effects unavailable: {_ols_err}")

    # Attach per-SNP imputation rate for QC transparency
    if "snp_imputation_rate" in results:
        imp_rate = pd.DataFrame({
            "SNP": sid.astype(str),
            "ImputationRate": results["snp_imputation_rate"],
        })
        gwas_df = gwas_df.merge(imp_rate, on="SNP", how="left")

    # Ensure users see both effect estimates clearly
    if "Beta_MLM" not in gwas_df.columns:
        st.warning(
            "FastLMM did not return SNP effect sizes (Beta_MLM). "
            "Only OLS marginal effects (Beta_OLS) are shown — these estimate "
            "per-allele effects but do **not** account for population structure (kinship). "
            "This can happen with very small sample sizes or singular kinship matrices."
        )

    # Multiple testing
    reject, p_fdr, _, _ = multipletests(gwas_df["PValue"].values, method="fdr_bh")
    bonf_thresh = 0.05 / len(gwas_df)

    gwas_df["FDR"] = p_fdr
    gwas_df["Significant_FDR"] = reject
    gwas_df["Significant_Bonf"] = gwas_df["PValue"] < bonf_thresh

    # M_eff (Li & Ji 2005) — computed once, cached by SNP count
    meff_val = None
    meff_thresh = None
    if sig_rule.startswith("M_eff"):
        with st.spinner("Computing M_eff (Li & Ji) — eigendecomposition of SNP correlations…"):
            try:
                meff_val, _meff_eigs = compute_meff_li_ji(geno_imputed)
                st.session_state["meff_val"] = meff_val
                meff_thresh = 0.05 / meff_val
                gwas_df["Significant_Meff"] = gwas_df["PValue"] < meff_thresh
            except Exception as e:
                logging.exception("M_eff computation failed")
                st.warning(f"M_eff computation failed ({e}); falling back to Bonferroni.")
                meff_val = len(gwas_df)
                meff_thresh = bonf_thresh

    # active threshold
    if sig_rule.startswith("FDR"):
        active_lod = None
        active_label = "FDR q<0.05"
        _active_sig_col = "Significant_FDR"
    elif sig_rule.startswith("M_eff") and meff_thresh is not None:
        active_lod = -np.log10(meff_thresh)
        active_label = f"M_eff Bonferroni (M={meff_val:,})"
        _active_sig_col = "Significant_Meff"
    else:
        active_lod = -np.log10(bonf_thresh)
        active_label = "Bonferroni α=0.05"
        _active_sig_col = "Significant_Bonf"

    # ============================================================
    # 2. PCA plot
    # ============================================================
    with st.expander("PCA: Population Structure", expanded=False):

        if pcs is not None and n_pcs >= 2:

            if (pcs.shape[0] != len(y)) or (pcs.shape[0] != geno_df.shape[0]):
                st.error(
                    f"Dimension mismatch: PCs have {pcs.shape[0]} rows, "
                    f"phenotype has {len(y)} samples, genotype has {geno_df.shape[0]} samples. "
                    "All three must match after sample alignment. "
                    "**How to fix:** Reload your data files (press F5) and ensure the same "
                    "sample IDs appear in both VCF and phenotype files."
                )
                st.stop()

            pc_df = pd.DataFrame(
                pcs[:, :2],
                columns=["PC1", "PC2"],
                index=geno_df.index
            )
            pc_df["Phenotype"] = y.flatten()

            # Compute % variance explained for axis labels
            _pc1_label, _pc2_label = "PC1", "PC2"
            if pca_eigenvalues is not None:
                _eig = np.asarray(pca_eigenvalues, dtype=float)
                _total = _eig.sum()
                if _total > 0:
                    _pc1_label = f"PC1 ({_eig[0] / _total * 100:.1f}%)"
                    _pc2_label = f"PC2 ({_eig[1] / _total * 100:.1f}%)"

            fig_pc = px.scatter(
                pc_df,
                x="PC1",
                y="PC2",
                color="Phenotype",
                color_continuous_scale="Viridis",
                hover_name=pc_df.index,
                title="PC1 vs PC2 colored by phenotype",
                labels={"PC1": _pc1_label, "PC2": _pc2_label},
            )
            fig_pc.update_traces(marker=dict(size=8))
            st.plotly_chart(fig_pc, use_container_width=True)
            st.session_state["gwas_figures"][f"PCA_{trait_col}.html"] = (
                fig_pc.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
            )

            download_plotly_fig(
                fig_pc,
                filename=f"PCA_{trait_col}.html",
                label="Download PCA plot (HTML)"
            )

            # PCA scree plot (eigenvalue decay)
            if pca_eigenvalues is not None:
                from gwas.plotting import plot_pca_scree
                _fig_scree = plot_pca_scree(pca_eigenvalues, n_pcs_used=n_pcs,
                                             title="PCA Scree Plot")
                st.pyplot(_fig_scree)
                download_matplotlib_fig(_fig_scree, filename=f"PCA_scree_{trait_col}.png",
                                         label="Download scree plot")
                plt.close(_fig_scree)

            pca_scores_df = pd.DataFrame(
                pcs_full,
                columns=[f"PC{i+1}" for i in range(pcs_full.shape[1])],
                index=geno_df.index,
            )
            pca_scores_df.index.name = "SampleID"

            st.download_button(
                "Download PCA scores (CSV)",
                pca_scores_df.to_csv().encode(),
                file_name=f"PCA_scores_{trait_col}.csv",
                mime="text/csv",
                key="dl_pca_scores",
            )

        else:
            st.write("PCA plot requires ≥2 PCs.")


    # Compute lambda_GC early (needed by manifest + MLM tab display)
    pvals_gc = (
        gwas_df
        .dropna(subset=["PValue"])
        .drop_duplicates(subset=["SNP"])
        ["PValue"]
        .astype(float)
        .values
    )
    lambda_gc = compute_lambda_gc(pvals_gc)

    # Run manifest (embedded in GWAS ZIP)
    pheno_label = st.session_state.get("pheno_file_label", "unknown_pheno")
    vcf_name = getattr(vcf_file, "name", "unknown_vcf")
    phe_name = getattr(phe_file, "name", "unknown_pheno_file")

    manifest = _make_run_manifest(
        trait_col=trait_col,
        pheno_label=pheno_label,
        vcf_name=vcf_name,
        phe_name=phe_name,
        maf_thresh=maf_thresh,
        mac_thresh=mac_thresh,
        miss_thresh=miss_thresh,
        ind_miss_thresh=ind_miss_thresh,
        drop_alt=drop_alt,
        norm_option=norm_option,
        n_pcs=n_pcs,
        sig_rule=sig_rule,
        lambda_gc=lambda_gc if "lambda_gc" in locals() else np.nan,
        n_samples=int(geno_df.shape[0]),
        n_snps=int(geno_df.shape[1]),
        qc_snp=results.get("qc_snp", {}) if isinstance(results, dict) else {},
        pheno_zero_handling=st.session_state.get("pheno_zero_handling", {}),
        pheno_transformations=st.session_state.get("pheno_transformations", {}),
        kinship_model=results.get("kinship_model", None) if isinstance(results, dict) else None,
        loco_fallback_by_chr=st.session_state.get("loco_diagnostics_by_chr", {}),
    )


    # ============================================================
    # Advanced: Stability Screening
    # ============================================================
    with st.expander("Advanced: Stability Screening", expanded=False):

        st.markdown("### Subsampling GWAS resampling (full MLM with GRM recomputation)")
        st.caption(
            "This is the gold-standard stability assessment: each iteration subsamples "
            "individuals, recomputes the GRM, and runs a full mixed-model scan. "
            "Per-SNP and per-LD-block discovery frequencies quantify which signals "
            "are robust to sample perturbation.\n\n"
            "**Note:** This is computationally expensive (~2-5 min per iteration "
            "depending on panel size). Start with 20-30 reps for exploration.\n\n"
            "Subsampling uses MLM (gold standard for stability). If FarmCPU is enabled, "
            "its stability is assessed via cross-model consensus instead."
        )

        run_subsampling = st.checkbox(
            "Run subsampling GWAS resampling (MLM + GRM recomputation)",
            value=False,
            key="run_subsampling_gwas",
        )

        if run_subsampling:
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                boot_n_reps = st.slider(
                    "Number of subsampling iterations",
                    min_value=10,
                    max_value=200,
                    value=50,
                    step=10,
                    key="boot_n_reps",
                )
            with col_b2:
                boot_sample_frac = st.slider(
                    "Sample fraction per iteration",
                    min_value=0.60,
                    max_value=0.95,
                    value=0.80,
                    step=0.05,
                    key="boot_sample_frac",
                )
            with col_b3:
                boot_disc_thresh = st.number_input(
                    "Discovery threshold (relaxed for subsamples)",
                    min_value=1e-8,
                    max_value=1e-2,
                    value=1e-4,
                    format="%.1e",
                    key="boot_disc_thresh",
                )

            col_b4, col_b5 = st.columns(2)
            with col_b4:
                boot_seed = st.number_input(
                    "Random seed (subsampling)",
                    min_value=0,
                    max_value=2**31 - 1,
                    value=random_seed,
                    step=1,
                    key="boot_seed",
                )
            with col_b5:
                import os
                _max_cores = os.cpu_count() or 1
                boot_n_jobs = st.number_input(
                    "CPU cores for subsampling",
                    min_value=1,
                    max_value=_max_cores,
                    value=min(4, _max_cores),
                    step=1,
                    key="boot_n_jobs",
                    help=(
                        f"Parallelize subsampling iterations across cores ({_max_cores} available). "
                        "Memory usage scales with number of cores."
                    ),
                )

            boot_use_loco = st.checkbox(
                "Use LOCO kinship in subsampling",
                value=False,
                key="boot_use_loco",
                help=(
                    "Rebuild per-chromosome LOCO kinship for each subsample. "
                    "More accurate but significantly slower (~12× per iteration)."
                ),
            )

            # Stable cache key for subsampling results
            boot_cache_key = (
                f"BOOT_GWAS::{trait_col}"
                f"::reps{boot_n_reps}::frac{boot_sample_frac:.2f}"
                f"::thresh{boot_disc_thresh:.1e}::seed{boot_seed}"
                f"::loco{boot_use_loco}"
                f"::n{geno_imputed.shape[0]}x{geno_imputed.shape[1]}"
            )

            if boot_cache_key not in st.session_state:

                # Validate Z_for_grm is available
                Z_grm_boot = st.session_state.get("Z_grm", None)
                if Z_grm_boot is None:
                    st.error(
                        "LD-pruned Z matrix (Z_grm) not found in session state. "
                        "This is computed during GWAS preprocessing — "
                        "please rerun the GWAS pipeline."
                    )
                    st.stop()

                Z_grm_boot = np.asarray(Z_grm_boot, dtype=np.float32)

                if Z_grm_boot.shape[0] != geno_imputed.shape[0]:
                    st.error(
                        f"Subsampling GRM matrix has {Z_grm_boot.shape[0]} samples but "
                        f"genotype matrix has {geno_imputed.shape[0]}. This happens when "
                        "the session data was modified between GWAS and subsampling steps. "
                        "**How to fix:** Refresh the page (F5) and re-run the full GWAS pipeline."
                    )
                    st.stop()

                progress_placeholder = st.empty()
                progress_bar = st.progress(0)

                _use_parallel = int(boot_n_jobs) > 1

                def _boot_progress(rep, total):
                    progress_bar.progress((rep + 1) / total)
                    progress_placeholder.info(
                        f"Subsampling GWAS iteration {rep + 1}/{total}…"
                    )

                if _use_parallel:
                    progress_placeholder.info(
                        f"Running {boot_n_reps} subsampling iterations "
                        f"on {int(boot_n_jobs)} cores…"
                    )

                with st.spinner("Running subsampling GWAS resampling…"):
                    boot_disc_df, boot_raw_pvals, boot_meta = subsample_gwas_resampling(
                        geno_imputed=geno_imputed,
                        y=y,
                        sid=sid,
                        chroms=chroms,
                        chroms_num=chroms_num,
                        positions=positions,
                        iid=iid,
                        Z_for_grm=Z_grm_boot,
                        pcs_full=pcs_full,
                        n_pcs=n_pcs,
                        n_reps=int(boot_n_reps),
                        sample_frac=float(boot_sample_frac),
                        discovery_thresh=float(boot_disc_thresh),
                        seed=int(boot_seed),
                        progress_callback=_boot_progress if not _use_parallel else None,
                        n_jobs=int(boot_n_jobs),
                        use_loco=boot_use_loco,
                        chroms_grm=st.session_state.get("chroms_grm") if boot_use_loco else None,
                    )

                progress_placeholder.success(
                    f"Subsampling GWAS complete: {boot_n_reps} iterations."
                )
                progress_bar.empty()

                st.session_state[boot_cache_key] = {
                    "disc_df": boot_disc_df,
                    "raw_pvals": boot_raw_pvals,
                    "meta": boot_meta,
                }

            else:
                boot_disc_df = st.session_state[boot_cache_key]["disc_df"]
                boot_raw_pvals = st.session_state[boot_cache_key]["raw_pvals"]
                boot_meta = st.session_state[boot_cache_key]["meta"]

            # ---- Display results ----

            # Per-iteration diagnostics
            meta_df = pd.DataFrame(boot_meta)
            n_ok = int((meta_df["status"] == "ok").sum())
            n_fail = int((meta_df["status"] != "ok").sum())

            st.write(f"**Successful iterations:** {n_ok} / {boot_n_reps}")
            if n_fail > 0:
                st.warning(f"{n_fail} iterations failed (see metadata).")

            # Lambda GC distribution across reps
            lam_vals = meta_df.loc[meta_df["status"] == "ok", "lambda_gc"].dropna()
            if lam_vals.size > 0:
                st.write(
                    f"**λGC across reps:** "
                    f"median={lam_vals.median():.3f}, "
                    f"mean={lam_vals.mean():.3f}, "
                    f"SD={lam_vals.std():.3f}"
                )

            # Top SNPs by discovery frequency
            st.markdown("#### Per-SNP discovery frequency (top 50)")
            st.dataframe(
                boot_disc_df.head(50)[
                    ["SNP", "Chr", "Pos", "DiscoveryFreq", "DiscoveryCount",
                     "MedianPValue", "MeanRank", "SD_log10P"]
                ],
                use_container_width=True,
            )

            # Discovery frequency histogram
            from utils.pub_theme import PALETTE, SIG_LINE_COLOR, FIGSIZE, export_matplotlib as _exp_mpl
            fig_boot_hist, ax_bh = plt.subplots(figsize=FIGSIZE["histogram"])
            freq_vals = boot_disc_df["DiscoveryFreq"].values
            ax_bh.hist(
                freq_vals[freq_vals > 0],
                bins=30,
                edgecolor="white",
                linewidth=0.5,
                color=PALETTE["blue"],
            )
            ax_bh.set_xlabel("Discovery frequency")
            ax_bh.set_ylabel("Number of SNPs")
            ax_bh.set_title("Subsampling GWAS: SNP discovery frequency distribution")
            ax_bh.axvline(0.5, color=SIG_LINE_COLOR, linestyle="--", label="50% threshold")
            ax_bh.legend()
            plt.tight_layout()
            st.pyplot(fig_boot_hist)
            _exp_mpl(fig_boot_hist, f"subsampling_discovery_freq_{trait_col}",
                      label_prefix="Download histogram")

            # Manhattan colored by discovery frequency
            st.markdown("#### Manhattan plot colored by subsampling stability")

            boot_manh = boot_disc_df.copy()
            boot_manh["-log10p"] = -np.log10(
                np.clip(boot_manh["MedianPValue"].values, 1e-300, 1.0)
            )
            boot_manh, tick_pos_b, tick_lab_b = compute_cumulative_positions(
                boot_manh, chr_col="Chr", pos_col="Pos"
            )

            fig_boot_manh = px.scatter(
                boot_manh,
                x="CumPos",
                y="-log10p",
                color="DiscoveryFreq",
                color_continuous_scale="RdYlBu",
                hover_data=["SNP", "Chr", "Pos", "DiscoveryFreq", "MedianPValue"],
                title=f"Subsampling stability Manhattan — {trait_col}",
                render_mode="webgl",
            )
            fig_boot_manh.update_layout(
                xaxis=dict(
                    tickmode="array",
                    tickvals=tick_pos_b,
                    ticktext=tick_lab_b,
                ),
                xaxis_title="Chromosome",
                yaxis_title="\u2212log<sub>10</sub>(median P)",
                coloraxis_colorbar_title="Discovery<br>frequency",
                height=500,
            )
            st.plotly_chart(fig_boot_manh, use_container_width=True)
            download_plotly_fig(fig_boot_manh, filename=f"Subsampling_Manhattan_{trait_col}.html", label="Download subsampling Manhattan")

            # ---- LD block aggregation ----
            st.markdown("#### Per-LD-block subsampling stability")

            haplo_df_for_agg = st.session_state.get("ld_blocks_final", None)
            if haplo_df_for_agg is None:
                haplo_df_for_agg = st.session_state.get("haplo_df_auto", None)
                if isinstance(haplo_df_for_agg, dict):
                    haplo_df_for_agg = haplo_df_for_agg.get(trait_col, None)

            with st.container():
                if haplo_df_for_agg is not None and not haplo_df_for_agg.empty:
                    block_stab = aggregate_subsampling_to_ld_blocks(
                        discovery_df=boot_disc_df,
                        ld_blocks_df=haplo_df_for_agg,
                        raw_pvals=boot_raw_pvals,
                        sid=sid,
                        chroms=chroms,
                        positions=positions,
                        discovery_thresh=float(boot_disc_thresh),
                    )

                    if not block_stab.empty:
                        st.dataframe(
                            block_stab.head(30),
                            use_container_width=True,
                        )

                        # Interpretation guide
                        n_stable_blocks = int((block_stab["BlockDiscoveryFreq"] >= 0.5).sum())
                        n_total_blocks = len(block_stab)
                        st.info(
                            f"**{n_stable_blocks} / {n_total_blocks}** LD blocks have "
                            f"BlockDiscoveryFreq ≥ 0.5 (discovered in ≥50% of subsampling iterations).\n\n"
                            "For your paper, report:\n"
                            "- Which blocks are consistently discovered (≥50%)\n"
                            "- Whether the lead SNP is stable (LeadSNP_DiscoveryFreq)\n"
                            "- Blocks with high BlockDiscoveryFreq but low LeadSNP_DiscoveryFreq "
                            "indicate the SIGNAL is stable but the specific lead SNP varies"
                        )

                        st.download_button(
                            "Download block-level subsampling stability (CSV)",
                            block_stab.to_csv(index=False).encode("utf-8"),
                            file_name=f"Subsampling_block_stability__{trait_col}.csv",
                            mime="text/csv",
                            key="dl_boot_block_stab",
                        )

                        # Store for other pages
                        st.session_state["subsampling_block_stability"] = block_stab
                    else:
                        st.warning("No LD blocks could be matched to subsampling SNP results.")
                else:
                    st.write(
                        "No LD blocks available for aggregation. "
                        "Run LD block detection on the LD page first."
                    )

            # ---- Downloads ----
            st.download_button(
                "Download per-SNP subsampling results (CSV)",
                boot_disc_df.to_csv(index=False).encode("utf-8"),
                file_name=f"Subsampling_snp_stability__{trait_col}.csv",
                mime="text/csv",
                key="dl_boot_snp",
            )

            st.download_button(
                "Download per-iteration metadata (CSV)",
                meta_df.to_csv(index=False).encode("utf-8"),
                file_name=f"Subsampling_rep_metadata__{trait_col}.csv",
                mime="text/csv",
                key="dl_boot_meta",
            )

            # Store in session for manifest
            st.session_state["subsampling_gwas_summary"] = {
                "n_reps": int(boot_n_reps),
                "n_successful": int(n_ok),
                "sample_frac": float(boot_sample_frac),
                "discovery_thresh": float(boot_disc_thresh),
                "seed": int(boot_seed),
                "n_snps_freq_gt_50pct": int((boot_disc_df["DiscoveryFreq"] > 0.5).sum()),
                "n_snps_freq_gt_80pct": int((boot_disc_df["DiscoveryFreq"] > 0.8).sum()),
                "lambda_gc_median": float(lam_vals.median()) if lam_vals.size > 0 else None,
            }

    # ============================================================
    # Build dynamic model result tabs
    # ============================================================
    _model_tab_names = ["MLM"]
    if "MLMM (iterative cofactors)" in model_choices:
        _model_tab_names.append("MLMM")
    if "FarmCPU (multi-locus)" in model_choices:
        _model_tab_names.append("FarmCPU")

    if len(_model_tab_names) > 1:
        _mtabs = st.tabs(_model_tab_names)
        _tab_map = dict(zip(_model_tab_names, _mtabs))
    else:
        from contextlib import nullcontext as _nullctx
        _tab_map = {n: _nullctx() for n in _model_tab_names}

    # ============================================================
    # 5–6. Manhattan plots
    # ============================================================
    with _tab_map["MLM"]:
        st.subheader("MLM results")

        # Static Manhattan (generated for ZIP export + download, not displayed inline)
        df_manh, tick_pos, tick_lab = compute_cumulative_positions(gwas_df)
        fig_manh = plot_manhattan_static(
            df_manh, active_lod, active_label, f"Manhattan — {trait_col} (MLM)"
        )
        plt.figure(fig_manh.number)
        plt.xticks(tick_pos, tick_lab, fontsize=8)
        st.session_state["gwas_figures"][f"Manhattan_MLM_{trait_col}.png"] = fig_manh

        # Interactive Manhattan
        st.write("### Manhattan Plot (MLM)")
        df_plot, tick_pos_i, tick_lab_i = compute_cumulative_positions(gwas_df)
        fig_int = plot_manhattan_interactive(
            df_plot, active_lod, active_label, f"Interactive Manhattan — {trait_col} (MLM)"
        )
        fig_int.update_layout(
            xaxis=dict(tickmode="array", tickvals=tick_pos_i, ticktext=tick_lab_i),
            height=500,
        )
        st.plotly_chart(fig_int, use_container_width=True)
        st.caption(_MANHATTAN_CAPTION)
        st.session_state["gwas_figures"][f"Interactive_Manhattan_MLM_{trait_col}.html"] = (
            fig_int.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
        )

        download_plotly_fig(
            fig_int,
            filename=f"Interactive_Manhattan_MLM_{trait_col}.html",
            label="Download interactive MLM Manhattan (HTML)"
        )
        download_matplotlib_fig(
            fig_manh,
            filename=f"Manhattan_MLM_{trait_col}.png",
            label="Download static MLM Manhattan (PNG)"
        )


        # ============================================================
        # λGC + QQ PLOT
        # ============================================================
        st.subheader("QQ Plot & genomic control")

        _gc_cols = st.columns(4)
        _gc_cols[0].metric("λGC (bulk 5–95%)", f"{lambda_gc:.3f}")
        chisq_all = stats.chi2.isf(np.clip(pvals_gc, 1e-300, 1.0), df=1)
        lambda_gc_standard = np.nanmedian(chisq_all) / 0.4549364
        _gc_cols[1].metric("λGC (standard)", f"{lambda_gc_standard:.3f}")
        _gc_cols[2].metric("SNPs tested", f"{len(gwas_df):,}")
        if meff_val is not None:
            _gc_cols[3].metric(
                "M_eff (Li & Ji)",
                f"{meff_val:,}",
                help=(
                    f"Effective number of independent tests (Li & Ji 2005). "
                    f"M_eff threshold = {meff_thresh:.2e}  "
                    f"(Bonferroni = {bonf_thresh:.2e}; "
                    f"M_eff is {len(gwas_df)/meff_val:.1f}× less conservative)."
                ),
            )
        else:
            _gc_cols[3].metric("Bonferroni threshold", f"{bonf_thresh:.2e}")

        fig_qq = plot_qq(gwas_df["PValue"].values, lambda_gc_used=lambda_gc)
        st.pyplot(fig_qq)
        st.session_state["gwas_figures"][f"QQ_{trait_col}.png"] = fig_qq

        download_matplotlib_fig(
            fig_qq,
            filename=f"QQ_{trait_col}.png",
            label="Download QQ plot (PNG)"
        )

        # --- Results summary card (single-trait) ---
        _st_n_sig = int(gwas_df[_active_sig_col].sum()) if _active_sig_col in gwas_df.columns else 0
        _render_results_summary_card(
            lambda_gc=lambda_gc,
            n_significant=_st_n_sig,
            sig_label=active_label,
            auto_pc_k=n_pcs,
        )

    # ============================================================
    # 6. Genome-wide LD DECAY — accurate per-chromosome computation
    # (same function as LD Analysis > Decay tab)
    try:
        _manual_decay_df, _manual_summary_df, _manual_median = _compute_ld_decay_for_gwas_page(
            geno_key=geno_key,
            chroms_tuple=tuple(np.asarray(chroms, dtype=str).tolist()),
            positions_tuple=tuple(int(p) for p in positions),
        )
        if _manual_median is not None:
            ld_decay_kb = float(_manual_median)
            st.session_state["ld_decay_df"] = _manual_decay_df
            st.session_state["ld_decay_summary"] = _manual_summary_df
            st.session_state["ld_decay_computed"] = True
        else:
            ld_decay_kb = float(st.session_state.get("ld_decay_kb", 200.0))
    except Exception:
        logging.exception("Manual path LD decay computation failed")
        ld_decay_kb = float(st.session_state.get("ld_decay_kb", 200.0))
    flank_kb = int(2 * ld_decay_kb)
    ld_block_max_dist_bp = int(ld_decay_kb * 1000)
    # Make MLM results available immediately for MLMM prefiltering
    st.session_state["gwas_df"] = gwas_df[["SNP", "PValue", "Chr", "Pos"]].copy()
    # ============================================================
    # 7. MLMM (optional, slow)
    # ============================================================
    gwas_mlm = gwas_df[["SNP", "Chr", "ChrNum", "Pos", "PValue"]].copy()
    gwas_mlm["Model"] = "MLM"
    results_all_models = [gwas_mlm]
    cofactor_logs = {}
    pheno_reader_key = put_object_in_session(
        pheno_reader, "PHENO_READER",
        vcf_hash, pheno_hash, trait_col
    )
    # MLM covar key (used by downstream helpers that reference the MLM PC set).
    covar_reader_key = put_object_in_session(
        covar_reader, "COVAR_READER",
        vcf_hash, pheno_hash, trait_col, int(n_pcs)
    )

    # --- Per-model covariate readers (manual Run GWAS path) ---
    # MLMM and FarmCPU may use different PC counts than MLM. Build dedicated
    # CovarData objects here so each model sees its own fixed-effect design.
    def _build_covar_for_model(k_pcs):
        if pcs_full is None or k_pcs <= 0:
            return None
        k = int(min(k_pcs, pcs_full.shape[1]))
        return CovarData(
            iid=iid,
            val=pcs_full[:, :k],
            names=[f"PC{i + 1}" for i in range(k)],
        )

    covar_reader_mlmm = _build_covar_for_model(int(n_pcs_mlmm))
    covar_reader_farmcpu = _build_covar_for_model(int(n_pcs_farmcpu))
    covar_reader_mlmm_key = put_object_in_session(
        covar_reader_mlmm, "COVAR_READER",
        vcf_hash, pheno_hash, trait_col, int(n_pcs_mlmm),
    )
    covar_reader_farmcpu_key = put_object_in_session(
        covar_reader_farmcpu, "COVAR_READER",
        vcf_hash, pheno_hash, trait_col, int(n_pcs_farmcpu),
    )

    if "MLMM (iterative cofactors)" in model_choices:
        with _tab_map["MLMM"]:
            _mlmm_ph = st.empty()
            _mlmm_ph.info("Computing MLMM (iterative cofactors)...")
        gwas_mlmm, cof_tbl = run_mlmm_core_cached(
            geno_key=geno_key,
            y_key=y_key,
            iid=iid,
            sid=sid,
            chroms=chroms,
            chroms_num=chroms_num,
            positions=positions,
            K0_key=K0_key,
            pheno_reader_key=pheno_reader_key,
            covar_reader_key=covar_reader_mlmm_key,
            gwas_df_mlm=gwas_df,
            p_enter=mlmm_p_enter,
            max_cof=mlmm_max_cof,
            window_kb=flank_kb,
        )
        # Enrich MLMM results to match MLM CSV format
        gwas_mlmm["PValue"] = np.clip(gwas_mlmm["PValue"].astype(float), 1e-300, 1.0)
        gwas_mlmm["-log10p"] = -np.log10(gwas_mlmm["PValue"])
        _chr_to_num = dict(zip(chroms.astype(str), chroms_num.astype(int)))
        gwas_mlmm["ChrNum"] = gwas_mlmm["Chr"].astype(str).map(_chr_to_num).fillna(0).astype(int)
        try:
            gwas_mlmm = add_ols_effects_to_gwas(gwas_mlmm, geno_imputed, y, covar_reader_mlmm, sid)
        except Exception:
            pass
        _rej_m, _fdr_m, _, _ = multipletests(gwas_mlmm["PValue"].values, method="fdr_bh")
        gwas_mlmm["FDR"] = _fdr_m
        gwas_mlmm["Significant_FDR"] = _rej_m
        gwas_mlmm["Significant_Bonf"] = gwas_mlmm["PValue"] < bonf_thresh
        if "snp_imputation_rate" in results:
            _imp = pd.DataFrame({"SNP": sid.astype(str), "ImputationRate": results["snp_imputation_rate"]})
            gwas_mlmm = gwas_mlmm.merge(_imp, on="SNP", how="left")

        results_all_models.append(gwas_mlmm)
        cofactor_logs["MLMM"] = cof_tbl
        with _tab_map["MLMM"]:
            _mlmm_ph.empty()
            st.subheader("MLMM (iterative cofactors)")
            st.write(f"MLMM finished with {len(cof_tbl)} selected cofactors.")

            # 7a. MLMM cofactors
            with st.expander("MLMM cofactors", expanded=False):
                st.dataframe(cof_tbl)

            # 7b. MLMM Manhattan (static — for export, not displayed inline)
            df_mlmm = gwas_mlmm.copy()
            df_mlmm["PValue"] = df_mlmm["PValue"].astype(float).clip(1e-300, 1.0)
            df_mlmm["-log10p"] = -np.log10(df_mlmm["PValue"])
            df_mlmm, tick_pos_m, tick_lab_m = compute_cumulative_positions(df_mlmm)

            fig_mlmm = plot_manhattan_static(
                df_mlmm,
                active_lod=active_lod,
                active_label=active_label,
                title=f"Manhattan — MLMM — {trait_col}"
            )
            plt.figure(fig_mlmm.number)
            plt.xticks(tick_pos_m, tick_lab_m, fontsize=8)
            st.session_state["gwas_figures"][f"Manhattan_MLMM_{trait_col}.png"] = fig_mlmm

            # 7c. MLMM Manhattan (interactive)
            st.write("### MLMM Manhattan Plot")
            fig_plotly_mlmm = plot_manhattan_interactive(
                df_mlmm,
                active_lod=active_lod,
                active_label=active_label,
                title=f"Interactive Manhattan — MLMM — {trait_col}"
            )
            fig_plotly_mlmm.update_layout(
                xaxis=dict(tickmode="array", tickvals=tick_pos_m, ticktext=tick_lab_m, showgrid=False),
                height=500,
            )
            st.plotly_chart(fig_plotly_mlmm, use_container_width=True)
            st.caption(_MANHATTAN_CAPTION)
            st.session_state["gwas_figures"][f"Interactive_Manhattan_MLMM_{trait_col}.html"] = (
                fig_plotly_mlmm.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
            )
            download_plotly_fig(fig_plotly_mlmm, filename=f"Interactive_Manhattan_MLMM_{trait_col}.html", label="Download interactive MLMM Manhattan")
            download_matplotlib_fig(
                fig_mlmm,
                filename=f"Manhattan_MLMM_{trait_col}.png",
                label="Download static MLMM Manhattan (PNG)"
            )

            # MLMM QQ plot & λGC
            st.write("### MLMM QQ Plot")
            lambda_gc_mlmm = compute_lambda_gc(gwas_mlmm["PValue"].values)
            fig_qq_mlmm = plot_qq(gwas_mlmm["PValue"].values, lambda_gc_used=lambda_gc_mlmm)
            st.write(f"**MLMM λGC (bulk 5–95%):** {lambda_gc_mlmm:.3f}")
            st.pyplot(fig_qq_mlmm)
            st.session_state["gwas_figures"][f"QQ_MLMM_{trait_col}.png"] = fig_qq_mlmm

    # ============================================================
    # 7d. FarmCPU (optional)
    # ============================================================
    if "FarmCPU (multi-locus)" in model_choices:
        with _tab_map["FarmCPU"]:
            _fc_ph = st.empty()
            _fc_ph.info("Computing FarmCPU (multi-locus)...")
        gwas_farmcpu, pqtn_tbl, conv_info = run_farmcpu_cached(
            geno_key=geno_key,
            y_key=y_key,
            iid=iid,
            sid=sid,
            chroms=chroms,
            chroms_num=chroms_num,
            positions=positions,
            K0_key=K0_key,
            pheno_reader_key=pheno_reader_key,
            covar_reader_key=covar_reader_farmcpu_key,
            p_threshold=farmcpu_p_threshold,
            max_iterations=farmcpu_max_iter,
            max_pseudo_qtns=farmcpu_max_pqtn,
            final_scan=farmcpu_final_scan,
            use_loco=use_loco,
        )
        # Enrich FarmCPU results to match MLM CSV format
        gwas_farmcpu["PValue"] = np.clip(gwas_farmcpu["PValue"].astype(float), 1e-300, 1.0)
        gwas_farmcpu["-log10p"] = -np.log10(gwas_farmcpu["PValue"])
        _chr_to_num_fc = dict(zip(chroms.astype(str), chroms_num.astype(int)))
        gwas_farmcpu["ChrNum"] = gwas_farmcpu["Chr"].astype(str).map(_chr_to_num_fc).fillna(0).astype(int)
        try:
            gwas_farmcpu = add_ols_effects_to_gwas(gwas_farmcpu, geno_imputed, y, covar_reader_farmcpu, sid)
        except Exception:
            pass
        _rej_fc, _fdr_fc, _, _ = multipletests(gwas_farmcpu["PValue"].values, method="fdr_bh")
        gwas_farmcpu["FDR"] = _fdr_fc
        gwas_farmcpu["Significant_FDR"] = _rej_fc
        gwas_farmcpu["Significant_Bonf"] = gwas_farmcpu["PValue"] < bonf_thresh
        if "snp_imputation_rate" in results:
            _imp_fc = pd.DataFrame({"SNP": sid.astype(str), "ImputationRate": results["snp_imputation_rate"]})
            gwas_farmcpu = gwas_farmcpu.merge(_imp_fc, on="SNP", how="left")

        results_all_models.append(gwas_farmcpu)
        cofactor_logs["FarmCPU"] = pqtn_tbl

        with _tab_map["FarmCPU"]:
            _fc_ph.empty()
            st.subheader("FarmCPU (multi-locus)")
            status_msg = "converged" if conv_info["converged"] else "reached max iterations"
            st.write(
                f"FarmCPU {status_msg}: {conv_info['n_pseudo_qtns']} pseudo-QTNs "
                f"in {conv_info['n_iterations']} iterations."
            )

            # Pseudo-QTN table
            with st.expander("FarmCPU pseudo-QTNs", expanded=False):
                st.dataframe(pqtn_tbl)

            # FarmCPU Manhattan (static — for export)
            df_fc = gwas_farmcpu.copy()
            df_fc["PValue"] = df_fc["PValue"].astype(float).clip(1e-300, 1.0)
            df_fc["-log10p"] = -np.log10(df_fc["PValue"])
            df_fc, tick_pos_fc, tick_lab_fc = compute_cumulative_positions(df_fc)

            fig_farmcpu = plot_manhattan_static(
                df_fc,
                active_lod=active_lod,
                active_label=active_label,
                title=f"Manhattan — FarmCPU — {trait_col}"
            )
            plt.figure(fig_farmcpu.number)
            plt.xticks(tick_pos_fc, tick_lab_fc, fontsize=8)
            st.session_state["gwas_figures"][f"Manhattan_FarmCPU_{trait_col}.png"] = fig_farmcpu

            # FarmCPU Manhattan (interactive)
            st.write("### FarmCPU Manhattan Plot")
            fig_plotly_fc = plot_manhattan_interactive(
                df_fc,
                active_lod=active_lod,
                active_label=active_label,
                title=f"Interactive Manhattan — FarmCPU — {trait_col}"
            )
            fig_plotly_fc.update_layout(
                xaxis=dict(tickmode="array", tickvals=tick_pos_fc, ticktext=tick_lab_fc, showgrid=False),
                height=500,
            )
            st.plotly_chart(fig_plotly_fc, use_container_width=True)
            st.caption(_MANHATTAN_CAPTION)
            st.session_state["gwas_figures"][f"Interactive_Manhattan_FarmCPU_{trait_col}.html"] = (
                fig_plotly_fc.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
            )
            download_plotly_fig(fig_plotly_fc, filename=f"Interactive_Manhattan_FarmCPU_{trait_col}.html", label="Download interactive FarmCPU Manhattan")
            download_matplotlib_fig(
                fig_farmcpu,
                filename=f"Manhattan_FarmCPU_{trait_col}.png",
                label="Download static FarmCPU Manhattan (PNG)"
            )

            # FarmCPU QQ plot & λGC
            st.write("### FarmCPU QQ Plot")
            lambda_gc_fc = compute_lambda_gc(gwas_farmcpu["PValue"].values)
            fig_qq_fc = plot_qq(gwas_farmcpu["PValue"].values, lambda_gc_used=lambda_gc_fc)
            st.write(f"**FarmCPU λGC (bulk 5–95%):** {lambda_gc_fc:.3f}")
            st.pyplot(fig_qq_fc)
            st.session_state["gwas_figures"][f"QQ_FarmCPU_{trait_col}.png"] = fig_qq_fc

    # ============================================================
    # 8. Cross-model comparison (if multiple models enabled)
    # ============================================================
    if len(results_all_models) > 1:
        with st.expander("Cross-model comparison", expanded=False):
            all_res = pd.concat(results_all_models, ignore_index=True)

            comb = []
            for m, sub in all_res.groupby("Model"):
                dfm = sub.copy()
                dfm["FDR"] = multipletests(dfm["PValue"].values, method="fdr_bh")[1]
                dfm["-log10p"] = -np.log10(np.clip(dfm["PValue"].astype(float), 1e-300, 1.0))
                comb.append(dfm)
            all_res = pd.concat(comb, ignore_index=True)

            # Model comparison summary table
            st.markdown("### Model Comparison Summary")
            _model_summary_rows = []
            for _m_name in all_res["Model"].unique():
                _m_sub = all_res[all_res["Model"] == _m_name]
                _m_pvals = _m_sub["PValue"].values
                _m_lgc = compute_lambda_gc(_m_pvals)
                _m_sig = int((_m_sub["FDR"] < 0.05).sum()) if "FDR" in _m_sub.columns else 0
                _m_top = _m_sub.sort_values("PValue").iloc[0]
                _model_summary_rows.append({
                    "Model": _m_name,
                    "\u03BBGC": round(_m_lgc, 3) if np.isfinite(_m_lgc) else "N/A",
                    "Sig SNPs (FDR<0.05)": _m_sig,
                    "Top SNP": _m_top["SNP"],
                    "Top P-value": f"{_m_top['PValue']:.2e}",
                })
            st.dataframe(pd.DataFrame(_model_summary_rows), hide_index=True, use_container_width=True)

            st.subheader("Cross-model results (first 30 per model)")

            for m in all_res["Model"].unique():
                st.markdown(f"**{m}**")
                sub = all_res[all_res["Model"] == m].copy()
                sub = sub.sort_values("PValue").head(30)
                sub["Significant (FDR<0.05)"] = sub["FDR"] < 0.05
                sub["PValue"] = sub["PValue"].round(6)
                sub["FDR (q-value)"] = sub["FDR"].round(6)

                st.dataframe(sub[["SNP", "Chr", "Pos", "PValue", "FDR (q-value)", "Significant (FDR<0.05)"]])

            st.download_button(
                "Download cross-model results (CSV)",
                all_res[["Model", "SNP", "Chr", "Pos", "PValue", "FDR"]].to_csv(index=False),
                file_name="CrossModel_Results.csv",
                mime="text/csv",
            )

            # Optional: overlay Manhattan by model (shared genome-wide x)
            st.markdown("### Overlay Manhattan by model (genome-wide x)")

            # Use helper to get cumulative positions per model together
            df_plot_multi = all_res.copy()
            chr_clean, chr_num_multi, order_labels_multi, order_map_multi = _clean_chr_series(df_plot_multi["Chr"])
            df_plot_multi["Chr"] = chr_clean
            df_plot_multi["ChrNum"] = chr_num_multi.astype(int)
            df_plot_multi["Pos"] = df_plot_multi["Pos"].astype(int)
            df_plot_multi = df_plot_multi.sort_values(["ChrNum", "Pos"]).reset_index(drop=True)

            gap = 1_000_000
            offset = 0
            tick_pos_multi, tick_lab_multi = [], []

            for chnum in np.sort(df_plot_multi["ChrNum"].unique()):
                mask = df_plot_multi["ChrNum"] == chnum
                pos_min = df_plot_multi.loc[mask, "Pos"].min()
                pos_max = df_plot_multi.loc[mask, "Pos"].max()
                df_plot_multi.loc[mask, "CumPos"] = df_plot_multi.loc[mask, "Pos"] + offset
                tick_pos_multi.append(offset + (pos_min + pos_max) / 2)
                tick_lab_multi.append(str(chnum))
                offset += (pos_max - pos_min) + gap


            _models_unique = df_plot_multi["Model"].unique().tolist()
            _model_color_map = {m: PALETTE_CYCLE[i % len(PALETTE_CYCLE)] for i, m in enumerate(_models_unique)}
            fig_overlay = px.scatter(
                df_plot_multi,
                x="CumPos", y="-log10p", color="Model",
                color_discrete_map=_model_color_map,
                hover_data=["SNP", "Chr", "Pos", "PValue"],
                title="Overlay Manhattan by model",
                render_mode="webgl"
            )
            fig_overlay.update_layout(
                xaxis_title="Chromosome (cumulative)",
                yaxis_title="-log10(p)",
                xaxis=dict(tickmode="array", tickvals=tick_pos_multi, ticktext=tick_lab_multi, showgrid=False),
                height=520,
            )
            st.plotly_chart(fig_overlay, use_container_width=True)
            st.caption(_MANHATTAN_CAPTION)
            download_plotly_fig(fig_overlay, filename=f"Overlay_Manhattan_{trait_col}.html", label="Download overlay Manhattan")

            # --- Cross-model consensus: which SNPs detected by which models? ---
            st.markdown("### Cross-model consensus")
            sig_by_model = {}
            for _df_m in results_all_models:
                _mname = _df_m["Model"].iloc[0]
                sig_snps = set(_df_m.loc[_df_m["PValue"] < bonf_thresh, "SNP"])
                sig_by_model[_mname] = sig_snps

            all_sig_snps = set().union(*sig_by_model.values())
            if all_sig_snps:
                consensus_rows = []
                for _snp in all_sig_snps:
                    detected_by = [m for m, s in sig_by_model.items() if _snp in s]
                    # Get best p-value and position from any model
                    best_row = all_res.loc[all_res["SNP"] == _snp].sort_values("PValue").iloc[0]
                    _crow = {
                        "SNP": _snp,
                        "Chr": best_row["Chr"],
                        "Pos": best_row["Pos"],
                        "Best_PValue": best_row["PValue"],
                        "Best_FDR": best_row.get("FDR", np.nan),
                        "Significant_Bonf": bool(best_row.get("Significant_Bonf", False)),
                        "Significant_FDR": bool(best_row.get("Significant_FDR", False)),
                        "Detected_by": ", ".join(detected_by),
                        "N_models": len(detected_by),
                    }
                    if "Significant_Meff" in best_row.index:
                        _crow["Significant_Meff"] = bool(best_row.get("Significant_Meff", False))
                    consensus_rows.append(_crow)
                consensus_df = pd.DataFrame(consensus_rows).sort_values(
                    ["N_models", "Best_PValue"], ascending=[False, True]
                )
                n_high = int((consensus_df["N_models"] >= 2).sum())
                st.caption(
                    f"{n_high} of {len(consensus_df)} significant SNPs detected by "
                    f"2+ models (high confidence)"
                )
                st.dataframe(consensus_df, use_container_width=True)
                st.download_button(
                    "Download consensus table (CSV)",
                    consensus_df.to_csv(index=False),
                    file_name=f"CrossModel_Consensus_{trait_col}.csv",
                    mime="text/csv",
                )
            else:
                st.write("No SNPs reached Bonferroni significance in any model.")

    # ============================================================
    # Understanding Your Results (interpretation panel)
    # ============================================================
    _render_interpretation_panel()

    # ============================================================
    # 9. Diagnostics
    # ============================================================
    with st.expander("Diagnostics", expanded=False):
        st.write("Phenotype mean±SD:", float(np.nanmean(y)), float(np.nanstd(y)))
        st.write("Kinship mean±SD:", float(np.nanmean(K)), float(np.nanstd(K)))
        st.write(
            "Genotype variance range:",
            float(np.nanmin(geno_imputed.var(axis=0))), "to",
            float(np.nanmax(geno_imputed.var(axis=0)))
        )
        if pcs is not None:
            corrs = np.corrcoef(np.c_[y, pcs], rowvar=False)[0, 1:]
            st.write("Corr(trait, PCs):", np.round(corrs, 3).tolist())
        st.write("P-value summary:")
        st.write(gwas_df[["PValue"]].describe())

        from utils.pub_theme import FIGSIZE as _FIG
        fig_hist, ax_hist = plt.subplots(figsize=_FIG["histogram"])
        ax_hist.hist(gwas_df["PValue"].astype(float), bins=50,
                     color=PALETTE["blue"], edgecolor="white", linewidth=0.5)
        ax_hist.set_xlabel("P-value")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title("Distribution of GWAS P-values")
        st.pyplot(fig_hist)
        export_matplotlib(fig_hist, f"pvalue_hist_{trait_col}",
                          label_prefix="Download p-value histogram")
        plt.close(fig_hist)

    # ============================================================
    # Save key objects for the LD / haplotype page (ROBUST)
    # ============================================================

    # --- Store all model results per trait ---
    if "gwas_results_by_model" not in st.session_state:
        st.session_state["gwas_results_by_model"] = {}
    st.session_state["gwas_results_by_model"][trait_col] = {}
    for _df_model in results_all_models:
        _model_name = _df_model["Model"].iloc[0]
        st.session_state["gwas_results_by_model"][trait_col][_model_name] = _df_model.copy()

    # --- Let user choose which model drives downstream analysis ---
    available_models = [df["Model"].iloc[0] for df in results_all_models]
    if len(available_models) > 1:
        downstream_model = st.radio(
            "Select model for LD & downstream analysis",
            available_models,
            index=0,
            help=(
                "LD blocks, haplotype analysis, and candidate gene search "
                "will use this model's p-values. All models remain available "
                "in the cross-model comparison above."
            ),
        )
    else:
        downstream_model = available_models[0]

    # Export selected model's results (backward-compatible)
    selected_df = st.session_state["gwas_results_by_model"][trait_col][downstream_model]
    if "gwas_results" not in st.session_state:
        st.session_state["gwas_results"] = {}
    st.session_state["gwas_results"][trait_col] = selected_df.copy()
    st.session_state["gwas_df"] = selected_df.copy()
    st.session_state["downstream_model"] = downstream_model

    # Trait pointers
    st.session_state["active_trait"] = trait_col
    st.session_state["trait_col"] = trait_col
    bump_data_version()  # GWAS complete — notify downstream pages
    # ============================================================
    # Auto-build GWAS-only ZIP (once per trait, no LD content)
    # ============================================================

    pheno_label = st.session_state.get("pheno_file_label", "unknown_pheno")

    # MLM always runs (foundational), so include it in the tag even though
    # it is no longer in the user-facing multiselect.
    _models_tag = "_".join(sorted(["MLM"] + [m.split()[0] for m in model_choices]))
    zip_key = f"gwas_zip_{pheno_label}__{trait_col}__{_models_tag}"

    if zip_key not in st.session_state:
        # Collect extra model results for ZIP
        _extra_dfs = {}
        if "MLMM (iterative cofactors)" in model_choices and "gwas_mlmm" in locals():
            _extra_dfs["MLMM"] = gwas_mlmm
        if "FarmCPU (multi-locus)" in model_choices and "gwas_farmcpu" in locals():
            _extra_dfs["FarmCPU"] = gwas_farmcpu

        zip_name, zip_buf = _build_gwas_results_zip(
            trait_col=trait_col,
            gwas_df=gwas_df,
            figures_dict=st.session_state.get("gwas_figures", {}),
            ld_blocks_df=None,
            hap_gwas_df=None,
            pheno_label=st.session_state.get("pheno_file_label"),
            extra_model_dfs=_extra_dfs or None,
        )

        st.session_state[zip_key] = {
            "name": zip_name,
            "data": zip_buf.getvalue(),
        }


    # --- Core genotype / phenotype objects (shared across traits)
    st.session_state["geno_imputed"] = geno_imputed
    # Store TRUE genotype sample order (CRITICAL for LD + haplotypes)
    st.session_state["geno_row_ids"] = np.asarray(geno_df.index, dtype=str).tolist()
    # Force numpy dtype on chroms — ArrowStringArray (newer pandas) cannot be
    # hashed by Streamlit @st.cache_data, breaking downstream cached wrappers.
    st.session_state["chroms"] = np.asarray(chroms, dtype=str)
    st.session_state["positions"] = positions
    st.session_state["sid"] = sid
    st.session_state["geno_df"] = geno_df
    st.session_state["pcs"] = pcs  # can be None

    # --- Optional LD hints (safe)
    if "ld_decay_kb" in locals():
        st.session_state["ld_decay_kb"] = float(ld_decay_kb)
    if "ld_block_max_dist_bp" in locals():
        st.session_state["ld_block_max_dist_bp"] = int(ld_block_max_dist_bp)

    # ============================================================
    # BLOCK 4 — REPRODUCIBILITY / METADATA (embedded in ZIP)
    # ============================================================

    meta = {
        "Run date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Trait analysed": trait_col,
        "Normalization method": norm_option,
        "MAF threshold": maf_thresh,
        "MAC threshold": mac_thresh,
        "Missingness threshold": miss_thresh,
        "Samples used": int(geno_df.shape[0]),
        "SNPs used": int(geno_df.shape[1]),
        "PCs used": int(n_pcs),
        "LOCO enabled": use_loco,
        "λGC (final model)": round(lambda_gc, 3) if "lambda_gc" in locals() else None,
        "Kinship mean ± SD": f"{np.nanmean(K):.4f} ± {np.nanstd(K):.4f}",
        "QC steps": {
            "Initial SNPs": int(results["n_snps_raw"]),
            "After QC SNPs": int(geno_df.shape[1]),
            "Final samples": int(geno_df.shape[0]),
        },
    }

    # Merge run manifest into metadata (both embedded in ZIP)
    if "manifest" in locals() and manifest is not None:
        meta["run_manifest"] = manifest


    # ============================================================
    # Generate HTML report
    # ============================================================
    from gwas.reports import generate_gwas_report

    _report_figures = {}
    for fig_key, fig_obj in st.session_state.get("gwas_figures", {}).items():
        _report_figures[fig_key] = fig_obj

    _n_sig_for_report = (
        int(gwas_df[_active_sig_col].sum())
        if _active_sig_col in gwas_df.columns else None
    )
    _report_html = generate_gwas_report(
        trait_col=trait_col,
        qc_snp=results.get("qc_snp"),
        gwas_df=gwas_df,
        figures=_report_figures,
        metadata=meta,
        mlmm_df=cofactor_logs.get("MLMM") if "cofactor_logs" in locals() else None,
        farmcpu_df=cofactor_logs.get("FarmCPU") if "cofactor_logs" in locals() else None,
        lambda_gc=lambda_gc if "lambda_gc" in locals() else None,
        n_samples=int(geno_df.shape[0]),
        n_snps=int(geno_df.shape[1]),
        info_field=results.get("info_field"),
        sig_label=active_label,
        n_significant_override=_n_sig_for_report,
    )

    st.download_button(
        label="Download Full Report (HTML)",
        data=_report_html.encode("utf-8"),
        file_name=f"GWAS_Report_{trait_col}_{datetime.now().strftime('%Y%m%d')}.html",
        mime="text/html",
    )

    # ============================================================
    # Download GWAS results (ZIP) with embedded metadata
    # ============================================================

    pheno_label = st.session_state.get("pheno_file_label", "unknown_pheno")
    # MLM always runs (foundational), so include it in the tag even though
    # it is no longer in the user-facing multiselect.
    _models_tag = "_".join(sorted(["MLM"] + [m.split()[0] for m in model_choices]))
    zip_key = f"gwas_zip_{pheno_label}__{trait_col}__{_models_tag}"

    if zip_key in st.session_state:

        # Retrieve original ZIP
        zip_bytes = st.session_state[zip_key]["data"]

        # Only append metadata once per run
        if not st.session_state.get(f"{zip_key}_meta_embedded", False):
            zip_bytes = _append_metadata_to_zip(zip_bytes, meta)
            st.session_state[zip_key]["data"] = zip_bytes
            st.session_state[f"{zip_key}_meta_embedded"] = True

        st.download_button(
            label="Download GWAS results (ZIP)",
            data=zip_bytes,
            file_name=st.session_state[zip_key]["name"],
            mime="application/zip",
        )

    # ============================================================
    # Cross-page navigation hints
    # ============================================================
    st.markdown("---")
    st.markdown("### Next steps")
    _nav_col1, _nav_col2 = st.columns(2)
    with _nav_col1:
        st.page_link("pages/LD_Analysis.py", label="LD & Haplotype Analysis →")




