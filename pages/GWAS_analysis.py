
import os
import logging
import plotly.express as px
import matplotlib.pyplot as plt
from utils.pub_theme import apply_matplotlib_theme, build_plotly_template, show_plot
from utils.plot_data import eigen_table, histogram_bins
apply_matplotlib_theme()
build_plotly_template()
from fastlmm.association import single_snp
from pysnptools.kernelreader import KernelData as PSKernelData
from pysnptools.snpreader import SnpData
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import json
from scipy import stats
from gwas.utils import (
    PhenoData, CovarData, hash_bytes, hash_df,
    put_array_in_session, put_kernel_in_session, put_object_in_session,
    stable_seed, bump_data_version,
)
from gwas.io import load_vcf_cached, _clean_chr_series
from gwas.qc import gwas_pipeline, allele_freq_from_called_dosage
from gwas.windows import ld_window_kb
from utils.pub_theme import plotly_html_bytes
from gwas.utils import _mean_impute_cols
from gwas.kinship import (
    build_loco_kernels_cached, compute_pcs_full_cached,
    _build_grm_from_Z, _standardize_geno_for_grm,
)
from gwas.models import (
    run_gwas_cached, run_mlmm_core_cached, run_farmcpu_cached,
    add_ols_effects_to_gwas,
    bic_proxy_from_design,
    build_mlmm_snpdata, standardize_genotypes_for_mlmm_cached,
    subset_snps_for_mlmm, _ols_fit, _nested_f_test, _one_hot_drop_first,
)
from gwas.subsampling import subsample_gwas_resampling, aggregate_subsampling_to_ld_blocks
from gwas.haplotype import (
    run_haplotype_block_gwas_cached,
    normalize_ld_blocks_schema, block_test_lm_with_pcs,
    freedman_lane_perm_pvalue,
)
from gwas.stability import _make_run_manifest
from gwas.significance import add_significance_columns
from gwas.plotting import (
    compute_cumulative_positions, plot_manhattan_static,
    plot_manhattan_interactive, plot_qq, compute_lambda_gc,
    download_matplotlib_fig, download_plotly_fig,
    _build_gwas_results_zip, _append_metadata_to_zip, _combine_gwas_zips,
    compute_meff_li_ji,
    PALETTE, PALETTE_CYCLE, export_matplotlib,
)
from annotation import (ASSEMBLY_NOTE, DESCRIPTIONS_FILE_HELP, GENE_FILE_HELP,
                        chromosomes_without_genes, compute_ld_decay_by_chromosome,
                        load_gene_model)

# -----------------
# Helper functions for user-friendly display
# -----------------
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
    "Significant_Custom": "Sig (custom)",
    "eta2": "Var Explained",
    "Hap_eta2": "Hap Var Explained",
}

# Internal columns to hide from user-facing tables
_COLUMNS_TO_HIDE = {
    "Nullh2", "Mixing", "sid_index", "GenDist",
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
    n_ld_blocks=None, pc_k=None,
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
            "Deflated genomic control: common for oligogenic traits "
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
            "Substantial inflation detected. Interpret results with caution; "
            "consider adding more PCs or checking for batch effects."
        )

    if pc_k is not None:
        lgc_note += f" The model included k = {pc_k} principal component(s)."

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

**What to do next:**

1. **Check LD blocks**: Significant SNPs often cluster into LD blocks on the same chromosome; these represent the same underlying QTL.
2. **Look at candidate genes**: The gene annotation identifies genes overlapping or flanking significant LD blocks.
3. **Cross-model consensus**: SNPs detected by multiple models (MLM + FarmCPU) are higher confidence.
4. **Subsampling stability**: If available, SNPs with a higher discovery frequency across subsampling resamples are more robust.
""")


@st.cache_data(show_spinner=False, max_entries=4)
def _gene_override_problem(data, chroms):
    """What is wrong with a One-Click gene file upload, or None: the loader's error (e.g.
    two columns for one field), or the genotype data's chromosomes it has no gene for."""
    import tempfile as _tf

    with _tf.TemporaryDirectory(prefix="trace_genecheck_") as d:
        path = os.path.join(d, "gene_model.csv")
        with open(path, "wb") as fh:
            fh.write(data)
        try:
            genes = load_gene_model(path)
        except Exception as e:
            return "error", f"This gene file cannot be used: {e}"
    missing = chromosomes_without_genes(list(chroms), genes) if chroms else []
    if missing:
        return "warning", (f"No genes for chromosome(s) {', '.join(missing)} of your genotype "
                           "data; check that the chromosome names match.")
    return None


@st.cache_data(show_spinner="Computing LD decay (per-chromosome)…", max_entries=8)
def _compute_ld_decay_for_gwas_page(geno_key: str, chroms_tuple: tuple,
                                    positions_tuple: tuple):
    """
    Accurate LD decay using compute_ld_decay_by_chromosome (the CLI's
    function; Post-GWAS shows the result in the LD Blocks & Haplotypes
    Settings). Returns (decay_df, summary_df,
    median_decay_kb). median_decay_kb may be None if no chromosome yielded
    a valid decay curve.

    Cached by geno_key + chroms + positions, stays hot across one-click
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
        max_snps_per_chr=1500,   # aligned to the CLI's per-chromosome subsample cap (cli.py, the `if len(pos_ch) > 1500`
        # in the "Estimating LD decay" loop, not ld_decay's MAX_LD_SNPS safety net)
        max_dist_kb=5000.0,
        n_bins=40,   # standardised onto the CLI ld_decay bin count (LD-decay Tier 1a)
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


# ---------------------------------------------------------------------------
# Session rehydration: re-display the persisted results of the last completed
# run (figures + table + downloads) instead of blanking the page when the user
# navigates back or an input gate would stop the run. Nothing is recomputed.
# ---------------------------------------------------------------------------
def _has_persisted_gwas():
    _g = st.session_state.get("gwas_df")
    return _g is not None and not (hasattr(_g, "empty") and _g.empty)


def _has_persisted_upload():
    """The raw genotype/phenotype inputs from a prior run are still in memory, so
    a re-run can proceed without re-uploading (Streamlit drops the uploaded-file
    buffers on page navigation)."""
    return ("_persist_vcf_bytes" in st.session_state
            and "_persist_pheno" in st.session_state)


def _covar_persist_decision(covar_file_present, running_from_memory, has_persisted_covar):
    """How to source the covariate when the uploader holds no file this rerun.

    "upload"  -> a file is present; load it.
    "session" -> run-from-memory rehydrate (the whole page is sourced from _persist_*);
                 reuse the persisted covariate.
    "clear"   -> live session with the uploader emptied; the user removed the covariate,
                 so drop the persisted one (nothing else ever clears it, so it would
                 otherwise apply silently to every later trait).
    "none"    -> no file and nothing persisted.
    Pure decision (no Streamlit / IO) so both branches are unit-testable without a
    browser; `running_from_memory` is `vcf_file is None` (the page's own read-back gate).
    """
    if covar_file_present:
        return "upload"
    if not has_persisted_covar:
        return "none"
    return "session" if running_from_memory else "clear"


def _render_last_run_results(stale=False):
    _g = st.session_state.get("gwas_df")
    _summ = st.session_state.get("gwas_run_summary", {})
    _figs = st.session_state.get("gwas_figures", {})
    if stale:
        st.warning(
            "Current settings differ from the run shown below. "
            "Click Run GWAS to apply them."
        )
    _trait = (_summ.get("trait") or st.session_state.get("active_trait")
              or st.session_state.get("trait_col", "?"))
    _bits = [f"trait **{_trait}**"]
    if _summ.get("n_samples"):
        _bits.append(f"{_summ['n_samples']} samples")
    if _summ.get("n_snps"):
        _bits.append(f"{_summ['n_snps']:,} SNPs")
    if _summ.get("models"):
        _bits.append(", ".join(_summ["models"]))
    if _summ.get("lambda_gc") is not None:
        _bits.append(f"λGC={_summ['lambda_gc']:.3f}")
    if _summ.get("timestamp"):
        _bits.append(_summ["timestamp"])
    st.info(
        "\U0001F4CB **Showing your last completed run** (restored from this session, "
        "not recomputed): " + " · ".join(_bits) +
        ".  Upload files and click **Run GWAS** below to run a new analysis."
    )
    for _name, _obj in _figs.items():
        try:
            if _name.endswith(".png"):
                if "QQ" in _name:
                    show_plot(_obj, "small")
                else:
                    show_plot(_obj, "wide")     # the Manhattan and scree plots
                st.caption("Data: the tables in \"Download GWAS results (ZIP)\" below.")
            elif _name.endswith(".html"):
                import streamlit.components.v1 as _components
                _html = _obj.decode("utf-8") if isinstance(_obj, (bytes, bytearray)) else str(_obj)
                _components.html(_html, height=520, scrolling=True)
        except Exception:
            pass
    if _g is not None:
        st.markdown("#### Results table")
        st.dataframe(_g, use_container_width=True)
        st.download_button(
            "Download results (CSV)", _g.to_csv(index=False).encode("utf-8"),
            file_name=f"GWAS_{_trait}.csv", mime="text/csv", key="rehydrate_csv",
        )
    # the ZIP of the run shown here (its session key is recorded with the run); before
    # runs recorded it, the first gwas_zip_ entry, which may be another trait's
    _zip_keys = ([_summ["zip_key"]] if _summ.get("zip_key") else
                 [k for k in list(st.session_state.keys())
                  if isinstance(k, str) and k.startswith("gwas_zip_")])
    for _zk in _zip_keys:
        _zv = st.session_state.get(_zk)
        if isinstance(_zv, dict) and "data" in _zv:
            st.download_button(
                "Download GWAS results (ZIP)", _zv["data"],
                file_name=_zv.get("name", "gwas_results.zip"),
                mime="application/zip", key="rehydrate_zip",
            )
            break


def _rehydrate_or_stop(msg=None, level="error"):
    """Show the persisted last-run results (if any) instead of blanking, then stop."""
    if _has_persisted_gwas():
        if msg:
            st.warning(f"{msg}; showing your last completed run below instead.")
        _render_last_run_results()
    elif msg:
        getattr(st, level, st.error)(msg)
    st.stop()


_CLEAR_PREFIXES = ("ARRKEY::", "KERNELKEY::", "OBJKEY::", "VCF_BYTES::", "gwas_zip_",
                   "BOOT_GWAS::")
_CLEAR_KEYS = (
    "_persist_vcf_bytes", "_persist_pheno", "_persist_is_gz", "_persist_covar",
    "_persist_vcf_name", "gwas_df", "gwas_results", "gwas_results_by_model", "gwas_figures",
    "gwas_run_summary", "gwas_triggered", "geno_imputed", "geno_dosage_raw",
    "geno_df", "K", "K0", "K_by_chr", "Z_grm", "chroms_grm", "pcs", "pcs_full",
    "pheno_reader", "covar_reader", "pheno_aligned", "pheno", "y", "iid",
    "sid", "chroms", "chroms_num", "positions", "GWAS_CTX", "pheno_raw",
    "pheno_applied_transforms", "_pheno_raw_fp", "subsampling_gwas_summary",
)


def _clear_session_data():
    """Release the genotype/phenotype held in memory and reset all run outputs."""
    for _k in list(st.session_state.keys()):
        if isinstance(_k, str) and (_k.startswith(_CLEAR_PREFIXES) or _k in _CLEAR_KEYS):
            st.session_state.pop(_k, None)
# =========================
# 1. Data upload & basic QC
# =========================
with st.container():
    col_files, col_qc = st.columns([2, 3])

    with col_files:
        vcf_file = st.file_uploader("Upload Genotype (VCF)", type=["vcf", "vcf.gz"], key="vcf_upload")
        if vcf_file is not None:
            _vcf_size_mb = vcf_file.size / (1024 * 1024)
            if _vcf_size_mb > 200:
                st.warning(
                    f"VCF file is {_vcf_size_mb:.0f} MB. Large files may cause slow "
                    "processing or memory issues. Consider pre-filtering to keep only "
                    "target chromosomes and MAF > 0.01 variants. Note: the genotype is "
                    "kept in session memory so runs can repeat without re-uploading; "
                    "use **Clear session data** (right) to release it."
                )
        phe_file = st.file_uploader("Upload Phenotype file (.csv or .txt)", type=["csv", "txt", "tsv"], key="pheno_upload")
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

        covar_file = st.file_uploader(
            "Upload covariates (optional; .csv/.tsv)",
            type=["csv", "txt", "tsv"], key="covar_upload",
            help="Numeric covariates (batch, environment, ...) added as fixed effects "
                 "to the single-trait models alongside the PCs. First column = sample "
                 "ID matching the VCF; remaining columns = numeric covariates. "
                 "Samples without a value for a covariate are dropped before QC, "
                 "kinship and the PCs, with a warning, as in the CLI's --covar.",
        )

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
        if st.button(
            "Clear session data",
            help="Release the genotype/phenotype held in memory and reset all results "
                 "(frees RAM; you will need to re-upload to run again).",
        ):
            _clear_session_data()
            st.success("Session data cleared.")
            st.rerun()

        st.markdown("**Genotype QC thresholds**")

        qc_preset = st.selectbox(
            "QC preset",
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

        impute_method = st.selectbox(
            "Imputation method", ["mean", "ldknni"], index=0,
            help=("How the GWAS matrix's missing calls are filled (mirrors CLI "
                  "--impute). 'mean' (default) = per-SNP mean; 'ldknni' = LD-kNNi "
                  "(Money et al. 2015), a discrete LD-weighted k-NN imputer. Does "
                  "not touch the raw dosage matrix used by LD/haplotype/QC."),
        )

        if impute_method == "ldknni":
            impute_k = st.number_input(
                "LD-kNNi neighbours (k)", min_value=1, max_value=50, value=5, step=1,
                key="impute_k",
                help=("How many nearest samples vote per missing call. 5 is the value "
                      "Money et al. 2015 chose for their apple dataset (Fig S1), not a "
                      "universal constant; it is near-optimal on both TRACE panels."),
            )
            impute_l = st.number_input(
                "LD-kNNi predictor SNPs (l)", min_value=2, max_value=200, value=20, step=1,
                key="impute_l",
                help=("How many top-r2 SNPs restrict the distance. 20 is the apple value "
                      "(Money et al. 2015); larger is NOT better -- on a sparse panel "
                      "(pepper) l=100 cost ~4 concordance points vs l=10, because "
                      "predictors beyond the strongest few are mostly noise."),
            )
        else:
            impute_k, impute_l = 5, 20

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
# MLM (FaST-MLM) is the foundational model and ALWAYS runs, its results
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
        "**FarmCPU** does iterative fixed-effect (OLS) scanning with "
        "REML-selected pseudo-QTNs (published algorithm, OLS final scan)."
    ),
)

# --- Per-model PC counts (grouped right under the model selector) ---
# MLM always runs, so its PC slider is always shown. The MLMM / FarmCPU PC sliders
# appear only when that model is selected in 'Additional GWAS models' above; each
# defaults to the MLM value.
n_pcs = st.sidebar.slider(
    "MLM PCs",
    min_value=0,
    max_value=20,
    value=0,
    step=1,
    key="n_pcs_mlm_slider",
    help=(
        "Number of principal components used as fixed-effect covariates "
        "in the MLM (FaST-MLM) model. MLMM and FarmCPU default to this "
        "value unless overridden below."
    ),
)
n_pcs_mlmm = n_pcs
n_pcs_farmcpu = n_pcs
if "MLMM (iterative cofactors)" in model_choices:
    n_pcs_mlmm = st.sidebar.slider(
        "MLMM PCs",
        min_value=0, max_value=20,
        value=int(n_pcs), step=1, key="n_pcs_mlmm_override",
        help="PC count used for MLMM covariates. Defaults to the MLM value.",
    )
if "FarmCPU (multi-locus)" in model_choices:
    n_pcs_farmcpu = st.sidebar.slider(
        "FarmCPU PCs",
        min_value=0, max_value=20,
        value=int(n_pcs), step=1, key="n_pcs_farmcpu_override",
        help="PC count used for FarmCPU covariates. Defaults to the MLM value.",
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
            help="Stop after this many scan / pseudo-QTN-selection cycles.",
        )
    else:
        farmcpu_max_iter = 10
    farmcpu_final_scan = "ols"  # D-103: FarmCPU MLM final scan frozen out (published OLS end-to-end)

# --- Significance rule ---
st.sidebar.subheader("Significance threshold")
sig_rule = st.sidebar.selectbox(
    "Significance threshold",
    ["FDR (q < 0.05)", "Bonferroni (α = 0.05)", "M_eff: Li & Ji (LD-aware Bonferroni)",
     "Custom p-value"],
    index=1,
    key="sig_rule_select",
    help=(
        "**M_eff (Li & Ji)**: divides α by the *effective* number of independent "
        "tests, so it is less stringent than Bonferroni; the reduction scales with "
        "how much LD is present.\n\n"
        "**Bonferroni**: Assumes all SNPs are independent; most conservative. "
        "May miss real signals in panels with extended LD.\n\n"
        "**FDR**: Controls the false discovery rate at 5%; least conservative. "
        "Good for exploratory screening when you expect many true associations.\n\n"
        "**Custom p-value**: flag SNPs below a fixed p-value you set (e.g. 5e-8).\n\n"
        "Full guidance: **Help → Interpreting Results**."
    ),
)
custom_sig_thresh = None
if sig_rule.startswith("Custom"):
    custom_sig_thresh = st.sidebar.number_input(
        "Custom p-value threshold",
        min_value=1e-300, max_value=1.0, value=5e-8, format="%.1e",
        key="custom_sig_thresh_input",
        help="SNPs with p < this value are flagged significant (e.g. 5e-8).",
    )
# propagate the reporting rule + custom threshold to downstream pages (LD tab)
st.session_state["sig_rule"] = sig_rule
st.session_state["custom_sig_thresh"] = custom_sig_thresh
show_secondary_threshold = st.sidebar.checkbox(
    "Also show secondary threshold line on Manhattans",
    value=False,
    help=(
        "When on, draws a second dotted grey line at whichever of "
        "Bonferroni or M_eff is NOT the primary rule above. Useful "
        "for figures where a peak lies between the two thresholds; "
        "common for LD-aware loci that clear M_eff but not naive "
        "Bonferroni. Has no effect when the primary rule is FDR "
        "(FDR is observation-dependent and is not drawn as a single "
        "−log10p line)."
    ),
    key="show_secondary_threshold_main",
)

# --- Subsampling ---
st.sidebar.subheader("Subsampling")
random_seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=2**31 - 1,
    value=42,
    step=1,
    help=(
        "Seed for the subsampling resampling step. Haplotype permutation "
        "p-values are seeded deterministically from each block's coordinates, "
        "and FarmCPU takes no seed -- this control affects neither. Set to a "
        "fixed value for reproducible subsampling across runs."
    ),
)


# =========================
# 2. Phenotype handling & transforms
# =========================
if (vcf_file and phe_file) or _has_persisted_upload():

    # --- Load phenotype file (with encoding fallbacks; from memory on re-run) ---
    if phe_file is not None:
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
                _rehydrate_or_stop(f"Could not read phenotype file: {_enc_err}")
        except Exception as _read_err:
            _rehydrate_or_stop(f"Could not parse phenotype file: {_read_err}")
        # persist the raw parsed phenotype so a re-run can proceed from memory
        st.session_state["_persist_pheno"] = pheno.copy()
    else:
        pheno = st.session_state["_persist_pheno"].copy()
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

    # --- Optional user covariates (parity with CLI --covar); resolved from the
    #     upload or from memory (run-from-memory), applied in the single-trait view ---
    from gwas.covariates import (
        load_covariate_frame as _load_cov,
        select_covariate_columns as _sel_cov,
        align_covariates as _align_cov,
        covariate_drop_message as _covariate_drop_message,
        CovariateDropError as _CovariateDropError,
    )
    covar_df = None
    _covar_source = None
    _covar_decision = _covar_persist_decision(
        covar_file is not None, vcf_file is None, "_persist_covar" in st.session_state
    )
    if _covar_decision == "upload":
        try:
            covar_file.seek(0)
            covar_df = _sel_cov(_load_cov(covar_file))
            st.session_state["_persist_covar"] = covar_df
            _covar_source = covar_file.name
        except Exception as _cov_err:
            _rehydrate_or_stop(f"Could not read covariate file: {_cov_err}")
    elif _covar_decision == "session":
        covar_df = st.session_state["_persist_covar"]
        _covar_source = "from session"
    elif _covar_decision == "clear":
        # Live session, uploader emptied -> user removed the covariate; drop the
        # persisted copy so it does not silently apply to every later trait.
        st.session_state.pop("_persist_covar", None)
    # Store RAW phenotype immediately (before zero-handling or transformations)
    # Always refresh pheno_raw when a new phenotype file is loaded
    st.session_state["pheno_raw"] = pheno.copy()
    # A new phenotype file clears the applied transformations (they were chosen for
    # the previous table); a rerun or a run from memory has the same content.
    from pages._pheno_qc.stats import SLUG_NAMES as _SLUG_NAMES
    _pheno_raw_fp = hash_df(pheno)
    _applied_before = st.session_state.get("pheno_applied_transforms") or {}
    if (st.session_state.get("_pheno_raw_fp") not in (None, _pheno_raw_fp)
            and _applied_before):
        st.session_state["pheno_applied_transforms"] = {}
        st.info(
            "A new phenotype file was loaded, so the applied transformations were cleared ("
            + "; ".join(f"{_t} -> {_SLUG_NAMES.get(_s, _s)}" for _t, _s in _applied_before.items())
            + "). Every trait now uses raw values."
        )
    st.session_state["_pheno_raw_fp"] = _pheno_raw_fp

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
    #   pheno_raw       : original upload (for fallback/reset)
    #   pheno_aligned   : GWAS QC-aligned subset (primary for downstream analysis)
    #   pheno_used_for_hap : column-standardized copy for LD/haplotype pages
    #   pheno_applied_transforms : {trait: CLI --norm slug}: Apply / Remove / Reset all
    #                       (2b); cleared on a new phenotype file and Clear session data

    # ------------------------------------------------
    # 2b. Phenotype QC and transformation
    # ------------------------------------------------
    # The panel's trait and method selectors drive only its preview. A transformation
    # is used only after Apply, which records it for that trait; the table keeps the
    # uploaded values, and each trait's slug is passed as that trait's normalisation,
    # applied where cli.py applies --norm (gwas.qc._pipeline_phenotype_qc, on the
    # trait's own analysis samples). Apply, Remove and Reset all are callbacks, which
    # run before this script, so the status line below is current on the click's rerun.
    from pages._pheno_qc import PhenoQCContext
    from pages._pheno_qc import apply as _pheno_apply
    from pages._pheno_qc import diagnostics as _pheno_diag
    from pages._pheno_qc.stats import applied_status_line as _applied_status_line
    _pheno_num_cols = pheno.select_dtypes(include=[np.number]).columns.tolist()
    st.caption(_applied_status_line(_pheno_apply.applied_transforms(), order=_pheno_num_cols))
    with st.expander("Phenotype QC and transformation", expanded=False):
        if not _pheno_num_cols:
            st.write("No numeric phenotype columns available for QC / transformation.")
        else:
            _pheno_ctx = PhenoQCContext(
                pheno_df=pheno, numeric_cols=_pheno_num_cols,
                id_col=str(pheno.index.name or "ID"), source="session")
            _qc_trait, _qc_method, _qc_transformed, _qc_ok = _pheno_diag.render(
                _pheno_ctx, embedded=True, key_prefix="gwas_pheno_qc")
            _pheno_apply.render(_pheno_ctx, _qc_trait, _qc_method, key_prefix="gwas_pheno_qc")
    # Each trait's normalisation for this run: its applied transformation, else none.
    _norm_by_trait = {_t: _s for _t, _s in _pheno_apply.applied_transforms().items()
                      if _t in _pheno_num_cols}

    def _norm_for(_trait):
        return _norm_by_trait.get(_trait, "none")

    # =========================
    # 3. Trait selection
    # =========================
    # The numeric columns are the traits (the table holds the uploaded values)
    numeric_traits = pheno.select_dtypes(include=[np.number]).columns.tolist()

    st.markdown("### Select traits for GWAS")

    # Initialize selection state with the first numeric trait on first load.
    _trait_key = "selected_traits_multiselect"
    if _trait_key not in st.session_state:
        st.session_state[_trait_key] = (
            [numeric_traits[0]] if numeric_traits else []
        )

    # Quick-select buttons for large phenotype tables (e.g. metabolomics
    # with thousands of columns). Clicking sets session state *before*
    # the multiselect renders, so the widget picks up the change in the
    # same rerun without needing st.rerun().
    _btn_col1, _btn_col2, _btn_col3 = st.columns([1, 1, 5])
    with _btn_col1:
        if st.button(
            f"Select all ({len(numeric_traits)})",
            use_container_width=True,
            disabled=not numeric_traits,
        ):
            st.session_state[_trait_key] = list(numeric_traits)
    with _btn_col2:
        if st.button(
            "Clear",
            use_container_width=True,
            disabled=not st.session_state[_trait_key],
        ):
            st.session_state[_trait_key] = []

    selected_traits = st.multiselect(
        "Select the numeric trait(s) for GWAS:",
        numeric_traits,
        key=_trait_key,
        help=(
            "One-Click Full Analysis runs every selected trait, one after another, "
            "each on its own phenotyped samples (the same per-trait pipeline as "
            "`cli.py --trait A B C`), and gives one combined ZIP with a folder per "
            "trait. The detailed view below analyses the first selected trait that "
            "passes phenotype QC."
        ),
    )
    _trait_caption = st.empty()  # filled once the detailed view's trait is known
    if len(selected_traits) > 10:
        st.warning(
            "More than 10 traits; consider the CLI batch workflow "
            "(`cli.py`) for production runs."
        )
    if not selected_traits:
        _rehydrate_or_stop("Select at least one numeric trait to proceed.", level="info")
    # Primary trait drives the pre-Run setup/preview (QC, PCA, data summary) and
    # the single-trait detailed view below the one-click expander. One-Click runs
    # every selected trait through gwas.pipeline with its own data.
    trait_col = selected_traits[0]

    # =========================
    # 4. GWAS preprocessing (VCF, QC, kinship, PCs)
    # =========================
    # P10/P11: unplaced/scaffold (ALT) markers are always excluded -- they carry no
    # valid genomic position, so no analysis can use them (and they must never enter
    # the significance divisor). Not a user choice.
    drop_alt = True

    # Load VCF bytes
    if vcf_file is not None:
        vcf_bytes = vcf_file.read()
        is_gz = vcf_file.name.endswith(".gz")
        # persist the raw genotype bytes so a re-run can proceed from memory
        st.session_state["_persist_vcf_bytes"] = vcf_bytes
        st.session_state["_persist_is_gz"] = is_gz
        st.session_state["_persist_vcf_name"] = vcf_file.name
    else:
        vcf_bytes = st.session_state["_persist_vcf_bytes"]
        is_gz = st.session_state.get("_persist_is_gz", False)

    # Phenotype normalisation: each trait's applied transformation from Section 2b
    # (else 'none'), passed to the pipeline as its --norm slug, which applies it as
    # cli.py --norm does. The phenotype table (and so pheno_hash) holds raw values.

    # --- Build stable cache keys ---
    vcf_hash = hash_bytes(vcf_bytes)
    # the raw table's fingerprint from the new-upload check above (the table has not
    # changed since: only an idempotent column-name strip ran in between)
    pheno_hash = _pheno_raw_fp

    # --- Store heavy inputs in session_state under those keys ---
    st.session_state[f"VCF_BYTES::{vcf_hash}"] = vcf_bytes
    st.session_state[f"PHENO_DF::{pheno_hash}"] = pheno.copy()
    # the covariate table: the preprocessing drops its incomplete samples, as the CLI
    _covar_hash = None
    if covar_df is not None:
        _covar_hash = hash_df(covar_df)
        st.session_state[f"COVAR_DF::{_covar_hash}"] = covar_df

    # --- Call cached preprocessing using hashes only ---
    # The setup (and the detailed view below) uses the first selected trait that
    # passes phenotype QC. A trait that does not (e.g. a constant phenotype) no
    # longer stops the page: One-Click still runs every selected trait and reports
    # that trait's own error.
    results = None
    _setup_failed = {}
    for _setup_trait in selected_traits:
        try:
            results = gwas_pipeline(
                vcf_hash=vcf_hash,
                pheno_hash=pheno_hash,
                trait_col=_setup_trait,
                maf_thresh=maf_thresh,
                miss_thresh=miss_thresh,
                norm_option=_norm_for(_setup_trait),
                drop_alt=drop_alt,
                ind_miss_thresh=ind_miss_thresh,
                mac_thresh=mac_thresh,
                info_thresh=info_thresh,
                impute_method=impute_method,
                impute_k=impute_k,
                impute_l=impute_l,
                covar_hash=_covar_hash,
            )
        except _CovariateDropError as _cov_err:
            # no sample keeps its covariate values: the same for every trait
            st.session_state.pop(f"VCF_BYTES::{vcf_hash}", None)
            _rehydrate_or_stop(str(_cov_err))
        except ValueError as _setup_err:
            # Skip only a phenotype-QC refusal, which is about this trait. Any other
            # error (e.g. unrecognised chromosome names) would fail the same way for
            # every trait, so raise it as before.
            _tb = _setup_err.__traceback__
            while _tb.tb_next is not None:
                _tb = _tb.tb_next
            if (_tb.tb_frame.f_code.co_name != "_pipeline_phenotype_qc"
                    # the per-sample missingness cut-off removes the same samples
                    # whichever trait is chosen
                    or "All samples removed by missingness filter" in str(_setup_err)):
                raise
            logging.info("Setup: trait %s fails phenotype QC: %s", _setup_trait, _setup_err)
            _setup_failed[_setup_trait] = str(_setup_err)
            continue
        trait_col = _setup_trait
        break
    if results is None:
        st.session_state.pop(f"VCF_BYTES::{vcf_hash}", None)
        _rehydrate_or_stop(
            "No selected trait can be analysed: "
            + "; ".join(f"{t}: {e}" for t, e in _setup_failed.items()))
    # the detailed view's trait: its normalisation keys the caches and the records below
    norm_option = _norm_for(trait_col)
    if _setup_failed:
        st.warning(
            "Not usable for the detailed view: "
            + "; ".join(f"`{t}` ({e})" for t, e in _setup_failed.items())
            + f". The detailed view uses **{trait_col}**; One-Click still runs every "
            "selected trait and reports each one's result."
        )
    if len(selected_traits) > 1:
        _trait_caption.caption(
            f"{len(selected_traits)} traits selected. One-Click Full Analysis runs "
            "all of them in sequence; the detailed view below analyses "
            f"**{trait_col}**."
        )

    # Free VCF bytes from session state, no longer needed after parsing
    st.session_state.pop(f"VCF_BYTES::{vcf_hash}", None)
    st.session_state["Z_grm"] = results["Z_for_pca"]
    # Force numpy dtype: chroms_grm may be ArrowStringArray on newer pandas,
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
    # Rebuild FastLMM readers (NOT cached, required)
    # ============================================================

    iid = results["iid"]
    K = results["K"]
    geno_df = results["geno_df"]
    pcs = results["pcs"]
    y = results["y"]

    # Aligned user covariates (iid order). The preprocessing (gwas_pipeline) has dropped
    # the samples without a covariate value, before SNP QC, kinship and the PCs, by the
    # CLI's helper at the CLI's stage; the page says how many, as a warning.
    user_covar_mat = None
    user_covar_names = None
    if covar_df is not None:
        _cov_M, _cov_names, _cov_ok = _align_cov(covar_df, iid[:, 0])
        if not _cov_ok.all():
            raise RuntimeError("Samples without a covariate value reached the scan; the "
                               "preprocessing should have dropped them.")
        user_covar_mat, user_covar_names = _cov_M, _cov_names
        if results.get("covar_dropped"):
            st.warning(_covariate_drop_message(results["covar_dropped"],
                                               results["covar_n_before"]))
    if user_covar_mat is not None:
        st.caption(
            f"Covariate: {_covar_source} · columns: {', '.join(user_covar_names)}"
        )
    # y depends on the trait's normalisation, which pheno_hash (raw table) does not
    # carry: it keys the string-keyed MLMM / FarmCPU caches
    y_key = put_array_in_session(
        np.asarray(y),
        "Y_VEC",
        vcf_hash,
        pheno_hash,
        trait_col,
        norm_option,
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
    # Force numpy dtype on chroms: ArrowStringArray (newer pandas) cannot be
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
        trait_col,
        norm_option,
    )

    # Covariates (PCs + optional user covariates)
    if user_covar_mat is None:
        if pcs is not None:
            covar_reader = CovarData(
                iid=iid,
                val=pcs,
                names=[f"PC{i + 1}" for i in range(pcs.shape[1])]
            )
        else:
            covar_reader = None
    else:
        _pc_names = [f"PC{i + 1}" for i in range(pcs.shape[1])] if pcs is not None else []
        _val = np.column_stack([pcs, user_covar_mat]) if pcs is not None else user_covar_mat
        covar_reader = CovarData(iid=iid, val=_val, names=_pc_names + list(user_covar_names))
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

        _qc_report = results.get("qc_report")

        if results.get("info_field"):
            st.info(f"Imputation quality field detected: **{results['info_field']}**")

        # Headline: the QC outcome at a glance, so a user learns it without opening
        # anything. "0 excluded" is load-bearing -- the report-only diagnostics
        # (heterozygosity, relatedness) flag samples but never remove them.
        _n_final = int(results["geno_df"].shape[1])
        _n_alt = int(_qc.get("ALT chromosomes", 0))
        _n_flag_het = int((_qc_report or {}).get("sample_het", {}).get("n_flagged", 0))
        st.markdown(
            f"**{_n_final:,} of {_n_pass:,} markers retained** · {_n_samples:,} samples · "
            f"{_n_flag_het} flagged for heterozygosity · 0 excluded"
        )

        with st.expander("Quality control report", expanded=False):
            from gwas.plotting import plot_maf_histogram
            from gwas import qc_report as _qcr

            # ── Markers: what QC removed ──
            st.markdown("**Markers**")
            if _qc_rows:
                _qc_table = "| Filter | SNPs removed | % of input |\n|--------|-------------|------------|\n"
                _qc_table += "\n".join(_qc_rows)
                st.markdown(_qc_table)
            if _n_alt > 0:
                st.caption(f"{_n_alt:,} markers on unplaced/scaffold sequences excluded "
                           "(no valid genomic position).")
            _geno_imp = results["geno_imputed"]
            _freq = _geno_imp.mean(axis=0) / 2.0
            _maf_vals = np.minimum(_freq, 1.0 - _freq)
            _fig_maf = plot_maf_histogram(_maf_vals, maf_threshold=maf_thresh)
            show_plot(_fig_maf, "small", data=histogram_bins(_fig_maf), data_name="MAF_histogram")
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

            # ── Samples / Sample pairs / Variant statistics / Trait: what QC
            #    flagged but kept (report-only), plus the trait distribution ──
            if _qc_report:
                st.markdown(_qcr.render_qc_report_markdown(_qc_report))
                for _qc_fn, _qc_fig in _qcr.qc_report_figures(_qc_report).items():
                    show_plot(_qc_fig, "wide",      # a 1 x 3 strip: unreadable when small
                              data=histogram_bins(_qc_fig),
                              data_name=str(_qc_fn).rsplit(".", 1)[0])
                    plt.close(_qc_fig)

        # ================================================================
        #       PC-selection diagnostics (report-only; never selects)
        # ================================================================

        with st.expander("PC-selection diagnostics", expanded=False):
            _Zdiag = st.session_state.get("Z_grm")
            if _Zdiag is None:
                st.info("Prepare/run the GWAS first to compute the PC spectrum.")
            else:
                try:
                    from gwas.pc_diagnostics import compute_pc_diagnostics
                    _diag = compute_pc_diagnostics(
                        _Zdiag, n=int(_Zdiag.shape[0]), m=int(_Zdiag.shape[1]),
                        prune_params={"r2": 0.2, "window_bp": 500_000, "step_bp": 100_000},
                        spectrum_depth=20,
                    )
                    _sp = _diag["spectrum"]
                    # Plot first: variance explained per PC (%) + cumulative on a secondary axis --
                    # what a user reads to choose a count (raw eigenvalues are scale-dependent; hidden below).
                    import plotly.graph_objects as go
                    _figd = go.Figure()
                    _figd.add_trace(go.Bar(x=_sp["rank"], y=_sp["pct_of_trace"],
                                           name="variance explained"))
                    _figd.add_trace(go.Scatter(x=_sp["rank"], y=_sp["cumulative_pct"],
                                               mode="lines+markers", name="cumulative", yaxis="y2"))
                    _figd.update_layout(
                        xaxis=dict(title="PC"),
                        yaxis=dict(title="variance explained (%)"),
                        yaxis2=dict(title="cumulative (%)", overlaying="y", side="right"),
                        height=280, margin=dict(t=30, b=30),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )
                    show_plot(_figd, "small",
                              data=_sp[["rank", "pct_of_trace", "cumulative_pct"]].rename(
                                  columns={"rank": "PC", "pct_of_trace": "Variance_explained_pct",
                                           "cumulative_pct": "Cumulative_pct"}),
                              data_name="PC_spectrum")
                    # Spectrum table: 10 rows, plain names, % variance + cumulative share.
                    _disp = _sp.head(10)[["rank", "pct_of_trace", "cumulative_pct"]].copy()
                    _disp.columns = ["PC", "Variance explained", "Cumulative"]
                    _disp["Variance explained"] = _disp["Variance explained"].map(lambda v: f"{v:.1f}%")
                    _disp["Cumulative"] = _disp["Cumulative"].map(lambda v: f"{v:.1f}%")
                    st.dataframe(_disp, use_container_width=True, hide_index=True)
                except Exception as _diag_err:
                    st.warning(f"PC diagnostics unavailable: {_diag_err}")


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
        _rehydrate_or_stop(
            f"Only {_n_geno} samples overlap between genotype and phenotype "
            f"({_n_pheno} phenotype samples). Check that sample IDs match between files."
        )
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
        "norm": norm_option,   # y's normalisation (the applied transformation, or none)
        "iid": st.session_state["iid"],
        "y": st.session_state["y"],
        "pcs": st.session_state["pcs"],
        "covar_reader": st.session_state["covar_reader"],
        # {SNP: (REF, ALT)}: the lead-SNP box plot names the genotypes (A/A, A/G, G/G)
        "allele_map": (results.get("allele_map") if isinstance(results, dict) else None),
    }

    # ================================================================
    #       ONE-CLICK FULL ANALYSIS PIPELINE
    # ================================================================

    with st.expander("One-Click Full Analysis", expanded=False):
        # --- Configuration panel ---
        _cfg_col1, _cfg_col2 = st.columns(2)
        with _cfg_col1:
            # MLM (FaST-MLM) is the foundational model and always runs, see
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
                    "Leave-One-Chromosome-Out kinship for the MLM scan. Uncheck "
                    "to use the whole-genome "
                    "GRM instead, useful when LOCO over-corrects on traits "
                    "with strong single-chromosome QTL. Overrides the "
                    "sidebar LOCO setting for this pipeline run only."
                ),
            )
            # PC count is a documented fixed default per model (no auto-selection);
            # the run report includes eigenvalue-spectrum + conventional-criteria diagnostics.
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
                        value=0, step=1, key=f"pipe_pcs_{_mname.lower()}",
                    )
            _pipe_sig_rule = st.selectbox(
                "Significance threshold",
                ["M_eff: Li & Ji (LD-aware Bonferroni)", "Bonferroni (α = 0.05)", "FDR (q < 0.05)"],
                index=1,
                key="pipe_sig_rule",
                help=(
                    "**M_eff (Li & Ji)**: divides α by the *effective* number of "
                    "independent tests, so it is less stringent than Bonferroni.\n\n"
                    "**Bonferroni**: Assumes all SNPs are independent; most conservative.\n\n"
                    "**FDR**: Controls false discovery rate at 5%; least conservative, "
                    "good for exploratory screening."
                ),
            )
            _pipe_show_secondary_threshold = st.checkbox(
                "Also show secondary threshold line on Manhattans",
                value=False,
                key="pipe_show_secondary_threshold",
                help=(
                    "When on, draws a second dotted grey line at whichever of "
                    "Bonferroni or M_eff is NOT the primary rule above. Useful "
                    "for figures where a peak lies between the two thresholds. "
                    "Has no effect when the primary rule is FDR."
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
                        "SL3 uses the default SL3.1 gene models; SL4 uses ITAG4.0/SL4 coordinates. "
                        "Choose the build of your VCF; for other builds, "
                        "such as Varitome's SL2.5, annotation is positional."
                    ),
                )
            else:
                _pipe_genome_build = None

            # Override gene files: mandatory for "Other (upload files)",
            # optional override for bundled species.
            _is_other_species = _pipe_species == "Other (upload files)"
            with st.expander(
                "Override gene files"
                + (" (required for 'Other')" if _is_other_species else ""),
                expanded=_is_other_species,
            ):
                _pipe_gm_up = st.file_uploader(
                    "Gene file (overrides the bundled gene model)",
                    type=["csv", "tsv", "txt"],
                    key="pipe_gene_model_override",
                    help=GENE_FILE_HELP,
                )
                st.file_uploader(
                    "Descriptions file (optional; overrides the bundled descriptions)",
                    type=["txt", "tsv"],
                    key="pipe_gene_desc_override",
                    help=DESCRIPTIONS_FILE_HELP,
                )
                st.caption(ASSEMBLY_NOTE)
                if _pipe_gm_up is not None:
                    # once the VCF is loaded, its chromosomes are checked against the file
                    _vcf_chroms = st.session_state.get("chroms")
                    _gm_problem = _gene_override_problem(
                        _pipe_gm_up.getvalue(),
                        tuple(sorted(set(np.asarray(_vcf_chroms).astype(str))))
                        if _vcf_chroms is not None else ())
                    if _gm_problem is not None:
                        (st.error if _gm_problem[0] == "error" else st.warning)(_gm_problem[1])
            _pipe_ld_r2 = st.number_input(
                "LD r² threshold", min_value=0.1, max_value=1.0,
                value=0.6, step=0.05, key="pipe_ld_r2",
            )
            _pipe_ld_seed = st.radio(
                "LD block seed SNPs",
                ["Suggestive (p-threshold + top N)", "Significant only (chosen threshold)"],
                key="pipe_ld_seed",
                horizontal=True,
                help="'Suggestive' finds LD blocks around top peaks (same as Post-GWAS Analysis page). "
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
            # "Significant only" keeps the CLI's seed-p / top-N values (see the run below).
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
            _pipe_generate_ld_heatmaps = st.checkbox(
                "Generate LD block heatmap PNGs",
                value=True,
                key="pipe_generate_ld_heatmaps",
                help=(
                    "Render an r² heatmap PNG for each LD block whose lead SNP is "
                    "significant in any model's scan (MLM, MLMM or FarmCPU), showing "
                    "the block with its neighbouring SNPs, and include them in the ZIP. "
                    "A block that several models report with the same bounds is drawn "
                    "once; blocks with different bounds each get their own PNG. When a "
                    "model other than MLM reports the block with a significant lead, or "
                    "calls its marked lead SNP, the title names the models that do. "
                    "Useful for visualising "
                    "haplotype structure but adds one PNG per block per "
                    "trait; turn off for batch runs over many traits "
                    "(e.g. metabolomics with hundreds-thousands of "
                    "phenotypes) to keep the ZIP small."
                ),
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
                _pipe_fc_max_iter = st.number_input(
                    "Max iterations", min_value=5, max_value=50,
                    value=10, step=5, key="pipe_fc_max_iter",
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
            # Every selected trait runs through gwas.pipeline.run_trait_pipeline, the
            # same per-trait code as `cli.py --trait A B C`, on its own phenotyped
            # samples: sample alignment, QC, imputation, kinship, PCs, models, LD
            # blocks, haplotype tests, figures, report and ZIP are all built from that
            # trait's data, and nothing is resolved from session keys (the reason
            # 822610f had to refuse multi-trait runs). Only the parsed VCF and the gene
            # models are shared. A trait that fails is reported and the others run.
            import tempfile as _tempfile
            import time as _time_mod

            from gwas.io import load_vcf_cached
            from gwas.pipeline import (
                ParsedGenotypes, cli_default_options, failed_traits_report, run_traits,
            )

            def _fmt_secs(_s):
                _s = max(0.0, float(_s))
                if _s < 60:
                    return f"{int(_s)}s"
                if _s < 3600:
                    return f"{int(_s // 60)}m {int(_s % 60)}s"
                return f"{int(_s // 3600)}h {int((_s % 3600) // 60)}m"

            def _last_line(_tb):
                _lines = [_l for _l in str(_tb).strip().splitlines() if _l.strip()]
                return _lines[-1] if _lines else "unknown error"

            # Trait names that differ only in letter case would share one folder of
            # the combined ZIP once it is unpacked on Windows or macOS.
            _case_seen = {}
            for _t in selected_traits:
                _case_seen.setdefault(_t.casefold(), []).append(_t)
            _case_clash = [v for v in _case_seen.values() if len(v) > 1]
            if _case_clash:
                st.error("Trait names that differ only in letter case cannot run together: "
                         + "; ".join(", ".join(f"`{t}`" for t in v) for v in _case_clash)
                         + ". Rename one of the columns.")
                st.stop()

            # Uploaded gene-model overrides go to a temporary folder for the run.
            _gm_tmp = _tempfile.TemporaryDirectory(prefix="trace_oneclick_")
            _gm_paths, _gm_names = {}, {}
            for _gm_key, _gm_fname in (("pipe_gene_model_override", "gene_model.csv"),
                                       ("pipe_gene_desc_override", "gene_descriptions.txt")):
                _gm_up = st.session_state.get(_gm_key)
                if _gm_up is not None:
                    _gm_p = os.path.join(_gm_tmp.name, _gm_fname)
                    with open(_gm_p, "wb") as _gm_fh:
                        _gm_fh.write(_gm_up.getvalue())
                    _gm_paths[_gm_key] = _gm_p
                    _gm_names[_gm_key] = str(getattr(_gm_up, "name", _gm_fname))

            # Seed settings: "Significant only" keeps the CLI's own seed-p / top-N
            # values (as `--ld-seed-mode significant` does); detection ignores them then.
            _ld_suggestive = _pipe_ld_seed.startswith("Suggestive")
            _ld_seed_opts = ({"ld_seed_p": float(_pipe_ld_sig_p), "ld_top_n": int(_pipe_ld_top_n)}
                             if _ld_suggestive else {})

            # The GUI's settings as the CLI's options; anything not named here keeps
            # the CLI default (gwas.pipeline.cli_default_options).
            _opts = cli_default_options(
                run_label="One-Click Full Analysis",
                vcf=(vcf_file.name if vcf_file is not None
                     else st.session_state.get("_persist_vcf_name", "genotypes.vcf")),
                pheno=f"{st.session_state.get('pheno_file_label', 'unknown_pheno')}.csv",
                output=None,
                model=(["mlm"]
                       + (["mlmm"] if "MLMM (iterative cofactors)" in _pipe_models else [])
                       + (["farmcpu"] if "FarmCPU (multi-locus)" in _pipe_models else [])),
                n_pcs=int(_pipe_manual_pcs.get("MLM", 0)),
                n_pcs_mlm=int(_pipe_manual_pcs.get("MLM", 0)),
                n_pcs_mlmm=int(_pipe_manual_pcs.get("MLMM", 0)),
                n_pcs_farmcpu=int(_pipe_manual_pcs.get("FarmCPU", 0)),
                no_loco=not _pipe_use_loco,
                sig_thresh=("meff" if _pipe_sig_rule.startswith("M_eff")
                            else "fdr" if _pipe_sig_rule.startswith("FDR") else "bonferroni"),
                secondary_threshold=bool(_pipe_show_secondary_threshold),
                species="custom" if _pipe_species == "Other (upload files)" else "tomato",
                genome_build=_pipe_genome_build or "SL3",
                gene_model=_gm_paths.get("pipe_gene_model_override"),
                gene_model_label=_gm_names.get("pipe_gene_model_override"),
                gene_desc=_gm_paths.get("pipe_gene_desc_override"),
                gene_desc_label=_gm_names.get("pipe_gene_desc_override"),
                ld_r2=float(_pipe_ld_r2),
                ld_seed_mode="suggestive" if _ld_suggestive else "significant",
                **_ld_seed_opts,
                ld_flank_kb=int(_pipe_ld_flank) if _pipe_ld_flank else None,
                hap_perms=int(_pipe_hap_perms),
                no_ld_heatmaps=not _pipe_generate_ld_heatmaps,
                mlmm_p_enter=float(_pipe_mlmm_p),
                mlmm_max_cof=int(_pipe_mlmm_max_cof),
                farmcpu_max_iter=int(_pipe_fc_max_iter),
                subsampling=bool(_pipe_run_subsampling),
                boot_reps=int(_pipe_boot_n_reps),
                boot_frac=float(_pipe_boot_frac),
                boot_thresh=float(_pipe_boot_thresh),
                boot_use_loco=bool(_pipe_boot_loco),
                boot_jobs=min(4, os.cpu_count() or 1),
                seed=int(random_seed),
                maf=float(maf_thresh),
                miss=float(miss_thresh),
                ind_miss=float(ind_miss_thresh),
                mac=int(mac_thresh),
                info_thresh=float(info_thresh),
                impute=impute_method,
                impute_k=int(impute_k),
                impute_l=int(impute_l),
                norm="none",  # every trait without an applied transformation...
                norm_by_trait=dict(_norm_by_trait),  # ...and the ones Apply set (Section 2b)
                covar=(_covar_source or "covariates") if covar_df is not None else None,
                covar_frame=covar_df,  # samples without a value are dropped, as in the CLI
            )

            _parsed = ParsedGenotypes.from_callset(load_vcf_cached(vcf_bytes))
            _n_total = len(selected_traits)
            _done, _failed, _elapsed = {}, {}, []
            _clock = {"t0": _time_mod.time(), "trait": ""}
            # Multi-trait progress widget, sits outside the status box so
            # it stays visible as each trait completes.
            _trait_progress = st.empty()

            with st.status("Running full analysis...", expanded=True) as _status:
                def _on_trait_start(_i, _t):
                    _clock["t0"], _clock["trait"] = _time_mod.time(), _t
                    if _elapsed:
                        _eta = sum(_elapsed) / len(_elapsed) * (_n_total - _i + 1)
                        _eta_text = f"~{_fmt_secs(_eta)} remaining"
                    else:
                        _eta_text = "estimating runtime…"
                    _trait_progress.info(f"**Trait {_i}/{_n_total}** (`{_t}`), {_eta_text}")

                def _on_progress(_msg):
                    _status.update(label=f"{_clock['trait']}: {str(_msg).strip()[:100]}")

                for _t, _res, _err in run_traits(_parsed, pheno, selected_traits, _opts,
                                                  on_start=_on_trait_start,
                                                  progress=_on_progress):
                    if _res is None:
                        _failed[_t] = _err
                        st.error(f"Trait `{_t}` failed: {_last_line(_err)}")
                    else:
                        # keep only what the summary and the downloads need
                        _done[_t] = dict(_res.summary, zip_name=_res.zip_name,
                                         zip_bytes=_res.zip_bytes)
                        _elapsed.append(_time_mod.time() - _clock["t0"])
                    del _res
                _gm_tmp.cleanup()
                del _parsed
                if _failed:
                    _status.update(label=(f"Full analysis finished: {len(_failed)} of "
                                          f"{_n_total} traits failed"), state="error")
                else:
                    _status.update(label="Full analysis complete!", state="complete")
                # Replace the trait-progress banner with a final summary
                # so it doesn't keep showing "ETA: ~Xm remaining" forever.
                _trait_progress.success(
                    f"**Done.** Processed {len(_done)}/{_n_total} traits in "
                    f"{_fmt_secs(sum(_elapsed))}."
                    + (f" Failed: {', '.join(f'`{t}`' for t in _failed)}." if _failed else "")
                )

            # --- Combined ZIP when several traits were selected (with the error report) ---
            _combined_zip_name = None
            _combined_zip_bytes = None
            if _n_total > 1:
                _combined_zip_name, _combined_zip_buf = _combine_gwas_zips(
                    {t: d["zip_bytes"] for t, d in _done.items()},
                    pheno_label=st.session_state.get("pheno_file_label", "unknown_pheno"),
                    extra_members=(
                        {"failed_traits.txt": failed_traits_report(selected_traits, _done, _failed)}
                        if _failed else None),
                )
                _combined_zip_bytes = _combined_zip_buf.getvalue()

            # --- Per-trait results summary (tabs when >1 trait) ---
            def _render_one_trait_summary(_t):
                _td = _done.get(_t)
                if _td is None:
                    st.error(f"`{_t}` failed: {_last_line(_failed.get(_t, ''))}")
                    with st.expander("Error details"):
                        st.code(_failed.get(_t, ""))
                    return
                if _td.get("covariate_dropped"):
                    st.warning(_covariate_drop_message(_td["covariate_dropped"],
                                                       _td["covariate_n_before"]))
                _render_results_summary_card(
                    lambda_gc=_td["lambda_gc"],
                    n_significant=_td["n_significant"],
                    sig_label=_td["sig_label"],
                    n_ld_blocks=_td["n_ld_blocks"],
                    pc_k=None,  # no auto-PC selection
                )

            if _n_total > 1:
                for _tab, _t in zip(st.tabs(selected_traits), selected_traits):
                    with _tab:
                        _render_one_trait_summary(_t)
            else:
                _render_one_trait_summary(selected_traits[0])

            # --- Interpretation panel (rendered once, regardless of trait count) ---
            _render_interpretation_panel()

            # --- Download button: combined ZIP for multi-trait, single ZIP for one trait ---
            if _combined_zip_bytes is not None:
                st.download_button(
                    label=(f"Download Full Results ({len(_done)} of {_n_total} traits, "
                           "combined ZIP)"),
                    data=_combined_zip_bytes,
                    file_name=_combined_zip_name,
                    mime="application/zip",
                    key="pipe_zip_dl",
                )
            elif _done:
                _td = _done[selected_traits[0]]
                st.download_button(
                    label="Download Full Results (ZIP)",
                    data=_td["zip_bytes"],
                    file_name=_td["zip_name"],
                    mime="application/zip",
                    key="pipe_zip_dl",
                )

            # What ran, for this page's later reruns and for tests (no ZIP bytes, and
            # nothing another view reads as input).
            st.session_state["one_click_results"] = {
                "traits": list(selected_traits),
                "completed": {t: {k: v for k, v in d.items() if k != "zip_bytes"}
                              for t, d in _done.items()},
                "failed": dict(_failed),
                "combined_zip_name": _combined_zip_name,
            }

            # ── Stop here. Don't run the detailed single-trait view below ──
            st.stop()

    # ================================================================
    #                 SINGLE-TRAIT ANALYSIS (ORDERED)
    # ================================================================

    st.header("Single-trait GWAS (detailed view)")

    # --- Run GWAS button (prevents auto-trigger on every sidebar change) ---
    # Reset trigger when trait changes so user must re-click
    if st.session_state.get("_last_gwas_trait") != trait_col:
        st.session_state["gwas_triggered"] = False
    _run_clicked = st.button("Run GWAS", type="primary", help="Click to start the GWAS analysis with current parameters.")
    if _run_clicked:
        st.session_state["gwas_triggered"] = True
    # Input fingerprint for the stale-result guard below: every input that changes the
    # scan. _covar_fp is duplicated here (also computed for the MLMM/FarmCPU readers) so a
    # covariate change is caught before the guard.
    _covar_fp = (
        "nocov" if user_covar_mat is None
        else f"cov{hash_bytes(user_covar_mat.tobytes(), digest_size=8)}"
    )
    _scan_fp = hash_bytes(repr((
        trait_col, int(n_pcs), _covar_fp,
        maf_thresh, miss_thresh, ind_miss_thresh, mac_thresh, info_thresh,
        impute_method, impute_k, impute_l,
        int(drop_alt), vcf_hash, pheno_hash, norm_option,
    )).encode(), digest_size=16)
    # Navigated back (did not click Run this rerun) and results already exist for this
    # trait -> re-display them, with a staleness notice if the inputs have since changed.
    if (not _run_clicked and _has_persisted_gwas()
            and st.session_state.get("gwas_run_summary", {}).get("trait") == trait_col):
        _stored_fp = st.session_state.get("gwas_run_summary", {}).get("scan_fp")
        _stale = _stored_fp is not None and _stored_fp != _scan_fp
        _render_last_run_results(stale=_stale)
        st.stop()
    if not st.session_state.get("gwas_triggered", False):
        _rehydrate_or_stop()
    st.session_state["_last_gwas_trait"] = trait_col
    # this run's subsampling (if ticked) writes its own summary for the run manifest;
    # never carry an earlier run's (or another trait's) summary into this one
    st.session_state.pop("subsampling_gwas_summary", None)

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
            user_covar=user_covar_mat,
            user_covar_names=user_covar_names,
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
            "Only OLS marginal effects (Beta_OLS) are shown; these estimate "
            "per-allele effects but do **not** account for population structure (kinship). "
            "This can happen with very small sample sizes or singular kinship matrices."
        )

    # Multiple testing
    bonf_thresh = 0.05 / len(gwas_df)

    # M_eff (Li & Ji 2005), computed once, cached by SNP count
    meff_val = None
    meff_thresh = None
    _meff_computed = False
    if sig_rule.startswith("M_eff"):
        with st.spinner("Computing M_eff (Li & Ji): eigendecomposition of SNP correlations…"):
            try:
                meff_val, _meff_eigs = compute_meff_li_ji(geno_imputed)
                st.session_state["meff_val"] = meff_val
                meff_thresh = 0.05 / meff_val
                _meff_computed = True
            except Exception as e:
                logging.exception("M_eff computation failed")
                st.warning(f"M_eff computation failed ({e}); falling back to Bonferroni.")
                meff_val = len(gwas_df)
                meff_thresh = bonf_thresh

    # Significance columns: one rule for MLM here and for MLMM / FarmCPU below, the
    # same as the CLI / One-Click pipeline (MLM's thresholds; FDR within each model).
    # Significant_Meff only when M_eff was computed, as before.
    _sig_cols = {
        "bonf_thresh": bonf_thresh,
        "meff_thresh": meff_thresh if _meff_computed else None,
        "custom_thresh": (float(custom_sig_thresh)
                          if sig_rule.startswith("Custom") and custom_sig_thresh is not None
                          else None),
    }
    gwas_df = add_significance_columns(gwas_df, **_sig_cols)

    # active threshold
    if sig_rule.startswith("Custom") and custom_sig_thresh is not None:
        active_lod = -np.log10(float(custom_sig_thresh))
        active_label = f"p < {float(custom_sig_thresh):.1e}"
        _active_sig_col = "Significant_Custom"
    elif sig_rule.startswith("FDR"):
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

    # Optional secondary threshold (show BOTH Bonferroni and M_eff
    # on the Manhattan when the user has asked for it). The line is
    # informational; it does not change which SNPs are amplified.
    secondary_lod = None
    secondary_label = None
    if show_secondary_threshold:
        if sig_rule.startswith("Custom"):
            secondary_lod = -np.log10(bonf_thresh)
            secondary_label = "Bonferroni α=0.05"
        elif sig_rule.startswith("Bonferroni") and meff_thresh is not None:
            secondary_lod = -np.log10(meff_thresh)
            secondary_label = f"M_eff (M={meff_val:,})"
        elif sig_rule.startswith("M_eff"):
            secondary_lod = -np.log10(bonf_thresh)
            secondary_label = "Bonferroni α=0.05"
        # FDR primary: skip; FDR isn't a single −log10p line

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
            show_plot(fig_pc, "medium")
            st.caption("Data: \"Download PCA scores (CSV)\" below.")
            st.session_state["gwas_figures"][f"PCA_{trait_col}.html"] = plotly_html_bytes(fig_pc)

            download_plotly_fig(
                fig_pc,
                filename=f"PCA_{trait_col}.html",
                html_bytes=st.session_state["gwas_figures"][f"PCA_{trait_col}.html"],
                label="Download PCA plot (HTML)"
            )

            # PCA scree plot (eigenvalue decay)
            if pca_eigenvalues is not None:
                from gwas.plotting import plot_pca_scree
                _fig_scree = plot_pca_scree(pca_eigenvalues, n_pcs_used=n_pcs,
                                             title="PCA Scree Plot")
                show_plot(_fig_scree, "small",      # the first 20 components, as drawn
                          data=eigen_table(pca_eigenvalues).head(20),
                          data_name=f"PCA_scree_{trait_col}")
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
        # this trait's applied transformation only ({} when it uses raw values)
        pheno_transformations=({trait_col: norm_option} if norm_option != "none" else {}),
        kinship_model=results.get("kinship_model", None) if isinstance(results, dict) else None,
        loco_fallback_by_chr=st.session_state.get("loco_diagnostics_by_chr", {}),
    )


    # ============================================================
    # Advanced: Stability Screening
    # ============================================================
    with st.expander("Advanced: Stability Screening", expanded=False):

        st.markdown("### Subsampling GWAS resampling (full MLM with GRM recomputation)")

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

            # Stable cache key for subsampling results. _scan_fp carries every input of
            # the scan (the trait's normalisation, the phenotype table, the genotypes,
            # PCs, covariates, QC and imputation settings), so an Apply / Remove in the
            # phenotype QC panel or a new upload is a miss.
            boot_cache_key = (
                f"BOOT_GWAS::{trait_col}::{_scan_fp}"
                f"::reps{boot_n_reps}::frac{boot_sample_frac:.2f}"
                f"::thresh{boot_disc_thresh:.1e}::seed{boot_seed}"
                f"::loco{boot_use_loco}"
                f"::n{geno_imputed.shape[0]}x{geno_imputed.shape[1]}"
            )

            if boot_cache_key not in st.session_state:
                # one entry per trait (each holds a reps x SNPs p-value matrix)
                for _old_boot_key in [k for k in list(st.session_state.keys())
                                      if isinstance(k, str)
                                      and k.startswith(f"BOOT_GWAS::{trait_col}::")]:
                    del st.session_state[_old_boot_key]

                # Validate Z_for_grm is available
                Z_grm_boot = st.session_state.get("Z_grm", None)
                if Z_grm_boot is None:
                    st.error(
                        "LD-pruned Z matrix (Z_grm) not found in session state. "
                        "This is computed during GWAS preprocessing; "
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
            st.markdown(
                "#### Per-SNP discovery frequency (top 50)",
                help=("Fraction of resamples in which a SNP stayed significant:\n"
                      "- **> 80%** robust, detected in most resamples\n"
                      "- **50-80%** moderately stable, mind sample-size limits\n"
                      "- **< 50%** unstable, may be driven by a few influential samples\n\n"
                      "Full guidance: **Help → Interpreting Results**."),
            )
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
            show_plot(fig_boot_hist, "medium", data=histogram_bins(fig_boot_hist),
                      data_name=f"subsampling_discovery_freq_{trait_col}")
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
                title=f"Subsampling stability Manhattan: {trait_col}",
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
            show_plot(fig_boot_manh, "wide")
            st.caption("Data: \"Download per-SNP subsampling results (CSV)\" below.")
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
        # the enlarged SNPs are the active rule's column, as in the tables
        fig_manh = plot_manhattan_static(
            df_manh, active_lod, active_label, f"Manhattan (MLM): {trait_col}",
            secondary_lod=secondary_lod, secondary_label=secondary_label,
            sig_mask=(df_manh[_active_sig_col].fillna(False).to_numpy(dtype=bool)
                      if _active_sig_col in df_manh.columns else None),
        )
        plt.figure(fig_manh.number)
        plt.xticks(tick_pos, tick_lab, fontsize=8)
        st.session_state["gwas_figures"][f"Manhattan_MLM_{trait_col}.png"] = fig_manh

        # Interactive Manhattan
        st.write("### Manhattan Plot (MLM)")
        df_plot, tick_pos_i, tick_lab_i = compute_cumulative_positions(gwas_df)
        fig_int = plot_manhattan_interactive(
            df_plot, active_lod, active_label, f"Interactive Manhattan (MLM): {trait_col}",
            secondary_lod=secondary_lod, secondary_label=secondary_label,
        )
        fig_int.update_layout(
            xaxis=dict(tickmode="array", tickvals=tick_pos_i, ticktext=tick_lab_i),
            height=500,
        )
        show_plot(fig_int, "wide")
        st.caption('Data: the per-SNP results table in "Download GWAS results (ZIP)" at the end of the page.')
        st.session_state["gwas_figures"][f"Interactive_Manhattan_MLM_{trait_col}.html"] = (
            plotly_html_bytes(fig_int))

        download_plotly_fig(
            fig_int,
            filename=f"Interactive_Manhattan_MLM_{trait_col}.html",
            label="Download interactive MLM Manhattan (HTML)",
            html_bytes=st.session_state["gwas_figures"][
                f"Interactive_Manhattan_MLM_{trait_col}.html"],
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
        _gc_cols[0].metric(
            "λGC (bulk 5-95%)", f"{lambda_gc:.3f}",
            help=(
                "Genomic-control λ: how well population structure is accounted for. "
                "**0.9-1.1** well-calibrated (p-values reliable); **<0.9** deflated "
                "(common for oligogenic traits or when LOCO kinship removes confounding, "
                "usually benign, but check known loci are still detected); **1.1-1.3** "
                "mildly inflated (usable); **>1.3** "
                "notably inflated; add PCs or check batch effects. "
                "Full guidance: **Help → Interpreting Results**."
            ),
        )
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
        show_plot(fig_qq, "small")
        st.caption('Data: the per-SNP results table in "Download GWAS results (ZIP)" at the end of the page.')
        st.session_state["gwas_figures"][f"QQ_{trait_col}.png"] = fig_qq

        download_matplotlib_fig(
            fig_qq,
            filename=f"QQ_{trait_col}.png",
            label="Download QQ plot (PNG)"
        )

        # --- Structure screen: does the trait track the leading genotype-PCA axes? ---
        # Uses the full PCA-spectrum scores (pcs_full), so it renders even at the
        # k=0 default (report-only; it does NOT set the PC count).
        if pcs_full is not None and getattr(pcs_full, "shape", (0, 0))[1] >= 1:
            _k_scr = min(5, int(pcs_full.shape[1]))
            _pc_corrs = np.corrcoef(np.c_[y, pcs_full[:, :_k_scr]], rowvar=False)[0, 1:]
            _max_abs = float(np.nanmax(np.abs(_pc_corrs)))
            st.markdown(
                f"**Trait-structure association (leading {_k_scr} PCs):** "
                f"max |r| = {_max_abs:.2f}  |  per-PC r = {[round(float(c), 2) for c in _pc_corrs]}"
            )
            st.caption(
                "Correlation of the phenotype with the leading genotype-PCA axes. A low value "
                "shows no detectable trait-structure association; a high value is ambiguous and is "
                "not evidence that PCs are required. Report-only; it does not set the PC count."
            )

        # --- P-value distribution (calibration diagnostic, beside the QQ plot) ---
        from utils.pub_theme import FIGSIZE as _FIG
        fig_hist, ax_hist = plt.subplots(figsize=_FIG["histogram"])
        ax_hist.hist(gwas_df["PValue"].astype(float), bins=50,
                     color=PALETTE["blue"], edgecolor="white", linewidth=0.5)
        ax_hist.set_xlabel("P-value")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title("Distribution of GWAS P-values")
        show_plot(fig_hist, "medium", data=histogram_bins(fig_hist),
                  data_name=f"pvalue_histogram_{trait_col}")
        export_matplotlib(fig_hist, f"pvalue_hist_{trait_col}",
                          label_prefix="Download p-value histogram")
        plt.close(fig_hist)

        # --- Results summary card (single-trait) ---
        _st_n_sig = int(gwas_df[_active_sig_col].sum()) if _active_sig_col in gwas_df.columns else 0
        _render_results_summary_card(
            lambda_gc=lambda_gc,
            n_significant=_st_n_sig,
            sig_label=active_label,
            pc_k=n_pcs,
        )

        # Continue to the Post-GWAS page from here, where the user finishes reading
        # their hits. A second copy remains in "Next steps" at the page bottom.
        # Guarded: st.page_link raises a url_pathname KeyError under Streamlit's
        # AppTest bare mode (no page context); catching it keeps that harness quirk
        # from blanking the results computed below. Renders normally in the real app.
        try:
            st.page_link(
                "pages/Post_GWAS_Analysis.py",
                label="Explore these hits → Post-GWAS Analysis",
            )
        except Exception:
            pass

    # ============================================================
    # 6. Genome-wide LD DECAY: accurate per-chromosome computation
    # (the CLI's function and settings; Post-GWAS shows this estimate in the LD Blocks &
    # Haplotypes Settings, next to the window it sets: gwas.windows.ld_window_kb)
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
            ld_decay_kb = None
    except Exception:
        logging.exception("Manual path LD decay computation failed")
        ld_decay_kb = None
    if ld_decay_kb is None:
        # no estimate in this run: no decay value, as in the CLI (never an earlier run's)
        for _k in ("ld_decay_df", "ld_decay_summary"):
            st.session_state.pop(_k, None)
        st.session_state["ld_decay_computed"] = False
    # the LD window by the CLI's rule: int(2 x decay), or 300 kb without an estimate
    flank_kb = ld_window_kb(ld_decay_kb)
    ld_block_max_dist_bp = int(ld_decay_kb * 1000) if ld_decay_kb is not None else None
    # Make MLM results available immediately for MLMM prefiltering
    st.session_state["gwas_df"] = gwas_df[["SNP", "PValue", "Chr", "Pos"]].copy()
    # ============================================================
    # 7. MLMM (optional, slow)
    # ============================================================
    # the MLM frame handed to Post-GWAS carries its effects (per ALT allele copy), for the
    # lead-SNP titles and flag and the Significant SNPs tab's effect columns
    _effect_cols = [c for c in ("Beta_MLM", "SE_MLM", "Beta_OLS", "SE_OLS") if c in gwas_df.columns]
    gwas_mlm = gwas_df[["SNP", "Chr", "ChrNum", "Pos", "PValue"] + _effect_cols].copy()
    gwas_mlm["Model"] = "MLM"
    results_all_models = [gwas_mlm]
    cofactor_logs = {}
    pheno_reader_key = put_object_in_session(
        pheno_reader, "PHENO_READER",
        vcf_hash, pheno_hash, trait_col, norm_option
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
        _pc = None
        if pcs_full is not None and k_pcs > 0:
            _pc = pcs_full[:, :int(min(k_pcs, pcs_full.shape[1]))]
        if _pc is None and user_covar_mat is None:
            return None
        if user_covar_mat is None:
            return CovarData(
                iid=iid,
                val=_pc,
                names=[f"PC{i + 1}" for i in range(_pc.shape[1])],
            )
        _val = np.column_stack([_pc, user_covar_mat]) if _pc is not None else user_covar_mat
        _names = ([f"PC{i + 1}" for i in range(_pc.shape[1])] if _pc is not None else []) + list(user_covar_names)
        return CovarData(iid=iid, val=_val, names=_names)

    # covariate fingerprint so the cached reader key changes when covars change
    _covar_fp = "nocov" if user_covar_mat is None else f"cov{hash_bytes(user_covar_mat.tobytes(), digest_size=8)}"
    covar_reader_mlmm = _build_covar_for_model(int(n_pcs_mlmm))
    covar_reader_farmcpu = _build_covar_for_model(int(n_pcs_farmcpu))
    covar_reader_mlmm_key = put_object_in_session(
        covar_reader_mlmm, "COVAR_READER",
        vcf_hash, pheno_hash, trait_col, int(n_pcs_mlmm), _covar_fp,
    )
    covar_reader_farmcpu_key = put_object_in_session(
        covar_reader_farmcpu, "COVAR_READER",
        vcf_hash, pheno_hash, trait_col, int(n_pcs_farmcpu), _covar_fp,
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
        gwas_mlmm = add_significance_columns(gwas_mlmm, **_sig_cols)  # MLM's thresholds
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

            # 7b. MLMM Manhattan (static, for export, not displayed inline)
            df_mlmm = gwas_mlmm.copy()
            df_mlmm["PValue"] = df_mlmm["PValue"].astype(float).clip(1e-300, 1.0)
            df_mlmm["-log10p"] = -np.log10(df_mlmm["PValue"])
            df_mlmm, tick_pos_m, tick_lab_m = compute_cumulative_positions(df_mlmm)

            fig_mlmm = plot_manhattan_static(
                df_mlmm,
                active_lod=active_lod,
                active_label=active_label,
                title=f"Manhattan (MLMM): {trait_col}",
                secondary_lod=secondary_lod,
                secondary_label=secondary_label,
                sig_mask=(df_mlmm[_active_sig_col].fillna(False).to_numpy(dtype=bool)
                          if _active_sig_col in df_mlmm.columns else None),
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
                title=f"Interactive Manhattan (MLMM): {trait_col}",
                secondary_lod=secondary_lod,
                secondary_label=secondary_label,
            )
            fig_plotly_mlmm.update_layout(
                xaxis=dict(tickmode="array", tickvals=tick_pos_m, ticktext=tick_lab_m, showgrid=False),
                height=500,
            )
            show_plot(fig_plotly_mlmm, "wide")
            st.caption('Data: the per-SNP results table in "Download GWAS results (ZIP)" at the end of the page.')
            st.session_state["gwas_figures"][f"Interactive_Manhattan_MLMM_{trait_col}.html"] = (
                plotly_html_bytes(fig_plotly_mlmm))
            download_plotly_fig(
                fig_plotly_mlmm, filename=f"Interactive_Manhattan_MLMM_{trait_col}.html",
                label="Download interactive MLMM Manhattan",
                html_bytes=st.session_state["gwas_figures"][
                    f"Interactive_Manhattan_MLMM_{trait_col}.html"])
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
            show_plot(fig_qq_mlmm, "small")
            st.caption('Data: the per-SNP results table in "Download GWAS results (ZIP)" at the end of the page.')
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
            max_iterations=farmcpu_max_iter,
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
        gwas_farmcpu = add_significance_columns(gwas_farmcpu, **_sig_cols)  # MLM's thresholds
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

            # FarmCPU Manhattan (static, for export)
            df_fc = gwas_farmcpu.copy()
            df_fc["PValue"] = df_fc["PValue"].astype(float).clip(1e-300, 1.0)
            df_fc["-log10p"] = -np.log10(df_fc["PValue"])
            df_fc, tick_pos_fc, tick_lab_fc = compute_cumulative_positions(df_fc)

            fig_farmcpu = plot_manhattan_static(
                df_fc,
                active_lod=active_lod,
                active_label=active_label,
                title=f"Manhattan (FarmCPU): {trait_col}",
                secondary_lod=secondary_lod,
                secondary_label=secondary_label,
                sig_mask=(df_fc[_active_sig_col].fillna(False).to_numpy(dtype=bool)
                          if _active_sig_col in df_fc.columns else None),
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
                title=f"Interactive Manhattan (FarmCPU): {trait_col}",
                secondary_lod=secondary_lod,
                secondary_label=secondary_label,
            )
            fig_plotly_fc.update_layout(
                xaxis=dict(tickmode="array", tickvals=tick_pos_fc, ticktext=tick_lab_fc, showgrid=False),
                height=500,
            )
            show_plot(fig_plotly_fc, "wide")
            st.caption('Data: the per-SNP results table in "Download GWAS results (ZIP)" at the end of the page.')
            st.session_state["gwas_figures"][f"Interactive_Manhattan_FarmCPU_{trait_col}.html"] = (
                plotly_html_bytes(fig_plotly_fc))
            download_plotly_fig(
                fig_plotly_fc, filename=f"Interactive_Manhattan_FarmCPU_{trait_col}.html",
                label="Download interactive FarmCPU Manhattan",
                html_bytes=st.session_state["gwas_figures"][
                    f"Interactive_Manhattan_FarmCPU_{trait_col}.html"])
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
            show_plot(fig_qq_fc, "small")
            st.caption('Data: the per-SNP results table in "Download GWAS results (ZIP)" at the end of the page.')
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
                # Benjamini-Hochberg within this model, as in its own table
                dfm["FDR"] = add_significance_columns(dfm[["PValue"]], bonf_thresh=bonf_thresh)["FDR"].to_numpy()
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
            show_plot(fig_overlay, "wide")
            st.caption("Data: \"Download cross-model results (CSV)\" below.")
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
    st.session_state["gwas_run_summary"] = {
        "trait": trait_col,
        "n_samples": int(geno_df.shape[0]) if "geno_df" in locals() else None,
        "n_snps": int(len(st.session_state["gwas_df"])),
        "models": sorted(set(["MLM"] + [m.split()[0] for m in model_choices])) if "model_choices" in locals() else ["MLM"],
        "lambda_gc": float(lambda_gc) if "lambda_gc" in locals() else None,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "scan_fp": _scan_fp if "_scan_fp" in locals() else None,
    }
    bump_data_version()  # GWAS complete, notify downstream pages
    # ============================================================
    # Auto-build GWAS-only ZIP (on every run, no LD content)
    # ============================================================

    pheno_label = st.session_state.get("pheno_file_label", "unknown_pheno")

    # MLM always runs (foundational), so include it in the tag even though
    # it is no longer in the user-facing multiselect.
    _models_tag = "_".join(sorted(["MLM"] + [m.split()[0] for m in model_choices]))
    # The fingerprint names the session key the download below reads.
    _gwas_fp = f"n{len(gwas_df)}_lam{lambda_gc:.4f}"
    zip_key = f"gwas_zip_{pheno_label}__{trait_col}__{_models_tag}__{_gwas_fp}"

    # Built on every run (the page reaches this point only on a Run; other reruns stop
    # at the last-run guard), because the tables' significance columns follow the rule,
    # which the fingerprint does not see. This trait's earlier ZIPs (and their metadata
    # flags) are dropped, and the run records its ZIP's key for the re-display path.
    _zip_prefix = f"gwas_zip_{pheno_label}__{trait_col}__"
    for _old_zip_key in [k for k in list(st.session_state.keys())
                         if isinstance(k, str) and k.startswith(_zip_prefix)]:
        del st.session_state[_old_zip_key]

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
    st.session_state["gwas_run_summary"]["zip_key"] = zip_key


    # --- Core genotype / phenotype objects (shared across traits)
    st.session_state["geno_imputed"] = geno_imputed
    # Store TRUE genotype sample order (CRITICAL for LD + haplotypes)
    st.session_state["geno_row_ids"] = np.asarray(geno_df.index, dtype=str).tolist()
    # Force numpy dtype on chroms: ArrowStringArray (newer pandas) cannot be
    # hashed by Streamlit @st.cache_data, breaking downstream cached wrappers.
    st.session_state["chroms"] = np.asarray(chroms, dtype=str)
    st.session_state["positions"] = positions
    st.session_state["sid"] = sid
    st.session_state["geno_df"] = geno_df
    st.session_state["pcs"] = pcs  # can be None

    # --- Optional LD hints (safe)
    if "ld_decay_kb" in locals():
        st.session_state["ld_decay_kb"] = None if ld_decay_kb is None else float(ld_decay_kb)
    if "ld_block_max_dist_bp" in locals():
        st.session_state["ld_block_max_dist_bp"] = (
            None if ld_block_max_dist_bp is None else int(ld_block_max_dist_bp))

    # ============================================================
    # BLOCK 4: REPRODUCIBILITY / METADATA (embedded in ZIP)
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
        manifest["Covariate columns"] = (
            ", ".join(user_covar_names) if user_covar_names else None)
        manifest["Covariates applied"] = user_covar_mat is not None
        # this run's subsampling summary (written by the stability section above, after
        # the manifest was built), or None when this run did no subsampling
        manifest["subsampling_gwas_resampling"] = (
            st.session_state.get("subsampling_gwas_summary") if run_subsampling else None)
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
    _gwas_fp = f"n{len(gwas_df)}_lam{lambda_gc:.4f}"
    zip_key = f"gwas_zip_{pheno_label}__{trait_col}__{_models_tag}__{_gwas_fp}"

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
        st.page_link("pages/Post_GWAS_Analysis.py", label="Post-GWAS Analysis →")




