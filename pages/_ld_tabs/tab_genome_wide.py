"""Tab 6 — Peak-centric LD blocks & haplotype analysis."""

import logging
import textwrap
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from streamlit.runtime.scriptrunner import StopException
from utils.pub_theme import PALETTE_CYCLE, FIGSIZE, export_matplotlib

from gwas.ld import (
    find_ld_clusters_genomewide,
    filter_contained_blocks,
)
from gwas.haplotype import (
    mlg_labels_for_block,
)
from . import LDContext


def _hash_df(df: pd.DataFrame) -> str:
    import hashlib
    h = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.blake2b(h.tobytes(), digest_size=16).hexdigest()


def render(
    ctx: LDContext,
    get_r2_cached,
    cached_extract_block_geno,
    compute_block_qc_effects,
    run_haplotype_block_gwas_cached_fn,
):
    try:
        st.subheader("Peak-centric LD blocks")

        st.caption(
            "Peak-centric blocks define LD structure around GWAS-significant SNPs. "
            "These reflect LD at association peaks, not population-level LD block structure."
        )

        # Local aliases
        gwas_df = ctx.gwas_df
        chroms = ctx.chroms
        positions = ctx.positions
        geno_ld = ctx.geno_ld
        sid = ctx.sid
        trait_col = ctx.trait_col
        ld_decay_kb = ctx.ld_decay_kb
        haplo_df_auto = ctx.haplo_df_auto

        auto_ld = st.checkbox(
            "Run / refresh LD block detection",
            value=st.session_state.get("auto_ld_run", False),
            key="auto_ld_run"
        )

        if auto_ld:
            sig_thresh = st.number_input(
                "Significance threshold (P <)",
                min_value=1e-12,
                max_value=0.1,
                value=1e-5,
                format="%.1e",
            )

            flank_kb_default = float(np.clip(ld_decay_kb * 2, 200, 2000))
            flank_kb_auto = st.number_input(
                "LD window around each lead SNP (kb)",
                min_value=10.0,
                max_value=2000.0,
                value=float(flank_kb_default),
                step=50.0,
                help=(
                    "Physical distance (in kb) to extend around each lead SNP when building "
                    "LD blocks. Default is 2× the estimated LD decay distance. Increase for "
                    "self-pollinating crops with slow LD decay."
                ),
            )

            ld_threshold_auto = st.number_input(
                "r² threshold for cluster definition",
                min_value=0.1,
                max_value=0.9,
                value=0.6,
                step=0.05,
                help=(
                    "SNPs with pairwise r² above this threshold are grouped into the same LD block. "
                    "0.6 is a common default for self-pollinating crops (tomato, pepper) where LD "
                    "decays slowly. Use 0.3–0.5 for outcrossing species."
                ),
            )

            adj_r2_min_auto = st.number_input(
                "Adjacent coherence split threshold (r²) (split if below)",
                min_value=0.0,
                max_value=0.9,
                value=0.2,
                step=0.05,
                help=(
                    "Split a candidate block when adjacent SNPs have r² below this value, "
                    "indicating a recombination breakpoint. Lower values = fewer splits "
                    "(more permissive blocks)."
                ),
            )
            st.session_state["adj_r2_min"] = float(adj_r2_min_auto)

            gap_factor_auto = st.slider(
                "Split blocks if adjacent SNP gap > (multiplier × median gap)",
                min_value=1.0,
                max_value=50.0,
                value=float(st.session_state.get("gap_factor_auto", 10.0)),
                step=0.5,
                help="Higher = fewer splits caused by uneven marker spacing.",
            )
            st.session_state["gap_factor_auto"] = float(gap_factor_auto)
            min_snps_block_auto = st.number_input(
                "Minimum SNPs per cluster",
                min_value=2,
                max_value=50,
                value=3,
                step=1,
            )

            top_n = st.number_input(
                "Also include top N SNPs by P-value",
                min_value=0,
                max_value=500,
                value=10,
                step=1,
            )

            with st.spinner("Detecting peak-centric LD blocks…"):
                haplo_df_auto = find_ld_clusters_genomewide(
                    gwas_df=gwas_df,
                    chroms=chroms,
                    positions=positions,
                    geno_imputed=np.asarray(geno_ld, float),
                    sid=sid,
                    ld_threshold=ld_threshold_auto,
                    flank_kb=flank_kb_auto,
                    min_snps=min_snps_block_auto,
                    top_n=top_n,
                    sig_thresh=sig_thresh,
                    max_dist_bp=None,
                    ld_decay_kb=ld_decay_kb,
                    adj_r2_min=float(adj_r2_min_auto),
                    gap_factor=float(gap_factor_auto),
                )

                # --- APPLY mega-block filter consistently ---
                haplo_df_auto, _ = filter_contained_blocks(
                    haplo_df_auto,
                    min_contained=int(st.session_state.get("mega_min_contained", 2)),
                    size_ratio_threshold=float(st.session_state.get("mega_size_ratio", 3.0)),
                    mode="remove" if st.session_state.get("mega_block_mode", "Remove") == "Remove" else "flag",
                )

            if "haplo_df_auto" not in st.session_state:
                st.session_state["haplo_df_auto"] = {}

            st.session_state["haplo_df_auto"][trait_col] = haplo_df_auto
            st.session_state["ld_blocks_final"] = haplo_df_auto.copy()
            st.session_state["ld_blocks"] = haplo_df_auto.copy()

            # --- LD metadata logging ---
            ld_metadata_auto = {
                "method": "peak-centric LD blocks",
                "ld_threshold_r2": float(ld_threshold_auto),
                "flank_kb": float(flank_kb_auto),
                "min_snps_per_block": int(min_snps_block_auto),
                "top_n_snps": int(top_n),
                "sig_threshold": float(sig_thresh),
                "ld_decay_kb": float(ld_decay_kb),
                "adjacent_r2_split": float(adj_r2_min_auto),
            }

            st.session_state["ld_block_metadata_auto"] = ld_metadata_auto

        # Show clusters if available
        if not isinstance(haplo_df_auto, pd.DataFrame) or haplo_df_auto.empty:
            st.info(
                "No LD blocks available yet. "
                "Enable 'Run / refresh LD block detection'."
            )
        else:
            st.success(f"Detected {haplo_df_auto.shape[0]} peak-centric LD blocks.")
            with st.expander("View LD block table"):
                st.dataframe(haplo_df_auto.head(20))
                st.download_button(
                    "Download LD blocks (CSV)",
                    haplo_df_auto.to_csv(index=False).encode(),
                    file_name="LD_clusters_genomewide.csv",
                    mime="text/csv",
                    key="dl_ld_clusters"
                )

        # ============================================================
        # Stability of LD block discovery
        # --------------------------------------------------------
        # Haplotype / multi-locus genotype analysis
        # --------------------------------------------------------
        st.markdown("### Haplotype / multi-locus genotype analysis")

        run_hap_gwas = st.checkbox(
            "Run haplotype / multi-locus genotype analysis",
            value=False,
            help="For each LD block, define multi-locus genotypes (MLGs) and test trait ~ block MLG.",
        )

        if run_hap_gwas:
            _render_haplotype_gwas(
                ctx=ctx,
                haplo_df_auto=haplo_df_auto,
                cached_extract_block_geno=cached_extract_block_geno,
                compute_block_qc_effects=compute_block_qc_effects,
                run_haplotype_block_gwas_cached_fn=run_haplotype_block_gwas_cached_fn,
            )

    except StopException:
        pass


def _render_haplotype_gwas(
    ctx,
    haplo_df_auto,
    cached_extract_block_geno,
    compute_block_qc_effects,
    run_haplotype_block_gwas_cached_fn,
):
    """Sub-section: Haplotype analysis parameters, execution, and visualization."""
    st.markdown("#### Haplotype analysis parameters")

    trait_col = ctx.trait_col
    ld_trait = ctx.ld_trait
    geno_ld = ctx.geno_ld
    geno_df = ctx.geno_df
    chroms = ctx.chroms
    positions = ctx.positions
    sid = ctx.sid
    pcs = ctx.pcs
    trait_col_resolved = ctx.ld_trait

    col1, col2 = st.columns(2)

    with col1:
        min_hap_count = st.number_input(
            "Minimum total samples per haplotype (collapse rarer → Other)",
            min_value=2,
            max_value=50,
            value=5,
            step=1,
            help=(
                "Haplotypes observed in fewer samples than this threshold "
                "are collapsed into the 'Other' group."
            ),
        )

    with col2:
        min_group_size = st.number_input(
            "Minimum samples per haplotype group (for testing)",
            min_value=2,
            max_value=50,
            value=3,
            step=1,
            help=(
                "Only haplotype groups with at least this many samples "
                "are included in the ANOVA test."
            ),
        )

    if run_haplotype_block_gwas_cached_fn is None:
        st.error(
            "Haplotype analysis functions are not available. "
            "Check `GWAS_analysis.py` imports."
        )
        return

    haplo_blocks_to_use = haplo_df_auto

    if haplo_blocks_to_use is None or haplo_blocks_to_use.empty:
        st.warning("No peak-centric LD blocks available for haplotype analysis.")
        return

    # ------------------------------------------------------------
    # Use GWAS-aligned phenotype for haplotype GWAS (MANDATORY)
    # ------------------------------------------------------------
    pheno_for_hap = None

    if "pheno_used_for_hap" in st.session_state:
        pheno_for_hap = st.session_state["pheno_used_for_hap"]
    elif "pheno_aligned" in st.session_state:
        pheno_for_hap = st.session_state["pheno_aligned"]
    elif "pheno" in st.session_state:
        pheno_for_hap = st.session_state["pheno"]

    if pheno_for_hap is None:
        st.error("No phenotype available for haplotype analysis.")
        st.stop()

    pheno_for_hap = pheno_for_hap.copy()
    pheno_label = "GWAS-aligned"

    # Store for downstream visualization
    st.session_state["pheno_used_for_hap"] = pheno_for_hap
    st.session_state["trait_col"] = trait_col
    st.session_state["trait_col_locked"] = trait_col
    st.session_state["pheno_used_for_hap_label"] = pheno_label

    st.session_state["haplo_blocks_to_use"] = haplo_blocks_to_use
    st.session_state["pcs_for_hap"] = pcs

    # ---------------------------------------------
    # Avoid recomputation unless inputs change
    # ---------------------------------------------
    blocks_hash = _hash_df(haplo_blocks_to_use.reset_index(drop=True))
    pheno_hash = _hash_df(pheno_for_hap[[trait_col_resolved]].copy())

    pcs_sig = "nopcs"
    if pcs is not None:
        pcs_arr = np.asarray(pcs)
        pcs_sig = f"pcs{pcs_arr.shape[0]}x{pcs_arr.shape[1]}"

    geno_sig = f"{geno_ld.shape[0]}x{geno_ld.shape[1]}"
    n_perm = int(st.session_state.get("n_perm_hap", 1000))

    hap_key = (
        f"HAPGWAS::{trait_col}"
        f"::mhc{int(min_hap_count)}::mgs{int(min_group_size)}"
        f"::{pcs_sig}"
        f"::{geno_sig}"
        f"::nperm{n_perm}"
        f"::B{blocks_hash}::P{pheno_hash}"
    )

    if hap_key not in st.session_state:

        with st.spinner("Running haplotype / MLG analysis on LD blocks…"):

            hap_gwas_df, hap_tables = run_haplotype_block_gwas_cached_fn(
                haplo_blocks_to_use,
                chroms,
                positions,
                geno_ld,
                sid,
                geno_df,
                pheno_for_hap,
                ld_trait,
                pcs,
                int(min_hap_count),
                int(min_group_size),
                int(n_perm),
                st.session_state.get("geno_encoding", "dosage012"),
                st.session_state.get("n_pcs_used", None),
            )
            st.session_state["hap_tables"] = hap_tables
            st.session_state["hap_gwas_df"] = hap_gwas_df
        st.session_state[hap_key] = hap_gwas_df

    else:
        hap_gwas_df = st.session_state[hap_key]

    # ------------------------------------------------------------
    # Block-level multiple testing correction (FDR)
    # ------------------------------------------------------------
    from statsmodels.stats.multitest import multipletests

    pvals = hap_gwas_df["PValue"].astype(float).values
    _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
    hap_gwas_df["FDR_qvalue"] = qvals
    st.caption(
        "Interpretation note: LD blocks are correlated due to linkage disequilibrium, "
        "so block-level multiple testing (e.g., BH-FDR) is conservative. "
        "Permutation-calibrated p-values (optional, below) provide an empirical robustness check."
    )

    show_cols = [c for c in [
        "Chr", "Start", "End", "Lead SNP", "n_snps",
        "n_haplotypes", "n_tested_haplotypes",
        "df1", "df2", "F_param", "PValue_param",
        "F_perm", "P_perm", "PValue", "FDR_qvalue",
        "eta2",
    ] if c in hap_gwas_df.columns]

    if hap_gwas_df is None or hap_gwas_df.empty:
        st.info(
            "No haplotype blocks had enough variation "
            "for association testing."
        )
        return

    st.success(
        f"Haplotype / MLG analysis ran on {hap_gwas_df.shape[0]} LD blocks."
    )
    st.dataframe(
        hap_gwas_df.sort_values("PValue")[show_cols].head(50),
        use_container_width=True
    )

    # Auto-compute block QC metrics (η², MAF)
    with st.spinner("Computing block effect sizes (η²) and MAF…"):
        hap_gwas_df = compute_block_qc_effects(
            hap_gwas_df=hap_gwas_df,
            chroms=chroms,
            positions=positions,
            geno_imputed=geno_ld,
            geno_df=geno_df,
            pheno_for_hap=pheno_for_hap,
            trait_col=ld_trait,
            min_hap_count=int(min_hap_count),
            min_group_size=int(min_group_size),
            compute_maf=True,
            geno_encoding=st.session_state.get("geno_encoding", "dosage012"),
        )
    st.session_state["hap_gwas_df_enriched"] = hap_gwas_df

    # ============================================================
    # BLOCK-LEVEL VISUALIZATION: MLG boxplots + Tukey HSD + Table
    # ============================================================
    _render_block_visualization(
        ctx=ctx,
        hap_gwas_df=hap_gwas_df,
        haplo_df_auto=haplo_df_auto,
        min_hap_count=int(min_hap_count),
        min_group_size=int(min_group_size),
        cached_extract_block_geno=cached_extract_block_geno,
    )


def _render_block_visualization(
    ctx,
    hap_gwas_df,
    haplo_df_auto,
    min_hap_count,
    min_group_size,
    cached_extract_block_geno,
):
    """Block-level MLG boxplots, Tukey HSD, forest plot, and expression panel selection."""
    st.markdown("### Haplotype / MLG Effect Visualization")

    trait_col = ctx.trait_col
    ld_trait = ctx.ld_trait
    geno_df = ctx.geno_df

    # ================================
    # Build enhanced block labels
    # ================================
    labels = []
    id_map = []

    for idx, row in hap_gwas_df.iterrows():
        block_id = row["Block_ID"] if "Block_ID" in hap_gwas_df.columns else idx
        chr_ = str(row["Chr"])
        start = int(row["Start"])
        end = int(row["End"])

        snps = int(row.get("n_snps", row.get("N_SNPs", row.get("SNPs", 0))))

        pval = float(row["PValue"])
        logp = -np.log10(max(pval, 1e-300))

        label = (
            f"Block {block_id} — Chr{chr_}: "
            f"{start / 1e6:.2f}–{end / 1e6:.2f} Mb | "
            f"{snps} SNPs | -log10(p)={logp:.2f}"
        )

        labels.append(label)
        id_map.append(str(block_id))

    # -------------------------------
    # Selector with informative labels
    # -------------------------------
    selected_label = st.selectbox(
        "Select LD block for phenotype–haplotype visualization:",
        options=labels,
    )

    selected_block = id_map[labels.index(selected_label)]

    # Retrieve the block row
    if "Block_ID" in hap_gwas_df.columns:
        block_row = hap_gwas_df.loc[
            hap_gwas_df["Block_ID"].astype(str) == selected_block
        ].iloc[0]
    else:
        block_row = hap_gwas_df.loc[int(selected_block)]

    block_chr = str(block_row["Chr"])
    block_start = int(block_row["Start"])
    block_end = int(block_row["End"])

    st.info(f"Selected block: Chr{block_chr}:{block_start:,}–{block_end:,}")

    # --------------------------------------------------------
    # Prepare phenotype
    # --------------------------------------------------------
    if "pheno_used_for_hap" not in st.session_state:
        st.error("No phenotype stored for haplotype analysis. Run haplotype analysis first.")
        st.stop()

    ph_used = st.session_state["pheno_used_for_hap"]
    trait_col_vis = ld_trait

    ph_ser = pd.to_numeric(ph_used[trait_col_vis], errors="coerce")
    ph_ser.index = ph_ser.index.astype(str)
    geno_sample_ids = np.asarray(
        st.session_state.get("geno_row_ids", []),
        dtype=str
    )

    if geno_sample_ids.size == 0:
        st.error(
            "Missing genotype sample IDs (geno_row_ids).\n"
            "Run the GWAS page again so LD analysis can align samples correctly."
        )
        st.stop()

    pheno_selected = ph_ser.reindex(geno_sample_ids)
    keep_pheno = np.isfinite(pheno_selected.values)

    if keep_pheno.shape[0] != st.session_state["ld_geno_hard"].shape[0]:
        st.warning(
            "Phenotype/genotype mismatch detected.\n"
            "Auto-aligning by genotype sample IDs."
        )
        pheno_selected = ph_ser.reindex(geno_sample_ids)
        keep_pheno = np.isfinite(pheno_selected.values)

    if keep_pheno.sum() < 5:
        st.warning("Too few non-missing phenotype values for haplotype visualization.")
        st.stop()

    # --------------------------------------------------------
    # Extract SNPs using phenotype mask
    # --------------------------------------------------------
    block_geno, _, block_sids = cached_extract_block_geno(
        st.session_state["ld_geno_hard"],
        st.session_state["ld_chroms"],
        st.session_state["ld_positions"],
        st.session_state["ld_sid"],
        block_chr,
        block_start,
        block_end,
        sample_keep_mask=keep_pheno,
        maf_threshold=st.session_state.get("maf_ld", 0.01),
        cache_key=f"{trait_col}::{block_chr}:{block_start}-{block_end}",
        snp_ids=block_row.get("SNP_IDs", None),
    )

    if block_geno.shape[1] < 1:
        st.warning("No SNPs found in this block window.")
        st.stop()

    block_geno_vis = block_geno
    sample_vis = pd.Index(geno_sample_ids)[keep_pheno]
    pheno_vis = pheno_selected.values[keep_pheno]

    # ---- build MLG labels AND propagate filtering mask ----
    mlg_labels_vis, keep_mlg = mlg_labels_for_block(
        block_geno_vis,
        min_hap_count=int(min_hap_count),
        return_mask=True,
        strict_selfing=st.session_state.get("selfing_mode", False),
        geno_encoding=st.session_state.get("geno_encoding", "dosage012"),
    )

    block_geno_vis = block_geno_vis[keep_mlg]
    sample_vis = sample_vis[keep_mlg]
    pheno_vis = pheno_vis[keep_mlg]

    # -------------------------------------------------
    # Rebuild haplotype strings after final filtering
    # -------------------------------------------------
    hap_strings = pd.Series(
        ["|".join(map(str, np.rint(row).astype(int))) for row in block_geno_vis],
        index=sample_vis,
        name="Allele_sequence"
    )

    # --------------------------------------------------------
    # PC-adjusted phenotype for visualization
    # --------------------------------------------------------
    pcs_vis = st.session_state.get("pcs_for_hap", None)
    trait_label = trait_col

    if pcs_vis is not None:
        try:
            pcs_df = pd.DataFrame(pcs_vis, index=geno_df.index)
            pcs_block = pcs_df.loc[sample_vis]

            valid_pc = ~pcs_block.isna().any(axis=1)
            pheno_vis = pheno_vis[valid_pc]
            sample_vis = sample_vis[valid_pc]
            block_geno_vis = block_geno_vis[valid_pc]
            mlg_labels_vis = mlg_labels_vis[valid_pc]
            pcs_block = pcs_block.loc[valid_pc]

            X = np.column_stack([np.ones((pcs_block.shape[0], 1)), pcs_block.values])
            beta, *_ = np.linalg.lstsq(X, pheno_vis.reshape(-1, 1), rcond=None)
            pheno_vis = (pheno_vis.reshape(-1, 1) - X @ beta).ravel()

            trait_label = f"{trait_col} (PC-adjusted)"

        except Exception:
            trait_label = trait_col

    # --------------------------------------------------------
    # Build MLG dataframe
    # --------------------------------------------------------
    mlg_df = pd.DataFrame({
        "Sample": sample_vis,
        "MLG": mlg_labels_vis,
        trait_col: pheno_vis,
    })

    group_counts = mlg_df["MLG"].value_counts()

    valid_groups = [
        g for g in group_counts.index
        if (group_counts[g] >= int(min_group_size)) and g != "Other"
    ]

    if len(valid_groups) < 2:
        st.warning("Too few haplotypes remain after filtering.")
        st.stop()

    st.caption(
        f"Showing {len(valid_groups)} haplotype groups "
        f"(≥{min_group_size} samples each, excluding 'Other')."
    )

    mlg_df = mlg_df.loc[mlg_df["MLG"].isin(valid_groups)].copy()
    mlg_df["MLG"] = pd.Categorical(mlg_df["MLG"])
    mlg_df["MLG"] = mlg_df["MLG"].cat.remove_unused_categories()

    # --------------------------------------------------------
    # Assign H1, H2, H3… labels
    # --------------------------------------------------------
    unique_mlg_sorted = (
        mlg_df["MLG"].value_counts()
        .loc[lambda s: (s.index != "Other") & (s >= int(min_group_size))]
        .sort_values(ascending=False)
        .index
    )

    mlg_label_map = {mlg: f"H{idx + 1}" for idx, mlg in enumerate(unique_mlg_sorted)}
    mlg_df["MLG_label"] = mlg_df["MLG"].map(mlg_label_map)

    # ---------------------------------------------------------
    # Save haplotype assignments
    # ---------------------------------------------------------
    st.session_state.setdefault("hap_tables", {})

    block_key = f"{block_chr}:{block_start}-{block_end}"
    store_df = mlg_df.merge(
        hap_strings.rename("Allele_sequence"),
        left_on="Sample",
        right_index=True,
        how="left"
    )

    store_df = store_df.rename(columns={"MLG_label": "Haplotype"}).copy()

    store_df["Trait_adj"] = pd.to_numeric(
        store_df[trait_col], errors="coerce"
    ).astype(float)

    st.session_state.setdefault("hap_tables_vis", {})
    st.session_state["hap_tables_vis"][block_key] = store_df[
        ["Sample", "Haplotype", "Allele_sequence", "Trait_adj"]].copy()

    hap_counts = mlg_df["MLG_label"].value_counts()

    # attach true haplotype strings
    mlg_df = mlg_df.merge(
        hap_strings.rename("Allele_sequence"),
        left_on="Sample",
        right_index=True,
        how="left"
    )

    mlg_df["Allele_sequence"] = mlg_df["Allele_sequence"].str.replace("|", "_", regex=False)

    # --------------------------------------------------------
    # Summary / MLG sequence table
    # --------------------------------------------------------
    mlg_summary = (
        mlg_df.groupby("MLG_label")
        .agg(
            Allele_sequence=(
                "Allele_sequence",
                lambda x: x.dropna().iloc[0] if x.dropna().size > 0 else ""
            ),
            n_samples=("Sample", "count"),
            phenotype_mean=(trait_col, "mean"),
        )
    )

    mlg_summary["SNPs_in_block"] = ";".join(block_sids.astype(str))

    st.markdown("### Haplotype (MLG) allele sequences")
    st.dataframe(mlg_summary, use_container_width=True)

    mlg_summary_out = mlg_summary.reset_index()
    st.download_button(
        "Download MLG sequence table (CSV)",
        mlg_summary_out.to_csv(index=False).encode(),
        file_name=f"MLG_sequences_block_{block_chr}_{block_start}_{block_end}.csv",
        mime="text/csv",
        key=f"dl_mlg_summary_{selected_block}"
    )

    # --------------------------------------------------------
    # Tukey HSD + CLD
    # --------------------------------------------------------
    _render_tukey_and_boxplot(
        mlg_df=mlg_df,
        trait_col=trait_col,
        trait_label=trait_label,
        mlg_order=[f"H{i + 1}" for i in range(len(unique_mlg_sorted))],
        hap_counts=hap_counts,
        selected_block=selected_block,
        block_chr=block_chr,
        block_start=block_start,
        block_end=block_end,
    )

    # Store for expression panel selection
    st.session_state["pheno_vis"] = pheno_vis
    st.session_state["sample_vis"] = sample_vis

    # ZIP export
    _render_zip_export(
        haplo_df_auto=haplo_df_auto,
        hap_gwas_df=hap_gwas_df,
        trait_col=trait_col,
    )

    # Download results
    st.download_button(
        "Download haplotype analysis results (CSV)",
        hap_gwas_df.to_csv(index=False).encode(),
        file_name=f"Haplotype_GWAS_{trait_col}.csv",
        mime="text/csv",
        key="dl_hap_gwas"
    )



def _render_tukey_and_boxplot(
    mlg_df, trait_col, trait_label, mlg_order, hap_counts,
    selected_block, block_chr, block_start, block_end,
):
    """Tukey HSD + CLD + boxplot + forest plot."""
    st.markdown("#### Tukey HSD post-hoc test")

    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    try:
        tukey = pairwise_tukeyhsd(
            endog=mlg_df[trait_col].astype(float),
            groups=mlg_df["MLG_label"].astype(str),
            alpha=0.05,
        )
    except Exception as e:
        logging.exception("Tukey HSD test failed")
        st.warning(
            f"Tukey HSD failed (likely constant phenotype after PC adjustment): {e}")
        st.stop()

    tukey_data = tukey.summary().data
    headers = tukey_data[0]
    rows = tukey_data[1:]

    tukey_df = pd.DataFrame(rows, columns=headers)
    tukey_df["reject"] = tukey_df["reject"].astype(bool)

    st.dataframe(tukey_df, use_container_width=True)

    # ------------------------------------------------------------
    # Compact Letter Display — Piepho (2004)
    # ------------------------------------------------------------
    st.markdown("#### Compact Letter Display (CLD)")

    groups = sorted(mlg_df["MLG_label"].unique())
    n_groups = len(groups)

    sig_matrix = pd.DataFrame(False, index=groups, columns=groups)
    for _, row in tukey_df.iterrows():
        g1 = str(row["group1"])
        g2 = str(row["group2"])
        reject = bool(row["reject"])
        sig_matrix.loc[g1, g2] = reject
        sig_matrix.loc[g2, g1] = reject

    letter_sets = {g: {i} for i, g in enumerate(groups)}
    next_letter = n_groups

    changed = True
    max_iter = 200
    iteration = 0

    while changed and iteration < max_iter:
        changed = False
        iteration += 1

        for i_g in range(n_groups):
            for j_g in range(i_g + 1, n_groups):
                g1 = groups[i_g]
                g2 = groups[j_g]

                if sig_matrix.loc[g1, g2]:
                    continue

                if letter_sets[g1].intersection(letter_sets[g2]):
                    continue

                found = False

                for L in list(letter_sets[g1]):
                    ok = True
                    for g_other in groups:
                        if g_other == g2:
                            continue
                        if L in letter_sets[g_other] and sig_matrix.loc[g2, g_other]:
                            ok = False
                            break
                    if ok:
                        letter_sets[g2].add(L)
                        found = True
                        changed = True
                        break

                if found:
                    continue

                for L in list(letter_sets[g2]):
                    ok = True
                    for g_other in groups:
                        if g_other == g1:
                            continue
                        if L in letter_sets[g_other] and sig_matrix.loc[g1, g_other]:
                            ok = False
                            break
                    if ok:
                        letter_sets[g1].add(L)
                        found = True
                        changed = True
                        break

                if found:
                    continue

                letter_sets[g1].add(next_letter)
                letter_sets[g2].add(next_letter)
                next_letter += 1
                changed = True

    # Absorption
    all_letters = set()
    for s in letter_sets.values():
        all_letters.update(s)

    for L in sorted(all_letters):
        groups_with_L = [g for g in groups if L in letter_sets[g]]
        if len(groups_with_L) <= 1:
            continue

        can_remove = True
        for g in groups_with_L:
            if len(letter_sets[g]) <= 1:
                can_remove = False
                break

        if not can_remove:
            continue

        trial = {g: letter_sets[g] - {L} for g in groups_with_L}
        ok = True
        for i_g in range(len(groups_with_L)):
            if not ok:
                break
            for j_g in range(i_g + 1, len(groups_with_L)):
                g1 = groups_with_L[i_g]
                g2 = groups_with_L[j_g]
                if sig_matrix.loc[g1, g2]:
                    continue
                s1 = trial.get(g1, letter_sets[g1])
                s2 = trial.get(g2, letter_sets[g2])
                if not s1.intersection(s2):
                    ok = False
                    break

        if ok:
            for g in groups_with_L:
                letter_sets[g].discard(L)

    # Map to letters
    used_ids = sorted(set().union(*letter_sets.values()))
    letters_str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    id_to_char = {}
    for idx, lid in enumerate(used_ids):
        id_to_char[lid] = letters_str[idx] if idx < len(letters_str) else f"L{idx}"

    cld_strings = {
        g: "".join(sorted(id_to_char[lid] for lid in letter_sets[g]))
        for g in groups
    }

    cld_df = pd.DataFrame({
        "MLG_label": groups,
        "CLD": [cld_strings[g] for g in groups],
    })
    st.dataframe(cld_df, use_container_width=True)

    # --------------------------------------------------------
    # Boxplot WITH CLD letters
    # --------------------------------------------------------
    fig_box2, ax2 = plt.subplots(figsize=FIGSIZE["boxplot"])

    sns.boxplot(
        data=mlg_df,
        x="MLG_label",
        y=trait_col,
        order=mlg_order,
        ax=ax2,
        width=0.6,
        showfliers=False,
        linewidth=1.2,
        boxprops=dict(edgecolor="black"),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )

    sns.stripplot(
        data=mlg_df,
        x="MLG_label",
        y=trait_col,
        order=mlg_order,
        ax=ax2,
        color="black",
        alpha=0.25,
        size=3,
        jitter=0.15,
    )

    ax2.tick_params(axis="both", which="both", labelsize=10, length=0)

    for spine in ax2.spines.values():
        spine.set_visible(False)

    data_min = float(mlg_df[trait_col].min())
    data_max = float(mlg_df[trait_col].max())
    data_range = data_max - data_min if data_max > data_min else 1.0

    y_text = data_max + 0.05 * data_range
    ax2.set_ylim(data_min - 0.05 * data_range, data_max + 0.25 * data_range)

    for i, tick in enumerate(ax2.get_xticklabels()):
        raw = tick.get_text().split("\n")[0].strip()
        if raw in cld_strings:
            ax2.text(
                i,
                y_text,
                cld_strings[raw],
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

    ax2.set_xlabel("Multi-locus genotype (haplotype)")
    ax2.set_ylabel(trait_col)
    ax2.set_title("Phenotype distribution across haplotypes")

    new_xticklabels = [
        f"{lbl.get_text()}\n(n={hap_counts.get(lbl.get_text(), 0)})"
        for lbl in ax2.get_xticklabels()
    ]

    ax2.set_xticks(range(len(new_xticklabels)))
    ax2.set_xticklabels(new_xticklabels, rotation=0)

    st.pyplot(fig_box2)
    export_matplotlib(fig_box2, f"haplotype_boxplot_{trait_col}", label_prefix="Download boxplot")
    plt.close(fig_box2)

    # --------------------------------------------------------
    # Forest plot
    # --------------------------------------------------------
    st.markdown("#### Haplotype Effect Sizes (Forest Plot)")

    import scipy.stats as _scipy_st_forest
    _forest_data = []
    for _h in mlg_order:
        _vals = (
            mlg_df.loc[mlg_df["MLG_label"] == _h, trait_col]
            .astype(float).dropna().values
        )
        if len(_vals) >= 2:
            _mean_h = float(np.mean(_vals))
            _se_h = float(_scipy_st_forest.sem(_vals))
            _forest_data.append({
                "Haplotype": _h,
                "n": len(_vals),
                "Mean": _mean_h,
                "SE": _se_h,
                "CI95_lo": _mean_h - 1.96 * _se_h,
                "CI95_hi": _mean_h + 1.96 * _se_h,
            })

    if len(_forest_data) >= 2:
        _forest_df = pd.DataFrame(_forest_data)

        _y_all_f = mlg_df[trait_col].astype(float).dropna().values
        _gm_f = float(np.mean(_y_all_f))
        _ss_tot_f = float(np.sum((_y_all_f - _gm_f) ** 2))
        _ss_bet_f = float(sum(
            r["n"] * (r["Mean"] - _gm_f) ** 2
            for _, r in _forest_df.iterrows()
        ))
        _eta2_vis = _ss_bet_f / _ss_tot_f if _ss_tot_f > 0 else float("nan")

        if np.isfinite(_eta2_vis):
            st.metric("Effect size (η²)", f"{_eta2_vis:.3f}",
                      help="Proportion of phenotypic variance explained by haplotype group")

        _x_range = (
            _forest_df["CI95_hi"].max() - _forest_df["CI95_lo"].min()
        ) or 1.0
        _fig_forest, _ax_forest = plt.subplots(
            figsize=(FIGSIZE["forest"][0], max(3, len(_forest_data) * 0.75))
        )
        for _i, _r in _forest_df.iterrows():
            _col_f = PALETTE_CYCLE[_i % len(PALETTE_CYCLE)]
            _y_pos = len(_forest_data) - 1 - _i
            _ax_forest.plot(
                [_r["CI95_lo"], _r["CI95_hi"]], [_y_pos, _y_pos],
                color=_col_f, linewidth=2.5, solid_capstyle="round"
            )
            _ax_forest.plot(
                _r["Mean"], _y_pos, "o",
                color=_col_f, markersize=9, zorder=5
            )
            _ax_forest.text(
                _r["CI95_hi"] + 0.03 * _x_range,
                _y_pos,
                f"n={_r['n']}",
                va="center", fontsize=9, color="grey"
            )

        _ax_forest.axvline(
            _gm_f, color="grey", linestyle="--",
            linewidth=1, alpha=0.7, label="Grand mean"
        )
        _ax_forest.set_yticks(
            range(len(_forest_data) - 1, -1, -1)
        )
        _ax_forest.set_yticklabels(
            _forest_df["Haplotype"].tolist(),
        )
        _ax_forest.set_xlabel(trait_label)
        _title_f = (
            f"Haplotype Effect Sizes \u2014 \u03b7\u00b2 = {_eta2_vis:.3f}"
            if np.isfinite(_eta2_vis)
            else "Haplotype Effect Sizes"
        )
        _ax_forest.set_title(_title_f)
        _ax_forest.legend()
        for _sp in _ax_forest.spines.values():
            _sp.set_visible(False)
        _ax_forest.tick_params(axis="both", length=0)
        plt.tight_layout()
        st.pyplot(_fig_forest)
        export_matplotlib(_fig_forest,
                          f"haplotype_forest_{block_chr}_{block_start}_{block_end}",
                          label_prefix="Download forest plot")
        plt.close(_fig_forest)

        st.download_button(
            "Download haplotype effects (CSV)",
            _forest_df.to_csv(index=False).encode("utf-8"),
            file_name=(
                f"Haplotype_effects_{block_chr}"
                f"_{block_start}_{block_end}.csv"
            ),
            mime="text/csv",
            key=f"dl_forest_{selected_block}",
        )


def _render_zip_export(haplo_df_auto, hap_gwas_df, trait_col):
    """One-click analysis bundle (ZIP)."""
    with st.expander("Export: one-click analysis bundle (ZIP)", expanded=False):
        st.markdown(
            "Creates a single ZIP containing the main tables produced on this page "
            "(LD blocks and haplotype analysis results with η²/MAF)."
        )

        if st.button("Build ZIP bundle"):
            buf_zip = BytesIO()

            hap_out = hap_gwas_df

            readme = textwrap.dedent(f"""
            LD & Haplotype Analysis Bundle

            Trait: {trait_col}

            Notes:
            - LD blocks are correlated due to LD; BH-FDR on blocks can be conservative.
            - Permutation-calibrated block p-values (if computed) are empirical robustness checks.
            - EtaSq is η² (SS_between / SS_total) for the tested MLG groups.
            - maf_median/maf_mean are computed from 0/1/2 dosage as minor allele frequency per SNP, summarized per block.

            Contents:
            - LD_clusters_peakcentric.csv (if available)
            - Haplotype_GWAS_blocks.csv (haplotype/MLG GWAS table with η²/MAF)
            """)

            with zipfile.ZipFile(buf_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr("README.txt", readme)

                if isinstance(haplo_df_auto, pd.DataFrame) and not haplo_df_auto.empty:
                    z.writestr(
                        f"LD_clusters_peakcentric__{trait_col}.csv",
                        haplo_df_auto.to_csv(index=False)
                    )

                if isinstance(hap_out, pd.DataFrame) and not hap_out.empty:
                    z.writestr(
                        f"Haplotype_GWAS__{trait_col}.csv",
                        hap_out.to_csv(index=False)
                    )

            st.download_button(
                "Download ZIP bundle",
                data=buf_zip.getvalue(),
                file_name=f"LD_Haplotype_bundle__{trait_col}.zip",
                mime="application/zip",
                key="dl_zip_bundle"
            )


