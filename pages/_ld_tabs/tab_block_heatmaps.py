"""Tab 1 — LD-block Heatmaps (Primary)."""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit.runtime.scriptrunner import StopException
from utils.pub_theme import LD_HEATMAP_CMAP, FIGSIZE, export_matplotlib

from gwas.ld import extract_block_geno_for_paper
from . import LDContext


class _TabExit(Exception):
    pass


def render(ctx: LDContext, get_r2_cached):
    try:
        st.subheader("LD-block Heatmaps")

        df_blocks = ctx.haplo_df_auto

        if df_blocks is None or df_blocks.empty:
            st.info("No peak-centric LD blocks available yet.")
            st.caption(
                "Go to the 'Genome-wide' tab and run automatic detection, "
                "or rely on automatic detection at page load."
            )
            return

        # --- Robust LD block selector (prevents label mismatch) ---
        dfb = df_blocks.reset_index(drop=True).copy()
        dfb["BlockUID"] = dfb.index.astype(int)

        def _fmt_block(i: int) -> str:
            r = dfb.iloc[int(i)]
            chr_ = str(r["Chr"])
            s = int(r["Start (bp)"])
            e = int(r["End (bp)"])
            kb = (e - s) / 1000.0
            return f"[{int(i)}] Chr{chr_}:{s:,}-{e:,} ({kb:.3f} kb)"

        selected_uid = st.selectbox(
            "Select LD block:",
            options=dfb["BlockUID"].tolist(),
            format_func=_fmt_block,
            key=f"ld_block_select_{ctx.trait_col}",
        )

        row = dfb.iloc[int(selected_uid)]

        chr_block = str(row["Chr"])
        start_bp = int(row["Start (bp)"])
        end_bp = int(row["End (bp)"])

        st.markdown(
            f"**Selected block:** Chr{chr_block}:{start_bp:,}–{end_bp:,}  "
            f"({(end_bp - start_bp) / 1000:.1f} kb)"
        )

        # ------------------------------------------------------------
        # Publication-safe SNP extraction (consistent with haplotype GWAS)
        # ------------------------------------------------------------
        pheno_df_for_ld = st.session_state.get("pheno_used_for_hap", ctx.pheno_df)

        if pheno_df_for_ld is None:
            st.warning("Phenotype data for LD analysis not available. Please rerun GWAS.")
            raise _TabExit()

        keep_mask_ld = ctx.keep_mask

        block_geno, block_pos, block_sids = extract_block_geno_for_paper(
            geno_imputed=ctx.geno_ld,
            chroms=ctx.chroms,
            positions=ctx.positions,
            sid=ctx.sid,
            block_chr=chr_block,
            start_bp=start_bp,
            end_bp=end_bp,
            sample_keep_mask=keep_mask_ld,
            maf_threshold=st.session_state.get("maf_ld", 0.01),
            snp_ids=row.get("SNP_IDs", None),
        )
        miss_rate = 1.0 - np.isfinite(block_geno).mean()
        if miss_rate > 0:
            st.warning(f"Genotype missingness in this block: {miss_rate:.2%}")

        # -------------------------------------------------
        # LD QC: report pairwise genotype overlap
        # -------------------------------------------------
        mask = np.isfinite(block_geno).astype(np.int32)
        vc = mask.T @ mask

        min_pairN = int(np.min(vc[np.triu_indices_from(vc, k=1)])) if block_geno.shape[1] > 1 else 0

        st.caption(
            f"LD QC: samples={block_geno.shape[0]}, SNPs={block_geno.shape[1]}, "
            f"min pairwise N={min_pairN}"
        )

        MIN_PAIR_N_FOR_PLOT = 10

        if (block_geno.shape[1] > 1) and (min_pairN < MIN_PAIR_N_FOR_PLOT):
            st.warning(
                f"Low SNP overlap (min pairwise N={min_pairN}). "
                "LD may be noisy — plotting anyway."
            )

        min_pairN_warn = max(5, int(0.05 * block_geno.shape[0]))
        if (block_geno.shape[1] > 1) and (min_pairN < min_pairN_warn):
            st.warning(
                f"Low genotype overlap detected (min pairwise N={min_pairN} < {min_pairN_warn}). "
                "LD may be noisy; consider increasing MAF/call-rate filters or using imputation."
            )

        if block_geno.shape[1] < 2:
            st.warning("Too few SNPs remain after QC — skipping this block.")
            st.stop()

        # Ensure consistent genomic ordering BEFORE LD calculation
        order2 = np.argsort(block_pos)
        block_geno = block_geno[:, order2]
        block_pos = block_pos[order2]
        block_sids = block_sids[order2]

        # Compute LD AFTER sorting
        r2 = get_r2_cached(
            block_geno, block_pos, min_pair_n=20,
            cache_key=f"r2::block::{chr_block}:{start_bp}-{end_bp}::{block_geno.shape[1]}snps",
        )

        seg_start_bp = int(block_pos.min())
        seg_end_bp = int(block_pos.max())

        # --- Display settings ---
        show_labels = ctx.show_ld_labels
        label_type = st.radio("Axis labels", ["Genomic position (kb)", "SNP IDs"], index=0)

        if label_type == "Genomic position (kb)":
            axis_labels = np.round(block_pos / 1000.0, 1)
            axis_title = "Position (kb)"
        else:
            axis_labels = np.array([f"{chr_block}_{int(p)}" for p in block_pos])
            axis_title = "SNP ID"

        # --- Triangular LD heatmap ---
        st.markdown("### LD Heatmap (r²)")

        fig, ax = plt.subplots(figsize=FIGSIZE["heatmap"])

        mask_upper = np.triu(np.ones_like(r2, dtype=bool))
        annot_vals = show_labels and (r2.shape[0] <= 25)
        sns.heatmap(
            r2,
            mask=mask_upper,
            cmap=LD_HEATMAP_CMAP,
            vmin=0,
            vmax=1,
            annot=annot_vals,
            fmt=".2f",
            annot_kws={"fontsize": 8},
            square=True,
            linewidths=0.15,
            linecolor="white",
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        # ---- Colorbar styling ----
        cbar = ax.collections[0].colorbar
        cbar.set_label("LD (r\u00b2)", fontsize=12)

        ax.tick_params(axis="both", which="both", length=0)
        ax.set_aspect("equal")

        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()

        # Enforce HARD cap on number of axis labels
        max_labels = 10
        n_labels = len(axis_labels)

        if n_labels <= max_labels:
            tick_idx = np.arange(n_labels)
        else:
            tick_idx = np.linspace(0, n_labels - 1, max_labels, dtype=int)

        ax.set_xticks(tick_idx + 0.5)
        ax.set_xticklabels(
            axis_labels[tick_idx],
            rotation=45,
            ha="right",
        )

        ax.set_yticks([])
        ax.set_ylabel("")

        ax.set_xlabel(axis_title)
        ax.set_title(f"LD block \u2014 Chr{chr_block}:{seg_start_bp:,}\u2013{seg_end_bp:,}")

        st.caption("LD (r²) computed from imputed dosages; haplotype labels use hard-called genotypes.")
        st.pyplot(fig)

        # --- Export buttons ---
        export_matplotlib(fig, f"LD_block_segment_Chr{chr_block}_{seg_start_bp}_{seg_end_bp}",
                          label_prefix="Download LD heatmap")

        st.download_button(
            "Download LD matrix (CSV)",
            pd.DataFrame(r2, index=axis_labels, columns=axis_labels).to_csv().encode(),
            file_name=f"LD_block_matrix_Chr{chr_block}_{start_bp}_{end_bp}.csv",
            mime="text/csv"
        )

    except StopException:
        pass
