"""Tab: Significant SNPs (complete, no-omission view).

Surfaces EVERY reporting-significant SNP, each marked as belonging to an LD block
or as *unblocked* (formed no block), with its candidate interval and Block_Status.
A block-only view hides the unblocked SNPs;
this tab shows all of them as data. The table is always complete, never
truncated. Pure glue over ``gwas.sigtable`` + ``gwas.significance``; the heavy
work + tests live in those modules.
"""

import streamlit as st

from . import LDContext

_SIG_LABELS = [
    "M_eff: Li & Ji (LD-aware Bonferroni)",
    "Bonferroni (α = 0.05)",
    "FDR (q < 0.05)",
    "Custom p-value",
]

_GLOSSARY = """
**Block_Status**: how each significant SNP relates to the LD-block detector's output:
- `in_block`: a member of a detected LD block.
- `unblocked_not_seeded`: significant, but below the block **seeding** threshold, so the
  detector never considered it. *These are the SNPs a block-only view silently drops.*
- `unblocked_isolated` / `unblocked_monomorphic_window`: seeded but no neighbours / no variation nearby.
- `unblocked_low_ld`: seeded with neighbours, but they did not reach the LD threshold to form a block
  (an inference from the block table; the detector is never re-run).

**Candidate interval**: for unblocked SNPs, the flanking typed-marker interval; its width reflects
marker density, not association strength. **Effect_Source**: whether a structure-adjusted β_MLM was
available (`both` / `MLM_only` / `OLS_only` / `unavailable`). The table is **complete**: no significant
SNP is omitted, regardless of block membership.
"""


def render(ctx: LDContext):
    st.subheader("Significant SNPs")

    gwas_df = ctx.gwas_df
    if gwas_df is None or getattr(gwas_df, "empty", True):
        st.info("Run a GWAS first. This tab reads the GWAS results in session.")
        return

    from gwas import guipost
    from gwas.sigtable import build_significant_snp_table, project_unblocked
    from gwas.significance import frame_for_rule, rule_from_streamlit

    default_idx = _SIG_LABELS.index(ctx.sig_rule_label) if ctx.sig_rule_label in _SIG_LABELS else 0
    label = st.selectbox("Significance threshold", _SIG_LABELS, index=default_idx,
                         key="sigtab_threshold",
                         help="Defaults to the rule chosen on the GWAS page.")
    _tab_custom_thresh = None
    if label.startswith("Custom"):
        _tab_custom_thresh = st.number_input(
            "Custom p-value", min_value=1e-300, max_value=1.0,
            value=float(ctx.custom_thresh) if ctx.custom_thresh else 5e-8,
            format="%.1e", key="sigtab_custom_thresh",
            help="SNPs with p < this value are significant (defaults to the GWAS-page value).",
        )
    rule = rule_from_streamlit(label, len(gwas_df), ctx.meff_val, custom_thresh=_tab_custom_thresh)

    # the CLI's edge rule (2 x LD decay, else 300 kb), so the intervals are the CLI's
    from gwas.isolated import edge_flank_bp_from_decay
    edge_flank_bp = edge_flank_bp_from_decay(ctx.ld_decay_kb)
    ld_decay_bp = int(ctx.ld_decay_kb * 1000) if ctx.ld_decay_kb else None
    geno_encoding = ctx.geno_encoding or "dosage012"
    seed_p_used, top_n_used = 1e-5, 0
    # built once per input and kept for reruns: a bounded session cache keyed by the table,
    # the labels, the GWAS results, the blocks, the rule and every argument of the build
    sig_key = guipost.sigtable_key(ctx, rule, edge_flank_bp, ld_decay_bp, geno_encoding,
                                   seed_p_used, top_n_used)
    sig_cache = guipost.bounded(st.session_state, "_sigtable_cache")
    sig = guipost.recall(sig_cache, sig_key)
    if sig is None:
        try:
            # frame_for_rule: this tab's own custom p-value, not a stored Significant_Custom
            sig = build_significant_snp_table(
                frame_for_rule(gwas_df, rule), ctx.haplo_df_auto, rule, ctx.chroms, ctx.positions, ctx.sid,
                geno_dosage_raw=ctx.geno_dosage_raw, genes=None,
                seed_p_used=seed_p_used, top_n_used=top_n_used, edge_flank_bp=edge_flank_bp,
                ld_decay_bp=ld_decay_bp, geno_encoding=geno_encoding,
            )
        except Exception as e:  # never crash the page on a malformed frame
            st.error(f"Could not build the significant-SNP table: {e}")
            return
        guipost.remember(sig_cache, sig_key, sig, guipost.RESULT_CACHE_MAX)
    sig = sig.copy()           # the cache keeps its own table

    if sig.empty:
        st.info(f"No SNPs pass {rule.label}.")
        return

    n_unblocked = int((sig["Block_Status"] != "in_block").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Significant SNPs", len(sig))
    c2.metric("In an LD block", len(sig) - n_unblocked)
    c3.metric("Unblocked", n_unblocked,
              help="Significant SNPs that formed no LD block, hidden by a block-only view.")
    st.markdown(f"**Threshold applied:** {rule.label}")

    # Complete table: never .head(); wide table scrolls inside the widget.
    st.dataframe(sig, use_container_width=True, height=430)
    st.download_button("Download Significant_SNPs.csv", sig.to_csv(index=False),
                       file_name="Significant_SNPs.csv", mime="text/csv")

    if n_unblocked:
        with st.expander(f"Unblocked SNPs ({n_unblocked}): the SNPs a block-only view would hide",
                         expanded=True):
            unb = project_unblocked(sig)
            st.dataframe(unb, use_container_width=True)
            st.download_button("Download Unblocked_SNPs.csv", unb.to_csv(index=False),
                               file_name="Unblocked_SNPs.csv", mime="text/csv",
                               key="dl_unblocked_snps")

    with st.expander("What the columns mean"):
        st.markdown(_GLOSSARY)
