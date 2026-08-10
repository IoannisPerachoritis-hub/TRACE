"""Tab — Significant SNPs (complete, no-omission view).

Surfaces EVERY reporting-significant SNP, each marked as belonging to an LD block
or as *unblocked* (formed no block), with its candidate interval and Block_Status.
This is the R1.6/R3.7 deliverable: a block-only view hides the unblocked SNPs;
this tab shows all of them as data. The table is always complete — never
truncated. Pure glue over ``gwas.sigtable`` + ``gwas.significance``; the heavy
work + tests live in those modules.
"""

import streamlit as st

from . import LDContext

_SIG_LABELS = [
    "M_eff — Li & Ji (LD-aware Bonferroni)",
    "Bonferroni (α = 0.05)",
    "FDR (q < 0.05)",
]

_GLOSSARY = """
**Block_Status** — how each significant SNP relates to the LD-block detector's output:
- `in_block` — a member of a detected LD block.
- `unblocked_not_seeded` — significant, but below the block **seeding** threshold, so the
  detector never considered it. *These are the SNPs a block-only view silently drops.*
- `unblocked_isolated` / `unblocked_monomorphic_window` — seeded but no neighbours / no variation nearby.
- `unblocked_low_ld` — seeded with neighbours, but they did not reach the LD threshold to form a block
  (an inference from the block table; the detector is never re-run).

**Candidate interval** — for unblocked SNPs, the flanking typed-marker interval; its width reflects
marker density, not association strength. **Effect_Source** — whether a structure-adjusted β_MLM was
available (`both` / `MLM_only` / `OLS_only` / `unavailable`). The table is **complete**: no significant
SNP is omitted, regardless of block membership.
"""


def render(ctx: LDContext):
    st.subheader("Significant SNPs")
    st.caption(
        "Every SNP passing the significance threshold — each marked in-block or **unblocked**, "
        "with its candidate interval. Complete: no significant SNP is omitted."
    )

    gwas_df = ctx.gwas_df
    if gwas_df is None or getattr(gwas_df, "empty", True):
        st.info("Run a GWAS first — this tab reads the GWAS results in session.")
        return

    from gwas.sigtable import build_significant_snp_table, project_unblocked
    from gwas.significance import rule_from_streamlit

    default_idx = _SIG_LABELS.index(ctx.sig_rule_label) if ctx.sig_rule_label in _SIG_LABELS else 0
    label = st.selectbox("Significance threshold", _SIG_LABELS, index=default_idx,
                         key="sigtab_threshold",
                         help="Defaults to the rule chosen on the GWAS page.")
    rule = rule_from_streamlit(label, len(gwas_df), ctx.meff_val)

    edge_flank_bp = int((ctx.ld_decay_kb or 300) * 1000)
    ld_decay_bp = int(ctx.ld_decay_kb * 1000) if ctx.ld_decay_kb else None
    try:
        sig = build_significant_snp_table(
            gwas_df, ctx.haplo_df_auto, rule, ctx.chroms, ctx.positions, ctx.sid,
            geno_dosage_raw=ctx.geno_dosage_raw, genes=None,
            seed_p_used=1e-5, top_n_used=0, edge_flank_bp=edge_flank_bp,
            ld_decay_bp=ld_decay_bp, geno_encoding=ctx.geno_encoding or "dosage012",
        )
    except Exception as e:  # never crash the page on a malformed frame
        st.error(f"Could not build the significant-SNP table: {e}")
        return

    if sig.empty:
        st.info(f"No SNPs pass {rule.label}.")
        return

    n_unblocked = int((sig["Block_Status"] != "in_block").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Significant SNPs", len(sig))
    c2.metric("In an LD block", len(sig) - n_unblocked)
    c3.metric("Unblocked", n_unblocked,
              help="Significant SNPs that formed no LD block — hidden by a block-only view.")
    st.markdown(f"**Threshold applied:** {rule.label}")

    # Complete table — never .head(); wide table scrolls inside the widget.
    st.dataframe(sig, use_container_width=True, height=430)
    st.download_button("Download Significant_SNPs.csv", sig.to_csv(index=False),
                       file_name="Significant_SNPs.csv", mime="text/csv")

    if n_unblocked:
        with st.expander(f"Unblocked SNPs ({n_unblocked}) — the SNPs a block-only view would hide",
                         expanded=True):
            unb = project_unblocked(sig)
            st.dataframe(unb, use_container_width=True)
            st.download_button("Download Unblocked_SNPs.csv", unb.to_csv(index=False),
                               file_name="Unblocked_SNPs.csv", mime="text/csv",
                               key="dl_unblocked_snps")

    with st.expander("What the columns mean"):
        st.markdown(_GLOSSARY)
