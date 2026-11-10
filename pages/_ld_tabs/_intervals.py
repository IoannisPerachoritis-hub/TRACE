"""The run's candidate intervals for the Post-GWAS tabs (cached).

One list for the Gene Annotation tab and the Regional Plot: every LD block and every run
of unblocked significant SNPs of the run on screen, each with its candidate interval by
the CLI's rule (gwas.isolated.candidate_interval_table: the marker before the first SNP
to the marker after the last; at a chromosome edge 2 x the LD decay; over 5 Mb clamped
around the lead SNP), so the GUI lists the intervals the CLI annotates.
"""
import numpy as np
import streamlit as st

from gwas.isolated import candidate_interval_table, edge_flank_bp_from_decay
from gwas.significance import frame_for_rule, rule_from_streamlit


@st.cache_data(show_spinner=False, max_entries=8)
def _candidate_intervals(blocks, gwas_df, rule_label, meff_val, custom_thresh,
                         chroms, positions, sid, edge_flank_bp):
    rule = rule_from_streamlit(rule_label, len(gwas_df), meff_val, custom_thresh=custom_thresh)
    return candidate_interval_table(blocks, frame_for_rule(gwas_df, rule), rule,
                                    chroms, positions, sid, edge_flank_bp=edge_flank_bp)


def run_candidate_intervals(ctx):
    """Every candidate interval of the run on screen (ctx's blocks, GWAS results and
    significance rule, over its post-QC markers), or None without GWAS results."""
    gwas_df = ctx.gwas_df
    if gwas_df is None or len(gwas_df) == 0 or ctx.chroms is None:
        return None
    return _candidate_intervals(
        ctx.haplo_df_auto, gwas_df, ctx.sig_rule_label, ctx.meff_val, ctx.custom_thresh,
        np.asarray(ctx.chroms), np.asarray(ctx.positions), np.asarray(ctx.sid),
        edge_flank_bp_from_decay(ctx.ld_decay_kb))
