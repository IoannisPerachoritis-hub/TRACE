"""Which run an LD-decay estimate comes from, as Post-GWAS names it.

Post-GWAS works on the GWAS page's single-trait run (Run GWAS): that run stores the
results the page shows and, in the same step, its LD-decay estimate. One-Click Full
Analysis writes its results to its ZIP only and hands nothing to Post-GWAS.
"""


def decay_run_label(run_summary):
    """'the GWAS page's Run GWAS for `<trait>` (<time>)' from the run's summary
    (st.session_state["gwas_run_summary"]), or a plain name without one."""
    if not isinstance(run_summary, dict) or not run_summary.get("trait"):
        return "the GWAS page's Run GWAS"
    when = run_summary.get("timestamp")
    return (f"the GWAS page's Run GWAS for `{run_summary['trait']}`"
            + (f" ({when})" if when else ""))
