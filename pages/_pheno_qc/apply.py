"""Apply, Remove and Reset all for the GWAS page's phenotype transformations.

The preview (diagnostics.render) never changes the GWAS phenotype. A transformation is
used only after Apply, which records it for that trait in
``st.session_state["pheno_applied_transforms"]`` as ``{trait: slug}``, with the CLI's
``--norm`` slugs. The GWAS page passes each trait's slug as that trait's normalisation,
which the pipeline applies where the CLI applies ``--norm``: to the trait's own analysis
samples, after phenotype QC. The phenotype table itself always holds the uploaded values.

The buttons act through callbacks, which Streamlit runs before the page script, so the
status line, this list and the run all see a click in the same rerun.
"""

import numpy as np
import streamlit as st

from .stats import SLUG_NAMES, TRANSFORM_SLUGS, transform_refusal

STATE_KEY = "pheno_applied_transforms"


def applied_transforms(state_key=STATE_KEY):
    """The applied list, ``{trait: slug}`` (a copy; empty when nothing is applied)."""
    return dict(st.session_state.get(state_key) or {})


def _apply(state_key, trait, method, raw):
    # checked again here: a disabled button's callback can still be fired (AppTest does)
    if transform_refusal(raw, method) is not None:
        return
    st.session_state[state_key] = {**applied_transforms(state_key), trait: TRANSFORM_SLUGS[method]}


def _remove(state_key, trait):
    current = applied_transforms(state_key)
    current.pop(trait, None)
    st.session_state[state_key] = current


def _reset(state_key):
    st.session_state[state_key] = {}


def render(ctx, trait, method, *, key_prefix="gwas_pheno_qc", state_key=STATE_KEY):
    """Below the preview of (trait, method): the Apply button, then the applied list
    with a Remove button per trait, then Reset all. The list follows ``ctx.numeric_cols``."""
    current = applied_transforms(state_key)
    raw = ctx.pheno_df[trait].to_numpy(dtype=float)
    slug = TRANSFORM_SLUGS[method]
    in_use = SLUG_NAMES.get(current.get(trait, "none"), current.get(trait))
    st.caption(f"The plots above are a preview. The GWAS uses {in_use} for {trait}.")
    st.caption("An applied transformation is used by the GWAS scans only, as cli.py --norm "
               "is; the QC report and the haplotype tests use the uploaded values.")

    reason = transform_refusal(raw, method)
    if reason is not None and slug == "log":
        _bad = ctx.pheno_df.index[np.isfinite(raw) & (raw <= 0)].astype(str).tolist()
        if _bad:
            reason += (f" ({len(_bad)} sample(s): {', '.join(_bad[:5])}"
                       + (", ..." if len(_bad) > 5 else "") + ")")
    if reason is None and current.get(trait) == slug:
        reason = f"{SLUG_NAMES[slug]} is already applied to {trait}"
    st.button(
        f"Apply {SLUG_NAMES[slug]} to {trait}",
        key=f"{key_prefix}_apply",
        disabled=reason is not None,
        help=reason or (
            "Use this transformation as this trait's normalisation in every GWAS run "
            "(One-Click and the detailed view), as cli.py --norm does: the association "
            "scans, the subsampling and the per-SNP plots use it; the QC report and the "
            "haplotype tests keep the uploaded values."),
        on_click=_apply, args=(state_key, trait, method, raw),
    )
    if reason is not None:
        st.caption(f"Apply is not available: {reason}.")
    elif trait in current:
        st.caption(f"Applying replaces {SLUG_NAMES.get(current[trait], current[trait])} "
                   f"for {trait}.")

    st.markdown("**Applied transformations**")
    order = ([t for t in ctx.numeric_cols if t in current]
             + [t for t in current if t not in ctx.numeric_cols])
    if not order:
        st.caption("None applied: every trait uses raw values.")
    for t in order:
        c_name, c_btn = st.columns([4, 1])
        c_name.markdown(f"{t} -> {SLUG_NAMES.get(current[t], current[t])}")
        c_btn.button("Remove", key=f"{key_prefix}_remove__{t}",
                     on_click=_remove, args=(state_key, t))
    st.button("Reset all", key=f"{key_prefix}_reset_all", disabled=not current,
              on_click=_reset, args=(state_key,))
