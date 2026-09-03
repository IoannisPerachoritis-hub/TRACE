"""Shared window selector for the LD-page tabs (T-08).

The Regional Plot and Local LD tabs are two renderings of ONE window selection.
Streamlit forbids the same keyed widget in two ``st.tabs`` bodies (DuplicateWidgetID,
and every tab body executes on each rerun), so this selector renders **once above
the tab bar** and both tabs consume the returned ``RegionWindow`` — a user flips
between the two views without re-selecting.

Behaviour-preserving extraction of the inline selector that used to live in
``tab_local_ld.py``: the same lead-SNP selectbox, snap-to-block checkbox, buffer
slider, and region-determination logic, so the local-LD window is unchanged.
"""
import dataclasses

import pandas as pd
import streamlit as st

from gwas.ld import _guess_lead_col
from . import LDContext


@dataclasses.dataclass(frozen=True)
class RegionWindow:
    """One resolved genomic window, shared by the Regional Plot and Local LD tabs."""

    lead_snp: str
    chr: str
    start_bp: int
    end_bp: int
    core_start: int | None   # LD-block start (bp) when snapped to a block, else None
    core_end: int | None     # LD-block end (bp) when snapped, else None
    use_block: bool
    label: str
    lead_pos: int
    buffer_kb: int
    block_snp_ids: str = ""  # raw comma-joined SNP_IDs when a detected block is picked;
                             # lets Local LD extract members by ID (reproduces Block Heatmaps)


def select_window(ctx: LDContext) -> "RegionWindow | None":
    """Render the lead-SNP / snap-to-block / buffer controls once, above the tabs.

    Returns a ``RegionWindow``, or ``None`` when no lead SNP can be resolved (empty
    GWAS table, or the chosen SNP is missing from it) — the caller shows the reason.
    """
    gwas_df = ctx.gwas_df
    if gwas_df is None or getattr(gwas_df, "empty", True):
        st.info("Run a GWAS first. The regional / local views read the GWAS results in session.")
        return None

    top_snps = gwas_df.sort_values("PValue").head(200)
    if top_snps.empty:
        st.info("GWAS table seems empty. Run GWAS first.")
        return None

    lead_col = _guess_lead_col(ctx.haplo_df_auto)
    has_blocks = (
        isinstance(ctx.haplo_df_auto, pd.DataFrame)
        and not ctx.haplo_df_auto.empty
        and lead_col is not None
    )

    # Region source: pick a window by lead SNP (default) or snap to a detected block.
    if has_blocks:
        region_source = st.radio(
            "Region source", ["Lead SNP", "Detected block"], horizontal=True,
            key="ld_window_source",
            help="A window around a top SNP, or snap directly to a detected LD block "
                 "(Local LD then reproduces the Block Heatmaps view for that block).",
        )
    else:
        region_source = "Lead SNP"

    def _buffer_slider():
        # Seed the default ONCE; never pass value= to a keyed widget here. A
        # data-dependent value= (2x ld_decay_kb) would overwrite the user's manual
        # buffer whenever ld_decay_kb changes (e.g. after the LD Decay tab computes
        # decay) -- the "buffer change doesn't stick" lag. With no value=, the slider
        # reads/writes session_state directly and the seed applies only on first render.
        if "ld_window_buffer_kb" not in st.session_state:
            st.session_state["ld_window_buffer_kb"] = min(5000, max(10, int((ctx.ld_decay_kb or 150) * 2)))
        return st.slider(
            "Buffer (kb)", min_value=10, max_value=5000, step=10,
            key="ld_window_buffer_kb",
            help="Extends the window this many kb on each side of the block (or SNP if no block).",
        )

    # ----- Detected-block mode: snap to a chosen block, member-aware -----
    if region_source == "Detected block":
        dfb = ctx.haplo_df_auto.reset_index(drop=True)

        def _fmt_block(i):
            r = dfb.iloc[int(i)]
            s, e = int(r["Start (bp)"]), int(r["End (bp)"])
            return f"[{int(i)}] Chr{r['Chr']}:{s:,}-{e:,} ({(e - s) / 1000:.1f} kb)"

        col_blk, col_buf = st.columns([2, 1])
        with col_blk:
            sel = st.selectbox("Detected LD block", options=list(range(len(dfb))),
                               format_func=_fmt_block, key="ld_window_block_select")
        with col_buf:
            buffer_kb = _buffer_slider()

        row = dfb.iloc[int(sel)]
        chr_sel = str(row["Chr"])
        core_start = int(row["Start (bp)"])
        core_end = int(row["End (bp)"])
        block_snp_ids = str(row.get("SNP_IDs", "") or "")
        member_ids = [s.strip() for s in block_snp_ids.split(",") if s.strip()]
        # Lead for the Regional plot = top-p member (in-block, a real association lead).
        # It does not affect Local LD, whose extraction is by SNP_IDs (member-aware).
        sub = gwas_df[gwas_df["SNP"].astype(str).isin(member_ids)].sort_values("PValue")
        if not sub.empty:
            lead_snp = str(sub.iloc[0]["SNP"])
            lead_pos = int(sub.iloc[0]["Pos"])
        else:
            lead_snp = str(top_snps.iloc[0]["SNP"])
            lead_pos = (core_start + core_end) // 2
        extra = buffer_kb * 1000
        start_bp = max(0, core_start - extra)
        end_bp = core_end + extra
        label = (f"Detected block Chr{chr_sel}:{core_start:,}-{core_end:,} "
                 f"({len(member_ids)} SNPs) ± {buffer_kb} kb buffer")
        st.markdown(f"**Window:** {label}")
        return RegionWindow(
            lead_snp=lead_snp, chr=chr_sel, start_bp=int(start_bp), end_bp=int(end_bp),
            core_start=core_start, core_end=core_end, use_block=True, label=label,
            lead_pos=int(lead_pos), buffer_kb=int(buffer_kb), block_snp_ids=block_snp_ids,
        )

    # ----- Lead-SNP mode (default): SNP-centered window -----
    # The old "Snap to LD block" checkbox is gone — the radio's "Detected block"
    # source expresses that intent, and the Regional Plot now derives block shading
    # from whether the lead sits inside a detected block (no two controls to disagree).
    col_lead, col_buf = st.columns([2, 1])
    with col_lead:
        lead_snp = st.selectbox(
            "Lead SNP (top 200 by P-value):",
            options=top_snps["SNP"].tolist(),
            index=0,
            key="ld_window_lead_snp",
            help="Shared by the Regional Plot and Local LD tabs.",
        )
    with col_buf:
        buffer_kb = _buffer_slider()

    snp_row = gwas_df.loc[gwas_df["SNP"] == lead_snp]
    if snp_row.empty:
        st.warning(f"SNP {lead_snp} not found in GWAS table.")
        return None

    chr_sel = str(snp_row.iloc[0]["Chr"])
    pos_snp = int(snp_row.iloc[0]["Pos"])
    start_bp = pos_snp - buffer_kb * 1000
    end_bp = pos_snp + buffer_kb * 1000
    label = f"SNP-centered window: Chr{chr_sel}:{pos_snp:,} ± {buffer_kb} kb"

    st.markdown(f"**Window:** {label}")
    return RegionWindow(
        lead_snp=str(lead_snp),
        chr=chr_sel,
        start_bp=int(start_bp),
        end_bp=int(end_bp),
        core_start=None,
        core_end=None,
        use_block=False,
        label=label,
        lead_pos=pos_snp,
        buffer_kb=int(buffer_kb),
        block_snp_ids="",
    )
