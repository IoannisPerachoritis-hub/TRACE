# page_ld_analysis.py
import numpy as np
import pandas as pd
import streamlit as st
from utils.pub_theme import apply_matplotlib_theme, build_plotly_template
apply_matplotlib_theme()
build_plotly_template()
import hashlib
import logging

from gwas import guipost as _gp
from gwas.ld import filter_contained_blocks
from gwas.utils import canonicalize_chr, align_pheno_to_geno, resolve_trait_column, check_data_version
from pages._ld_tabs import LDContext
from pages._ld_tabs._decay import decay_run_label
from gwas.windows import ld_window_kb
from pages._ld_tabs import tab_local_ld
from pages._ld_tabs import tab_gene_annotation
from pages._ld_tabs import tab_genome_wide
from pages._ld_tabs import tab_significant_snps
from pages._ld_tabs import tab_regional
from pages._ld_tabs._window import select_window





def _hash_df(df: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.blake2b(h.tobytes(), digest_size=16).hexdigest()
def _sync_variant_axes(n_snps, chroms, positions, sid):
    """
    Ensure the genotype table's SNP count (``n_snps``, its columns) matches the
    chroms/positions/sid lengths. Fails hard on mismatch to prevent silent data
    corruption. The table itself is not read.
    """
    chroms = np.asarray(chroms, dtype=object)
    positions = np.asarray(positions)
    sid = np.asarray(sid, dtype=object)

    nG = int(n_snps)
    nC = len(chroms)
    nP = len(positions)
    nS = len(sid)

    if not (nG == nC == nP == nS):
        st.error(
            f"CRITICAL: Variant axis mismatch: "
            f"geno columns={nG}, chroms={nC}, positions={nP}, sid={nS}.\n\n"
            f"This indicates stale session state from a previous run. "
            f"Please rerun GWAS to rebuild all arrays from the same SNP set."
        )
        st.stop()

    return chroms, positions, sid

# --------------------------------------------------
# LD matrix cache (bounded; keyed by the table's identity and the window)
# --------------------------------------------------
def get_r2_cached(
    G: np.ndarray,
    pos: np.ndarray | None = None,
    min_pair_n: int = 20,
    cache_key: str | None = None,
):
    """
    Compute LD r² of the window ``G`` (samples x SNPs) with LRU caching.

    The tabs pass cache_key = gwas.guipost.r2_key(...), which names the table's identity,
    its rows and the window's SNPs. Without one the window's own values are the key.
    """
    from gwas.ld import pairwise_r2

    if cache_key is None:
        cache_key = _gp.window_r2_key(G, min_pair_n)

    cache = _gp.bounded(st.session_state, "r2_cache")
    MAX_R2_CACHE = 10

    hit = _gp.recall(cache, cache_key)
    if hit is not None:
        return hit

    r2 = pairwise_r2(np.asarray(G, float), min_pair_n=int(min_pair_n))
    r2 = np.clip(r2, 0.0, 1.0)
    return _gp.remember(cache, cache_key, r2, MAX_R2_CACHE)

# -------------------------------------------
# LD utilities from gwas/ld.py
# -------------------------------------------
from gwas.ld import (
    find_ld_clusters_genomewide,
    find_ld_blocks_from_genotypes,
)

# Gene annotation, LD decay, cross-trait comparison
try:
    from ld.annotation import (
        load_gene_annotation,
        annotate_ld_blocks,
        compute_ld_decay_by_chromosome,
        plot_ld_decay_matplotlib,
    )
    _HAS_ANNOTATION = True
except ImportError:
    try:
        from annotation import (
            load_gene_annotation,
            annotate_ld_blocks,
            compute_ld_decay_by_chromosome,
            plot_ld_decay_matplotlib,
        )
        _HAS_ANNOTATION = True
    except ImportError:
        _HAS_ANNOTATION = False

# ============================================================
# Helper: cumulative genomic positions for Manhattan-style plots
# ============================================================
from gwas.plotting import compute_cumulative_positions as _prepare_cumpos


def compute_block_qc_effects(**kwargs) -> pd.DataFrame:
    """The LD Blocks & Haplotypes tab's block effects (gwas.guipost.block_qc_effects: block
    length, SNP counts, MAF, eta^2 of the tested groups), with the session's PCs and selfing
    mode, as this page always read them. The tab passes the imputed table as it reads it."""
    return _gp.block_qc_effects(**kwargs, pcs=st.session_state.get("pcs", None),
                                strict_selfing=st.session_state.get("selfing_mode", False))


def _released():
    """The session's table file was closed after the GWAS page left it here (Clear session
    data, or the server made room for another dataset): say so plainly and stop."""
    st.info(_gp.RELEASED_MESSAGE)
    st.stop()

# --------------------------------------------------------
# 1. Sanity: require genotype / phenotype (GWAS resolved below)
# --------------------------------------------------------

def ld_analysis_page():
    st.title("Post-GWAS Analysis")
    check_data_version("ld_analysis")

    # Page-scoped controls (mega-block filter, haplotype permutations) are rendered
    # in the page body / their tabs, not the sidebar; see the "Block table filter"
    # row above the tabs and the LD Blocks & Haplotypes tab.

    # The genotype table comes only through the session's table object (the GWAS page's
    # gwas.guitable.SessionTable): its identity, the table in memory or the view of the
    # table on disk, and its SNP and sample labels.
    required_keys = ["geno_table"]

    missing = [k for k in required_keys if st.session_state.get(k) is None]
    if missing:
        st.error(
            "Missing required genotype objects in session state:\n\n"
            + ", ".join(missing)
            + "\n\nRun the GWAS page first."
        )
        st.stop()
    geno_table = st.session_state["geno_table"]
    # The GWAS results are those of the last Run GWAS, whose table's identity the GWAS page
    # records with them. It rebuilds the session's table on every rerun, so a table built
    # since with other settings has another identity, and the results are not its own.
    if _gp.run_table_changed(st.session_state, geno_table):
        st.warning(_gp.RUN_CHANGED_MESSAGE)
        st.stop()

    # ---- Resolve phenotype (STRICT: use GWAS-aligned phenotype if available) ----
    if "pheno_used_for_hap" in st.session_state:
        pheno_df = st.session_state["pheno_used_for_hap"]
    elif "pheno_aligned" in st.session_state:
        pheno_df = st.session_state["pheno_aligned"]
    elif "pheno" in st.session_state:
        pheno_df = st.session_state["pheno"]
    elif "pheno_raw" in st.session_state:
        pheno_df = st.session_state["pheno_raw"]
    else:
        st.error(
            "No phenotype dataframe found.\n"
            "Run GWAS first so LD uses the correct phenotype."
        )
        st.stop()


    # --------------------------------------------------------
    # 2. Resolve GWAS results (ROBUST)
    # --------------------------------------------------------
    gwas_df = None
    trait_col = None

    # Always trust active_trait if present
    trait_col = (
            st.session_state.get("trait_col_locked")
            or st.session_state.get("active_trait")
            or st.session_state.get("trait_col")
    )

    # CRITICAL: must have an active trait before indexing trait-specific results
    if trait_col is None:
        st.error("No active trait found. Please run GWAS first: click Run GWAS on the "
                 "GWAS page. One-Click Full Analysis writes its results to its ZIP only; "
                 "this page works on the GWAS page's Run GWAS.")
        st.stop()

    # 1) Try trait-specific gwas_results
    gwas_results = st.session_state.get("gwas_results", {})
    if trait_col in gwas_results:
        gwas_df = gwas_results[trait_col]

    # 2) Fallback: global gwas_df
    if gwas_df is None and "gwas_df" in st.session_state:
        gwas_df = st.session_state["gwas_df"]

    # 3) Final sanity check
    if gwas_df is None or not isinstance(gwas_df, pd.DataFrame) or gwas_df.empty:
        st.error(
            "GWAS results not found for LD analysis.\n\n"
            "Run a single-trait GWAS first (Run GWAS on the GWAS page) and ensure results "
            "are saved. One-Click Full Analysis writes its results to its ZIP only."
        )
        st.stop()

    gwas_df = gwas_df.copy()

    if "Chr" in gwas_df.columns:
        gwas_df["Chr"] = canonicalize_chr(gwas_df["Chr"])

    try:
        trait_col_resolved = resolve_trait_column(trait_col, pheno_df)
    except KeyError as e:
        st.error(str(e))
        st.stop()

    ld_trait = trait_col_resolved

    # --------------------------------------------------------
    # 3. Pull shared objects
    # --------------------------------------------------------
    # a table on disk can be closed while the session still holds it (Clear session data in
    # any session, or the server's registry making room): stop before any read
    if _gp.table_released(geno_table):
        _released()
    positions = geno_table.positions
    sid = geno_table.sid

    chroms = canonicalize_chr(geno_table.chroms)

    # The genotype frame's index is all this page reads of it (the sample IDs); the shared
    # functions it is passed to read nothing else. On disk the GWAS page keeps the store
    # frame (the index and a row column) or None, so the IDs then come from GWAS_CTX or
    # the table.
    try:
        _samples = _gp.sample_index(
            st.session_state.get("geno_df"),
            (st.session_state.get("GWAS_CTX") or {}).get("geno_index"),
            geno_table.iid)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    geno_df = pd.DataFrame(index=_samples)
    pcs = st.session_state.get("pcs", None)
    geno_encoding = st.session_state.get("geno_encoding", None)

    # ============================================================
    # SINGLE phenotype alignment for the entire LD page
    # ============================================================
    _ph_aligned_global, _y_aligned_global, _keep_mask_global = align_pheno_to_geno(
        pheno_df,
        geno_df,
        ld_trait,
    )
    # -------------------------------------------------
    # The genotype table, read in windows and blocks (gwas.guipost.PostTable)
    # -------------------------------------------------
    # LD uses imputed dosages directly (stable LD estimation): post.ld reads each window
    # as the float64 copy np.asarray(geno_imputed, float) gave, post.ld_aligned its
    # phenotype-aligned rows, and post.hard the hard calls (np.rint) when the encoding
    # supports them, else the dosages. No whole copy of the table is made or kept.
    geno_encoding = st.session_state.get("geno_encoding", "dosage012")
    post = _gp.post_table(geno_table, _keep_mask_global, geno_encoding=geno_encoding,
                          samples=geno_df.index)

    chroms_now, pos_now, sid_now = _sync_variant_axes(
        post.shape[1],
        chroms,
        positions,
        sid,
    )

    # Store the LD page's SNP labels (both LD-specific and legacy keys)
    st.session_state["ld_chroms"] = chroms_now
    st.session_state["ld_positions"] = pos_now
    st.session_state["ld_sid"] = sid_now

    st.session_state["chroms"] = chroms_now
    st.session_state["positions"] = pos_now
    st.session_state["sid"] = sid_now

    # and use these locals from here on
    chroms = chroms_now
    positions = pos_now
    sid = sid_now

    # -------------------------------------------------
    # FIX: keep SNP metadata ALWAYS aligned with genotype columns
    # -------------------------------------------------
    m_now = int(post.shape[1])

    chroms_now = np.asarray(chroms, dtype=object)
    pos_now = np.asarray(positions, dtype=int)
    sid_now = np.asarray(sid, dtype=object)

    # If any of these lengths don't match genotype columns,
    # DO NOT keep stale session_state; hard fail with a clear message.
    if not (len(chroms_now) == len(pos_now) == len(sid_now) == m_now):
        st.error(
            "SNP axis mismatch detected:\n"
            f"genotype columns  = {m_now}\n"
            f"chroms            = {len(chroms_now)}\n"
            f"positions         = {len(pos_now)}\n"
            f"sid               = {len(sid_now)}\n\n"
            "This usually means stale session_state from a previous run "
            "or that GWAS/LD are using different SNP sets.\n"
            "Fix: rerun GWAS and ensure chroms/positions/sid are created from the same SNP set as geno_imputed."
        )
        st.stop()

    # --------------------------------------------------------
    # 4. LD decay hints from GWAS page (optional)
    # --------------------------------------------------------
    # ---- Auto-reset LD caches when trait changes ----
    prev_trait = st.session_state.get("_ld_prev_trait", None)
    if prev_trait != trait_col:
        # drop only LD-page caches (don’t nuke everything)
        for k in ["r2_cache"]:
            st.session_state.pop(k, None)

        # optional: if you want auto LD blocks to rebuild cleanly
        # keep dicts but remove current trait entry to avoid mismatch
        if isinstance(st.session_state.get("haplo_df_auto"), dict):
            st.session_state["haplo_df_auto"].pop(trait_col, None)
        if isinstance(st.session_state.get("haplo_df_auto_hash"), dict):
            st.session_state["haplo_df_auto_hash"].pop(trait_col, None)

        st.session_state["_ld_prev_trait"] = trait_col
    # the LD decay the GWAS page's Run GWAS estimated (the run whose results this page
    # shows), or None without an estimate: the window is then 300 kb, as in the CLI
    ld_decay_kb = (st.session_state.get("ld_decay_kb")
                   if st.session_state.get("ld_decay_computed", False) else None)
    ld_decay_kb = None if ld_decay_kb is None else float(ld_decay_kb)
    _decay_run = decay_run_label(st.session_state.get("gwas_run_summary"))
    # Keep a single, consistent adjacent-LD split threshold across the page
    adj_r2_min_global = float(st.session_state.get("adj_r2_min", 0.2))
    adj_r2_min_global = float(np.clip(adj_r2_min_global, 0.0, 0.99))

    if ld_decay_kb is not None:
        _cn = ""
        _sm = st.session_state.get("ld_decay_summary")
        if _sm is not None and "censored_r2_0.2" in getattr(_sm, "columns", []):
            _ncc = int(_sm["censored_r2_0.2"].sum())
            if _ncc:
                _cn = f" ({_ncc} chr grid-limited, upper bound)"
        st.markdown(
            f"**Trait:** `{trait_col}`  \n"
            f"LD decay (r² ≤ 0.2): ~{ld_decay_kb:.1f} kb (from {_decay_run}){_cn}"
        )
    else:
        st.markdown(
            f"**Trait:** `{trait_col}`  \n"
            f"LD decay: no estimate in {_decay_run}, so the LD window is "
            f"{ld_window_kb(None)} kb, as in the CLI"
        )
    # --------------------------------------------------------
    # AUTO-BUILD peak-centric LD blocks once per trait
    # --------------------------------------------------------
    if "haplo_df_auto" not in st.session_state or not isinstance(st.session_state["haplo_df_auto"], dict):
        st.session_state["haplo_df_auto"] = {}
    # Robust hash: sort only by columns that exist
    sort_cols = [c for c in ["Chr", "Pos", "SNP", "PValue"] if c in gwas_df.columns]
    gwas_hash = _hash_df(
        gwas_df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else gwas_df.reset_index(drop=True))
    # the blocks depend on the GWAS results and on the table (its identity and the rows the
    # detection reads): rebuilt when any of them changes
    auto_sig = f"{gwas_hash}::T{post.identity}::R{post.keep_key}"

    existing = st.session_state["haplo_df_auto"].get(trait_col, None)
    existing_empty = (existing is None) or (isinstance(existing, pd.DataFrame) and existing.empty)

    st.session_state.setdefault("haplo_df_auto_hash", {})
    old_hash = st.session_state["haplo_df_auto_hash"].get(trait_col, None)

    needs_rebuild = (
            (trait_col not in st.session_state["haplo_df_auto"])
            or existing_empty
            or (old_hash != auto_sig)
    )

    if needs_rebuild:
        with st.spinner("Auto-detecting LD blocks (first visit or GWAS changed)…"):
            # the phenotype-aligned rows, read window by window (the detector's bounded r2
            # cache keeps the last windows). When every sample is aligned and the LD Blocks
            # tab is at its defaults, the tab's detection is this one: SharedDetection keeps
            # the result for it, so a first visit detects once.
            geno_ld_masked = post.ld_aligned
            haplo_tmp = _gp.SharedDetection(
                st.session_state, find_ld_clusters_genomewide, post.identity,
            ).find_ld_clusters_genomewide(
                gwas_df=gwas_df,
                chroms=chroms,
                positions=positions,
                geno_imputed=geno_ld_masked,
                sid=sid,
                ld_threshold=0.6,
                flank_kb=ld_window_kb(ld_decay_kb),   # the CLI's rule: int(2 x decay)
                min_snps=2,
                top_n=10,
                sig_thresh=1e-5,
                max_dist_bp=None,
                ld_decay_kb=ld_decay_kb,
                adj_r2_min=adj_r2_min_global,
            )

            # kept for the most recent traits only (a fixed number)
            _gp.keep_recent(st.session_state["haplo_df_auto"], trait_col, haplo_tmp,
                            _gp.TRAIT_CACHE_MAX)
            _gp.keep_recent(st.session_state["haplo_df_auto_hash"], trait_col, auto_sig,
                            _gp.TRAIT_CACHE_MAX)

            if isinstance(haplo_tmp, pd.DataFrame) and not haplo_tmp.empty:
                st.session_state["ld_blocks_final"] = haplo_tmp.copy()
                # which trait and table they are of (the GWAS page's block stability checks)
                st.session_state["ld_blocks_final_key"] = (trait_col, geno_table.identity)
                st.session_state["ld_blocks"] = haplo_tmp.copy()  # shared with GWAS enrichment

    # --------------------------------------------------------
    # 5. LD block tables (trait-aware)
    # --------------------------------------------------------
    auto_store = st.session_state.get("haplo_df_auto", pd.DataFrame())
    if isinstance(auto_store, dict):
        haplo_df_auto = auto_store.get(trait_col, pd.DataFrame())
    else:
        haplo_df_auto = auto_store

    # --------------------------------------------------------
    # Mega-block containment is structurally impossible after the disjoint-block
    # redesign (WO4): _occupancy_select claims each accepted span and rejects any
    # candidate overlapping a claimed marker, so block detection is disjoint by
    # construction. The former page-scoped mega-block filter control (mode /
    # min_contained / size_ratio) is RETIRED -- it configured a case that cannot
    # occur. This is now an invariant CHECK only: if nesting is EVER found it is
    # disclosed + LOGGED with the block coordinates, never silently removed, and the
    # block set passes through unchanged.
    # --------------------------------------------------------
    if isinstance(haplo_df_auto, pd.DataFrame) and not haplo_df_auto.empty:
        _cf, _ = filter_contained_blocks(
            haplo_df_auto.copy(), min_contained=2, size_ratio_threshold=3.0, mode="flag")
        if "is_mega_block" in _cf.columns and bool(_cf["is_mega_block"].any()):
            _mb = _cf[_cf["is_mega_block"]]
            logging.getLogger(__name__).warning(
                "LD containment detected (should be impossible post-occupancy): %s",
                [f"{r['Chr']}:{int(r['Start (bp)'])}-{int(r['End (bp)'])}"
                 for _, r in _mb.iterrows()])
            st.caption(
                "⚠ Nested LD blocks detected. This should not occur after the "
                "disjoint-block redesign; the block set is left unchanged. Please report "
                "(see _occupancy_select's span_idx.size==0 short-circuit).")

    if not isinstance(st.session_state.get("haplo_df_auto"), dict):
        st.session_state["haplo_df_auto"] = {}
    _gp.keep_recent(st.session_state["haplo_df_auto"], trait_col, haplo_df_auto,
                    _gp.TRAIT_CACHE_MAX)

    # ============================================================
    # Build shared context for all tabs
    # ============================================================
    # The tabs get the table's readers (windows and blocks, never a whole table), its
    # identity for their cache keys, and the GWAS page's table as it is (the haplotype PCA
    # panel's fallback).
    _ld_ctx = LDContext(
        geno_ld=post.ld,
        geno_hard=post.hard,
        chroms=chroms,
        positions=positions,
        sid=sid,
        geno_df=geno_df,
        pheno_df=pheno_df,
        trait_col=trait_col,
        ph_aligned=_ph_aligned_global,
        y_aligned=_y_aligned_global,
        keep_mask=_keep_mask_global,
        geno_ld_aligned=post.ld_aligned,
        gwas_df=gwas_df,
        haplo_df_auto=haplo_df_auto,
        ld_decay_kb=ld_decay_kb,
        adj_r2_min_global=adj_r2_min_global,
        ld_trait=ld_trait,
        pcs=pcs,
        geno_encoding=geno_encoding,
        has_annotation=_HAS_ANNOTATION,
        geno_dosage_raw=post.raw,
        # the GWAS page's M_eff when it was computed on this table, else none (the rule's
        # path without M_eff)
        meff_val=_gp.meff_for_table(st.session_state, geno_table),
        sig_rule_label=st.session_state.get("sig_rule", ""),
        custom_thresh=st.session_state.get("custom_sig_thresh"),
        table_identity=post.identity,
        geno_view=post.view,
    )

    # ============================================================
    # Tabs
    # ============================================================
    # Shared window selector, rendered ONCE above the tab bar so the Regional
    # Plot and Local LD tabs consume ONE selection. Streamlit forbids the same
    # keyed widget in two st.tabs bodies (DuplicateWidgetID), and every tab body
    # executes on each rerun, so the selector cannot live inside either tab.
    with st.expander("Region / lead-SNP selection (Regional Plot + Local LD)", expanded=True):
        _ld_window = select_window(_ld_ctx)

    # five tabs: the LD-decay estimate is the GWAS page's, shown in the LD Blocks &
    # Haplotypes tab's Settings next to the window it sets
    tab_sig, tab_reg, tab1, tab_genes, tab2 = st.tabs([
        "Significant SNPs",
        "Regional Plot",
        "Local LD",
        "Gene Annotation",
        "LD Blocks & Haplotypes",
    ])
    with tab_sig:
        tab_significant_snps.render(_ld_ctx)

    with tab_reg:
        tab_regional.render(_ld_ctx, _ld_window)

    with tab1:
        tab_local_ld.render(_ld_ctx, get_r2_cached, _ld_window)

    with tab_genes:
        tab_gene_annotation.render(_ld_ctx)

    with tab2:
        # the haplotype test is kept in the session under its key (which names the table's
        # identity); it is not wrapped in st.cache_data, which would hash the whole table
        tab_genome_wide.render(
            _ld_ctx,
            get_r2_cached=get_r2_cached,
            cached_extract_block_geno=None,
            compute_block_qc_effects=compute_block_qc_effects,
            run_haplotype_block_gwas_cached_fn=_gp.run_haplotype_tests,
        )




# Always run the page when Streamlit imports the file. A table file closed while this run
# reads it ends the run with the same plain message as the check at the top.
with _gp.released_guard(lambda: st.session_state.get("geno_table"), _released):
    ld_analysis_page()
