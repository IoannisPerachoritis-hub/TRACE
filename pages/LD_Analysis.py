# page_ld_analysis.py
import numpy as np
import pandas as pd
import streamlit as st
from utils.pub_theme import apply_matplotlib_theme, build_plotly_template
apply_matplotlib_theme()
build_plotly_template()
import hashlib

from gwas.ld import (
    filter_contained_blocks, extract_block_geno_for_paper,
    maf_from_matrix, anova_eta_sq_from_labels,
)
from gwas.utils import canonicalize_chr, align_pheno_to_geno, resolve_trait_column, check_data_version
from gwas.haplotype import mlg_labels_for_block
from pages._ld_tabs import LDContext
from pages._ld_tabs import tab_block_heatmaps
from pages._ld_tabs import tab_gene_annotation
from pages._ld_tabs import tab_decay
from pages._ld_tabs import tab_genome_wide





def _hash_df(df: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.blake2b(h.tobytes(), digest_size=16).hexdigest()
def _sync_variant_axes(G, chroms, positions, sid):
    """
    Ensure genotype matrix columns match chroms/positions/sid lengths.
    Fails hard on mismatch to prevent silent data corruption.
    """
    G = np.asarray(G)
    chroms = np.asarray(chroms, dtype=object)
    positions = np.asarray(positions)
    sid = np.asarray(sid, dtype=object)

    nG = G.shape[1]
    nC = len(chroms)
    nP = len(positions)
    nS = len(sid)

    if not (nG == nC == nP == nS):
        st.error(
            f"CRITICAL: Variant axis mismatch — "
            f"geno columns={nG}, chroms={nC}, positions={nP}, sid={nS}.\n\n"
            f"This indicates stale session state from a previous run. "
            f"Please rerun GWAS to rebuild all arrays from the same SNP set."
        )
        st.stop()

    return G, chroms, positions, sid

# -----------------------------------------------------------
# Cached wrapper for block genotype extraction
# Prevents recomputation when only UI changes (e.g. boxplots)
# -----------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=50)
def cached_extract_block_geno(
    geno_hard, chroms, positions, sid,
    block_chr, block_start, block_end,
    sample_keep_mask=None, maf_threshold=0.01,
    cache_key: str = "",
    snp_ids=None,
):
    return extract_block_geno_for_paper(
        geno_hard, chroms, positions, sid,
        block_chr, block_start, block_end,
        sample_keep_mask=sample_keep_mask,
        maf_threshold=maf_threshold,
        snp_ids=snp_ids,
    )
# --------------------------------------------------
# LD matrix cache (robust, avoids re-hashing big arrays)
# --------------------------------------------------
from collections import OrderedDict

def get_r2_cached(
    G: np.ndarray,
    pos: np.ndarray | None = None,
    min_pair_n: int = 20,
    cache_key: str | None = None,
):
    """
    Compute LD r² with LRU caching.

    Prefer passing an explicit cache_key (e.g. "r2::chr3:1000-5000")
    for reliable, collision-free caching.
    """
    from gwas.ld import pairwise_r2

    if cache_key is None:
        # Fallback: shape + boundary positions (fast, mostly safe)
        pos_sig = ""
        if pos is not None:
            p = np.asarray(pos, int)
            pos_sig = f"::{p[0]}:{p[-1]}:{len(p)}"
        cache_key = f"r2::{G.shape}{pos_sig}::minpair{int(min_pair_n)}"

    if "r2_cache" not in st.session_state:
        st.session_state["r2_cache"] = OrderedDict()

    cache = st.session_state["r2_cache"]
    MAX_R2_CACHE = 10

    if cache_key in cache:
        cache.move_to_end(cache_key)
        return cache[cache_key]

    r2 = pairwise_r2(np.asarray(G, float), min_pair_n=int(min_pair_n))
    r2 = np.clip(r2, 0.0, 1.0)
    cache[cache_key] = r2
    cache.move_to_end(cache_key)

    while len(cache) > MAX_R2_CACHE:
        cache.popitem(last=False)

    return r2

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
        format_annotation_summary,
        compute_ld_decay_by_chromosome,
        plot_ld_decay_matplotlib,
    )
    _HAS_ANNOTATION = True
except ImportError:
    try:
        from annotation import (
            load_gene_annotation,
            annotate_ld_blocks,
            format_annotation_summary,
            compute_ld_decay_by_chromosome,
            plot_ld_decay_matplotlib,
        )
        _HAS_ANNOTATION = True
    except ImportError:
        _HAS_ANNOTATION = False

# -------------------------------------------
# Haplotype GWAS (import directly from gwas.haplotype
# to avoid triggering GWAS_analysis.py top-level UI)
# -------------------------------------------
from gwas.haplotype import run_haplotype_block_gwas, run_haplotype_block_gwas_cached


# ============================================================
# Helper: cumulative genomic positions for Manhattan-style plots
# ============================================================
from gwas.plotting import compute_cumulative_positions as _prepare_cumpos



# Backward-compatible aliases for renamed functions
_mlg_labels_for_block = mlg_labels_for_block
_maf_from_matrix = maf_from_matrix
_anova_eta_sq_from_labels = anova_eta_sq_from_labels


def compute_block_qc_effects(
    hap_gwas_df: pd.DataFrame,
    chroms,
    positions,
    geno_imputed: np.ndarray,
    geno_df: pd.DataFrame,
    pheno_for_hap: pd.DataFrame,
    trait_col: str,
    min_hap_count: int,
    min_group_size: int,
    compute_maf: bool = True,
    geno_encoding: str | None = None,
) -> pd.DataFrame:
    """
    Add publication-oriented QC/effect metrics to hap_gwas_df:
      - Block_length_kb
      - n_snps_total (in physical window)
      - n_snps_poly
      - maf_median, maf_mean (optional)
      - EtaSq (η²) for MLG ANOVA groups actually tested
      - n_groups_tested, n_samples_used
      - Frac_other (based on tested vs total haplotypes, if available)
    """
    if hap_gwas_df is None or hap_gwas_df.empty:
        return hap_gwas_df

    out = hap_gwas_df.copy()
    out["Block_length_kb"] = (out["End"].astype(int) - out["Start"].astype(int)) / 1000.0

    # “Other” fraction (only if your run_haplotype_block_gwas provides these columns)
    if ("n_haplotypes" in out.columns) and ("n_tested_haplotypes" in out.columns):
        denom = out["n_haplotypes"].replace(0, np.nan).astype(float)
        out["Frac_other"] = 1.0 - (out["n_tested_haplotypes"].astype(float) / denom)
    else:
        out["Frac_other"] = np.nan

    chroms_np = np.asarray(chroms).astype(str)
    pos_np = np.asarray(positions).astype(int)

    trait_col_qc = resolve_trait_column(trait_col, pheno_for_hap)

    # -------------------------------------------------
    # FIX — enforce genotype sample order alignment
    # prevents genotype–phenotype mismatches
    # -------------------------------------------------
    ph_aligned, y_vec, keep_mask = align_pheno_to_geno(
        pheno_for_hap,
        geno_df,
        trait_col_qc,
    )

    n_snps_total = []
    n_snps_poly = []
    maf_median = []
    maf_mean = []
    eta_sq_list = []
    n_groups_list = []
    n_used_list = []

    for _, row in out.iterrows():
        ch = str(row["Chr"])
        s = int(row["Start"])
        e = int(row["End"])

        mask = (chroms_np == ch) & (pos_np >= s) & (pos_np <= e)
        G = np.asarray(geno_imputed[keep_mask, :][:, mask], float)

        n_snps_total.append(int(G.shape[1]))

        if G.shape[1] < 1:
            n_snps_poly.append(0)
            maf_median.append(np.nan)
            maf_mean.append(np.nan)
            eta_sq_list.append(np.nan)
            n_groups_list.append(0)
            n_used_list.append(int(np.isfinite(y_vec).sum()))
            continue

        var = np.nanvar(G, axis=0)
        poly = var > 0
        n_snps_poly.append(int(np.sum(poly)))

        # MAF summaries
        if compute_maf and np.any(poly):
            enc = geno_encoding
            maf = _maf_from_matrix(G[:, poly], enc)
            maf_median.append(float(np.nanmedian(maf)))
            maf_mean.append(float(np.nanmean(maf)))
        else:
            maf_median.append(np.nan)
            maf_mean.append(np.nan)

        # Effect size η² using the same MLG grouping logic as permutation test
        # (collapse rare haplotypes and drop too-small groups + "Other")
        keep_y = np.isfinite(y_vec)
        if keep_y.sum() < 5:
            eta_sq_list.append(np.nan)
            n_groups_list.append(0)
            n_used_list.append(int(keep_y.sum()))
            continue

        G2 = G[keep_y, :]
        y2 = y_vec[keep_y]

        # ---- PC-residualize phenotype for η² (match haplotype analysis test) ----
        pcs_for_eta = st.session_state.get("pcs", None)
        if pcs_for_eta is not None:
            try:
                pcs_arr = np.asarray(pcs_for_eta, float)
                # align to same keep_mask used above
                pcs_sub = pcs_arr[keep_mask, :][keep_y, :]
                X = np.column_stack([np.ones(pcs_sub.shape[0]), pcs_sub])
                beta, *_ = np.linalg.lstsq(X, y2, rcond=None)
                y2 = y2 - X @ beta
            except Exception:
                pass  # fall back to unadjusted η² if alignment fails

        if G2.shape[1] < 2:
            eta_sq_list.append(np.nan)
            n_groups_list.append(0)
            n_used_list.append(int(len(y2)))
            continue

        mlg, keep_mlg = _mlg_labels_for_block(
            G2,
            min_hap_count=min_hap_count,
            return_mask=True,
            strict_selfing=st.session_state.get("selfing_mode", False),
            geno_encoding=geno_encoding,
        )

        y2 = y2[keep_mlg]

        df = pd.DataFrame({
            "MLG": mlg,
            "y": y2
        })

        counts = df["MLG"].value_counts()
        valid = [g for g in counts.index if (counts[g] >= int(min_group_size) and g != "Other")]

        if len(valid) < 2:
            eta_sq_list.append(np.nan)
            n_groups_list.append(int(len(valid)))
            n_used_list.append(int(len(y2)))
            continue

        df = df[df["MLG"].isin(valid)].copy()
        eta_sq, n_groups, n_used = _anova_eta_sq_from_labels(df["y"].values, df["MLG"].values)

        eta_sq_list.append(eta_sq)
        n_groups_list.append(n_groups)
        n_used_list.append(n_used)

    out["n_snps_total"] = n_snps_total
    out["n_snps_poly"] = n_snps_poly
    out["maf_median"] = maf_median
    out["maf_mean"] = maf_mean
    out["EtaSq"] = eta_sq_list
    out["n_groups_tested_effect"] = n_groups_list
    out["n_samples_used_effect"] = n_used_list

    return out

# --------------------------------------------------------
# 1. Sanity: require genotype / phenotype (GWAS resolved below)
# --------------------------------------------------------

def ld_analysis_page():
    st.title("LD & Haplotype Analysis")
    check_data_version("ld_analysis")

    # ---- Sidebar: mega-block filter ----
    st.sidebar.subheader("Mega-block filter")
    mega_mode = st.sidebar.radio(
        "Mega-block handling",
        ["Remove", "Flag only"],
        index=0,
        key="mega_block_mode",
    )
    mega_min = st.sidebar.number_input(
        "Min contained blocks",
        min_value=1, max_value=10, value=2,
        key="mega_min_contained",
    )
    mega_ratio = st.sidebar.number_input(
        "Size ratio threshold",
        min_value=1.5, max_value=20.0, value=3.0, step=0.5,
        key="mega_size_ratio",
    )

    # ---- Sidebar: haplotype / MLG analysis settings ----
    st.sidebar.subheader("Haplotype / MLG Analysis")
    st.sidebar.number_input(
        "Haplotype permutations",
        min_value=100, max_value=10000, value=1000, step=100,
        key="n_perm_hap",
        help="Number of Freedman-Lane permutations for haplotype block analysis. "
             "Higher = finer p-values but slower. Max -log10(p) ≈ log10(n_perm).",
    )

    # ---- Sidebar: display options ----
    st.sidebar.subheader("Display")
    st.sidebar.checkbox(
        "Show numeric LD values in heatmaps",
        value=False,
        key="show_ld_labels",
    )

    required_keys = ["geno_dosage_raw", "geno_imputed", "chroms", "positions", "sid", "geno_df"]

    missing = [k for k in required_keys if k not in st.session_state]
    if missing:
        st.error(
            "Missing required genotype objects in session state:\n\n"
            + ", ".join(missing)
            + "\n\nRun the GWAS page first."
        )
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
        st.error("No active trait found. Please run GWAS first.")
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
            "Run a single-trait GWAS first and ensure results are saved."
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
    positions = st.session_state["positions"]
    sid = st.session_state["sid"]

    chroms = canonicalize_chr(st.session_state["chroms"])

    geno_df = st.session_state["geno_df"]
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
    _geno_ld_aligned = None  # lazy — built after geno_ld is ready
    # -------------------------------------------------
    # Canonical genotype matrices (SIMPLIFIED)
    # -------------------------------------------------

    # LD uses imputed dosages directly (stable LD estimation)
    geno_ld = np.asarray(st.session_state["geno_imputed"], float)

    # Pre-masked genotype matrix (phenotype-aligned)
    _geno_ld_aligned = geno_ld[_keep_mask_global, :]

    # -------------------------------------------------
    # Hard-call conversion ONLY when encoding supports it
    # -------------------------------------------------
    geno_encoding = st.session_state.get("geno_encoding", "dosage012")

    if geno_encoding == "dosage012":
        geno_hard = np.rint(geno_ld)
    else:
        geno_hard = geno_ld.copy()

    geno_hard, chroms_now, pos_now, sid_now = _sync_variant_axes(
        geno_hard,
        chroms,
        positions,
        sid,
    )

    # Store LD-local arrays (both LD-specific and legacy keys)
    st.session_state["ld_geno_hard"] = geno_hard
    st.session_state["ld_chroms"] = chroms_now
    st.session_state["ld_positions"] = pos_now
    st.session_state["ld_sid"] = sid_now

    st.session_state["geno_hard"] = geno_hard
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
    G_now = np.asarray(geno_hard)
    m_now = int(G_now.shape[1])

    chroms_now = np.asarray(chroms, dtype=object)
    pos_now = np.asarray(positions, dtype=int)
    sid_now = np.asarray(sid, dtype=object)

    # If any of these lengths don't match genotype columns,
    # DO NOT keep stale session_state; hard fail with a clear message.
    if not (len(chroms_now) == len(pos_now) == len(sid_now) == m_now):
        st.error(
            "SNP axis mismatch detected:\n"
            f"geno_hard columns = {m_now}\n"
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
    ld_decay_kb = float(st.session_state.get("ld_decay_kb", 200.0))
    ld_decay_kb = max(ld_decay_kb, 10.0)
    # Keep a single, consistent adjacent-LD split threshold across the page
    adj_r2_min_global = float(st.session_state.get("adj_r2_min", 0.2))
    adj_r2_min_global = float(np.clip(adj_r2_min_global, 0.0, 0.99))

    if st.session_state.get("ld_decay_computed", False):
        st.markdown(
            f"**Trait:** `{trait_col}`  \n"
            f"LD decay (r² ≤ 0.2): ~{ld_decay_kb:.1f} kb (from Decay tab)"
        )
    else:
        st.markdown(
            f"**Trait:** `{trait_col}`  \n"
            f"LD decay estimate (r² ≤ 0.2): ~{ld_decay_kb:.1f} kb "
            f"*(quick estimate — use Decay tab for full computation)*"
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

    existing = st.session_state["haplo_df_auto"].get(trait_col, None)
    existing_empty = (existing is None) or (isinstance(existing, pd.DataFrame) and existing.empty)

    st.session_state.setdefault("haplo_df_auto_hash", {})
    old_hash = st.session_state["haplo_df_auto_hash"].get(trait_col, None)

    needs_rebuild = (
            (trait_col not in st.session_state["haplo_df_auto"])
            or existing_empty
            or (old_hash != gwas_hash)
    )

    if needs_rebuild:
        with st.spinner("Auto-detecting LD blocks (first visit or GWAS changed)…"):
            geno_ld_masked = _geno_ld_aligned
            haplo_tmp = find_ld_clusters_genomewide(
                gwas_df=gwas_df,
                chroms=chroms,
                positions=positions,
                geno_imputed=geno_ld_masked,
                sid=sid,
                ld_threshold=0.6,
                flank_kb=float(ld_decay_kb * 2),
                min_snps=3,
                top_n=10,
                sig_thresh=1e-5,
                max_dist_bp=None,
                ld_decay_kb=ld_decay_kb,
                adj_r2_min=adj_r2_min_global,
            )

            st.session_state["haplo_df_auto"][trait_col] = haplo_tmp
            st.session_state["haplo_df_auto_hash"][trait_col] = gwas_hash

            if isinstance(haplo_tmp, pd.DataFrame) and not haplo_tmp.empty:
                st.session_state["ld_blocks_final"] = haplo_tmp.copy()
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
    # Apply mega-block filter (sidebar controls)
    # --------------------------------------------------------
    if isinstance(haplo_df_auto, pd.DataFrame) and not haplo_df_auto.empty:
        haplo_df_auto, n_removed = filter_contained_blocks(
            haplo_df_auto,
            min_contained=int(mega_min),
            size_ratio_threshold=float(mega_ratio),
            mode="remove" if mega_mode == "Remove" else "flag",
        )

        if mega_mode == "Remove" and n_removed > 0:
            st.toast(f"Removed {n_removed} mega-block(s).", icon="⚠️")

    if not isinstance(st.session_state.get("haplo_df_auto"), dict):
        st.session_state["haplo_df_auto"] = {}
    st.session_state["haplo_df_auto"][trait_col] = haplo_df_auto

    # ============================================================
    # Build shared context for all tabs
    # ============================================================
    _ld_ctx = LDContext(
        geno_ld=geno_ld,
        geno_hard=geno_hard,
        chroms=chroms,
        positions=positions,
        sid=sid,
        geno_df=geno_df,
        pheno_df=pheno_df,
        trait_col=trait_col,
        ph_aligned=_ph_aligned_global,
        y_aligned=_y_aligned_global,
        keep_mask=_keep_mask_global,
        geno_ld_aligned=_geno_ld_aligned,
        gwas_df=gwas_df,
        haplo_df_auto=haplo_df_auto,
        ld_decay_kb=ld_decay_kb,
        adj_r2_min_global=adj_r2_min_global,
        ld_trait=ld_trait,
        pcs=pcs,
        geno_encoding=geno_encoding,
        show_ld_labels=st.session_state.get("show_ld_labels", False),
        has_annotation=_HAS_ANNOTATION,
    )

    # ============================================================
    # Tabs
    # ============================================================
    tab_blocks, tab_genes, tab_decay_t, tab2 = st.tabs([
        "Block Heatmaps",
        "Gene Annotation",
        "LD Decay",
        "LD Blocks & Haplotypes",
    ])
    with tab_blocks:
        tab_block_heatmaps.render(_ld_ctx, get_r2_cached)

    with tab_genes:
        tab_gene_annotation.render(_ld_ctx)

    with tab_decay_t:
        tab_decay.render(_ld_ctx)

    with tab2:
        tab_genome_wide.render(
            _ld_ctx,
            get_r2_cached=get_r2_cached,
            cached_extract_block_geno=cached_extract_block_geno,
            compute_block_qc_effects=compute_block_qc_effects,
            run_haplotype_block_gwas_cached_fn=run_haplotype_block_gwas_cached,
        )




# Always run the page when Streamlit imports the file
ld_analysis_page()
