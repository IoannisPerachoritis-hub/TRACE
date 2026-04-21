import numpy as np
import pandas as pd
import matplotlib.patches as patches
from scipy.stats import linregress
import re
import hashlib
from annotation import canon_chr
def _split_leads(x: str) -> list[str]:
    toks = re.split(r"[;,\s|]+", str(x))
    return [t for t in toks if t and t.lower() != "nan"]

def _merge_leads(a, b) -> str:
    toks = set(_split_leads(a)) | set(_split_leads(b))
    return ";".join(sorted(toks))
def _interval_iou_bp(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """
    IoU for 1D genomic intervals in bp, treated as continuous physical spans.
    Coordinates are half-open [start, end) in spirit but computed as
    continuous distances (no +1). Not SNP counts.
    Compatible with bp-based block boundaries from LD detection.
    Not directly compatible with BED-style or 1-based inclusive coordinates
    without conversion.
    """
    a_start, a_end = int(a_start), int(a_end)
    b_start, b_end = int(b_start), int(b_end)

    # ensure proper ordering
    if a_end < a_start:
        a_start, a_end = a_end, a_start
    if b_end < b_start:
        b_start, b_end = b_end, b_start

    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    len_a = max(0, a_end - a_start)
    len_b = max(0, b_end - b_start)
    union = len_a + len_b - inter

    return 0.0 if union <= 0 else float(inter / union)
def _adaptive_adj_threshold(r2_sub: np.ndarray, base: float = 0.3, frac: float = 0.5):
    """
    Adaptive adjacent LD threshold.

    Uses median adjacent r² inside a block:
        threshold = max(base, frac * median_adjacent_r2)

    This stabilizes LD block detection across:
        - tomato (often higher LD)
        - pepper (often lower LD)
        - uneven marker density.
    """
    if r2_sub.shape[0] < 2:
        return base

    adj = np.diag(r2_sub, k=1)
    med = float(np.nanmedian(adj)) if np.any(np.isfinite(adj)) else np.nan

    if not np.isfinite(med):
        return base

    return float(min(0.5, max(base, frac * med)))
# -------------------------------------------------
# 1) Pairwise LD (r²) — EXACT (pairwise-complete)
# -------------------------------------------------
def pairwise_r2(geno_matrix, min_pair_n: int = 20):
    """
    EXACT, mask-aware Pearson r² using pairwise-complete observations.

    - Uses only samples with non-missing genotypes for each SNP pair.
    - Enforces minimum pairwise overlap (min_pair_n).
    - Returns full m×m matrix aligned to original SNP order.
    - Monomorphic (or near-monomorphic) pairs -> NaN off-diagonal, diag=1.
    """

    G = np.asarray(geno_matrix, dtype=float)
    if G.ndim != 2:
        raise ValueError("geno_matrix must be a 2D array (n_samples x n_snps).")

    n, m = G.shape

    if m > 4000:
        raise RuntimeError(
            f"pairwise_r2 called with {m} SNPs — this will be extremely slow. "
            f"Window or subsample first (max recommended: ~4000)."
        )
    if m == 0:
        return np.zeros((0, 0), dtype=float)
    if m == 1:
        return np.ones((1, 1), dtype=float)

    mask = np.isfinite(G)

    # -------------------------
    # FAST PATH: no missing data
    # -------------------------
    if np.all(mask):
        G0 = G - G.mean(axis=0)
        std = G.std(axis=0)

        C = np.full((m, m), np.nan, dtype=float)
        np.fill_diagonal(C, 1.0)

        valid = std > 1e-8
        if valid.sum() >= 2:
            Z = G0[:, valid] / std[valid]
            r = np.corrcoef(Z.T)
            r2v = np.clip(r, -1.0, 1.0) ** 2

            idx = np.where(valid)[0]
            C[np.ix_(idx, idx)] = r2v
            np.fill_diagonal(C, 1.0)

        return C

    # -------------------------
    # OPTIMIZED MISSING-DATA PATH
    # Uses mean-imputation ONLY for LD estimation (not stored).
    # Falls back to pairwise-complete for high-missingness pairs.
    # -------------------------
    C = np.full((m, m), np.nan, dtype=float)
    np.fill_diagonal(C, 1.0)

    min_pair_n = int(min_pair_n)

    # Per-SNP missingness rate
    miss_rate = 1.0 - mask.mean(axis=0)

    # SNPs with low missingness: mean-impute and use fast path
    LOW_MISS = 0.05  # 5% threshold
    low_miss = miss_rate <= LOW_MISS
    high_miss = ~low_miss

    # --- Fast block: low-missingness SNPs (mean-imputed, vectorized) ---
    idx_low = np.where(low_miss)[0]
    if idx_low.size >= 2:
        G_imp = G[:, idx_low].copy()
        col_means = np.nanmean(G_imp, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        nan_locs = np.where(~np.isfinite(G_imp))
        G_imp[nan_locs] = col_means[nan_locs[1]]

        std_imp = G_imp.std(axis=0)
        valid_std = std_imp > 1e-8

        if valid_std.sum() >= 2:
            Z_imp = G_imp[:, valid_std] - G_imp[:, valid_std].mean(axis=0)
            sd = Z_imp.std(axis=0)
            sd[sd == 0] = 1.0
            Z_imp = Z_imp / sd
            r_block = np.corrcoef(Z_imp.T)
            r2_block = np.clip(r_block, -1.0, 1.0) ** 2

            idx_valid = idx_low[valid_std]
            C[np.ix_(idx_valid, idx_valid)] = r2_block
            np.fill_diagonal(C, 1.0)

    # --- Slow block: pairs involving high-missingness SNPs ---
    idx_high = np.where(high_miss)[0]

    # Two complementary loops cover all pairs involving high-missingness SNPs:
    #   Loop 1: (high_i, any j > high_i) — covers high-high and high-low pairs
    #   Loop 2: (low_i, high_j > low_i) — covers low-high pairs not in loop 1
    pairs_to_check = []
    for i in idx_high:
        for j in range(i + 1, m):
            pairs_to_check.append((i, j))
    for i in idx_low:
        for j in idx_high:
            if j > i:
                pairs_to_check.append((i, j))

    for (i, j) in pairs_to_check:
        valid_ij = mask[:, i] & mask[:, j]
        nv = int(valid_ij.sum())
        if nv < min_pair_n:
            continue

        xi = G[valid_ij, i]
        xj = G[valid_ij, j]

        if np.std(xi) < 1e-8 or np.std(xj) < 1e-8:
            continue

        r = np.corrcoef(xi, xj)[0, 1]
        if np.isfinite(r):
            val = float(np.clip(r, -1.0, 1.0) ** 2)
            C[i, j] = val
            C[j, i] = val

    return C

def pairwise_r(geno_matrix, min_pair_n: int = 20):
    """
    Mask-aware Pearson correlation (r), consistent with pairwise_r2().
    Always returns full m×m matrix aligned to original SNP order.
    Monomorphic SNPs → NaN off-diagonal, diag=1.
    """
    G = np.asarray(geno_matrix, float)
    mask = np.isfinite(G)
    n, m = G.shape

    # FAST PATH: no missing data
    if np.all(mask):
        G0 = G - G.mean(axis=0)
        std = G.std(axis=0)

        C = np.full((m, m), np.nan, float)
        np.fill_diagonal(C, 1.0)

        valid = std > 1e-8
        if valid.sum() >= 2:
            Z = G0[:, valid] / std[valid]
            r = np.corrcoef(Z.T)
            r = np.clip(r, -1, 1)

            idx = np.where(valid)[0]
            C[np.ix_(idx, idx)] = r
            np.fill_diagonal(C, 1.0)

        return C

    # MISSING-DATA PATH
    if m < 2:
        return np.ones((m, m), float)

    C = np.full((m, m), np.nan, float)
    np.fill_diagonal(C, 1.0)

    for i in range(m):
        for j in range(i + 1, m):
            valid = mask[:, i] & mask[:, j]
            nv = int(valid.sum())
            if nv < int(min_pair_n):
                continue

            xi = G[valid, i]
            xj = G[valid, j]

            if np.nanstd(xi) < 1e-8 or np.nanstd(xj) < 1e-8:
                continue

            r = np.corrcoef(xi, xj)[0, 1]
            if np.isfinite(r):
                C[i, j] = C[j, i] = float(np.clip(r, -1.0, 1.0))

    return C
# -------------------------------------------------
# Shared contiguous LD segmentation helper
# -------------------------------------------------
def contiguous_segments_by_adjacent(
    region_pos: np.ndarray,
    r2: np.ndarray,
    adj_r2_min: float = 0.3,
    gap_factor: float = 2.5,
    min_len: int = 2,
):
    """
    Split SNP region into contiguous LD-coherent segments.

    Breakpoints occur when:
      - adjacent r² drops below adj_r2_min, OR
      - physical gap between SNPs is unusually large.

    Returns:
      segments : list of index arrays
      order    : sorting order applied to positions
    """
    region_pos = np.asarray(region_pos, int)
    order = np.argsort(region_pos)
    region_pos = region_pos[order]
    r2 = r2[np.ix_(order, order)]

    if region_pos.size < 2:
        return [], order

    gaps = np.diff(region_pos)
    median_gap = float(np.median(gaps)) if gaps.size else 0.0

    segs = []
    start = 0

    for k in range(region_pos.size - 1):
        coh = float(r2[k, k + 1])
        gap_bp = int(region_pos[k + 1] - region_pos[k])

        # Split if LD is low OR undefined (low sample overlap)
        split_by_ld = (not np.isfinite(coh)) or (coh < float(adj_r2_min))
        split_by_gap = (median_gap > 0) and (gap_bp > gap_factor * median_gap)

        if split_by_ld or split_by_gap:
            if (k + 1 - start) >= int(min_len):
                segs.append(np.arange(start, k + 1))
            start = k + 1

    if (region_pos.size - start) >= int(min_len):
        segs.append(np.arange(start, region_pos.size))

    return segs, order

# -------------------------------------------------
# 2) LD Decay
# -------------------------------------------------
def ld_decay(region_pos, r2, ld_threshold=0.2, n_bins=40, max_dist_kb=None):
    """
    Estimate LD decay distance as the smallest distance bin
    where median r² <= ld_threshold.

    Returns:
      dist_kb_est : estimated decay distance (kb)
      slope       : slope of r² vs log10(distance) (descriptive only; not a
                    recombination parameter estimate; heavily autocorrelated)
      df_ld       : pairwise LD dataframe
    """
    region_pos = np.asarray(region_pos, int)
    r2 = np.asarray(r2, float)
    # SAFETY: cap SNP count to avoid O(n²) explosion
    MAX_LD_SNPS = 2000
    if region_pos.size > MAX_LD_SNPS:
        idx = np.linspace(0, region_pos.size - 1, MAX_LD_SNPS, dtype=int)
        region_pos = region_pos[idx]
        r2 = r2[np.ix_(idx, idx)]

    # SAFETY: ensure positions are sorted
    order = np.argsort(region_pos)
    region_pos = region_pos[order]
    r2 = r2[np.ix_(order, order)]

    # Pairwise distances
    dist_bp = np.abs(np.subtract.outer(region_pos, region_pos))
    iu = np.triu_indices_from(r2, k=1)

    df_ld = pd.DataFrame({
        "dist_bp": dist_bp[iu],
        "r2": r2[iu],
    })
    df_ld = df_ld[df_ld["dist_bp"] > 0]
    df_ld["dist_kb"] = df_ld["dist_bp"] / 1000.0

    finite_mask = (
        np.isfinite(df_ld["dist_kb"]) &
        (df_ld["dist_kb"] > 0) &
        np.isfinite(df_ld["r2"])
    )
    df_ld = df_ld.loc[finite_mask].copy()

    if len(df_ld) <= 20:
        return np.nan, np.nan, df_ld

    # Optionally restrict to a maximum distance (e.g. 5–10 Mb)
    if max_dist_kb is not None:
        df_ld = df_ld[df_ld["dist_kb"] <= max_dist_kb]
        if len(df_ld) <= 20:
            return np.nan, np.nan, df_ld

    # ---- slope of r² vs log10(distance) for descriptive purposes ----
    xlog = np.log10(df_ld["dist_kb"].clip(lower=1e-6))
    y = df_ld["r2"]
    slope, intercept, _, _, _ = linregress(xlog, y)

    # ---- Bin distances and compute median r² per bin (robust to outliers) ----
    dmin, dmax = df_ld["dist_kb"].min(), df_ld["dist_kb"].max()
    bins = np.linspace(dmin, dmax, n_bins + 1)
    df_ld["bin"] = pd.cut(df_ld["dist_kb"], bins=bins, labels=False, include_lowest=True)

    bin_stats = (
        df_ld.groupby("bin")["r2"]
        .median()
        .reset_index()
        .dropna()
    )

    # Middle distance of each bin
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_stats["dist_kb"] = bin_centers[bin_stats["bin"].values]

    # Find first bin where median r² <= threshold
    below = bin_stats[bin_stats["r2"] <= ld_threshold]
    if below.empty:
        dist_kb_est = np.nan
    else:
        dist_kb_est = float(below["dist_kb"].iloc[0])

    return dist_kb_est, slope, df_ld




# -------------------------------------------------
# 3) LD block finder (gap-aware)
# -------------------------------------------------
def find_ld_clusters(region_pos, r2_lead, ld_threshold, min_snps=3):
    """
    Given region positions and r2 vector vs lead SNP,
    return LD blocks using gap-aware expansion.

    Returns list of (start_bp, end_bp, n_snps, mean_r2)
    """

    # SAFETY: ensure SNPs are sorted by position
    order = np.argsort(region_pos)
    region_pos = np.asarray(region_pos)[order]
    r2_lead = np.asarray(r2_lead)[order]

    hits = (r2_lead >= ld_threshold)
    hits = np.asarray(hits, bool)


    if len(region_pos) >= 2:
        median_gap = np.median(np.diff(region_pos))
    else:
        median_gap = 0

    max_gap_bp = 3 * median_gap if median_gap > 0 else 0
    max_gap_snps = 3

    clusters = []
    i = 0

    while i < len(hits):

        if not hits[i]:
            i += 1
            continue

        start = i
        gap_snps = 0
        j = i

        while j + 1 < len(hits):
            next_ok = hits[j + 1]
            gap_bp = region_pos[j + 1] - region_pos[j]

            if next_ok:
                gap_snps = 0
                j += 1
            elif gap_snps + 1 <= max_gap_snps and gap_bp <= max_gap_bp:
                gap_snps += 1
                j += 1
            else:
                break

        idx = np.arange(start, j + 1)
        keep_idx = idx[hits[idx]]

        if len(keep_idx) >= min_snps:
            start_bp = int(region_pos[keep_idx.min()])
            end_bp = int(region_pos[keep_idx.max()])
            mean_r2 = float(np.mean(r2_lead[keep_idx]))
            clusters.append((start_bp, end_bp, len(keep_idx), mean_r2))

        i = j + 1

    return clusters


# -------------------------------------------------
# 4) Rectangle drawing for heatmaps
# -------------------------------------------------
def highlight_ld_block(ax, start_idx, end_idx, color="white", lw=2):
    width = end_idx - start_idx + 1
    rect = patches.Rectangle(
        (start_idx, start_idx),
        width,
        width,
        fill=False,
        edgecolor=color,
        linewidth=lw,
    )
    ax.add_patch(rect)

def get_block_snp_mask(block_row, chroms, positions, sid):
    """Select SNPs belonging to an LD block, using member IDs if available.

    When the block row contains a 'SNP_IDs' column (comma-separated SNP IDs),
    selection is exact. Otherwise falls back to coordinate-range filtering
    for backward compatibility with old block DataFrames.
    """
    snp_ids_str = block_row.get("SNP_IDs", "")
    if pd.notna(snp_ids_str) and str(snp_ids_str).strip():
        member_set = set(str(snp_ids_str).split(","))
        return np.isin(np.asarray(sid, dtype=str), list(member_set))
    # Fallback: coordinate range (legacy behavior)
    chr_sel = str(block_row["Chr"])
    start_bp = int(block_row["Start (bp)"])
    end_bp = int(block_row["End (bp)"])
    return (
        (np.asarray(chroms, dtype=str) == chr_sel) &
        (np.asarray(positions) >= start_bp) &
        (np.asarray(positions) <= end_bp)
    )


def find_ld_blocks_graph(
    region_pos: np.ndarray,
    r2: np.ndarray,
    ld_threshold: float = 0.6,
    max_dist_bp: int | None = None,
    min_snps: int = 3,
    adj_r2_min: float = 0.3,
    gap_factor: float = 10.0,
    region_sids: np.ndarray | None = None,
):
    """
    LD block detection using a graph-based approach + contiguity refinement.

    Step 1: Graph
      Each SNP is a node. Add an undirected edge i-j if:
        - r²(i, j) >= ld_threshold
        - and (optionally) distance <= max_dist_bp

      Candidate blocks are connected components.
    Step 2: Contiguity refinement to avoid patchy LD blocks
      Split each connected component into *contiguous* sub-blocks by genomic order:
        - break between consecutive SNPs if r²(i, i+1) < adj_r2_min

      This prevents long-range edges from creating huge patchy "blocks" in dense regions.
    """
    region_pos = np.asarray(region_pos, dtype=int)
    r2 = np.asarray(r2, dtype=float)
    if region_sids is not None:
        region_sids = np.asarray(region_sids, dtype=str)

    # SAFETY: ensure SNPs are sorted by genomic position
    order = np.argsort(region_pos)
    region_pos = region_pos[order]
    r2 = r2[np.ix_(order, order)]
    if region_sids is not None:
        region_sids = region_sids[order]

    n = len(region_pos)
    if n < 2:
        return []

    # Adjacency list
    adj = [[] for _ in range(n)]

    # Build edges (distance-capped, early-break because positions sorted)
    for i in range(n):
        for j in range(i + 1, n):
            dist = region_pos[j] - region_pos[i]
            if max_dist_bp is not None and dist > max_dist_bp:
                break
            if r2[i, j] >= ld_threshold:
                # allow edge; coherence enforced later during refinement
                adj[i].append(j)
                adj[j].append(i)

    visited = np.zeros(n, dtype=bool)
    blocks = []

    def _emit_segment(seg_idx: np.ndarray):
        """Compute block tuple for a segment of indices."""
        if seg_idx.size < min_snps:
            return
        start_bp = int(region_pos[seg_idx].min())
        end_bp = int(region_pos[seg_idx].max())
        sub = r2[np.ix_(seg_idx, seg_idx)]
        iu = np.triu_indices_from(sub, k=1)
        vals = sub[iu]
        mean_r2 = float(np.nanmean(vals)) if (vals.size > 0 and np.any(np.isfinite(vals))) else np.nan
        member_ids = region_sids[seg_idx].tolist() if region_sids is not None else []
        blocks.append((start_bp, end_bp, int(seg_idx.size), mean_r2, member_ids))

    def _refine_component(comp_idx: np.ndarray):
        """
        Split a connected component into contiguous sub-blocks
        using the shared contiguous_segments_by_adjacent helper.
        """
        comp_idx = np.asarray(comp_idx, dtype=int)
        comp_idx = comp_idx[np.argsort(region_pos[comp_idx])]

        if comp_idx.size < min_snps:
            return

        # Adaptive adjacent LD threshold for this component
        sub_r2 = r2[np.ix_(comp_idx, comp_idx)]
        adj_thr_here = _adaptive_adj_threshold(
            sub_r2,
            base=float(adj_r2_min),
            frac=0.5,
        )

        segs, seg_order = contiguous_segments_by_adjacent(
            region_pos=region_pos[comp_idx],
            r2=sub_r2,
            adj_r2_min=adj_thr_here,
            gap_factor=float(gap_factor),
            min_len=int(min_snps),
        )

        for seg_local in segs:
            _emit_segment(comp_idx[seg_order[seg_local]])

    # DFS/BFS to find connected components, then refine each
    for start in range(n):
        if visited[start]:
            continue

        stack = [start]
        comp = []

        while stack:
            k = stack.pop()
            if visited[k]:
                continue
            visited[k] = True
            comp.append(k)
            for nb in adj[k]:
                if not visited[nb]:
                    stack.append(nb)

        if len(comp) >= min_snps:
            _refine_component(np.array(comp, dtype=int))

    # Sort blocks by start position
    blocks.sort(key=lambda x: x[0])
    return blocks


def find_ld_clusters_genomewide(
    gwas_df,
    chroms,
    positions,
    geno_imputed,
    sid,
    ld_threshold=0.6,
    flank_kb=300,
    min_snps=3,
    top_n=0,
    sig_thresh=1e-5,
    max_dist_bp=None,
    ld_decay_kb=None,
    adj_r2_min=0.2,
    min_pair_n: int = 20,
    merge_iou=0.3,
    gap_factor: float = 10.0
):

    """
    Peak-centric LD block detection.
    Returns DataFrame with columns: Chr, Start (bp), End (bp), Lead SNP.
    LD blocks are defined as connected components in an LD graph (r² ≥ threshold)
    within a flank window around GWAS-significant SNPs.
    """

    # --- 1) Select significant SNPs ---
    significant = gwas_df[gwas_df["PValue"] < sig_thresh].copy()
    significant["Chr"] = significant["Chr"].astype(str).map(canon_chr)


    # Add top-N if requested
    if top_n > 0:
        best_n = gwas_df.nsmallest(top_n, "PValue")
        significant = (
            pd.concat([significant, best_n])
            .drop_duplicates(subset=["SNP"])
        )

    if significant.empty:
        return pd.DataFrame(columns=["Chr", "Start (bp)", "End (bp)", "Lead SNP"])

    flank_bp = int(round(float(flank_kb) * 1000.0))


    all_blocks = []
    # Cache r² per (chr, start, end, m) window to avoid recomputing for overlapping lead SNPs
    _r2_window_cache = {}

    chroms = np.asarray([canon_chr(c) for c in chroms], dtype=object)
    positions = np.asarray(positions)
    sid = np.asarray(sid).astype(str)

    # --- 2) Loop over significant SNPs ---
    for _, snp_row in significant.iterrows():

        chr_sel = canon_chr(snp_row["Chr"])
        pos_sel = int(snp_row["Pos"])
        snp_id  = str(snp_row["SNP"])

        # Filter single chromosome
        chr_mask = chroms.astype(str) == chr_sel
        chr_positions = positions[chr_mask]
        chr_geno = np.asarray(geno_imputed[:, chr_mask], float)
        chr_sids      = sid[chr_mask]

        # Window around lead SNP
        reg_mask = (
            (chr_positions >= pos_sel - flank_bp) &
            (chr_positions <= pos_sel + flank_bp)
        )
        region_pos  = chr_positions[reg_mask]
        region_geno = chr_geno[:, reg_mask]
        region_sids = chr_sids[reg_mask]
        if region_geno.shape[1] < 2:
            continue

        # Remove monomorphic
        snp_var = np.nanvar(region_geno, axis=0)
        keep = snp_var > 0
        region_geno = region_geno[:, keep]
        region_pos  = region_pos[keep]
        region_sids = region_sids[keep]

        if region_geno.shape[1] < 2:
            continue

        # -------------------------------------------------
        # Ensure SNPs sorted by genomic position
        # (required for LD graph logic)
        # -------------------------------------------------
        order = np.argsort(region_pos)
        region_pos = region_pos[order]
        region_geno = region_geno[:, order]
        region_sids = region_sids[order]

        # -------------------------------------------------
        # PERFORMANCE SAFETY:
        # cap SNP count in LD region to avoid O(m²) blowups
        # -------------------------------------------------
        MAX_REGION_SNPS = 1500

        if region_geno.shape[1] > MAX_REGION_SNPS:
            idx = np.linspace(
                0,
                region_geno.shape[1] - 1,
                MAX_REGION_SNPS,
                dtype=int
            )

            # Always re-sort after subsetting
            region_geno = region_geno[:, idx]
            region_pos = region_pos[idx]
            region_sids = region_sids[idx]

            # IMPORTANT: keep genomic order intact
            order = np.argsort(region_pos)
            region_pos = region_pos[order]
            region_geno = region_geno[:, order]
            region_sids = region_sids[order]

        # --- Compute full pairwise LD for the region ---
        cache_key = (
            chr_sel,
            int(region_pos.min()),
            int(region_pos.max()),
            int(region_geno.shape[1]),
            int(region_geno.shape[0]),
            hashlib.blake2b(region_sids.astype(str).tobytes(), digest_size=8).hexdigest(),
        )
        if cache_key in _r2_window_cache:
            r2 = _r2_window_cache[cache_key]
        else:
            try:
                r2 = pairwise_r2(region_geno, min_pair_n=int(min_pair_n))
            except RuntimeError:
                continue
            _r2_window_cache[cache_key] = r2
        # --- choose LD distance cap robustly ---
        # LD decay estimates can be underestimated on sparse SNP sets
        # so enforce a reasonable minimum distance

        DIST_FLOOR_BP = 200_000  # 200 kb floor (adjust later if needed)

        if ld_decay_kb is not None and np.isfinite(ld_decay_kb):
            global_ld_bp = int(ld_decay_kb * 1000)
            max_dist_bp_use = min(max(global_ld_bp, DIST_FLOOR_BP), flank_bp)
        else:
            max_dist_bp_use = flank_bp

        # Explicit override always wins
        if max_dist_bp is not None:
            max_dist_bp_use = int(max_dist_bp)

        # --- Hybrid LD block detection ---
        # Graph connectivity first, then contiguity refinement
        blocks = find_ld_blocks_graph(
            region_pos=region_pos,
            r2=r2,
            ld_threshold=ld_threshold,
            max_dist_bp=max_dist_bp_use,
            min_snps=min_snps,
            adj_r2_min=float(adj_r2_min),
            gap_factor=float(gap_factor),
            region_sids=region_sids,
        )

        for (start_bp, end_bp, n_snps, mean_r2, member_ids) in blocks:
            all_blocks.append([chr_sel, start_bp, end_bp, snp_id,
                               ",".join(member_ids) if member_ids else ""])

    if not all_blocks:
        return pd.DataFrame(columns=["Chr", "Start (bp)", "End (bp)", "Lead SNP", "SNP_IDs"])

    df = pd.DataFrame(
        all_blocks,
        columns=["Chr", "Start (bp)", "End (bp)", "Lead SNP", "SNP_IDs"]
    ).drop_duplicates()

    # Merge overlapping blocks (IoU > 0.3)
    merged = []
    df = df.sort_values(["Chr", "Start (bp)"])

    for _, row in df.iterrows():
        cur = row.to_dict()

        # try merge backward as long as overlaps remain high
        while merged and cur["Chr"] == merged[-1]["Chr"]:
            last = merged[-1]
            iou = _interval_iou_bp(
                cur["Start (bp)"], cur["End (bp)"],
                last["Start (bp)"], last["End (bp)"]
            )
            if iou > float(merge_iou):
                last["Start (bp)"] = int(min(last["Start (bp)"], cur["Start (bp)"]))
                last["End (bp)"] = int(max(last["End (bp)"], cur["End (bp)"]))
                last["Lead SNP"] = _merge_leads(last.get("Lead SNP", ""), cur.get("Lead SNP", ""))
                # Union member SNP IDs
                last_ids = set(filter(None, last.get("SNP_IDs", "").split(",")))
                cur_ids = set(filter(None, cur.get("SNP_IDs", "").split(",")))
                last["SNP_IDs"] = ",".join(sorted(last_ids | cur_ids))
                cur = last
                merged.pop()
            else:
                break

        merged.append(cur)

    return pd.DataFrame(merged)

def find_ld_blocks_from_genotypes(
    chroms,
    positions,
    geno_imputed,
    sid,
    ld_threshold=0.6,
    max_dist_bp=300000,
    min_snps=3,
    adj_r2_min=0.2,
    gap_factor=10.0,
):

    """
    Genome-wide LD block detection independent of GWAS significance.

    Parameters
    ----------
    chroms : array-like
        Chromosome labels for each SNP.
    positions : array-like
        SNP positions (bp).
    geno_imputed : ndarray
        Genotype matrix (n_samples × n_snps).
    sid : array-like
        SNP IDs.
    ld_threshold : float
        Minimum r^2 to connect two SNPs in LD graph.
    max_dist_bp : int
        Maximum distance (bp) to consider LD edges.
    window_size : int
        Number of SNPs to include per local LD calculation chunk.
    min_snps : int
        Minimum SNPs per block.

    Returns
    -------
    DataFrame with columns:
        ["Chr", "Start (bp)", "End (bp)", "n_snps", "Lead SNP"]
    """
    import numpy as np
    import pandas as pd

    chroms = np.asarray([canon_chr(c) for c in chroms], dtype=object)
    positions = np.asarray(positions)
    sid = np.asarray(sid).astype(str)

    results = []

    # Loop chromosome-wise
    for ch in np.unique(chroms.astype(str)):
        mask = chroms.astype(str) == str(ch)
        idx = np.where(mask)[0]

        if len(idx) < min_snps:
            continue

        chr_pos = positions[idx]
        chr_sid = sid[idx]
        chr_geno = geno_imputed[:, idx]

        # Sort by position
        order = np.argsort(chr_pos)
        chr_pos = chr_pos[order]
        chr_sid = chr_sid[order]
        chr_geno = chr_geno[:, order]

        # Recompute n after filtering
        n = chr_geno.shape[1]
        if n < min_snps:
            continue
        # -------------------------------------------------
        # Vectorized LD computation + union-find
        # Mean-impute once, standardize, then compute r² via
        # chunked matrix multiplication (~5-20x faster)
        # -------------------------------------------------

        # Mean-impute for LD estimation only (not stored)
        G_imp = chr_geno.copy()
        col_means = np.nanmean(G_imp, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        nan_locs = np.where(~np.isfinite(G_imp))
        if nan_locs[0].size > 0:
            G_imp[nan_locs] = col_means[nan_locs[1]]

        # Standardize columns
        mu = G_imp.mean(axis=0)
        sd = G_imp.std(axis=0)
        sd[sd < 1e-8] = 1.0  # monomorphic → won't form edges
        Z = (G_imp - mu) / sd
        n_samp = Z.shape[0]

        parent = np.arange(n)
        rank_uf = np.zeros(n, dtype=int)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                if rank_uf[rx] < rank_uf[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank_uf[rx] == rank_uf[ry]:
                    rank_uf[rx] += 1

        # Build LD edges in vectorized chunks
        CHUNK = 500
        max_neighbors = 2000

        for chunk_start in range(0, n, CHUNK):
            chunk_end = min(n, chunk_start + CHUNK)

            # Find rightmost target within max_dist_bp of any SNP in this chunk
            farthest = chunk_end
            for i in range(chunk_start, chunk_end):
                while farthest < n and (chr_pos[farthest] - chr_pos[i]) <= max_dist_bp:
                    farthest = min(farthest + 1, n)
            farthest = min(farthest, chunk_start + max_neighbors + CHUNK)

            if farthest <= chunk_start:
                continue

            # Vectorized r² for this chunk
            Z_query = Z[:, chunk_start:chunk_end]
            Z_target = Z[:, chunk_start:farthest]
            R = (Z_query.T @ Z_target) / n_samp
            R2 = np.clip(R, -1.0, 1.0) ** 2

            # Build edges from R2 matrix
            for i_local in range(chunk_end - chunk_start):
                i_global = chunk_start + i_local
                for j_local in range(i_local + 1, R2.shape[1]):
                    j_global = chunk_start + j_local
                    if j_global >= n:
                        break
                    if (chr_pos[j_global] - chr_pos[i_global]) > max_dist_bp:
                        break
                    if R2[i_local, j_local] >= ld_threshold:
                        union(i_global, j_global)

        # Also keep compute_ld_cached for the refinement step below
        _ld_cache = {}

        def compute_ld_cached(i, j, min_pair_n=20):
            key = (min(i, j), max(i, j))
            if key in _ld_cache:
                return _ld_cache[key]
            xi = chr_geno[:, i]
            xj = chr_geno[:, j]
            valid = np.isfinite(xi) & np.isfinite(xj)
            nv = valid.sum()
            if nv < min_pair_n:
                val = np.nan
            else:
                xi, xj = xi[valid], xj[valid]
                si, sj = np.std(xi), np.std(xj)
                if si < 1e-8 or sj < 1e-8:
                    val = np.nan
                else:
                    r = np.corrcoef(xi, xj)[0, 1]
                    val = float(np.clip(r, -1.0, 1.0) ** 2)
            _ld_cache[key] = val
            return val

        # Extract connected components
        components = {}
        for i in range(n):
            root = find(i)
            components.setdefault(root, []).append(i)

        # Emit blocks (deterministic ordering by genomic start)
        comps = list(components.values())
        comps.sort(key=lambda c: int(chr_pos[np.min(c)]))

        # Hard cap on block size: 2× max_dist_bp (prevents chromosome-arm blocks)
        MAX_BLOCK_BP = int(2 * max_dist_bp)

        def _emit_component_segment(seg):
            if len(seg) < int(min_snps):
                return
            seg = np.asarray(seg, dtype=int)
            block_pos = chr_pos[seg]
            start_bp = int(block_pos.min())
            end_bp = int(block_pos.max())

            # If block exceeds hard cap, split at weakest adjacent r²
            if (end_bp - start_bp) > MAX_BLOCK_BP and len(seg) >= 2 * min_snps:
                adj_r2_vals = np.array([
                    compute_ld_cached(seg[k], seg[k + 1])
                    for k in range(len(seg) - 1)
                ])
                # Replace NaN with 0 for argmin
                adj_r2_vals = np.where(np.isfinite(adj_r2_vals), adj_r2_vals, 0.0)
                split_at = int(np.argmin(adj_r2_vals)) + 1
                # Recurse on both halves
                _emit_component_segment(seg[:split_at])
                _emit_component_segment(seg[split_at:])
                return

            mid_bp = 0.5 * (start_bp + end_bp)
            lead_snp = chr_sid[seg[np.argmin(np.abs(block_pos - mid_bp))]]

            results.append({
                "Chr": str(ch),
                "Start (bp)": start_bp,
                "End (bp)": end_bp,
                "n_snps": int(len(seg)),
                "Lead SNP": lead_snp,
                "SNP_IDs": ",".join(chr_sid[seg].astype(str).tolist()),
            })

        def _refine_component_by_adjacent_ld(comp):
            comp = np.asarray(comp, dtype=int)
            comp = comp[np.argsort(chr_pos[comp])]

            if comp.size < int(min_snps):
                return

            # Build local r² matrix for the component (from cache, fast)
            m_c = comp.size
            sub_r2 = np.full((m_c, m_c), np.nan, dtype=float)
            np.fill_diagonal(sub_r2, 1.0)
            for a in range(m_c):
                for b in range(a + 1, m_c):
                    val = compute_ld_cached(comp[a], comp[b])
                    sub_r2[a, b] = val
                    sub_r2[b, a] = val

            adj_thr_here = _adaptive_adj_threshold(
                sub_r2,
                base=float(adj_r2_min),
                frac=0.5,
            )

            segs, seg_order = contiguous_segments_by_adjacent(
                region_pos=chr_pos[comp],
                r2=sub_r2,
                adj_r2_min=adj_thr_here,
                gap_factor=float(gap_factor),
                min_len=int(min_snps),
            )

            for seg_local in segs:
                _emit_component_segment(comp[seg_order[seg_local]])

        for comp in comps:
            if len(comp) < min_snps:
                continue
            _refine_component_by_adjacent_ld(comp)

    return pd.DataFrame(results)

# ============================================================
# Block-level utility functions (moved from pages/LD analysis.py)
# ============================================================

def filter_contained_blocks(
    blocks_df,
    min_contained=2,
    size_ratio_threshold=3.0,
    mode="remove",
):
    """
    Remove or flag LD blocks that fully contain multiple smaller blocks.
    Designed to eliminate transitive LD chaining artifacts.
    """

    if blocks_df is None or blocks_df.empty:
        return blocks_df, 0

    df = blocks_df.copy()

    # Normalize column names
    if "Start (bp)" in df.columns:
        df = df.rename(columns={"Start (bp)": "Start", "End (bp)": "End"})
        had_bp_suffix = True
    else:
        had_bp_suffix = False

    df["Chr"] = df["Chr"].astype(str)
    df["Start"] = df["Start"].astype(int)
    df["End"] = df["End"].astype(int)

    df["_size"] = df["End"] - df["Start"]
    df["_idx"] = range(len(df))

    n_contained = [0] * len(df)
    contained_sizes = [[] for _ in range(len(df))]
    is_mega = [False] * len(df)

    # Check containment chromosome-wise
    for ch, group in df.groupby("Chr"):
        idxs = group["_idx"].tolist()

        for i in idxs:
            si = df.loc[df["_idx"] == i, "Start"].iloc[0]
            ei = df.loc[df["_idx"] == i, "End"].iloc[0]

            for j in idxs:
                if i == j:
                    continue

                sj = df.loc[df["_idx"] == j, "Start"].iloc[0]
                ej = df.loc[df["_idx"] == j, "End"].iloc[0]

                if si <= sj and ei >= ej:
                    n_contained[i] += 1
                    contained_sizes[i].append(ej - sj)

    df["n_contained_blocks"] = n_contained

    # Flag mega-blocks
    for i in range(len(df)):
        if n_contained[i] >= min_contained and contained_sizes[i]:
            med = sorted(contained_sizes[i])[len(contained_sizes[i]) // 2]
            if med > 0 and df.loc[df["_idx"] == i, "_size"].iloc[0] / med >= size_ratio_threshold:
                is_mega[i] = True

    df["is_mega_block"] = is_mega
    df = df.drop(columns=["_size", "_idx"])

    if had_bp_suffix:
        df = df.rename(columns={"Start": "Start (bp)", "End": "End (bp)"})

    if mode == "remove":
        n_removed = int(df["is_mega_block"].sum())
        df = df[~df["is_mega_block"]].drop(columns=["is_mega_block"]).reset_index(drop=True)
        return df, n_removed

    return df, 0


def extract_block_geno_for_paper(
    geno_imputed, chroms, positions, sid,
    block_chr, start_bp, end_bp,
    sample_keep_mask=None,
    maf_threshold=None,
    snp_ids=None,
):
    chroms_np = np.asarray(chroms, dtype=object).astype(str)
    pos_np = np.asarray(positions, dtype=object).astype(int)
    sid_np = np.asarray(sid, dtype=object).astype(str)

    G = np.asarray(geno_imputed, dtype=float)

    # SAFETY: SNP metadata must match genotype columns
    m = int(G.shape[1])
    if not (chroms_np.size == pos_np.size == sid_np.size == m):
        raise ValueError(
            "SNP axis mismatch in extract_block_geno_for_paper(): "
            f"G has {m} SNP columns, but chroms={chroms_np.size}, "
            f"positions={pos_np.size}, sid={sid_np.size}. "
            "Ensure these arrays are derived from the same SNP set."
        )

    # Use member SNP IDs when available (exact), fall back to coordinate range
    if snp_ids and str(snp_ids) not in ("", "nan"):
        member_set = set(str(snp_ids).split(","))
        mask = np.isin(sid_np, list(member_set))
    else:
        block_chr_s = str(block_chr)
        mask = (chroms_np == block_chr_s) & (pos_np >= int(start_bp)) & (pos_np <= int(end_bp))
    mask = np.asarray(mask, dtype=bool)

    if sample_keep_mask is not None:
        sample_keep_mask = np.asarray(sample_keep_mask, dtype=bool)
        if sample_keep_mask.size != G.shape[0]:
            raise ValueError(
                "Sample mask mismatch in extract_block_geno_for_paper(): "
                f"G has {G.shape[0]} samples, but sample_keep_mask has {sample_keep_mask.size}."
            )
        G = G[sample_keep_mask, :]

    G = G[:, mask]
    sids = sid_np[mask]
    pos = pos_np[mask]

    if G.shape[1] < 2:
        return G, pos, sids

    # Remove monomorphic SNPs
    var = np.nanvar(G, axis=0)
    poly = var > 0
    G, pos, sids = G[:, poly], pos[poly], sids[poly]

    # Optional MAF filter
    if maf_threshold is not None and G.shape[1] > 0:
        p = np.nanmean(G, axis=0) / 2.0
        p = np.clip(p, 0.0, 1.0)
        maf = np.minimum(p, 1.0 - p)
        keep_maf = maf > float(maf_threshold)
        G, pos, sids = G[:, keep_maf], pos[keep_maf], sids[keep_maf]

    # Sort SNPs by genomic position
    order = np.argsort(pos)
    return G[:, order], pos[order], sids[order]


def _blocks_to_interval_set(df_blocks: pd.DataFrame):
    """
    Convert LD block table -> list of tuples (Chr, Start, End).
    Accepts both styles:
      - Start (bp) / End (bp)
      - Start / End
    """
    if df_blocks is None or not isinstance(df_blocks, pd.DataFrame) or df_blocks.empty:
        return []
    start_col = "Start (bp)" if "Start (bp)" in df_blocks.columns else "Start"
    end_col = "End (bp)" if "End (bp)" in df_blocks.columns else "End"
    out = []
    for _, r in df_blocks.iterrows():
        out.append((str(r["Chr"]), int(r[start_col]), int(r[end_col])))
    return out


def _guess_lead_col(df: pd.DataFrame) -> str | None:
    """
    Try to find the column in haplotype/LD-block tables that represents
    the lead SNP. Handles 'Lead SNP', 'Lead  SNP', 'Lead_SNP', etc.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None

    for c in df.columns:
        norm = c.replace(" ", "").replace("_", "").lower()
        if norm == "leadsnp":
            return c
    return None


def _merge_lead_strings(a, b) -> str:
    toks = set(_split_leads(a)) | set(_split_leads(b))
    return ";".join(sorted(toks))


def merge_blocks(df, gap_bp=None, chroms=None, positions=None):
    """
    Merge overlapping/adjacent LD blocks.

    Fixes:
      - do NOT sum n_snps (overlaps double-count)
      - deduplicate Lead SNP strings robustly
      - keep original column naming compatible (Start (bp)/End (bp))
      - if chroms/positions provided, recompute n_snps from actual coordinates
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    start_col = "Start (bp)" if "Start (bp)" in df.columns else "Start"
    end_col   = "End (bp)" if "End (bp)" in df.columns else "End"

    # detect any lead column variant
    lead_col = next(
        (c for c in ["Lead SNP", "Representative SNP", "Lead_SNP"] if c in df.columns),
        None,
    )

    has_nsnps = "n_snps" in df.columns

    df[start_col] = df[start_col].astype(int)
    df[end_col] = df[end_col].astype(int)

    df = df.rename(columns={start_col: "Start (bp)", end_col: "End (bp)"})
    df["Chr"] = df["Chr"].astype(str)
    df = df.sort_values(["Chr", "Start (bp)"]).reset_index(drop=True)

    if gap_bp is None:
        med = np.median(df["End (bp)"] - df["Start (bp)"])
        gap_bp = max(int(0.1 * med), 1000)

    merged = []

    for _, row in df.iterrows():
        rowd = row.to_dict()

        # normalize lead key to "Lead SNP" if present
        if lead_col and lead_col != "Lead SNP":
            rowd["Lead SNP"] = rowd.get(lead_col, "")

        if not merged:
            merged.append(rowd)
            continue

        last = merged[-1]

        if rowd["Chr"] == last["Chr"] and rowd["Start (bp)"] <= last["End (bp)"] + gap_bp:
            last["End (bp)"] = max(int(last["End (bp)"]), int(rowd["End (bp)"]))

            if lead_col:
                last["Lead SNP"] = _merge_lead_strings(last.get("Lead SNP", ""), rowd.get("Lead SNP", ""))

            if has_nsnps:
                # DON'T sum: overlap causes double count.
                # Use max as a conservative non-inflating proxy.
                last["n_snps"] = int(max(int(last.get("n_snps", 0)), int(rowd.get("n_snps", 0))))

        else:
            merged.append(rowd)

    result = pd.DataFrame(merged)

    # Recompute n_snps from actual positions if arrays are available
    if chroms is not None and positions is not None and "n_snps" in result.columns:
        chroms_np = np.asarray(chroms).astype(str)
        pos_np = np.asarray(positions).astype(int)

        new_counts = []
        for _, row in result.iterrows():
            ch = str(row["Chr"])
            s = int(row["Start (bp)"])
            e = int(row["End (bp)"])
            mask = (chroms_np == ch) & (pos_np >= s) & (pos_np <= e)
            new_counts.append(int(mask.sum()))
        result["n_snps"] = new_counts

    return result


def maf_from_matrix(G: np.ndarray, encoding: str | None) -> np.ndarray:
    """
    Compute per-SNP MAF robustly for:
      - dosage012 : p = mean(dosage)/2
      - dosage02  : p = mean(dosage)/2  (hets absent, still valid)
      - binary01  : p = mean(value)     (no /2)
    """
    X = np.asarray(G, float)

    if encoding == "binary01":
        p = np.nanmean(X, axis=0)
    else:
        p = np.nanmean(X, axis=0) / 2.0

    p = np.clip(p, 0.0, 1.0)
    maf = np.minimum(p, 1.0 - p)
    return maf


def anova_eta_sq_from_labels(y: np.ndarray, labels: np.ndarray) -> tuple[float, int, int]:
    """
    Compute eta-squared (n^2) for one-way ANOVA:
      n^2 = SS_between / SS_total
    Returns: (eta_sq, n_groups, n_used)
    """
    y = np.asarray(y, float).reshape(-1)
    labels = np.asarray(labels, object)
    keep = np.isfinite(y) & (labels != None)
    y = y[keep]
    labels = np.asarray(labels, object)[keep]

    if y.size < 5:
        return np.nan, 0, int(y.size)

    # groups
    df = pd.DataFrame({"y": y, "g": labels.astype(str)})
    groups = [v.values for _, v in df.groupby("g")["y"]]
    if len(groups) < 2:
        return np.nan, len(groups), int(y.size)

    y_all = df["y"].values
    grand = float(np.mean(y_all))
    ss_total = float(np.sum((y_all - grand) ** 2))

    # avoid division by 0 when phenotype is constant
    if ss_total <= 0:
        return np.nan, len(groups), int(y.size)

    ss_between = 0.0
    for _, sub in df.groupby("g"):
        m = float(np.mean(sub["y"].values))
        ss_between += float(len(sub)) * (m - grand) ** 2

    eta_sq = ss_between / ss_total
    return float(eta_sq), int(len(groups)), int(y.size)

