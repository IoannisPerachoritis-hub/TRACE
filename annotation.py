"""Gene annotation and LD decay for TRACE.

Reads Sol_genes.csv and ITAG4.0 / SL3.1 description files,
annotates LD blocks with gene content, and computes LD decay curves.
"""

import numpy as np
import pandas as pd
import re



# ============================================================
# 1. GENE ANNOTATION LOADING
# ============================================================

def canon_chr(x) -> str:
    s = str(x).strip()
    s = re.sub(r"(?i)^chromosome[_\-]?", "", s)
    s = re.sub(r"(?i)^chr", "", s)
    s = re.sub(r"(?i)^[a-z]+[\d\.]*ch(?=\d)", "", s)  # SL4.0ch, ST4.03ch, ITAG
    s = re.sub(r"(?i)^(?:os|gm|at|zm|pv|ps|ah)[_\-]?(?=\d)", "", s)
    s = re.sub(r"(?i)^ca[p]?[_\-]?(?:chr)?(?=\d)", "", s)
    s = re.sub(r"(?i)^(\d+)h$", r"\1", s)  # barley 1H → 1
    s = re.sub(r"^0+", "", s)
    return s if s != "" else "0"

_canon_chr = canon_chr  # backward-compat alias


def load_gene_model(path) -> pd.DataFrame:
    """
    Load Sol_genes.csv (or similar gene coordinate file).
    Expected columns: CHROM, START, END, STRAND, GENE
    Returns DataFrame with: Chr, Start, End, Strand, Gene_ID
    """
    # encoding="utf-8" — Sol_genes.csv ships UTF-8; default cp1252 on Windows
    # would fail on non-ASCII gene descriptions.
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    df.columns = df.columns.str.strip()

    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("chrom", "chr", "chromosome"):
            col_map[c] = "Chr"
        elif cl in ("start", "start_pos", "begin"):
            col_map[c] = "Start"
        elif cl in ("end", "end_pos", "stop"):
            col_map[c] = "End"
        elif cl in ("strand",):
            col_map[c] = "Strand"
        elif cl in ("gene", "gene_id", "name", "gene_name"):
            col_map[c] = "Gene_ID"

    df = df.rename(columns=col_map)

    required = {"Chr", "Start", "End", "Gene_ID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Gene model file missing columns: {missing}. Found: {list(df.columns)}")

    df["Chr"] = df["Chr"].astype(str).map(canon_chr)
    df["Start"] = pd.to_numeric(df["Start"], errors="coerce").astype("Int64")
    df["End"] = pd.to_numeric(df["End"], errors="coerce").astype("Int64")
    df["Gene_ID"] = df["Gene_ID"].astype(str).str.strip()

    if "Strand" not in df.columns:
        df["Strand"] = "."

    df = df.dropna(subset=["Start", "End"]).copy()
    df["Start"] = df["Start"].astype(int)
    df["End"] = df["End"].astype(int)

    # Sort numerically by chromosome, then by start position
    df["_chr_sort"] = pd.to_numeric(df["Chr"], errors="coerce").fillna(999)
    df = df.sort_values(["_chr_sort", "Start"]).drop(columns=["_chr_sort"]).reset_index(drop=True)
    return df


def load_gene_descriptions(path) -> dict:
    """
    Load gene descriptions file (tab-separated: gene_id \\t description).
    Returns dict: gene_id -> description.
    """
    desc = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) < 2:
                continue
            gid = parts[0].strip()
            d = parts[1].strip()
            desc[gid] = d
            base = re.sub(r"(\.\d+)+$", "", gid)
            if base != gid and base not in desc:
                desc[base] = d
    return desc


def load_gene_annotation(gene_model_path, description_path=None) -> pd.DataFrame:
    """Load gene coordinates + descriptions into a single DataFrame."""
    genes = load_gene_model(gene_model_path)

    if description_path is not None:
        desc_dict = load_gene_descriptions(description_path)
        genes["Description"] = genes["Gene_ID"].map(
            lambda g: desc_dict.get(g, desc_dict.get(re.sub(r"(\.\d+)+$", "", g), ""))
        )
    else:
        genes["Description"] = ""

    return genes


# ============================================================
# 2. LD BLOCK ANNOTATION
# ============================================================

def _find_overlapping_genes(genes_chr, block_start, block_end):
    mask = (genes_chr["Start"].values <= block_end) & (genes_chr["End"].values >= block_start)
    return genes_chr.loc[mask].copy()


def _find_flanking_genes(genes_chr, block_start, block_end, n_flank=2, max_dist_bp=500_000):
    upstream = genes_chr[genes_chr["End"] < block_start].copy()
    upstream["dist_to_block"] = block_start - upstream["End"]
    upstream = upstream[upstream["dist_to_block"] <= max_dist_bp]
    upstream = upstream.nlargest(n_flank, "Start")

    downstream = genes_chr[genes_chr["Start"] > block_end].copy()
    downstream["dist_to_block"] = downstream["Start"] - block_end
    downstream = downstream[downstream["dist_to_block"] <= max_dist_bp]
    downstream = downstream.nsmallest(n_flank, "Start")

    return upstream, downstream


def annotate_ld_blocks(
    ld_blocks,
    genes,
    n_flank=2,
    max_flank_dist_bp=500_000,
):
    """
    Annotate LD blocks with overlapping + flanking genes.

    Parameters
    ----------
    ld_blocks : DataFrame with Chr, Start/Start (bp), End/End (bp)
    genes : DataFrame from load_gene_annotation()
    n_flank : how many flanking genes upstream/downstream
    max_flank_dist_bp : max distance to search for flanking genes

    Returns
    -------
    DataFrame with added columns:
        n_genes_overlapping, overlapping_genes, overlapping_descriptions,
        annotation_status, upstream_gene_1, upstream_dist_1, upstream_desc_1, ...
        downstream_gene_1, downstream_dist_1, downstream_desc_1, ...
    """
    if ld_blocks is None or ld_blocks.empty:
        return ld_blocks

    out = ld_blocks.copy()

    if "Start (bp)" in out.columns:
        out = out.rename(columns={"Start (bp)": "Start", "End (bp)": "End"})
    out["Chr"] = out["Chr"].astype(str).map(canon_chr)
    out["Start"] = out["Start"].astype(int)
    out["End"] = out["End"].astype(int)

    genes = genes.copy()
    genes["Chr"] = genes["Chr"].astype(str).map(canon_chr)
    genes_by_chr = {ch: g for ch, g in genes.groupby("Chr")}

    n_genes_list = []
    overlapping_genes_list = []
    overlapping_desc_list = []
    annotation_status_list = []

    flank_data = {}
    for direction in ("upstream", "downstream"):
        for i in range(n_flank):
            flank_data[f"{direction}_gene_{i+1}"] = []
            flank_data[f"{direction}_dist_{i+1}"] = []
            flank_data[f"{direction}_desc_{i+1}"] = []

    for _, row in out.iterrows():
        ch = str(row["Chr"])
        s, e = int(row["Start"]), int(row["End"])

        genes_chr = genes_by_chr.get(ch, pd.DataFrame())

        if genes_chr.empty:
            n_genes_list.append(0)
            overlapping_genes_list.append("")
            overlapping_desc_list.append("")
            annotation_status_list.append("no_genes_nearby")
            for direction in ("upstream", "downstream"):
                for i in range(n_flank):
                    flank_data[f"{direction}_gene_{i+1}"].append("")
                    flank_data[f"{direction}_dist_{i+1}"].append(np.nan)
                    flank_data[f"{direction}_desc_{i+1}"].append("")
            continue

        overlap = _find_overlapping_genes(genes_chr, s, e)

        n_genes_list.append(len(overlap))
        overlapping_genes_list.append(";".join(overlap["Gene_ID"].tolist()))
        overlapping_desc_list.append(
            " | ".join(
                f"{g}: {d}" if d else g
                for g, d in zip(overlap["Gene_ID"], overlap["Description"])
            )
        )

        upstream, downstream = _find_flanking_genes(
            genes_chr, s, e, n_flank=n_flank, max_dist_bp=max_flank_dist_bp,
        )

        if len(overlap) > 0:
            annotation_status_list.append("overlapping")
        elif len(upstream) > 0 or len(downstream) > 0:
            annotation_status_list.append("intergenic")
        else:
            annotation_status_list.append("no_genes_nearby")

        for direction, flank_df in [("upstream", upstream), ("downstream", downstream)]:
            if direction == "upstream":
                flank_sorted = flank_df.sort_values("Start", ascending=False).reset_index(drop=True)
            else:
                flank_sorted = flank_df.sort_values("Start", ascending=True).reset_index(drop=True)

            for i in range(n_flank):
                if i < len(flank_sorted):
                    r = flank_sorted.iloc[i]
                    flank_data[f"{direction}_gene_{i+1}"].append(str(r["Gene_ID"]))
                    flank_data[f"{direction}_dist_{i+1}"].append(int(r["dist_to_block"]))
                    flank_data[f"{direction}_desc_{i+1}"].append(str(r.get("Description", "")))
                else:
                    flank_data[f"{direction}_gene_{i+1}"].append("")
                    flank_data[f"{direction}_dist_{i+1}"].append(np.nan)
                    flank_data[f"{direction}_desc_{i+1}"].append("")

    out["n_genes_overlapping"] = n_genes_list
    out["overlapping_genes"] = overlapping_genes_list
    out["overlapping_descriptions"] = overlapping_desc_list
    out["annotation_status"] = annotation_status_list

    for k, v in flank_data.items():
        out[k] = v

    out = out.rename(columns={"Start": "Start (bp)", "End": "End (bp)"})
    return out


def consolidate_ld_block_table(ld_blocks, hap_gwas=None, annotated_blocks=None):
    """Merge LD blocks, gene annotation, and haplotype GWAS into one table.

    Parameters
    ----------
    ld_blocks : DataFrame
        Raw LD blocks (Chr, Start (bp), End (bp), Lead SNP, SNP_IDs).
    hap_gwas : DataFrame or None
        Haplotype GWAS results from ``run_haplotype_block_gwas``.
    annotated_blocks : DataFrame or None
        Gene-annotated LD blocks from ``annotate_ld_blocks``.

    Returns
    -------
    DataFrame combining all available information, or *None* if
    *ld_blocks* is None/empty.
    """
    if ld_blocks is None or (hasattr(ld_blocks, "empty") and ld_blocks.empty):
        return annotated_blocks  # may also be None

    base = annotated_blocks if (annotated_blocks is not None and not annotated_blocks.empty) else ld_blocks.copy()

    if hap_gwas is not None and not hap_gwas.empty:
        hm = hap_gwas.rename(columns={"Start": "Start (bp)", "End": "End (bp)"})
        keep = [c for c in [
            "Chr", "Start (bp)", "End (bp)",
            "PValue", "FDR_BH", "F_perm", "F_param",
            "n_haplotypes", "n_tested_haplotypes", "n_samples_block",
            "eta2", "n_permutations",
        ] if c in hm.columns]
        hsub = hm[keep].copy()
        hsub = hsub.rename(columns={
            "PValue": "Hap_PValue", "FDR_BH": "Hap_FDR_BH",
            "F_perm": "Hap_F_perm", "F_param": "Hap_F_param",
            "n_haplotypes": "Hap_n_haplotypes",
            "n_tested_haplotypes": "Hap_n_tested",
            "n_samples_block": "Hap_n_samples",
            "eta2": "Hap_eta2",
            "n_permutations": "Hap_n_perms",
        })
        for df in [base, hsub]:
            df["Chr"] = df["Chr"].astype(str)
            df["Start (bp)"] = pd.to_numeric(df["Start (bp)"], errors="coerce").astype("Int64")
            df["End (bp)"] = pd.to_numeric(df["End (bp)"], errors="coerce").astype("Int64")
        base = base.merge(hsub, on=["Chr", "Start (bp)", "End (bp)"], how="left")

    return base


def format_annotation_summary(annotated_blocks):
    """
    Compact paper-ready annotation summary.
    Shows overlapping genes, or flanking genes for intergenic blocks.
    """
    if annotated_blocks is None or annotated_blocks.empty:
        return pd.DataFrame()

    rows = []
    for _, r in annotated_blocks.iterrows():
        block_label = f"Chr{r['Chr']}:{int(r['Start (bp)']):,}-{int(r['End (bp)']):,}"
        block_kb = (int(r['End (bp)']) - int(r['Start (bp)'])) / 1000.0
        status = r.get("annotation_status", "unknown")

        if status == "overlapping":
            n_g = int(r["n_genes_overlapping"])
            rows.append({
                "Block": block_label,
                "Block_kb": round(block_kb, 1),
                "Status": f"{n_g} gene{'s' if n_g > 1 else ''} overlapping",
                "Genes": str(r["overlapping_genes"]),
                "Descriptions": str(r["overlapping_descriptions"]),
            })

        elif status == "intergenic":
            parts = []
            for direction, arrow in [("upstream", "\u2191"), ("downstream", "\u2193")]:
                gene = str(r.get(f"{direction}_gene_1", ""))
                dist = r.get(f"{direction}_dist_1", np.nan)
                desc = str(r.get(f"{direction}_desc_1", ""))
                if gene:
                    d_kb = f"{int(dist)/1000:.1f} kb" if pd.notna(dist) else "?"
                    parts.append(f"{arrow} {gene} ({d_kb} {direction})")

            descs = []
            for direction, arrow in [("upstream", "\u2191"), ("downstream", "\u2193")]:
                gene = str(r.get(f"{direction}_gene_1", ""))
                desc = str(r.get(f"{direction}_desc_1", ""))
                if gene and desc:
                    descs.append(f"{arrow} {gene}: {desc}")

            rows.append({
                "Block": block_label,
                "Block_kb": round(block_kb, 1),
                "Status": "Intergenic",
                "Genes": " ; ".join(parts) if parts else "\u2014",
                "Descriptions": " | ".join(descs) if descs else "\u2014",
            })
        else:
            rows.append({
                "Block": block_label,
                "Block_kb": round(block_kb, 1),
                "Status": "No genes within 500 kb",
                "Genes": "\u2014",
                "Descriptions": "\u2014",
            })

    return pd.DataFrame(rows)


# ============================================================
# 3. LD DECAY VISUALIZATION
# ============================================================

def compute_ld_decay_by_chromosome(
    chroms,
    positions,
    geno_imputed,
    max_snps_per_chr=2000,
    max_dist_kb=5000.0,
    n_bins=50,
    ld_thresholds=(0.1, 0.2),
    min_pair_n=20,
):
    """
    Compute LD decay per chromosome.

    Returns:
        decay_df : binned r2 vs distance per chromosome
        summary_df : decay distances per chromosome
    """
    from gwas.ld import pairwise_r2

    chroms = np.asarray([canon_chr(c) for c in chroms], dtype=object)
    positions = np.asarray(positions, int)
    geno = np.asarray(geno_imputed, float)

    all_bins = []
    summaries = []

    for ch in sorted(set(chroms.astype(str))):
        mask = chroms.astype(str) == ch
        idx = np.where(mask)[0]
        if len(idx) < 10:
            continue

        chr_pos = positions[idx]
        chr_geno = geno[:, idx]

        order = np.argsort(chr_pos)
        chr_pos = chr_pos[order]
        chr_geno = chr_geno[:, order]

        if len(chr_pos) > max_snps_per_chr:
            sub_idx = np.linspace(0, len(chr_pos) - 1, max_snps_per_chr, dtype=int)
            chr_pos = chr_pos[sub_idx]
            chr_geno = chr_geno[:, sub_idx]

        try:
            r2 = pairwise_r2(chr_geno, min_pair_n=min_pair_n)
        except RuntimeError:
            continue

        iu = np.triu_indices_from(r2, k=1)
        dist_bp = np.abs(chr_pos[iu[0]] - chr_pos[iu[1]])
        r2_vals = r2[iu]

        valid = np.isfinite(r2_vals) & (dist_bp > 0)
        if valid.sum() < 20:
            continue

        dist_kb = dist_bp[valid] / 1000.0
        r2_v = r2_vals[valid]

        cap = dist_kb <= max_dist_kb
        dist_kb, r2_v = dist_kb[cap], r2_v[cap]

        if len(dist_kb) < 20:
            continue

        bins = np.linspace(dist_kb.min(), dist_kb.max(), n_bins + 1)
        bin_idx = np.clip(np.digitize(dist_kb, bins) - 1, 0, n_bins - 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        chr_bins = []
        for b in range(n_bins):
            bm = bin_idx == b
            if bm.sum() < 5:
                continue
            entry = {
                "Chr": ch,
                "dist_kb": float(bin_centers[b]),
                "median_r2": float(np.median(r2_v[bm])),
                "mean_r2": float(np.mean(r2_v[bm])),
                "n_pairs": int(bm.sum()),
                "bin_index": int(b),
            }
            all_bins.append(entry)
            chr_bins.append(entry)

        summary = {"Chr": ch, "n_snps": int(len(chr_pos))}
        if chr_bins:
            chr_bin_df = pd.DataFrame(chr_bins)
            for thr in ld_thresholds:
                below = chr_bin_df[chr_bin_df["median_r2"] <= thr]
                summary[f"decay_kb_r2_{thr}"] = float(below["dist_kb"].iloc[0]) if not below.empty else np.nan
        summaries.append(summary)

    return pd.DataFrame(all_bins), pd.DataFrame(summaries)


def plot_ld_decay_matplotlib(
    decay_df,
    summary_df=None,
    ld_threshold_line=0.2,
    title="LD Decay by Chromosome",
    figsize=(10, 6),
):
    """Publication-grade LD decay plot. Returns (fig, ax)."""
    import matplotlib.pyplot as plt

    if decay_df is None or decay_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No LD decay data", transform=ax.transAxes, ha="center", va="center")
        return fig, ax

    from utils.pub_theme import PALETTE_CYCLE, SIG_LINE_COLOR, FIGSIZE

    fig, ax = plt.subplots(figsize=FIGSIZE["line"])

    chroms = sorted(decay_df["Chr"].unique(), key=lambda x: (0, int(x)) if x.isdigit() else (1, x))

    for i, ch in enumerate(chroms):
        sub = decay_df[decay_df["Chr"] == ch].sort_values("dist_kb")
        ax.plot(sub["dist_kb"], sub["median_r2"],
                color=PALETTE_CYCLE[i % len(PALETTE_CYCLE)],
                alpha=0.7, linewidth=1.5, label=f"Chr{ch}")

    ax.axhline(y=ld_threshold_line, color=SIG_LINE_COLOR, linestyle="--",
               linewidth=1.2, alpha=0.7, label=f"r\u00b2 = {ld_threshold_line}")

    genome_wide = decay_df.groupby("dist_kb")["median_r2"].median().reset_index()
    ax.plot(genome_wide["dist_kb"], genome_wide["median_r2"],
            color="black", linewidth=2.5, label="Genome-wide median", zorder=10)

    ax.set_xlabel("Physical distance (kb)")
    ax.set_ylabel("Median r\u00b2")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(left=0)
    ax.legend(ncol=2, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    return fig, ax
