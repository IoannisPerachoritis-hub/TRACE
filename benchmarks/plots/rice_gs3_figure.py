"""The rice grain length figure: GS3 in TRACE's run on the 3,000 Rice Genomes accessions.

Five panels in the style of the author's ggplot figures: 180 mm wide, 12, 11 and 10 pt text,
plain axes.

  (A) Manhattan plots of MLM and, when its ZIP is given, FarmCPU, with the Bonferroni line and the
      marked SNP (the MLM scan's top SNP, in GS3).
  (B) The GS3 window (GS3 with 500 kb on each side) with MLM -log10 p, the LD block that holds
      the marked SNP shaded and the SNP marked.
  (C) The LD heatmap of the window, with the block outlined. r2 comes from gwas.ld.pairwise_r2 on
      the tested samples, with the missing calls filled with the SNP mean as TRACE does, on at
      most 1,500 SNPs taken evenly by index (TRACE's MAX_REGION_SNPS rule), keeping the marked
      SNP and the block's end members.
  (D) The genes of the window from the gene model: the block's genes in black, GS3 highlighted.
  (E) Grain length by the raw genotype at the marked SNP, with n per group and the missing calls.
      No new test: the subtitle gives the run's MLM p for the site.

    python benchmarks/plots/rice_gs3_figure.py --mlm-zip MLM.zip --vcf VCF.gz --pheno PHENO.csv
        --gff locus.gff --out-dir DIR [--farmcpu-zip FARMCPU.zip]

Inputs: the results ZIP of the MLM run (its tables GWAS_<trait>.csv, LD_blocks_annotated_MLM.csv,
and QC_sample_heterozygosity.csv, which lists the tested samples), the results ZIP of the FarmCPU
run (its GWAS_FarmCPU_<trait>.csv), the VCF and the phenotype file the runs read, and the gene
model the runs used (GFF3). The script prints the sha256 of each input.
Output, in --out-dir: rice_gs3_figure.png (300 dpi), rice_gs3_figure.pdf (points and heatmap
rasterized) and rice_gs3_figure_summary.json (the numbers the figure shows).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from annotation import canon_chr  # noqa: E402
from gwas.ld import pairwise_r2  # noqa: E402
from gwas.plotting import compute_cumulative_positions  # noqa: E402

MM = 1 / 25.4                       # inches per mm
FIG_W = 180                         # mm
FS_TITLE, FS_AXIS, FS_TEXT = 12, 11, 10
BAND = "#D5ECF9"                    # the LD block's shading
MARK = "#D55E00"                    # the marked SNP and GS3
GENE_BOX = "#FFF200"                # the box of GS3's label
R2_COLOURS = ["#FFFFFF", "#FEE8C8", "#FDBB84", "#E34A33", "#7F0000"]
LINK_IN, LINK_OUT = "#7FA7C9", "#BFBFBF"     # heatmap links in the block, outside (grey75)
GENE_IN, GENE_OUT = "#1A1A1A", "#A6A6A6"     # genes in the block (grey10), outside (grey65)
VIOLIN = "#E1E6F2"
POINT = "#595959"                   # grey35
MANHATTAN = ("#3D3D3D", "#A3A3A3")  # alternate chromosomes
MAX_REGION_SNPS = 1500              # gwas/ld.py, find_ld_clusters_genomewide

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": FS_TEXT,
    "mathtext.default": "regular",
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "xtick.minor.width": 0.7, "ytick.minor.width": 0.7,
    "xtick.labelsize": FS_TEXT, "ytick.labelsize": FS_TEXT,
    "axes.labelsize": FS_AXIS,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ---------------------------------------------------------------- inputs

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def zip_table(zpath: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(zpath) as z:
        if member not in z.namelist():
            raise SystemExit(f"{zpath.name} has no member {member}")
        return pd.read_csv(io.BytesIO(z.read(member)))


def parse_window(text: str) -> tuple[str, int, int]:
    chrom, span = text.split(":")
    a, b = span.replace(",", "").split("-")
    return canon_chr(chrom), int(a), int(b)


def read_vcf_window(vcf: Path, samples: list[str], chrom: str, start: int, end: int,
                    keep: set[str], raw_snp: str):
    """The records of chrom:start-end whose ID is in ``keep``: the dosage (ALT copies, NaN for a
    missing call) of ``samples``, and for ``raw_snp`` its REF, ALT and two allele codes per
    sample (-1 for missing). A record with more than one ALT allele is skipped, as TRACE does."""
    opener = gzip.open if vcf.name.endswith(".gz") else open
    ids, pos, rows = [], [], []
    raw = None
    take = n_all = None
    seen = False
    skipped_multi = 0
    with opener(vcf, "rb") as f:
        for line in f:
            if line.startswith(b"##"):
                continue
            if line.startswith(b"#CHROM"):
                head = line.rstrip(b"\r\n").decode().split("\t")
                col = {s: i for i, s in enumerate(head[9:])}
                lost = [s for s in samples if s not in col]
                if lost:
                    raise SystemExit(f"{len(lost)} tested samples are not in the VCF, e.g. {lost[:3]}")
                take = np.array([col[s] for s in samples])
                n_all = len(head) - 9
                continue
            t1 = line.find(b"\t")
            if canon_chr(line[:t1].decode()) != chrom:
                if seen:
                    break
                continue
            seen = True
            t2 = line.find(b"\t", t1 + 1)
            p = int(line[t1 + 1:t2])
            if p < start:
                continue
            if p > end:
                break
            fields = line.rstrip(b"\r\n").split(b"\t", 9)
            sid = fields[2].decode()
            if sid not in keep:
                continue
            if b"," in fields[4]:
                skipped_multi += 1
                continue
            body = fields[9]
            if fields[8] == b"GT" and len(body) == 4 * n_all - 1:
                a = np.frombuffer(body + b"\t", dtype=np.uint8).reshape(n_all, 4)
                c1, c2 = a[:, 0].astype(np.int16) - 48, a[:, 2].astype(np.int16) - 48
            else:                                   # any other layout: field by field
                gts = [x.split(b":")[0].replace(b"|", b"/") for x in body.split(b"\t")]
                c1 = np.array([int(g[:1]) if g[:1].isdigit() else -2 for g in gts], dtype=np.int16)
                c2 = np.array([int(g[2:3]) if g[2:3].isdigit() else -2 for g in gts], dtype=np.int16)
            c1, c2 = c1[take], c2[take]
            miss = (c1 < 0) | (c2 < 0)
            c1 = np.where(miss, -1, c1)
            c2 = np.where(miss, -1, c2)
            d = (c1 > 0).astype(np.float32) + (c2 > 0).astype(np.float32)
            d[miss] = np.nan
            ids.append(sid)
            pos.append(p)
            rows.append(d)
            if sid == raw_snp:
                raw = {"ref": fields[3].decode(), "alt": fields[4].decode(), "c1": c1, "c2": c2}
    if not rows:
        raise SystemExit(f"no record of {chrom}:{start}-{end} in the VCF")
    return np.array(ids), np.array(pos), np.vstack(rows).T, raw, skipped_multi


def read_gff_genes(gff: Path, chrom: str, start: int, end: int) -> pd.DataFrame:
    out = []
    with open(gff, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                continue
            t = line.rstrip("\n").split("\t")
            if len(t) < 9 or t[2] != "gene" or canon_chr(t[0]) != chrom:
                continue
            s, e = int(t[3]), int(t[4])
            if e < start or s > end:
                continue
            attrs = dict(kv.split("=", 1) for kv in t[8].split(";") if "=" in kv)
            out.append({"id": attrs.get("ID", attrs.get("Name", "")), "start": s, "end": e})
    return pd.DataFrame(out, columns=["id", "start", "end"])


def thin_by_index(m: int, must: np.ndarray, k: int = MAX_REGION_SNPS) -> np.ndarray:
    """At most k of m SNPs, evenly by index (np.linspace, as TRACE thins an LD window), with
    the indices in ``must`` always kept."""
    if m <= k:
        return np.arange(m)
    kk = k
    while True:
        idx = np.union1d(np.linspace(0, m - 1, kk, dtype=int), must)
        if idx.size <= k:
            return idx
        kk -= idx.size - k


# ---------------------------------------------------------------- drawing helpers

def classic(ax, left=True, bottom=True):
    """theme_classic: the left and bottom axis lines only, ticks out, black text."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(direction="out", colors="black", length=3.5)


def panel_title(ax, text, dx_mm=-12.0, dy_mm=2.5):
    """A bold panel title above the axes' top left corner, offset in mm."""
    ax.annotate(text, xy=(0, 1), xycoords="axes fraction",
                xytext=(dx_mm * 72 / 25.4, dy_mm * 72 / 25.4), textcoords="offset points",
                fontsize=FS_TITLE, fontweight="bold", ha="left", va="bottom")


def bonferroni_line(ax, y, x_text, ha="left"):
    ax.axhline(y, ls=(0, (4, 3)), lw=0.8, color="#4D4D4D", zorder=2)
    ax.text(x_text, y, f"Bonferroni ({y:.2f})", fontsize=FS_TEXT - 1.5, color="#4D4D4D",
            ha=ha, va="bottom")


def fmt_p(p: float) -> str:
    m, e = f"{p:.2e}".split("e")
    return rf"{float(m):.2f} $\times$ 10$^{{{int(e)}}}$"


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mlm-zip", required=True, type=Path)
    ap.add_argument("--farmcpu-zip", type=Path)
    ap.add_argument("--vcf", required=True, type=Path)
    ap.add_argument("--pheno", required=True, type=Path)
    ap.add_argument("--gff", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--trait", default="GRLT")
    ap.add_argument("--trait-label", default="Grain length (mm)")
    ap.add_argument("--snp", default="3:16733441", help="the marked SNP")
    ap.add_argument("--gene", default="Os03g0407400", help="the highlighted gene's ID")
    ap.add_argument("--gene-label", default="GS3")
    ap.add_argument("--window", default="3:16229501-17235109", help="chrom:start-end, inclusive")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args(argv)

    inputs = {"mlm_zip": a.mlm_zip, "farmcpu_zip": a.farmcpu_zip, "vcf": a.vcf,
              "pheno": a.pheno, "gff": a.gff}
    info = {}
    for k, p in inputs.items():
        if p is None:
            continue
        info[k] = {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)}
        print(f"input {k}: {p} ({info[k]['bytes']:,} bytes) sha256 {info[k]['sha256']}")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"inputs": info}

    # --- the run's tables
    mlm = zip_table(a.mlm_zip, f"tables/GWAS_{a.trait}.csv")[["SNP", "Chr", "Pos", "PValue"]].copy()
    mlm["Chr"] = mlm["Chr"].map(canon_chr)
    n_snps = len(mlm)
    bonf = -np.log10(a.alpha / n_snps)
    blocks = zip_table(a.mlm_zip, "tables/LD_blocks_annotated_MLM.csv")
    tested = zip_table(a.mlm_zip, "tables/QC_sample_heterozygosity.csv")["sample"].astype(str).tolist()
    farm = None
    if a.farmcpu_zip is not None:
        farm = zip_table(a.farmcpu_zip, f"tables/GWAS_FarmCPU_{a.trait}.csv")
        farm = farm[["SNP", "Chr", "Pos", "PValue"]].copy()
        farm["Chr"] = farm["Chr"].map(canon_chr)

    mark = mlm.loc[mlm["SNP"] == a.snp].iloc[0]
    mark_p = float(mark["PValue"])
    mark_rank = int((mlm["PValue"] < mark_p).sum()) + 1
    has = blocks["SNP_IDs"].astype(str).str.split(",").map(lambda xs: a.snp in [x.strip() for x in xs])
    if has.sum() != 1:
        raise SystemExit(f"{a.snp} is in {int(has.sum())} LD blocks, not one")
    blk = blocks.loc[has].iloc[0]
    b_start, b_end = int(blk["Start (bp)"]), int(blk["End (bp)"])
    pos_of = dict(zip(mlm["SNP"], mlm["Pos"]))
    members = sorted((x.strip() for x in str(blk["SNP_IDs"]).split(",")), key=lambda s: pos_of[s])
    ends = (members[0], members[-1])
    block_genes = [g for g in str(blk["overlapping_genes"]).split(";") if g and g != "nan"]
    print(f"SNPs tested {n_snps:,}; Bonferroni -log10 {bonf:.4f}; tested samples {len(tested):,}")
    print(f"marked SNP {a.snp}: MLM p {mark_p:.4g}, rank {mark_rank}")
    print(f"its LD block: chr{canon_chr(blk['Chr'])}:{b_start:,}-{b_end:,}, {len(members)} SNPs "
          f"(end members {ends[0]}, {ends[1]}), lead {blk['lead_snp']}, {len(block_genes)} genes")
    summary.update({"n_snps": n_snps, "bonferroni_mlog10": bonf, "n_tested_samples": len(tested),
                    "marked_snp": a.snp, "marked_mlm_p": mark_p, "marked_mlm_rank": mark_rank,
                    "block": {"chr": canon_chr(blk["Chr"]), "start": b_start, "end": b_end,
                              "n_members": len(members), "lead_snp": str(blk["lead_snp"]),
                              "genes": block_genes}})

    # --- the window: genotypes from the VCF, for the SNPs the run tested
    chrom, w0, w1 = parse_window(a.window)
    win = mlm[(mlm["Chr"] == chrom) & (mlm["Pos"] >= w0) & (mlm["Pos"] <= w1)].sort_values("Pos")
    ids, pos, G, raw, multi = read_vcf_window(a.vcf, tested, chrom, w0, w1, set(win["SNP"]), a.snp)
    order = np.argsort(pos, kind="stable")
    ids, pos, G = ids[order], pos[order], G[:, order]
    if len(ids) != len(win) or raw is None:
        raise SystemExit(f"the VCF gave {len(ids)} of the window's {len(win)} tested SNPs")
    means = np.nanmean(G, axis=0)
    miss_cells = int(np.isnan(G).sum())
    G = np.where(np.isnan(G), means[None, :], G)          # the SNP mean, as TRACE fills
    must = np.flatnonzero(np.isin(ids, [a.snp, *ends]))
    if must.size != 3:
        raise SystemExit("the marked SNP or a block end member is not in the window")
    keep = thin_by_index(len(ids), must)
    r2 = pairwise_r2(G[:, keep])
    s_ids, s_pos = ids[keep], pos[keep]
    n_show = len(keep)
    print(f"window {chrom}:{w0:,}-{w1:,}: {len(ids):,} tested SNPs, {n_show:,} in the heatmap; "
          f"{miss_cells:,} missing calls filled; multi-allelic records skipped {multi}")
    summary["window"] = {"chr": chrom, "start": w0, "end": w1, "n_snps": int(len(ids)),
                         "n_heatmap": int(n_show), "missing_calls_filled": miss_cells,
                         "heatmap_kept": [str(x) for x in ids[must]]}

    genes = read_gff_genes(a.gff, chrom, w0, w1)
    genes["in_block"] = genes["id"].isin(block_genes)
    span_only = genes[(genes["end"] >= b_start) & (genes["start"] <= b_end) & ~genes["in_block"]]
    print(f"genes in the window {len(genes)}, of the block {int(genes['in_block'].sum())}; "
          f"overlapping the block's span but not listed: {len(span_only)}")
    summary["genes"] = {"in_window": int(len(genes)), "in_block": int(genes["in_block"].sum()),
                        "span_not_listed": span_only["id"].tolist(),
                        "highlighted": a.gene, "highlighted_in_block": a.gene in block_genes}

    # --- the raw genotypes at the marked SNP, with the phenotype
    ph = pd.read_csv(a.pheno)
    ph = ph.set_index(ph.columns[0])[a.trait]
    y = ph.reindex(tested).to_numpy(dtype=float)
    ref, alt = raw["ref"], raw["alt"]
    c1, c2 = raw["c1"], raw["c2"]
    miss = c1 < 0
    n_alt = (c1 > 0).astype(int) + (c2 > 0).astype(int)
    labels = [f"{ref}/{ref}", f"{ref}/{alt}", f"{alt}/{alt}"]
    groups = []
    for k, lab in enumerate(labels):
        sel = (~miss) & (n_alt == k) & np.isfinite(y)
        if sel.any():
            groups.append((lab, y[sel]))
    n_missing = int(miss.sum())
    print(f"{a.snp} REF {ref} ALT {alt}: " + ", ".join(f"{g} n {len(v)}" for g, v in groups)
          + f", missing {n_missing}")
    summary["genotypes"] = {"ref": ref, "alt": alt, "missing": n_missing,
                            "groups": {g: {"n": int(len(v)), "median": float(np.median(v)),
                                           "mean": float(np.mean(v))} for g, v in groups}}

    # ---------------------------------------------------------------- layout (mm)
    n_man = 2 if farm is not None else 1
    L, R_ = 15.0, 3.0
    W_A = FIG_W - L - R_
    W_L = 100.0                                   # the window panels' width
    X_E = L + W_L + 18.0
    W_E = FIG_W - R_ - X_E
    H_MAN, GAP_MAN = 27.0, 4.0
    top = 7.0
    y_man = [top + i * (H_MAN + GAP_MAN) for i in range(n_man)]
    y_b = y_man[-1] + H_MAN + 20.0
    H_B = 40.0
    TOP_U = 0.05 * n_show                          # the link band, in SNP steps
    H_C = (n_show / 2 + TOP_U) * (W_L / n_show)     # square cells
    y_c = y_b + H_B + 7.0
    H_D = 13.0
    y_d = y_c + H_C + 6.0
    FIG_H = y_d + H_D + 11.0
    y_e, H_E = y_b + 22.0, 52.0                     # (E), with its title and notes above it
    fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM))

    def axes(x, y_top, w, h):
        return fig.add_axes([x / FIG_W, 1 - (y_top + h) / FIG_H, w / FIG_W, h / FIG_H])

    # ---------------------------------------------------------------- (A) Manhattans
    man = [("MLM", mlm)] + ([("FarmCPU", farm)] if farm is not None else [])
    summary["manhattan"] = {}
    for i, (name, df) in enumerate(man):
        ax = axes(L, y_man[i], W_A, H_MAN)
        d, ticks, tick_labels = compute_cumulative_positions(df, "Chr", "Pos")
        d["mlog10p"] = -np.log10(d["PValue"].clip(lower=1e-300))
        chr_rank = {c: k for k, c in enumerate(tick_labels)}
        col = np.array(MANHATTAN)[d["Chr"].map(lambda c: chr_rank.get(str(c), 0) % 2).to_numpy()]
        ax.scatter(d["CumPos"], d["mlog10p"], s=2.0, c=col, linewidths=0, rasterized=True, zorder=1)
        top_y = float(d["mlog10p"].max())
        ax.set_ylim(0, top_y * 1.12)
        ax.set_xlim(d["CumPos"].min() - 4e6, d["CumPos"].max() + 4e6)
        bonferroni_line(ax, bonf, d["CumPos"].max() + 4e6, ha="right")
        m = d.loc[d["SNP"] == a.snp].iloc[0]
        ax.scatter([m["CumPos"]], [m["mlog10p"]], marker="D", s=34, c=MARK, edgecolors="black",
                   linewidths=0.6, zorder=4)
        ax.annotate(a.gene_label, xy=(m["CumPos"], m["mlog10p"]), xytext=(7, 0), textcoords="offset points",
                    fontsize=FS_TEXT - 1, fontweight="bold", fontstyle="italic", va="center",
                    bbox=dict(boxstyle="square,pad=0.25", fc=GENE_BOX, ec="none"), zorder=5)
        ax.text(0.005, 0.97, name, transform=ax.transAxes, fontsize=FS_TEXT, fontweight="bold",
                ha="left", va="top")
        ax.set_ylabel(r"$-\log_{10}\,p$")
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        classic(ax)
        if i < n_man - 1:
            ax.tick_params(labelbottom=False)
        if i == 0:
            trait_name = a.trait_label.split(" (")[0].lower()
            panel_title(ax, f"(A) Genome scans of {trait_name}: {len(tested):,} accessions, "
                            f"{n_snps:,} SNPs")
        if i == n_man - 1:
            ax.set_xlabel("Chromosome")
        rank = int((df["PValue"] < float(df.loc[df["SNP"] == a.snp, "PValue"].iloc[0])).sum()) + 1
        summary["manhattan"][name] = {"n_snps": int(len(df)), "max_mlog10p": top_y,
                                      "marked_p": float(m["PValue"]), "marked_rank": rank}

    xlim = (w0 / 1e6, w1 / 1e6)
    bs, be = b_start / 1e6, b_end / 1e6

    # ---------------------------------------------------------------- (B) the window
    ax = axes(L, y_b, W_L, H_B)
    wm = win.assign(Mb=win["Pos"] / 1e6, mlog10p=-np.log10(win["PValue"].clip(lower=1e-300)))
    ymax = float(wm["mlog10p"].max()) * 1.15
    ax.add_patch(Rectangle((bs, 0), be - bs, ymax, fc=BAND, ec="none", zorder=0))
    ax.text((bs + be) / 2, ymax * 0.985, "LD block", fontsize=FS_TEXT - 0.5, color="#404040",
            ha="center", va="top")
    bonferroni_line(ax, bonf, xlim[0] + 0.006)
    rest = wm[wm["SNP"] != a.snp]
    ax.scatter(rest["Mb"], rest["mlog10p"], s=3, c=POINT, linewidths=0, rasterized=True, zorder=2)
    mk = wm[wm["SNP"] == a.snp].iloc[0]
    ax.scatter([mk["Mb"]], [mk["mlog10p"]], marker="D", s=40, c=MARK, edgecolors="black",
               linewidths=0.6, zorder=4)
    ax.annotate(a.snp, xy=(mk["Mb"], mk["mlog10p"]), xytext=(-6, 0), textcoords="offset points",
                fontsize=FS_TEXT - 1, color="#333333", ha="right", va="center")
    ax.set_xlim(*xlim)
    ax.set_ylim(0, ymax)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_ylabel(r"$-\log_{10}\,p$ (MLM)")
    ax.yaxis.set_label_coords(-0.09, 0.5)
    classic(ax)
    ax.tick_params(labelbottom=False)
    panel_title(ax, f"(B) The {a.gene_label} window, chr{chrom}:{w0:,}-{w1:,}")

    # ---------------------------------------------------------------- (C) LD heatmap
    ax = axes(L, y_c, W_L, H_C)
    n = n_show
    wd = (xlim[1] - xlim[0]) / n
    xi = xlim[0] + (np.arange(n) + 0.5) * wd
    aa, bb = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    X = xlim[0] + wd * (aa + bb) / 2.0
    Y = (aa - bb) / 2.0
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    C = np.ma.masked_where((jj <= ii) | ~np.isfinite(r2), r2)
    cmap = LinearSegmentedColormap.from_list("pairwise_r2", R2_COLOURS)
    cmap.set_bad((1, 1, 1, 0))
    mesh = ax.pcolormesh(X, Y, C, cmap=cmap, vmin=0, vmax=1, shading="flat", rasterized=True,
                         linewidth=0, antialiased=False)
    ib = np.flatnonzero((s_pos >= b_start) & (s_pos <= b_end))
    i0, i1 = int(ib.min()), int(ib.max())
    segs = [[(s_pos[k] / 1e6, TOP_U), (xi[k], 0.0)] for k in range(n)]
    inb = (s_pos >= b_start) & (s_pos <= b_end)
    ax.add_collection(LineCollection([s for s, f in zip(segs, inb) if not f], colors=LINK_OUT,
                                     linewidths=0.15, zorder=1))
    ax.add_collection(LineCollection([s for s, f in zip(segs, inb) if f], colors=LINK_IN,
                                     linewidths=0.15, zorder=1))
    ax.add_patch(Polygon([(xi[i0] - wd / 2, 0), (xi[i1] + wd / 2, 0),
                          ((xi[i0] + xi[i1]) / 2, -(i1 - i0 + 1) / 2)],
                         closed=True, fill=False, ec="black", lw=0.8, zorder=3))
    ax.set_xlim(*xlim)
    ax.set_ylim(-n / 2, TOP_U)
    ax.set_axis_off()
    ax.text(-0.09, 0.5, r"LD ($\mathit{r}^2$)", transform=ax.transAxes, rotation=90,
            fontsize=FS_AXIS, ha="center", va="center")
    # the key in the empty bottom right corner, outside the triangle
    cax = ax.inset_axes([0.71, 0.08, 0.26, 0.035])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=FS_TEXT - 1, length=2, width=0.6)
    cax.set_title(r"pairwise $\mathit{r}^2$", fontsize=FS_TEXT, pad=3)
    panel_title(ax, "(C) Pairwise LD in the window", dy_mm=0.5)
    heat_note = (f"Heatmap (C): {n:,} of the window's\n{len(ids):,} SNPs, evenly by index,\n"
                 f"keeping {a.snp} and the\nblock's end SNPs. Missing calls\n"
                 f"filled with the SNP mean, on the\n{len(tested):,} tested accessions.")
    fig.text((X_E - 0.30 * W_E) / FIG_W, 1 - (y_e + H_E + 14.0) / FIG_H, heat_note,
             fontsize=FS_TEXT - 1.5, color="#333333", ha="left", va="top")

    # ---------------------------------------------------------------- (D) genes
    ax = axes(L, y_d, W_L, H_D)
    ax.add_patch(Rectangle((bs, -0.75), be - bs, 3.35, fc=BAND, ec="none", zorder=0))
    ax.axhline(0, color="#8C8C8C", lw=0.4, zorder=1)
    g = genes.assign(s=genes["start"].clip(lower=w0) / 1e6, e=genes["end"].clip(upper=w1) / 1e6)
    g["e"] = np.maximum(g["e"], g["s"] + 0.004)
    for _, r in g[g["id"] != a.gene].iterrows():
        ax.add_patch(Rectangle((r["s"], -0.45), r["e"] - r["s"], 0.9,
                               fc=GENE_IN if r["in_block"] else GENE_OUT, ec="none", zorder=2))
    hl = g[g["id"] == a.gene]
    for _, r in hl.iterrows():
        mid = (r["s"] + r["e"]) / 2
        ax.add_patch(Rectangle((r["s"], -0.6), r["e"] - r["s"], 1.2, fc=MARK, ec="none", zorder=3))
        ax.plot([mid, mid], [0.6, 1.05], color="#404040", lw=0.4, zorder=3)
        ax.text(mid, 1.05, a.gene_label, fontsize=FS_TEXT - 1, fontweight="bold", fontstyle="italic",
                ha="center", va="bottom", zorder=4,
                bbox=dict(boxstyle="square,pad=0.25", fc=GENE_BOX, ec="none"))
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.75, 2.6)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_yticks([])
    ax.set_ylabel("Genes", rotation=90)
    ax.yaxis.set_label_coords(-0.09, 0.5)
    ax.set_xlabel(f"Chromosome {chrom} position (Mb)")
    classic(ax, left=False)
    panel_title(ax, "(D) Genes (RAP-DB)", dy_mm=0.5)

    # ---------------------------------------------------------------- (E) phenotype by genotype
    ax = axes(X_E, y_e, W_E, H_E)
    xs = np.arange(len(groups))
    data = [v for _, v in groups]
    big = [k for k, v in enumerate(data) if len(v) >= 3]
    if big:
        vp = ax.violinplot([data[k] for k in big], positions=xs[big], widths=0.85,
                           showextrema=False, showmedians=False)
        for body in vp["bodies"]:
            body.set_facecolor(VIOLIN)
            body.set_edgecolor("none")
            body.set_alpha(1.0)
        ax.boxplot([data[k] for k in big], positions=xs[big], widths=0.28, patch_artist=True,
                   showfliers=False,
                   boxprops=dict(facecolor="white", edgecolor="#262626", linewidth=0.8),
                   whiskerprops=dict(color="#262626", linewidth=0.8),
                   capprops=dict(color="#262626", linewidth=0.8),
                   medianprops=dict(color="black", linewidth=1.6), zorder=3)
    rng = np.random.default_rng(1)
    for k, v in enumerate(data):
        ax.scatter(k + rng.uniform(-0.09, 0.09, size=len(v)), v, s=3.0, c="#4D4D4D", alpha=0.55,
                   linewidths=0, rasterized=True, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{lab}\nn = {len(v):,}" for lab, v in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel(a.trait_label)
    classic(ax)
    ax.text(-0.30, 1.0 + 18.0 / H_E, f"(E) {a.trait_label.split(' (')[0]}\nat {a.snp}",
            transform=ax.transAxes, fontsize=FS_TITLE, fontweight="bold", ha="left", va="bottom")
    ax.text(-0.30, 1.0 + 1.5 / H_E, "the top SNP of the MLM scan\n(not the block's lead SNP)\n"
            f"MLM p = {fmt_p(mark_p)}\nmissing calls: {n_missing:,}",
            transform=ax.transAxes, fontsize=FS_TEXT - 1, ha="left", va="bottom")

    png = a.out_dir / "rice_gs3_figure.png"
    pdf = a.out_dir / "rice_gs3_figure.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf, dpi=300)
    plt.close(fig)
    summary["figure"] = {"width_mm": FIG_W, "height_mm": round(FIG_H, 1)}
    (a.out_dir / "rice_gs3_figure_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"wrote {png}, {pdf} ({FIG_W} x {FIG_H:.1f} mm) and rice_gs3_figure_summary.json")


if __name__ == "__main__":
    main()
