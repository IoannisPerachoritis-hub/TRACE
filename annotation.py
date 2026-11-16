"""Gene annotation and LD decay for TRACE.

Reads Sol_genes.csv and ITAG4.0 / SL3.1 description files,
annotates LD blocks with gene content, and computes LD decay curves.
"""

import gzip
import io
import numpy as np
import pandas as pd
import re

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote



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


# ============================================================
# Isolated-SNP evidence vocabulary (T-44 / shared with T-23)
# ------------------------------------------------------------
# Additive constants only — no function body in this module changes. They give the
# isolated-SNP rescue and the significant-SNP table one shared, closed vocabulary
# for the evidence columns, so "distance_only" means the same thing on both tracks.
INTERVAL_EVIDENCE_CLASSES = ("ld_supported", "distance_only")   # class of evidence delimiting an interval
LOCALIZATION_SUPPORT = ("block_member_snps", "flanking_markers_only")  # how the interval bounds were set
GENE_RELATIONS = ("overlapping",)  # gene ↔ interval relation: a gene counts when it overlaps


# ============================================================
# What the gene-file uploaders accept: the "?" of every one (the Gene Annotation tab and the
# GWAS page's One-Click overrides). The texts say what load_gene_model and
# load_gene_descriptions do; a change to a loader rule changes its text too
# (tests/test_gene_files.py runs every example in them).
# ============================================================
_GENE_DOC = "https://github.com/IoannisPerachoritis-hub/TRACE/blob/main/docs/gene_model_upload.md"

GENE_FILE_HELP = f"""One row per gene, with a header row. Comma-, semicolon- or tab-separated plain text (UTF-8; Excel's "CSV UTF-8" is fine), not compressed.

Columns (names are not case-sensitive; other columns are ignored):

- chromosome: `Chr`, `Chrom` or `Chromosome`
- start: `Start`, `Start_pos` or `Begin`
- end: `End`, `End_pos` or `Stop`
- gene ID: `Gene_ID`, `Gene`, `Name` or `Gene_name`
- optional: `Strand`

Use one column for each. Positions are in base pairs, in the same genome assembly as your VCF. Rows whose start or end is not a number are skipped.

Chromosome names are matched after common prefixes and leading zeros are removed: `chr01`, `Chr1`, `SL2.50ch01` and `1` are all chromosome 1.

Example (a row of TRACE's SL3.1 table):

```
Chr,Start,End,Strand,Gene_ID
1,30726,43941,-,Solyc01g005020.3
```

A GFF3 file is read directly too, plain or gzip-compressed, recognised by its content. Its `gene` rows give the chromosome (column 1), start and end (columns 4 and 5), strand (column 7) and gene ID (the `ID`, without a leading `gene:` or `gene-`); the description is the first of `description`, `Note` and `product`. A descriptions file overrides those descriptions for the genes it lists ([more]({_GENE_DOC}#gff3-files)). Example (an Ensembl gene row; the gaps are TABs):

```
1\tITAG\tgene\t30726\t43941\t.\t-\t.\tID=gene:Solyc01g005020.3;description=Salt Overly Sensitive 1
```
"""

DESCRIPTIONS_FILE_HELP = """Optional. One gene per line: the gene ID, a TAB, then the description. No header is needed; lines starting with `#` are ignored. Lines without a TAB are skipped, so a comma-separated file adds no descriptions.

The IDs must match the gene file's. Version endings are ignored: `Solyc01g005020.3` matches `Solyc01g005020`.

Example (from TRACE's SL3.1 descriptions): `Solyc01g005020.3`, a TAB, then `Salt Overly Sensitive 1`.
"""

ASSEMBLY_NOTE = ("Markers and gene models must come from the same genome assembly; otherwise "
                 "genes are matched at the wrong positions.")

_FIELD_NAMES = {"Chr": "chromosome", "Start": "start", "End": "end", "Strand": "strand",
                "Gene_ID": "gene ID"}


_GFF3_DESCRIPTION_KEYS = ("description", "Note", "product")


def _gene_file_bytes(path):
    """(the gene file's bytes, gunzipped when they start with gzip's magic bytes 1f 8b
    whatever the file is called; whether they were gzip). ``path`` may also be an open
    file."""
    if hasattr(path, "read"):
        raw = path.read()
        raw = raw.encode("utf-8") if isinstance(raw, str) else raw
    else:
        with open(path, "rb") as fh:
            raw = fh.read()
    if raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw), True
    return raw, False


def _is_gff3(text):
    """A GFF3 by its content: a ``##gff-version`` line among the leading comment lines,
    or a first data line of 9 tab-separated columns whose ninth holds key=value pairs."""
    for line in io.StringIO(text):            # lazily: only the leading lines are read
        line = line.rstrip("\r\n")
        if not line.strip():
            continue
        if line.startswith("#"):
            if line.lower().startswith("##gff-version"):
                return True
            continue
        cols = line.split("\t")
        return len(cols) == 9 and re.search(r"(?:^|;)\s*[^=;\s]+=", cols[8]) is not None
    return False


def _gff3_attributes(field):
    """A GFF3 column-9 field as {key: value}, keys and values percent-decoded."""
    out = {}
    for part in field.strip().split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        out.setdefault(unquote(key.strip()), unquote(value.strip()))
    return out


def _read_gff3(text):
    """A GFF3's genes: (the five-column gene table, {Gene_ID: description}).

    Rows of type ``gene``: Chr = canon_chr of column 1, Start and End = columns 4 and 5,
    Strand = column 7, Gene_ID = the ID without a leading ``gene:`` or ``gene-``, and the
    description the first of description, Note and product. A gene row without an ID is
    skipped. With no gene row the error lists the feature types found."""
    rows, descriptions, types = [], {}, {}
    for line in text.splitlines():
        if line.startswith("##FASTA"):
            break
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) != 9:
            continue
        types[cols[2]] = types.get(cols[2], 0) + 1
        if cols[2] != "gene":
            continue
        attrs = _gff3_attributes(cols[8])
        gene_id = re.sub(r"^gene[:\-]", "", attrs.get("ID", "").strip())
        if not gene_id:
            continue
        rows.append((cols[0], cols[3], cols[4], cols[6].strip() or ".", gene_id))
        desc = next((attrs[k] for k in _GFF3_DESCRIPTION_KEYS if attrs.get(k, "").strip()), "")
        descriptions.setdefault(gene_id, desc.strip())
    if not rows:
        found = ", ".join(f"{t} ({n})" for t, n in sorted(types.items())) or "none"
        raise ValueError("This GFF3 has no gene with an ID (rows of type 'gene'); its "
                         f"feature types are: {found}.")
    return (_finish_gene_table(pd.DataFrame(rows, columns=["Chr", "Start", "End", "Strand",
                                                           "Gene_ID"])),
            descriptions)


def _load_gene_model_and_descriptions(path):
    """A gene file's five-column gene table and, for a GFF3, its descriptions ({} for a
    table). The file is recognised by its content, not its name: gzip by its first two
    bytes, then GFF3 or a table."""
    raw, was_gzip = _gene_file_bytes(path)
    text = raw.decode("utf-8-sig", errors="replace")
    if _is_gff3(text):
        return _read_gff3(text)
    # a plain table on disk is read from its path, as before; a gzipped or already-read
    # one from its bytes
    return _read_gene_table(io.BytesIO(raw) if was_gzip or hasattr(path, "read") else path), {}


def load_gene_model(path) -> pd.DataFrame:
    """
    Load a gene file: a table (Sol_genes.csv or similar gene coordinate file) or a GFF3,
    plain or gzip, recognised by its content.
    Expected table columns: CHROM, START, END, STRAND, GENE
    Returns DataFrame with: Chr, Start, End, Strand, Gene_ID
    """
    return _load_gene_model_and_descriptions(path)[0]


def _read_gene_table(path):
    """A gene table: comma-, semicolon- or tab-separated, with a header row."""
    # encoding="utf-8-sig" — UTF-8, with the byte-order mark Excel's "CSV UTF-8" writes
    # removed (the default cp1252 on Windows would fail on non-ASCII text)
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
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

    # one column per field: two (e.g. Gene_ID and Name) would both become Gene_ID and the
    # load would fail later with an unrelated error
    by_field = {}
    for src, dst in col_map.items():
        by_field.setdefault(dst, []).append(src)
    for field, srcs in by_field.items():
        if len(srcs) > 1:
            raise ValueError(f"Use one column for the {_FIELD_NAMES[field]}; this file has "
                             f"{' and '.join(srcs)}.")

    df = df.rename(columns=col_map)

    required = {"Chr", "Start", "End", "Gene_ID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Gene model file missing columns: {missing}. Found: {list(df.columns)}")

    return _finish_gene_table(df)


def _finish_gene_table(df):
    """Canonical chromosome names and integer positions (rows without them dropped), a
    strand ('.' when absent), sorted by chromosome number, then start."""
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


def chromosomes_without_genes(genotype_chroms, genes):
    """The chromosomes of the genotype data that have no gene in ``genes`` (canon_chr
    names, numbers first): a non-empty list usually means the chromosome names of the
    gene file and the VCF do not match."""
    have = (set(genes["Chr"].astype(str).map(canon_chr))
            if genes is not None and len(genes) else set())
    chroms = {canon_chr(c) for c in pd.unique(np.asarray(genotype_chroms, dtype=object))}
    return sorted((c for c in chroms if c not in have),
                  key=lambda c: (0, int(c), "") if c.isdigit() else (1, 0, c))


def load_gene_annotation(gene_model_path, description_path=None) -> pd.DataFrame:
    """Load gene coordinates + descriptions into a single DataFrame. A GFF3's own
    descriptions are used; a descriptions file overrides them for the genes it lists."""
    genes, gff_desc = _load_gene_model_and_descriptions(gene_model_path)

    if description_path is not None:
        desc_dict = load_gene_descriptions(description_path)
        genes["Description"] = genes["Gene_ID"].map(
            lambda g: desc_dict.get(g, desc_dict.get(re.sub(r"(\.\d+)+$", "", g),
                                                     gff_desc.get(g, "")))
        )
    elif gff_desc:
        genes["Description"] = genes["Gene_ID"].map(lambda g: gff_desc.get(g, ""))
    else:
        genes["Description"] = ""

    return genes


# ============================================================
# 1b. BUNDLED GENE-MODEL RESOLUTION
# ============================================================
# One resolver, read once per run, consumed by the isolated-SNP rescue, the
# block annotation AND the run manifest. Keeping them on one answer is what
# makes it structurally impossible for run_metadata.json to name a gene model
# that annotation never loaded.

# Where the wheel ships the gene models (see pyproject.toml packages.find).
# Module-level so a test can point it at an empty directory and exercise the
# not-found path without touching the real install.
_BUNDLED_DATA_DIR = Path(__file__).resolve().parent / "data"

# species -> genome build -> (gene model filename, gene description filename)
_BUNDLED_GENE_FILES = {
    "tomato": {
        "SL3": ("Sol_genes_SL3.csv", "SL3.1_descriptions.txt"),
        "SL4": ("Sol_genes.csv", "ITAG4.0_annotation.txt"),
    },
}


@dataclass(frozen=True)
class GeneModelResolution:
    """What a run resolved its gene model to, and whether it is actually there.

    Attributes
    ----------
    gene_model : Path or None
        The gene table to load; None when nothing usable was found.
    descriptions : Path or None
        Matching description file, or None when there is none on disk.
    searched : Path or None
        The path that was looked for. None when the species has no bundled
        default and no override was given, so nothing was searched for.
    found : bool
        True iff ``gene_model`` exists on disk. The single fact callers branch on.
    status : str
        Human-readable provenance for the run manifest and the log.
    """

    gene_model: "Path | None"
    descriptions: "Path | None"
    searched: "Path | None"
    found: bool
    status: str


def resolve_gene_model(species, genome_build="SL3", override=None, data_dir=None):
    """Resolve the gene model for a run, reporting whether it is present.

    Parameters
    ----------
    species : str
        Species key, e.g. "tomato". Species with no bundled table resolve to
        "nothing found" unless ``override`` is given.
    genome_build : str
        Build key, e.g. "SL3" or "SL4". Unknown builds fall back to "SL3",
        matching the CLI's own default.
    override : str or Path or None
        An explicit ``--gene-model`` path. Takes precedence over the bundled
        table. Descriptions still come from the species defaults, so
        ``--species tomato --gene-model mine.csv`` keeps the tomato descriptions.
    data_dir : str or Path or None
        Directory holding the bundled tables. Defaults to the shipped
        ``_BUNDLED_DATA_DIR``.

    Returns
    -------
    GeneModelResolution
    """
    root = Path(data_dir) if data_dir is not None else _BUNDLED_DATA_DIR
    build = genome_build if genome_build in _BUNDLED_GENE_FILES.get(species, {}) else "SL3"
    names = _BUNDLED_GENE_FILES.get(species, {}).get(build)

    default_gm = root / names[0] if names else None
    desc = root / names[1] if names else None
    if desc is not None and not desc.exists():
        desc = None

    gene_model = Path(override) if override else default_gm

    if gene_model is None:
        return GeneModelResolution(
            gene_model=None, descriptions=None, searched=None, found=False,
            status=f"none (species='{species}', no --gene-model)",
        )
    if not gene_model.exists():
        return GeneModelResolution(
            gene_model=None, descriptions=None, searched=gene_model, found=False,
            status=f"not found: searched {gene_model}",
        )
    return GeneModelResolution(
        gene_model=gene_model, descriptions=desc, searched=gene_model, found=True,
        status=f"used: {gene_model}",
    )


# ============================================================
# 2. LD BLOCK ANNOTATION
# ============================================================

def _find_overlapping_genes(genes_chr, block_start, block_end):
    mask = (genes_chr["Start"].values <= block_end) & (genes_chr["End"].values >= block_start)
    return genes_chr.loc[mask].copy()


def summarize_gene_model(genes_df):
    """Per-chromosome summary of a loaded gene model, for upload-time validation.

    Given the normalised frame from :func:`load_gene_annotation` (columns
    Chr/Start/End/Gene_ID), return a tidy per-chromosome table so a user can
    sanity-check that an uploaded gene model is in the same coordinate frame /
    assembly as their genotype data. A coordinate range that does not cover the
    same span as the SNP positions is the visible symptom of a wrong build.

    Returns
    -------
    DataFrame with columns: Chr, n_genes, min_start, max_end, span_Mb
    (one row per chromosome, ordered by chromosome number).
    """
    cols = ["Chr", "n_genes", "min_start", "max_end", "span_Mb"]
    if genes_df is None or len(genes_df) == 0:
        return pd.DataFrame(columns=cols)
    g = genes_df.copy()
    g["Chr"] = g["Chr"].astype(str).map(canon_chr)
    g["Start"] = pd.to_numeric(g["Start"], errors="coerce")
    g["End"] = pd.to_numeric(g["End"], errors="coerce")
    rows = []
    for ch, sub in g.groupby("Chr"):
        mn = int(sub["Start"].min())
        mx = int(sub["End"].max())
        rows.append({
            "Chr": ch, "n_genes": int(len(sub)),
            "min_start": mn, "max_end": mx,
            "span_Mb": round((mx - mn) / 1e6, 2),
        })
    out = pd.DataFrame(rows, columns=cols)
    out["_ord"] = pd.to_numeric(out["Chr"], errors="coerce")
    out = out.sort_values(["_ord", "Chr"]).drop(columns="_ord").reset_index(drop=True)
    return out


def annotate_ld_blocks(ld_blocks, genes):
    """
    Annotate each row with the genes that overlap its candidate interval.

    The interval is ``Interval_start`` / ``Interval_end`` when the row has them (an LD
    block's interval from the marker before its first SNP to the marker after its last:
    gwas.isolated.block_candidate_intervals), else the row's own Start/End (an unblocked
    SNP's interval is its ``Start (bp)`` / ``End (bp)``). A gene counts when it overlaps
    the interval; nothing outside it is listed.

    Parameters
    ----------
    ld_blocks : DataFrame with Chr, Start/Start (bp), End/End (bp) and, for LD blocks,
        Interval_start / Interval_end
    genes : DataFrame from load_gene_annotation()

    Returns
    -------
    The rows (Start/End as ``Start (bp)`` / ``End (bp)``) with added columns
    n_genes_overlapping, overlapping_genes, overlapping_descriptions and
    annotation_status: "overlapping", or "no_genes_nearby" when no gene lies inside
    the interval.
    """
    if ld_blocks is None or ld_blocks.empty:
        return ld_blocks

    out = ld_blocks.copy()

    if "Start (bp)" in out.columns:
        out = out.rename(columns={"Start (bp)": "Start", "End (bp)": "End"})
    out["Chr"] = out["Chr"].astype(str).map(canon_chr)
    out["Start"] = out["Start"].astype(int)
    out["End"] = out["End"].astype(int)
    by_interval = {"Interval_start", "Interval_end"}.issubset(out.columns)

    genes = genes.copy()
    genes["Chr"] = genes["Chr"].astype(str).map(canon_chr)
    genes_by_chr = {ch: g for ch, g in genes.groupby("Chr")}

    n_genes_list = []
    overlapping_genes_list = []
    overlapping_desc_list = []
    annotation_status_list = []

    for _, row in out.iterrows():
        ch = str(row["Chr"])
        if by_interval:
            s, e = int(row["Interval_start"]), int(row["Interval_end"])
        else:
            s, e = int(row["Start"]), int(row["End"])

        genes_chr = genes_by_chr.get(ch, pd.DataFrame())
        if genes_chr.empty:
            n_genes_list.append(0)
            overlapping_genes_list.append("")
            overlapping_desc_list.append("")
            annotation_status_list.append("no_genes_nearby")
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
        annotation_status_list.append("overlapping" if len(overlap) > 0 else "no_genes_nearby")

    out["n_genes_overlapping"] = n_genes_list
    out["overlapping_genes"] = overlapping_genes_list
    out["overlapping_descriptions"] = overlapping_desc_list
    out["annotation_status"] = annotation_status_list

    out = out.rename(columns={"Start": "Start (bp)", "End": "End (bp)"})
    return out


ANNOTATION_TABLE_COLUMNS = ["Type", "Chr", "Interval start", "Interval end", "Lead SNP",
                            "Number of genes", "Genes"]


def annotate_candidate_intervals(intervals, genes):
    """The Gene Annotation tab's one table: each candidate interval (an LD block or a run of
    unblocked significant SNPs, gwas.isolated.candidate_interval_table) with the genes
    inside it. ``Genes`` gives each as "ID: description" where it has one, joined by
    " | ". Returns ``(table, summary)``, summary "N intervals, M genes" (M distinct
    genes)."""
    if intervals is None or len(intervals) == 0:
        return pd.DataFrame(columns=ANNOTATION_TABLE_COLUMNS), "0 intervals, 0 genes"
    rows = pd.DataFrame({"Chr": intervals["Chr"].astype(str).to_numpy(),
                         "Start": intervals["Interval_start"].astype(int).to_numpy(),
                         "End": intervals["Interval_end"].astype(int).to_numpy()})
    ann = annotate_ld_blocks(rows, genes)
    table = pd.DataFrame({
        "Type": intervals["Type"].to_numpy(),
        "Chr": ann["Chr"].to_numpy(),
        "Interval start": ann["Start (bp)"].to_numpy(),
        "Interval end": ann["End (bp)"].to_numpy(),
        "Lead SNP": intervals["Lead_SNP"].to_numpy(),
        "Number of genes": ann["n_genes_overlapping"].astype(int).to_numpy(),
        "Genes": ann["overlapping_descriptions"].to_numpy(),
    })[ANNOTATION_TABLE_COLUMNS]
    ids = {g for v in ann["overlapping_genes"] for g in str(v).split(";") if g}
    n_i, n_g = len(table), len(ids)
    return table, f"{n_i} interval{'' if n_i == 1 else 's'}, {n_g} gene{'' if n_g == 1 else 's'}"


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
            "eta2", "eta2_ci_low", "eta2_ci_high", "omega2", "lead_snp_r2",
            "n_permutations",
        ] if c in hm.columns]
        hsub = hm[keep].copy()
        hsub = hsub.rename(columns={
            "PValue": "Hap_PValue", "FDR_BH": "Hap_FDR_BH",
            "F_perm": "Hap_F_perm", "F_param": "Hap_F_param",
            "n_haplotypes": "Hap_n_haplotypes",
            "n_tested_haplotypes": "Hap_n_tested",
            "n_samples_block": "Hap_n_samples",
            "eta2": "Hap_eta2",
            "eta2_ci_low": "Hap_eta2_ci_low", "eta2_ci_high": "Hap_eta2_ci_high",
            "omega2": "Hap_omega2", "lead_snp_r2": "Hap_lead_r2",
            "n_permutations": "Hap_n_perms",
        })
        for df in [base, hsub]:
            df["Chr"] = df["Chr"].astype(str)
            df["Start (bp)"] = pd.to_numeric(df["Start (bp)"], errors="coerce").astype("Int64")
            df["End (bp)"] = pd.to_numeric(df["End (bp)"], errors="coerce").astype("Int64")
        base = base.merge(hsub, on=["Chr", "Start (bp)", "End (bp)"], how="left")

    return base


# ============================================================
# 3. LD DECAY VISUALIZATION
# ============================================================

def compute_ld_decay_by_chromosome(
    chroms,
    positions,
    geno_imputed,
    max_snps_per_chr=1500,   # aligned to the CLI's per-chromosome subsample cap (cli.py, the `if len(pos_ch) > 1500`
    # in the "Estimating LD decay" loop — not ld_decay's MAX_LD_SNPS safety net)
    max_dist_kb=5000.0,
    n_bins=40,   # standardised onto the CLI ld_decay bin count (LD-decay Tier 1a)
    ld_thresholds=(0.1, 0.2),
    min_pair_n=20,
):
    """
    Compute LD decay per chromosome (the visualization / Decay-tab path).

    Same definition as gwas/ld.py::ld_decay: the CENTRE of the first LINEAR distance
    bin whose MEDIAN r² <= threshold, per chromosome, over the observed min/max pair
    distance. n_bins defaults to 40 (standardised onto the CLI ld_decay bin count,
    LD-decay Tier 1a). Same RESOLUTION FLOOR (~range/n_bins/2): when the crossing is
    the first bin the value is grid-limited -- an UPPER BOUND, not an estimate --
    flagged in summary_df["censored_r2_0.2"]. Uses np.digitize with a >=5-pair-per-bin
    minimum, and subsamples to max_snps_per_chr=1500, matching the CLI ld_decay path
    (LD-decay follow-up A); with the same bin count AND subsample cap it reproduces the
    CLI decay value on the tomato panel (median 72.17). The only remaining difference is
    a <10-vs-<50 minimum-SNP-per-chromosome skip, irrelevant unless a chromosome has <50 SNPs.

    Returns:
        decay_df : binned r2 vs distance per chromosome
        summary_df : per-chr decay distances (decay_kb_r2_{0.1,0.2}) + censored_r2_0.2

    ``geno_imputed`` may be a table kept on disk (gwas.genostore.LazyColumns): then each
    chromosome reads only the columns it uses (at most ``max_snps_per_chr``), as one float64
    F-ordered block, the dtype and layout the in-memory selections give pairwise_r2.
    """
    from gwas.genostore import is_lazy
    from gwas.ld import pairwise_r2

    chroms = np.asarray([canon_chr(c) for c in chroms], dtype=object)
    positions = np.asarray(positions, int)
    lazy = is_lazy(geno_imputed)
    geno = geno_imputed.astype(float) if lazy else np.asarray(geno_imputed, float)

    all_bins = []
    summaries = []

    for ch in sorted(set(chroms.astype(str))):
        mask = chroms.astype(str) == ch
        idx = np.where(mask)[0]
        if len(idx) < 10:
            continue

        chr_pos = positions[idx]
        if lazy:
            # on disk: the columns the in-memory selections below end with, read once
            order = np.argsort(chr_pos)
            chr_pos = chr_pos[order]
            cols = idx[order]
            if len(chr_pos) > max_snps_per_chr:
                sub_idx = np.linspace(0, len(chr_pos) - 1, max_snps_per_chr, dtype=int)
                chr_pos = chr_pos[sub_idx]
                cols = cols[sub_idx]
            chr_geno = geno.block(cols, layout="F")
        else:
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
            # LD-decay Tier 1b: grid-limited (censored) when the r2<=0.2 crossing is
            # the FIRST bin centre -> the true decay is below this value (upper bound).
            _fc = float(bin_centers[0])
            _d02 = summary.get("decay_kb_r2_0.2", np.nan)
            summary["censored_r2_0.2"] = bool(np.isfinite(_d02) and _d02 == _fc)
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
