"""Canonical serialisation and digest helpers for the golden block tables (T-95).

`canon_block_table` is the single definition of "the same block table": order is
normalised on (Chr, Start (bp), End (bp)), and the two set-valued columns
(`Lead SNP`, `SNP_IDs`) are tokenised, sorted and rejoined so that an incidental
ordering difference — `_merge_leads` builds a `;`-joined *set* and the detector
builds a `,`-joined sorted set, neither with a cross-version order guarantee —
cannot cause a false failure.

`GOLDEN_LOCK` is a sha256 over the canonical concatenation of every
`expected_*.csv` in the tree (path-tagged), so a deliberate regeneration always
shows up as a one-line diff on a file whose only job is to be noticed in review.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

GOLDEN_DIR = Path(__file__).resolve().parent
BLOCK_COLUMNS = ["Chr", "Start (bp)", "End (bp)", "Lead SNP", "SNP_IDs"]

_LEAD_SPLIT = re.compile(r"[;,\s|]+")


def _tok(value, splitter):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    toks = [t for t in splitter(str(value)) if t]
    return ",".join(sorted(toks)) if toks else ""


def canon_block_table(df_or_path):
    """Return the canonical CSV text (``\\n`` line endings, no index) for a block
    table given as a DataFrame or a path to a CSV."""
    if isinstance(df_or_path, (str, Path)):
        df = pd.read_csv(df_or_path, dtype=str, keep_default_na=False)
    else:
        df = df_or_path.copy()

    out = pd.DataFrame(columns=BLOCK_COLUMNS)
    if df is not None and len(df) > 0:
        # SNP_IDs may be absent on the 4-column empty schema; tolerate it.
        chr_ = df["Chr"].astype(str)
        start = df["Start (bp)"].astype(float).round().astype("int64")
        end = df["End (bp)"].astype(float).round().astype("int64")
        lead = df["Lead SNP"].map(lambda v: _tok(v, _LEAD_SPLIT.split))
        snpids = (
            df["SNP_IDs"].map(lambda v: _tok(v, lambda s: s.split(",")))
            if "SNP_IDs" in df.columns
            else pd.Series([""] * len(df))
        )
        out = pd.DataFrame(
            {"Chr": chr_.values, "Start (bp)": start.values, "End (bp)": end.values,
             "Lead SNP": lead.values, "SNP_IDs": snpids.values}
        )
        out = out.sort_values(["Chr", "Start (bp)", "End (bp)", "Lead SNP"]).reset_index(drop=True)

    return out.to_csv(index=False, lineterminator="\n")


def load_case(case):
    """Return ``(inp, expected_blocks_df, meta)`` for a case directory.

    ``inp`` is the loaded ``input.npz`` (Tier A only); ``None`` for Tier-B cases
    that ship no genotypes.
    """
    d = GOLDEN_DIR / case
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    npz = d / "input.npz"
    inp = dict(np.load(npz, allow_pickle=False)) if npz.exists() else None
    expected = pd.read_csv(d / "expected_blocks.csv", dtype=str, keep_default_na=False)
    return inp, expected, meta


def expected_csv_paths():
    """Every frozen ``expected_*.csv`` in the tree, sorted by relative path."""
    return sorted(GOLDEN_DIR.glob("*/expected_*.csv"), key=lambda p: p.relative_to(GOLDEN_DIR).as_posix())


def canonical_concat():
    """Path-tagged canonical concatenation of every expected table."""
    parts = []
    for p in expected_csv_paths():
        rel = p.relative_to(GOLDEN_DIR).as_posix()
        parts.append(f"### {rel}\n{canon_block_table(p)}")
    return "\n".join(parts)


def golden_lock_digest():
    return hashlib.sha256(canonical_concat().encode("utf-8")).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
