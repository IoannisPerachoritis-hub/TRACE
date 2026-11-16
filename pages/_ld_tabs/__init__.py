"""
LD analysis tab modules.

Each tab is a separate module exporting a ``render(ctx)`` function
that receives an LDContext dataclass with all shared state.
"""

import dataclasses
import numpy as np
import pandas as pd


@dataclasses.dataclass
class LDContext:
    """Shared state passed to every LD tab renderer.

    The genotype tables are read in windows and blocks, never copied whole. The page passes
    gwas.guipost readers (the trait's table in memory or on disk); a test may pass plain
    arrays, which are read the same way."""

    # --- genotype tables (column-aligned; read a window or a block at a time) ---
    geno_ld: object            # the imputed table as the LD code reads it (float64 windows)
    geno_hard: object          # its hard calls, a column or a block at a time
    chroms: np.ndarray
    positions: np.ndarray
    sid: np.ndarray
    geno_df: pd.DataFrame      # the sample index (the page passes an index-only frame)

    # --- phenotype ---
    pheno_df: pd.DataFrame
    trait_col: str
    ph_aligned: pd.DataFrame
    y_aligned: np.ndarray
    keep_mask: np.ndarray
    geno_ld_aligned: object    # geno_ld's phenotype-aligned rows

    # --- GWAS results ---
    gwas_df: pd.DataFrame

    # --- LD block tables ---
    haplo_df_auto: pd.DataFrame

    # --- LD parameters ---
    ld_decay_kb: float | None   # None without an estimate (the window is then 300 kb)
    adj_r2_min_global: float

    # --- resolved trait name (after column matching) ---
    ld_trait: str = ""

    # --- principal components (optional) ---
    pcs: np.ndarray | None = None

    # --- misc ---
    geno_encoding: str | None = None
    has_annotation: bool = False

    # --- significant-SNP tab: raw dosage + the resolved reporting rule ---
    geno_dosage_raw: object | None = None   # the raw dosage table, read like geno_ld
    meff_val: int | None = None
    sig_rule_label: str = ""
    custom_thresh: float | None = None

    # --- the table's identity (every tab cache names it: gwas.guipost.table_key) ---
    table_identity: str = ""
    # --- the GWAS page's imputed table as it is (the haplotype PCA panel's fallback) ---
    geno_view: object | None = None
    # --- the GWAS results' key (gwas.guipost.frame_key), filled when first needed ---
    gwas_key: str = ""
