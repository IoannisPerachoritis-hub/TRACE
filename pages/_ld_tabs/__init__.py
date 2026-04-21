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
    """Shared state passed to every LD tab renderer."""

    # --- genotype arrays (column-aligned) ---
    geno_ld: np.ndarray
    geno_hard: np.ndarray
    chroms: np.ndarray
    positions: np.ndarray
    sid: np.ndarray
    geno_df: pd.DataFrame

    # --- phenotype ---
    pheno_df: pd.DataFrame
    trait_col: str
    ph_aligned: pd.DataFrame
    y_aligned: np.ndarray
    keep_mask: np.ndarray
    geno_ld_aligned: np.ndarray

    # --- GWAS results ---
    gwas_df: pd.DataFrame

    # --- LD block tables ---
    haplo_df_auto: pd.DataFrame

    # --- LD parameters ---
    ld_decay_kb: float
    adj_r2_min_global: float

    # --- resolved trait name (after column matching) ---
    ld_trait: str = ""

    # --- principal components (optional) ---
    pcs: np.ndarray | None = None

    # --- misc ---
    geno_encoding: str | None = None
    show_ld_labels: bool = False
    has_annotation: bool = False
