"""
Phenotype normality & distribution-check page package.

Standalone in-app tool: upload a phenotype table, test each trait for
normality (Shapiro-Wilk / D'Agostino K² / Anderson-Darling + skew/kurtosis),
show histogram + Q-Q diagnostics, and recommend/apply the same normalising
transforms the GWAS pipeline uses (log10, Yeo-Johnson, rank-INT).

Each submodule exports a ``render(...)`` function that receives the shared
``PhenoQCContext`` dataclass, following the same pattern as ``pages/_vnn/``
and ``pages/_ld_tabs/``. This page is fully standalone. It does NOT write
into GWAS session state and has no upstream data-version dependency.
"""

import dataclasses

import pandas as pd

# Sample/accession ID column names recognised on upload, kept identical to
# the GWAS page loader (pages/GWAS_analysis.py) so behaviour matches.
_ID_COL_CANDIDATES = {
    "accessions", "accession", "sample", "samples",
    "iid", "genotype", "id", "sample_id", "sampleid",
}


@dataclasses.dataclass
class PhenoQCContext:
    """Shared state passed to every phenotype-QC submodule renderer."""

    pheno_df: pd.DataFrame          # index = sample/accession ID, columns = traits
    numeric_cols: list              # testable (numeric) trait column names
    id_col: str                     # name of the detected ID column
    source: str                     # "upload" or "session" (provenance label)


def load_phenotype_file(uploaded):
    """Parse an uploaded phenotype file into an ID-indexed DataFrame.

    Mirrors the GWAS page loader (pages/GWAS_analysis.py:549-617): separator
    auto-detection with an encoding fallback, ID-column auto-detection (else
    first column), and stripped string index. Returns
    ``(pheno_df, id_col, latin1_used)``.

    Raises ValueError with a human-readable message on unrecoverable parse
    failure so the caller can surface it via ``st.error``.
    """
    latin1_used = False
    try:
        pheno = pd.read_csv(uploaded, sep=None, engine="python", encoding="utf-8-sig")
    except (UnicodeDecodeError, UnicodeError):
        uploaded.seek(0)
        try:
            pheno = pd.read_csv(uploaded, sep=None, engine="python", encoding="latin-1")
            latin1_used = True
        except Exception as enc_err:  # noqa: BLE001
            raise ValueError(f"Could not read phenotype file: {enc_err}") from enc_err
    except Exception as read_err:  # noqa: BLE001
        raise ValueError(f"Could not parse phenotype file: {read_err}") from read_err

    pheno = pheno.copy()
    pheno.columns = pheno.columns.astype(str).str.strip()

    # Find an explicit accession column first (case-insensitive), else first column.
    id_col = None
    for col in pheno.columns:
        if col.lower().strip() in _ID_COL_CANDIDATES:
            id_col = col
            break
    if id_col is None:
        id_col = pheno.columns[0]

    pheno[id_col] = pheno[id_col].astype(str).str.strip()
    pheno = pheno.set_index(id_col)
    pheno.index = pheno.index.astype(str).str.strip()

    return pheno, id_col, latin1_used
