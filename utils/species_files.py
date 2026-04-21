"""
Species-specific annotation file paths.

Shared between the GWAS pipeline page and the LD analysis page so that
the definition lives in one place and does not create cross-module imports.
"""

from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SPECIES_FILES = {
    "Solanum lycopersicum (tomato)": {
        "gene_model": _DATA_DIR / "Sol_genes_SL3.csv",
        "gene_desc": _DATA_DIR / "SL3.1_descriptions.txt",
        "gene_model_SL3": _DATA_DIR / "Sol_genes_SL3.csv",
        "gene_desc_SL3": _DATA_DIR / "SL3.1_descriptions.txt",
        "gene_model_SL4": _DATA_DIR / "Sol_genes.csv",
        "gene_desc_SL4": _DATA_DIR / "ITAG4.0_annotation.txt",
    },
    "Capsicum annuum (pepper)": {
        "gene_model": _DATA_DIR / "cann_gene_model.csv",
        "gene_desc": _DATA_DIR / "cann_gene_annotation.txt",
    },
}
