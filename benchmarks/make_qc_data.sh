#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Regenerate benchmarks/qc_data/<dataset>/ matrices used by the
# concordance and per-tool comparison scripts.
#
# The QC'd matrices total ~106 MB and are too large to track in git;
# this script reproduces them locally from the source VCF + phenotype
# pairs. After running, benchmarks/plink_qc/compare_qc.py and the R
# scripts under benchmarks/run_*.R can be invoked directly.
#
# Usage:
#   bash benchmarks/make_qc_data.sh <dataset> <vcf_path> <pheno_path> <trait>
#
# Example (after fetching the public VCFs):
#   bash benchmarks/make_qc_data.sh tomato_locule_number \
#     data/tomato/varitome_filtered.vcf \
#     data/tomato/varitome_phenotypes.csv \
#     locule_number
#
# Source VCFs:
#   - Tomato Varitome panel: Pereira et al. (2021), Solanaceae Genomics
#     Network ftp://ftp.solgenomics.net/genomes/Solanum_lycopersicum/
#   - Pepper G2P-SOL panel: Tripodi et al. (2021), G2P-SOL Project
#     https://www.g2p-sol.eu/
# ──────────────────────────────────────────────────────────────
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <dataset_name> <vcf_path> <pheno_path> <trait_column>"
    echo "Example: $0 tomato_locule_number data/tomato/varitome_filtered.vcf \\"
    echo "           data/tomato/varitome_phenotypes.csv locule_number"
    exit 1
fi

DATASET="$1"
VCF="$2"
PHENO="$3"
TRAIT="$4"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${REPO_ROOT}/benchmarks/qc_data/${DATASET}"

if [[ ! -f "${VCF}" ]]; then
    echo "ERROR: VCF not found: ${VCF}" >&2
    exit 2
fi
if [[ ! -f "${PHENO}" ]]; then
    echo "ERROR: Phenotype CSV not found: ${PHENO}" >&2
    exit 2
fi

mkdir -p "${OUT_DIR}"
TMP_OUT="$(mktemp -d)"

echo "→ Running TRACE --export-qc on ${DATASET}…"
trace-gwas \
    --vcf "${VCF}" \
    --pheno "${PHENO}" \
    --trait "${TRAIT}" \
    --output "${TMP_OUT}" \
    --model mlm \
    --export-qc \
    --no-plots \
    --no-report

echo "→ Extracting QC matrices to ${OUT_DIR}…"
ZIP_PATH="$(find "${TMP_OUT}" -name '*.zip' | head -n 1)"
if [[ -z "${ZIP_PATH}" ]]; then
    echo "ERROR: No ZIP produced under ${TMP_OUT}" >&2
    exit 3
fi

unzip -j -o "${ZIP_PATH}" \
    "tables/QC_genotype_matrix.csv" \
    "tables/QC_snp_map.csv" \
    "tables/QC_phenotype.csv" \
    -d "${OUT_DIR}"

rm -rf "${TMP_OUT}"

echo "✓ QC matrices written to ${OUT_DIR}/"
ls -lh "${OUT_DIR}/"
