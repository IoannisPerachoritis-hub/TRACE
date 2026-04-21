#!/usr/bin/env bash
# Quick-start example: generate synthetic data and run TRACE.
# Usage:  bash examples/run_example.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Step 1: Generate synthetic data ==="
python "$SCRIPT_DIR/simulate_example.py"

echo ""
echo "=== Step 2: Run TRACE MLM GWAS ==="
trace-gwas \
    --vcf  "$SCRIPT_DIR/example.vcf.gz" \
    --pheno "$SCRIPT_DIR/example_pheno.csv" \
    --trait  Trait1 \
    --output "$SCRIPT_DIR/output/" \
    --model  mlm

echo ""
echo "=== Done! Results in $SCRIPT_DIR/output/ ==="
