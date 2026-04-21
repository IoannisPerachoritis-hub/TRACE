#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# One-command reviewer reproduction script for TRACE.
#
# Runs the simulator-based example end-to-end and asserts the
# expected calibration window (lambda_GC ∈ [0.85, 1.15]) and
# at least one significant SNP near the embedded QTL on chr2.
#
# Reviewer workflow:
#   git clone https://github.com/IoannisPerachoritis-hub/TRACE.git
#   cd TRACE
#   pip install -e . -c constraints.txt
#   bash benchmarks/reproduce.sh
#
# Exit code 0 = PASS, non-zero = FAIL with diagnostic message.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

OUT_DIR="${REPO_ROOT}/benchmarks/reproduce_out"
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

echo "==[1/4]== Generating synthetic example data"
python examples/simulate_example.py

echo "==[2/4]== Running TRACE MLM pipeline"
trace-gwas \
    --vcf  "examples/example.vcf.gz" \
    --pheno "examples/example_pheno.csv" \
    --trait Trait1 \
    --output "${OUT_DIR}/" \
    --model mlm \
    --seed 42 \
    --no-plots \
    --no-report

echo "==[3/4]== Verifying calibration + signal recovery"
ZIP_PATH="$(find "${OUT_DIR}" -name '*.zip' | head -n 1)"
if [[ -z "${ZIP_PATH}" ]]; then
    echo "FAIL: TRACE produced no output ZIP." >&2
    exit 1
fi

python - "${ZIP_PATH}" <<'PY'
import sys, zipfile
import pandas as pd
import numpy as np
from scipy.stats import chi2

zip_path = sys.argv[1]
with zipfile.ZipFile(zip_path) as z:
    name = next((n for n in z.namelist() if n.endswith("GWAS_Trait1.csv") or n.startswith("tables/GWAS_")), None)
    if name is None:
        print("FAIL: GWAS results CSV not found in ZIP.", file=sys.stderr); sys.exit(1)
    with z.open(name) as fh:
        df = pd.read_csv(fh)

p = df["PValue"].astype(float).values
p = p[np.isfinite(p) & (p > 0) & (p <= 1)]
p = np.clip(p, 1e-300, 1.0)

# Genomic inflation lambda_GC: median(chi^2_1 from p) / median chi^2_1 expected (0.4549).
# Untrimmed because n_SNPs is small in the demo (510 < 500 trim threshold).
chisq = chi2.isf(p, df=1)
lam = float(np.median(chisq) / 0.454936423119572)
print(f"  lambda_GC = {lam:.3f}  (expected window: [0.85, 1.15])")
assert 0.85 <= lam <= 1.15, f"FAIL: lambda_GC = {lam:.3f} outside [0.85, 1.15]"

# At least one SNP on chr2 should be near-significant (Bonferroni 5%).
# QTL was placed at chr02_860000 by examples/simulate_example.py.
bonf = 0.05 / len(df)
chr_str = df["Chr"].astype(str).str.lstrip("0").replace("", "0")
chr2_hits = df[(chr_str == "2") & (df["PValue"] < bonf)]
print(f"  Bonferroni 5% hits on chr2: {len(chr2_hits)} (QTL was placed on chr2)")
assert len(chr2_hits) >= 1, (
    f"FAIL: zero Bonferroni-significant SNPs on chr2 - "
    f"signal recovery broken (Bonf threshold {bonf:.2e})."
)

print("PASS: calibration in window AND chr2 QTL recovered.")
PY

echo "==[4/4]== Done."
