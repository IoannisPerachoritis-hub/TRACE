"""Which copy of the tomato QC data the real-data golden tests read, and whether it is the one the
goldens were made from.

The tomato goldens (tests/golden/tomato_locule, tests/golden/varitome_locule) were produced from
the corrected 164 x 43,749 QC (tests/golden/REGENERATION_LOG.md). A machine can hold an older
copy in its gitignored benchmarks/qc_data folder; the three tests that read it then skip with a
reason instead of failing on data that is not the goldens' (their manifest records the sha256 of
the GWAS table they were made from: gwas_csv_sha256)."""
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "tests" / "golden" / "tomato_locule" / "run_manifest.json"


def qc_dir():
    """TRACE_VARITOME_DIR if set, else benchmarks/qc_data/tomato_locule_number."""
    return Path(os.environ.get("TRACE_VARITOME_DIR")
                or REPO / "benchmarks" / "qc_data" / "tomato_locule_number")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def qc_copy_problem(folder=None):
    """None when ``folder`` holds the QC data the goldens were made from; otherwise why not
    (absent, or another copy, with both hashes)."""
    folder = qc_dir() if folder is None else Path(folder)
    gwas = folder / "platform_GWAS_locule_number.csv"
    if not (folder / "QC_genotype_matrix.csv").exists() or not gwas.exists():
        return (f"the tomato QC inputs are absent from {folder} (gitignored; regenerate them or set "
                "TRACE_VARITOME_DIR)")
    want = json.loads(MANIFEST.read_text(encoding="utf-8"))["gwas_csv_sha256"]
    got = _sha256(gwas)
    if got != want:
        return (f"{folder} holds another copy of the tomato QC data (platform_GWAS_locule_number.csv "
                f"sha256 {got[:12]}..., the goldens were made from {want[:12]}...); see "
                "tests/golden/REGENERATION_LOG.md")
    return None
