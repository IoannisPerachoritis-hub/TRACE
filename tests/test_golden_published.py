"""T-94 — Published-output regression guard (opt-in).

Rebuilds the tomato locule-number downstream tables from the *real* Varitome QC
inputs and asserts they are bitwise-stable against the frozen golden
(`tests/golden/tomato_locule/`). Marked ``golden`` because it needs
``benchmarks/qc_data/`` (gitignored), so it is excluded from CI with
``-m "not golden"`` and skips cleanly on a machine without the data.

The frozen golden itself — and the demonstration that the tripwire FIRES on
``min_snps=2`` and on ``flank_kb ± 10 %`` — is produced in Batch B once T-61
reports the flank that reproduces Table S8; the reproducing parameters are read
back from the golden's own ``run_manifest.json``. Until then these tests skip.

Spec: ``docs/revision/specs/coverage_and_scope.md`` T-60/T-94.
"""
import importlib.util
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "golden" / "tomato_locule"
QC = REPO / "benchmarks" / "qc_data" / "tomato_locule_number"


def _load_capture_module():
    spec = importlib.util.spec_from_file_location(
        "capture_golden", REPO / "benchmarks" / "capture_golden.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _golden_ready():
    return GOLDEN.exists() and (GOLDEN / "run_manifest.json").exists() and QC.exists()


def _rebuild(tmp_path):
    """Re-run the capture with the golden's own recorded parameters."""
    manifest = json.loads((GOLDEN / "run_manifest.json").read_text(encoding="utf-8"))
    cap = _load_capture_module()
    genes = manifest.get("genes_csv")
    descr = manifest.get("descriptions_csv")
    cap.capture_golden(
        QC, QC / "platform_GWAS_locule_number.csv", tmp_path,
        flank_kb=manifest["flank_kb"], ld_decay_kb=manifest.get("ld_decay_kb"),
        n_perm=manifest.get("n_perm", 1000),
        genes_csv=Path(genes) if genes else None,
        descriptions_csv=Path(descr) if descr else None,
        trait_col=manifest.get("trait_col"),
    )
    return tmp_path


# --------------------------------------------------------------------------
# always-on: the design guard (runs in CI; no data required)
# --------------------------------------------------------------------------
def test_capture_golden_flank_kb_is_required():
    """`flank_kb` must be keyword-only with NO default — the whole point of the
    harness is that no flank value is silently enshrined (§0.1 / T-61)."""
    mod = _load_capture_module()
    p = inspect.signature(mod.capture_golden).parameters["flank_kb"]
    assert p.default is inspect.Parameter.empty, "flank_kb must have no default (T-94)"
    assert p.kind is inspect.Parameter.KEYWORD_ONLY


# --------------------------------------------------------------------------
# opt-in: the real-data bitwise pin
# --------------------------------------------------------------------------
@pytest.mark.golden
def test_ld_blocks_bitwise_stable(tmp_path):
    if not _golden_ready():
        pytest.skip("tomato_locule golden not captured yet (T-61/Batch B) or qc_data/ absent")
    out = _rebuild(tmp_path)
    got = pd.read_csv(out / "ld_blocks.csv")
    exp = pd.read_csv(GOLDEN / "ld_blocks.csv")
    # integer basepair boundaries — a one-bp shift is exactly the failure guarded
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True), exp.reset_index(drop=True),
        check_exact=True, check_dtype=False,
    )


@pytest.mark.golden
def test_haplotype_blocks_bitwise_stable(tmp_path):
    if not _golden_ready():
        pytest.skip("tomato_locule golden not captured yet (T-61/Batch B) or qc_data/ absent")
    out = _rebuild(tmp_path)
    got = pd.read_csv(out / "haplotype_blocks.csv")
    exp = pd.read_csv(GOLDEN / "haplotype_blocks.csv")
    # permutation p-values are deterministic given stable_seed -> exact to 1e-12
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True), exp.reset_index(drop=True),
        check_exact=False, rtol=0, atol=1e-12, check_dtype=False,
    )


@pytest.mark.golden
def test_annotated_blocks_bitwise_stable(tmp_path):
    if not _golden_ready():
        pytest.skip("tomato_locule golden not captured yet (T-61/Batch B) or qc_data/ absent")
    if not (GOLDEN / "annotated_blocks.csv").exists():
        pytest.skip("annotated golden not present")
    out = _rebuild(tmp_path)
    got = pd.read_csv(out / "annotated_blocks.csv")
    exp = pd.read_csv(GOLDEN / "annotated_blocks.csv")
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True), exp.reset_index(drop=True),
        check_exact=True, check_dtype=False,
    )
