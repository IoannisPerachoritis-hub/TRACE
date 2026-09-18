"""T-95 — Golden-fixture harness guards.

Structural invariants on ``tests/golden/`` itself: the on-disk budget, the
per-case file completeness, the ``input_sha256`` binding in each ``meta.json``,
the test-only isolation of the fixtures, and the refusal behaviour of the
guarded regeneration driver.

Spec: ``docs/revision/specs/integration_and_testing.md`` §1.1, §1.4.
"""
import json
import os
import subprocess
import sys

import pytest

from tests.golden._canon import GOLDEN_DIR, sha256_file
from tests.golden._cases import STD_PARAMS, TIER_A_CASES

REPO = GOLDEN_DIR.parents[1]
PER_CASE_BUDGET = 40 * 1024
TOTAL_BUDGET = 250 * 1024


def _tracked_files():
    """Files under tests/golden/ that would be committed (excludes __pycache__)."""
    return [
        p for p in GOLDEN_DIR.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc")
    ]


def test_total_on_disk_budget():
    total = sum(p.stat().st_size for p in _tracked_files())
    assert total <= TOTAL_BUDGET, f"tests/golden/ is {total} bytes (> {TOTAL_BUDGET})"


@pytest.mark.parametrize("case", TIER_A_CASES)
def test_case_has_required_files(case):
    d = GOLDEN_DIR / case
    for fname in ("input.npz", "expected_blocks.csv", "meta.json"):
        assert (d / fname).exists(), f"{case}/{fname} missing"


@pytest.mark.parametrize("case", TIER_A_CASES)
def test_input_npz_within_budget(case):
    size = (GOLDEN_DIR / case / "input.npz").stat().st_size
    assert size <= PER_CASE_BUDGET, f"{case}/input.npz is {size} bytes (> {PER_CASE_BUDGET})"


@pytest.mark.parametrize("case", TIER_A_CASES)
def test_meta_input_sha256_matches(case):
    meta = json.loads((GOLDEN_DIR / case / "meta.json").read_text(encoding="utf-8"))
    on_disk = sha256_file(GOLDEN_DIR / case / "input.npz")
    assert meta["input_sha256"] == on_disk, (
        f"{case}/meta.json input_sha256 does not match input.npz — the input was "
        f"changed without regenerating the golden"
    )


@pytest.mark.parametrize("case", TIER_A_CASES)
def test_meta_params_match_the_standard_set(case):
    """The params block is read by the golden test and passed to the detector, so
    a golden can never be silently reinterpreted under different parameters."""
    meta = json.loads((GOLDEN_DIR / case / "meta.json").read_text(encoding="utf-8"))
    assert meta["params"] == STD_PARAMS, f"{case}/meta.json params drifted from STD_PARAMS"
    assert meta["tier"] == "A"
    assert meta["comparison"] == {"mode": "exact", "float_tolerance": None}


def test_fixtures_not_imported_by_source():
    """Negative: no shipped module (gwas/, pages/, cli.py, annotation.py, app.py)
    references the golden fixtures. They are test-only by construction."""
    targets = []
    for pkg in ("gwas", "pages", "utils"):
        targets += list((REPO / pkg).rglob("*.py"))
    for mod in ("cli.py", "annotation.py", "app.py"):
        if (REPO / mod).exists():
            targets.append(REPO / mod)
    offenders = []
    for p in targets:
        text = p.read_text(encoding="utf-8", errors="ignore")
        if "tests.golden" in text or "tests/golden" in text:
            offenders.append(str(p.relative_to(REPO)))
    assert not offenders, f"shipped modules reference the golden fixtures: {offenders}"


def test_regenerate_refuses_without_full_intent(tmp_path):
    """``regenerate.py --case <name>`` with no other arguments exits non-zero and
    changes nothing. Run with PYTEST_CURRENT_TEST stripped so the argparse/flag
    guards (not just the under-pytest assert) are what block it."""
    before = {p: p.stat().st_mtime_ns for p in _tracked_files()}
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
    env["TRACE_GOLDEN_REGEN"] = "0"  # ensure the env guard is not satisfied either
    proc = subprocess.run(
        [sys.executable, str(GOLDEN_DIR / "regenerate.py"), "--case", "blocks_dense_ld"],
        cwd=str(REPO), env=env, capture_output=True, text=True,
    )
    assert proc.returncode != 0, "regenerate.py must refuse without --reason and the intent flags"
    after = {p: p.stat().st_mtime_ns for p in _tracked_files()}
    assert before == after, "regenerate.py changed files while refusing to run"
