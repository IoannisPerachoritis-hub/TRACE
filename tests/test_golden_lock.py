"""T-95 — GOLDEN_LOCK tripwire.

``GOLDEN_LOCK`` is a sha256 over the canonical concatenation of every
``expected_*.csv`` in ``tests/golden/``. Any hand-edit of a golden that does not
also update ``GOLDEN_LOCK`` (i.e. any edit not made through the guarded
``regenerate.py``) fails this test, so a deliberate regeneration always shows up
as a one-line diff on a file whose only purpose is to be noticed in review.
"""
from pathlib import Path

from tests.golden._canon import GOLDEN_DIR, golden_lock_digest


def test_golden_lock_matches_expected_tables():
    lock = (GOLDEN_DIR / "GOLDEN_LOCK").read_text(encoding="utf-8").strip()
    recomputed = golden_lock_digest()
    assert lock == recomputed, (
        "GOLDEN_LOCK does not match the current expected_*.csv contents. Either a "
        "golden was hand-edited without updating the lock, or a deliberate "
        "regeneration did not run tests/golden/regenerate.py (which rewrites the "
        "lock). Do not 'fix' this by editing GOLDEN_LOCK by hand."
    )


def test_golden_lock_is_a_bare_sha256():
    lock = (GOLDEN_DIR / "GOLDEN_LOCK").read_text(encoding="utf-8").strip()
    assert len(lock) == 64 and all(c in "0123456789abcdef" for c in lock)
