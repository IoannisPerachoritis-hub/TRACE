"""Documentation <-> CLI-parser parity + markdown-link resolution (WO-AVAIL-01 / A9).

Six of the nine defects in the 2026-09 Availability audit were the same bug: a
flag or feature named in prose that ``cli.py``'s parser does not have (a stale
README parity table, a phantom ``--ld-merge-mode`` in ``docs/cli_reference.md``,
``--pc-diagnostics-parallel`` / ``--sig-rule`` / ``--genes`` / etc.). That class
is mechanically checkable, so this test guards against its recurrence.

``test_documented_flags_exist_in_parser`` extracts every ``--flag`` token from
the user-facing docs and asserts each one is a real option in the parser.
``test_relative_markdown_links_resolve`` asserts every relative markdown link in
README + docs/ points at a file that exists.

Exemptions (documented, deliberate):
  * ``--help`` / ``-h`` (argparse-provided).
  * Flags on a line that invokes a third-party tool (docker / pip / apt-get /
    useradd / streamlit / uv / conda / ...) -- those are that tool's flags shown
    in an install/run example, not trace-gwas flags. The trace-gwas flag
    documentation (parity table, help prose) never lives on such lines.
  * Flags under a CHANGELOG ``### Removed`` heading (naming a removed flag there
    is correct).
  * Tokens ending in ``-`` -- a line-wrap artifact of the generated help text
    (e.g. ``--ld-`` where ``--ld-top-n`` wrapped) or a ``/-x`` range fragment.
"""
from __future__ import annotations

import re
from pathlib import Path

from cli import _build_parser

REPO = Path(__file__).resolve().parents[1]

DOC_FILES = [REPO / "README.md", REPO / "CHANGELOG.md", REPO / "pages" / "z_Help.py"]
DOC_FILES += sorted((REPO / "docs").glob("*.md"))

LINK_FILES = [REPO / "README.md"] + sorted((REPO / "docs").glob("*.md"))

_FLAG_RE = re.compile(r"--[a-z][a-z0-9-]*")
_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

# Lines invoking a third-party tool -> their flags are exempt.
_THIRD_PARTY = (
    "docker", "pip install", "pip3", "apt-get", "apt ", "useradd",
    "streamlit run", "uv pip", "uv venv", "uv ", "python -m pip",
    "conda", "chmod", "chown", "gunicorn", "run ", "--no-cache-dir",
)


def _valid_flags() -> set[str]:
    parser = _build_parser()
    valid = {opt for a in parser._actions for opt in a.option_strings}
    valid |= {"--help", "-h"}
    return valid


def _iter_doc_flag_tokens():
    """Yield (file, lineno, flag, rawline) for every candidate trace-gwas flag."""
    for f in DOC_FILES:
        text = f.read_text(encoding="utf-8")
        in_removed = False  # CHANGELOG "### Removed" tracking
        for i, line in enumerate(text.splitlines(), 1):
            if f.name == "CHANGELOG.md":
                s = line.strip()
                if s.startswith("### "):
                    in_removed = s.lower() == "### removed"
                elif s.startswith("## "):
                    in_removed = False
            if in_removed:
                continue
            low = line.lower()
            if any(m in low for m in _THIRD_PARTY):
                continue
            for flag in _FLAG_RE.findall(line):
                if flag.endswith("-"):
                    continue
                yield f, i, flag, line.strip()


def test_documented_flags_exist_in_parser():
    valid = _valid_flags()
    # sanity: we actually loaded the real parser (not a vacuous empty set)
    assert "--vcf" in valid and "--sig-thresh" in valid, "parser did not load its flags"

    violations = [
        (str(f.relative_to(REPO)), i, flag, raw)
        for f, i, flag, raw in _iter_doc_flag_tokens()
        if flag not in valid
    ]
    if violations:
        lines = "\n".join(
            f"  {path}:{ln}  {flag}\n      | {raw[:120]}" for path, ln, flag, raw in violations
        )
        raise AssertionError(
            f"{len(violations)} documented --flag(s) are not in cli._build_parser().\n"
            f"Fix the doc, or (if a real flag was added) the parser:\n{lines}"
        )


def test_relative_markdown_links_resolve():
    bad = []
    for f in LINK_FILES:
        base = f.parent
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            for target in _LINK_RE.findall(line):
                t = target.strip()
                if t.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                path_part = t.split("#", 1)[0].split("?", 1)[0]
                if not path_part:
                    continue
                if not (base / path_part).resolve().exists():
                    bad.append(f"  {f.relative_to(REPO)}:{i}  -> {target}")
    assert not bad, "unresolved relative markdown link(s):\n" + "\n".join(bad)
