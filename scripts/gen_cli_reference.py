"""Generate docs/cli_reference.md from the CLI argparse parser.

Run to refresh the reference after changing cli.py:
    python scripts/gen_cli_reference.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from cli import _build_parser  # noqa: E402


def main():
    parser = _build_parser()
    parser.prog = "trace-gwas"
    help_text = parser.format_help()
    out = _ROOT / "docs" / "cli_reference.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        "# TRACE CLI reference\n\n"
        "Auto-generated from `cli.py` by `python scripts/gen_cli_reference.py`. "
        "Do not edit by hand — re-run the generator after changing the parser.\n\n"
        "```text\n" + help_text.rstrip() + "\n```\n",
        encoding="utf-8",
    )
    print(f"wrote {out} ({len(help_text)} chars)")


if __name__ == "__main__":
    main()
