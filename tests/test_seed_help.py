"""--seed says what it covers: the subsampling draws. Haplotype permutations are seeded per
block from the block's position and the trait (gwas.haplotype, stable_seed), so their p-values
are reproducible and independent of the block order, and do not depend on --seed."""
from pathlib import Path

import cli

REPO = Path(__file__).resolve().parents[1]


def _seed_help():
    p = cli._build_parser()
    for a in p._actions:
        if "--seed" in a.option_strings:
            return a.help
    raise AssertionError("no --seed option")


def test_seed_help_names_subsampling_only():
    h = _seed_help()
    assert "subsampling" in h
    assert "permutation" not in h.split("(default")[0].replace("permutation p-values", "")
    assert "seeded per block" in h


def test_docs_say_the_same():
    tut = (REPO / "docs" / "tutorial.md").read_text(encoding="utf-8")
    row = [l for l in tut.splitlines() if l.startswith("| `--seed`")][0]
    assert "subsampling" in row and "permutation" not in row.replace("permutations are seeded per block", "")
    ref = (REPO / "docs" / "cli_reference.md").read_text(encoding="utf-8")
    assert "subsampling and permutation" not in ref
