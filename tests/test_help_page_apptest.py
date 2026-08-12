"""AppTest coverage for the Help page split (GUI brief, Task 4a).

Task 4a trims z_Help.py: Quick Start becomes a pointer to the README (one source
of truth), Output Format is condensed, and the high-value "Interpreting Results
for Breeding" section is kept in full. The CSV Column Glossary is ALSO kept — the
brief proposed replacing it with a link to docs/outputs.md, but that file
explicitly delegates the v1.0.1 GWAS/LD/haplotype/subsampling/consensus columns
back to this glossary, so deleting it would lose documentation and create a
circular reference.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

_HELP = str(_ROOT / "pages" / "z_Help.py")


def test_help_page_keeps_interpretation_and_glossary():
    at = AppTest.from_file(_HELP, default_timeout=60)
    at.run()
    assert not at.exception
    headers = [h.value for h in at.header]
    # the highest-value text is kept in full
    assert "Interpreting Results for Breeding" in headers
    # the column glossary is kept (docs/outputs.md points back to it)
    assert "CSV Column Glossary" in headers


def test_help_page_quick_start_points_to_readme():
    at = AppTest.from_file(_HELP, default_timeout=60)
    at.run()
    assert not at.exception
    # Quick Start is now a pointer to the README, not a duplicated walkthrough
    assert any("README" in m.value for m in at.markdown)


def test_help_significance_section_is_neutral():
    """Task B1: the significance-threshold guidance is a neutral comparison — no
    evidence-free 'recommended'/'best balance', and it names Bonferroni as the
    shipped default (which the M_eff '(recommended)' text contradicted)."""
    at = AppTest.from_file(_HELP, default_timeout=60)
    at.run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "M_eff (recommended)" not in md, "M_eff still tagged '(recommended)'"
    assert "best balance" not in md, "'strikes the best balance' claim still present"
    assert "The default is Bonferroni" in md, "Bonferroni default not stated"
