"""AppTest coverage for the landing-page workflow map (GUI brief, Task 3).

The landing page (`app.py`) previously had zero in-app navigation links — only
prose pointing at the sidebar. Task 3 adds a three-step workflow map with real
`st.page_link` calls plus a read-only "you have results" hint.

Harness limitation (why this asserts structure, not link elements): `st.page_link`
resolves its target through `PagesManager.get_pages()`, which returns a default
page dict lacking `url_pathname` on every AppTest run after the first in a
process — so `st.page_link` raises `KeyError: 'url_pathname'` there. app.py wraps
each link in `_safe_page_link` (try/except) so the page renders regardless; this
test therefore asserts the Workflow SECTION and the session hint render, not the
guarded link elements (which AppTest cannot reliably surface across a multi-test
run). The links themselves are simple `st.page_link` calls, correct in the real
multipage app.

Filename note — sorts LAST on purpose: `AppTest.from_file(app.py)` touches
Streamlit's global page state, which can break a later `from_string` tab test; the
guard + last-ordering keep both this test and the tab tests green.
"""
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

_APP = str(_ROOT / "app.py")


def test_landing_page_renders_workflow_section():
    at = AppTest.from_file(_APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert any("Workflow" in m.value for m in at.markdown), \
        "landing page missing the Workflow section header"


def test_landing_page_shows_results_hint_when_gwas_present():
    at = AppTest.from_file(_APP, default_timeout=60)
    at.session_state["gwas_df"] = pd.DataFrame({"SNP": ["s1"], "PValue": [0.01]})
    at.run()
    assert not at.exception
    assert any("GWAS results loaded in this session" in i.value for i in at.info)


def test_landing_page_no_results_hint_when_empty():
    at = AppTest.from_file(_APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert not any("GWAS results loaded in this session" in i.value for i in at.info)
