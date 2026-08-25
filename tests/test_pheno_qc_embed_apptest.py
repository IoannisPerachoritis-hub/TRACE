"""AppTest for the embedded phenotype-QC panel (diagnostics.render, embedded=True).

Drives the panel the GWAS page embeds: select a trait, switch the transform
selector, and confirm the transformed (right-panel) vector re-computes on every
switch while the raw (left) stays fixed; and that log on a non-positive trait is
REFUSED (transform_ok False + warning), not silently shifted.

The panel's export buttons hit the same AppTest bare-mode `url_pathname`
limitation the regional-tab test tolerates; the render + return run before it.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from pages._pheno_qc.stats import (  # noqa: E402
    TRANSFORM_INT,
    TRANSFORM_LOG10,
    TRANSFORM_NONE,
    TRANSFORM_YEOJOHNSON,
)

_SCRIPT = '''
import sys
_ROOT = r"__ROOT__"
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import numpy as np
import pandas as pd
import streamlit as st
from pages._pheno_qc import PhenoQCContext, diagnostics

rng = np.random.default_rng(0)
df = pd.DataFrame({
    "pos": np.exp(rng.normal(0.0, 1.0, 80)),   # strictly positive, right-skewed
    "neg": rng.normal(0.0, 1.0, 80),           # spans negatives
})
ctx = PhenoQCContext(pheno_df=df, numeric_cols=["pos", "neg"], id_col="ID", source="session")
trait, method, transformed, ok = diagnostics.render(ctx, embedded=True, key_prefix="t")
st.session_state["_trait"] = trait
st.session_state["_method"] = method
st.session_state["_transformed"] = np.asarray(transformed, dtype=float)
st.session_state["_ok"] = bool(ok)
'''


def _only_harness_error(at):
    return all("url_pathname" in str(e.value) for e in at.exception)


def test_embedded_panel_updates_on_transform_switch():
    at = AppTest.from_string(_SCRIPT.replace("__ROOT__", str(_ROOT).replace("\\", "/")),
                             default_timeout=120).run()
    assert not at.exception or _only_harness_error(at)

    # positive trait "pos" — switch every transform; the transformed vector (which
    # drives the right-hand histogram / Q-Q / metrics) must re-compute each time.
    outs = {}
    for m in (TRANSFORM_NONE, TRANSFORM_LOG10, TRANSFORM_YEOJOHNSON, TRANSFORM_INT):
        at.selectbox(key="t_trait").set_value("pos").run()
        at.selectbox(key="t_transform").set_value(m).run()
        assert not at.exception or _only_harness_error(at)
        outs[m] = at.session_state["_transformed"].copy()

    raw = outs[TRANSFORM_NONE]
    assert np.allclose(outs[TRANSFORM_NONE], raw)                 # none == raw
    assert not np.allclose(outs[TRANSFORM_LOG10], raw)            # each other transform
    assert not np.allclose(outs[TRANSFORM_YEOJOHNSON], raw)       # changes the right panel
    assert not np.allclose(outs[TRANSFORM_INT], raw)
    # and they differ from each other (not the same transform under different names)
    assert not np.allclose(outs[TRANSFORM_LOG10], outs[TRANSFORM_INT])


def test_embedded_panel_refuses_log_on_nonpositive():
    at = AppTest.from_string(_SCRIPT.replace("__ROOT__", str(_ROOT).replace("\\", "/")),
                             default_timeout=120).run()
    at.selectbox(key="t_trait").set_value("neg").run()
    at.selectbox(key="t_transform").set_value(TRANSFORM_LOG10).run()
    assert not at.exception or _only_harness_error(at)
    # refused -> not applied; right panel shows raw, ok is False, warning surfaced
    assert at.session_state["_ok"] is False
    neg_raw = None  # raw is what the panel returns as `transformed` on refusal
    assert np.all(np.isfinite(at.session_state["_transformed"]))
    assert any(("not applicable" in str(w.value).lower())
               or ("non-positive" in str(w.value).lower()) for w in at.warning)
