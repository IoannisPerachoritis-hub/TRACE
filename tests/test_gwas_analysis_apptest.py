"""AppTest coverage for the GWAS page (GUI).

The page is a large Streamlit script with no unit-testable seams; these tests
drive it via streamlit.testing.v1.AppTest, seeding session_state to exercise the
run-from-memory path (AppTest cannot perform real file uploads). The deep display
block hits an AppTest bare-mode limitation (a `url_pathname` KeyError) that does
NOT occur in the real app; the scan + result columns run before it, so tests
assert on the produced state and tolerate only that specific harness error.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

_PAGE = str(_ROOT / "pages" / "GWAS_analysis.py")


def _synth(n=50, per=40, chroms=("1", "2", "3"), seed=0):
    rng = np.random.default_rng(seed)
    samples = [f"S{i:03d}" for i in range(n)]
    lines = ["##fileformat=VCFv4.2",
             '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">',
             "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples)]
    for ch in chroms:
        for j in range(per):
            gt = ["0/0" if g == 0 else "0/1" if g == 1 else "1/1"
                  for g in rng.binomial(2, 0.3, n)]
            lines.append(f"{ch}\t{(j+1)*1000}\tchr{ch}_{j}\tA\tT\t.\tPASS\t.\tGT\t" + "\t".join(gt))
    vcf_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    pheno = pd.DataFrame({"Accession": samples,
                          "TestTrait": np.random.default_rng(seed + 1).normal(10, 2, n)})
    return vcf_bytes, pheno, samples


def _seed_from_memory(at, run=False, extra=None):
    """Seed the persisted inputs so the page enters the run block from memory."""
    vcf_bytes, pheno, samples = _synth()
    at.session_state["_persist_vcf_bytes"] = vcf_bytes
    at.session_state["_persist_is_gz"] = False
    at.session_state["_persist_pheno"] = pheno
    at.session_state["selected_traits_multiselect"] = ["TestTrait"]
    for k, v in (extra or {}).items():
        at.session_state[k] = v
    if run:
        at.session_state["gwas_triggered"] = True
        at.session_state["_last_gwas_trait"] = "TestTrait"
    return samples


def _only_harness_error(at):
    return all("url_pathname" in str(e.value) for e in at.exception)


def test_page_loads_cold():
    at = AppTest.from_file(_PAGE, default_timeout=120)
    at.run()
    assert not at.exception


def test_run_from_memory_produces_results():
    """T-09 regression: with the uploaded file gone but inputs persisted, a Run
    proceeds from memory and yields gwas_df (no re-upload)."""
    at = AppTest.from_file(_PAGE, default_timeout=300)
    _seed_from_memory(at, run=True)
    at.run()
    assert _only_harness_error(at)
    assert "gwas_df" in at.session_state and len(at.session_state["gwas_df"]) > 0


def test_custom_sig_thresh_runs_from_memory():
    """The Custom-p-value path (selectbox -> number_input -> Significant_Custom
    column -> active threshold) runs end-to-end without a real error. Column
    correctness is covered by the significance unit tests (rule_from_streamlit /
    SignificanceRule custom)."""
    at = AppTest.from_file(_PAGE, default_timeout=300)
    _seed_from_memory(at, run=True, extra={
        "sig_rule_select": "Custom p-value",
        "custom_sig_thresh_input": 5e-8,
    })
    at.run()
    assert _only_harness_error(at)
    assert "gwas_df" in at.session_state and len(at.session_state["gwas_df"]) > 0
