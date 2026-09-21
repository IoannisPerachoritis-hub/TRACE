"""AppTest guard: One-Click Full Analysis runs one trait per run.

The per-trait loop in the one-click pipeline resolves its heavy inputs through
session keys registered ONCE above the loop from ``selected_traits[0]``
(``y_key`` / ``geno_key`` / ``K0_key`` / ``pheno_reader_key``). ``run_gwas_cached``
regresses the phenotype reader those keys resolve to -- not its ``y`` argument,
which ``_run_gwas_impl`` casts once and never reads again -- so every trait after
the first was scanned against the first trait's phenotype and genotypes and then
reported under its own name. Measured on planted causals: traits 2 and 3 came
back with p-values bit-identical to trait 1 and named trait 1's causal SNP, on
the wrong chromosome, with no error.

Until the loop registers those keys per trait, the pipeline refuses to start on
more than one trait rather than computing on the wrong phenotype. This test pins
that refusal, and pins that a single-trait run still reaches the GWAS call.

``run_gwas_cached`` is replaced with a counter: the refusal must not reach it,
and the single-trait control must. That keeps the test independent of whether
the rest of the pipeline completes, and fast.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402
import gwas.models as _models  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

_PAGE = str(_ROOT / "pages" / "GWAS_analysis.py")
_TRAITS = ["TraitA", "TraitB", "TraitC"]
_REFUSAL = "runs one trait at a time"

_CALLS = []


def _counting_run_gwas_cached(geno_imputed, y, pcs_full, n_pcs, sid, positions,
                              chroms, chroms_num, iid, _K0, _K_by_chr,
                              _pheno_reader_key, trait_name=None,
                              user_covar=None, user_covar_names=None):
    """Stand-in for gwas.models.run_gwas_cached: records the call, returns a
    null frame (uniform p, so nothing is significant and the post-GWAS stages
    short-circuit) instead of running FastLMM."""
    _CALLS.append(trait_name)
    _sid = np.asarray(sid).astype(str)
    return pd.DataFrame({
        "SNP": _sid,
        "Chr": np.asarray(chroms).astype(str),
        "ChrNum": np.asarray(chroms_num).astype(int),
        "Pos": np.asarray(positions).astype(int),
        "PValue": np.full(len(_sid), 0.5),
    })


@pytest.fixture
def calls(monkeypatch):
    _CALLS.clear()
    monkeypatch.setattr(_models, "run_gwas_cached", _counting_run_gwas_cached)
    # memoised frames persist across AppTest runs in-process; a stale hit would
    # let an arm pass without executing anything
    st.cache_data.clear()
    yield _CALLS
    st.cache_data.clear()


def _synth(n=40, per=20, chroms=("1", "2"), seed=0):
    """Small in-memory VCF + a 3-trait phenotype (no missing values)."""
    rng = np.random.default_rng(seed)
    samples = [f"S{i:03d}" for i in range(n)]
    lines = [
        "##fileformat=VCFv4.2",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples),
    ]
    for ch in chroms:
        for j in range(per):
            gt = ["0/0" if g == 0 else "0/1" if g == 1 else "1/1"
                  for g in rng.binomial(2, 0.3, n)]
            lines.append(
                f"{ch}\t{(j + 1) * 1000}\tchr{ch}_{j}\tA\tT\t.\tPASS\t.\tGT\t"
                + "\t".join(gt)
            )
    pheno = pd.DataFrame({"Accession": samples})
    for k, t in enumerate(_TRAITS):
        pheno[t] = rng.normal(10.0 + 10.0 * k, 2.0, n)
    return ("\n".join(lines) + "\n").encode("utf-8"), pheno


def _run_one_click(traits):
    vcf_bytes, pheno = _synth()
    at = AppTest.from_file(_PAGE, default_timeout=600)
    at.session_state["_persist_vcf_bytes"] = vcf_bytes
    at.session_state["_persist_is_gz"] = False
    at.session_state["_persist_pheno"] = pheno
    at.session_state["selected_traits_multiselect"] = list(traits)
    # keep the run cheap: MLM only, no subsampling, no heatmaps, no gene model
    at.session_state["pipe_models"] = []
    at.session_state["pipe_subsampling"] = False
    at.session_state["pipe_generate_ld_heatmaps"] = False
    at.run()
    at.button(key="run_full_pipeline").click().run()
    return at


def test_more_than_one_trait_is_refused_before_any_gwas_runs(calls):
    at = _run_one_click(_TRAITS)

    errs = " ".join(e.value for e in at.error)
    assert _REFUSAL in errs, f"no refusal message; errors were: {errs!r}"
    assert "3 are selected" in errs, "the message does not say how many are selected"
    assert "cli.py --trait" in errs, "the message does not name the alternative"

    # the decisive half: nothing was computed on the wrong phenotype
    assert calls == [], (
        f"the pipeline reached the GWAS call for {calls} despite the refusal; "
        "traits after the first would be scanned against the first trait's phenotype"
    )


def test_single_trait_still_runs(calls):
    at = _run_one_click(["TraitB"])

    errs = " ".join(e.value for e in at.error)
    assert _REFUSAL not in errs, f"the guard fired on a single trait: {errs!r}"

    # the pipeline reached the GWAS call, exactly once, for the selected trait
    assert calls == ["TraitB"], f"expected one GWAS call for TraitB, got {calls}"
