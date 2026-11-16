"""The GWAS page's plain stop messages, at function level.

- When the page stops but shows the last completed run instead (its _rehydrate_or_stop, taken from
  the page's source), every caller's message is followed by "Showing your last completed run below
  instead." as a sentence of its own: no ".;" in any of them.
"""
import ast
from pathlib import Path

import pandas as pd
import pytest

from gwas import guigwas, guitable, storeplan
from gwas.covariates import covariate_drop_message

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "pages" / "GWAS_analysis.py"
EX1 = REPO / "tests" / "data" / "example_v1" / "example.vcf.gz"
EX1_PHENO = REPO / "tests" / "data" / "example_v1" / "example_pheno.csv"
NOTE = "Showing your last completed run below instead."


class _Stopped(Exception):
    pass


class _St:
    """The st calls _rehydrate_or_stop makes, recorded."""

    def __init__(self):
        self.shown = []

    def warning(self, text):
        self.shown.append(("warning", text))

    def error(self, text):
        self.shown.append(("error", text))

    def info(self, text):
        self.shown.append(("info", text))

    def stop(self):
        raise _Stopped


def _page():
    return ast.parse(PAGE.read_text(encoding="utf-8"))


def _rehydrate_or_stop(st, persisted):
    """The page's own _rehydrate_or_stop, run with ``st`` and a session that has (or has not) a
    completed run to show."""
    fn = next(n for n in _page().body if isinstance(n, ast.FunctionDef) and n.name == "_rehydrate_or_stop")
    ns = {"st": st, "_gg": guigwas, "_has_persisted_gwas": lambda: persisted,
          "_render_last_run_results": lambda: st.shown.append(("last run", ""))}
    exec(compile(ast.Module([fn], type_ignores=[]), str(PAGE), "exec"), ns)
    return ns["_rehydrate_or_stop"]


def _phenotype_qc_errors():
    """What the page's setup records for traits that fail phenotype QC (str of the ValueError)."""
    pheno = pd.read_csv(EX1_PHENO, sep=None, engine="python", encoding="utf-8-sig")
    pheno = pheno.set_index(pheno.columns[0])
    pheno.index = pheno.index.astype(str).str.strip()
    pheno["Const"] = 1.0
    pheno["Cat"] = "a"
    reg = guitable.TableRegistry()
    errors = {}
    try:
        parsed = reg.get(guitable.file_identity(EX1), "memory", EX1)
        for trait in ("Const", "Cat"):
            with pytest.raises(ValueError) as exc:
                guigwas.preprocess({}, trait, parsed, pheno.copy(), trait, 0.05, 0.10, "none", True, 0.20, 5)
            errors[trait] = str(exc.value)
    finally:
        reg.close_all()
    return errors


def _stop_messages(tmp_path, monkeypatch):
    """One message of every kind the page passes to _rehydrate_or_stop."""
    work = tmp_path / "w"
    vcf = str(EX1)
    plans = [storeplan.plan(vcf, choice="memory", store_base=work, available=1),
             storeplan.plan(vcf, choice="disk", store_base=work, free_disk=1),
             storeplan.plan(vcf, choice="auto", store_base=work, impute="ldknni", available=1),
             storeplan.plan(vcf, choice="disk", store_base=work, impute="ldknni"),
             storeplan.plan(vcf, choice="disk", store_base=work, export_qc=True, one_piece_cells=1)]
    assert [p.stop_kind for p in plans] == ["memory", "disk", "ldknni", "ldknni", "export_qc"]
    messages = [guitable.page_stop(p, work) for p in plans]
    ident = guitable.file_identity(EX1)
    gui = dict(estimate=guitable.gui_estimate_mib)
    checks = [guigwas.build_check(vcf, storeplan.plan(vcf, choice=c, store_base=work,
                                                      models=("mlm", "mlmm", "farmcpu"), available=64 * 2**30,
                                                      free_disk=100 * 2**30, **gui), work, ident, **gui)
              for c in ("auto", "memory", "disk")]
    monkeypatch.setattr(storeplan, "available_memory_bytes", lambda: 2**20)
    monkeypatch.setattr(storeplan, "free_disk_bytes", lambda folder: 2**20)
    for check in checks:
        with pytest.raises(guigwas.PlanStop) as stop:
            check()
        messages.append(str(stop.value))
    messages += [
        "Could not read phenotype file: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte",
        "Could not parse phenotype file: Error tokenizing data. C error: Expected 3 fields in line 5, saw 4\n",
        "Select at least one numeric trait to proceed.",
        guitable.UPLOAD_REMOVED_MESSAGE,
        guitable.vcf_gone_message(str(tmp_path / "moved" / "panel.vcf.gz")),
        covariate_drop_message(["S1", "S2"], 2) + " No sample remains: check that the covariate file's "
        "sample IDs match the VCF's.",
        "Only 5 samples overlap between genotype and phenotype (40 phenotype samples). Check that sample "
        "IDs match between files.",
    ]
    # the setup's stop when no selected trait passes phenotype QC: the page's own expression
    (call,) = [n for n in ast.walk(_page()) if isinstance(n, ast.Call) and ast.unparse(n.func) == "_rehydrate_or_stop"
               and n.args and "No selected trait can be analysed" in ast.unparse(n.args[0])]
    failed = _phenotype_qc_errors()
    no_trait = eval(compile(ast.Expression(call.args[0]), str(PAGE), "eval"),
                    {"_gg": guigwas, "_setup_failed": failed})
    messages.append(no_trait)
    return messages, (no_trait, failed)


def test_the_last_run_note_is_a_sentence_of_its_own_after_every_stop(tmp_path, monkeypatch):
    """The round 2 re-check's q2 e: the note was joined with "; ", which gave "...upload it again.;
    showing your last completed run below instead." and the same after every message that ends with
    a full stop (the page's plan stops, the build check's, a VCF gone from its location). Now each
    message is kept as it is, ends as a sentence, and the note follows as a sentence of its own.
    Without a run to show, the message is shown alone, at its level. The setup's stop when no
    selected trait passes phenotype QC names each trait's error as a sentence of its own too."""
    messages, (no_trait, failed) = _stop_messages(tmp_path, monkeypatch)
    for msg in messages:
        text = msg.strip()
        st = _St()
        with pytest.raises(_Stopped):
            _rehydrate_or_stop(st, persisted=True)(msg, level="warning")
        (kind, shown), last = st.shown
        assert kind == "warning" and last == ("last run", "")
        assert shown == f"{text}{'' if text.endswith('.') else '.'} {NOTE}", shown
        assert ".;" not in shown and "; showing" not in shown
        st = _St()
        with pytest.raises(_Stopped):
            _rehydrate_or_stop(st, persisted=False)(msg, level="info")
        assert st.shown == [("info", msg)]
    st = _St()
    with pytest.raises(_Stopped):
        _rehydrate_or_stop(st, persisted=True)()                    # no message: the last run alone
    assert st.shown == [("last run", "")]
    assert ".;" not in no_trait and all(f"{t}: {e}" in no_trait for t, e in failed.items())
    assert no_trait == "No selected trait can be analysed: " + " ".join(f"{t}: {e}" for t, e in failed.items())
    assert guigwas.as_sentence("Done!") == "Done!" and guigwas.as_sentence('It said "stop."') == 'It said "stop."'
    assert guigwas.as_sentence("(see below)") == "(see below)." and guigwas.as_sentence("") == ""
