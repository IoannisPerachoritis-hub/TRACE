"""The pipeline's warnings (gwas.qc._log_or_warn) reach the CLI's log. Without a Streamlit page
they went to st.warning, which prints nothing outside a page, so the CLI showed none of them.
In the GUI they stay on the page."""
import logging

from gwas import qc as gqc


def test_warning_is_logged_without_a_streamlit_page(caplog):
    caplog.set_level(logging.WARNING, logger="gwas.qc")
    gqc._log_or_warn("12 multi-allelic records collapsed (kept highest-MAF allele per chr:pos).")
    assert any("12 multi-allelic records collapsed" in r.getMessage() for r in caplog.records)


def test_streamlit_page_shows_it_instead(monkeypatch, caplog):
    shown = []

    class _St:
        @staticmethod
        def warning(msg):
            shown.append(msg)
    monkeypatch.setattr(gqc, "st", _St)
    monkeypatch.setattr(gqc, "_in_streamlit_page", lambda: True)
    caplog.set_level(logging.WARNING, logger="gwas.qc")
    gqc._log_or_warn("partial overlap")
    assert shown == ["partial overlap"]
    assert not any("partial overlap" in r.getMessage() for r in caplog.records)


def test_no_page_context_in_a_plain_process():
    assert gqc._in_streamlit_page() is False
