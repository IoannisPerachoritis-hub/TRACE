"""The GWAS page's input boxes at function level (gwas.guitable: vcf_input, save_vcf_upload,
new_page_visit, covar_persist_decision, clear_inputs, release_upload, save_upload, stale_folders).

A page run is replayed as the page makes it: an upload arrives through the VCF box's callback, which
saves it and empties the box before the page runs; then the visit check before the boxes are drawn,
the VCF in use from the location box or the saved upload, then the covariate decision. A page
navigation is Streamlit dropping the page's widget states.
"""
import ast
import io

from pathlib import Path

import pytest

from gwas import guigwas
from gwas import guitable as gt

REPO = Path(__file__).resolve().parents[1]
EX1 = REPO / "tests" / "data" / "example_v1" / "example.vcf.gz"
EX2 = REPO / "examples" / "example.vcf.gz"


class _Upload(io.BytesIO):
    def __init__(self, data, name, file_id):
        super().__init__(data)
        self.name, self.size, self.file_id = name, len(data), file_id


def _page_run(state, work, location="", upload=None, covar=False):
    """One run of the page's input section; returns (VcfInput, covariate decision). An upload arrives
    through the VCF box's callback (gt.save_vcf_upload), which runs before the page: the box was drawn
    by an earlier run of the visit (its first run draws it, with nothing given), the callback saves
    the upload and empties the box, and the page runs once, on the saved file."""
    if upload is not None:
        if gt.VISIT_WIDGET_KEY not in state:        # the page is drawn first: the box exists
            _page_run(state, work, location)
        state[gt.box_key(state, "vcf_upload")] = upload                     # the browser's upload
        gt.save_vcf_upload(state, work, location)                           # the box's callback
    gt.new_page_visit(state)
    inp = gt.vcf_input(state, location, state.get(gt.box_key(state, "vcf_upload")), work)
    state[gt.VISIT_WIDGET_KEY] = "auto"             # the "Large datasets" radio, drawn this run
    if inp.error:
        return inp, None
    if inp.saved:                                   # saved by the page itself: the box is emptied
        gt.empty_upload_box(state, "vcf_upload")
        inp = gt.vcf_input(state, location, None, work)
    decision = gt.covar_persist_decision(covar, not inp.live, "_persist_covar" in state)
    if decision == "upload":
        state["_persist_covar"] = "covariates"      # the page loads and keeps the upload
    elif decision == "clear":
        state.pop("_persist_covar", None)           # the page drops the kept covariates
    return inp, decision


def _navigate(state):
    """Another page ran: Streamlit dropped this page's widget states (its boxes are empty)."""
    for k in (gt.VISIT_WIDGET_KEY, "vcf_location"):
        state.pop(k, None)


# ff56e5c's decision in each workflow (its call passed `vcf_file is None`; its VCF box held the file
# for the whole visit), as measured by running ff56e5c's own decision function in batch 13
FF56E5C = {"live, all three filled": "upload", "live, covariate removed": "clear",
           "navigation, nothing given again": "session", "navigation, only a new phenotype": "session",
           "navigation, only a new VCF": "clear"}


def test_the_covariate_decision_is_ff56e5cs_in_every_workflow(tmp_path):
    work = tmp_path / "work"
    got = {}
    try:
        for source in ("upload", "location"):
            state = {}

            def give(path, fid):
                return (dict(upload=_Upload(path.read_bytes(), path.name, fid)) if source == "upload"
                        else dict(location=str(path)))

            # a live session: the VCF given in this visit, the covariates uploaded, then removed
            _inp, got["live, all three filled"] = _page_run(state, work, covar=True, **give(EX1, "v1"))
            kept = give(EX1, "v1") if source == "location" else {}          # the location box keeps its text
            _inp, got["live, covariate removed"] = _page_run(state, work, covar=False, **kept)
            assert "_persist_covar" not in state                            # dropped ("clear")
            state["_persist_covar"] = "covariates"                          # kept from an earlier visit
            _navigate(state)
            _inp, got["navigation, nothing given again"] = _page_run(state, work)
            _navigate(state)
            _inp, got["navigation, only a new phenotype"] = _page_run(state, work)     # the box is the phenotype's
            assert state.get("_persist_covar") == "covariates"
            _navigate(state)
            _inp, got["navigation, only a new VCF"] = _page_run(state, work, **give(EX2, "v2"))
            assert got == FF56E5C, source
    finally:
        gt.remove_saved_uploads()
    # the page passes "the VCF was not given in this visit" as running_from_memory
    page = ast.parse((REPO / "pages" / "GWAS_analysis.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(page) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "_covar_persist_decision"]
    assert len(calls) == 1 and ast.unparse(calls[0].args[1]) == "not _vcf_in.live"


def test_a_saved_upload_leaves_the_servers_memory(tmp_path, monkeypatch):
    """After the copy into the work folder, Streamlit's record of the upload (the whole file, held by
    the server for the session) is removed, as Streamlit's chat input removes its files. The VCF box's
    callback does it before the page runs, and drops the box's own copy (its widget state) with it:
    the box is drawn empty and the page reads the saved file."""
    import types

    import streamlit.runtime.scriptrunner as sr
    from streamlit.proto.Common_pb2 import FileURLs
    from streamlit.runtime.memory_uploaded_file_manager import MemoryUploadedFileManager
    from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec

    mgr = MemoryUploadedFileManager("/_stcore/upload_file")
    data = EX1.read_bytes()
    mgr.add_file("session-1", UploadedFileRec(file_id="f-1", name="panel.vcf.gz", type="", data=data))
    monkeypatch.setattr(sr, "get_script_run_ctx",
                        lambda suppress_warning=False: types.SimpleNamespace(uploaded_file_mgr=mgr,
                                                                             session_id="session-1"))
    (rec,) = mgr.get_files("session-1", ["f-1"])
    upload = UploadedFile(rec, FileURLs())
    state = {}
    try:
        inp = gt.vcf_input(state, "", upload, tmp_path / "work")
        assert inp.saved and inp.live and not inp.error
        assert mgr.get_files("session-1", ["f-1"]) == []                      # the record is gone
        assert Path(state["_persist_vcf_path"]).read_bytes() == data          # the saved copy is used
        assert state["_vcf_source"] == "upload" and state["_persist_vcf_name"] == "panel.vcf.gz"
        mgr.add_file("session-1", UploadedFileRec(file_id="f-3", name="next.vcf.gz", type="", data=data))
        (rec3,) = mgr.get_files("session-1", ["f-3"])
        state["vcf_upload"] = UploadedFile(rec3, FileURLs())                   # the box, as the callback finds it
        assert gt.save_vcf_upload(state, tmp_path / "work").saved             # the box's callback
        assert mgr.get_files("session-1", ["f-3"]) == []                      # the record is gone
        assert "vcf_upload" not in state and gt.box_key(state, "vcf_upload") == "vcf_upload_1"   # emptied
        page = gt.vcf_input(state, "", state.get(gt.box_key(state, "vcf_upload")), tmp_path / "work")
        assert page.live and "next.vcf.gz" in page.caption and not page.saved
        assert Path(state["_persist_vcf_path"]).read_bytes() == data
    finally:
        gt.remove_saved_uploads()
    assert gt.release_upload(_Upload(b"x", "x.vcf", "f-2"))                   # a record not there: no error
    assert not gt.release_upload(object())


def test_after_clear_the_location_box_builds_no_table(tmp_path, monkeypatch):
    """Clear empties the location box (and gives the upload boxes new keys, and drops the trait
    selection), so the run after it has no VCF and no phenotype: the page's gate is closed and
    nothing is read. Without the reset, the location box's text would give the VCF again."""
    monkeypatch.setenv("TRACE_STREAMLIT_CACHE_DIR", str(tmp_path / "memo"))
    work = tmp_path / "work"
    state = {"vcf_location": str(EX1), "_persist_pheno": "pheno", "selected_traits_multiselect": ["T"]}
    inp, _d = _page_run(state, work, location=state["vcf_location"])
    assert inp.live and gt.inputs_kept(state)                               # the page would build the table
    gt.get_registry().get(gt.file_identity(EX1), "disk", EX1, work)

    without_reset = dict(state)
    gt.clear_all(without_reset, keys=guigwas.SESSION_KEYS, prefixes=guigwas.SESSION_PREFIXES)
    gt.vcf_input(without_reset, without_reset["vcf_location"], None, work)
    assert gt.vcf_path_ready(without_reset)                                 # the location gives it again

    gt.clear_inputs(state)                                                  # the button's callback
    gt.clear_all(state, keys=guigwas.SESSION_KEYS, prefixes=guigwas.SESSION_PREFIXES)
    assert state["vcf_location"] == "" and "selected_traits_multiselect" not in state
    assert [gt.box_key(state, b) for b in gt.UPLOAD_BOXES] == ["vcf_upload_1", "pheno_upload_1",
                                                                "covar_upload_1"]
    inp, _d = _page_run(state, work, location=state["vcf_location"])        # the run after Clear
    phe_file = None                                                         # a new, empty box
    assert not ((gt.vcf_path_ready(state) and phe_file) or gt.inputs_kept(state))   # the gate is closed
    assert not inp.live and not inp.caption
    assert gt.get_registry().keys() == []                                   # no table was built


def test_a_location_that_fails_the_check_stops_and_drops_the_earlier_location(tmp_path):
    work = tmp_path / "work"
    state = {}
    gt.vcf_input(state, str(EX1), None, work)
    assert state["_persist_vcf_path"] == str(EX1.resolve())
    for bad in (str(tmp_path / "typo.vcf.gz"), str(tmp_path / "panel.csv")):
        inp = gt.vcf_input(state, bad, None, work)
        assert inp.error and not inp.live
        assert not gt.vcf_path_ready(state) and "_vcf_source" not in state  # never the earlier file
    # an earlier upload is kept for when the box is emptied again; the page stops meanwhile
    try:
        gt.vcf_input(state, "", _Upload(EX2.read_bytes(), "two.vcf.gz", "u1"), work)
        saved = state["_persist_vcf_path"]
        inp = gt.vcf_input(state, str(tmp_path / "typo.vcf"), None, work)
        assert inp.error and state["_persist_vcf_path"] == saved
        inp = gt.vcf_input(state, "", None, work)
        assert not inp.error and "two.vcf.gz" in inp.caption
    finally:
        gt.remove_saved_uploads()


class _FailingUpload:
    """An upload whose copy fails after two pieces (as a full disk fails a write)."""

    name, size, file_id = "panel.vcf.gz", 10 ** 9, "f-full"

    def __init__(self):
        self.n = 0

    def seek(self, _):
        self.n = 0

    def read(self, k):
        self.n += 1
        if self.n > 2:
            raise OSError(28, "No space left on device")
        return b"x" * k


def test_a_failed_copy_leaves_nothing_and_is_not_tried_again(tmp_path):
    work = tmp_path / "work"
    with pytest.raises(OSError):
        gt.save_upload(_FailingUpload(), work)
    assert list(work.iterdir()) == [] and gt.saved_upload_folders() == []
    state = {}
    first = gt.vcf_input(state, "", _FailingUpload(), work)
    assert first.error.startswith(f"The upload cannot be saved in the work folder {work}")
    again = gt.vcf_input(state, "", _FailingUpload(), work)                 # the next rerun
    assert again.error == first.error and list(work.iterdir()) == []
    other = tmp_path / "other"                                              # another work folder: tried
    elsewhere = gt.vcf_input(state, "", _FailingUpload(), other)
    assert str(other) in elsewhere.error and list(other.iterdir()) == []
    assert gt.saved_upload_folders() == [] and not gt.vcf_path_ready(state)


def test_the_upload_is_saved_in_its_boxs_callback_with_no_rerun(tmp_path):
    """The VCF box's on_change callback saves a new upload before the page runs (save_vcf_upload), so
    the page never reruns for it: the rerun it made after a save dropped the states of the widgets
    drawn after the box (the QC sliders, the sidebar), and the run right after the save used their
    defaults. A typed location comes first (the upload stays in its box); a copy that fails stays in
    its box, for the page to show; an upload the page itself saves (in another work folder, after a
    failed copy) is freed the same way, with the box emptied and no rerun."""
    page = ast.parse((REPO / "pages" / "GWAS_analysis.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(page) if isinstance(n, ast.Call)]
    assert not [n for n in calls if ast.unparse(n.func) == "st.rerun"]                  # no rerun at all
    (box,) = [n for n in calls if ast.unparse(n.func) == "st.file_uploader" and "Genotype" in ast.unparse(n.args[0])]
    assert {k.arg: ast.unparse(k.value) for k in box.keywords}["on_change"] == "_save_vcf_upload"
    (callback,) = [n for n in page.body if isinstance(n, ast.FunctionDef) and n.name == "_save_vcf_upload"]
    assert "_gt.save_vcf_upload(" in ast.unparse(callback)
    (saved,) = [n for n in ast.walk(page) if isinstance(n, ast.If) and ast.unparse(n.test) == "_vcf_in.saved"]
    assert "_gt.empty_upload_box(st.session_state, 'vcf_upload')" in ast.unparse(saved)
    work = tmp_path / "work"
    try:
        held = _Upload(EX1.read_bytes(), "a.vcf.gz", "a1")
        state = {"vcf_upload": held}
        assert gt.save_vcf_upload(state, work, str(EX1)) is None and state["vcf_upload"] is held   # a location
        assert gt.save_vcf_upload({}, work) is None                                              # an empty box
        blocked = tmp_path / "blocked"
        blocked.write_text("a file where the work folder should be")
        failed = gt.save_vcf_upload(state, blocked)                                              # a failed copy
        assert failed.error and state["vcf_upload"] is held and gt.box_key(state, "vcf_upload") == "vcf_upload"
        assert gt.vcf_input(state, "", held, blocked).error == failed.error                      # the page shows it
        inp = gt.vcf_input(state, "", held, work)                       # another work folder: the page saves it
        assert inp.saved and inp.live
        gt.empty_upload_box(state, "vcf_upload")                        # ... and empties the box, no rerun
        assert "vcf_upload" not in state and gt.box_key(state, "vcf_upload") == "vcf_upload_1"
        after = gt.vcf_input(state, "", None, work)
        assert after.live and "a.vcf.gz" in after.caption and not after.saved
    finally:
        gt.remove_saved_uploads()


def test_the_page_names_saved_uploads_left_by_another_server(tmp_path):
    from gwas import storeplan

    work = tmp_path / "work"
    (work / "_trace_upload_gui_99999_deadbeef").mkdir(parents=True)
    (work / "_trace_upload_gui_99999_deadbeef" / "panel.vcf.gz").write_bytes(b"x" * 1000)
    (work / "_trace_geno_store_gui_99999_cafebabe").mkdir()
    (work / "_trace_geno_store_gui_99999_cafebabe" / "calls.i8").write_bytes(b"x" * 10)
    try:
        own, _ident = gt.save_upload(_Upload(b"y" * 50, "mine.vcf", "m1"), work)
        listed = {(d.name, b, k) for d, b, k in gt.stale_folders(work)}
        assert listed == {("_trace_upload_gui_99999_deadbeef", 1000, "upload"),
                          ("_trace_geno_store_gui_99999_cafebabe", 10, "table")}
        assert own.parent.name not in {n for n, _b, _k in listed}           # this server's own upload
        # the CLI's listing is unchanged: table folders only
        assert [d.name for d, _b in storeplan.stale_store_folders(work)] == ["_trace_geno_store_gui_99999_cafebabe"]
    finally:
        gt.remove_saved_uploads()


def test_an_upload_folder_that_cannot_be_deleted_stays_this_servers(tmp_path):
    path, _ident = gt.save_upload(_Upload(b"z" * 100, "held.vcf", "h1"), tmp_path / "work")
    fh = open(path, "rb")
    try:
        gt.remove_saved_uploads()
        if path.parent.exists():                     # Windows refuses to delete an open file
            assert path.parent in gt.saved_upload_folders()
            assert gt.stale_folders(tmp_path / "work") == []
    finally:
        fh.close()
        gt.remove_saved_uploads()
    assert not path.parent.exists() and gt.saved_upload_folders() == []


def test_a_new_visit_forgets_which_vcf_was_given_in_the_last_one():
    state = {gt.VCF_LIVE_KEY: "x", gt.VISIT_WIDGET_KEY: "auto"}
    assert not gt.new_page_visit(state) and state[gt.VCF_LIVE_KEY] == "x"   # a rerun of the visit
    del state[gt.VISIT_WIDGET_KEY]
    assert gt.new_page_visit(state) and gt.VCF_LIVE_KEY not in state
    assert gt.box_key({"_pheno_upload_widget": 2}, "pheno_upload") == "pheno_upload_2"
    assert gt.box_key({}, "covar_upload") == "covar_upload"


def test_with_the_location_box_emptied_the_earlier_location_is_named(tmp_path):
    """The earlier location stays in use (from the session, as ff56e5c's upload did when its box was
    emptied) and the caption names it; a VCF of unknown source gets no caption."""
    state = {}
    gt.vcf_input(state, str(EX1), None, tmp_path)
    inp = gt.vcf_input(state, "", None, tmp_path)
    assert not inp.error and not inp.live and str(EX1.resolve()) in inp.caption
    assert gt.vcf_input({"_persist_vcf_path": str(EX1)}, "", None, tmp_path).caption == ""
