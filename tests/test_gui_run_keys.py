"""What the GWAS page records with a run, and the keys of its session results (gwas.guigwas).

- The contract with the Post-GWAS page: M_eff is written with the table it belongs to
  (meff_identity), and a Run's results with the Run's table (gwas_run_identity); the per-block
  subsampling stability aggregates only the Post-GWAS page's blocks of this trait and this table.
- The subsampling result's key names the discovery threshold exactly and the number of workers.
- The "Current settings differ" notice covers every setting a Run reads.
"""
import ast
from pathlib import Path

import pandas as pd

from gwas import guigwas

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "pages" / "GWAS_analysis.py"


def _page():
    return ast.parse(PAGE.read_text(encoding="utf-8"))


def _session_key(node):
    """The literal key of ``st.session_state["key"]`` (or of its first subscript), else None."""
    while isinstance(node, ast.Subscript):
        value = node.value
        if (isinstance(value, ast.Attribute) and value.attr == "session_state"
                and isinstance(value.value, ast.Name) and value.value.id == "st"):
            return node.slice.value if isinstance(node.slice, ast.Constant) else None
        node = value
    return None


def _blocks(tree):
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                yield block


def _writes(block):
    return {(_session_key(t), ast.unparse(s.value)) for s in block if isinstance(s, ast.Assign)
            for t in s.targets if _session_key(t)}


def test_meff_and_the_runs_results_are_written_with_their_tables_identity():
    """KEY-1 and KEY-2 (the GWAS page's side): every block that writes meff_val also writes
    meff_identity = _table_id, and every block that writes gwas_results writes gwas_run_identity."""
    found = {"meff_val": 0, "gwas_results": 0}
    for block in _blocks(_page()):
        keys = {k for k, _v in _writes(block)}
        if "meff_val" in keys:
            found["meff_val"] += 1
            assert ("meff_identity", "_table_id") in _writes(block)
        if "gwas_results" in keys and any(isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Subscript)
                                          and isinstance(s.targets[0].value, ast.Subscript) for s in block):
            found["gwas_results"] += 1
            assert ("gwas_run_identity", "_table_id") in _writes(block)
    assert found == {"meff_val": 1, "gwas_results": 1}
    assert {"meff_identity", "gwas_run_identity"} <= set(guigwas.SESSION_KEYS)        # Clear drops them


def test_the_block_stability_uses_only_this_trait_and_tables_blocks():
    """KEY-3 (the GWAS page's side): ld_blocks_final when its key is (trait, table identity); else
    the automatic blocks of the trait when their signature names this table; else none."""
    mine, other = pd.DataFrame({"b": [1]}), pd.DataFrame({"b": [2]})
    sig = "gwashash::Tid-1::Rrows"
    assert guigwas.stability_blocks({}, "T", "id-1") is None
    state = {"ld_blocks_final": mine, "ld_blocks_final_key": ("T", "id-1")}
    assert guigwas.stability_blocks(state, "T", "id-1") is mine
    for key in (("T2", "id-1"), ("T", "id-2"), None):              # another trait, another table, no key
        state = {"ld_blocks_final": other, "ld_blocks_final_key": key}
        assert guigwas.stability_blocks(state, "T", "id-1") is None
        state.update(haplo_df_auto={"T": mine}, haplo_df_auto_hash={"T": sig})
        assert guigwas.stability_blocks(state, "T", "id-1") is mine
        assert guigwas.stability_blocks(state, "T", "id-3") is None              # found on another table
        assert guigwas.stability_blocks(state, "T3", "id-1") is None             # another trait's
    page = PAGE.read_text(encoding="utf-8")
    assert "_gg.stability_blocks(st.session_state, trait_col, _table_id)" in page
    assert 'st.session_state.get("ld_blocks_final", None)' not in page


def test_the_not_aggregated_caption_says_to_tick_subsampling_again():
    """The caption shown when the session's blocks are another trait's or table's lists every step: the
    subsampling box is drawn only on a Run and another page clears it, so after the Post-GWAS page the
    user clicks Run GWAS, ticks subsampling again and runs once more."""
    text = guigwas.BLOCKS_NOT_AGGREGATED
    assert text.startswith("Not aggregated: the Post-GWAS page's LD blocks belong to another trait")
    steps = ["open the Post-GWAS page", "click Run GWAS", "tick subsampling again", "click Run GWAS once more"]
    assert all(s in text for s in steps) and [text.index(s) for s in steps] == sorted(text.index(s) for s in steps)
    captions = [ast.unparse(n.args[0]) for n in ast.walk(_page()) if isinstance(n, ast.Call)
                and ast.unparse(n.func) == "st.caption" and n.args]
    assert "_gg.BLOCKS_NOT_AGGREGATED" in captions
    assert "Not aggregated" not in PAGE.read_text(encoding="utf-8")       # one text, in gwas.guigwas


def test_the_subsampling_key_names_the_threshold_exactly_and_the_workers():
    """KEY-4 and KEY-11: two thresholds that print alike with two digits, or another number of
    workers, are other results."""
    args = dict(trait="T", table_key="id", n_pcs=2, n_reps=50, sample_frac=0.8, seed=42, use_loco=False,
                shape=(60, 301))
    a = guigwas.subsampling_key(discovery_thresh=1.99485e-06, n_jobs=4, **args)
    b = guigwas.subsampling_key(discovery_thresh=2.00283e-06, n_jobs=4, **args)
    c = guigwas.subsampling_key(discovery_thresh=1.99485e-06, n_jobs=2, **args)
    assert len({a, b, c}) == 3
    assert a == guigwas.subsampling_key(discovery_thresh=1.99485e-06, n_jobs=4, **args)
    assert a.startswith("BOOT_GWAS::T::id::pcs2::reps50::frac0.80::thresh1.99485e-06::seed42::locoFalse")
    assert "_gg.subsampling_key(" in PAGE.read_text(encoding="utf-8")


def test_the_subsampling_settings_the_notice_reads():
    assert guigwas.subsampling_settings({}, 42, 4) == ("none",)
    assert guigwas.subsampling_settings({"run_subsampling_gwas": False, "boot_n_reps": 70}, 42, 4) == ("none",)
    ticked = {"run_subsampling_gwas": True}
    assert guigwas.subsampling_settings(ticked, 42, 4) == (50, 0.80, 1e-4, 42, 4, False)    # not drawn yet
    ticked.update(boot_n_reps=70, boot_disc_thresh=1e-5, boot_n_jobs=2)
    assert guigwas.subsampling_settings(ticked, 42, 4) == (70, 0.80, 1e-5, 42, 2, False)


def test_the_notice_covers_every_setting_the_run_reads():
    """KEY-7: the fingerprint the re-display compares names LOCO, the models, MLMM's and FarmCPU's
    PCs and settings, the rule and its threshold, and the subsampling settings."""
    tree = _page()
    (fp,) = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "_run_fp" for t in n.targets)]
    names = {n.id for n in ast.walk(fp.value) if isinstance(n, ast.Name)}
    assert {"use_loco", "model_choices", "n_pcs_mlmm", "n_pcs_farmcpu", "mlmm_p_enter", "mlmm_max_cof",
            "farmcpu_max_iter", "sig_rule", "custom_sig_thresh"} <= names
    assert "subsampling_settings" in ast.unparse(fp.value)
    (scan,) = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "_scan_fp" for t in n.targets)]
    assert "_run_fp" in {n.id for n in ast.walk(scan.value) if isinstance(n, ast.Name)}


def test_the_page_imports_no_cached_wrapper_that_hashes_a_table():
    """The page calls none of the st.cache_data / st.cache_resource wrappers whose key holds a whole
    table (Streamlit samples a large array): it does not import them either, so none is called by
    mistake. They stay in gwas/ for benchmarks/capture_golden.py and the tests."""
    imported = {a.asname or a.name for n in ast.walk(_page()) if isinstance(n, ast.ImportFrom) for a in n.names}
    hashing = {"build_mlmm_snpdata", "standardize_genotypes_for_mlmm_cached", "subset_snps_for_mlmm",
               "run_haplotype_block_gwas_cached", "compute_meff_li_ji", "load_vcf_cached"}
    assert not imported & hashing


def test_the_page_imports_no_haplotype_test():
    """The GWAS page runs no haplotype test of its own: the Post-GWAS page tests the blocks
    (gwas.guipost.run_haplotype_tests) and One-Click runs the pipeline's test. The PCs-only
    test's functions, imported here unused until batch 14, are not imported."""
    imported = {a.asname or a.name for n in ast.walk(_page()) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not imported & {"block_test_lm_with_pcs", "freedman_lane_perm_pvalue",
                           "run_haplotype_block_gwas", "kinship_block_test"}
