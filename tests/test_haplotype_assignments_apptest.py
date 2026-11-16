"""Sample-to-haplotype assignments in the Post-GWAS LD Blocks tab (reviewer point 2-INT.10).

The CLI, the GWAS page (detailed run, then One-Click) and the Post-GWAS page run for
real on the bundled example, Trait2, with CC-8's LD settings (r2 0.1, flank 200 kb),
MLM only and raw phenotype values, nothing stubbed. st.download_button is wrapped only
to record what each button offers.

A. The block panel's MLG groups are the haplotype test's: each block's in-test rows
   give the samples and labels the panel shows (its stored hap_tables_vis).
B. The all-blocks download equals the CLI ZIP member and the One-Click ZIP member,
   byte for byte; each block's download is that block's rows of it (header plus its
   lines) under haplotype_assignments_block_<chr>_<start>_<end>_<trait>.csv.
The old "Download per-sample MLG ledger (CSV)" button is gone.
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import cli  # noqa: E402
import gwas.ld as _ld  # noqa: E402
import gwas.pipeline as _pl  # noqa: E402
import pages._ld_tabs.tab_genome_wide as _tgw  # noqa: E402

# over about 30 s: run once per batch, not per task (-m 'not slow_gui')
pytestmark = pytest.mark.slow_gui

_VCF = _ROOT / "tests" / "data" / "example_v1" / "example.vcf.gz"      # the first example
_PHENO = _ROOT / "tests" / "data" / "example_v1" / "example_pheno.csv"
_TRAIT = "Trait2"
_LD_R2, _FLANK_KB, _HAP_PERMS = 0.1, 200, 100
_MEMBER = f"tables/Haplotype_assignments_{_TRAIT}.csv"
_BLOCK_PICK = "Select LD block for phenotype-haplotype visualization:"
_PER_BLOCK = "Download sample-to-haplotype assignments (CSV)"
_ALL_BLOCKS = "Download assignments for all blocks (CSV)"
_OLD_LEDGER = "Download per-sample MLG ledger (CSV)"

# Other AppTest files assign these module attributes for good; take the real ones at
# collection time and set them explicitly (as test_ld_heatmap_same_window_apptest does).
_REAL = {
    "find": _ld.find_ld_clusters_genomewide,
    "blockvis": getattr(_tgw, "_pristine_render_block_visualization",
                        _tgw._render_block_visualization),
    "gallery": getattr(_tgw, "_pristine_render_lead_snp_gallery",
                       _tgw._render_lead_snp_gallery),
    "run_traits": _pl.run_traits,
    "download": st.download_button,
}


def _ok(at, where):
    assert all("url_pathname" in str(e.value) for e in at.exception), (
        f"{where}: " + "; ".join(str(e.value) for e in at.exception))


def _carried_state(at):
    """What a page switch keeps: user keys (not the GWAS page's widget-bound keys)."""
    widget_keys = set(at.session_state._state._key_id_mapper.id_key_mapping.values())
    return {k: v for k, v in at.session_state.filtered_state.items() if k not in widget_keys}


@pytest.fixture(scope="module")
def sites(tmp_path_factory):
    downloads, zips = [], {}

    def _download(label, data, *a, **k):
        downloads.append({"label": label, "data": data, "file_name": k.get("file_name"),
                          "key": k.get("key")})
        return _REAL["download"](label, data, *a, **k)

    def _run_traits(parsed, pheno_df, traits, options, on_start=None, progress=None):
        for trait, res, err in _REAL["run_traits"](parsed, pheno_df, traits, options,
                                                   on_start=on_start, progress=progress):
            if res is not None:
                zips[trait] = res.zip_bytes
            yield trait, res, err

    out = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_ld, "find_ld_clusters_genomewide", _REAL["find"])
        mp.setattr(_tgw, "find_ld_clusters_genomewide", _REAL["find"])
        mp.setattr(_tgw, "_render_block_visualization", _REAL["blockvis"])
        mp.setattr(_tgw, "_render_lead_snp_gallery", _REAL["gallery"])
        mp.setattr(_pl, "run_traits", _run_traits)
        st.cache_data.clear()
        # the CLI
        cli_dir = tmp_path_factory.mktemp("assign_cli")
        cli.run_pipeline(cli._build_parser().parse_args([
            "--vcf", str(_VCF), "--pheno", str(_PHENO), "--trait", _TRAIT,
            "--output", str(cli_dir), "--ld-r2", str(_LD_R2),
            "--ld-flank-kb", str(_FLANK_KB), "--hap-perms", str(_HAP_PERMS)]))
        (cli_zip,) = list(cli_dir.glob("*.zip"))
        with zipfile.ZipFile(cli_zip) as zf:
            out["cli"] = zf.read(_MEMBER)
        # the GWAS page: detailed run (publishes what Post-GWAS reads), then One-Click
        at = AppTest.from_file(str(_ROOT / "pages" / "GWAS_analysis.py"), default_timeout=900)
        ss = at.session_state
        ss["_persist_vcf_path"] = str(_VCF)
        ss["_persist_pheno"] = pd.read_csv(_PHENO, sep=None, engine="python", encoding="utf-8-sig")
        ss["selected_traits_multiselect"] = [_TRAIT]
        for key, value in {"pipe_models": [], "pipe_subsampling": False,
                           "pipe_generate_ld_heatmaps": False, "pipe_ld_r2": float(_LD_R2),
                           "pipe_ld_flank_mode": "Manual (fixed kb)",
                           "pipe_ld_flank": int(_FLANK_KB), "pipe_hap_perms": int(_HAP_PERMS)}.items():
            ss[key] = value
        at.run()
        _ok(at, "GWAS page")
        [w for w in at.sidebar.multiselect if w.label == "Additional GWAS models"][0].set_value([])
        [b for b in at.button if b.label == "Run GWAS"][0].click().run()
        _ok(at, "detailed run")
        at.button(key="run_full_pipeline").click().run()
        _ok(at, "One-Click")
        with zipfile.ZipFile(io.BytesIO(zips[_TRAIT])) as zf:
            out["one_click"] = zf.read(_MEMBER)
        # Post-GWAS: detect blocks at the run's LD settings; the tab tests them
        mp.setattr(st, "download_button", _download)
        pg = AppTest.from_file(str(_ROOT / "pages" / "Post_GWAS_Analysis.py"), default_timeout=900)
        for k, v in _carried_state(at).items():
            pg.session_state[k] = v
        pg.session_state["n_perm_hap"] = int(_HAP_PERMS)
        pg.run()
        _ok(pg, "Post-GWAS")
        [w for w in pg.number_input if w.label == "r² threshold for cluster definition"][0].set_value(float(_LD_R2))
        [w for w in pg.number_input if w.label == "LD window around each lead SNP (kb)"][0].set_value(float(_FLANK_KB))
        del downloads[:]
        pg.run()
        _ok(pg, "Post-GWAS, block detection")
        picker = [s for s in pg.selectbox if s.label == _BLOCK_PICK]
        assert picker, "the block panel did not render"
        out["blocks"] = []
        for option in picker[0].options:
            del downloads[:]
            [s for s in pg.selectbox if s.label == _BLOCK_PICK][0].set_value(option).run()
            _ok(pg, f"Post-GWAS, block {option}")
            vis = pg.session_state["hap_tables_vis"]
            out["blocks"].append({"option": option, "downloads": list(downloads),
                                  "vis": {k: v.copy() for k, v in vis.items()},
                                  "labels": [b["label"] for b in downloads]})
    st.cache_data.clear()
    return out


def _by_label(snap, label):
    hits = [d for d in snap["downloads"] if d["label"] == label]
    assert len(hits) == 1, (label, [d["label"] for d in snap["downloads"]])
    return hits[0]


def test_the_buttons_replace_the_old_ledger(sites):
    assert len(sites["blocks"]) >= 5
    for snap in sites["blocks"]:
        assert _OLD_LEDGER not in snap["labels"]
        assert _PER_BLOCK in snap["labels"] and _ALL_BLOCKS in snap["labels"]


def test_all_blocks_file_equals_the_cli_and_one_click_zip_member(sites):
    assert sites["one_click"] == sites["cli"]
    for snap in sites["blocks"]:
        got = _by_label(snap, _ALL_BLOCKS)
        assert got["file_name"] == f"Haplotype_assignments_{_TRAIT}.csv"
        assert got["data"] == sites["cli"]


def test_each_blocks_file_is_its_rows_of_the_all_blocks_file(sites):
    lines = sites["cli"].decode("utf-8").splitlines(keepends=True)
    full = pd.read_csv(io.BytesIO(sites["cli"]), keep_default_na=False, na_values=[""])
    seen = set()
    for snap in sites["blocks"]:
        got = _by_label(snap, _PER_BLOCK)
        rows = pd.read_csv(io.BytesIO(got["data"]), keep_default_na=False, na_values=[""])
        (chrom,), (start,), (end,) = (set(rows["Chr"].astype(str)), set(rows["Block_start"]),
                                      set(rows["Block_end"]))
        assert got["file_name"] == f"haplotype_assignments_block_{chrom}_{start}_{end}_{_TRAIT}.csv"
        mask = ((full["Chr"].astype(str) == chrom) & (full["Block_start"] == start)
                & (full["Block_end"] == end)).to_numpy()
        expected = lines[0] + "".join(line for line, m in zip(lines[1:], mask) if m)
        assert got["data"].decode("utf-8") == expected
        seen.add((chrom, start, end))
    blocks = set(zip(full["Chr"].astype(str), full["Block_start"], full["Block_end"]))
    assert seen == blocks


def test_the_mlg_panel_shows_the_tests_groups(sites):
    full = pd.read_csv(io.BytesIO(sites["cli"]), keep_default_na=False, na_values=[""])
    for snap in sites["blocks"]:
        rows = pd.read_csv(io.BytesIO(_by_label(snap, _PER_BLOCK)["data"]),
                           keep_default_na=False, na_values=[""])
        key = f"{rows['Chr'].iloc[0]}:{rows['Block_start'].iloc[0]}-{rows['Block_end'].iloc[0]}"
        shown = snap["vis"][key]
        tested = rows[rows["in_haplotype_test"]]
        assert shown["Sample"].astype(str).tolist() == tested["Sample"].astype(str).tolist()
        assert shown["Haplotype"].astype(str).tolist() == tested["Haplotype"].astype(str).tolist()
        assert shown["Allele_sequence"].astype(str).tolist() == tested["Allele_sequence"].tolist()
        # and those are the test's counts
        n = full[(full["Block_start"] == rows["Block_start"].iloc[0]) & full["in_haplotype_test"]]
        assert len(shown) == len(n) > 0
    assert np.all([len(s["vis"]) >= 1 for s in sites["blocks"]])
