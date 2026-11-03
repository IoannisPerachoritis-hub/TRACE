"""One LD-block heatmap window everywhere (CC-8 acceptance B, reviewer point 2-INT.3).

On the bundled example, the CLI ZIP, the One-Click ZIP and the Post-GWAS Local LD
tab ("Detected block") must show the same SNP window and the same r2 matrix for the
same LD block. All three draw it with gwas.plotting.plot_ld_block_heatmap; this test
wraps that function at each site and compares what each site passes and gets back
(window indices, SNP IDs, block membership, lead SNP, counts, title, r2), not PNG
bytes.

Every site runs for real, with nothing stubbed:
- CLI: ``cli.run_pipeline`` in-process, plots on.
- One-Click: ``pages/GWAS_analysis.py`` under AppTest (real MLM), after the page's
  detailed single-trait run, which publishes the GWAS results the Post-GWAS page reads.
- On-screen: ``pages/Post_GWAS_Analysis.py`` under AppTest with the session state a
  page switch carries over (user keys, not widget-bound keys). The user sets the LD
  Blocks tab's r2 threshold, then picks Region source "Detected block" and the block.

Settings, the same at all three sites (each exists in the CLI and in the GUI): LD r2
0.1, flank 200 kb, seeds p < 1e-5 plus the top 10, Bonferroni, MLM only, raw
phenotype values (the GUI's Section 2b default is a rank-based INT, the CLI default
is --norm none). At a 200 kb flank the detector's distance cap is 200 kb at every
site, whatever LD-decay estimate the GUI passes (200 kb floor), so all three detect
the same blocks. Haplotype permutations are cut to 100 to keep the run short; the
heatmap does not use them. At the default r2 of 0.6 the example forms no LD block.
"""
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
import gwas.plotting as _gp  # noqa: E402
import pages._ld_tabs.tab_genome_wide as _tgw  # noqa: E402
import pages._ld_tabs.tab_local_ld as _tll  # noqa: E402
from annotation import canon_chr  # noqa: E402

_EXAMPLES = _ROOT / "examples"
_VCF = _EXAMPLES / "example.vcf.gz"
_PHENO = _EXAMPLES / "example_pheno.csv"
_TRAIT = "Trait1"
_LD_R2, _FLANK_KB, _HAP_PERMS = 0.1, 200, 100

# AppTest scripts run in this process, and other AppTest files assign module
# attributes for good (tab_genome_wide.find_ld_clusters_genomewide is left stubbed by
# test_block_tables_apptest / test_hap_sample_alignment_apptest). Take the real
# functions at collection time, before any test runs, and set them explicitly.
_REAL = {
    "plot": _gp.plot_ld_block_heatmap,
    "find": _ld.find_ld_clusters_genomewide,
    "extract": _ld.extract_block_geno_for_paper,
    "blockvis": getattr(_tgw, "_pristine_render_block_visualization",
                        _tgw._render_block_visualization),
    "gallery": getattr(_tgw, "_pristine_render_lead_snp_gallery",
                       _tgw._render_lead_snp_gallery),
}

_INFO = ("block_chr", "block_start", "block_end", "n_block", "n_left", "n_right",
         "n_window", "n_target", "n_shown", "n_block_shown", "block_share", "thinned",
         "exhausted", "window_start", "window_end", "lead_pos", "lead_index")


class _Recorder:
    """Wraps plot_ld_block_heatmap (what each site draws) and
    find_ld_clusters_genomewide (each site's SNP axis and block table)."""

    def __init__(self):
        self.site = None
        self.plots = []
        self.finds = []

    def plot(self, geno, chroms, positions, block_chr, block_start, block_end,
             lead_pos=None, **kw):
        fig, info = _REAL["plot"](geno, chroms, positions, block_chr, block_start,
                                  block_end, lead_pos=lead_pos, **kw)
        G = np.asarray(geno)
        idx = np.asarray(info["indices"])
        sid = kw.get("sid")
        self.plots.append({
            "site": self.site,
            "block": (canon_chr(str(block_chr)), int(block_start), int(block_end)),
            "label_mode": kw.get("label_mode", "position"),
            "geno_shape": G.shape,
            "indices": idx.copy(),
            "positions_axis": np.asarray(positions).astype(np.int64).copy(),
            "sid_axis": None if sid is None else np.asarray(sid).astype(str).copy(),
            "in_block": np.asarray(info["in_block"]).copy(),
            "r2": np.array(info["r2"], dtype=float, copy=True),
            "info": {k: info[k] for k in _INFO},
            "title": info["title"],
        })
        return fig, info

    def find(self, *a, **kw):
        out = _REAL["find"](*a, **kw)
        self.finds.append({
            "site": self.site,
            "sid_axis": np.asarray(kw["sid"]).astype(str).copy(),
            "positions_axis": np.asarray(kw["positions"]).astype(np.int64).copy(),
            "blocks": out.copy(),
        })
        return out


@pytest.fixture
def rec(monkeypatch):
    r = _Recorder()
    # CLI and One-Click import the heatmap function at call time; Local LD and the
    # LD Blocks tab bind their names at module import.
    monkeypatch.setattr(_gp, "plot_ld_block_heatmap", r.plot)
    monkeypatch.setattr(_tll, "plot_ld_block_heatmap", r.plot)
    monkeypatch.setattr(_tll, "extract_block_geno_for_paper", _REAL["extract"])
    monkeypatch.setattr(_ld, "find_ld_clusters_genomewide", r.find)
    monkeypatch.setattr(_tgw, "find_ld_clusters_genomewide", r.find)
    monkeypatch.setattr(_tgw, "_render_block_visualization", _REAL["blockvis"])
    monkeypatch.setattr(_tgw, "_render_lead_snp_gallery", _REAL["gallery"])
    # memoised frames persist across runs in-process; start and end clean
    st.cache_data.clear()
    yield r
    st.cache_data.clear()


def _only_harness_error(at):
    return all("url_pathname" in str(e.value) for e in at.exception)


def _assert_clean(at, where):
    assert not at.exception or _only_harness_error(at), (
        f"{where}: " + "; ".join(str(e.value) for e in at.exception))


def _run_cli(rec, out_dir):
    rec.site = "cli"
    args = cli._build_parser().parse_args([
        "--vcf", str(_VCF), "--pheno", str(_PHENO), "--trait", _TRAIT,
        "--output", str(out_dir), "--ld-r2", str(_LD_R2),
        "--ld-flank-kb", str(_FLANK_KB), "--hap-perms", str(_HAP_PERMS),
    ])
    cli.run_pipeline(args)
    zips = list(out_dir.glob("*.zip"))
    assert len(zips) == 1, f"expected one CLI ZIP, got {zips}"
    with zipfile.ZipFile(zips[0]) as zf:
        return sorted(n for n in zf.namelist() if n.startswith("figures/LD_heatmap_"))


def _run_gwas_page(rec):
    at = AppTest.from_file(str(_ROOT / "pages" / "GWAS_analysis.py"), default_timeout=900)
    ss = at.session_state
    ss["_persist_vcf_bytes"] = _VCF.read_bytes()
    ss["_persist_is_gz"] = True
    ss["_persist_pheno"] = pd.read_csv(_PHENO, sep=None, engine="python", encoding="utf-8-sig")
    ss["selected_traits_multiselect"] = [_TRAIT]
    ss["gwas_pheno_qc_transform"] = "None (raw values)"   # = the CLI's --norm none
    ss["pipe_models"] = []                                # One-Click: MLM only
    ss["pipe_subsampling"] = False
    ss["pipe_generate_ld_heatmaps"] = True
    ss["pipe_ld_r2"] = float(_LD_R2)
    ss["pipe_ld_flank_mode"] = "Manual (fixed kb)"
    ss["pipe_ld_flank"] = int(_FLANK_KB)
    ss["pipe_hap_perms"] = int(_HAP_PERMS)
    rec.site = "gwas_page"
    at.run()
    _assert_clean(at, "GWAS page, first run")
    # detailed single-trait view, MLM only: publishes the results Post-GWAS reads
    models = [w for w in at.sidebar.multiselect if w.label == "Additional GWAS models"]
    assert models, "sidebar 'Additional GWAS models' not found"
    models[0].set_value([])
    run_btn = [b for b in at.button if b.label == "Run GWAS"]
    assert run_btn, "'Run GWAS' button not found"
    run_btn[0].click().run()
    _assert_clean(at, "GWAS page, detailed run")
    assert _TRAIT in at.session_state["gwas_results"], "detailed run published no GWAS results"
    rec.site = "one_click"
    at.button(key="run_full_pipeline").click().run()
    _assert_clean(at, "One-Click")
    assert not at.error, "One-Click errors: " + "; ".join(e.value for e in at.error)
    return at


def _carried_state(at):
    """What a page switch keeps: user keys. Widget-bound keys belong to the page
    that rendered the widget and are dropped when it is no longer rendered.
    (Streamlit 1.50 internals; the version is pinned in constraints.txt.)"""
    widget_keys = set(at.session_state._state._key_id_mapper.id_key_mapping.values())
    return {k: v for k, v in at.session_state.filtered_state.items() if k not in widget_keys}


def _run_local_ld(rec, state, target):
    rec.site = "on_screen"
    at = AppTest.from_file(str(_ROOT / "pages" / "Post_GWAS_Analysis.py"), default_timeout=900)
    for k, v in state.items():
        at.session_state[k] = v
    at.session_state["n_perm_hap"] = int(_HAP_PERMS)   # LD Blocks tab; heatmap-neutral
    at.run()
    _assert_clean(at, "Post-GWAS, first run")
    # LD Blocks & Haplotypes tab: detect blocks at the run's LD r2 and flank
    r2_in = [w for w in at.number_input if w.label == "r² threshold for cluster definition"]
    flank_in = [w for w in at.number_input if w.label == "LD window around each lead SNP (kb)"]
    assert r2_in and flank_in, "LD Blocks tab detection controls not found"
    r2_in[0].set_value(float(_LD_R2))
    flank_in[0].set_value(float(_FLANK_KB))
    at.run()
    _assert_clean(at, "Post-GWAS, block detection")
    blocks = at.session_state["haplo_df_auto"][_TRAIT].reset_index(drop=True)
    hit = [i for i, r in blocks.iterrows()
           if (canon_chr(str(r["Chr"])), int(r["Start (bp)"]), int(r["End (bp)"])) == target]
    assert hit, f"Post-GWAS did not detect block {target}; it found {blocks.iloc[:, :4].values.tolist()}"
    # region selector: snap to the detected block -> Local LD draws its heatmap
    at.session_state["ld_window_source"] = "Detected block"
    at.session_state["ld_window_block_select"] = int(hit[0])
    at.run()
    _assert_clean(at, "Post-GWAS, Local LD in detected-block mode")
    captions = [c.value for c in at.caption if "red tick" in c.value]
    # the SNP-ID tick labels must not change the window or the matrix
    at.radio(key="local_ld_block_labels").set_value("SNP IDs").run()
    _assert_clean(at, "Post-GWAS, SNP-ID labels")
    return captions


def _ids(rec_plot, axes):
    sid_axis = rec_plot["sid_axis"] if rec_plot["sid_axis"] is not None else axes
    return sid_axis[rec_plot["indices"]]


def _assert_same(a, b, ids_a, ids_b, what):
    assert a["geno_shape"] == b["geno_shape"], f"{what}: genotype matrix shapes differ"
    assert np.array_equal(a["indices"], b["indices"]), f"{what}: window indices differ"
    assert np.array_equal(ids_a, ids_b), f"{what}: window SNPs differ:\n{ids_a}\n{ids_b}"
    assert np.array_equal(a["in_block"], b["in_block"]), f"{what}: block membership differs"
    assert a["info"] == b["info"], f"{what}: window summary differs:\n{a['info']}\n{b['info']}"
    assert a["title"] == b["title"], f"{what}: titles differ"
    diff = np.nanmax(np.abs(a["r2"] - b["r2"])) if a["r2"].shape == b["r2"].shape else None
    assert np.array_equal(a["r2"], b["r2"], equal_nan=True), (
        f"{what}: r2 matrices differ (max |diff| {diff})")


@pytest.mark.skipif(not _VCF.exists(), reason="examples/example.vcf.gz not present")
def test_cli_one_click_and_local_ld_draw_the_same_window(rec, tmp_path):
    cli_pngs = _run_cli(rec, tmp_path / "cli")
    at = _run_gwas_page(rec)

    cli_plots = [p for p in rec.plots if p["site"] == "cli"]
    assert cli_plots and cli_pngs, "the CLI drew no LD heatmap at these settings"
    target = cli_plots[0]["block"]
    assert f"figures/LD_heatmap_Chr{target[0]}_{target[1]}_{target[2]}.png" in cli_pngs

    captions = _run_local_ld(rec, _carried_state(at), target)

    def axes(site):
        return [f for f in rec.finds if f["site"] == site][-1]

    # the three sites hold the same SNP axis, so indices are comparable across them
    for site in ("one_click", "on_screen"):
        assert np.array_equal(axes(site)["sid_axis"], axes("cli")["sid_axis"]), \
            f"{site}: SNP axis differs from the CLI's"
    b = axes("cli")["blocks"]
    in_target = ((b["Chr"].astype(str).map(canon_chr) == target[0])
                 & (b["Start (bp)"].astype(int) == target[1])
                 & (b["End (bp)"].astype(int) == target[2]))
    lead = str(b.loc[in_target, "lead_snp"].iloc[0])

    def plots(site, mode="position"):
        hits = [p for p in rec.plots
                if p["site"] == site and p["block"] == target and p["label_mode"] == mode]
        assert hits, f"{site} drew no heatmap for block {target} ({mode} labels)"
        return hits[0]

    cli_p, oc_p = plots("cli"), plots("one_click")
    os_p, os_ids_p = plots("on_screen"), plots("on_screen", "snp_id")
    ids = {name: _ids(p, axes(site)["sid_axis"]) for name, p, site in (
        ("cli", cli_p, "cli"), ("one_click", oc_p, "one_click"),
        ("on_screen", os_p, "on_screen"), ("on_screen_ids", os_ids_p, "on_screen"))}

    # the window really carries neighbours (otherwise the comparison is trivial)
    info = cli_p["info"]
    assert info["n_left"] > 0 and info["n_right"] > 0
    assert info["n_block_shown"] < info["n_shown"]

    _assert_same(cli_p, oc_p, ids["cli"], ids["one_click"], "CLI vs One-Click")
    _assert_same(cli_p, os_p, ids["cli"], ids["on_screen"], "CLI vs Local LD")
    _assert_same(os_p, os_ids_p, ids["on_screen"], ids["on_screen_ids"],
                 "Local LD position vs SNP-ID labels")

    # each site marks the block table's own lead SNP
    for name, p in (("cli", cli_p), ("one_click", oc_p), ("on_screen", os_p)):
        li = p["info"]["lead_index"]
        assert li is not None, f"{name}: lead SNP {lead} not marked"
        assert ids[name][int(np.flatnonzero(p["indices"] == li)[0])] == lead, \
            f"{name}: marks a different lead SNP"
    assert any(f"({lead})" in c for c in captions), f"Local LD caption does not name {lead}"
