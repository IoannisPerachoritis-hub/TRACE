"""AppTest: the Post-GWAS lead-SNP box plot colours its dots by haplotype (2-INT.9).

The haplotype tab's "Lead SNP genotype effect" gallery colours each dot by the haplotype
of its sample in the LD block that contains the SNP (by position), from the tab's own
haplotype test (the assignments table), with the colours of the block-level plots: the
interactive PCA and the raincloud of the same block. A SNP outside every block, or in a
block with no haplotype groups, keeps the single colour and gets a note. The dots do not
move (data coordinates compared with the plain render).

Data: 135 samples; block 1 (markers 0-4) carries three multi-locus genotypes of 60, 45
and 30 samples, so the test labels them H1, H2 and H3 by size; block 2 (markers 10-14)
gives every sample its own sequence, so it has no haplotype group (all "Other") and is
not tested; marker 20 is significant and in no block. The REAL haplotype test runs (99
permutations); LD detection is stubbed to these two blocks, and the GWAS table is given.
The y axis names the trait, and its scale when the GWAS page applied a transformation
(GWAS_CTX: rank-INT), in which case the values plotted are the GWAS's y.
"""
import json
import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from matplotlib.colors import to_hex  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import gwas.snpplots as _sp  # noqa: E402
import pages._ld_tabs.tab_genome_wide as _tgw  # noqa: E402
from utils.pub_theme import PALETTE_CYCLE  # noqa: E402

# AppTest scripts run in THIS process: stash the real functions at collection time, so
# the script can wrap them (and every later test sees them restored).
_WRAPPED = ("_render_block_visualization", "_render_lead_snp_gallery", "_render_snp_effect",
            "_render_fig_or_cached")
for _name in _WRAPPED:
    if not hasattr(_tgw, "_pristine" + _name):
        setattr(_tgw, "_pristine" + _name, getattr(_tgw, _name))
if not hasattr(_tgw, "_pristine_render_snp_boxplot"):
    _tgw._pristine_render_snp_boxplot = _sp.render_snp_boxplot

_SCRIPT = '''
import sys
_ROOT = r"__ROOT__"
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.colors import to_hex
from gwas.haplotype import run_haplotype_block_gwas
from gwas.ld import extract_block_geno_for_paper
import gwas.snpplots as sp
from pages._ld_tabs import LDContext
import pages._ld_tabs.tab_genome_wide as tgw

N, m = 135, 30
rng = np.random.default_rng(3)
ids = np.array(["S%03d" % i for i in range(N)])
positions = (1_000_000 + np.arange(m) * 10_000).astype(float)
chroms = np.array(["2"] * m)
sid = np.array(["SL25ch02p%d" % int(p) for p in positions])
PATTERNS = np.array([[0, 0, 2, 2, 2], [2, 2, 0, 0, 2], [2, 0, 2, 0, 0]], dtype=float)
grp = rng.permutation(np.repeat([0, 1, 2], [60, 45, 30]))
geno = rng.binomial(2, 0.3, (N, m)).astype(float)
geno[:, :5] = PATTERNS[grp]
geno[:, 10:15] = np.array([[(i // 3 ** k) % 3 for k in range(5)] for i in range(N)], float)
pheno = pd.DataFrame({"t": 10.0 + grp * 2.0 + rng.normal(0.0, 1.0, N)},
                     index=pd.Index(ids, dtype=object))
pheno.iloc[[5, 15], 0] = np.nan
if "__NORM__" != "none":      # the GWAS page applied a transformation to t
    from gwas.qc import normalise_phenotype
    _ok = pheno["t"].notna().to_numpy()
    _y_gwas, _note = normalise_phenotype(pheno["t"].to_numpy()[_ok], "__NORM__")
    st.session_state["GWAS_CTX"] = {"trait": "t", "norm": "__NORM__",
                                    "y": np.asarray(_y_gwas, float).reshape(-1, 1),
                                    "geno_index": ids[_ok]}
    st.session_state["_y_gwas"] = dict(zip(ids[_ok].tolist(),
                                           np.asarray(_y_gwas, float).ravel().tolist()))
pval = np.full(m, 0.5)
pval[:5] = [1e-9, 2e-9, 3e-9, 4e-9, 5e-9]
pval[12], pval[20] = 2e-8, 1e-8
gwas_df = pd.DataFrame({"SNP": sid, "Chr": chroms, "Pos": positions.astype(int), "PValue": pval})
blocks = pd.DataFrame({
    "Chr": ["2", "2"], "Start (bp)": [1_000_000, 1_100_000], "End (bp)": [1_040_000, 1_140_000],
    "lead_snp": [sid[0], sid[12]], "SNP_IDs": [",".join(sid[:5]), ",".join(sid[10:15])],
    "n_snps": [5, 5]})
st.session_state["_grp"] = dict(zip(ids.tolist(), grp.tolist()))
st.session_state["_sid"] = sid.tolist()

st.session_state["ld_geno_hard"] = geno
st.session_state["ld_chroms"] = chroms
st.session_state["ld_positions"] = positions
st.session_state["ld_sid"] = sid
st.session_state["geno_row_ids"] = ids.tolist()
st.session_state["geno_imputed"] = geno          # the PCA panel's PCs
st.session_state["pheno_used_for_hap"] = pheno.copy()
ctx = LDContext(
    geno_ld=geno, geno_hard=geno, chroms=chroms, positions=positions, sid=sid,
    geno_df=pd.DataFrame(geno, index=pd.Index(ids, dtype=object)),
    pheno_df=pheno, trait_col="t", ph_aligned=pheno, y_aligned=pheno["t"].to_numpy(),
    keep_mask=np.ones(N, dtype=bool), geno_ld_aligned=geno,
    gwas_df=gwas_df, haplo_df_auto=blocks,
    ld_decay_kb=100.0, adj_r2_min_global=0.2, ld_trait="t",
)

def _hap_real(blocks_, chroms_, positions_, geno_, sid_, geno_df_, pheno_, trait_, pcs_,
              min_hap_count, min_group_size, n_perm, geno_encoding, n_pcs_used):
    return run_haplotype_block_gwas(
        haplo_df=blocks_, chroms=chroms_, positions=positions_, geno_imputed=geno_,
        sid=sid_, geno_df=geno_df_, pheno_df=pheno_, trait_col=trait_, pcs=pcs_,
        min_hap_count=min_hap_count, min_group_size=min_group_size, n_perm=99,
        geno_encoding=geno_encoding, n_pcs_used=n_pcs_used)

def _extract_real(geno_hard, chroms_, positions_, sid_, block_chr, block_start, block_end,
                  sample_keep_mask=None, maf_threshold=0.01, cache_key="", snp_ids=None):
    return extract_block_geno_for_paper(
        geno_hard, chroms_, positions_, sid_, block_chr, block_start, block_end,
        sample_keep_mask=sample_keep_mask, maf_threshold=maf_threshold, snp_ids=snp_ids)

def _dots(fig):
    out = []
    for line in fig.axes[0].lines:
        if line.get_marker() == "o":
            c = to_hex(line.get_color())
            out += [(float(x), float(y), c) for x, y in zip(line.get_xdata(), line.get_ydata())]
    return out

NEW = ("hap_labels", "hap_color_map", "hap_note", "y_label")
def _rec_plot(snp_id, geno_col, y, **kw):
    fig = tgw._pristine_render_snp_boxplot(snp_id, geno_col, y, **kw)
    plain = tgw._pristine_render_snp_boxplot(
        snp_id, geno_col, y, **{k: v for k, v in kw.items() if k not in NEW})
    leg = fig.axes[0].get_legend()
    st.session_state.setdefault("_rec_box", []).append({
        "snp": str(snp_id), "g": np.asarray(geno_col, float).tolist(),
        "y": np.asarray(y, float).tolist(), "dots": _dots(fig), "plain": _dots(plain),
        "legend": ([t.get_text() for t in leg.get_texts()] if leg is not None else None),
        "legend_colours": ([to_hex(h.get_color()) for h in leg.legend_handles]
                           if leg is not None else None),
        "texts": [t.get_text() for t in fig.texts], "ylabel": fig.axes[0].get_ylabel(),
        "lim": [fig.axes[0].get_xlim(), fig.axes[0].get_ylim()],
        "plain_lim": [plain.axes[0].get_xlim(), plain.axes[0].get_ylim()]})
    plt.close(plain)
    return fig

def _rec_effect(*a, **kw):
    hk = dict(kw.get("hap_kwargs") or {})
    if hk.get("hap_labels") is not None:
        hk["hap_labels"] = [str(h) for h in hk["hap_labels"]]
    st.session_state.setdefault("_rec_effect", []).append(
        {"snp": str(a[1]), "samp": [str(s) for s in a[7]], "hap_kwargs": hk})
    return tgw._pristine_render_snp_effect(*a, **kw)

def _rec_cached(cache_namespace, cache_key, fname_stem, label_prefix, build_fig_fn):
    def _build():
        fig = build_fig_fn()
        if fig is not None and fname_stem.startswith("haplotype_boxplot"):
            # the block's static haplotype plot: each group's dot colour, by its tick label
            ax = fig.axes[0]
            ticks = [t.get_text().split(chr(10))[0] for t in ax.get_xticklabels()]
            block = {}
            for ln in ax.lines:
                if ln.get_marker() == "o" and len(ln.get_xdata()):
                    i = int(round(float(np.mean(np.asarray(ln.get_xdata(), float)))))
                    block[ticks[i]] = to_hex(ln.get_color())
            st.session_state["_rec_rain"] = block
        return fig
    return tgw._pristine_render_fig_or_cached(
        cache_namespace, cache_key, fname_stem, label_prefix, _build)

tgw.find_ld_clusters_genomewide = lambda **kw: blocks.copy()
tgw._render_block_visualization = tgw._pristine_render_block_visualization
tgw._render_lead_snp_gallery = tgw._pristine_render_lead_snp_gallery
tgw._render_snp_effect = _rec_effect
tgw._render_fig_or_cached = _rec_cached
sp.render_snp_boxplot = _rec_plot
try:
    tgw.render(
        ctx,
        get_r2_cached=lambda *a, **k: np.zeros((5, 5)),
        cached_extract_block_geno=_extract_real,
        compute_block_qc_effects=lambda hap_gwas_df=None, **k: hap_gwas_df,
        run_haplotype_block_gwas_cached_fn=_hap_real,
    )
finally:
    tgw._render_snp_effect = tgw._pristine_render_snp_effect
    tgw._render_fig_or_cached = tgw._pristine_render_fig_or_cached
    sp.render_snp_boxplot = tgw._pristine_render_snp_boxplot
'''


@pytest.fixture(autouse=True)
def _restore():
    yield
    for name in _WRAPPED:
        setattr(_tgw, name, getattr(_tgw, "_pristine" + name))
    _sp.render_snp_boxplot = _tgw._pristine_render_snp_boxplot


def _assert_clean(at):
    errs = [str(e.value) for e in at.exception if "url_pathname" not in str(e.value)]
    assert not errs, "unexpected exception: " + "; ".join(errs)


def _start(norm):
    at = AppTest.from_string(_SCRIPT.replace("__ROOT__", str(_ROOT).replace("\\", "/"))
                             .replace("__NORM__", norm), default_timeout=240)
    at.session_state["lead_snp_viz_type"] = "Static"    # the static plot's colours
    at.session_state["hap_viz_type_2_1000000_1040000_t"] = "Static"   # the block's plot
    at.run()
    _assert_clean(at)
    for name in _WRAPPED:      # the script restored what it wrapped
        assert getattr(_tgw, name) is getattr(_tgw, "_pristine" + name)
    return at


@pytest.fixture(scope="module")
def app():
    return _start("none")


def _colour(label):
    m = re.match(r"^H(\d+)$", label)
    return to_hex(PALETTE_CYCLE[(int(m.group(1)) - 1) % len(PALETTE_CYCLE)]) if m else "#bdbdbd"


def _last(at, snp):
    eff = [e for e in at.session_state["_rec_effect"] if e["snp"] == snp]
    box = [b for b in at.session_state["_rec_box"] if b["snp"] == snp]
    assert eff and box, f"{snp} was not drawn"
    return eff[-1], box[-1]


def _pick(at, snp):
    sel = at.selectbox(key="lead_gallery_snp")
    sel.select(next(o for o in sel.options if o.startswith(snp + " "))).run()
    _assert_clean(at)
    return _last(at, snp)


def _check_geometry(box):
    assert sorted(d[:2] for d in box["dots"]) == sorted(d[:2] for d in box["plain"])
    assert box["lim"] == box["plain_lim"]


def test_block_lead_dots_have_their_samples_haplotype_colours(app):
    sid, grp = app.session_state["_sid"], app.session_state["_grp"]
    eff, box = _last(app, sid[0])                # the default pick: the smallest p
    want = [f"H{grp[s] + 1}" for s in eff["samp"]]
    assert eff["hap_kwargs"]["hap_labels"] == want
    # dots matched to samples via the plain render (class order, then sample order)
    import numpy as np
    g, y = np.asarray(box["g"]), np.asarray(box["y"])
    order = [i for cls in (0, 1, 2) for i in np.where(np.rint(g) == cls)[0]]
    assert len(order) == len(box["plain"]) == len(eff["samp"])
    who = {tuple(d[:2]): order[k] for k, d in enumerate(box["plain"])}
    for d in box["dots"]:
        assert d[2] == _colour(want[who[tuple(d[:2])]])
    assert box["legend"] == ["H1", "H2", "H3"] and box["texts"] == []
    assert box["ylabel"] == "t"
    _check_geometry(box)


def test_colours_are_the_block_plots_colours(app):
    sid = app.session_state["_sid"]
    _eff, box = _last(app, sid[0])
    legend = dict(zip(box["legend"], box["legend_colours"]))
    rain = app.session_state["_rec_rain"]
    assert sorted(rain) == ["H1", "H2", "H3"]
    pca = {}
    for el in app.get("plotly_chart"):
        for tr in json.loads(el.proto.spec)["data"]:
            if tr.get("name", "").split(" ")[0] in legend:
                pca[tr["name"].split(" ")[0]] = to_hex(tr["marker"]["color"])
    assert sorted(pca) == ["H1", "H2", "H3"]
    for h in ("H1", "H2", "H3"):
        assert legend[h] == rain[h] == pca[h] == _colour(h)


def test_a_snp_in_no_block_keeps_one_colour_and_gets_the_note(app):
    eff, box = _pick(app, app.session_state["_sid"][20])
    assert eff["hap_kwargs"].get("hap_labels") is None
    assert eff["hap_kwargs"]["hap_note"] == _sp.HAP_NOTE_NO_BLOCK
    assert {d[2] for d in box["dots"]} == {"#0072b2"} and box["legend"] is None
    assert box["texts"] == [_sp.HAP_NOTE_NO_BLOCK]
    _check_geometry(box)


def test_a_snp_in_a_block_with_no_haplotype_group_gets_its_note(app):
    eff, box = _pick(app, app.session_state["_sid"][12])
    assert eff["hap_kwargs"].get("hap_labels") is None
    assert eff["hap_kwargs"]["hap_note"] == _sp.HAP_NOTE_NO_GROUPS
    assert {d[2] for d in box["dots"]} == {"#0072b2"} and box["legend"] is None
    assert box["texts"] == [_sp.HAP_NOTE_NO_GROUPS]


def test_the_axis_names_an_applied_transformation_and_plots_the_gwas_values():
    at = _start("int")
    sid, y_gwas = at.session_state["_sid"], at.session_state["_y_gwas"]
    eff, box = _last(at, sid[0])
    assert box["ylabel"] == "t (rank-INT)"
    assert box["y"] == pytest.approx([y_gwas[s] for s in eff["samp"]])
    assert box["legend"] == ["H1", "H2", "H3"]
