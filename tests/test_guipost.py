"""The Post-GWAS page reads the trait's table in windows and blocks (gwas.guipost).

- ArrayColumns (the table in memory) gives every read site of the page and of the shared LD
  code today's operand: values, number format and layout, as LazyColumns does on disk.
- The PostTable of a memory SessionTable and of a disk SessionTable read the same windows.
- Every tab's computation gives the same tables in memory, on disk, and with today's
  whole-table expressions (example_v1, Trait1): significant SNPs, regional r2, local LD, the
  automatic blocks and the tab's detection, block LD quality, haplotype tests and
  assignments, block effects, the block table, the lead-SNP gallery's column.
- Nothing reads the table whole (TRACE_STORE_FORBID_MATERIALIZE=1, and the read log).
- The cache keys name the table's identity and the window; the caches are bounded; and
  POST_GWAS_SESSION_KEYS lists every session key the page and its tabs write.
- A table file closed meanwhile (Clear) is found and ends the page plainly.
- The page's automatic blocks and the LD Blocks tab's default detection share one result,
  and only when every input is the same.
- The Significant SNPs tab builds its table once per input.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gwas import guipost as gp
from gwas.genostore import as_columns, is_lazy, read_log, reset_read_log
from tests import _post_gwas_tables as pt

ROOT = Path(__file__).resolve().parents[1]


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape
    assert np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()
    assert a.flags["C_CONTIGUOUS"] == b.flags["C_CONTIGUOUS"]
    assert a.flags["F_CONTIGUOUS"] == b.flags["F_CONTIGUOUS"]


# ------------------------------------------------------------------ ArrayColumns read sites
N, M = 30, 200
ROWS = np.ones(N, dtype=bool)
ROWS[[2, 7, 19]] = False


def _array(order, dtype, seed=3):
    rng = np.random.default_rng(seed)
    A = rng.integers(0, 3, size=(N, M)).astype(dtype)
    A[rng.random((N, M)) < 0.05] = np.nan
    A[:, 50] = 1.0                                   # a monomorphic column
    return np.asarray(A, order=order)


def _sel(width, offset):
    chm = np.zeros(M, dtype=bool)
    chm[40:160] = True
    reg = np.zeros(int(chm.sum()), dtype=bool)
    reg[offset:offset + width] = True
    idx = np.arange(40 + offset, 40 + offset + width)
    mask = np.zeros(M, dtype=bool)
    mask[idx] = True
    return {"idx": idx, "j": int(idx[-1]), "chm": chm, "reg": reg, "mask": mask,
            "k": int(idx[0]), "p": int(width)}


# (site, today's expression on the array, the page's read through ArrayColumns)
SITES = {
    "ld window (block_mean_r2, merge_coherence, ld quality, extract, sigtable)":
        (lambda A, s: np.asarray(A, float)[:, s["idx"]],
         lambda R, s: as_columns(R, float)[:, s["idx"]]),
    "find_ld_clusters_genomewide window":
        (lambda A, s: np.asarray(np.asarray(A, float)[:, s["chm"]], float)[:, s["reg"]],
         lambda R, s: as_columns(R, float)[:, np.flatnonzero(s["chm"])[s["reg"]]]),
    "auto blocks on the aligned rows":
        (lambda A, s: np.asarray(np.asarray(A, float)[ROWS, :][:, s["chm"]], float)[:, s["reg"]],
         lambda R, s: as_columns(R.astype(float).lazy_rows(ROWS), float)[:, np.flatnonzero(s["chm"])[s["reg"]]]),
    "block heatmap on the aligned rows":
        (lambda A, s: np.asarray(np.asarray(A, float)[ROWS, :])[:, s["idx"]],
         lambda R, s: as_columns(R.astype(float).lazy_rows(ROWS))[:, s["idx"]]),
    "r2 to the lead (one column)":
        (lambda A, s: np.asarray(A, dtype=float)[:, s["j"]],
         lambda R, s: as_columns(R, float)[:, s["j"]]),
    "haplotype test: rows, then the block's columns":
        (lambda A, s: np.asarray(np.asarray(np.asarray(A, float))[ROWS, :][:, s["mask"]], float),
         lambda R, s: np.asarray(as_columns(R.astype(float))[ROWS, :][:, s["mask"]], float)),
    "haplotype test: the lead SNP's dose (one column of the rows)":
        (lambda A, s: np.asarray(np.asarray(np.asarray(A, float))[ROWS, :][:, s["j"]], dtype=float),
         lambda R, s: np.asarray(as_columns(R.astype(float))[ROWS, :][:, s["j"]], dtype=float)),
    "block effects: aligned rows, then the block's columns":
        (lambda A, s: np.asarray(np.asarray(A, float)[ROWS, :][:, s["mask"]], float),
         lambda R, s: np.asarray(R.astype(float)[ROWS, :][:, s["mask"]], float)),
    "local LD extract: rows, then the window":
        (lambda A, s: np.asarray(A, float)[ROWS, :][:, s["mask"]],
         lambda R, s: as_columns(R.astype(float), float)[ROWS, :][:, s["mask"]]),
    "significant SNPs: per-SNP MAF in column slices":
        (lambda A, s: np.asarray(A, float)[:, s["k"]:s["k"] + s["p"]],
         lambda R, s: R.astype(float)[:, s["k"]:s["k"] + s["p"]]),
}


@pytest.mark.parametrize("site", list(SITES))
@pytest.mark.parametrize("width", [1, 2, 3, 17])
@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_array_columns_give_todays_operand(site, width, order, dtype):
    A = _array(order, dtype)
    R = gp.ArrayColumns(A)
    today, page = SITES[site]
    for offset in (0, 31, 120 - width):
        s = _sel(width, offset)
        got = page(R, s)
        assert not is_lazy(got)
        _same(got, today(A, s))


@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("encoding", ["dosage012", "binary01"])
def test_hard_calls_are_todays_columns(order, encoding):
    """HardCalls(reader)[:, j] gives today's np.rint(geno_ld)[:, j] (dosage012), else the
    dosages of geno_ld.copy()[:, j], as the gallery takes them (.astype(float))."""
    A = _array(order, np.float32)
    A[:5, 10] = 0.6                                 # a mean-imputed value
    geno_ld = np.asarray(A, float)
    geno_hard = np.rint(geno_ld) if encoding == "dosage012" else geno_ld.copy()
    hard = gp.HardCalls(gp.ArrayColumns(A, dtype=np.float64), encoding)
    assert hard.shape == geno_hard.shape
    for j in (0, 10, 50, M - 1):
        _same(np.asarray(hard[:, j]).astype(float), np.asarray(geno_hard)[:, j].astype(float))


def test_array_columns_never_copy_the_table_whole(monkeypatch):
    """A whole-table read happens only on request (np.asarray), is logged, and is refused
    under TRACE_STORE_FORBID_MATERIALIZE=1 as on disk; windows are not affected."""
    A = _array("F", np.float32)
    R = gp.ArrayColumns(A, dtype=np.float64)
    reset_read_log()
    w = R[:, np.arange(5)]
    assert w.shape == (N, 5) and read_log()["materialized"] == []
    assert R.arr is A                                # the array is kept as it is
    _same(np.asarray(R), np.asarray(A, float))       # on request: today's whole copy
    assert len(read_log()["materialized"]) == 1
    monkeypatch.setenv("TRACE_STORE_FORBID_MATERIALIZE", "1")
    with pytest.raises(MemoryError):
        np.asarray(R)
    _same(R[:, np.arange(3)], np.asarray(A, float)[:, np.arange(3)])


# ------------------------------------------------------------- memory and disk PostTables
@pytest.fixture(scope="module")
def example_v1(tmp_path_factory):
    """example_v1, Trait1, the page's defaults: the memory and disk tables and their
    PostTables (the page's phenotype alignment), with a GWAS table whose every 7th SNP is
    significant (the computations take the GWAS results as given)."""
    from gwas.utils import align_pheno_to_geno

    vcf, pheno_path = pt.EXAMPLE_V1 / "example.vcf.gz", pt.EXAMPLE_V1 / "example_pheno.csv"
    if not vcf.exists():
        pytest.skip("tests/data/example_v1 not present")
    pheno = pt.read_pheno(pheno_path)
    mem = pt.memory_tables(vcf, pheno, "Trait1")
    disk = pt.disk_tables(vcf, pheno, "Trait1", tmp_path_factory.mktemp("store") / "t")
    _, _, keep = align_pheno_to_geno(mem.pheno, mem.geno_df, "Trait1")
    posts = {"memory": gp.post_table(mem.table, keep, samples=mem.geno_df.index),
             "disk": gp.post_table(disk.table, keep, samples=disk.geno_df.index)}
    t = mem.table
    sid = np.asarray(t.sid).astype(str)
    chroms = np.asarray(t.chroms).astype(str)
    positions = np.asarray(t.positions)
    rng = np.random.default_rng(11)
    p = rng.uniform(0.01, 1.0, sid.size)
    for j in range(0, sid.size, 7):
        p[j] = 10.0 ** -(5.5 + (j % 4))
    gwas_df = pd.DataFrame({"SNP": sid, "Chr": chroms, "ChrNum": pd.to_numeric(chroms, errors="coerce"),
                            "Pos": positions, "PValue": p,
                            "Beta_MLM": rng.normal(0, 1, sid.size), "SE_MLM": 0.2,
                            "Beta_OLS": rng.normal(0, 1, sid.size), "SE_OLS": 0.2, "Model": "MLM"})
    yield {"mem": mem, "disk": disk, "posts": posts, "keep": keep, "gwas_df": gwas_df,
           "sid": sid, "chroms": chroms, "positions": positions, "pheno": mem.pheno}
    pt.close(disk)


def test_memory_and_disk_tables_are_equal(example_v1):
    m, d = example_v1["mem"].table, example_v1["disk"].table
    assert m.identity == d.identity
    _same(d.view.materialize(), m.view)
    _same(d.raw.materialize(), m.raw)
    for name in ("sid", "positions", "iid"):
        assert np.array_equal(np.asarray(getattr(d, name)), np.asarray(getattr(m, name)))
    assert np.array_equal(np.asarray(d.chroms).astype(str), np.asarray(m.chroms).astype(str))
    assert example_v1["disk"].geno_df.index.equals(example_v1["mem"].geno_df.index)


def test_post_tables_read_the_same_windows(example_v1):
    pm, pd_ = example_v1["posts"]["memory"], example_v1["posts"]["disk"]
    view = np.asarray(example_v1["mem"].table.view)
    geno_ld = np.asarray(view, float)                       # today's whole copies
    aligned = geno_ld[example_v1["keep"], :]
    hard = np.rint(geno_ld)
    raw = np.asarray(example_v1["mem"].table.raw)
    assert pm.shape == pd_.shape == view.shape
    assert pm.identity == pd_.identity
    for idx in (np.arange(3), np.arange(100, 160), np.array([5, 400, 7])):
        for post in (pm, pd_):
            _same(as_columns(post.ld, float)[:, idx], geno_ld[:, idx])
            _same(as_columns(post.ld_aligned)[:, idx], aligned[:, idx])
            _same(as_columns(post.raw, float)[:, idx], np.asarray(raw, float)[:, idx])
            _same(np.asarray(post.hard[:, int(idx[0])]).astype(float), hard[:, int(idx[0])].astype(float))
    assert list(pm.samples.columns) == [] and pm.samples.index.equals(pd_.samples.index)


def _computations(post, gwas_df, chroms, positions, sid, pheno, trait="Trait1"):
    """What the page and its tabs compute from the table, with the arguments they pass
    (pages/Post_GWAS_Analysis.py and pages/_ld_tabs/*), as CSV text and arrays."""
    from gwas.haplotype import haplotype_assignments
    from gwas.ld import compute_block_ld_quality, extract_block_geno_for_paper, find_ld_clusters_genomewide, pairwise_r2
    from gwas.plotting import compute_r2_to_lead, plot_ld_block_heatmap
    from gwas.significance import frame_for_rule, rule_from_streamlit
    from gwas.sigtable import build_significant_snp_table, project_unblocked
    from pages._ld_tabs.block_table import merge_block_table
    import matplotlib.pyplot as plt

    out = {}
    rule = rule_from_streamlit("Bonferroni (α = 0.05)", len(gwas_df), None)
    sig = build_significant_snp_table(
        frame_for_rule(gwas_df, rule), pd.DataFrame(), rule, chroms, positions, sid,
        geno_dosage_raw=post["raw"], genes=None, seed_p_used=1e-5, top_n_used=0,
        edge_flank_bp=300_000, ld_decay_bp=None, geno_encoding="dosage012")
    out["significant_snps.csv"] = sig.to_csv(index=False)
    out["unblocked_snps.csv"] = project_unblocked(sig).to_csv(index=False) if len(sig) else ""
    # automatic blocks (aligned rows) and the tab's own detection (all rows, r2 0.1, 200 kb)
    auto = find_ld_clusters_genomewide(
        gwas_df=gwas_df, chroms=chroms, positions=positions, geno_imputed=post["ld_aligned"],
        sid=sid, ld_threshold=0.6, flank_kb=300, min_snps=2, top_n=10, sig_thresh=1e-5,
        max_dist_bp=None, ld_decay_kb=None, adj_r2_min=0.2)
    out["auto_blocks.csv"] = auto.to_csv(index=False)
    blocks = find_ld_clusters_genomewide(
        gwas_df=gwas_df, chroms=chroms, positions=positions, geno_imputed=post["ld"], sid=sid,
        ld_threshold=0.1, flank_kb=200.0, min_snps=2, top_n=10, sig_thresh=1e-5,
        max_dist_bp=None, ld_decay_kb=None, adj_r2_min=0.2, gap_factor=10.0)
    assert len(blocks) > 0, "no LD block: the comparison would be vacuous"
    out["blocks.csv"] = blocks.to_csv(index=False)
    ldq = compute_block_ld_quality(blocks, chroms, positions, sid, post["ld"], gwas_df,
                                   geno_dosage_raw=post["raw"], r2_coherent=0.6)
    out["ld_quality.csv"] = ldq.to_csv(index=False)
    # the haplotype test (the tab's positional call), its assignments, the block effects
    hap, tables = gp.run_haplotype_tests(blocks, chroms, positions, post["ld"], sid,
                                         post["samples"], pheno, trait, None, 5, 3, 99,
                                         "dosage012", None)
    assert len(hap) > 0, "no haplotype test ran"
    out["haplotype_tests.csv"] = hap.to_csv(index=False)
    out["assignments.csv"] = haplotype_assignments(hap, tables, post["samples"], pheno, trait).to_csv(index=False)
    eff = gp.block_qc_effects(hap_gwas_df=hap, chroms=chroms, positions=positions,
                              geno_imputed=post["ld"], geno_df=post["samples"], pheno_for_hap=pheno,
                              trait_col=trait, min_hap_count=5, min_group_size=3, compute_maf=True,
                              geno_encoding="dosage012", pcs=None, strict_selfing=False)
    out["block_effects.csv"] = eff.to_csv(index=False)
    out["block_table.csv"] = merge_block_table(blocks, eff, ldq, untested=hap.attrs.get("untested")).to_csv(index=False)
    # local LD around the first block's lead (lead-SNP mode) and the block heatmap
    b0 = blocks.iloc[0]
    lead = str(b0["lead_snp"])
    lead_pos = int(positions[np.flatnonzero(sid == lead)[0]])
    g, rp, rs = extract_block_geno_for_paper(
        geno_imputed=post["ld"], chroms=chroms, positions=positions, sid=sid,
        block_chr=str(b0["Chr"]), start_bp=lead_pos - 300_000, end_bp=lead_pos + 300_000,
        sample_keep_mask=post["keep"], maf_threshold=0.01, snp_ids=None)
    out["local_ld_window.npy"] = g
    out["local_ld_r2.npy"] = np.clip(pairwise_r2(np.asarray(g, float), min_pair_n=20), 0.0, 1.0)
    fig, info = plot_ld_block_heatmap(post["ld_aligned"], chroms, positions, str(b0["Chr"]),
                                      int(b0["Start (bp)"]), int(b0["End (bp)"]), lead_pos=lead_pos,
                                      label_mode="position", sid=sid, lead_star=True)
    plt.close(fig)
    out["block_heatmap_r2.npy"] = info["r2"]
    # the regional plot's r2 to the lead, and the gallery's hard-call column
    mask = (chroms == str(b0["Chr"])) & (positions >= lead_pos - 300_000) & (positions <= lead_pos + 300_000)
    out["regional_r2.npy"] = compute_r2_to_lead(post["raw"], sid, lead, mask, min_pair_n=15)[0]
    out["gallery_dose.npy"] = np.asarray(post["hard"][:, int(np.flatnonzero(sid == lead)[0])]).astype(float)
    return out


def _today(example_v1):
    """Today's page objects: the whole float64 copies of ff56e5c's page."""
    t = example_v1["mem"].table
    geno_ld = np.asarray(t.view, float)
    return {"ld": geno_ld, "ld_aligned": geno_ld[example_v1["keep"], :], "hard": np.rint(geno_ld),
            "raw": t.raw, "samples": example_v1["mem"].geno_df, "keep": example_v1["keep"]}


def _post(post, keep):
    return {"ld": post.ld, "ld_aligned": post.ld_aligned, "hard": post.hard, "raw": post.raw,
            "samples": post.samples, "keep": keep}


def _compare(a, b):
    assert sorted(a) == sorted(b)
    for k in a:
        if isinstance(a[k], str):
            assert a[k] == b[k], k
        else:
            _same(a[k], b[k])


def test_every_tab_computation_is_the_same_in_memory_on_disk_and_today(example_v1, monkeypatch):
    args = (example_v1["gwas_df"], example_v1["chroms"], example_v1["positions"],
            example_v1["sid"], example_v1["pheno"])
    today = _computations(_today(example_v1), *args)
    monkeypatch.setenv("TRACE_STORE_FORBID_MATERIALIZE", "1")   # no whole-table read
    for mode in ("memory", "disk"):
        reset_read_log()
        got = _computations(_post(example_v1["posts"][mode], example_v1["keep"]), *args)
        _compare(got, today)
        log = read_log()
        # no whole-table read, and no piece wider than the CLI's 4,096 columns (the per-SNP
        # MAF's pieces; on this 510-SNP table one piece holds every column)
        assert log["materialized"] == [], mode
        assert log["blocks"] > 0
        assert all(s["max_cols"] <= 4096 and s["whole"] == 0 for s in log["by_site"].values()), \
            (mode, log["by_site"])


# ------------------------------------------------------------------------------ the keys
class _Ctx:
    def __init__(self, geno, identity="", keep=None):
        self.geno_ld = geno
        self.table_identity = identity
        self.keep_mask = keep
        self.chroms = np.array(["1", "1", "2"])
        self.positions = np.array([10, 20, 30])
        self.sid = np.array(["a", "b", "c"])


def test_r2_keys_name_the_table_its_rows_and_the_window():
    G = _array("F", np.float32)
    keep = np.ones(N, dtype=bool)
    base = gp.r2_key(_Ctx(G, "T1", keep), "local", "1", 0, 100, "s1", ["s1", "s2"])
    assert "T1" in base
    keep2 = keep.copy()
    keep2[0] = False
    assert gp.r2_key(_Ctx(G, "T2", keep), "local", "1", 0, 100, "s1", ["s1", "s2"]) != base
    assert gp.r2_key(_Ctx(G, "T1", keep2), "local", "1", 0, 100, "s1", ["s1", "s2"]) != base
    assert gp.r2_key(_Ctx(G, "T1", keep), "local", "1", 0, 100, "s1", ["s1", "s3"]) != base
    assert gp.r2_key(_Ctx(G, "T1", keep), "local", "1", 0, 100, "s1", ["s2", "s1"]) != base
    assert gp.r2_key(_Ctx(G, "T1", keep), "local", "1", 0, 100, "s1", ["s1", "s2"]) == base
    # without an identity (a test's plain array) the content names the table
    G2 = G.copy()
    G2[0, 0] = 2.0 if G[0, 0] != 2.0 else 0.0
    assert gp.table_key(_Ctx(G)) == gp.table_key(_Ctx(G.copy())) != gp.table_key(_Ctx(G2))
    # the window's own values when no key is given
    assert gp.window_r2_key(G[:, :3]) != gp.window_r2_key(G2[:, :3])


def test_haplotype_and_detection_keys_name_the_identity_not_the_shape():
    pcs = np.zeros((N, 2))
    a = gp.haplotype_key("t", 5, 3, pcs, "identityA", 1000, "B", "P")
    b = gp.haplotype_key("t", 5, 3, pcs, "identityB", 1000, "B", "P")
    assert a != b and "identityA" in a and a.startswith("HAPGWAS::")
    assert f"::{N}x{M}::" not in a                                # no shape term for the table
    d1 = gp.detection_key("t", 0.6, 300.0, 10, 1e-5, None, "G", "identityA")
    d2 = gp.detection_key("t", 0.6, 300.0, 10, 1e-5, None, "G", "identityB")
    assert d1 != d2 and d1.startswith("LDDETECT::")
    assert all(k.startswith(gp.POST_GWAS_SESSION_PREFIXES) for k in (a, b, d1, d2))


def test_frame_key_reads_every_row():
    df = pd.DataFrame({"SNP": [f"s{i}" for i in range(300_000)], "PValue": np.linspace(0, 1, 300_000)})
    df2 = df.copy()
    df2.loc[123_457, "PValue"] = 0.5
    assert gp.frame_key(df) == gp.frame_key(df.copy()) != gp.frame_key(df2)
    assert gp.frame_key(df) != gp.frame_key(df.rename(columns={"PValue": "P"}))


def test_sample_index_sources():
    """The table's iid first, else the genotype frame's index (the GWAS page's frame, or the
    store frame on disk), else GWAS_CTX's geno_index; none of them: a plain error for the
    page. A geno_df of another trait (written before the table, by a GWAS page run that
    stopped between the two) does not replace the table's own samples."""
    idx = pd.Index(["a", "b"], dtype=object)
    store_frame = pd.DataFrame({"__trace_store_row__": [0, 1]}, index=idx)
    iid = np.array([["a", "a"], ["b", "b"]])
    assert list(gp.sample_index(iid)) == ["a", "b"]
    other = pd.DataFrame(index=pd.Index(["b", "c"], dtype=object))      # another trait's frame
    assert list(gp.sample_index(iid, other, np.array(["c", "d"], dtype=object))) == ["a", "b"]
    assert gp.sample_index(None, store_frame) is store_frame.index
    assert list(gp.sample_index(None, None, np.array(["a", "b"], dtype=object))) == ["a", "b"]
    with pytest.raises(ValueError, match="Run the GWAS page first"):
        gp.sample_index(None, None, None)


def test_the_tables_iid_gives_the_genotype_frames_index(example_v1):
    """The page now takes the samples from the table's iid: the same index as the genotype
    frame's (values, dtype, name), in memory and on disk."""
    for tables in (example_v1["mem"], example_v1["disk"]):
        got = gp.sample_index(tables.table.iid)
        assert got.identical(tables.geno_df.index)
        assert got.dtype == object and got.name is None


def test_bounded_caches():
    from collections import OrderedDict

    state = {"c": {"x": 1}}
    store = gp.bounded(state, "c")
    assert isinstance(store, OrderedDict) and state["c"] is store and store["x"] == 1
    for i in range(12):
        gp.remember(store, i, i, 4)
    assert list(store) == [8, 9, 10, 11]
    assert gp.recall(store, 9) == 9 and list(store)[-1] == 9 and gp.recall(store, 0) is None
    d = {}
    for i in range(10):
        gp.keep_recent(d, i % 6, i, 3)
    assert list(d) == [1, 2, 3] and d[3] == 9
    s = {}
    for i in range(6):
        k = f"HAPGWAS::{i}"
        s[k], s[k + "::tables"] = i, i
        gp.remember_family(s, "_order", k, 2, suffixes=("::tables", "::assign"))
    assert sorted(k for k in s if k.startswith("HAPGWAS::")) == [
        "HAPGWAS::4", "HAPGWAS::4::tables", "HAPGWAS::5", "HAPGWAS::5::tables"]
    assert s["_order"] == ["HAPGWAS::4", "HAPGWAS::5"]


# --------------------------------------------------------- a table closed meanwhile (Clear)
def test_a_closed_disk_table_is_released(tmp_path):
    """Clear session data and the registry's eviction close a table file: GenotypeStore.closed
    says so (read only), table_released finds it through the session's table object, and
    released_guard turns the read error into the page's plain stop. A table in memory is never
    released; other errors pass through unchanged."""
    from types import SimpleNamespace

    from gwas.genostore import LazyColumns, write_store
    from gwas.guitable import SessionTable

    G = _array("F", np.float32)
    store = write_store(tmp_path / "s", G)
    view = LazyColumns(store, np.arange(M), fill=np.zeros(M, dtype=np.float32))
    disk = SessionTable(identity="t", file_identity="f", mode="disk",
                        parsed=SimpleNamespace(store=store), view=view,
                        raw=LazyColumns(store, np.arange(M)))
    memory = SessionTable(identity="t", file_identity="f", mode="memory", view=G, raw=G)
    ld = gp.reader(disk.view, np.float64)
    assert not store.closed and not gp.table_released(disk) and not gp.table_released(memory)
    assert ld[:, np.arange(3)].shape == (N, 3)
    store.close()
    assert store.closed and gp.table_released(disk)
    assert gp.table_released(SessionTable(identity="t", file_identity="f", mode="disk", view=view))
    assert not gp.table_released(memory) and not gp.table_released(None)
    with pytest.raises(ValueError, match="is closed"):
        ld[:, np.arange(3)]
    with pytest.raises(AttributeError):
        store.closed = False                         # read only

    class Released(Exception):
        pass

    def on_released():
        raise Released()

    with pytest.raises(Released):                    # the table given, or read when it fails
        with gp.released_guard(disk, on_released):
            ld[:, np.arange(3)]
    with pytest.raises(Released):
        with gp.released_guard(lambda: disk, on_released):
            ld[:, np.arange(3)]
    with pytest.raises(ValueError, match="another error"):
        with gp.released_guard(memory, on_released):
            raise ValueError("another error")
    with pytest.raises(KeyError):
        with gp.released_guard(disk, on_released):
            raise KeyError("not a read")


# ---------------------------------------------------- one LD-block detection, shared
def _counting(find):
    """``find`` wrapped (its signature kept), counting its runs."""
    import functools

    runs = []

    @functools.wraps(find)
    def counted(**kw):
        runs.append(kw)
        return find(**kw)

    return counted, runs


def _page_and_tab_calls(example_v1, post):
    """The page's automatic-block call and the LD Blocks tab's call at its defaults, with the
    arguments each passes (pages/Post_GWAS_Analysis.py, pages/_ld_tabs/tab_genome_wide.py):
    the page the aligned rows, the window as an int and no gap_factor; the tab all rows, the
    window as a float and gap_factor written out. LD r2 0.1 and a 200 kb window, where
    example_v1 forms blocks (it forms none at 0.6)."""
    data = dict(gwas_df=example_v1["gwas_df"], chroms=example_v1["chroms"],
                positions=example_v1["positions"], sid=example_v1["sid"])
    page = dict(data, geno_imputed=post.ld_aligned, ld_threshold=0.1, flank_kb=200, min_snps=2,
                top_n=10, sig_thresh=1e-5, max_dist_bp=None, ld_decay_kb=None, adj_r2_min=0.2)
    tab = dict(data, geno_imputed=post.ld, ld_threshold=0.1, flank_kb=200.0, min_snps=2, top_n=10,
               sig_thresh=1e-5, max_dist_bp=None, ld_decay_kb=None, adj_r2_min=0.2, gap_factor=10.0)
    return page, tab


@pytest.mark.parametrize("mode", ["memory", "disk"])
def test_the_tabs_default_detection_is_the_pages(example_v1, mode):
    """The page's automatic blocks and the LD Blocks tab's detection at its defaults are one
    detection when every sample is aligned: the tab is served the page's result, equal to a
    detection of its own (values, dtypes, attrs); after a detection with another r2, the
    default is served again."""
    from gwas.ld import find_ld_clusters_genomewide

    post = example_v1["posts"][mode]
    assert example_v1["keep"].all()                  # every sample has the trait
    find, runs = _counting(find_ld_clusters_genomewide)
    shared = gp.SharedDetection({}, find, post.identity)
    page_kw, tab_kw = _page_and_tab_calls(example_v1, post)
    assert shared.key(page_kw) == shared.key(tab_kw)
    page = shared.find_ld_clusters_genomewide(**page_kw)
    tab = shared.find_ld_clusters_genomewide(**tab_kw)
    assert len(runs) == 1, "the tab's detection at its defaults ran again"
    own = find_ld_clusters_genomewide(**tab_kw)
    assert len(own) > 0, "no LD block: the comparison would be vacuous"
    for got in (page, tab):
        pd.testing.assert_frame_equal(got, own, check_exact=True)
        assert got.to_csv(index=False) == own.to_csv(index=False) and got.attrs == own.attrs
    assert tab is not page
    # the tab's r2 changed: a detection of its own; set back: the shared result again
    shared.find_ld_clusters_genomewide(**dict(tab_kw, ld_threshold=0.5))
    assert len(runs) == 2
    back = shared.find_ld_clusters_genomewide(**tab_kw)
    assert len(runs) == 2
    assert back.to_csv(index=False) == own.to_csv(index=False)


_CHANGES = ["ld_threshold", "flank_kb", "adj_r2_min", "rows", "gwas", "labels", "table",
            "function", "top_n", "sig_thresh", "gap_factor", "min_pair_n"]


@pytest.mark.parametrize("change", _CHANGES)
def test_a_detection_with_other_inputs_is_not_shared(example_v1, change):
    """Any input that differs (the tab's r2, window or adj_r2_min, the rows, the GWAS results,
    the labels, the table, the function, the seeds, the gap, a default): another key, and
    both detections run."""
    from gwas.ld import find_ld_clusters_genomewide

    post = example_v1["posts"]["memory"]
    find, runs = _counting(find_ld_clusters_genomewide)
    runs2 = []
    state = {}
    first = gp.SharedDetection(state, find, post.identity)
    kw = _page_and_tab_calls(example_v1, post)[1]
    other, kw2 = first, dict(kw)
    if change in ("ld_threshold", "adj_r2_min", "sig_thresh"):
        kw2[change] = {"ld_threshold": 0.5, "adj_r2_min": 0.3, "sig_thresh": 1e-6}[change]
    elif change == "flank_kb":
        kw2["flank_kb"] = 200.5                      # another window in base pairs
    elif change == "rows":
        keep = np.ones(post.shape[0], dtype=bool)
        keep[3] = False
        kw2["geno_imputed"] = post.ld.lazy_rows(keep)
    elif change == "gwas":
        g = kw["gwas_df"].copy()
        g.loc[g.index[1], "PValue"] = 1e-9
        kw2["gwas_df"] = g
    elif change == "labels":
        p = np.asarray(kw["positions"]).copy()
        p[0] += 1
        kw2["positions"] = p
    elif change == "table":
        other = gp.SharedDetection(state, find, "another table")
    elif change == "function":
        find2, runs2 = _counting(find_ld_clusters_genomewide)
        other = gp.SharedDetection(state, find2, post.identity)
    elif change == "top_n":
        kw2["top_n"] = 9
    elif change == "gap_factor":
        kw2["gap_factor"] = 5.0
    elif change == "min_pair_n":
        kw2["min_pair_n"] = 15                       # a default neither caller passes
    assert other.key(kw2) != first.key(kw)
    first.find_ld_clusters_genomewide(**kw)
    other.find_ld_clusters_genomewide(**kw2)
    assert len(runs) + len(runs2) == 2


def test_the_window_is_the_only_flank_use():
    """The key takes flank_kb as the base pairs int(round(float(flank_kb) * 1000.0)): pin that
    this is the detection's only use of it, and that equal base pairs share a key."""
    import ast
    import inspect

    from gwas.ld import find_ld_clusters_genomewide

    tree = ast.parse(inspect.getsource(find_ld_clusters_genomewide))
    uses = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "flank_kb"]
    assert len(uses) == 1, "find_ld_clusters_genomewide reads flank_kb more than once"
    assign = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
              and any(getattr(t, "id", None) == "flank_bp" for t in n.targets)]
    assert len(assign) == 1
    assert ast.unparse(assign[0].value) == "int(round(float(flank_kb) * 1000.0))"
    shared = gp.SharedDetection({}, find_ld_clusters_genomewide, "T")
    assert shared.key({"flank_kb": 300}) == shared.key({"flank_kb": 300.0}) != shared.key({"flank_kb": 300.5})


def test_the_detection_cache_is_bounded():
    state = {}
    shared = gp.SharedDetection(state, lambda **kw: pd.DataFrame({"x": [kw["top_n"]]}), "T")
    for i in range(gp.RESULT_CACHE_MAX + 5):
        shared.find_ld_clusters_genomewide(geno_imputed=None, top_n=i)
    assert len(state[gp.DETECTION_CACHE]) == gp.RESULT_CACHE_MAX
    assert shared.find_ld_clusters_genomewide(geno_imputed=None, top_n=gp.RESULT_CACHE_MAX + 4)["x"].tolist() == [
        gp.RESULT_CACHE_MAX + 4]


# ------------------------------------------------- the significant-SNP table, built once
def test_the_significant_snp_table_is_built_once_per_input(example_v1, monkeypatch):
    """The Significant SNPs tab builds its table once per input and serves reruns from a
    bounded session cache with the same CSV; another table, GWAS results, blocks or rule
    build again, and the first input is still served."""
    import streamlit as st

    import gwas.sigtable as sigtable
    from gwas.ld import find_ld_clusters_genomewide
    from pages._ld_tabs import LDContext
    from pages._ld_tabs import tab_significant_snps as T

    assert gp.SIGTABLE_CACHE == "_sigtable_cache"   # the literal the tab writes
    built, downloads = [], []
    real = sigtable.build_significant_snp_table

    def build(*a, **k):
        built.append(1)
        return real(*a, **k)

    monkeypatch.setattr(sigtable, "build_significant_snp_table", build)
    monkeypatch.setattr(st, "download_button",
                        lambda label, data, *a, **k: downloads.append((label, data)) or False)
    st.session_state.pop(gp.SIGTABLE_CACHE, None)
    post = example_v1["posts"]["disk"]

    def ctx(**over):
        base = dict(geno_ld=post.ld, geno_hard=post.hard, chroms=example_v1["chroms"],
                    positions=example_v1["positions"], sid=example_v1["sid"], geno_df=post.samples,
                    pheno_df=example_v1["pheno"], trait_col="Trait1", ph_aligned=example_v1["pheno"],
                    y_aligned=None, keep_mask=example_v1["keep"], geno_ld_aligned=post.ld_aligned,
                    gwas_df=example_v1["gwas_df"], haplo_df_auto=pd.DataFrame(), ld_decay_kb=None,
                    adj_r2_min_global=0.2, ld_trait="Trait1", geno_dosage_raw=post.raw,
                    meff_val=None, sig_rule_label="Bonferroni (α = 0.05)",
                    table_identity=post.identity)
        base.update(over)
        return LDContext(**base)

    def csv():
        return [d for d in downloads if d[0] == "Download Significant_SNPs.csv"][-1][1]

    try:
        T.render(ctx())
        first = csv()
        T.render(ctx())                              # a rerun: a new context, nothing changed
        assert len(built) == 1 and csv() == first
        T.render(ctx(table_identity="another table"))
        assert len(built) == 2
        gwas2 = example_v1["gwas_df"].copy()
        gwas2.loc[gwas2.index[1], "PValue"] = 1e-12
        T.render(ctx(gwas_df=gwas2))
        assert len(built) == 3
        blocks = find_ld_clusters_genomewide(
            gwas_df=example_v1["gwas_df"], chroms=example_v1["chroms"],
            positions=example_v1["positions"], geno_imputed=post.ld, sid=example_v1["sid"],
            ld_threshold=0.1, flank_kb=200.0)
        assert len(blocks) > 0
        T.render(ctx(haplo_df_auto=blocks))
        assert len(built) == 4 and csv() != first    # SNPs now in blocks
        T.render(ctx(sig_rule_label="FDR (q < 0.05)"))
        assert len(built) == 5
        T.render(ctx())                              # the first input again: still kept
        assert len(built) == 5 and csv() == first
        assert len(st.session_state[gp.SIGTABLE_CACHE]) <= gp.RESULT_CACHE_MAX
    finally:
        st.session_state.pop(gp.SIGTABLE_CACHE, None)


# --------------------------------------------------------------- the session keys (Clear)
_FILES = ["pages/Post_GWAS_Analysis.py"] + sorted(
    str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "pages" / "_ld_tabs").glob("*.py"))
_WRITE = re.compile(r"""st\.session_state\[\s*(?P<key>[^\]]+?)\s*\]\s*=(?!=)"""
                    r"""|st\.session_state\.setdefault\(\s*(?P<key2>[^,\)]+)"""
                    r"""|bounded\(\s*st\.session_state\s*,\s*(?P<key3>[^,\)]+)"""
                    r"""|remember_family\(\s*st\.session_state\s*,\s*(?P<key4>[^,\)]+)\s*,\s*(?P<key5>[^,\)]+)""")
# the keys written through a variable, and where its value comes from
_VARIABLE_KEYS = {"hap_key": "HAPGWAS::", "ld_key": "LDDETECT::", "_assign_key": "HAPGWAS::",
                  'f"{hap_key}::tables"': "HAPGWAS::", "cache_namespace": None}


def test_the_session_key_lists_cover_every_write():
    sources = {rel: "\n".join(line for line in (ROOT / rel).read_text(encoding="utf-8").splitlines()
                              if not line.lstrip().startswith("#"))       # code, not comments
               for rel in _FILES}
    # widget keys (key="..." of a widget) are excluded; one is seeded through the session
    widget_keys = {k for src in sources.values() for k in re.findall(r'\bkey="([^"{]+)"', src)}
    seen = set()
    for rel, src in sources.items():
        for m in _WRITE.finditer(src):
            for raw in (m.group("key"), m.group("key2"), m.group("key3"), m.group("key4"), m.group("key5")):
                if raw is None:
                    continue
                raw = raw.strip()
                lit = re.fullmatch(r"""["']([^"']+)["']""", raw)
                if lit:
                    if lit.group(1) in widget_keys:
                        continue
                    seen.add(lit.group(1))
                    assert lit.group(1) in gp.POST_GWAS_SESSION_KEYS, (rel, lit.group(1))
                else:
                    assert raw in _VARIABLE_KEYS, (rel, raw)
                    if _VARIABLE_KEYS[raw]:
                        assert _VARIABLE_KEYS[raw] in gp.POST_GWAS_SESSION_PREFIXES
    # cache_namespace is "_hap_box_cache" (its only caller), listed with the others
    src = (ROOT / "pages/_ld_tabs/tab_genome_wide.py").read_text(encoding="utf-8")
    assert re.findall(r'cache_namespace="([^"]+)"', src) == ["_hap_box_cache"]
    seen.add("_hap_box_cache")
    # SharedDetection (gwas/guipost.py) keeps its results under DETECTION_CACHE in the session
    # the page and the tab pass it
    for rel in ("pages/Post_GWAS_Analysis.py", "pages/_ld_tabs/tab_genome_wide.py"):
        assert "SharedDetection(\n" in sources[rel] and "st.session_state, find_ld_clusters_genomewide" in sources[rel]
    seen.add(gp.DETECTION_CACHE)
    # the stray copies are gone, and every listed key is written somewhere
    assert not {"geno_hard", "ld_geno_hard"} & seen
    assert set(gp.POST_GWAS_SESSION_KEYS) <= seen, set(gp.POST_GWAS_SESSION_KEYS) - seen


def test_no_table_reaches_a_tab_whole():
    """The page builds no whole-table copy: no np.asarray / np.rint / .copy() of the table, and
    the tabs read the context's tables only by index (no np.asarray of a whole table)."""
    page = (ROOT / "pages/Post_GWAS_Analysis.py").read_text(encoding="utf-8")
    assert 'st.session_state["geno_imputed"]' not in page
    assert "np.rint(geno_ld)" not in page and "geno_ld.copy()" not in page
    for rel in _FILES[1:]:
        src = (ROOT / rel).read_text(encoding="utf-8")
        for name in ("geno_ld", "geno_hard", "geno", "geno_ld_aligned"):
            assert f"np.asarray({name})" not in src, (rel, name)
            assert f"np.asarray({name}, float)" not in src, (rel, name)
