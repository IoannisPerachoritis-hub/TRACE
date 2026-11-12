"""Post-GWAS stages with the genotype table on disk: every read site gives today's operand,
and LD-block detection gives today's blocks.

The read sites (gwas/ld.py, gwas/plotting.py, gwas/sigtable.py, gwas/haplotype.py and the LD
decay, triage and box-plot steps of gwas/pipeline.py) now take ``as_columns(table, float)``
where they took ``np.asarray(table, float)``. For an array that is the same expression; for a
table on disk the columns are read as the site asks for them. Each form is checked with 1-, 2-
and 3-column operands against today's expression on the in-memory table: values, number
format and layout (BLAS takes another path for another layout).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gwas.ld as ld
from gwas.genostore import LazyColumns, as_columns, is_lazy, write_store

N, M = 36, 400


def _tables(tmp_path, seed=1):
    rng = np.random.default_rng(seed)
    G = rng.integers(0, 3, size=(N, M)).astype(np.float32)
    G[rng.random((N, M)) < 0.05] = np.nan
    fill = np.nan_to_num(np.nanmean(G.astype(np.float64), axis=0)).astype(np.float32)
    imp = np.asfortranarray(np.where(np.isnan(G), fill, G).astype(np.float32))
    raw = np.asfortranarray(G)
    st = write_store(tmp_path / "s", G)
    return st, raw, imp, LazyColumns(st, np.arange(M)), LazyColumns(st, np.arange(M), fill=fill)


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape
    assert np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()
    assert a.flags["C_CONTIGUOUS"] == b.flags["C_CONTIGUOUS"]
    assert a.flags["F_CONTIGUOUS"] == b.flags["F_CONTIGUOUS"]


ROWS = np.ones(N, dtype=bool)
ROWS[[1, 4, 20]] = False

# (site, expression with today's array, expression with the table on disk); ``idx`` is the
# site's column selection, ``j`` one column, ``chm`` a chromosome mask and ``reg`` a window
# mask within it, as at the site.
SITES = {
    "ld.block_mean_r2 / merge_coherence / compute_block_ld_quality / extract_block_geno":
        (lambda A, s: np.asarray(A, dtype=float)[:, s["idx"]],
         lambda L, s: as_columns(L, float)[:, s["idx"]]),
    "ld.find_ld_clusters_genomewide window":
        (lambda A, s: np.asarray(np.asarray(A, float)[:, s["chm"]], float)[:, s["reg"]],
         lambda L, s: as_columns(as_columns(L, float), float)[:, np.flatnonzero(s["chm"])[s["reg"]]]),
    "plotting.plot_ld_block_heatmap":
        (lambda A, s: np.asarray(A)[:, s["idx"]], lambda L, s: as_columns(L)[:, s["idx"]]),
    "plotting.compute_r2_to_lead (lead column)":
        (lambda A, s: np.asarray(A, dtype=float)[:, s["j"]], lambda L, s: as_columns(L, float)[:, s["j"]]),
    "sigtable window":
        (lambda A, s: np.asarray(A, float)[:, s["idx"]], lambda L, s: as_columns(L, float)[:, s["idx"]]),
    "haplotype rows then block columns":
        (lambda A, s: np.asarray(A)[ROWS, :][:, s["idx"]], lambda L, s: as_columns(L)[ROWS, :][:, s["idx"]]),
    "pipeline LD decay":
        (lambda A, s: A.astype(float)[:, s["chm"]][:, s["reg"]],
         lambda L, s: L.astype(float)[:, np.flatnonzero(s["chm"])[s["reg"]]]),
    "pipeline triage / box plots (one SNP)":
        (lambda A, s: np.asarray(A, float)[:, s["j"]], lambda L, s: as_columns(L, float)[:, s["j"]]),
}


def _selection(width, offset):
    chm = np.zeros(M, dtype=bool)
    chm[100:250] = True
    reg = np.zeros(int(chm.sum()), dtype=bool)
    reg[offset:offset + width] = True
    idx = np.arange(100 + offset, 100 + offset + width)
    return {"idx": idx, "j": int(idx[-1]), "chm": chm, "reg": reg}


@pytest.mark.parametrize("site", list(SITES))
@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("table", ["raw", "imputed"])
def test_read_sites_give_todays_operand(tmp_path, site, width, table):
    st, raw, imp, Lraw, Limp = _tables(tmp_path)
    try:
        A, L = (raw, Lraw) if table == "raw" else (imp, Limp)
        today, disk = SITES[site]
        for offset in (0, 37, 150 - width):
            s = _selection(width, offset)
            got = disk(L, s)
            assert not is_lazy(got)
            _same(got, today(A, s))
    finally:
        st.close()


@pytest.mark.parametrize("width", [2, 3, 50])
def test_pairwise_r2_on_read_windows(tmp_path, width):
    st, raw, imp, Lraw, Limp = _tables(tmp_path)
    try:
        idx = np.arange(10, 10 + width)
        _same(ld.pairwise_r2(as_columns(Lraw, float)[:, idx]), ld.pairwise_r2(np.asarray(raw, float)[:, idx]))
        _same(ld.pairwise_r2(as_columns(Limp, float)[:, idx]), ld.pairwise_r2(np.asarray(imp, float)[:, idx]))
    finally:
        st.close()


@pytest.mark.parametrize("piece", ["1", "2", "3", "4096"])
def test_sigtable_maf_in_blocks(tmp_path, monkeypatch, piece):
    """build_significant_snp_table's per-SNP MAF read in blocks equals the whole-table value."""
    st, raw, imp, Lraw, Limp = _tables(tmp_path)
    try:
        monkeypatch.setenv("TRACE_STORE_PIECE", piece)
        from gwas.genostore import piece as _piece
        g, p = Lraw.astype(float), _piece(4096)
        blocked = np.concatenate([ld.maf_from_matrix(g[:, k:k + p], "dosage012") for k in range(0, M, p)])
        _same(blocked, ld.maf_from_matrix(np.asarray(raw, float), "dosage012"))
    finally:
        st.close()


def _gwas_df(chroms, positions, sid, leads):
    p = np.full(len(sid), 0.5)
    for j, pv in leads.items():
        p[j] = pv
    return pd.DataFrame({"SNP": sid, "Chr": chroms, "Pos": positions, "PValue": p})


def _ld_example():
    """The LD example (examples/): 100 lines, 600 SNPs, blocks at the defaults."""
    from gwas.io import load_vcf_cached
    from gwas.pipeline import ParsedGenotypes

    load = getattr(load_vcf_cached, "__wrapped__", load_vcf_cached)
    vcf = Path(__file__).resolve().parents[1] / "examples" / "example.vcf.gz"
    return vcf, ParsedGenotypes.from_callset(load(vcf.read_bytes()))


@pytest.mark.parametrize("cache_max", [1, 8, 10**6])
def test_ld_blocks_equal_on_disk_and_with_a_bounded_cache(tmp_path, monkeypatch, cache_max):
    """find_ld_clusters_genomewide on the table on disk equals today's on the array, and keeping
    fewer r2 windows in its cache (a miss recomputes the window) changes nothing either."""
    from gwas.genostore import ingest_vcf

    vcf, parsed = _ld_example()
    G = np.asfortranarray(np.asarray(parsed.genotypes))
    fill = np.nan_to_num(np.nanmean(G.astype(np.float64), axis=0)).astype(np.float32)
    imp = np.asfortranarray(np.where(np.isnan(G), fill, G).astype(np.float32))
    chroms = np.asarray(parsed.chroms).astype(str)
    positions = np.asarray(parsed.positions)
    sid = np.array([f"s{j}" for j in range(G.shape[1])])
    # many seeds: every 7th SNP significant, several sharing windows
    leads = {j: 10.0 ** -(6 + (j % 5)) for j in range(0, G.shape[1], 7)}
    gwas_df = _gwas_df(chroms, positions, sid, leads)
    kw = dict(ld_threshold=0.6, flank_kb=150, min_snps=2, top_n=0, sig_thresh=1e-5)

    ref = ld.find_ld_clusters_genomewide(gwas_df=gwas_df, chroms=chroms, positions=positions,
                                         geno_imputed=imp.astype(float), sid=sid, **kw)
    assert len(ref) > 0
    monkeypatch.setattr(ld, "_R2_CACHE_MAX", cache_max)
    mem = ld.find_ld_clusters_genomewide(gwas_df=gwas_df, chroms=chroms, positions=positions,
                                         geno_imputed=imp.astype(float), sid=sid, **kw)
    pd.testing.assert_frame_equal(mem, ref, check_exact=True)

    p = ingest_vcf(vcf, tmp_path / "s")
    try:
        L = LazyColumns(p.store, np.arange(p.store.M), fill=fill)
        disk = ld.find_ld_clusters_genomewide(gwas_df=gwas_df, chroms=chroms, positions=positions,
                                              geno_imputed=L.astype(float), sid=sid, **kw)
        pd.testing.assert_frame_equal(disk, ref, check_exact=True)
        q_ref = ld.compute_block_ld_quality(ref, chroms, positions, sid, imp, gwas_df)
        q_disk = ld.compute_block_ld_quality(disk, chroms, positions, sid, L, gwas_df)
        pd.testing.assert_frame_equal(q_disk, q_ref, check_exact=True)
    finally:
        p.store.close()


def test_stage_marks_write_rows_only_when_asked(tmp_path):
    """gwas.stagelog (TRACE_STAGE_LOG): one CSV row per mark, then END; nothing without it."""
    import subprocess
    import sys

    out = tmp_path / "marks.csv"
    code = ("from gwas import stagelog; stagelog.mark('A'); stagelog.mark('B')")
    root = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ, TRACE_STAGE_LOG=str(out), TRACE_STAGE_SAMPLE_MS="5", PYTHONPATH=root)
    subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=root)
    rows = out.read_text().splitlines()
    assert rows[0].startswith("t_s,stage,ws_mb")
    assert [r.split(",")[1] for r in rows[1:]] == ["A", "B", "END"]
    env.pop("TRACE_STAGE_LOG")
    subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=root)
    assert out.read_text().splitlines() == rows
