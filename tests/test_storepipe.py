"""The per-trait stages with the genotype table on disk (gwas/storepipe.py) against the
in-memory functions they replace, stage by stage, bit for bit.

Each case is run twice: today's path (load_vcf_cached, ParsedGenotypes, the gwas.qc stage
functions) and the store's (ingest_vcf, TraitStore), with the CLI's arguments. Every stage's
outputs must be identical, at piece sizes 1, 64 and 997 SNP columns per read. The synthetic
case holds what the stages filter: duplicate chr:pos records (one pair with tied MAF),
monomorphic, rare, high-missingness and heterozygous-excess SNPs, X and scaffold markers,
missing and non-numeric trait values, an imputation-quality field.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cli
import gwas.io as gio
from gwas import qc as gqc
from gwas.genostore import LazyColumns, ingest_vcf, is_lazy
from gwas.pipeline import ParsedGenotypes
from gwas.storepipe import ROW_COL, TraitStore, is_store_frame

REPO = Path(__file__).resolve().parents[1]
_load = getattr(gio.load_vcf_cached, "__wrapped__", gio.load_vcf_cached)


# ------------------------------------------------------------------------------- inputs
def _edge_files(tmp_path, n=30, seed=11):
    rng = np.random.default_rng(seed)
    samples = [f"L{i:03d}" for i in range(n)]
    lines = ["##fileformat=VCFv4.2",
             '##INFO=<ID=DR2,Number=1,Type=Float,Description="Dosage R2">',
             '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
             "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples)]

    def rec(ch, pos, alt, gts, vid=".", dr2=None):
        dr2 = rng.random() if dr2 is None else dr2
        lines.append(f"{ch}\t{pos}\t{vid}\tA\t{alt}\t.\tPASS\tDR2={dr2:.3f}\tGT\t" + "\t".join(gts))

    def gts(p_alt=0.3, miss=0.05, het=None):
        out = []
        for _ in range(n):
            if rng.random() < miss:
                out.append("./.")
                continue
            if het is not None and rng.random() < het:
                out.append("0/1")
                continue
            a = int(rng.random() < p_alt) + int(rng.random() < p_alt)
            out.append(["0/0", "0/1", "1/1"][a])
        return out

    for ch in ("chr1", "chr2", "chr3"):
        for j in range(60):
            pos = 10_000 + 1_500 * j
            kind = j % 12
            if kind == 0:
                rec(ch, pos, "T", ["0/0"] * n)                          # monomorphic
            elif kind == 1:
                rec(ch, pos, "T", gts(p_alt=0.02))                     # rare
            elif kind == 2:
                rec(ch, pos, "T", gts(miss=0.5))                       # high missingness
            elif kind == 3:
                rec(ch, pos, "T", gts(het=0.95, miss=0.0))             # heterozygous excess
            elif kind == 4:
                rec(ch, pos, "T", gts(), vid=f"rs{ch}{j}")
                rec(ch, pos, "G", gts(p_alt=0.45))                     # duplicate chr:pos
            elif kind == 5:
                g = gts()
                rec(ch, pos, "T", g)
                rec(ch, pos, "C", list(g))                             # duplicate, tied MAF
            elif kind == 6:
                rec(ch, pos, "T,G", gts())                             # multi-allelic: dropped
            elif kind == 7:
                rec(ch, pos, ".", gts())                               # no ALT: dropped
            else:
                rec(ch, pos, "T", gts(p_alt=float(rng.uniform(0.1, 0.5))))
    for j in range(8):
        rec("chrX", 5_000 + 100 * j, "T", gts())
        rec("scaffold_9", 7_000 + 100 * j, "T", gts())
    vcf = tmp_path / "edge.vcf"
    vcf.write_text("\n".join(lines) + "\n")
    trait = rng.normal(size=n).round(4).astype(object)
    trait[[2, 9]] = np.nan
    trait[5] = "x"
    pd.DataFrame({"T": trait, "U": rng.normal(size=n)}, index=pd.Index(samples, name="id")).to_csv(
        tmp_path / "edge_pheno.csv")
    return vcf, tmp_path / "edge_pheno.csv"


def _cases(tmp_path):
    ev, ep = _edge_files(tmp_path)
    return {
        "example_v1": (REPO / "tests" / "data" / "example_v1" / "example.vcf.gz",
                       REPO / "tests" / "data" / "example_v1" / "example_pheno.csv", "Trait1"),
        "example_v2": (REPO / "examples" / "example.vcf.gz", REPO / "examples" / "example_pheno.csv", "Trait2"),
        "edge": (ev, ep, "T"),
    }


def _args(**over):
    a = cli._build_parser().parse_args(["--vcf", "v", "--pheno", "p", "--trait", "T", "--output", "o"])
    for k, v in over.items():
        setattr(a, k, v)
    return a


ARGSETS = {
    "defaults": {},
    "strict": dict(maf=0.1, miss=0.2, mac=3, info_thresh=0.3, max_het=0.6, het_excess_p=1e-3,
                   ind_miss=0.3, norm="int"),
}


# ------------------------------------------------------------------------- the two paths
def _stages(parsed, pheno, trait, a, fns):
    """S05 to S09 of gwas.pipeline._run_trait, with ``fns`` the five stage functions."""
    harmonize, phenoqc, snpqc, matrices = fns
    out = {}
    geno_df, pheno, chroms, positions, sid, info, amap = harmonize(
        parsed.genotypes, parsed.samples, parsed.chroms, parsed.positions,
        parsed.vid, parsed.ref, parsed.alt, pheno, parsed.info_scores)
    out["S05"] = dict(geno_df=geno_df, pheno=pheno, chroms=chroms, positions=positions, sid=sid,
                      info=info, amap=amap)
    geno_df, pheno, y = phenoqc(geno_df, pheno, trait, a.norm, a.ind_miss)
    out["S06"] = dict(geno_df=geno_df, pheno=pheno, y=y)
    canonical = tuple(str(i) for i in range(1, a.n_chromosomes + 1)) if a.n_chromosomes else None
    geno_df, chroms, chroms_num, positions, sid, n_raw, qc_snp, info = snpqc(
        geno_df, chroms, positions, sid, a.maf, a.miss, a.mac, True, info, a.info_thresh,
        canonical=canonical, max_het=a.max_het, het_excess_p=a.het_excess_p)
    out["S08"] = dict(geno_df=geno_df, chroms=chroms, chroms_num=chroms_num, positions=positions,
                      sid=sid, n_raw=n_raw, qc_snp=qc_snp, info=info)
    raw, rate, imp, iid = matrices(geno_df, method=a.impute, impute_k=a.impute_k, impute_l=a.impute_l)
    out["S09"] = dict(raw=raw, rate=rate, imp=imp, iid=iid)
    return out


def _memory(vcf, pheno_path, trait, a):
    parsed = ParsedGenotypes.from_callset(_load(Path(vcf).read_bytes()))
    pheno = pd.read_csv(pheno_path, index_col=0, encoding="utf-8-sig")
    fns = (gqc._pipeline_harmonize_ids, gqc._pipeline_phenotype_qc, gqc._pipeline_snp_qc,
           gqc._pipeline_build_geno_matrices)
    return _stages(parsed, pheno, trait, a, fns)


def _store(vcf, pheno_path, trait, a, store_dir):
    parsed = ingest_vcf(vcf, store_dir)
    st = TraitStore(parsed)
    pheno = pd.read_csv(pheno_path, index_col=0, encoding="utf-8-sig")
    out = _stages(parsed, pheno, trait, a, (st.harmonize_ids, st.phenotype_qc, st.snp_qc,
                                            st.build_geno_matrices))
    return out, st, parsed


# ---------------------------------------------------------------------------- comparison
def _bits(a, b, what):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype, what
    assert a.shape == b.shape, what
    if a.dtype.kind in "fc":
        assert np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes(), what
    else:
        assert np.array_equal(a, b), what
    assert a.flags["F_CONTIGUOUS"] == b.flags["F_CONTIGUOUS"], what
    assert a.flags["C_CONTIGUOUS"] == b.flags["C_CONTIGUOUS"], what


def _same(a, b, what):
    if isinstance(b, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b, check_exact=True, obj=what)
    elif isinstance(b, pd.Index):
        assert a.equals(b) and a.dtype == b.dtype, what
    elif isinstance(b, dict):
        assert list(a) == list(b), what
        for k in b:
            _same(a[k], b[k], f"{what}[{k}]")
    elif isinstance(b, np.ndarray):
        _bits(a, b, what)
    elif isinstance(b, float):
        assert (a == b or (np.isnan(a) and np.isnan(b))) and type(a) is type(b), what
    else:
        assert a == b and type(a) is type(b), what


def _compare(mem, sto, st):
    for stage in ("S05", "S06", "S08"):
        m, s = mem[stage], sto[stage]
        gm, gs_ = m["geno_df"], s["geno_df"]
        assert is_store_frame(gs_) and not is_store_frame(gm)
        assert gs_.index.equals(gm.index) and gs_.index.dtype == gm.index.dtype, stage
        assert list(gs_.columns) == [ROW_COL]
        for k in m:
            if k != "geno_df":
                _same(s[k], m[k], f"{stage}.{k}")
    # the store frame's table, read whole, is today's frame
    pd.testing.assert_frame_equal(st.frame(sto["S08"]["geno_df"]), mem["S08"]["geno_df"], check_exact=True)
    m, s = mem["S09"], sto["S09"]
    assert is_lazy(s["raw"]) and is_lazy(s["imp"])
    _bits(np.asarray(s["raw"]), m["raw"], "S09.raw")
    _bits(np.asarray(s["imp"]), m["imp"], "S09.imp")
    _bits(s["rate"], m["rate"], "S09.rate")
    _bits(s["iid"], m["iid"], "S09.iid")


@pytest.mark.parametrize("argset", list(ARGSETS))
@pytest.mark.parametrize("case", ["example_v1", "example_v2", "edge"])
@pytest.mark.parametrize("piece", ["1", "64", "997"])
def test_early_stages_equal_in_memory(tmp_path, monkeypatch, case, argset, piece):
    monkeypatch.setenv("TRACE_STORE_PIECE", piece)
    vcf, pheno, trait = _cases(tmp_path)[case]
    a = _args(trait=trait, **ARGSETS[argset])
    mem = _memory(vcf, pheno, trait, a)
    sto, st, parsed = _store(vcf, pheno, trait, a, tmp_path / "s")
    try:
        _compare(mem, sto, st)
    finally:
        parsed.store.close()


def _kinship_pca_loco(out, kinship, chroms, positions):
    """S10 to S12: kinship, PCA and the LOCO kernels, as gwas.pipeline._run_trait calls them."""
    from gwas.kinship import _build_loco_kernels_impl, _compute_pcs_full_impl

    K, Z, chroms_grm, positions_grm, model = kinship(out["S08"]["geno_df"], out["S09"]["imp"],
                                                       chroms, positions)
    res = dict(K=K, Z=Z, chroms_grm=chroms_grm, positions_grm=positions_grm, model=model)
    try:      # today's PCA stops on a table with fewer than 20 SNPs: the same stop in both modes
        res["pcs"], res["eig"] = _compute_pcs_full_impl(Z, max_pcs=20)
    except ValueError as e:
        res["pcs_error"] = str(e)
        return res
    K0, K_by_chr, _ = _build_loco_kernels_impl(out["S09"]["iid"], Z, chroms_grm, K)
    res.update(K0=np.asarray(K0.val), K0_iid=np.asarray(K0.iid),
               K_by_chr={c: np.asarray(k.val) for c, k in sorted(K_by_chr.items())})
    return res


@pytest.mark.parametrize("argset", list(ARGSETS))
@pytest.mark.parametrize("case", ["example_v1", "example_v2", "edge", "tiny"])
@pytest.mark.parametrize("piece", ["1", "64", "997"])
def test_kinship_pca_loco_equal_in_memory(tmp_path, monkeypatch, case, argset, piece):
    monkeypatch.setenv("TRACE_STORE_PIECE", piece)
    if case == "tiny":            # few SNPs: the unpruned-set and identity-GRM branches
        vcf, pheno, trait = _cases(tmp_path)["edge"]
        lines = Path(vcf).read_text().splitlines()
        head = [l for l in lines if l.startswith("#")]
        body = [l for l in lines if not l.startswith("#")][:30]
        (tmp_path / "tiny.vcf").write_text("\n".join(head + body) + "\n")
        vcf = tmp_path / "tiny.vcf"
    else:
        vcf, pheno, trait = _cases(tmp_path)[case]
    a = _args(trait=trait, **ARGSETS[argset])
    mem = _memory(vcf, pheno, trait, a)
    sto, st, parsed = _store(vcf, pheno, trait, a, tmp_path / "s")
    try:
        km = _kinship_pca_loco(mem, gqc._pipeline_build_kinship, mem["S08"]["chroms"], mem["S08"]["positions"])
        ks = _kinship_pca_loco(sto, st.build_kinship, sto["S08"]["chroms"], sto["S08"]["positions"])
        for k in km:
            _same(ks[k], km[k], f"S10-S12.{k}")
        if case == "tiny":
            assert ks["model"] == "identity_GRM_low_snp_count" or ks["Z"].shape[1] < 40
    finally:
        parsed.store.close()


def test_edge_case_exercises_the_count_passes(tmp_path, monkeypatch):
    """The synthetic case drops samples and records, so the counts come from store passes,
    not from the ingest's totals (both routes are compared above)."""
    from gwas import genostore as gs

    gs.reset_read_log()
    vcf, pheno, trait = _cases(tmp_path)["edge"]
    sto, st, parsed = _store(vcf, pheno, trait, _args(trait=trait), tmp_path / "s")
    try:
        notes = [n[0] for n in gs.read_log().get("notes", []) if isinstance(n, (list, tuple))]
        assert "count_pass_rows" in notes and "count_pass_snps" in notes
        assert sto["S05"]["sid"].size < parsed.store.M          # duplicates collapsed
        assert sto["S08"]["qc_snp"]["Fail ANY"] > 0
    finally:
        parsed.store.close()


def test_leading_zero_ids_match_in_both_modes(tmp_path):
    vcf, pheno, trait = _cases(tmp_path)["edge"]
    p = pd.read_csv(pheno, index_col=0)
    p.index = [s.lstrip("L").lstrip("0") or "0" for s in p.index]       # 'L007' -> '7'
    text = Path(vcf).read_text().replace("\tL", "\t")                     # VCF samples '007'
    (tmp_path / "z.vcf").write_text(text)
    p.to_csv(tmp_path / "z_pheno.csv")
    a = _args(trait=trait)
    mem = _memory(tmp_path / "z.vcf", tmp_path / "z_pheno.csv", trait, a)
    sto, st, parsed = _store(tmp_path / "z.vcf", tmp_path / "z_pheno.csv", trait, a, tmp_path / "s")
    try:
        _compare(mem, sto, st)
        assert len(sto["S06"]["geno_df"]) > 20
    finally:
        parsed.store.close()


def test_ldknni_is_refused_on_disk(tmp_path):
    from gwas.pipeline import TraitPipelineError

    vcf, pheno, trait = _cases(tmp_path)["example_v1"]
    a = _args(trait=trait, impute="ldknni")
    with pytest.raises(TraitPipelineError, match="ldknni"):
        _store(vcf, pheno, trait, a, tmp_path / "s")


def test_lazy_tables_read_from_the_store(tmp_path):
    vcf, pheno, trait = _cases(tmp_path)["example_v1"]
    sto, st, parsed = _store(vcf, pheno, trait, _args(trait=trait), tmp_path / "s")
    try:
        raw = sto["S09"]["raw"]
        assert isinstance(raw, LazyColumns) and raw.layout == "F" and raw.fill is None
        assert sto["S09"]["imp"].fill is not None
    finally:
        parsed.store.close()
