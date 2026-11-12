"""Multi-trait CLI runs (reviewer point 2-INT.2): ``--trait A B C`` runs every trait
through the same per-trait code as a single-trait run (gwas.pipeline), each on its
own phenotyped samples.

Planted causal SNPs, one per trait on a different chromosome (tests/_multitrait_data):
A. each trait's ZIP from the multi-trait run is byte-identical to that trait's
   single-trait run (all model p-values, LD blocks, haplotype tables, LD heatmaps,
   metadata; only the report's "Generated" time line may differ), and its top hit
   is on the planted chromosome;
B. traits with different missing samples are each analysed on their own samples
   (n equals the single-trait run), with no abort;
C. a constant trait fails without stopping the others: the combined ZIP holds the
   completed traits plus failed_traits.txt, and the run exits non-zero.
"""
import io
import json
import zipfile

import pandas as pd
import pytest

from cli import _build_parser, run_pipeline
from tests import _multitrait_data as mt

_GENERATED = "<strong>Generated:</strong>"
_PLANTED = list(mt.CAUSAL)


def _args(vcf, pheno, traits, out):
    # all three models, so "all model p-values" are compared
    return _build_parser().parse_args(
        ["--vcf", str(vcf), "--pheno", str(pheno), "--trait", *traits, "--output", str(out),
         "--model", "mlm", "mlmm", "farmcpu"])


def _members(zip_bytes):
    """Member name -> bytes, with the report's 'Generated:' line dropped."""
    out = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            data = zf.read(name)
            if name.endswith("report.html"):
                data = b"\n".join(l for l in data.split(b"\n") if _GENERATED.encode() not in l)
            out[name] = data
    return out


def _trait_zip(folder, trait):
    hits = sorted(folder.glob(f"GWAS_*__{trait}__*.zip"))
    assert len(hits) == 1, (trait, [p.name for p in folder.iterdir()])
    return hits[0].read_bytes()


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("multitrait_cli")
    vcf, pheno, info = mt.write(base)
    multi = base / "multi"
    # the constant trait sits second, so traits run both before and after a failure
    order = [_PLANTED[0], mt.CONSTANT_TRAIT, *_PLANTED[1:]]
    with pytest.raises(SystemExit) as ei:
        run_pipeline(_args(vcf, pheno, order, multi))
    singles = {}
    for t in _PLANTED:
        out = base / f"single_{t}"
        run_pipeline(_args(vcf, pheno, [t], out))
        singles[t] = _trait_zip(out, t)
    return {"multi": multi, "exit": ei.value, "singles": singles, "info": info}


def test_each_trait_matches_its_single_trait_run_byte_for_byte(runs):
    for t in _PLANTED:
        got, ref = _members(_trait_zip(runs["multi"], t)), _members(runs["singles"][t])
        assert sorted(got) == sorted(ref), t
        differ = [n for n in ref if got[n] != ref[n]]
        assert not differ, f"{t}: multi-trait members differ from the single-trait run: {differ}"
        # the comparison covers the outputs the brief names
        names = "\n".join(ref)
        for table in (f"GWAS_{t}", f"GWAS_MLMM_{t}", f"GWAS_FarmCPU_{t}"):
            assert f"tables/{table}.csv" in ref, table
        assert "LD_blocks_annotated_MLM" in names and "figures/LD_heatmap_" in names


def test_top_hit_is_on_the_planted_chromosome(runs):
    for t in _PLANTED:
        with zipfile.ZipFile(io.BytesIO(_trait_zip(runs["multi"], t))) as zf:
            gwas = pd.read_csv(io.BytesIO(zf.read(f"tables/GWAS_{t}.csv")))
        top = gwas.sort_values("PValue").iloc[0]
        assert str(top["Chr"]) == runs["info"][t]["chrom"], (t, top["SNP"])


def test_traits_with_different_missing_samples_use_their_own_samples(runs):
    for t in _PLANTED:
        metas = []
        for zb in (_trait_zip(runs["multi"], t), runs["singles"][t]):
            with zipfile.ZipFile(io.BytesIO(zb)) as zf:
                metas.append(json.loads(zf.read("run_metadata.json")))
        assert metas[0]["Samples"] == metas[1]["Samples"] == runs["info"][t]["n"], t
        assert metas[0]["Trait"] == t


def test_a_failing_trait_does_not_stop_the_others(runs):
    exit_msg = str(runs["exit"].code)
    assert runs["exit"].code not in (0, None)
    assert "1 of 4 traits failed" in exit_msg and mt.CONSTANT_TRAIT in exit_msg
    assert not list(runs["multi"].glob(f"GWAS_*__{mt.CONSTANT_TRAIT}__*.zip"))
    combined = sorted(runs["multi"].glob("GWAS_*__MultiTrait_*.zip"))
    assert len(combined) == 1 and "MultiTrait_3traits" in combined[0].name
    with zipfile.ZipFile(combined[0]) as zf:
        names = zf.namelist()
        report = zf.read("failed_traits.txt").decode()
        for t in _PLANTED:
            inner = _members(_trait_zip(runs["multi"], t))
            mine = {n.split("/", 1)[1]: zf.read(n) for n in names if n.startswith(f"{t}/")}
            mine = {n: (b"\n".join(l for l in d.split(b"\n") if _GENERATED.encode() not in l)
                        if n.endswith("report.html") else d) for n, d in mine.items()}
            assert mine == inner, t
    assert not any(n.startswith(f"{mt.CONSTANT_TRAIT}/") for n in names)
    assert f"{mt.CONSTANT_TRAIT}: FAILED" in report and "near-zero variance" in report
    assert all(f"{t}: completed" in report for t in _PLANTED)


def test_duplicate_trait_names_are_refused_before_any_work(tmp_path):
    vcf, pheno, _ = mt.write(tmp_path)
    with pytest.raises(SystemExit) as ei:
        run_pipeline(_args(vcf, pheno, ["T_chr1", "T_chr1"], tmp_path / "out"))
    assert "more than once" in str(ei.value.code)
