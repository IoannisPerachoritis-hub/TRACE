"""T-36 — LD-triage CLI wiring: additive escape hatch, pinned flags, emission."""
import ast
import io
import pathlib
import zipfile

import numpy as np
import pandas as pd

from cli import _build_parser, run_pipeline
from tests.test_cli_e2e import _write_test_data

_SKIP = ("report.html", "run_metadata.json")


def _run(data_dir, out_dir, extra):
    data_dir.mkdir(parents=True, exist_ok=True)
    vcf, pheno = _write_test_data(data_dir)
    args = _build_parser().parse_args([
        "--vcf", str(vcf), "--pheno", str(pheno), "--trait", "TestTrait", "--output", str(out_dir),
        "--model", "mlm", "--no-report", "--no-plots", "--maf", "0.01", "--mac", "1", "--n-pcs", "2", *extra])
    run_pipeline(args)
    zp = list(out_dir.glob("*.zip"))[0]
    with zipfile.ZipFile(zp) as zf:
        return {n: zf.read(n) for n in zf.namelist()}


def test_triage_additive_and_escape_hatch(tmp_path):
    """triage ON vs --no-triage: every pre-existing member byte-identical; the only
    new members are LD_triage_* (N1)."""
    d = tmp_path / "data"
    on = _run(d, tmp_path / "on", [])
    off = _run(d, tmp_path / "off", ["--no-triage"])
    new = set(on) - set(off)
    assert all("LD_triage" in n for n in new), f"unexpected new members: {new}"
    for n in set(on) & set(off):
        if any(s in n for s in _SKIP):
            continue
        assert on[n] == off[n], f"pre-existing member perturbed by triage: {n}"


def test_hap_and_triage_flag_defaults_pinned():
    """--hap-min-count/--hap-min-group-size default 5/3: a typo here IS type-c
    (repartitions MLGs -> moves eta2/F/p in Table S8)."""
    a = _build_parser().parse_args(["--vcf", "x", "--pheno", "y", "--trait", "t", "--output", "o"])
    assert a.hap_min_count == 5 and a.hap_min_group_size == 3
    assert a.triage_lead_r2_frac == 0.5 and a.triage_r2_coherent is None
    assert a.no_triage is False


def test_cli_threads_hap_flags_into_haplotype_call():
    tree = ast.parse(pathlib.Path("cli.py").read_text(encoding="utf-8"))
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if fn == "run_haplotype_block_gwas":
                kw = {k.arg for k in node.keywords}
                assert {"min_hap_count", "min_group_size"} <= kw, f"hap flags not threaded: {kw}"
                found = True
    assert found, "run_haplotype_block_gwas call not found"


def _planted(tmp):
    rng = np.random.default_rng(7)
    n, m = 90, 60
    G = rng.integers(0, 3, size=(n, m))
    G[:, 11] = G[:, 10]; G[:, 12] = G[:, 10]; G[:, 13] = G[:, 10]      # a real block
    y = 3.0 * G[:, 10] + rng.normal(0, 1.0, n)
    gt = {0: "0/0", 1: "0/1", 2: "1/1"}
    samples = [f"S{i}" for i in range(n)]
    lines = ["##fileformat=VCFv4.2", "##contig=<ID=SL2.50ch01>", "##contig=<ID=SL2.50ch02>",
             '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
             "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples)]
    for j in range(m):
        chrom = "SL2.50ch01" if j < 40 else "SL2.50ch02"
        lines.append("\t".join([chrom, str(1000 + (j % 40) * 5000), f"snp{j}", "A", "T", ".", "PASS", ".", "GT"]
                                + [gt[int(G[i, j])] for i in range(n)]))
    (tmp / "s.vcf").write_text("\n".join(lines) + "\n")
    pd.DataFrame({"ID": samples, "TestTrait": y}).to_csv(tmp / "p.csv", index=False)
    return tmp / "s.vcf", tmp / "p.csv"


def test_triage_emits_full_columns_on_signal(tmp_path):
    vcf, pheno = _planted(tmp_path)
    out = tmp_path / "out"
    args = _build_parser().parse_args([
        "--vcf", str(vcf), "--pheno", str(pheno), "--trait", "TestTrait", "--output", str(out),
        "--model", "mlm", "--no-report", "--no-plots", "--maf", "0.01", "--mac", "1", "--n-pcs", "1",
        "--sig-thresh", "fdr", "--ld-seed-p", "0.01"])
    run_pipeline(args)
    with zipfile.ZipFile(list(out.glob("*.zip"))[0]) as zf:
        member = next(n for n in zf.namelist() if "LD_triage_MLM" in n)
        df = pd.read_csv(io.BytesIO(zf.read(member)))
    assert len(df) >= 1
    for grp in ("ldq_", "mlg_", "eta2_", "triage_"):
        assert any(c.startswith(grp) for c in df.columns), f"missing {grp}* columns"
    # N8: every eta2-bearing table carries the sample-count companions
    assert {"n_tested_haplotypes", "n_samples_tested"} <= set(df.columns)
    assert df["triage_reason_code"].notna().all()
