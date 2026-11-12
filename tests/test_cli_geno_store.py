"""The CLI with the genotype table on disk (--geno-store disk): the stops come with the same
message, in the same order, as with the table in memory, and no run leaves its table behind.

The in-memory path checks the phenotype file and the trait names before it parses the VCF;
the chunked read parses as it reads, so a stop it meets while parsing is raised later, at the
place the in-memory path raises it. Each run makes its own new folder for the table (inside
--geno-store-dir, else --output) and removes only that folder, after the run and on any stop.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cli

STORE_GLOB = "_trace_geno_store_*"


def _run(argv, monkeypatch):
    """cli.main() in this process: ('ok', None) or (exception type, message)."""
    monkeypatch.setattr(sys, "argv", ["cli.py"] + [str(a) for a in argv])
    try:
        cli.main()
    except SystemExit as e:
        return "SystemExit", str(e.code)
    except Exception as e:  # noqa: BLE001 -- the comparison is the point
        return type(e).__name__, str(e)
    return "ok", None


def _vcf(path, n=30, m=200, header_only=False, sites_only=False, chrom_line=True, seed=2):
    rng = np.random.default_rng(seed)
    samples = [f"S{i:02d}" for i in range(n)]
    head = ["##fileformat=VCFv4.2", '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">']
    cols = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"
    if chrom_line:
        head.append(cols if sites_only else cols + "\tFORMAT\t" + "\t".join(samples))
    lines = list(head)
    if not header_only:
        for j in range(m):
            rec = f"chr{1 + j // 70}\t{1000 + 2000 * j}\t.\tA\tT\t.\tPASS\t."
            if not sites_only:
                p = rng.uniform(0.2, 0.5)
                g = ["0/0", "0/1", "1/1"]
                rec += "\tGT\t" + "\t".join(g[int(rng.random() < p) + int(rng.random() < p)] for _ in range(n))
            lines.append(rec)
    path.write_text("\n".join(lines) + "\n")
    pd.DataFrame({"T": rng.normal(size=n), "U": rng.normal(size=n)},
                 index=pd.Index(samples, name="id")).to_csv(path.with_name(path.stem + "_pheno.csv"))
    return path, path.with_name(path.stem + "_pheno.csv")


def _both(tmp_path, monkeypatch, vcf, pheno, trait, *extra):
    out = {}
    for mode in ("memory", "disk"):
        o = tmp_path / f"out_{mode}"
        out[mode] = _run(["--vcf", vcf, "--pheno", pheno, "--trait", trait, "--output", o,
                          "--geno-store", mode, "--no-plots", "--no-report", *extra], monkeypatch)
        assert not list(o.glob(STORE_GLOB)), f"{mode}: the table was left behind"
    return out


@pytest.mark.parametrize("case,trait", [
    ("header_only", "T"), ("header_only", "Misspelt"),
    ("sites_only", "T"), ("sites_only", "Misspelt"),
    ("no_chrom_line", "T"), ("no_chrom_line", "Misspelt"),
])
def test_stops_match_the_in_memory_path(tmp_path, monkeypatch, case, trait):
    vcf, pheno = _vcf(tmp_path / f"{case}.vcf", header_only=case == "header_only",
                      sites_only=case == "sites_only", chrom_line=case != "no_chrom_line")
    r = _both(tmp_path, monkeypatch, vcf, pheno, trait)
    assert r["memory"][0] != "ok"
    assert r["disk"] == r["memory"]


def test_unreadable_phenotype_file_stops_first_in_both_modes(tmp_path, monkeypatch):
    vcf, _ = _vcf(tmp_path / "h.vcf", header_only=True)
    bad = tmp_path / "bad_pheno.csv"
    bad.write_bytes(b"")
    r = _both(tmp_path, monkeypatch, vcf, bad, "T")
    assert r["memory"][0] != "ok" and r["disk"] == r["memory"]


def test_ldknni_is_refused_before_reading(tmp_path, monkeypatch):
    vcf, pheno = _vcf(tmp_path / "v.vcf")
    o = tmp_path / "o"
    kind, msg = _run(["--vcf", vcf, "--pheno", pheno, "--trait", "T", "--output", o,
                      "--geno-store", "disk", "--impute", "ldknni"], monkeypatch)
    assert kind == "SystemExit" and "ldknni" in msg and "--geno-store disk" in msg
    assert not list(o.glob(STORE_GLOB))


def _zip_members(folder):
    import zipfile

    zips = sorted(Path(folder).glob("*.zip"))
    assert zips
    out = {}
    for z in zips:
        with zipfile.ZipFile(z) as f:
            out.update({(z.name.split("__")[1] if "__" in z.name else z.name, n): f.read(n) for n in f.namelist()})
    return out


def test_a_run_on_disk_equals_memory_and_leaves_nothing(tmp_path, monkeypatch):
    vcf, pheno = _vcf(tmp_path / "v.vcf")
    r = _both(tmp_path, monkeypatch, vcf, pheno, "T")
    assert r == {"memory": ("ok", None), "disk": ("ok", None)}
    mem, disk = _zip_members(tmp_path / "out_memory"), _zip_members(tmp_path / "out_disk")
    assert mem.keys() == disk.keys()
    # every member identical, except the keys of run_metadata.json that say where the table was kept
    norm = _compare_script()._normalise
    differ = [k for k in mem if norm(k[1], mem[k]) != norm(k[1], disk[k])]
    assert differ == [], differ
    assert sorted(p.name for p in (tmp_path / "out_disk").iterdir()) == sorted(
        p.name for p in (tmp_path / "out_memory").iterdir())


def test_store_folder_inside_a_folder_with_files(tmp_path, monkeypatch):
    """--geno-store-dir naming the output folder, or a folder holding the user's files: the
    run removes only the subfolder it made."""
    vcf, pheno = _vcf(tmp_path / "v.vcf")
    keep = tmp_path / "mine"
    keep.mkdir()
    (keep / "notes.txt").write_text("keep me")
    for gsd in (tmp_path / "o1", keep):
        o = tmp_path / "o1"
        r = _run(["--vcf", vcf, "--pheno", pheno, "--trait", "T", "--output", o, "--no-plots",
                  "--no-report", "--geno-store", "disk", "--geno-store-dir", gsd], monkeypatch)
        assert r == ("ok", None)
        assert list(o.glob("*.zip")) and not list(gsd.glob(STORE_GLOB))
    assert (keep / "notes.txt").read_text() == "keep me"


def test_a_stop_during_the_trait_run_removes_the_table(tmp_path, monkeypatch):
    vcf, pheno = _vcf(tmp_path / "v.vcf")
    r = _both(tmp_path, monkeypatch, vcf, pheno, "T", "--maf", "0.6")    # every SNP fails QC
    assert r["memory"][0] != "ok" and r["disk"] == r["memory"]


def test_multi_trait_run_on_disk(tmp_path, monkeypatch):
    vcf, pheno = _vcf(tmp_path / "v.vcf")
    out = {}
    for mode in ("memory", "disk"):
        o = tmp_path / f"m_{mode}"
        assert _run(["--vcf", vcf, "--pheno", pheno, "--trait", "T", "U", "--output", o, "--no-plots",
                     "--no-report", "--geno-store", mode], monkeypatch) == ("ok", None)
        assert not list(o.glob(STORE_GLOB))
        out[mode] = {k: v for k, v in _zip_members(o).items() if not k[0].endswith(".zip")}
    assert out["memory"].keys() == out["disk"].keys()


def test_unwritable_store_folder_stops_before_reading(tmp_path, monkeypatch):
    vcf, pheno = _vcf(tmp_path / "v.vcf")
    blocker = tmp_path / "a_file"
    blocker.write_text("not a folder")
    kind, msg = _run(["--vcf", vcf, "--pheno", pheno, "--trait", "T", "--output", tmp_path / "o",
                      "--geno-store", "disk", "--geno-store-dir", blocker / "sub"], monkeypatch)
    assert kind == "SystemExit" and "--geno-store-dir" in msg


@pytest.mark.parametrize("piece", ["1", "64"])
def test_stage_digests_equal_in_both_modes(tmp_path, monkeypatch, piece):
    """TRACE_TAPS=1: every intermediate object the pipeline records (kept samples and SNPs,
    the tables, kinship, PCs, LOCO kernels, M_eff, the standardised table) has the same digest
    with the table on disk, at small piece sizes, as in memory."""
    import json

    vcf, pheno = _vcf(tmp_path / "v.vcf")
    monkeypatch.setenv("TRACE_TAPS", "1")
    rows = {}
    for mode in ("memory", "disk"):
        if mode == "disk":
            monkeypatch.setenv("TRACE_STORE_PIECE", piece)
        o = tmp_path / mode / "results"
        assert _run(["--vcf", vcf, "--pheno", pheno, "--trait", "T", "--output", o, "--no-plots",
                     "--no-report", "--model", "mlm", "mlmm", "farmcpu", "--geno-store", mode],
                    monkeypatch) == ("ok", None)
        rows[mode] = json.loads((tmp_path / mode / "taps_T.json").read_text())
    assert len(rows["memory"]) > 30
    assert rows["disk"] == rows["memory"]


def _metadata(folder):
    import json
    import zipfile

    z = sorted(Path(folder).glob("*.zip"))[0]
    with zipfile.ZipFile(z) as f:
        return json.loads(f.read("run_metadata.json"))


def _compare_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "compare_result_zips", Path(__file__).resolve().parents[1] / "scripts" / "compare_result_zips.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_auto_chooses_disk_when_free_memory_is_low(tmp_path, monkeypatch, caplog):
    """The free memory reported to the plan is lowered (not real memory exhausted): auto keeps
    the table on disk, says so in the log and in run_metadata.json, and the results equal a run
    with the table in memory."""
    import logging

    from gwas import storeplan

    vcf, pheno = _vcf(tmp_path / "v.vcf")
    caplog.set_level(logging.INFO, logger="cli")
    monkeypatch.setattr(storeplan, "available_memory_bytes", lambda: 64 * 2 ** 20)
    assert _run(["--vcf", vcf, "--pheno", pheno, "--trait", "T", "--output", tmp_path / "auto",
                 "--no-plots"], monkeypatch) == ("ok", None)
    meta = _metadata(tmp_path / "auto")
    assert meta["Genotype table"] == "disk" and meta["Genotype table choice"] == "auto"
    assert "so TRACE keeps the genotype table on disk. Results are the same either way." in meta["Genotype table reason"]
    assert any("keeps the genotype table on disk" in r.getMessage() for r in caplog.records)
    assert not list((tmp_path / "auto").glob(STORE_GLOB))

    monkeypatch.undo()
    assert _run(["--vcf", vcf, "--pheno", pheno, "--trait", "T", "--output", tmp_path / "mem",
                 "--no-plots", "--geno-store", "memory"], monkeypatch) == ("ok", None)
    assert _metadata(tmp_path / "mem")["Genotype table"] == "memory"
    za = sorted((tmp_path / "auto").glob("*.zip"))[0]
    zm = sorted((tmp_path / "mem").glob("*.zip"))[0]
    import io
    buf = io.StringIO()
    assert _compare_script().compare(za, zm, out=buf) == 0, buf.getvalue()


def test_auto_keeps_a_small_run_in_memory(tmp_path, monkeypatch):
    vcf, pheno = _vcf(tmp_path / "v.vcf")
    assert _run(["--vcf", vcf, "--pheno", pheno, "--trait", "T", "--output", tmp_path / "o",
                 "--no-plots", "--no-report"], monkeypatch) == ("ok", None)
    meta = _metadata(tmp_path / "o")
    assert meta["Genotype table"] == "memory" and "in memory" in meta["Genotype table reason"]
    report_keys = set(meta) - set(__import__("gwas.storeplan", fromlist=["MODE_KEYS"]).MODE_KEYS)
    assert "Genotype table" not in report_keys


def test_forced_memory_stops_before_reading_when_it_will_not_fit(tmp_path, monkeypatch):
    from gwas import storeplan

    vcf, pheno = _vcf(tmp_path / "v.vcf")
    monkeypatch.setattr(storeplan, "available_memory_bytes", lambda: 2 ** 20)
    kind, msg = _run(["--vcf", vcf, "--pheno", pheno, "--trait", "T", "--output", tmp_path / "o",
                      "--geno-store", "memory"], monkeypatch)
    assert kind == "SystemExit" and "would need about" in msg and "--geno-store disk" in msg
    assert not list((tmp_path / "o").glob("*.zip"))


def test_report_is_the_same_in_both_modes(tmp_path, monkeypatch):
    """The mode keys go into run_metadata.json only, not into the report's provenance table."""
    import zipfile

    vcf, pheno = _vcf(tmp_path / "v.vcf")
    html = {}
    for mode in ("memory", "disk"):
        o = tmp_path / mode
        assert _run(["--vcf", vcf, "--pheno", pheno, "--trait", "T", "--output", o, "--no-plots",
                     "--geno-store", mode], monkeypatch) == ("ok", None)
        with zipfile.ZipFile(sorted(o.glob("*.zip"))[0]) as f:
            html[mode] = [l for l in f.read("report.html").decode().splitlines() if "Generated:" not in l]
    assert html["memory"] == html["disk"]
    assert not any("Genotype table" in l for l in html["memory"])
