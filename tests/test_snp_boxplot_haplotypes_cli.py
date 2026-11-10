"""The report's per-SNP box plots (CLI and One-Click ZIP): dots coloured by haplotype.

The shared pipeline colours each per-SNP box plot by the haplotype labels of the LD block
that contains the SNP (by position: the significant-SNP table's Block_ID), taken from the
Haplotype_assignments table it writes. A SNP outside every block, or in a block with no
haplotype groups, keeps the single colour and gets a note. The y axis names the trait,
and the --norm when there is one. Recorded calls of gwas.snpplots.render_snp_boxplot on
the bundled example; opt-in: the tomato lead SNP (TRACE_TOMATO_VCF, TRACE_TOMATO_PHENO).
"""
import io
import os
import re
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

import gwas.haplotype as gh  # noqa: E402
import gwas.pipeline as pl  # noqa: E402
import gwas.snpplots as sp  # noqa: E402
from gwas.io import load_vcf_cached  # noqa: E402
from utils.pub_theme import PALETTE_CYCLE, hap_sort_key  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
_EX = _ROOT / "tests" / "data" / "example_v1"   # the first example
_TOMATO = (os.environ.get("TRACE_TOMATO_VCF"), os.environ.get("TRACE_TOMATO_PHENO"))
_CC8 = {"ld_r2": 0.1, "ld_flank_kb": 200, "hap_perms": 100}
_NEW = ("hap_labels", "hap_color_map", "hap_note", "y_label")


def _dots(fig):
    out = []
    for line in fig.axes[0].lines:
        if line.get_marker() == "o":
            c = to_hex(line.get_color())
            out += [(float(x), float(y), c) for x, y in zip(line.get_xdata(), line.get_ydata())]
    return out


def _run(monkeypatch, vcf, pheno_csv, trait, **opts):
    """Run the pipeline; record each box plot and the samples (rows of y)."""
    plots, samples = [], []
    real_plot, real_hap = sp.render_snp_boxplot, gh.run_haplotype_block_gwas

    def _plot(snp_id, geno_col, y, **kw):
        fig = real_plot(snp_id, geno_col, y, **kw)
        plain = real_plot(snp_id, geno_col, y, **{k: v for k, v in kw.items() if k not in _NEW})
        leg = fig.axes[0].get_legend()
        plots.append({
            "snp": str(snp_id), "g": np.asarray(geno_col, float), "y": np.asarray(y, float),
            "kw": dict(kw), "dots": _dots(fig), "plain": _dots(plain),
            "legend": [t.get_text() for t in leg.get_texts()] if leg is not None else None,
            "texts": [t.get_text() for t in fig.texts], "ylabel": fig.axes[0].get_ylabel(),
            "lim": (fig.axes[0].get_xlim(), fig.axes[0].get_ylim()),
            "plain_lim": (plain.axes[0].get_xlim(), plain.axes[0].get_ylim())})
        plt.close(plain)
        return fig

    def _hap(*a, **k):
        samples.append(np.asarray(k["geno_df"].index, dtype=str))
        return real_hap(*a, **k)

    monkeypatch.setattr(sp, "render_snp_boxplot", _plot)
    monkeypatch.setattr(gh, "run_haplotype_block_gwas", _hap)
    parsed = pl.ParsedGenotypes.from_callset(load_vcf_cached(Path(vcf).read_bytes()))
    pheno = pd.read_csv(pheno_csv, index_col=0, encoding="utf-8-sig")
    res = pl.run_trait_pipeline(parsed, pheno, trait, pl.cli_default_options(
        vcf=str(vcf), pheno=str(pheno_csv), no_plots=True, **opts))
    with zipfile.ZipFile(io.BytesIO(res.zip_bytes)) as zf:
        members = {n: zf.read(n) for n in zf.namelist()}
    assert plots, "no per-SNP box plot was drawn"
    return members, plots, (samples[0] if samples else None)


def _colour(label):
    """The colour a label must have: H<n> takes palette colour n (wrapping after 6), Other grey."""
    m = re.match(r"^H(\d+)$", label)
    return to_hex(PALETTE_CYCLE[(int(m.group(1)) - 1) % len(PALETTE_CYCLE)]) if m else "#bdbdbd"


def _table(members, name):
    return pd.read_csv(io.BytesIO(members[f"tables/{name}"]), keep_default_na=False,
                       na_values=[""])


def _check(members, plots, samples, trait):
    """Every plot against the tables: {snp: legend, or the note}."""
    sig = _table(members, "Significant_SNPs_MLM.csv").set_index("SNP")
    name = f"Haplotype_assignments_{trait}.csv"
    assign = (_table(members, name) if f"tables/{name}" in members else
              pd.DataFrame(columns=["Chr", "Block_start", "Block_end", "Sample", "Haplotype"]))
    seen = {}
    for p in plots:
        kw, block = p["kw"], sig.loc[p["snp"], "Block_ID"]
        # geometry: the same dots (data coordinates), limits and boxes as the plain plot
        assert sorted(d[:2] for d in p["dots"]) == sorted(d[:2] for d in p["plain"]), p["snp"]
        assert p["lim"] == p["plain_lim"]
        rows = pd.DataFrame()
        if isinstance(block, str) and block:
            chrom, span = block.split(":")
            start, end = (int(v) for v in span.split("-"))
            rows = assign[(assign["Chr"].astype(str) == chrom) & (assign["Block_start"] == start)
                          & (assign["Block_end"] == end)]
        if rows.empty or not rows["Haplotype"].astype(str).str.match(r"^H\d+$").any():
            note = sp.HAP_NOTE_NO_GROUPS if isinstance(block, str) and block else sp.HAP_NOTE_NO_BLOCK
            assert kw.get("hap_labels") is None and kw.get("hap_note") == note, p["snp"]
            assert {c for *_xy, c in p["dots"]} == {"#0072b2"} and p["legend"] is None
            assert p["texts"] == [note]
            seen[p["snp"]] = note
            continue
        # the labels: the table's, of this block, for each sample in the order of y
        by_sample = rows.set_index(rows["Sample"].astype(str))["Haplotype"].astype(str)
        want = by_sample.reindex(samples).fillna("Other").replace("", "Other").tolist()
        assert list(kw["hap_labels"]) == want, p["snp"]
        # each dot: its sample's label colour (dots matched to samples via the plain plot)
        valid = np.isfinite(p["g"]) & np.isfinite(p["y"])
        order = [i for cls in (0, 1, 2) for i in np.where(valid & (np.rint(p["g"]) == cls))[0]]
        who = {d[:2]: order[k] for k, d in enumerate(p["plain"])}
        drawn = sorted({want[who[d[:2]]] for d in p["dots"]}, key=hap_sort_key)
        for d in p["dots"]:
            assert d[2] == _colour(want[who[d[:2]]]), (p["snp"], d)
        assert p["legend"] == drawn and p["texts"] == []
        seen[p["snp"]] = drawn
    return seen


@pytest.mark.parametrize("trait", ["Trait1", "Trait2"])
def test_cc8_settings_colour_each_dot_by_its_block_haplotype(trait, monkeypatch):
    members, plots, samples = _run(monkeypatch, _EX / "example.vcf.gz",
                                   _EX / "example_pheno.csv", trait, **_CC8)
    seen = _check(members, plots, samples, trait)
    coloured = {s: v for s, v in seen.items() if isinstance(v, list)}
    assert coloured, "no box plot was coloured"
    if trait == "Trait1":
        assert seen["chr02_860000"] == ["H1", "H2", "H3", "H4", "Other"]
    for p in plots:
        assert p["ylabel"] == trait


def test_no_block_at_defaults_and_the_axis_names_the_norm(monkeypatch):
    members, plots, samples = _run(monkeypatch, _EX / "example.vcf.gz",
                                   _EX / "example_pheno.csv", "Trait1", norm="int")
    seen = _check(members, plots, samples, "Trait1")
    assert seen["chr02_860000"] == sp.HAP_NOTE_NO_BLOCK
    assert {p["ylabel"] for p in plots} == {"Trait1 (rank-INT)"}


def test_a_block_with_no_haplotype_groups_gets_the_other_note(monkeypatch):
    # every multi-locus genotype below 60 samples is "Other": no block is tested
    members, plots, samples = _run(monkeypatch, _EX / "example.vcf.gz",
                                   _EX / "example_pheno.csv", "Trait1", hap_min_count=60, **_CC8)
    assert "tables/Haplotype_assignments_Trait1.csv" not in members
    seen = _check(members, plots, samples, "Trait1")
    assert seen["chr02_860000"] == sp.HAP_NOTE_NO_GROUPS


@pytest.mark.skipif(not all(_TOMATO), reason="set TRACE_TOMATO_VCF and TRACE_TOMATO_PHENO "
                                             "to the Varitome files to run")
def test_tomato_lead_snp(monkeypatch):
    """CLI defaults, locule_number: the lead SNP's dots follow its block's haplotypes."""
    members, plots, samples = _run(monkeypatch, _TOMATO[0], _TOMATO[1], "locule_number")
    seen = _check(members, plots, samples, "locule_number")
    sig = _table(members, "Significant_SNPs_MLM.csv")
    lead = str(sig.loc[sig["PValue"].idxmin(), "SNP"])
    assert isinstance(seen[lead], list) and len(seen[lead]) >= 3
