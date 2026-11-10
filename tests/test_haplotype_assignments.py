"""Sample-to-haplotype assignments (reviewer point 2-INT.10).

gwas.haplotype.haplotype_assignments builds one row per sample of the panel per tested
block from the haplotype test's own per-sample tables: the sample's label, its allele
sequence and the markers in that sequence, whether the test took it, and why not.

Synthetic blocks (every exclusion reason, a duplicate marker the test thins) and the
bundled example through the shared pipeline (the CLI and One-Click ZIP member
tables/Haplotype_assignments_<trait>.csv). The F-test's inputs are recorded
(block_test_lm_with_pcs) and compared with the in-test rows: same samples, same labels,
same phenotype values, in the same order. Opt-in: the tomato lead block, when
TRACE_TOMATO_VCF and TRACE_TOMATO_PHENO name the Varitome files.
"""
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gwas.haplotype as gh
import gwas.pipeline as pl
from gwas.haplotype import HAP_ASSIGNMENT_COLUMNS, haplotype_assignments, run_haplotype_block_gwas
from gwas.io import load_vcf_cached

_ROOT = Path(__file__).resolve().parent.parent
_EX = _ROOT / "tests" / "data" / "example_v1"   # the first example

# block 1 (markers s0-s4; s4 duplicates s0): haplotypes A..F over s0-s3
_PATTERNS = {"A": [0, 0, 0, 0], "B": [2, 2, 0, 0], "C": [0, 2, 2, 0],
             "D": [2, 0, 2, 2], "E": [0, 0, 2, 2], "F": [2, 2, 2, 2]}
_COUNTS = {"A": 25, "B": 15, "C": 8, "D": 4, "E": 2, "F": 2}   # phenotyped, genotyped
_EFFECT = {"A": 0.0, "B": 1.0, "C": -1.0, "D": 0.5, "E": 2.0, "F": -2.0}


def _synthetic():
    rows = [h for h, n in _COUNTS.items() for _ in range(n)]
    rows += ["A", "A"]          # no phenotype
    rows += ["B", "B"]          # 2 of 5 calls missing (> 30%)
    n = len(rows)
    rng = np.random.default_rng(11)
    order = rng.permutation(n)
    hap = np.array(rows)[order]
    special = {int(np.where(order == n - 4)[0][0]): "nopheno",
               int(np.where(order == n - 3)[0][0]): "nopheno",
               int(np.where(order == n - 2)[0][0]): "nogeno",
               int(np.where(order == n - 1)[0][0]): "nogeno"}
    ids = np.array([f"S{i:02d}" for i in range(n)][::-1])     # genotype order is not sorted
    G = np.zeros((n, 10))
    for i, h in enumerate(hap):
        G[i, :4] = _PATTERNS[h]
        G[i, 4] = _PATTERNS[h][0]                 # duplicate of s0 (r2 = 1, thinned)
        G[i, 5:] = 2.0 if i % 2 else 0.0          # block 2: two groups of 30
    one_missing = int(np.where((hap == "A") & ~np.isin(np.arange(n), list(special)))[0][0])
    G[one_missing, 3] = np.nan                    # 1 of 5 missing: kept, imputed to 0
    y = np.array([_EFFECT[h] for h in hap]) + rng.normal(0, 0.3, n) + 0.4 * (np.arange(n) % 2)
    for i, kind in special.items():
        if kind == "nopheno":
            y[i] = np.nan
        else:
            G[i, [1, 2]] = np.nan
    sid = np.array([f"s{j}" for j in range(10)])
    blocks = pd.DataFrame({
        "Chr": ["1", "1"], "Start (bp)": [1000, 6000], "End (bp)": [5000, 10000],
        "lead_snp": ["s1", "s6"], "SNP_IDs": [",".join(sid[:5]), ",".join(sid[5:])],
    })
    return {
        "blocks": blocks, "chroms": np.array(["1"] * 10), "positions": np.arange(1, 11) * 1000.0,
        "G": G, "sid": sid, "geno_df": pd.DataFrame(G, index=ids),
        "pheno": pd.DataFrame({"t": y}, index=ids), "hap": dict(zip(ids, hap)),
        "special": {ids[i]: kind for i, kind in special.items()},
    }


def _record_f_test(monkeypatch):
    calls = []
    real = gh.block_test_lm_with_pcs

    def _rec(y, groups, pcs=None):
        calls.append({"y": np.asarray(y, float).copy(), "groups": np.asarray(groups).astype(str)})
        return real(y=y, groups=groups, pcs=pcs)

    monkeypatch.setattr(gh, "block_test_lm_with_pcs", _rec)
    return calls


@pytest.fixture(scope="module")
def synth():
    d = _synthetic()
    with pytest.MonkeyPatch.context() as mp:
        calls = _record_f_test(mp)
        res, tables = run_haplotype_block_gwas(
            haplo_df=d["blocks"], chroms=d["chroms"], positions=d["positions"],
            geno_imputed=d["G"], sid=d["sid"], geno_df=d["geno_df"], pheno_df=d["pheno"],
            trait_col="t", pcs=None, min_hap_count=4, min_group_size=5, n_perm=50)
    d.update(res=res, tables=tables, calls=calls,
             table=haplotype_assignments(res, tables, d["geno_df"], d["pheno"], "t"))
    return d


def test_one_row_per_panel_sample_per_tested_block(synth):
    t, res = synth["table"], synth["res"]
    assert list(t.columns) == HAP_ASSIGNMENT_COLUMNS
    assert len(res) == 2
    panel = list(synth["geno_df"].index)
    for (_, r), (key, rows) in zip(res.iterrows(), t.groupby(["Block_start"], sort=False)):
        assert list(rows["Sample"]) == panel
        assert set(rows["Chr"]) == {"1"} and set(rows["Lead_SNP"]) == {r["lead_snp"]}
        assert set(rows["Block_end"]) == {int(r["End"])} and set(rows["Trait"]) == {"t"}
        assert int(rows["in_haplotype_test"].sum()) == int(r["n_samples_tested"])


def test_every_exclusion_reason_and_the_labels(synth):
    b1 = synth["table"][synth["table"]["Block_start"] == 1000].set_index("Sample")
    reasons = b1["exclusion_reason"].fillna("")
    assert reasons.value_counts().to_dict() == {
        "": 48, "rare haplotype (Other)": 4, "group below minimum size": 4,
        "missing phenotype": 2, "missing genotype": 2}
    for sample, kind in synth["special"].items():
        assert reasons[sample] == ("missing phenotype" if kind == "nopheno" else "missing genotype")
        # left out before labelling: no haplotype
        assert b1.loc[sample, "Haplotype"] == "" and b1.loc[sample, "Allele_sequence"] == ""
    labels = {h: set(b1.loc[[s for s, x in synth["hap"].items()
                             if x == h and s not in synth["special"]], "Haplotype"])
              for h in _PATTERNS}
    assert labels == {"A": {"H1"}, "B": {"H2"}, "C": {"H3"}, "D": {"H4"},
                      "E": {"Other"}, "F": {"Other"}}
    assert set(b1.loc[b1["Haplotype"] == "H4", "exclusion_reason"]) == {"group below minimum size"}
    assert set(b1.loc[b1["Haplotype"] == "Other", "exclusion_reason"]) == {"rare haplotype (Other)"}
    assert b1["in_haplotype_test"].tolist() == (reasons == "").tolist()


def test_snp_order_is_the_tested_markers(synth):
    t, res = synth["table"], synth["res"]
    for _, r in res.iterrows():
        order = t.loc[t["Block_start"] == int(r["Start"]), "SNP_order"].unique()
        assert len(order) == 1
        assert len(order[0].split("|")) == int(r["n_snps"])
    b1 = t[t["Block_start"] == 1000]
    assert b1["SNP_order"].iloc[0] == "s0|s1|s2|s3"          # s4 (duplicate of s0) thinned
    seqs = b1.loc[b1["Allele_sequence"] != "", "Allele_sequence"]
    assert seqs.str.count(r"\|").eq(3).all()


def test_in_test_rows_are_the_f_tests_inputs(synth):
    """A and C: the rows with in_haplotype_test are the samples, labels and phenotype
    values the F-test received, in its order; the counts are the test's own."""
    t, res, calls = synth["table"], synth["res"], synth["calls"]
    assert len(calls) == len(res)
    for (_, r), call in zip(res.iterrows(), calls):
        rows = t[(t["Block_start"] == int(r["Start"])) & t["in_haplotype_test"]]
        assert rows["Haplotype"].tolist() == call["groups"].tolist()
        np.testing.assert_array_equal(rows["Phenotype_used"].to_numpy(float), call["y"])
        n_by_label = {h: s["n"] for h, s in json.loads(r["hap_stats_json"]).items()}
        assert rows["Haplotype"].value_counts().to_dict() == n_by_label


def test_phenotype_raw_appears_only_when_it_differs(synth):
    args = (synth["res"], synth["tables"], synth["geno_df"], synth["pheno"], "t")
    assert "Phenotype_raw" not in haplotype_assignments(*args).columns
    same = haplotype_assignments(*args, pheno_raw=synth["pheno"]["t"])
    assert "Phenotype_raw" not in same.columns
    raw = synth["pheno"]["t"] * 3.0 + 1.0
    diff = haplotype_assignments(*args, pheno_raw=raw)
    assert list(diff.columns) == HAP_ASSIGNMENT_COLUMNS + ["Phenotype_raw"]
    for _, rows in diff.groupby("Block_start"):
        np.testing.assert_array_equal(rows["Phenotype_raw"].to_numpy(float),
                                      raw.reindex(rows["Sample"]).to_numpy(float))


def test_a_blocks_file_is_its_rows_of_the_all_blocks_file(synth):
    args = (synth["res"], synth["tables"], synth["geno_df"], synth["pheno"], "t")
    full = synth["table"]
    for key, start in (("1:1000-5000", 1000), ("1:6000-10000", 6000)):
        one = haplotype_assignments(*args, blocks={key})
        assert one.to_csv(index=False) == full[full["Block_start"] == start].to_csv(index=False)
    assert haplotype_assignments(*args, blocks={"1:1-2"}).empty


def test_inputs_are_not_modified(synth):
    tables = {k: v.copy() for k, v in synth["tables"].items()}
    pheno = synth["pheno"].copy()
    haplotype_assignments(synth["res"], synth["tables"], synth["geno_df"], synth["pheno"], "t",
                          pheno_raw=synth["pheno"]["t"] + 1)
    for k in tables:
        pd.testing.assert_frame_equal(synth["tables"][k], tables[k])
    pd.testing.assert_frame_equal(synth["pheno"], pheno)


def test_raw_values_with_repeated_or_unmatched_ids_add_no_column(synth):
    """A repeated sample in the raw values keeps its first value (as the ID harmonisation
    does), and raw values that match no panel sample do not add Phenotype_raw."""
    s = synth
    plain = haplotype_assignments(s["res"], s["tables"], s["geno_df"], s["pheno"], "t")
    raw = s["pheno"]["t"]
    dup = pd.concat([raw, raw.iloc[[0]] + 99.0])
    assert haplotype_assignments(s["res"], s["tables"], s["geno_df"], s["pheno"], "t",
                                 pheno_raw=dup).equals(plain)
    padded = raw.copy()
    padded.index = ["0" + str(i) for i in padded.index]
    assert haplotype_assignments(s["res"], s["tables"], s["geno_df"], s["pheno"], "t",
                                 pheno_raw=padded).equals(plain)


def test_outputs_doc_documents_every_column():
    doc = (_ROOT / "docs" / "outputs.md").read_text(encoding="utf-8")
    assert "## `Haplotype_assignments_<trait>.csv`" in doc
    section = doc.split("## `Haplotype_assignments_<trait>.csv`", 1)[1].split("\n## ", 1)[0]
    for col in HAP_ASSIGNMENT_COLUMNS + ["Phenotype_raw"]:
        assert f"| `{col}` |" in section, col
    for reason in gh.HAP_EXCLUSION_REASONS:
        assert reason in section, reason
    assert "haplotype_assignments_block_<chr>_<start>_<end>_<trait>.csv" in section


def test_no_tested_block_gives_the_header_only():
    empty = haplotype_assignments(pd.DataFrame(), {}, pd.DataFrame(index=["a"]),
                                  pd.DataFrame({"t": [1.0]}, index=["a"]), "t")
    assert empty.empty and list(empty.columns) == HAP_ASSIGNMENT_COLUMNS


# ── the shared pipeline: the CLI and One-Click ZIP member ─────────────────────

def _pipeline_run(monkeypatch, vcf, pheno_csv, trait, **opts):
    hap_calls, f_calls = [], _record_f_test(monkeypatch)
    real = gh.run_haplotype_block_gwas

    def _rec(*a, **k):
        out = real(*a, **k)
        hap_calls.append(out)
        return out

    monkeypatch.setattr(gh, "run_haplotype_block_gwas", _rec)
    parsed = pl.ParsedGenotypes.from_callset(load_vcf_cached(Path(vcf).read_bytes()))
    pheno = pd.read_csv(pheno_csv, index_col=0, encoding="utf-8-sig")
    res = pl.run_trait_pipeline(parsed, pheno, trait, pl.cli_default_options(
        vcf=str(vcf), pheno=str(pheno_csv), **opts))
    with zipfile.ZipFile(io.BytesIO(res.zip_bytes)) as zf:
        members = {n: zf.read(n) for n in zf.namelist()}
    return res, members, hap_calls, f_calls


@pytest.mark.parametrize("trait", ["Trait2", "Trait1"])
def test_pipeline_writes_the_assignments_of_mlms_tested_blocks(trait, monkeypatch):
    res, members, hap_calls, f_calls = _pipeline_run(
        monkeypatch, _EX / "example.vcf.gz", _EX / "example_pheno.csv", trait,
        ld_r2=0.1, ld_flank_kb=200, hap_perms=100, no_plots=True)
    name = f"tables/Haplotype_assignments_{trait}.csv"
    assert name in members
    table = pd.read_csv(io.BytesIO(members[name]), keep_default_na=False, na_values=[""])
    assert members[name] == res.tables[f"Haplotype_assignments_{trait}.csv"].to_csv(
        index=False).encode("utf-8")
    assert list(table.columns) == HAP_ASSIGNMENT_COLUMNS          # raw values: no Phenotype_raw
    hap_res, _tables = hap_calls[0]                                # MLM, the primary model
    assert len(hap_res) >= 5
    assert len(f_calls) == len(hap_res)                            # MLM is the only model
    triage = pd.read_csv(io.BytesIO(members["tables/LD_triage_MLM.csv"]))
    for (_, r), call in zip(hap_res.iterrows(), f_calls):
        rows = table[(table["Chr"].astype(str) == str(r["Chr"])) & (table["Block_start"] == int(r["Start"]))]
        tested = rows[rows["in_haplotype_test"]]
        assert len(rows) == 50
        assert len(tested) == int(r["n_samples_tested"])
        tri = triage[(triage["Start (bp)"] == int(r["Start"])) & (triage["End (bp)"] == int(r["End"]))]
        assert len(tested) == int(tri["n_samples_tested"].iloc[0])
        assert tested["Haplotype"].astype(str).tolist() == call["groups"].tolist()
        np.testing.assert_array_equal(tested["Phenotype_used"].to_numpy(float), call["y"])
    blocks = {(str(r["Chr"]), int(r["Start"])) for _, r in hap_res.iterrows()}
    assert set(zip(table["Chr"].astype(str), table["Block_start"])) == blocks


def test_every_other_model_gets_a_file_of_its_tested_blocks(monkeypatch):
    """With three models, MLMM and FarmCPU each get Haplotype_assignments_<trait>_<model>.csv:
    exactly that model's tested blocks (its haplotype results), every panel sample per
    block. The primary model's file is the MLM one, as before."""
    res, members, _hap, _f = _pipeline_run(
        monkeypatch, _EX / "example.vcf.gz", _EX / "example_pheno.csv", "Trait1",
        model=["mlm", "mlmm", "farmcpu"], ld_r2=0.1, ld_flank_kb=200, hap_perms=100,
        no_plots=True)
    assert "tables/Haplotype_assignments_Trait1.csv" in members
    assert "tables/Haplotype_assignments_Trait1_MLM.csv" not in members
    covered = 0
    for model in ("MLMM", "FarmCPU"):
        ann = pd.read_csv(io.BytesIO(members[f"tables/LD_blocks_annotated_{model}.csv"]))
        tested = ann[ann["Hap_PValue"].notna()]
        name = f"tables/Haplotype_assignments_Trait1_{model}.csv"
        if tested.empty:
            assert name not in members
            continue
        table = pd.read_csv(io.BytesIO(members[name]), keep_default_na=False, na_values=[""])
        assert list(table.columns) == HAP_ASSIGNMENT_COLUMNS
        assert members[name] == res.tables[name.split("/", 1)[1]].to_csv(index=False).encode("utf-8")
        want = {(str(r["Chr"]), int(r["Start (bp)"]), int(r["End (bp)"])) for _, r in tested.iterrows()}
        got = set(zip(table["Chr"].astype(str), table["Block_start"], table["Block_end"]))
        assert got == want, model
        assert (table.groupby(["Block_start"]).size() == 50).all()
        covered += 1
    assert covered >= 1        # the example's MLMM / FarmCPU runs test at least one block


_TOMATO = (os.environ.get("TRACE_TOMATO_VCF"), os.environ.get("TRACE_TOMATO_PHENO"))


@pytest.mark.skipif(not all(_TOMATO), reason="set TRACE_TOMATO_VCF and TRACE_TOMATO_PHENO "
                                             "to the Varitome files to run")
def test_tomato_lead_block(monkeypatch):
    """The tomato lead block (the tested block with the smallest lead-SNP p-value),
    CLI defaults, locule_number."""
    res, members, hap_calls, f_calls = _pipeline_run(
        monkeypatch, _TOMATO[0], _TOMATO[1], "locule_number", no_plots=True)
    table = pd.read_csv(io.BytesIO(members["tables/Haplotype_assignments_locule_number.csv"]),
                        keep_default_na=False, na_values=[""])
    hap_res, _tables = hap_calls[0]
    blocks = pd.read_csv(io.BytesIO(members["tables/LD_blocks_annotated_MLM.csv"]))
    assert "lead_snp_pvalue" in blocks.columns
    keys = list(zip(hap_res["Chr"].astype(str), hap_res["Start"].astype(int)))
    tested_blocks = blocks[[(str(c), int(s)) in keys
                            for c, s in zip(blocks["Chr"], blocks["Start (bp)"])]]
    lead = tested_blocks.loc[tested_blocks["lead_snp_pvalue"].idxmin()]
    i = keys.index((str(lead["Chr"]), int(lead["Start (bp)"])))
    r, call = hap_res.iloc[i], f_calls[i]
    tested = table[(table["Chr"].astype(str) == str(r["Chr"]))
                   & (table["Block_start"] == int(r["Start"])) & table["in_haplotype_test"]]
    assert len(tested) == int(r["n_samples_tested"])
    assert tested["Haplotype"].astype(str).tolist() == call["groups"].tolist()
    np.testing.assert_array_equal(tested["Phenotype_used"].to_numpy(float), call["y"])
    n_by_label = {h: s["n"] for h, s in json.loads(r["hap_stats_json"]).items()}
    assert tested["Haplotype"].value_counts().to_dict() == n_by_label
