"""T-71 — Golden block-table tests (Batch A regression harness).

Tier A pins the *algorithm* on committed synthetic fixtures: any edit to
connectivity, contiguity refinement, gap splitting, IoU merging or the min-SNP
gate changes a golden and fails here. Runs in CI (self-contained, no external
data). Tier B pins the *published* Table S8 block table behind the ``manuscript``
marker and is skipped unless the Varitome inputs are present.

The comparison is via ``canon_block_table`` (see ``tests/golden/_canon.py``), the
single definition of "the same block table" — order- and token-normalised so an
incidental ordering difference cannot cause a false failure.

Spec: ``docs/revision/specs/integration_and_testing.md`` §1.2.
"""
import os

import pandas as pd
import pytest

from gwas.ld import filter_contained_blocks, find_ld_clusters_genomewide
from tests.golden._canon import GOLDEN_DIR, canon_block_table, load_case
from tests.golden._cases import TIER_A_CASES

BLOCK_SCHEMA = ["Chr", "Start (bp)", "End (bp)", "Lead SNP", "SNP_IDs"]


def _detect(inp, params, *, min_snps=None):
    """Run the detector on a case's input with the case's pinned params."""
    p = dict(params)
    if min_snps is not None:
        p["min_snps"] = min_snps
    gwas_df = pd.DataFrame({
        "SNP": inp["sid"].astype(str),
        "Chr": inp["chroms"].astype(str),
        "Pos": inp["positions"].astype(int),
        "PValue": inp["pvalue"].astype(float),
    })
    return find_ld_clusters_genomewide(
        gwas_df=gwas_df,
        chroms=inp["chroms"], positions=inp["positions"],
        geno_imputed=inp["geno"].astype(float), sid=inp["sid"],
        ld_threshold=p["ld_threshold"], flank_kb=p["flank_kb"], min_snps=p["min_snps"],
        top_n=p["top_n"], sig_thresh=p["sig_thresh"], adj_r2_min=p["adj_r2_min"],
        min_pair_n=p["min_pair_n"], merge_iou=p["merge_iou"], gap_factor=p["gap_factor"],
    )


# --------------------------------------------------------------------------
# Tier A — algorithmic pin (CI)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("case", TIER_A_CASES)
def test_golden_block_table_is_unchanged(case):
    inp, expected, meta = load_case(case)
    got = _detect(inp, meta["params"])
    assert canon_block_table(got) == canon_block_table(expected), (
        f"{case}: detected block table changed vs golden — "
        f"tests/golden/{case}/expected_blocks.csv"
    )


@pytest.mark.parametrize("case", TIER_A_CASES)
def test_golden_filtered_block_table_is_unchanged(case):
    """The mega-block containment filter is pinned as a SECOND column set in the
    same case, so a regression localises to detection vs containment filtering."""
    inp, _expected, meta = load_case(case)
    got = _detect(inp, meta["params"])
    filtered, _n_removed = filter_contained_blocks(got.copy(), min_contained=2)
    expected_filtered = pd.read_csv(
        GOLDEN_DIR / case / "expected_blocks_filtered.csv", dtype=str, keep_default_na=False
    )
    assert canon_block_table(filtered) == canon_block_table(expected_filtered), (
        f"{case}: filter_contained_blocks output changed vs golden — "
        f"tests/golden/{case}/expected_blocks_filtered.csv"
    )


def test_below_min_snps_is_empty_with_full_schema():
    """The below-min-SNPs case yields an EMPTY table carrying the full five-column
    schema — this is the shape that catches a relaxed min_snps."""
    inp, _expected, meta = load_case("blocks_below_min_snps")
    got = _detect(inp, meta["params"])
    assert len(got) == 0
    assert list(got.columns) == BLOCK_SCHEMA


def test_min_snps_gate_has_discriminating_power():
    """Negative: the same fixture at min_snps-1 must DIFFER from its golden. A
    golden insensitive to the gate it protects is a broken golden."""
    inp, expected, meta = load_case("blocks_below_min_snps")
    golden = canon_block_table(expected)
    relaxed = _detect(inp, meta["params"], min_snps=meta["params"]["min_snps"] - 1)
    assert canon_block_table(relaxed) != golden, (
        "blocks_below_min_snps golden is insensitive to min_snps — the fixture has "
        "no discriminating power and cannot protect the gate"
    )
    assert len(relaxed) >= 1, "relaxing min_snps should surface the size-2 component"


# --------------------------------------------------------------------------
# Tier B — manuscript pin (opt-in; skipped without the Varitome inputs)
# --------------------------------------------------------------------------
@pytest.mark.manuscript
def test_varitome_locule_blocks_match_table_s8():
    """Tier-B pin: re-running detection at the reproducing params (meta.json,
    from T-61: flank_kb=144, ld_decay_kb=72.17) on the Varitome QC inputs emits
    Table S8's six blocks. The genotypes are gitignored, so this skips cleanly
    unless the QC data is present on the machine."""
    import importlib.util
    import json

    repo = GOLDEN_DIR.parents[1]
    case_dir = GOLDEN_DIR / "varitome_locule"
    meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
    qc_dir = os.environ.get("TRACE_VARITOME_DIR",
                            str(repo / "benchmarks" / "qc_data" / "tomato_locule_number"))
    from pathlib import Path
    if not (Path(qc_dir) / "QC_genotype_matrix.csv").exists():
        pytest.skip("Varitome QC inputs absent (gitignored; set TRACE_VARITOME_DIR). "
                    "Read back from an actual run, never from the PDF.")

    # Reuse the T-61 loader/detector so the reproducing path lives in one place.
    spec = importlib.util.spec_from_file_location(
        "_recon_t61", repo / "benchmarks" / "reconstruct_table_s8_params.py")
    recon = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recon)

    geno, sid, chroms_str, positions = recon._load()
    gwas_df = pd.read_csv(recon.QC_DIR / "platform_GWAS_locule_number.csv")
    p = meta["params"]
    got = recon._detect(gwas_df, chroms_str, positions, geno, sid,
                        flank_kb=p["flank_kb"], ld_decay_kb=p["ld_decay_kb"])
    expected = pd.read_csv(case_dir / "expected_blocks.csv", dtype=str, keep_default_na=False)
    assert canon_block_table(got) == canon_block_table(expected), (
        "Varitome locule detection no longer reproduces Table S8's six blocks — "
        "tests/golden/varitome_locule/expected_blocks.csv"
    )
