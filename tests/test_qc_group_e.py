"""Group E QC-overhaul tests: P14 (--ld-seed-mode {suggestive,significant}) and
P15 (the --ld-top-n FLOOR forms blocks even without threshold-passing seeds)."""
import numpy as np
import pandas as pd
import pytest

from gwas.ld import find_ld_clusters_genomewide


# ── P14: the CLI seed-mode flag ─────────────────────────────────────────────
def test_p14_ld_seed_mode_default_suggestive():
    from cli import _build_parser
    args = _build_parser().parse_args(
        ["--vcf", "x", "--pheno", "p", "--trait", "Y", "--output", "o"])
    assert args.ld_seed_mode == "suggestive"


def test_p14_ld_seed_mode_accepts_significant_rejects_bogus():
    from cli import _build_parser
    base = ["--vcf", "x", "--pheno", "p", "--trait", "Y", "--output", "o"]
    args = _build_parser().parse_args(base + ["--ld-seed-mode", "significant"])
    assert args.ld_seed_mode == "significant"
    with pytest.raises(SystemExit):
        _build_parser().parse_args(base + ["--ld-seed-mode", "bogus"])


# ── P15: the top-N floor forms a block even when nothing passes sig_thresh ───
def _block_inputs():
    """3 perfect-LD chr-1 SNPs at p=1e-6 (below 1e-5, above a strict 1e-9)."""
    rng = np.random.default_rng(0)
    n = 60
    b1 = rng.integers(0, 3, n).astype(float)
    noise = rng.integers(0, 3, (n, 3)).astype(float)
    G = np.column_stack([b1, b1, b1, noise])
    sid = np.array([f"m{i}" for i in range(6)])
    chroms = np.array(["1", "1", "1", "1", "1", "1"])
    positions = np.array([1000, 1100, 1200, 500000, 900000, 1500000])
    gwas_df = pd.DataFrame({"SNP": sid, "Chr": chroms, "Pos": positions,
                            "PValue": [1e-6, 1e-6, 1e-6, 0.5, 0.6, 0.7]})
    return gwas_df, chroms, positions, G, sid


def test_p15_no_floor_no_block_when_nothing_passes_threshold():
    gwas_df, chroms, positions, G, sid = _block_inputs()
    blocks = find_ld_clusters_genomewide(
        gwas_df, chroms, positions, G, sid,
        min_snps=3, sig_thresh=1e-9, top_n=0)          # significant mode: no floor
    assert len(blocks) == 0                            # p=1e-6 never clears 1e-9


def test_p15_top_n_floor_forms_nonsignificant_block():
    gwas_df, chroms, positions, G, sid = _block_inputs()
    blocks = find_ld_clusters_genomewide(
        gwas_df, chroms, positions, G, sid,
        min_snps=3, sig_thresh=1e-9, top_n=10)         # suggestive floor
    assert len(blocks) == 1                            # the floor seeds the block
    assert str(blocks.iloc[0]["Chr"]) == "1"
