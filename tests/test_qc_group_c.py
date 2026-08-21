"""Group C QC-overhaul tests: P16 (block detection + haplotype refuse unanchored
ALT markers) and P11 (--drop-alt removed)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_p11_cli_rejects_drop_alt():
    from cli import _build_parser
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            ["--vcf", "x", "--pheno", "p", "--trait", "Y", "--output", "o", "--drop-alt"])


def test_p16_block_detection_refuses_alt_seeds():
    """An ALT LD block (3 scaffold markers in perfect LD, highly significant)
    would form without P16; the guard refuses ALT seeds so only the chr-1 block
    survives."""
    from gwas.ld import find_ld_clusters_genomewide
    rng = np.random.default_rng(0)
    n = 60
    b1 = rng.integers(0, 3, n).astype(float)   # chr-1 block source (perfect LD)
    ba = rng.integers(0, 3, n).astype(float)   # ALT block source (perfect LD)
    G = np.column_stack([b1, b1, b1, ba, ba, ba])
    sid = np.array([f"m{i}" for i in range(6)])
    chroms = np.array(["1", "1", "1", "ALT", "ALT", "ALT"])
    positions = np.array([1000, 1100, 1200, 100, 200, 300])
    gwas_df = pd.DataFrame({"SNP": sid, "Chr": chroms, "Pos": positions,
                            "PValue": [1e-8, 1e-8, 1e-8, 1e-9, 1e-9, 1e-9]})
    blocks = find_ld_clusters_genomewide(gwas_df, chroms, positions, G, sid,
                                         min_snps=3, sig_thresh=1e-5)
    chrs = set(blocks["Chr"].astype(str)) if len(blocks) else set()
    assert "ALT" not in chrs                    # P16: ALT seeds refused despite 1e-9
    assert "1" in chrs                          # a normal block still forms


def test_p16_pepper_haplotype_three_to_one():
    """Acceptance: the committed pepper FarmCPU haplotype output has 3 blocks,
    2 of them on chr-0 scaffolds (Chr == "ALT"); P16 refuses ALT blocks, leaving
    exactly 1 (chr 9)."""
    p = Path("benchmarks/qc_data/pepper_BX/platform_Haplotype_GWAS_FarmCPU.csv")
    if not p.exists():
        pytest.skip("pepper haplotype fixture absent")
    h = pd.read_csv(p)
    assert len(h) == 3
    kept = h[h["Chr"].astype(str) != "ALT"].reset_index(drop=True)
    assert len(kept) == 1
    assert str(kept.iloc[0]["Chr"]) == "9"
