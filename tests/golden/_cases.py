"""Tier-A golden case architectures (T-95).

Each case is a small, deterministic synthetic dataset engineered so that a single
branch of the LD-block detector (`gwas.ld.find_ld_clusters_genomewide` →
`find_ld_blocks_graph` → `contiguous_segments_by_adjacent`) is the one under
test. The *expected* output is not hand-derived: it is whatever the current
detector emits (that is the regression contract). This module owns only the
INPUT architecture; the expected CSVs are produced by re-running the detector in
`regenerate.py`.

Reproducibility: every case is built from its `generator_seed` via
`numpy.random.default_rng`, so `input.npz` can be rebuilt bit-for-bit. Genotypes
are dosage ints in {0,1,2}; a "block" is a set of SNPs sharing two latent
haplotype columns with a small independent per-allele flip rate, giving pairwise
r² comfortably above the 0.6 edge threshold but below 1 — the regime the block
detector actually operates in.

Budget (asserted by the harness): n = 120 samples so `min_pair_n = 20` is never
the binding constraint, m ≤ 400 SNPs, ≤ 40 kB per compressed `input.npz`.
"""
import numpy as np

N_SAMPLES = 120

# The single standard parameter set every Tier-A case is detected under. Stored
# verbatim into each meta.json so a golden can never be silently re-interpreted
# under different parameters (the golden test reads these back and passes them).
STD_PARAMS = {
    "ld_threshold": 0.6,
    "flank_kb": 300,
    "min_snps": 2,
    "top_n": 0,
    "sig_thresh": 1e-5,
    "adj_r2_min": 0.2,
    "min_pair_n": 20,
    "merge_iou": 0.3,
    "gap_factor": 10.0,
}

# Case -> generator seed. Distinct per case so latent draws are independent.
TIER_A_CASES = [
    "blocks_dense_ld",
    "blocks_two_separated",
    "blocks_gap_split",
    "blocks_below_min_snps",
    "blocks_monomorphic_window",
    "blocks_single_typed_marker",
]

CASE_SEED = {name: 20260421 + i for i, name in enumerate(TIER_A_CASES)}

CASE_REASON = {
    "blocks_dense_ld": "One dense connected component -> a single 5-SNP block; "
                       "exercises component detection with no contiguity or gap split.",
    "blocks_two_separated": "Two internally-correlated groups with no cross-group LD in one "
                            "seed window -> two disjoint components -> two blocks.",
    "blocks_gap_split": "One connected component split into two segments by a physical gap "
                        "exceeding gap_factor x median gap (contiguous_segments_by_adjacent).",
    "blocks_below_min_snps": "A seed whose flank window has >=2 markers but whose LD-connected "
                             "component is size 1 (= min_snps-1) -> dropped by the min_snps gate at "
                             "min_snps=2 (distinct from the m<2 single-marker skip); the "
                             "discriminating case for the min-SNP gate.",
    "blocks_monomorphic_window": "Window of >=2 markers that collapses to <2 after the variance "
                                 "filter -> emits nothing (monomorphic-window path).",
    "blocks_single_typed_marker": "A significant SNP alone in its flank window (m<2) -> the "
                                  "isolated/single-typed-marker skip -> emits nothing.",
}

_NOT_SIG = 0.5   # p-value for non-seed SNPs (never below sig_thresh)
_SIG = 1e-8      # p-value for seed SNPs


def _block_snps(rng, n, k, flip=0.02):
    """`k` SNPs sharing two latent haplotype columns, each with an independent
    per-allele flip rate. Dosage in {0,1,2}; pairwise r² ~ (1-2*flip)^2 ≈ 0.92 —
    comfortably above the 0.6 edge threshold so every within-block pair connects."""
    hap = rng.integers(0, 2, size=(n, 2))
    cols = []
    for _ in range(k):
        h = hap.copy()
        fmask = rng.random((n, 2)) < flip
        h[fmask] = 1 - h[fmask]
        cols.append(h.sum(axis=1).astype(np.int8))
    return np.column_stack(cols)


def _indep_snps(rng, n, k, maf=0.3):
    """`k` independent SNPs (background; low cross-r²)."""
    return rng.binomial(2, maf, size=(n, k)).astype(np.int8)


def _assemble(chrom_pos_geno, sid_prefix="s"):
    """chrom_pos_geno: list of (chrom, position, dosage_col, pvalue). Returns the
    input dict sorted by (chrom, position) with generated SNP ids."""
    chroms, positions, cols, pvals = [], [], [], []
    for chrom, pos, col, pv in chrom_pos_geno:
        chroms.append(str(chrom))
        positions.append(int(pos))
        cols.append(np.asarray(col, dtype=np.int8))
        pvals.append(float(pv))
    geno = np.column_stack(cols)
    sid = np.array([f"{sid_prefix}{c}_{p:08d}" for c, p in zip(chroms, positions)], dtype="<U24")
    return {
        "geno": geno.astype(np.int8),
        "chroms": np.array(chroms, dtype="<U8"),
        "positions": np.array(positions, dtype=np.int32),
        "sid": sid,
        "pvalue": np.array(pvals, dtype=np.float64),
    }


def build_case_input(name):
    """Deterministically build the input arrays for a Tier-A case."""
    rng = np.random.default_rng(CASE_SEED[name])
    n = N_SAMPLES

    if name == "blocks_dense_ld":
        block = _block_snps(rng, n, 5)                 # 5 correlated SNPs
        bg = _indep_snps(rng, n, 8)                    # background on chr2
        rows = []
        block_pos = [1000, 1100, 1200, 1300, 1400]
        for j, pos in enumerate(block_pos):
            rows.append(("1", pos, block[:, j], _SIG if j == 2 else _NOT_SIG))
        for j in range(bg.shape[1]):
            rows.append(("2", 5000 + j * 200, bg[:, j], _NOT_SIG))
        return _assemble(rows)

    if name == "blocks_two_separated":
        a = _block_snps(rng, n, 3)                     # group A
        b = _block_snps(rng, n, 3)                     # group B (independent latent)
        bg = _indep_snps(rng, n, 6)
        rows = []
        for j, pos in enumerate([1000, 1100, 1200]):
            rows.append(("1", pos, a[:, j], _SIG if j == 1 else _NOT_SIG))
        for j, pos in enumerate([5000, 5100, 5200]):
            rows.append(("1", pos, b[:, j], _NOT_SIG))
        for j in range(bg.shape[1]):
            rows.append(("2", 8000 + j * 200, bg[:, j], _NOT_SIG))
        return _assemble(rows)

    if name == "blocks_gap_split":
        block = _block_snps(rng, n, 7)                 # one component, all correlated
        bg = _indep_snps(rng, n, 6)
        rows = []
        # tight cluster, then a >10x-median gap, then a second tight cluster
        gpos = [1000, 1100, 1200, 1300, 50000, 50100, 50200]
        for j, pos in enumerate(gpos):
            rows.append(("1", pos, block[:, j], _SIG if j == 2 else _NOT_SIG))
        for j in range(bg.shape[1]):
            rows.append(("2", 9000 + j * 200, bg[:, j], _NOT_SIG))
        return _assemble(rows)

    if name == "blocks_below_min_snps":
        seed = _block_snps(rng, n, 1)[:, 0]            # lone significant seed, no LD partner
        neigh = _indep_snps(rng, n, 1)[:, 0]           # independent neighbour keeps window m>=2
        bg = _indep_snps(rng, n, 6)
        rows = [("1", 1000, seed, _SIG),               # seed's LD component is size 1 (min_snps-1)
                ("1", 1100, neigh, _NOT_SIG)]          # -> dropped by the min_snps gate at min_snps=2
        for j in range(bg.shape[1]):
            rows.append(("2", 4000 + j * 200, bg[:, j], _NOT_SIG))
        return _assemble(rows)

    if name == "blocks_monomorphic_window":
        seed = _block_snps(rng, n, 1)[:, 0]            # single polymorphic seed
        mono_a = np.full(n, 2, dtype=np.int8)          # monomorphic neighbours
        mono_b = np.full(n, 0, dtype=np.int8)
        bg = _indep_snps(rng, n, 6)
        rows = [("1", 1000, seed, _SIG),
                ("1", 1100, mono_a, _NOT_SIG),
                ("1", 1200, mono_b, _NOT_SIG)]
        for j in range(bg.shape[1]):
            rows.append(("2", 4000 + j * 200, bg[:, j], _NOT_SIG))
        return _assemble(rows)

    if name == "blocks_single_typed_marker":
        seed = _block_snps(rng, n, 1)[:, 0]            # lone significant marker on chr1
        bg = _indep_snps(rng, n, 8)                    # all other markers on chr2
        rows = [("1", 1000, seed, _SIG)]
        for j in range(bg.shape[1]):
            rows.append(("2", 5000 + j * 200, bg[:, j], _NOT_SIG))
        return _assemble(rows)

    raise KeyError(f"unknown Tier-A case: {name!r}")
