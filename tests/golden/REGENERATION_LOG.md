# Golden regeneration log

Append-only. One entry per deliberate regeneration of a pinned golden.

## bootstrap — initial Tier-A fixtures
- reason: initial creation of the six Tier-A block-detector goldens (T-95)
- cases: blocks_dense_ld, blocks_two_separated, blocks_gap_split, blocks_below_min_snps, blocks_monomorphic_window, blocks_single_typed_marker

## Batch B — Tier-B varitome_locule golden added
- reason: T-61 confirmed the reproducing params (flank_kb=144, ld_decay_kb=72.17); populated tests/golden/varitome_locule/expected_blocks.csv (Table S8 six MLM blocks, read back from an actual run). GOLDEN_LOCK recomputed to include it.

## 2026-08-25 — lead_snp = real min-p seed (column rename + value change; P_perm shift)
- reason: replaced the fake "Lead SNP" column (the seed-UNION, printed truncated to its first token) with a single real `lead_snp` = the most-significant SEEDING SNP of each block (min GWAS p-value; ties -> smaller position), plus a new `lead_snp_pvalue` column. See lead_snp_work_order.md. The lead is now a genuine min-p seed (may be a non-member).
- cases: all six Tier-A `blocks_*` (expected_blocks + expected_blocks_filtered), varitome_locule/expected_blocks.csv, tomato_locule/ld_blocks.csv (+ lead_snp_pvalue), tomato_locule/haplotype_blocks.csv.
- P_perm: the block RNG seed dropped its SNP-id argument (P3), so permutation p-values re-drew: 5/6 tomato blocks stay at the 0.000999 floor; block 47092604-47168061 (F_obs 8.50) moved 0.004995 -> 0.005994 (within [0.002,0.009], below 0.01). eta2/F_param/F_perm/PValue_param/boundaries/n_snps/n_haplotypes byte-identical. annotated_blocks + run_manifest unchanged.
- inputs (input.npz) unchanged; GOLDEN_LOCK recomputed.

