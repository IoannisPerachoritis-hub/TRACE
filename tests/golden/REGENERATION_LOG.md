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

## 2026-08-28 -- D-109: disjoint LD blocks via greedy occupancy selection (default flip iou -> occupancy)
- reason: find_ld_clusters_genomewide processed overlapping +/-flank windows independently, emitting blocks that shared member markers (a marker tested in >1 block; the BH denominator over-counted). Greedy occupancy selection (accept candidates by best-member p-value, claim the whole span, discard overlapping candidates) makes blocks disjoint. Default ld_merge_mode flipped "iou" -> "occupancy" (iou/correlation still selectable). TYPE (c) -- published numbers move by design (D-109).
- tomato_locule: 6 overlapping blocks -> 3 disjoint (chr2 46,461,623-46,582,239 / 47,017,547-47,129,438 / 47,301,921-47,515,290); 18 candidates discarded. The published lead block 47,301,921-47,657,766 (Mean r2 0.427) resolves to the tighter 47,301,921-47,515,290 (Mean r2 0.626) -- the low-coherence tail is dropped, not merged.
- cases regenerated: tomato_locule/{ld_blocks,haplotype_blocks,annotated_blocks}.csv + run_manifest.json (now records ld_merge_mode); varitome_locule/expected_blocks.csv; GOLDEN_LOCK recomputed.
- Tier-A synthetic blocks_* UNCHANGED (one seed each -> already disjoint; occupancy == iou there) -- verified via test_golden_blocks, not regenerated.
- significance: no call flipped. All 3 remaining blocks FDR_BH < 0.05 (P_perm at the 0.000999 floor); the lead-locus block stays significant (eta2 0.347). The weak 0.005994 block was one of the discarded overlapping candidates.
- plumbing: capture_golden + test_golden_published._rebuild now thread ld_merge_mode via the manifest.

