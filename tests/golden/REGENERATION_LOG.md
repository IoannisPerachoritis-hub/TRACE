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

## 2026-09-01 -- R2.2: missing-as-reference parse fix + MAF>=0.05 boundary alignment (TYPE c; synced from DEV)
- PROVENANCE: these fixtures were PRODUCED IN THE DEVELOPMENT REPOSITORY (Solanaceae-gwas, commit 9e661cc) and
  CANNOT be regenerated in TRACE-release -- release ships no source VCF and no QC checkpoint (benchmarks/qc_data is
  gitignored and untracked here). They were copied byte-for-byte from DEV, and the accompanying gwas/qc.py fix was
  applied identically. The real-data golden tests (test_golden_published, test_golden_blocks::...varitome...) skip on
  a clean clone / CI for want of qc_data; they cannot re-derive these values in this repo.
- reason (same defect as DEV): gwas/qc.py:137 built dosage with scikit-allel `to_n_alt()` at the default `fill=0`, so
  every missing genotype (`./.`, half-missing `0/.`) became 0 (reference) and the `G[G<0]=np.nan` guard was a no-op.
  Consequence: the per-SNP `--miss` and per-sample `--ind-miss` filters were inert, mean/LD-kNNi imputation never ran,
  and MAF + the VanRaden GRM's 2p were computed with missing counted as reference. Fix: `to_n_alt(fill=-1)` so missing
  -> NaN. Coupled MAF-boundary alignment (same change): keep_mask `(maf > maf_thresh)` -> `(maf >= maf_thresh)` and the
  QC report "Fail MAF" `(maf <= maf_thresh)` -> `(maf < maf_thresh)` (exact complement; Fail MAF + Pass MAF == Total).
- golden consequence (tomato_locule): the 21.7%-missing sample BGV006336 is now excluded (165 -> 164 samples); markers
  43,974 -> 43,749; on the corrected genotypes the occupancy detector emits 4 disjoint blocks (was 3) -- block 3's lead
  SL25ch02p47391467 falls to p=1.19e-6 (just above Bonferroni -> non-significant) and a 4th, NON-significant block at
  47,644,058-47,718,411 (P_perm 0.22) surfaces. No reported locus/conclusion changes (the chr2 locule locus stands,
  min p 2.7e-13 -> 4.2e-13; lead-block eta2 0.350 -> 0.367).
- cases synced (byte-for-byte from DEV): tomato_locule/{ld_blocks,haplotype_blocks,annotated_blocks}.csv +
  run_manifest.json (n_samples 165->164, n_snps 43974->43749, n_significant_bonf 12->11, n_significant_meff 44->43,
  n_blocks 3->4, meff_value 242->249, gwas_csv_sha256 30ebef29...); varitome_locule/{expected_blocks.csv,meta.json};
  GOLDEN_LOCK recomputed (6d42a073... -> f2733277...). Tier-A synthetic blocks_* UNCHANGED (the VCF parse never runs
  there).
- verification: release's byte-identical block-detection code (find_ld_clusters_genomewide, filter_contained_blocks,
  run_haplotype_block_gwas, annotate_ld_blocks -- all confirmed byte-identical to DEV), run on DEV's FIXED qc_data
  (platform_GWAS sha256 30ebef29... == the manifest), reproduces these fixtures EXACTLY (ld_blocks 4x6, haplotype 4x17,
  annotated 4x14, rebuilt manifest 164/43749/11/4/meff249). test_golden_lock recomputes the digest to f2733277...

