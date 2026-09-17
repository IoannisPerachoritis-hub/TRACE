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
- PROVENANCE: these fixtures were PRODUCED IN THE DEVELOPMENT REPOSITORY (commit 9e661cc) and
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


## 2026-09-03 -- LD-block detection redesign (seed-component scope + mandatory coherence + correlation-only merge). TYPE (c)
- reason: three defects in `gwas/ld.py` block detection were fixed together (work order v2). (1) Blocks emitted every
  connected component in a seed's window stamped with the seed as "Lead SNP" even when the seed was not a member;
  (2) within-block coherence was not required by default; (3) the merge fused overlapping candidates on interval
  overlap with no r2 test. Redesign: emit ONLY the seed's connected component (every block now contains its lead),
  split each block toward within-block coherence FOLLOWING the seed and never annihilating it, then merge overlapping
  candidates by correlation (cross-seam AND union mean r2 both >= --ld-merge-r2) and resolve residual overlap by
  occupancy. `--ld-merge-mode` removed; `--ld-merge-r2` now governs both the within-block split and the merge test.
- before/after (tomato_locule): 4 blocks -> 3. Blocks 1 and 2 byte-identical; block 3 extends to absorb the marker
  47,616,243 (eta2 0.3626 -> 0.3729, stays significant); the pre-redesign block 4 (non-member lead 47,616,243,
  eta2 0.0098, P_perm 0.2198 = non-significant) is removed. All acceptance STOP conditions PASS (no P_perm crossing,
  no block loses its lead from members, zero overlaps after the merge); meff/n_significant unchanged.
- files regenerated: tomato_locule/{ld_blocks,haplotype_blocks,annotated_blocks}.csv + run_manifest.json;
  varitome_locule/{expected_blocks.csv,meta.json}; blocks_two_separated (2->1) and blocks_gap_split (2->1); the other
  four Tier-A cases are byte-identical. GOLDEN_LOCK f2733277... -> c768db78...
- provenance: these fixtures were PRODUCED IN THE DEVELOPMENT REPOSITORY (the development tree) from the corrected
  164x43749 QC and copied byte-identical here. TRACE-release cannot regenerate them locally (its on-disk QC is the
  stale pre-R2.2 165-sample checkpoint), so the real-data @golden/@manuscript tests skip on a clean clone and fail
  locally only when the stale QC is present -- the same situation recorded for the R2.2 sync. No push (freeze).

## 2026-09-04 -- Commit B: LD-block min_snps floor 3 -> 2 (GUI = CLI). TYPE (c)
- reason: the LD-block detector's minimum cluster size (min_snps) 3 -> 2 at every LIVE site (gwas/ld.py
  find_ld_clusters_genomewide + find_ld_blocks_graph signature defaults, cli.py detection call + run_manifest
  "LD_min_snps", the three GUI detection call sites -- tab_genome_wide, Post_GWAS_Analysis auto-detect, GWAS_analysis
  one-click -- + their metadata, benchmarks/capture_golden.py); the two DEAD ld.py functions and the Table-S8
  reproduction benchmarks stay at 3. The GUI and CLI now share the same floor (the UI-cleanup governing rule). The
  haplotype stage (min_hap_count=5, min_group_size=3) remains the real thin-block protection, so a 2-marker floor is
  defensible.
- before/after (tomato_locule): 3 blocks -> 4. A new significant 2-marker block surfaces (46,799,142-46,914,856,
  eta2 0.196, P_perm floor) that min_snps=3 discarded as below-size; the three prior blocks are unchanged except
  block 4's lead shifts member -> member (SL25ch02p47391467 -> SL25ch02p47301921; coordinates unchanged, so P_perm
  floor unchanged). STOP conditions all PASS (no P_perm crossing, no non-member lead, zero overlaps); meff (249) /
  n_significant_meff (43) / n_significant_bonf (11) unchanged.
- files regenerated: tomato_locule/{ld_blocks,haplotype_blocks,annotated_blocks}.csv + run_manifest.json (3 -> 4;
  manifest min_snps 2); varitome_locule/{expected_blocks.csv,meta.json} (3 -> 4). Tier-A: blocks_below_min_snps input
  redesigned to a size-1 LD component + one independent neighbour (still emits 0 at min_snps=2; discriminates at the
  relaxed min_snps=1) -- its input.npz + meta.json move but its expected_blocks.csv is byte-identical; all six Tier-A
  meta.json record params.min_snps=2; NO Tier-A expected_*.csv moved. GOLDEN_LOCK c768db78... -> 7439a5cb... (driven
  solely by varitome_locule/expected_blocks.csv).
- provenance: these fixtures were PRODUCED IN THE DEVELOPMENT REPOSITORY (the development tree) from the corrected
  164x43749 QC and copied byte-identical here; TRACE-release cannot regenerate them locally (stale pre-R2.2
  165-sample on-disk QC), so the real-data @golden/@manuscript tests skip on a clean clone and fail locally only when
  the stale QC is present. No push (freeze).
