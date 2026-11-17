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
- reason: find_ld_clusters_genomewide processed overlapping +/-flank windows independently, emitting blocks that shared member markers (a marker tested in >1 block; the BH denominator over-counted). Greedy occupancy selection (accept candidates by best-member p-value, claim the whole span, discard overlapping candidates) makes blocks disjoint. Default ld_merge_mode flipped "iou" -> "occupancy" (iou/correlation still selectable). Published numbers move by design (D-109).
- tomato_locule: 6 overlapping blocks -> 3 disjoint (chr2 46,461,623-46,582,239 / 47,017,547-47,129,438 / 47,301,921-47,515,290); 18 candidates discarded. The published lead block 47,301,921-47,657,766 (Mean r2 0.427) resolves to the tighter 47,301,921-47,515,290 (Mean r2 0.626) -- the low-coherence tail is dropped, not merged.
- cases regenerated: tomato_locule/{ld_blocks,haplotype_blocks,annotated_blocks}.csv + run_manifest.json (now records ld_merge_mode); varitome_locule/expected_blocks.csv; GOLDEN_LOCK recomputed.
- Tier-A synthetic blocks_* UNCHANGED (one seed each -> already disjoint; occupancy == iou there) -- verified via test_golden_blocks, not regenerated.
- significance: no call flipped. All 3 remaining blocks FDR_BH < 0.05 (P_perm at the 0.000999 floor); the lead-locus block stays significant (eta2 0.347). The weak 0.005994 block was one of the discarded overlapping candidates.
- plumbing: capture_golden + test_golden_published._rebuild now thread ld_merge_mode via the manifest.

## 2026-09-01 -- missing-as-reference parse fix + MAF>=0.05 boundary alignment (published numbers move; synced from DEV)
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


## 2026-09-03 -- LD-block detection redesign (seed-component scope + mandatory coherence + correlation-only merge). Published numbers move.
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
  stale 165-sample checkpoint from before the missing-as-reference parse fix), so the real-data @golden/@manuscript
  tests skip on a clean clone and fail locally only when the stale QC is present -- the same situation recorded for
  the parse-fix sync. No push (freeze).

## 2026-09-04 -- Commit B: LD-block min_snps floor 3 -> 2 (GUI = CLI). Published numbers move.
- reason: the LD-block detector's minimum cluster size (min_snps) 3 -> 2 at every LIVE site (gwas/ld.py
  find_ld_clusters_genomewide + find_ld_blocks_graph signature defaults, cli.py detection call + run_manifest
  "LD_min_snps", the three GUI detection call sites -- tab_genome_wide, Post_GWAS_Analysis auto-detect, GWAS_analysis
  one-click -- + their metadata, benchmarks/capture_golden.py); the two DEAD ld.py functions and the Table-S8
  reproduction benchmarks stay at 3. [Correction, 17 Nov 2026: the same commit (7c6bbbd) moved
  benchmarks/reconstruct_table_s8_params.py to min_snps 2 as well (its _detect, which
  test_golden_blocks.py::test_varitome_locule_blocks_match_table_s8 uses); only benchmarks/rerun_downstream_demo.py
  stays at 3.] The GUI and CLI now share the same floor (the UI-cleanup governing rule). The
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
  164x43749 QC and copied byte-identical here; TRACE-release cannot regenerate them locally (stale 165-sample
  on-disk QC from before the missing-as-reference parse fix), so the real-data @golden/@manuscript tests skip on a
  clean clone and fail locally only when the stale QC is present. No push (freeze).

## 2026-11-17 -- the kinship-aware haplotype test becomes the default (Reviewer 2, point 3a). Published numbers move.
- reason: run_haplotype_block_gwas's default test is now the kinship-aware Freedman-Lane test
  (gwas/haplotype_kinship.py): a mixed model with the scan's own kernel for the block's chromosome (its LOCO
  kernel), the null model fitted by REML, the F test and the permutations on the data whitened by it. The CLI's
  `--hap-test` defaults to it; `--hap-test pcs` keeps the earlier test. capture_golden takes the test as a required
  keyword (the manifest records it) and builds the scan's kernels as the pipeline does (gwas.kinship.scan_kernels,
  from the imputed QC matrix as float32); its frozen haplotype columns gain kinship_model, h2_null and
  eta2_pcs_only (the earlier test's eta2).
- tomato_locule/haplotype_blocks.csv, every changed value, old -> new (P_perm and PValue stay 0.000999000999000999,
  the 1/1001 floor, in all four blocks; the counts, df and lead SNPs are unchanged):
  - 46,461,623-46,582,239: F_param = F_perm 19.862817014728254 -> 9.121456845576859; PValue_param
    8.736420877330159e-11 -> 1.4842817510206519e-05; eta2 0.2985566085233715 -> 0.16350176832126323; added
    kinship_model LOCO kernel, h2_null 0.8818453080777714, eta2_pcs_only 0.2985566085233715.
  - 46,799,142-46,914,856: F 18.690069933283134 -> 15.408534364775809; PValue_param 5.4681306067581076e-08 ->
    8.005306837908999e-07; eta2 0.1963447442194616 -> 0.1676507461605347; added LOCO kernel, h2_null
    0.8693474408882625, eta2_pcs_only 0.1963447442194616.
  - 47,017,547-47,129,438: F 13.118450398569394 -> 7.548474393364311; PValue_param 7.371161381870086e-12 ->
    4.547988964697973e-07; eta2 0.3487238378928092 -> 0.23553303351398328; added LOCO kernel, h2_null
    0.8800909568573887, eta2_pcs_only 0.3487238378928092.
  - 47,301,921-47,616,243: F 20.667557367627072 -> 6.191613734315433; PValue_param 2.202588113653581e-13 ->
    0.00012941637291169163; eta2 0.372942409397862 -> 0.151230329475897; added LOCO kernel, h2_null
    0.9337777828049028, eta2_pcs_only 0.372942409397862.
- tomato_locule/run_manifest.json: "hap_test": "kinship" added; meff_value 249.0 -> 248.0 and n_significant_meff
  43 -> 44. Cause (measured in the Table S8 check, revision_reports/tableS8_check_report.md, "Recommendation for the
  CC-5 regeneration"): the old M_eff rule took the ceiling of a sum that is 248 in exact arithmetic and lands just
  above 248 on this harness's float64, C-order matrix; the whole-number rule now in the code gives 248. The data,
  the blocks and the thread count play no part; the 249 / 43 recorded above (2026-09-01, 2026-09-04) are the old
  rule's. n_significant_bonf (11), n_blocks (4) and the settings are unchanged; the two machine paths keep the
  placeholder convention ("<qc_data>/...", "data/Sol_genes_SL3.csv").
- unchanged: tomato_locule/ld_blocks.csv and annotated_blocks.csv (byte-identical), varitome_locule, the Tier-A
  cases, GOLDEN_LOCK (it covers the expected_*.csv files only).
- test: tests/test_golden_published.py compares P_perm and PValue to 1e-12 as before, the other haplotype floats to
  a relative 1e-6: the kinship-aware test's F, parametric p, eta2 and h2_null depend on the BLAS thread count in
  their last digits (measured on this golden: at most 2.3e-7 relative between 1 and 12 threads). The golden tests
  pass at 12 and at 1 thread.
- provenance: produced in TRACE-scaling/batch14/tree (b14/work) by benchmarks/capture_golden.py with this manifest's
  settings, 12 threads, from the corrected 164x43749 QC copy in benchmarks/qc_data/tomato_locule_number; gwas
  imported from the tree (logged). The same cells come from the CLI at its defaults on the Varitome VCF and
  phenotype file (to a relative 3.5e-7 at most, the report's 4-digit columns at their 4 digits, P_perm as counts), and the
  four blocks from benchmarks/reconstruct_table_s8_params.py.
