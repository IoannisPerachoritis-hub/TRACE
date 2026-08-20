# TRACE CLI reference

Auto-generated from `cli.py` by `python scripts/gen_cli_reference.py`. Do not edit by hand — re-run the generator after changing the parser.

```text
usage: trace-gwas [-h] [--vcf VCF] [--pheno PHENO] [--trait TRAIT]
                  [--output OUTPUT] [--maf MAF] [--miss MISS] [--mac MAC]
                  [--ind-miss IND_MISS] [--info-thresh INFO_THRESH]
                  [--norm {none,zscore,log,yeojohnson,int}]
                  [--model {mlm,mlmm,farmcpu} [{mlm,mlmm,farmcpu} ...]]
                  [--n-pcs N_PCS] [--no-loco] [--n-pcs-mlm N_PCS_MLM]
                  [--n-pcs-mlmm N_PCS_MLMM] [--n-pcs-farmcpu N_PCS_FARMCPU]
                  [--covar COVAR] [--covar-cols COVAR_COLS]
                  [--sig-thresh {meff,bonferroni,fdr,PVALUE}] [--auto-pcs]
                  [--pc-strategy {band,closest_to_1}] [--max-pcs MAX_PCS]
                  [--pc-band-lo PC_BAND_LO] [--pc-band-hi PC_BAND_HI]
                  [--pc-parsimony-tol PC_PARSIMONY_TOL] [--subsampling]
                  [--boot-reps BOOT_REPS] [--boot-frac BOOT_FRAC]
                  [--boot-thresh BOOT_THRESH] [--boot-jobs BOOT_JOBS]
                  [--seed SEED] [--ld-r2 LD_R2] [--ld-flank-kb LD_FLANK_KB]
                  [--ld-seed-p LD_SEED_P] [--ld-top-n LD_TOP_N]
                  [--hap-perms HAP_PERMS] [--no-annotation]
                  [--genome-build {SL3,SL4}] [--species {tomato,custom}]
                  [--gene-model GENE_MODEL] [--no-triage]
                  [--triage-r2-coherent TRIAGE_R2_COHERENT]
                  [--triage-lead-r2-frac TRIAGE_LEAD_R2_FRAC]
                  [--hap-min-count HAP_MIN_COUNT]
                  [--hap-min-group-size HAP_MIN_GROUP_SIZE]
                  [--no-isolated-rescue]
                  [--isolated-max-interval-mb ISOLATED_MAX_INTERVAL_MB]
                  [--isolated-edge-flank-kb ISOLATED_EDGE_FLANK_KB]
                  [--isolated-low-res-kb ISOLATED_LOW_RES_KB]
                  [--snp-plots {none,capped,all}]
                  [--max-snp-plots MAX_SNP_PLOTS] [--collapse-r2 COLLAPSE_R2]
                  [--no-report] [--no-plots] [--export-qc] [--drop-alt]
                  [--n-chromosomes N_CHROMOSOMES] [-v] [--interactive]

TRACE — Trait Resolution and Candidate Evaluation (CLI)

options:
  -h, --help            show this help message and exit
  --vcf VCF             Path to VCF file (.vcf or .vcf.gz)
  --pheno PHENO         Path to phenotype CSV/TSV
  --trait TRAIT         Trait column name in phenotype file
  --output OUTPUT       Output directory
  --norm {none,zscore,log,yeojohnson,int}
                        Phenotype normalization: none, zscore, log,
                        yeojohnson, int (default: none)
  --model {mlm,mlmm,farmcpu} [{mlm,mlmm,farmcpu} ...]
                        GWAS models to run (default: mlm). mlmm/farmcpu
                        require mlm.
  --n-pcs N_PCS         Number of PCs as covariates (default: 4)
  --no-loco             Use global kinship instead of LOCO (for benchmarking)
  --n-pcs-mlm N_PCS_MLM
                        PCs for MLM (overrides --n-pcs)
  --n-pcs-mlmm N_PCS_MLMM
                        PCs for MLMM (overrides --n-pcs)
  --n-pcs-farmcpu N_PCS_FARMCPU
                        PCs for FarmCPU (overrides --n-pcs)
  --covar COVAR         CSV/TSV of user covariates: first column = sample ID
                        (matching the VCF), remaining columns = numeric
                        covariates. Added as fixed effects to every model,
                        alongside the PCs. Samples missing a covariate value
                        are dropped.
  --covar-cols COVAR_COLS
                        Comma-separated subset of covariate columns to use
                        (default: all columns).
  --sig-thresh {meff,bonferroni,fdr,PVALUE}
                        Significance reporting threshold: bonferroni
                        (default), meff (LD-aware), fdr (q<0.05), or a numeric
                        p-value such as 5e-8.
  --no-report           Skip HTML report generation
  --no-plots            Skip plot generation
  --export-qc           Export post-QC genotype matrix, SNP map, and phenotype
                        for benchmarking
  --drop-alt            Drop ALT chromosomes
  --n-chromosomes N_CHROMOSOMES
                        Number of chromosomes (default: auto-detect from VCF).
                        When set, only chromosomes 1..N are kept.
  -v, --verbose         Verbose logging
  --interactive         Interactive wizard — prompts for all options step by
                        step

QC thresholds:
  --maf MAF             MAF threshold (default: 0.05)
  --miss MISS           Per-SNP missingness max (default: 0.10)
  --mac MAC             Minor allele count minimum (default: 5)
  --ind-miss IND_MISS   Per-individual missingness max (default: 0.20)
  --info-thresh INFO_THRESH
                        Imputation quality threshold (default: 0.0 = disabled)

Auto PC selection:
  --auto-pcs            Auto-select PCs via lambda scan (overrides
                        --n-pcs/--n-pcs-*)
  --pc-strategy {band,closest_to_1}
                        Auto PC strategy (default: band)
  --max-pcs MAX_PCS     Max PCs to scan in auto mode (default: 10)
  --pc-band-lo PC_BAND_LO
                        Lower lambda_GC bound for band strategy (default:
                        0.95)
  --pc-band-hi PC_BAND_HI
                        Upper lambda_GC bound for band strategy (default:
                        1.05)
  --pc-parsimony-tol PC_PARSIMONY_TOL
                        Parsimony tolerance for band fallback (default: 0.02)

Subsampling stability:
  --subsampling         Run subsampling GWAS stability screening (MLM only)
  --boot-reps BOOT_REPS
                        Subsampling iterations (default: 50)
  --boot-frac BOOT_FRAC
                        Sample fraction per iteration (default: 0.80)
  --boot-thresh BOOT_THRESH
                        Discovery p-threshold (default: 1e-4)
  --boot-jobs BOOT_JOBS
                        Parallel workers for subsampling (default: 1, -1=all
                        cores)
  --seed SEED           RNG seed for subsampling and permutation (default:
                        42). Set to make CLI runs bit-for-bit reproducible.

LD & post-GWAS:
  --ld-r2 LD_R2         LD r^2 threshold for block detection (default: 0.6)
  --ld-flank-kb LD_FLANK_KB
                        LD flank window in kb (default: auto from LD decay)
  --ld-seed-p LD_SEED_P
                        Seed SNP p-threshold for LD blocks (default: 1e-5)
  --ld-top-n LD_TOP_N   Also seed top-N SNPs (default: 10)
  --hap-perms HAP_PERMS
                        Haplotype permutations (default: 1000)
  --no-annotation       Skip gene annotation
  --genome-build {SL3,SL4}
                        Tomato gene-model assembly: SL3 = SL3.1 (default), SL4
                        = ITAG4.0. Note: Varitome SNPs are in SL2.5, so gene
                        coordinates in either build are offset from the SNP
                        positions (SL3 by ~0.5 Mb, SL4 more) and annotation is
                        positional, not coordinate-exact.
  --species {tomato,custom}
                        Species for annotation files (default: tomato)
  --gene-model GENE_MODEL
                        Gene coordinate CSV (required if --species custom)
  --no-triage           Disable the LD-quality triage table (per-block view
                        recommendation). Triage is ON by default; --no-triage
                        reproduces byte-identical output (no LD_triage_*.csv).
  --triage-r2-coherent TRIAGE_R2_COHERENT
                        Triage LD-coherence gate (default: the run's --ld-r2).
                        Binding it to the edge threshold makes triage a self-
                        consistency check on the detector, not a new opinion.
  --triage-lead-r2-frac TRIAGE_LEAD_R2_FRAC
                        Triage: fraction of block members that must track the
                        lead SNP (CONVENTION, not a derived quantity; default
                        0.5).
  --hap-min-count HAP_MIN_COUNT
                        Min samples for a multi-locus genotype to be tested
                        (else labelled 'Other'; default 5).
  --hap-min-group-size HAP_MIN_GROUP_SIZE
                        Min tested-group size for the haplotype F-test
                        (default 3).
  --no-isolated-rescue  Disable the isolated-SNP rescue (reporting-significant
                        SNPs that form no LD block). Rescue is ON by default;
                        this is the byte-exact reproduction escape hatch.
  --isolated-max-interval-mb ISOLATED_MAX_INTERVAL_MB
                        Max flanking-marker interval width (Mb) before
                        clamping (default: 5.0).
  --isolated-edge-flank-kb ISOLATED_EDGE_FLANK_KB
                        Flank (kb) for chromosome-edge isolated intervals
                        (default: the run's LD flank, else 300).
  --isolated-low-res-kb ISOLATED_LOW_RES_KB
                        Interval width (kb) above which an isolated interval
                        is flagged low-resolution (default: max(2x LD decay,
                        400)).
  --snp-plots {none,capped,all}
                        Per-SNP effect boxplots in the HTML report: none,
                        capped (top --max-snp-plots), or all (default:
                        capped). The Significant_SNPs table is always complete
                        regardless.
  --max-snp-plots MAX_SNP_PLOTS
                        Max per-SNP boxplots embedded when --snp-plots=capped
                        (default: 24).
  --collapse-r2 COLLAPSE_R2
                        PLOTS ONLY: collapse near-redundant significant SNPs
                        (r2 >= this to a representative) before plotting; the
                        table is unaffected (default: 0.9).

Examples:
  python cli.py --vcf data.vcf.gz --pheno pheno.csv --trait Yield --output results/
```
