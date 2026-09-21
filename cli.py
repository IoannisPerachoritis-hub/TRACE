"""
Command-line interface for TRACE (Trait Resolution and Candidate Evaluation).

Runs the full GWAS pipeline headlessly — suitable for HPC clusters,
batch processing, and reproducible scripted analyses.

Usage:
    trace-gwas --vcf input.vcf.gz --pheno pheno.csv --trait Yield --output results/
"""
import argparse
import logging
import sys
from pathlib import Path

# Headless matplotlib backend (must be before any mpl import)
import matplotlib
matplotlib.use("Agg")


def _sig_thresh_type(s):
    """argparse type for --sig-thresh: a named rule or a numeric p-value in (0, 1]."""
    if s in ("meff", "bonferroni", "fdr"):
        return s
    try:
        v = float(s)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"--sig-thresh must be meff/bonferroni/fdr or a p-value like 5e-8 (got {s!r})")
    if not (0.0 < v <= 1.0):
        raise argparse.ArgumentTypeError(f"--sig-thresh p-value must be in (0, 1] (got {v})")
    return v


def _build_parser():
    parser = argparse.ArgumentParser(
        description="TRACE: Trait Resolution and Candidate Evaluation (CLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python cli.py --vcf data.vcf.gz --pheno pheno.csv --trait Yield --output results/\n"
        ),
    )

    # ── GWAS required arguments ─────────────────────────
    parser.add_argument("--vcf", help="Path to VCF file (.vcf or .vcf.gz)")
    parser.add_argument("--pheno", help="Path to phenotype CSV/TSV")
    parser.add_argument("--trait", help="Trait column name in phenotype file")
    parser.add_argument("--output", help="Output directory")

    # QC parameters
    qc = parser.add_argument_group("QC thresholds")
    qc.add_argument("--maf", type=float, default=0.05, help="MAF threshold (default: 0.05)")
    qc.add_argument("--miss", type=float, default=0.10, help="Per-SNP missingness max (default: 0.10)")
    qc.add_argument("--mac", type=int, default=5, help="Minor allele count minimum (default: 5)")
    qc.add_argument("--ind-miss", type=float, default=0.20, help="Per-individual missingness max (default: 0.20)")
    qc.add_argument("--info-thresh", type=float, default=0.0,
                     help="Imputation quality threshold (default: 0.0 = disabled)")
    # Imputation of the GWAS mixed-model genotype matrix. Default mean is byte-identical
    # to prior behaviour; ldknni is LD-kNNi (Money et al. 2015), opt-in.
    qc.add_argument("--impute", default="mean", choices=["mean", "ldknni"],
                     help="Genotype imputation for the GWAS matrix (default: mean). "
                          "'ldknni' = LD-kNNi (Money et al. 2015): a discrete, "
                          "LD-weighted k-NN imputer. Does NOT touch geno_dosage_raw.")
    qc.add_argument("--impute-k", type=int, default=5,
                     help="LD-kNNi neighbours per call (default: 5).")
    qc.add_argument("--impute-l", type=int, default=20,
                     help="LD-kNNi predictor SNPs by r2 (default: 20).")
    # P13: heterozygosity screen (one-sided heterozygote EXCESS; NOT two-sided HWE).
    # Both OFF by default -> byte-identical QC. Excess het flags paralog / repeat-
    # mismapping artifacts; heterozygote deficit (expected in selfers) is never screened.
    qc.add_argument("--max-het", type=float, default=None,
                     help="Heterozygosity screen: remove variants with observed "
                          "heterozygosity above this rate, e.g. 0.20 (default: off).")
    qc.add_argument("--het-excess-p", type=float, default=None,
                     help="Heterozygosity screen: remove variants failing a one-sided "
                          "heterozygote-excess test at this p-value (default: off).")

    # Normalization (short slugs for shell-friendliness)
    parser.add_argument(
        "--norm", default="none",
        choices=["none", "zscore", "log", "yeojohnson", "int"],
        help=("Phenotype normalisation slug: 'int' (rank-based inverse normal) "
              "recommended; 'log'/'yeojohnson' change the distribution; 'zscore' is "
              "scaling only (p-value-neutral). Default: none."),
    )

    # Model selection
    parser.add_argument(
        "--model", nargs="+", default=["mlm"],
        choices=["mlm", "mlmm", "farmcpu"],
        help="GWAS models to run (default: mlm). mlmm/farmcpu require mlm.",
    )
    parser.add_argument("--n-pcs", type=int, default=0, help="Number of PCs as covariates (default: 0)")
    parser.add_argument("--no-loco", action="store_true",
                        help="Use global kinship instead of LOCO (for benchmarking)")
    parser.add_argument("--n-pcs-mlm", type=int, default=None, help="PCs for MLM (overrides --n-pcs)")
    parser.add_argument("--n-pcs-mlmm", type=int, default=None, help="PCs for MLMM (overrides --n-pcs)")
    parser.add_argument("--n-pcs-farmcpu", type=int, default=None, help="PCs for FarmCPU (overrides --n-pcs)")
    parser.add_argument(
        "--covar",
        help="CSV/TSV of user covariates: first column = sample ID (matching the VCF), "
             "remaining columns = numeric covariates. Added as fixed effects to every "
             "model, alongside the PCs. Samples missing a covariate value are dropped.")
    parser.add_argument(
        "--covar-cols",
        help="Comma-separated subset of covariate columns to use (default: all columns).")

    # Significance threshold
    parser.add_argument(
        "--sig-thresh", default="bonferroni", type=_sig_thresh_type,
        metavar="{meff,bonferroni,fdr,PVALUE}",
        help="Significance reporting threshold: bonferroni (default), meff (LD-aware), "
             "fdr (q<0.05), or a numeric p-value such as 5e-8.",
    )

    # Subsampling GWAS
    boot_grp = parser.add_argument_group("Subsampling stability")
    boot_grp.add_argument("--subsampling", action="store_true",
                           help="Run subsampling GWAS stability screening (MLM only)")
    boot_grp.add_argument("--boot-reps", type=int, default=50,
                           help="Subsampling iterations (default: 50)")
    boot_grp.add_argument("--boot-frac", type=float, default=0.80,
                           help="Sample fraction per iteration (default: 0.80)")
    boot_grp.add_argument("--boot-thresh", type=float, default=1e-4,
                           help="Discovery p-threshold (default: 1e-4)")
    boot_grp.add_argument("--boot-jobs", type=int, default=1,
                           help="Parallel workers for subsampling (default: 1, -1=all cores)")
    boot_grp.add_argument("--seed", type=int, default=42,
                           help="RNG seed for subsampling and permutation (default: 42). "
                                "Set to make CLI runs bit-for-bit reproducible.")

    # LD + Haplotype + Annotation
    ld_grp = parser.add_argument_group("LD & post-GWAS")
    ld_grp.add_argument("--ld-r2", type=float, default=0.6,
                         help="LD r^2 threshold for block detection (default: 0.6)")
    ld_grp.add_argument("--ld-flank-kb", type=int, default=None,
                         help="LD flank window in kb (default: auto from LD decay)")
    ld_grp.add_argument("--ld-seed-mode", default="suggestive",
                         choices=["suggestive", "significant"],
                         help="LD-block seed SNPs (mirrors the GUI): 'suggestive' "
                              "(default) = the --ld-seed-p threshold plus the "
                              "--ld-top-n floor; 'significant' = only SNPs passing "
                              "the genome-wide --sig-thresh, with no top-N floor.")
    ld_grp.add_argument("--ld-seed-p", type=float, default=1e-5,
                         help="Seed SNP p-threshold for LD blocks in suggestive "
                              "mode (default: 1e-5)")
    ld_grp.add_argument("--ld-top-n", type=int, default=10,
                         help="Suggestive-mode FLOOR: always also seed the top-N "
                              "SNPs by p-value, even when fewer than N pass "
                              "--ld-seed-p (default: 10; 0 disables the floor).")
    ld_grp.add_argument("--ld-merge-r2", type=float, default=0.5,
                         help="Within-block coherence threshold (mean off-diagonal "
                              "r^2). Governs BOTH the seed-following within-block "
                              "coherence split and the cross-block merge test: a block "
                              "is split (following its seed) until its members reach "
                              "this mean r^2, and two overlapping blocks fuse only when "
                              "their cross-seam AND union mean r^2 both reach it "
                              "(default: 0.5).")
    ld_grp.add_argument("--hap-perms", type=int, default=1000,
                         help="Haplotype permutations (default: 1000)")
    ld_grp.add_argument("--no-annotation", action="store_true",
                         help="Skip gene annotation")
    ld_grp.add_argument("--genome-build", default="SL3", choices=["SL3", "SL4"],
                         help="Tomato gene-model assembly: SL3 = SL3.1 (default), SL4 = ITAG4.0. "
                              "Note: Varitome SNPs are in SL2.5, so gene coordinates in either build are "
                              "offset from the SNP positions (SL3 by ~0.5 Mb, SL4 more) and annotation is "
                              "positional, not coordinate-exact.")
    ld_grp.add_argument("--species", default="tomato", choices=["tomato", "custom"],
                         help="Species for annotation files (default: tomato)")
    ld_grp.add_argument("--gene-model", help="Gene coordinate CSV (required if --species custom)")
    ld_grp.add_argument("--no-triage", action="store_true",
                         help="Disable the LD-quality triage table (per-block view recommendation). Triage is "
                              "ON by default; --no-triage reproduces byte-identical output (no LD_triage_*.csv).")
    ld_grp.add_argument("--triage-r2-coherent", type=float, default=None,
                         help="Triage LD-coherence gate (default: the run's --ld-r2). Binding it to the edge "
                              "threshold makes triage a self-consistency check on the detector, not a new opinion.")
    ld_grp.add_argument("--triage-lead-r2-frac", type=float, default=0.5,
                         help="Triage: fraction of block members that must track the lead SNP (CONVENTION, not "
                              "a derived quantity; default 0.5).")
    ld_grp.add_argument("--hap-min-count", type=int, default=5,
                         help="Min samples for a multi-locus genotype to be tested (else labelled 'Other'; "
                              "default 5).")
    ld_grp.add_argument("--hap-min-group-size", type=int, default=3,
                         help="Min tested-group size for the haplotype F-test (default 3).")
    # ── Isolated-SNP rescue (T-45) ──
    ld_grp.add_argument("--no-isolated-rescue", action="store_true",
                         help="Disable the isolated-SNP rescue (reporting-significant SNPs that "
                              "form no LD block). Rescue is ON by default; this is the byte-exact "
                              "reproduction escape hatch.")
    ld_grp.add_argument("--isolated-max-interval-mb", type=float, default=5.0,
                         help="Max flanking-marker interval width (Mb) before clamping (default: 5.0).")
    ld_grp.add_argument("--isolated-edge-flank-kb", type=int, default=None,
                         help="Flank (kb) for chromosome-edge isolated intervals "
                              "(default: the run's LD flank, else 300).")
    ld_grp.add_argument("--isolated-low-res-kb", type=int, default=None,
                         help="Interval width (kb) above which an isolated interval is flagged "
                              "low-resolution (default: max(2x LD decay, 400)).")
    ld_grp.add_argument("--snp-plots", choices=("none", "capped", "all"), default="capped",
                         help="Per-SNP effect boxplots in the HTML report: none, capped "
                              "(top --max-snp-plots), or all (default: capped). The "
                              "Significant_SNPs table is always complete regardless.")
    ld_grp.add_argument("--max-snp-plots", type=int, default=24,
                         help="Max per-SNP boxplots embedded when --snp-plots=capped (default: 24).")
    ld_grp.add_argument("--collapse-r2", type=float, default=0.9,
                         help="PLOTS ONLY: collapse near-redundant significant SNPs (r2 >= this to a "
                              "representative) before plotting; the table is unaffected (default: 0.9).")

    # ── Output options ────────────────────────────────────
    parser.add_argument("--no-report", action="store_true", help="Skip HTML report generation")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--export-qc", action="store_true",
                         help="Export post-QC genotype matrix, SNP map, and phenotype for benchmarking")
    # P11: --drop-alt removed. Unplaced/scaffold (ALT) markers carry no valid genomic
    # position, so no position-dependent step can use them and they must never enter
    # the significance divisor; TRACE now always excludes them (no flag).
    parser.add_argument("--n-chromosomes", type=int, default=None,
                         help="Number of chromosomes (default: auto-detect from VCF). "
                              "When set, only chromosomes 1..N are kept.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--interactive", action="store_true",
                         help="Interactive wizard: prompts for all options step by step")

    return parser


# ── Interactive wizard ──────────────────────────────────


def _prompt_file(label, must_exist=True):
    """Prompt for a file path with existence validation."""
    import click
    from colorama import Fore, Style

    while True:
        path = click.prompt(label)
        p = Path(path)
        if must_exist and not p.exists():
            print(f"{Fore.RED}  x File not found: {path}{Style.RESET_ALL}")
            continue
        size = p.stat().st_size
        if size > 1_000_000:
            print(f"{Fore.GREEN}  + Found ({size / 1_000_000:.1f} MB){Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}  + Found ({size / 1_000:.0f} KB){Style.RESET_ALL}")
        return str(p)


def _peek_phenotype(path):
    """Quick-read phenotype file to show column names."""
    import pandas as pd

    try:
        sep = "\t" if Path(path).suffix in (".tsv", ".txt") else ","
        return pd.read_csv(path, sep=sep, index_col=0, nrows=5)
    except Exception:
        return None


def _build_equivalent_command(args):
    """Build the equivalent non-interactive CLI command string."""
    parts = ["python cli.py"]

    parts.append(f"--vcf {args.vcf}")
    parts.append(f"--pheno {args.pheno}")
    parts.append(f"--trait {args.trait}")
    parts.append(f"--output {args.output}")
    if args.model != ["mlm"]:
        parts.append(f"--model {' '.join(args.model)}")
    if args.n_pcs != 0:
        parts.append(f"--n-pcs {args.n_pcs}")
    if args.subsampling:
        parts.append(f"--subsampling --boot-reps {args.boot_reps}")
        if args.boot_jobs != 1:
            parts.append(f"--boot-jobs {args.boot_jobs}")
    if getattr(args, "sig_thresh", "bonferroni") != "bonferroni":
        parts.append(f"--sig-thresh {args.sig_thresh}")
    if args.maf != 0.05:
        parts.append(f"--maf {args.maf}")
    if args.miss != 0.10:
        parts.append(f"--miss {args.miss}")
    if args.mac != 5:
        parts.append(f"--mac {args.mac}")
    if args.norm != "none":
        parts.append(f"--norm {args.norm}")
    if args.no_annotation:
        parts.append("--no-annotation")
    if args.verbose:
        parts.append("-v")

    return " \\\n    ".join(parts)


def _print_summary(args):
    """Print a formatted summary of all selected options."""
    from colorama import Fore, Style

    print(f"\n{Fore.CYAN}── Summary ──{Style.RESET_ALL}")

    print(f"  VCF:        {args.vcf}")
    print(f"  Phenotype:  {args.pheno}")
    print(f"  Trait:      {args.trait}")
    print(f"  Output:     {args.output}")
    print(f"  Models:     {', '.join(m.upper() for m in args.model)}")
    print(f"  PCs:        {args.n_pcs} (fixed)")
    _thresh_labels = {"meff": "M_eff (LD-aware)", "bonferroni": "Bonferroni", "fdr": "FDR q<0.05"}
    print(f"  Threshold:  {_thresh_labels.get(args.sig_thresh, args.sig_thresh)}")
    print(f"  QC:         MAF={args.maf}, miss={args.miss}, MAC={args.mac}")
    if args.subsampling:
        print(f"  Subsampling:  {args.boot_reps} reps, {args.boot_jobs} workers")
    if not args.no_annotation:
        print(f"  Species:    {args.species}")
    if args.no_annotation:
        print("  Annotation: Disabled")


def _interactive_wizard(parser):
    """Step-by-step interactive CLI wizard. Returns populated argparse.Namespace."""
    import click
    from colorama import init, Fore, Style
    init(autoreset=True)

    args = parser.parse_args([])  # start with all defaults

    print(f"\n{Fore.CYAN}{'=' * 44}")
    print("  TRACE  --  Interactive Mode")
    print(f"{'=' * 44}{Style.RESET_ALL}")

    # ── Step 1: Input files ──
    print(f"\n{Fore.YELLOW}-- Input Files --{Style.RESET_ALL}")
    args.vcf = _prompt_file("VCF file path")
    args.pheno = _prompt_file("Phenotype file path")

    pheno_df = _peek_phenotype(args.pheno)
    if pheno_df is not None:
        cols = list(pheno_df.columns)
        print(f"  Available columns: {', '.join(cols[:20])}")
        if len(cols) > 20:
            print(f"  ... and {len(cols) - 20} more")

    while True:
        args.trait = click.prompt("Trait column name")
        if pheno_df is not None and args.trait not in pheno_df.columns:
            print(f"{Fore.RED}  x '{args.trait}' not found in columns!{Style.RESET_ALL}")
            continue
        if pheno_df is not None:
            print(f"{Fore.GREEN}  + Column found{Style.RESET_ALL}")
        break

    args.output = click.prompt("Output directory", default="results/")

    # ── Step 2: Models ──
    print(f"\n{Fore.YELLOW}-- Models --{Style.RESET_ALL}")
    model_choice = click.prompt(
        "  [1] MLM only (fastest)\n"
        "  [2] MLM + MLMM\n"
        "  [3] MLM + FarmCPU\n"
        "  [4] MLM + MLMM + FarmCPU\n"
        "Choice",
        type=click.IntRange(1, 4), default=1,
    )
    _model_map = {
        1: ["mlm"], 2: ["mlm", "mlmm"],
        3: ["mlm", "farmcpu"], 4: ["mlm", "mlmm", "farmcpu"],
    }
    args.model = _model_map[model_choice]

    # ── Step 3: PCs ──
    print(f"\n{Fore.YELLOW}-- Principal Components --{Style.RESET_ALL}")
    print("  TRACE uses a fixed number of PCs (no auto-selection); the run report includes")
    print("  eigenvalue-spectrum + conventional-criteria diagnostics to inform the choice.")
    args.n_pcs = click.prompt("  Number of PCs", type=int, default=0)

    # ── Step 4: Optional features ──
    print(f"\n{Fore.YELLOW}-- Optional Features --{Style.RESET_ALL}")

    # Normalization
    norm_choice = click.prompt(
        "Phenotype normalization",
        type=click.Choice(["none", "zscore", "log", "yeojohnson", "int"]),
        default="none",
    )
    args.norm = norm_choice

    # Significance threshold
    sig_choice = click.prompt(
        "Significance threshold\n"
        "  [1] M_eff -- LD-aware Bonferroni (recommended)\n"
        "  [2] Bonferroni (most conservative)\n"
        "  [3] FDR q < 0.05 (least conservative)\n"
        "Choice",
        type=click.IntRange(1, 3), default=1,
    )
    _SIG_MAP = {1: "meff", 2: "bonferroni", 3: "fdr"}
    args.sig_thresh = _SIG_MAP[sig_choice]

    # Subsampling
    if click.confirm("Run subsampling stability screening?", default=False):
        args.subsampling = True
        args.boot_reps = click.prompt("  Reps", type=int, default=50)
        args.boot_jobs = click.prompt("  Parallel workers (-1=all cores)", type=int, default=1)

    # QC thresholds
    if click.confirm("Customize QC thresholds?", default=False):
        args.maf = click.prompt("  MAF threshold", type=float, default=0.05)
        args.miss = click.prompt("  SNP missingness max", type=float, default=0.10)
        args.mac = click.prompt("  Minor allele count min", type=int, default=5)
        args.ind_miss = click.prompt("  Individual missingness max", type=float, default=0.20)
    else:
        print(f"  Using defaults: MAF={args.maf}, miss={args.miss}, MAC={args.mac}")

    # LD parameters
    if click.confirm("Customize LD/haplotype parameters?", default=False):
        args.ld_r2 = click.prompt("  LD r^2 threshold", type=float, default=0.6)
        args.hap_perms = click.prompt("  Haplotype permutations", type=int, default=1000)
        args.ld_top_n = click.prompt("  Top-N SNP seeds", type=int, default=10)
    else:
        print(f"  Using defaults: r^2={args.ld_r2}, {args.hap_perms} permutations")

    # Annotation
    if click.confirm("Skip gene annotation?", default=False):
        args.no_annotation = True

    # Species (for annotation)
    if not args.no_annotation:
        args.species = click.prompt(
            "Species (for gene annotation)",
            type=click.Choice(["tomato", "custom"]),
            default="tomato",
        )

    # Verbose
    args.verbose = click.confirm("Verbose logging?", default=False)

    # ── Step 5: Summary + confirm ──
    _print_summary(args)
    equiv_cmd = _build_equivalent_command(args)
    print(f"\n{Fore.CYAN}Equivalent command (save for reproducibility):{Style.RESET_ALL}")
    print(f"  {equiv_cmd}")

    print()
    if not click.confirm("Proceed?", default=True):
        print("Aborted.")
        sys.exit(0)

    # --norm is already a slug (none/zscore/log/yeojohnson/int) -- passed straight
    # to normalise_phenotype, no display-string mapping (the mapping caused the
    # en-dash no-op bug: hyphen 'Yeo-Johnson' never matched the en-dash key).

    return args


def run_pipeline(args):
    """Execute the full GWAS pipeline using pure computation functions."""
    import numpy as np
    import pandas as pd

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("cli")

    # ── Validate inputs ──────────────────────────────────
    # --norm is a slug (none/zscore/log/yeojohnson/int) passed straight to
    # normalise_phenotype -- no display-string mapping.

    # MLMM/FarmCPU require MLM as the primary model
    if "mlm" not in args.model and any(
        m in args.model for m in ("mlmm", "farmcpu")
    ):
        sys.exit("Error: mlmm and farmcpu require 'mlm' as the primary model. "
                 "Add 'mlm' to --model.")

    vcf_path = Path(args.vcf)
    pheno_path = Path(args.pheno)
    output_dir = Path(args.output)

    if not vcf_path.exists():
        sys.exit(f"Error: VCF file not found: {vcf_path}")
    if not pheno_path.exists():
        sys.exit(f"Error: Phenotype file not found: {pheno_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", output_dir)

    # ── Load data ────────────────────────────────────────
    from gwas.io import load_vcf_cached
    from gwas.qc import (
        _pipeline_parse_vcf_biallelic,
        _pipeline_harmonize_ids,
        _pipeline_phenotype_qc,
        _pipeline_snp_qc,
        _pipeline_build_geno_matrices,
        _pipeline_build_kinship,
    )

    log.info("Loading VCF: %s", vcf_path)
    vcf_bytes = vcf_path.read_bytes()

    # load_vcf_cached uses @st.cache_data decorator but works as regular function
    # when Streamlit is not running (decorator becomes a no-op passthrough)
    callset = load_vcf_cached(vcf_bytes)

    log.info("Loading phenotype: %s", pheno_path)
    sep = "\t" if pheno_path.suffix in (".tsv", ".txt") else ","
    # encoding="utf-8-sig" strips BOM if present; default cp1252 on Windows
    # would fail on UTF-8 files with non-ASCII trait names.
    pheno = pd.read_csv(pheno_path, sep=sep, index_col=0, encoding="utf-8-sig")

    if args.trait not in pheno.columns:
        sys.exit(f"Error: Trait '{args.trait}' not found in phenotype columns: {list(pheno.columns)}")

    # ── Pipeline stages ──────────────────────────────────
    log.info("Parsing VCF (biallelic filter)…")
    genotypes, samples, chroms, positions, vid, ref, alt, info_scores, info_field = \
        _pipeline_parse_vcf_biallelic(callset)
    log.info("  %d samples, %d variants", len(samples), genotypes.shape[1])

    if info_field:
        log.info("  Imputation quality field detected: %s", info_field)

    log.info("Harmonizing sample IDs…")
    geno_df, pheno, chroms, positions, sid, info_scores, allele_map = \
        _pipeline_harmonize_ids(genotypes, samples, chroms, positions,
                                vid, ref, alt, pheno, info_scores)
    log.info("  %d overlapping samples, %d SNPs", geno_df.shape[0], geno_df.shape[1])

    log.info("Phenotype QC…")
    geno_df, pheno, y = _pipeline_phenotype_qc(
        geno_df, pheno, args.trait, args.norm, args.ind_miss,
    )
    log.info("  %d samples after phenotype QC", geno_df.shape[0])

    # ── Optional user covariates: load + align to the phenotype-QC'd sample set,
    #    dropping samples that lack a covariate value BEFORE PCA/kinship so every
    #    downstream array (iid/geno/K/PCs) is built on the covar-complete set ──
    covar_df_sel = None
    if getattr(args, "covar", None):
        from gwas.covariates import (
            load_covariate_frame, select_covariate_columns, align_covariates,
        )
        _cov_cols = ([c.strip() for c in args.covar_cols.split(",")]
                     if getattr(args, "covar_cols", None) else None)
        covar_df_sel = select_covariate_columns(
            load_covariate_frame(args.covar), _cov_cols)
        _cov_M, _cov_names, _cov_ok = align_covariates(covar_df_sel, geno_df.index)
        _n_drop = int((~_cov_ok).sum())
        if _n_drop:
            log.warning("  --covar: dropping %d sample(s) missing covariate values", _n_drop)
            geno_df = geno_df.iloc[_cov_ok]
            pheno = pheno.iloc[_cov_ok]
            y = y[_cov_ok]
        if geno_df.shape[0] == 0:
            raise SystemExit("No samples remain after dropping covariate-missing rows.")
        log.info("  --covar: %d covariate(s) [%s] on %d samples",
                 covar_df_sel.shape[1], ", ".join(_cov_names), geno_df.shape[0])

    log.info("SNP QC (MAF=%.3f, miss=%.2f, MAC=%d, INFO=%.2f)…",
             args.maf, args.miss, args.mac, args.info_thresh)
    canonical = (tuple(str(i) for i in range(1, args.n_chromosomes + 1))
                 if args.n_chromosomes else None)
    geno_df, chroms, chroms_num, positions, sid, n_raw, qc_snp, info_scores = \
        _pipeline_snp_qc(geno_df, chroms, positions, sid,
                         args.maf, args.miss, args.mac, True,  # P11: ALT always excluded
                         info_scores, args.info_thresh, canonical=canonical,
                         max_het=args.max_het, het_excess_p=args.het_excess_p)
    log.info("  QC: %s", qc_snp)

    log.info("Building genotype matrices (impute=%s)…", args.impute)
    geno_dosage_raw, snp_imputation_rate, geno_imputed, iid = \
        _pipeline_build_geno_matrices(geno_df, method=args.impute,
                                      impute_k=args.impute_k, impute_l=args.impute_l)
    qc_snp["Imputation method"] = args.impute

    log.info("Building kinship matrix…")
    K, Z_for_pca, chroms_grm, positions_grm, kinship_model = \
        _pipeline_build_kinship(geno_df, geno_imputed, chroms, positions)
    log.info("  Kinship model: %s", kinship_model)

    # ── PCA ──────────────────────────────────────────────
    from gwas.kinship import _compute_pcs_full_impl
    pcs_full, pca_eigenvalues = _compute_pcs_full_impl(Z_for_pca, max_pcs=20)
    _max_avail = pcs_full.shape[1] if pcs_full is not None else 0
    n_pcs = min(args.n_pcs, _max_avail) if pcs_full is not None else 0

    from gwas.plotting import (
        compute_cumulative_positions, plot_manhattan_static, plot_qq, compute_lambda_gc,
    )

    # ── LOCO kernels ─────────────────────────────────────
    from gwas.kinship import _build_loco_kernels_impl
    K0, K_by_chr, _ = _build_loco_kernels_impl(iid, Z_for_pca, chroms_grm, K)

    if getattr(args, "no_loco", False):
        K_by_chr = {ch: K0 for ch in K_by_chr}
        log.info("  --no-loco: using global kinship for all chromosomes")

    # ── Phenotype reader ─────────────────────────────────
    from gwas.utils import PhenoData, CovarData
    pheno_reader = PhenoData(iid, y)

    # Aligned user-covariate matrix (in iid order); None unless --covar was given.
    user_covar_mat = None
    user_covar_names = None
    if covar_df_sel is not None:
        from gwas.covariates import align_covariates as _align_cov
        user_covar_mat, user_covar_names, _ = _align_cov(covar_df_sel, iid[:, 0])

    def _make_covar(k):
        if user_covar_mat is None:
            # original path — byte-identical when no user covariates are supplied
            if pcs_full is None or k <= 0:
                return None
            return CovarData(iid, pcs_full[:, :k])
        _pc = pcs_full[:, :k] if (pcs_full is not None and k > 0) else None
        if _pc is None:
            return CovarData(iid, user_covar_mat, names=list(user_covar_names))
        return CovarData(iid, np.c_[_pc, user_covar_mat],
                         names=[f"PC{i + 1}" for i in range(int(k))] + list(user_covar_names))

    extra_csvs = {}  # additional CSVs to include in ZIP
    figures = {}     # figures to include in ZIP

    # ── QC report (P1-P5): report-only statistics; never removes a sample or
    #    marker and never breaks the run (wrapped) ──
    try:
        from gwas import qc_report as _qcr
        _qc_trait = (pheno[args.trait].to_numpy()
                     if args.trait in getattr(pheno, "columns", []) else y.ravel())
        _qc_rep = _qcr.compute_qc_report(
            geno_df, _qc_trait, args.trait, qc_snp=qc_snp,
            geno_dosage_raw=geno_dosage_raw, geno_imputed=geno_imputed,
            impute_method=args.impute, impute_k=args.impute_k, impute_l=args.impute_l)
        extra_csvs.update(_qcr.qc_report_dataframes(_qc_rep))
        figures.update(_qcr.qc_report_figures(_qc_rep))
        log.info("  QC report: %d het-outlier(s) |z|>3, %d relatedness pair(s), median F_IS %.3f",
                 _qc_rep["sample_het"]["n_flagged"],
                 _qc_rep["dup_pairs"]["n_pairs_flagged"],
                 _qc_rep["variant_fis"]["median_fis"])
    except Exception as _qc_err:  # pragma: no cover
        log.warning("QC report skipped: %s", _qc_err)

    if getattr(args, "export_qc", False):
        log.info("Preparing QC'd genotype export for benchmarking…")
        geno_export = pd.DataFrame(
            geno_imputed, index=list(geno_df.index), columns=sid,
        )
        geno_export.index.name = "SampleID"
        extra_csvs["QC_genotype_matrix.csv"] = geno_export.reset_index()
        # P3: also export the RAW dosage matrix (NaN preserved) -- the imputed
        # export destroys the missing-call evidence.  QC_genotype_matrix.csv is
        # unchanged; this is an additional file.
        geno_raw_export = pd.DataFrame(
            geno_dosage_raw, index=list(geno_df.index), columns=sid,
        )
        geno_raw_export.index.name = "SampleID"
        extra_csvs["QC_genotype_raw.csv"] = geno_raw_export.reset_index()
        ref_vals = [allele_map.get(s, (".", "."))[0] for s in sid]
        alt_vals = [allele_map.get(s, (".", "."))[1] for s in sid]
        extra_csvs["QC_snp_map.csv"] = pd.DataFrame({
            "SNP_ID": sid, "Chr": chroms, "Pos": positions,
            "Ref": ref_vals, "Alt": alt_vals,
        })
        extra_csvs["QC_phenotype.csv"] = pd.DataFrame({
            "SampleID": list(geno_df.index), args.trait: y.ravel(),
        })
        log.info("  QC export: %d samples x %d SNPs (impute=%s; raw missing %.3f%%)",
                 geno_imputed.shape[0], geno_imputed.shape[1], args.impute,
                 float(np.isnan(geno_dosage_raw).mean()) * 100.0)

    # Per-model PC counts (fall back to --n-pcs when not specified)
    n_pcs_mlm = min(args.n_pcs_mlm if args.n_pcs_mlm is not None else n_pcs, _max_avail)
    n_pcs_mlmm = min(args.n_pcs_mlmm if args.n_pcs_mlmm is not None else n_pcs_mlm, _max_avail)
    n_pcs_fc = min(args.n_pcs_farmcpu if args.n_pcs_farmcpu is not None else n_pcs, _max_avail)

    # ── PC-selection diagnostics (report-only; REPORTS, never selects) ──
    # The lambda-GC auto-PC selector was removed (D-FINAL): TRACE uses the fixed
    # --n-pcs default and reports the genotype-PCA eigenvalue spectrum so the user can
    # judge the choice.  No criterion, no recommended k; one eigendecomposition, no model fits.
    _pc_diag = None
    try:
        from gwas.pc_diagnostics import compute_pc_diagnostics
        _pc_diag = compute_pc_diagnostics(
            Z_for_pca, n=int(Z_for_pca.shape[0]), m=int(Z_for_pca.shape[1]),
            prune_params={"r2": 0.2, "window_bp": 500_000, "step_bp": 100_000},
            spectrum_depth=20,
        )
        extra_csvs["PC_diagnostics_spectrum.csv"] = _pc_diag["spectrum"]
        log.info("  PC diagnostics: %d eigenvalues, trace/m=%.3f "
                 "(eigenvalue spectrum -> PC_diagnostics_spectrum.csv; TRACE uses the fixed --n-pcs)",
                 _pc_diag["meta"]["n_eigenvalues"], _pc_diag["meta"]["trace_over_m"])
    except Exception as _pcd_err:  # pragma: no cover
        log.warning("  PC diagnostics skipped: %s", _pcd_err)

    log.info("  PCs: MLM=%d, MLMM=%d, FarmCPU=%d", n_pcs_mlm, n_pcs_mlmm, n_pcs_fc)

    covar_mlm = _make_covar(n_pcs_mlm)
    covar_mlmm = _make_covar(n_pcs_mlmm)
    covar_fc = _make_covar(n_pcs_fc)

    # ── Run GWAS models ──────────────────────────────────
    from gwas.models import _run_gwas_impl, add_ols_effects_to_gwas
    from statsmodels.stats.multitest import multipletests

    gwas_df = None
    extra_model_dfs = {}
    cofactor_tables = {}

    # Compute M_eff (always, for reference columns)
    try:
        from gwas.plotting import compute_meff_li_ji
        _meff_val, _ = compute_meff_li_ji(geno_imputed)
        meff_thresh = 0.05 / _meff_val
        log.info("M_eff = %d independent tests (of %d SNPs)", _meff_val, geno_df.shape[1])
    except Exception:
        _meff_val = geno_df.shape[1]
        meff_thresh = 0.05 / _meff_val
        log.warning("M_eff computation failed; using naive Bonferroni")

    bonf_thresh_naive = 0.05 / geno_df.shape[1]

    # Primary threshold based on user choice
    sig_rule = getattr(args, "sig_thresh", "bonferroni")
    _sig_is_numeric = isinstance(sig_rule, (int, float)) and not isinstance(sig_rule, bool)
    if sig_rule == "bonferroni":
        primary_thresh = bonf_thresh_naive
        log.info("Significance: Bonferroni %.2e (%d SNPs)", primary_thresh, geno_df.shape[1])
    elif sig_rule == "fdr":
        primary_thresh = None  # FDR uses per-SNP q-values
        log.info("Significance: FDR q < 0.05")
    else:  # meff (default)
        primary_thresh = meff_thresh
        log.info("Significance: M_eff %.2e (M_eff=%d)", primary_thresh, _meff_val)

    if _sig_is_numeric:
        # a user-specified fixed p-value threshold overrides the named-rule value
        primary_thresh = float(sig_rule)
        log.info("Significance: p < %.2e (user-specified)", primary_thresh)

    if "mlm" in args.model:
        log.info("Running MLM (LOCO)…")
        gwas_df = _run_gwas_impl(
            geno_imputed, y, pcs_full, n_pcs_mlm, sid, positions,
            chroms, chroms_num, iid, K0, K_by_chr, pheno_reader, args.trait,
            user_covar=user_covar_mat, user_covar_names=user_covar_names,
        )
        gwas_df["PValue"] = np.clip(gwas_df["PValue"].astype(float), 1e-300, 1.0)
        gwas_df["-log10p"] = -np.log10(gwas_df["PValue"])
        try:
            gwas_df = add_ols_effects_to_gwas(gwas_df, geno_imputed, y, covar_mlm, sid)
        except Exception as e:
            log.warning("OLS effects failed: %s", e)
        _rej, _fdr, _, _ = multipletests(gwas_df["PValue"].values, method="fdr_bh")
        gwas_df["FDR"] = _fdr
        gwas_df["Significant_FDR"] = _rej
        gwas_df["Significant_Bonf"] = gwas_df["PValue"] < bonf_thresh_naive
        gwas_df["Significant_Meff"] = gwas_df["PValue"] < meff_thresh
        if _sig_is_numeric:
            gwas_df["Significant_Custom"] = gwas_df["PValue"] < primary_thresh
        gwas_df["Model"] = "MLM"

        # Imputation rate
        _imp = pd.DataFrame({"SNP": sid.astype(str), "ImputationRate": snp_imputation_rate})
        gwas_df = gwas_df.merge(_imp, on="SNP", how="left")

        log.info("  MLM: %d significant (M_eff), %d (FDR<0.05)",
                 gwas_df["Significant_Meff"].sum(), _rej.sum())

    if "mlmm" in args.model and gwas_df is not None:
        from gwas.models import run_mlmm_research_grade_fast
        log.info("Running MLMM…")
        gwas_mlmm, cof_tbl = run_mlmm_research_grade_fast(
            geno_imputed, sid, chroms, chroms_num, positions, iid,
            pheno_reader, K0, covar_mlmm, verbose=False,
        )
        gwas_mlmm["PValue"] = np.clip(gwas_mlmm["PValue"].astype(float), 1e-300, 1.0)
        gwas_mlmm["-log10p"] = -np.log10(gwas_mlmm["PValue"])
        extra_model_dfs["MLMM"] = gwas_mlmm
        cofactor_tables["MLMM"] = cof_tbl
        log.info("  MLMM: %d cofactors selected", len(cof_tbl))

    if "farmcpu" in args.model and gwas_df is not None:
        from gwas.models import run_farmcpu
        log.info("Running FarmCPU…")
        gwas_farmcpu, pqtn_tbl, conv_info = run_farmcpu(
            geno_imputed, sid, chroms, chroms_num, positions, iid,
            pheno_reader, K0, covar_fc,
            final_scan="ols", verbose=False,  # D-103: FarmCPU MLM final scan frozen out (published OLS)
            use_loco=not getattr(args, "no_loco", False),
        )
        gwas_farmcpu["PValue"] = np.clip(gwas_farmcpu["PValue"].astype(float), 1e-300, 1.0)
        gwas_farmcpu["-log10p"] = -np.log10(gwas_farmcpu["PValue"])
        extra_model_dfs["FarmCPU"] = gwas_farmcpu
        cofactor_tables["FarmCPU"] = pqtn_tbl
        log.info("  FarmCPU: %s, %d pseudo-QTNs",
                 "converged" if conv_info["converged"] else "max iter",
                 conv_info["n_pseudo_qtns"])

    if gwas_df is None:
        log.error("No primary GWAS model ran. At least 'mlm' is required.")
        sys.exit(1)

    # ── Cross-model consensus ───────────────────────────
    # Mirrors the Streamlit consensus block (pages/GWAS_analysis.py).
    # Only emitted when at least two of {MLM, MLMM, FarmCPU} ran.
    _consensus_models = [("MLM", gwas_df)]
    for _mname in ["MLMM", "FarmCPU"]:
        if _mname in extra_model_dfs:
            _consensus_models.append((_mname, extra_model_dfs[_mname]))

    if len(_consensus_models) >= 2:
        try:
            _sig_by_model = {}
            for _mname, _mdf in _consensus_models:
                if _sig_is_numeric:
                    _sig_by_model[_mname] = set(_mdf.loc[_mdf["PValue"] < primary_thresh, "SNP"].astype(str))
                elif sig_rule == "fdr" and "FDR" in _mdf.columns:
                    _sig_by_model[_mname] = set(_mdf.loc[_mdf["FDR"] < 0.05, "SNP"].astype(str))
                elif sig_rule == "bonferroni":
                    _sig_by_model[_mname] = set(_mdf.loc[_mdf["PValue"] < bonf_thresh_naive, "SNP"].astype(str))
                else:  # meff (default)
                    _sig_by_model[_mname] = set(_mdf.loc[_mdf["PValue"] < meff_thresh, "SNP"].astype(str))

            _all_sig = sorted(set().union(*_sig_by_model.values()))
            if _all_sig:
                _rows = []
                # SNP -> best (smallest) p across models for sorting
                _best_p = {}
                for _mname, _mdf in _consensus_models:
                    _sub = _mdf[_mdf["SNP"].astype(str).isin(_all_sig)][["SNP", "PValue"]]
                    for _snp, _p in zip(_sub["SNP"].astype(str), _sub["PValue"]):
                        _best_p[_snp] = min(_best_p.get(_snp, np.inf), float(_p))
                for _snp in _all_sig:
                    _detected_by = sorted(
                        _mname for _mname, _s in _sig_by_model.items() if _snp in _s
                    )
                    _rows.append({
                        "SNP": _snp,
                        "N_models": len(_detected_by),
                        "Detected_by": ",".join(_detected_by),
                        "Best_PValue": _best_p.get(_snp, np.nan),
                    })
                _consensus_df = (
                    pd.DataFrame(_rows)
                    .sort_values(["N_models", "Best_PValue"], ascending=[False, True])
                    .reset_index(drop=True)
                )
                extra_csvs["CrossModel_Consensus.csv"] = _consensus_df
                _n_full = int((_consensus_df["N_models"] == len(_consensus_models)).sum())
                log.info("  Cross-model consensus: %d SNPs significant in %d/%d models",
                         _n_full, len(_consensus_models), len(_consensus_models))
        except Exception as e:
            log.warning("Cross-model consensus failed: %s", e)

    # ── LD decay estimation ─────────────────────────────
    import gwas.ld as ld
    ld_flank_kb = args.ld_flank_kb
    ld_decay_kb = None
    ld_blocks_mlm = None  # track MLM blocks for subsampling aggregation

    _ld_decay_n_censored = None  # LD-decay Tier 1b: grid-limited per-chr count (run_metadata)
    if ld_flank_kb is None:
        log.info("Estimating LD decay…")
        ld_distances = []
        _ld_decay_n_censored = 0
        _geno_float = geno_imputed.astype(float)
        for ch in np.unique(chroms):
            ch_mask = chroms == ch
            pos_ch = positions[ch_mask]
            geno_ch = _geno_float[:, ch_mask]
            if len(pos_ch) < 50:
                continue
            if len(pos_ch) > 1500:
                idx = np.linspace(0, len(pos_ch) - 1, 1500, dtype=int)
                pos_ch, geno_ch = pos_ch[idx], geno_ch[:, idx]
            r2_mat = ld.pairwise_r2(geno_ch)
            dk, _, _dfld = ld.ld_decay(pos_ch, r2_mat, ld_threshold=0.2, max_dist_kb=5000)
            if np.isfinite(dk):
                ld_distances.append(dk)
                if _dfld.attrs.get("ld_decay_censored"):
                    _ld_decay_n_censored += 1
        if ld_distances:
            ld_decay_kb = float(np.median(ld_distances))
            ld_flank_kb = int(2 * ld_decay_kb)
            # LD-decay Tier 1b: N grid-limited -> the median is partly a median of floors
            # (an upper bound). Display/logging only; ld_decay_kb / flank unchanged.
            _cens = (f" ({_ld_decay_n_censored} of {len(ld_distances)} chromosomes "
                     f"grid-limited <= floor)") if _ld_decay_n_censored else ""
            log.info("  LD decay ~ %.0f kb -> flank = %d kb%s", ld_decay_kb, ld_flank_kb, _cens)
        else:
            ld_flank_kb = 300
            log.warning("  Could not estimate LD decay; using 300 kb")

    # ── Per-model post-GWAS: LD blocks → haplotype → annotation ──
    from gwas.haplotype import run_haplotype_block_gwas
    from annotation import canon_chr

    post_gwas_models = [("MLM", gwas_df)]
    for mname in ["MLMM", "FarmCPU"]:
        if mname in extra_model_dfs:
            post_gwas_models.append((mname, extra_model_dfs[mname]))

    _geno_float = geno_imputed.astype(float)
    pheno_clean = pheno  # phenotype DataFrame for haplotype testing
    _heatmap_done = set()  # deduplicate LD heatmaps across models
    chroms_canon = np.array([canon_chr(str(c)) for c in chroms])

    # ── Isolated-SNP rescue setup (T-45) ──────────────────
    # Reporting rule + gene model resolved ONCE, before the per-model loop, so the
    # rescue can mine genes even for models with no LD blocks. The rescue is a
    # positional set difference against the FINAL emitted block table and NEVER
    # calls the block detector (gwas/isolated.py has no gwas.ld import).
    from gwas import isolated as _iso
    from gwas.significance import rule_from_cli_args
    _sig_rule_obj = rule_from_cli_args(args, geno_df.shape[1], _meff_val)
    _do_rescue = not getattr(args, "no_isolated_rescue", False)
    _iso_edge_flank_bp = (
        int(args.isolated_edge_flank_kb) * 1000 if getattr(args, "isolated_edge_flank_kb", None)
        else (int(ld_flank_kb) * 1000 if ld_flank_kb else 300_000)
    )
    _iso_max_interval_bp = int(float(getattr(args, "isolated_max_interval_mb", 5.0)) * 1_000_000)
    _iso_low_res_bp = (int(args.isolated_low_res_kb) * 1000
                       if getattr(args, "isolated_low_res_kb", None) else None)
    _iso_ld_decay_bp = int(ld_decay_kb * 1000) if ld_decay_kb else None
    _iso_counts = {}
    _iso_genes_df = None
    if _do_rescue and not args.no_annotation:
        try:
            from annotation import load_gene_annotation as _iso_load_genes
            _iso_data_dir = Path(__file__).resolve().parent / "data"
            _iso_build = getattr(args, "genome_build", "SL3")
            _iso_sp = {
                "tomato": {
                    "gm": (_iso_data_dir / "Sol_genes_SL3.csv" if _iso_build == "SL3"
                           else _iso_data_dir / "Sol_genes.csv"),
                    "desc": (_iso_data_dir / "SL3.1_descriptions.txt" if _iso_build == "SL3"
                             else _iso_data_dir / "ITAG4.0_annotation.txt"),
                },
            }.get(args.species, {})
            _iso_gm = Path(args.gene_model) if args.gene_model else _iso_sp.get("gm")
            _iso_desc = _iso_sp.get("desc")
            if _iso_gm and _iso_gm.exists():
                _iso_genes_df = _iso_load_genes(
                    str(_iso_gm), str(_iso_desc) if _iso_desc and _iso_desc.exists() else None)
        except Exception as e:
            log.warning("Isolated rescue: gene model load failed (%s); intervals will lack genes.", e)

    # ── Post-GWAS frames captured for the HTML report (T-80) ─────────────
    # The report previously omitted every post-GWAS table (defect 0.2). Capture
    # the primary model's frames (MLM if run, else the first model) plus each
    # model's annotation/haplotype. Additive: when a frame is absent the report
    # renders exactly as before (every report section is {% if %}-guarded).
    _primary_model = "MLM" if any(m == "MLM" for m, _ in post_gwas_models) else (
        post_gwas_models[0][0] if post_gwas_models else None)
    _report_sig_table = _report_unblocked = _report_isolated = None
    _report_ld_annotated = _report_hap_gwas = None
    _per_model_post = {}

    # P14: LD-block seed mode (mirrors the GUI radio). "suggestive" (default) =
    # the --ld-seed-p threshold + the --ld-top-n floor -> byte-identical to the
    # prior CLI behaviour; "significant" = seeds restricted to the genome-wide
    # --sig-thresh, with NO top-N floor. FDR (no single p-threshold) falls back to
    # naive Bonferroni for the seed threshold, matching the GUI.
    _ld_gw_thresh = primary_thresh if primary_thresh is not None else bonf_thresh_naive
    if args.ld_seed_mode == "significant":
        _ld_seed_thresh, _ld_seed_top_n = _ld_gw_thresh, 0
    else:
        _ld_seed_thresh, _ld_seed_top_n = args.ld_seed_p, args.ld_top_n

    for model_name, model_df in post_gwas_models:
        log.info("Post-GWAS: %s", model_name)
        m_hap_gwas = None
        m_ld_annotated = None
        m_ld_blocks = pd.DataFrame()   # always defined; empty => no blocks (rescue still runs)

        # LD block detection
        # P15: report both seed counts so the suggestive vs significant distinction
        # is visible regardless of --ld-seed-mode (the top-N floor can also form
        # non-significant blocks -- e.g. pepper forms 3 FarmCPU blocks this way).
        _n_sugg = int((model_df["PValue"] < args.ld_seed_p).sum())
        _n_sig = int((model_df["PValue"] < _ld_gw_thresh).sum())
        log.info("  %d seed SNPs at p < %.1e, of which %d pass the genome-wide "
                 "threshold (%.2e)", _n_sugg, args.ld_seed_p, _n_sig, _ld_gw_thresh)

        has_seeds = (model_df["PValue"] < _ld_seed_thresh).any()
        if not has_seeds:
            # Also try top-N seeding
            if _ld_seed_top_n > 0:
                log.info("  No seed SNPs at p < %.1e; using top-%d seeding.",
                         _ld_seed_thresh, _ld_seed_top_n)
            else:
                log.info("  No seed SNPs (p < %.1e) and no top-N seeding; no LD blocks "
                         "(isolated-SNP rescue still runs).", _ld_seed_thresh)

        try:
            m_ld_blocks = ld.find_ld_clusters_genomewide(
                gwas_df=model_df, chroms=chroms, positions=positions,
                geno_imputed=_geno_float, sid=sid,
                ld_threshold=args.ld_r2, flank_kb=ld_flank_kb,
                ld_decay_kb=ld_decay_kb, min_snps=2,
                top_n=_ld_seed_top_n, sig_thresh=_ld_seed_thresh,
                ld_merge_r2=args.ld_merge_r2,
            )
            m_ld_blocks, _ = ld.filter_contained_blocks(m_ld_blocks, min_contained=2)
        except Exception as e:
            log.warning("  LD block detection failed for %s: %s", model_name, e)
            m_ld_blocks = pd.DataFrame()

        log.info("  %d LD blocks detected", len(m_ld_blocks))

        if model_name == "MLM":
            ld_blocks_mlm = m_ld_blocks

        # ── Isolated-SNP rescue (T-45): runs for EVERY model, with or without
        #    blocks. Set difference vs the FINAL block table; the block detector
        #    is untouched. Emits parallel CSVs; never writes into m_ld_blocks. ──
        if _do_rescue:
            try:
                _rescue = _iso.run_isolated_snp_rescue(
                    model_df, m_ld_blocks, _sig_rule_obj, chroms, positions, sid,
                    genes=_iso_genes_df, seed_p_used=args.ld_seed_p, top_n_used=args.ld_top_n,
                    edge_flank_bp=_iso_edge_flank_bp, max_interval_bp=_iso_max_interval_bp,
                    low_res_bp=_iso_low_res_bp, ld_decay_bp=_iso_ld_decay_bp,
                )
                if _rescue.n_uncovered > 0:
                    extra_csvs[f"Isolated_SNP_intervals_{model_name}.csv"] = _rescue.intervals
                    if _rescue.genes_long is not None and not _rescue.genes_long.empty:
                        extra_csvs[f"Isolated_SNP_candidate_genes_{model_name}.csv"] = _rescue.genes_long
                    if model_name == _primary_model:
                        _report_isolated = _rescue.intervals
                _iso_counts[model_name] = {
                    "n_uncovered": _rescue.n_uncovered, "n_intervals": _rescue.n_intervals,
                    "n_seeding_path": _rescue.n_seeding_path, "n_block_path": _rescue.n_block_path,
                }
                log.info("  Isolated-SNP rescue: %d significant SNP(s) with no LD block "
                         "(%d seeding-threshold, %d block-formation) -> %d interval(s)",
                         _rescue.n_uncovered, _rescue.n_seeding_path,
                         _rescue.n_block_path, _rescue.n_intervals)
            except Exception as e:
                log.warning("  Isolated-SNP rescue failed for %s: %s", model_name, e)

        # ── Significant-SNP table (T-20): every reporting-significant SNP, marked
        #    in_block/unblocked with candidate interval + gene evidence. Same escape
        #    hatch as the rescue so --no-isolated-rescue stays byte-clean. Runs for
        #    every model (block or not) before the empty-block skip below. ──
        if _do_rescue:
            try:
                from gwas.sigtable import build_significant_snp_table, project_unblocked
                _sigtab = build_significant_snp_table(
                    model_df, m_ld_blocks, _sig_rule_obj, chroms, positions, sid,
                    geno_dosage_raw=geno_dosage_raw, genes=_iso_genes_df,
                    seed_p_used=args.ld_seed_p, top_n_used=args.ld_top_n,
                    edge_flank_bp=_iso_edge_flank_bp, max_interval_bp=_iso_max_interval_bp,
                    low_res_bp=_iso_low_res_bp, ld_decay_bp=_iso_ld_decay_bp,
                    genome_build=getattr(args, "genome_build", "SL3"), species=args.species,
                )
                if not _sigtab.empty:
                    _unb = project_unblocked(_sigtab)
                    extra_csvs[f"Significant_SNPs_{model_name}.csv"] = _sigtab
                    extra_csvs[f"Unblocked_SNPs_{model_name}.csv"] = _unb
                    if model_name == _primary_model:
                        _report_sig_table, _report_unblocked = _sigtab, _unb
                    log.info("  Significant-SNP table: %d rows (%d unblocked) for %s",
                             len(_sigtab), int((_sigtab["Block_Status"] != "in_block").sum()), model_name)
            except Exception as e:
                log.warning("  Significant-SNP table failed for %s: %s", model_name, e)

        if m_ld_blocks.empty:
            continue

        # LD heatmaps for blocks with significant lead SNPs
        if not args.no_plots:
            try:
                import seaborn as sns
                import matplotlib.pyplot as plt
                _sig_col_name = (
                    "Significant_Custom" if _sig_is_numeric
                    else "Significant_Meff" if sig_rule == "meff"
                    else ("Significant_FDR" if sig_rule == "fdr" else "Significant_Bonf")
                )
                _sig_snps = set()
                if _sig_col_name in model_df.columns:
                    _sig_snps = set(model_df.loc[model_df[_sig_col_name], "SNP"].astype(str))
                _n_hm = 0
                for _, _brow in m_ld_blocks.iterrows():
                    _lead = str(_brow.get("lead_snp", "")).strip()
                    if _lead not in _sig_snps:
                        continue
                    _bchr = canon_chr(str(_brow["Chr"]))
                    _bstart = int(_brow.get("Start (bp)", _brow.get("Start", 0)))
                    _bend = int(_brow.get("End (bp)", _brow.get("End", 0)))
                    _block_key = (_bchr, _bstart, _bend)
                    if _block_key in _heatmap_done:
                        continue
                    _heatmap_done.add(_block_key)
                    _in_block = (chroms_canon == _bchr) & (positions >= _bstart) & (positions <= _bend)
                    _block_idx = np.where(_in_block)[0]
                    if len(_block_idx) < 2:
                        continue
                    _block_r2 = ld.pairwise_r2(_geno_float[:, _block_idx])
                    _labels = [f"{_bchr}_{positions[i]}" for i in _block_idx]
                    _fig_hm, _ax_hm = plt.subplots(
                        figsize=(max(4, len(_block_idx) * 0.3 + 1),
                                 max(3, len(_block_idx) * 0.25 + 1))
                    )
                    _mask = np.triu(np.ones_like(_block_r2, dtype=bool))
                    sns.heatmap(
                        _block_r2, mask=_mask, cmap="YlOrBr", vmin=0, vmax=1,
                        square=True, linewidths=0.5,
                        xticklabels=_labels, yticklabels=_labels,
                        cbar_kws={"label": "r^2"}, ax=_ax_hm,
                    )
                    _ax_hm.set_title(f"LD heatmap: Chr{_bchr} {_bstart:,}-{_bend:,}", fontsize=10)
                    _ax_hm.tick_params(labelsize=6)
                    _fig_hm.tight_layout()
                    figures[f"LD_heatmap_Chr{_bchr}_{_bstart}_{_bend}.png"] = _fig_hm
                    _n_hm += 1
                if _n_hm:
                    log.info("  %d LD heatmaps generated for %s", _n_hm, model_name)
            except Exception as e:
                log.warning("  LD heatmap generation failed for %s: %s", model_name, e)

        # Haplotype testing
        try:
            _hap_pcs = pcs_full[:, :n_pcs_mlm] if pcs_full is not None and n_pcs_mlm > 0 else None
            m_hap_gwas, _ = run_haplotype_block_gwas(
                haplo_df=m_ld_blocks, chroms=chroms, positions=positions,
                geno_imputed=_geno_float, sid=sid,
                geno_df=geno_df, pheno_df=pheno_clean, trait_col=args.trait,
                pcs=_hap_pcs, n_perm=args.hap_perms, n_pcs_used=n_pcs_mlm,
                min_hap_count=args.hap_min_count, min_group_size=args.hap_min_group_size,
                user_covar=user_covar_mat,
            )
            if m_hap_gwas is not None and not m_hap_gwas.empty:
                n_sig_hap = int((m_hap_gwas.get("FDR_BH", pd.Series(dtype=float)) < 0.05).sum())
                log.info("  Haplotype: %d/%d blocks significant", n_sig_hap, len(m_hap_gwas))
        except Exception as e:
            log.warning("  Haplotype testing failed for %s: %s", model_name, e)

        # Gene annotation
        if not args.no_annotation:
            try:
                from annotation import (
                    load_gene_annotation, annotate_ld_blocks,
                )

                # Species file mapping
                _data_dir = Path(__file__).resolve().parent / "data"
                _build = getattr(args, "genome_build", "SL3")
                _sp_files = {
                    "tomato": {
                        "gene_model": (
                            _data_dir / "Sol_genes_SL3.csv" if _build == "SL3"
                            else _data_dir / "Sol_genes.csv"
                        ),
                        "gene_desc": (
                            _data_dir / "SL3.1_descriptions.txt" if _build == "SL3"
                            else _data_dir / "ITAG4.0_annotation.txt"
                        ),
                    },
                }.get(args.species, {})

                gm_path = Path(args.gene_model) if args.gene_model else _sp_files.get("gene_model")
                desc_path = _sp_files.get("gene_desc")

                if gm_path and gm_path.exists():
                    genes_df = load_gene_annotation(
                        str(gm_path),
                        str(desc_path) if desc_path and desc_path.exists() else None,
                    )

                    m_ld_annotated = annotate_ld_blocks(
                        m_ld_blocks, genes_df, n_flank=2, max_flank_dist_bp=500_000,
                    )
                else:
                    log.info("  Gene model not found for species '%s'; skipping annotation.", args.species)
            except Exception as e:
                log.warning("  Annotation failed for %s: %s", model_name, e)

        # Consolidate LD blocks + annotation + haplotype into one table
        from annotation import consolidate_ld_block_table
        try:
            m_consolidated = consolidate_ld_block_table(m_ld_blocks, m_hap_gwas, m_ld_annotated)
            if m_consolidated is not None and not m_consolidated.empty:
                extra_csvs[f"LD_blocks_annotated_{model_name}.csv"] = m_consolidated
        except Exception as e:
            log.warning("  LD block consolidation failed for %s: %s", model_name, e)

        # ── LD-quality triage (T-36): Layer 1 (predictive) + Layer 2 (mlg_*) ->
        #    router -> a NEW supplementary LD_triage_{model}.csv. Additive + gated
        #    by --no-triage (existing outputs byte-identical when off). Never a
        #    filter; changes no p/F/eta2/boundary. N1 safe: consolidate drops mlg_*. ──
        if not args.no_triage and m_ld_blocks is not None and not m_ld_blocks.empty:
            try:
                import numpy as _np
                from annotation import canon_chr as _cc
                from gwas.ld import compute_block_ld_quality as _cblq, maf_from_matrix as _maf
                from gwas.triage import (TriageThresholds as _TT, triage_blocks as _tb,
                                         add_eta2_comparability as _eta)
                _ldq = _cblq(m_ld_blocks, chroms, positions, sid, _geno_float, model_df,
                             geno_dosage_raw=geno_dosage_raw, r2_coherent=args.ld_r2)
                _m = _ldq.copy()
                _m["_c"] = _m["Chr"].astype(str).map(_cc)
                _m["_s"] = _m["Start (bp)"].astype(int); _m["_e"] = _m["End (bp)"].astype(int)
                if m_hap_gwas is not None and not m_hap_gwas.empty:
                    _h = m_hap_gwas.rename(columns={"Start": "Start (bp)", "End": "End (bp)"}).copy()
                    _h["_c"] = _h["Chr"].astype(str).map(_cc)
                    _h["_s"] = _h["Start (bp)"].astype(int); _h["_e"] = _h["End (bp)"].astype(int)
                    _l2 = ["_c", "_s", "_e"] + [c for c in _h.columns if c.startswith("mlg_") or c in (
                        "eta2", "df1", "df2", "F_perm", "F_param", "n_samples_tested",
                        "n_samples_block", "n_tested_haplotypes")]
                    _m = _m.merge(_h[_l2], on=["_c", "_s", "_e"], how="left")
                _m = _m.drop(columns=["_c", "_s", "_e"])
                _m = _eta(_m)
                # per-row lead usability (n_lead_classes_ge, lead_maf) from raw dosage
                _sida = _np.asarray(sid).astype(str)
                _G = _np.asarray(geno_dosage_raw, float) if geno_dosage_raw is not None else None
                _nc, _mf = [], []
                for _lead in _m["ldq_lead_snp"].astype(str):
                    _c1, _m1 = _np.nan, _np.nan
                    if _G is not None and _lead:
                        _ix = _np.where(_sida == _lead)[0]
                        if len(_ix):
                            _col = _G[:, int(_ix[0])]
                            _gg = _np.rint(_col[_np.isfinite(_col)])
                            _c1 = int(sum(int((_gg == _k).sum()) >= args.hap_min_group_size for _k in (0, 1, 2)))
                            _m1 = float(_maf(_G[:, [int(_ix[0])]], "dosage012")[0])
                    _nc.append(_c1); _mf.append(_m1)
                _m["n_lead_classes_ge"] = _nc
                _m["lead_maf"] = _mf
                _thr = _TT(
                    r2_coherent=(args.triage_r2_coherent if args.triage_r2_coherent is not None else args.ld_r2),
                    lead_r2_frac=args.triage_lead_r2_frac, min_group_n=args.hap_min_group_size, enabled=True)
                extra_csvs[f"LD_triage_{model_name}.csv"] = _tb(_m, _thr)
                log.info("  Triage: %d blocks -> LD_triage_%s.csv", len(_m), model_name)
            except Exception as e:
                log.warning("  LD-quality triage failed for %s: %s", model_name, e)

        # Capture per-model annotation/haplotype frames for the HTML report (T-80)
        _mp = {}
        if m_ld_annotated is not None and not getattr(m_ld_annotated, "empty", True):
            _mp["ld_blocks_annotated_df"] = m_ld_annotated
        if m_hap_gwas is not None and not getattr(m_hap_gwas, "empty", True):
            _mp["haplotype_gwas_df"] = m_hap_gwas
        if _mp:
            _per_model_post[model_name] = _mp
        if model_name == _primary_model:
            _report_ld_annotated = m_ld_annotated
            _report_hap_gwas = m_hap_gwas

    # ── Subsampling GWAS ───────────────────────────────────
    boot_disc_df = None
    if args.subsampling:
        from gwas.subsampling import subsample_gwas_resampling, aggregate_subsampling_to_ld_blocks
        log.info("Subsampling GWAS: %d reps, %.0f%% subsample…",
                 args.boot_reps, args.boot_frac * 100)

        _boot_use_loco = not getattr(args, "no_loco", False)
        try:
            boot_disc_df, boot_raw_pvals, boot_meta = subsample_gwas_resampling(
                geno_imputed=geno_imputed, y=y, sid=sid,
                chroms=chroms, chroms_num=chroms_num, positions=positions,
                iid=iid, Z_for_grm=Z_for_pca, pcs_full=pcs_full,
                n_pcs=n_pcs_mlm,
                n_reps=args.boot_reps, sample_frac=args.boot_frac,
                discovery_thresh=args.boot_thresh, seed=args.seed,
                n_jobs=args.boot_jobs,
                use_loco=_boot_use_loco,
                chroms_grm=chroms_grm if _boot_use_loco else None,
            )

            extra_csvs["Subsampling_SNP_stability.csv"] = boot_disc_df
            boot_meta_df = pd.DataFrame(boot_meta)
            extra_csvs["Subsampling_rep_metadata.csv"] = boot_meta_df

            n_ok = int((boot_meta_df["status"] == "ok").sum())
            n_gt50 = int((boot_disc_df["DiscoveryFreq"] > 0.5).sum())
            log.info("  %d/%d reps OK, %d SNPs with freq > 50%%", n_ok, args.boot_reps, n_gt50)

            # Aggregate to LD blocks if available
            if ld_blocks_mlm is not None and not ld_blocks_mlm.empty:
                try:
                    block_stab = aggregate_subsampling_to_ld_blocks(
                        boot_disc_df, ld_blocks_mlm, boot_raw_pvals,
                        sid, chroms, positions, discovery_thresh=args.boot_thresh,
                    )
                    if not block_stab.empty:
                        extra_csvs["Subsampling_block_stability.csv"] = block_stab
                        log.info("  Block-level stability: %d blocks", len(block_stab))
                except Exception as e:
                    log.warning("  Subsampling block aggregation failed: %s", e)

            # Subsampling discovery frequency histogram
            if not args.no_plots:
                try:
                    import matplotlib.pyplot as plt
                    _freq_vals = boot_disc_df["DiscoveryFreq"].values
                    _fig_bh, _ax_bh = plt.subplots(figsize=(6, 4))
                    _ax_bh.hist(
                        _freq_vals[_freq_vals > 0], bins=30,
                        edgecolor="white", linewidth=0.5,
                    )
                    _ax_bh.set_xlabel("Discovery frequency")
                    _ax_bh.set_ylabel("Number of SNPs")
                    _ax_bh.set_title(
                        f"Subsampling GWAS: SNP discovery frequency ({args.trait})"
                    )
                    _ax_bh.axvline(0.5, color="red", linestyle="--", label="50% threshold")
                    _ax_bh.legend()
                    plt.tight_layout()
                    figures[f"Subsampling_discovery_freq_{args.trait}.png"] = _fig_bh
                    log.info("  Subsampling histogram generated")
                except Exception as e:
                    log.warning("  Subsampling histogram failed: %s", e)
        except Exception as e:
            log.error("  Subsampling GWAS failed: %s", e)

    # ── Plots ────────────────────────────────────────────
    lambda_gc = compute_lambda_gc(gwas_df["PValue"].values)
    log.info("Lambda GC: %.3f", lambda_gc)

    if not args.no_plots:
        log.info("Generating plots…")
        df_plot = gwas_df.copy()
        df_plot, tick_pos, tick_lab = compute_cumulative_positions(df_plot)

        import matplotlib.pyplot as plt

        _thresh_labels = {"meff": f"M_eff (M={_meff_val:,})", "bonferroni": "Bonferroni", "fdr": "FDR"}
        _man_lod = -np.log10(primary_thresh) if primary_thresh is not None else None
        fig_man = plot_manhattan_static(df_plot, _man_lod,
                                        _thresh_labels.get(sig_rule, sig_rule),
                                        f"Manhattan (MLM): {args.trait}")
        plt.figure(fig_man.number)
        plt.xticks(tick_pos, tick_lab, fontsize=8)
        figures["Manhattan_MLM.png"] = fig_man

        fig_qq = plot_qq(gwas_df["PValue"].values, lambda_gc_used=lambda_gc)
        figures["QQ_MLM.png"] = fig_qq

        # Per-model Manhattan + QQ (MLMM, FarmCPU)
        for _m_name, _m_df in extra_model_dfs.items():
            try:
                _mdf_plot = _m_df.copy()
                _mdf_plot, _mtp, _mtl = compute_cumulative_positions(_mdf_plot)
                _lam_m = compute_lambda_gc(_m_df["PValue"].values)
                _fig_m = plot_manhattan_static(
                    _mdf_plot, _man_lod,
                    _thresh_labels.get(sig_rule, sig_rule),
                    f"Manhattan ({_m_name}): {args.trait}",
                )
                plt.figure(_fig_m.number)
                plt.xticks(_mtp, _mtl, fontsize=8)
                figures[f"Manhattan_{_m_name}.png"] = _fig_m

                _fig_qq_m = plot_qq(_m_df["PValue"].values, lambda_gc_used=_lam_m)
                figures[f"QQ_{_m_name}.png"] = _fig_qq_m
                log.info("  %s plots generated", _m_name)
            except Exception as e:
                log.warning("  %s plots failed: %s", _m_name, e)

        if pcs_full is not None and n_pcs_mlm >= 2:
            from gwas.plotting import plot_pca_scatter
            fig_pca = plot_pca_scatter(
                pcs_full[:, :n_pcs_mlm], y=y.ravel(),
                title=f"PCA: {args.trait}",
                eigenvalues=pca_eigenvalues,
            )
            figures["PCA_scatter.png"] = fig_pca

        if pca_eigenvalues is not None:
            # P4: scree plot (eigenvalues already computed; parity with the GUI which
            # renders plot_pca_scree). Gated on availability, not n_pcs_mlm >= 2, so it
            # emits even at k=0 (e.g. the tomato deflation-guard panel).
            from gwas.plotting import plot_pca_scree
            figures["PCA_scree.png"] = plot_pca_scree(
                pca_eigenvalues, n_pcs_used=n_pcs_mlm, title="PCA Scree Plot")

    # ── Per-SNP effect boxplots + plotting ledger (T-21/T-84–86) ──────────
    # Primary model only. The significant-SNP table stays complete — this only
    # bounds how many figures the report embeds (--snp-plots / --max-snp-plots),
    # and --collapse-r2 thins near-redundant SNPs for PLOTTING only. Emits
    # SNP_view_index; gated by --no-isolated-rescue (no _report_sig_table).
    # (Block_sample_retention is deferred to the LD-triage sub-batch, where the
    # haplotype test's frac_retained is hoisted into its output frame.)
    _snp_boxplots = []
    if (_do_rescue and _report_sig_table is not None and not _report_sig_table.empty
            and geno_dosage_raw is not None):
        try:
            from gwas.snpplots import (build_snp_view_index, render_snp_boxplot,
                                       select_snps_for_plotting)
            from gwas.snpview import collapse_snps_for_plotting, effect_flag
            _collapse = collapse_snps_for_plotting(
                _report_sig_table, geno_dosage_raw, sid, r2_threshold=args.collapse_r2)
            _rep_ids = set(_collapse.loc[
                _collapse["SNP"] == _collapse["Representative_SNP"], "SNP"].astype(str))
            _rep_table = _report_sig_table[_report_sig_table["SNP"].astype(str).isin(_rep_ids)]
            _plot_ids = select_snps_for_plotting(
                _rep_table, mode=args.snp_plots, max_plots=args.max_snp_plots)
            extra_csvs[f"SNP_view_index_{_primary_model}.csv"] = build_snp_view_index(
                _report_sig_table, _plot_ids, _collapse)
            if _plot_ids and not args.no_report:
                _sid_arr = np.asarray(sid).astype(str)
                _geno_raw = np.asarray(geno_dosage_raw, dtype=float)
                _yv = np.asarray(y, dtype=float).ravel()
                for _pid in _plot_ids:
                    _mi = np.where(_sid_arr == str(_pid))[0]
                    _rowsel = _report_sig_table[_report_sig_table["SNP"].astype(str) == str(_pid)]
                    if len(_mi) == 0 or _rowsel.empty:
                        continue
                    _row = _rowsel.iloc[0]
                    _bm = _row.get("Beta_MLM"); _bo = _row.get("Beta_OLS"); _se = _row.get("SE_MLM")
                    _bm = float(_bm) if pd.notna(_bm) else None
                    _bo = float(_bo) if pd.notna(_bo) else None
                    _se = float(_se) if pd.notna(_se) else None
                    _flag = effect_flag(_bm if _bm is not None else np.nan,
                                        _bo if _bo is not None else np.nan)
                    _cap = f"{_pid} · {_row.get('Block_Status', '')} · {_flag}"
                    _fig = render_snp_boxplot(str(_pid), _geno_raw[:, int(_mi[0])], _yv,
                                              beta_mlm=_bm, se_mlm=_se, beta_ols=_bo)
                    _snp_boxplots.append((str(_pid), _fig, _cap))
        except Exception as e:
            log.warning("Per-SNP boxplots/ledger failed: %s", e)

    # ── Build HTML report (in-memory) ────────────────────
    # ── Run provenance (built ALWAYS; feeds the HTML report AND run_metadata.json) ──
    _gb_meta = getattr(args, "genome_build", "SL3")
    if args.gene_model:
        _gm_meta = str(args.gene_model)
    elif args.species == "tomato":
        _gm_meta = "Sol_genes_SL3.csv" if _gb_meta == "SL3" else "Sol_genes.csv"
    else:
        _gm_meta = "none"

    # Addition B: block-detection provenance. Effective detection seed/top-N (P14) +
    # the find_ld_clusters_genomewide defaults for the params the call inherits.
    import inspect as _md_ins
    from gwas import ld as _md_ld
    _md_lddef = {k: p.default for k, p in
                 _md_ins.signature(_md_ld.find_ld_clusters_genomewide).parameters.items()}
    meta = {
        "VCF": str(vcf_path),
        "Phenotype": str(pheno_path),
        "Trait": args.trait,
        "Models": args.model,
        "Species": args.species,
        "Genome build": _gb_meta,
        "Gene model": _gm_meta,
        "Covariate file (--covar)": str(args.covar) if getattr(args, "covar", None) else None,
        "Covariate columns": ", ".join(_cov_names) if getattr(args, "covar", None) else None,
        "Covariate samples dropped": _n_drop if getattr(args, "covar", None) else 0,
        "Covariate analysis N": int(geno_df.shape[0]) if getattr(args, "covar", None) else "N/A",
        "MAF threshold": args.maf,
        "MAC threshold": args.mac,
        "Missingness threshold": args.miss,
        "Info threshold": args.info_thresh,
        "Imputation method": (args.impute if args.impute == "mean"
                              else f"ldknni (k={args.impute_k}, l={args.impute_l})"),
        "Normalization": args.norm,
        "Significance": sig_rule,
        "PCs_MLM": n_pcs_mlm,
        "PCs_MLMM": n_pcs_mlmm if "mlmm" in args.model else "N/A",
        "PCs_FarmCPU": n_pcs_fc if "farmcpu" in args.model else "N/A",
        "Samples": int(geno_df.shape[0]),
        "SNPs (post-QC)": int(geno_df.shape[1]),
        "SNPs (raw)": int(n_raw),
        "Isolated SNPs (no LD block)": (
            "; ".join(
                f"{m}: {c['n_uncovered']} ({c['n_seeding_path']} seeding-threshold, "
                f"{c['n_block_path']} block-formation) -> {c['n_intervals']} interval(s)"
                for m, c in _iso_counts.items()
            ) if _iso_counts else ("off" if not _do_rescue else "0")
        ),
        "Lambda GC": round(lambda_gc, 4),
        "Kinship model": kinship_model,
        "LD decay (kb)": round(ld_decay_kb, 1) if ld_decay_kb else "N/A",
        "LD decay grid-limited (n_chr)": _ld_decay_n_censored,
        "LD flank (kb)": ld_flank_kb if ld_flank_kb else "N/A",
        "LD_r2 (--ld-r2)": args.ld_r2,
        "LD_seed_p (effective)": _ld_seed_thresh,
        "LD_top_n (effective)": _ld_seed_top_n,
        "LD_merge_r2 (--ld-merge-r2)": args.ld_merge_r2,
        "LD_merge_iou": _md_lddef.get("merge_iou"),
        "LD_adj_r2_min": _md_lddef.get("adj_r2_min"),
        "LD_gap_factor": _md_lddef.get("gap_factor"),
        "LD_min_snps": 2,
        "Hap_min_group_size (--hap-min-group-size)": args.hap_min_group_size,
        "Hap_n_perm": args.hap_perms,
        "LD blocks (MLM)": len(ld_blocks_mlm) if ld_blocks_mlm is not None else "N/A",
        "Subsampling reps": args.boot_reps if args.subsampling else "N/A",
        "PCs (fixed)": int(n_pcs),
        "PC diagnostics": (
            f"trace/m={_pc_diag['meta']['trace_over_m']}; see PC_diagnostics_spectrum.csv"
            if _pc_diag is not None else "n/a"),
    }

    report_html = None
    if not args.no_report:
        from gwas.reports import generate_gwas_report

        report_html = generate_gwas_report(
            trait_col=args.trait,
            qc_snp=qc_snp,
            gwas_df=gwas_df,
            figures=figures if not args.no_plots else None,
            metadata=meta,
            mlmm_df=cofactor_tables.get("MLMM"),
            farmcpu_df=cofactor_tables.get("FarmCPU"),
            ld_blocks_df=ld_blocks_mlm,
            ld_blocks_annotated_df=_report_ld_annotated,
            haplotype_gwas_df=_report_hap_gwas,
            per_model_post_gwas=_per_model_post or None,
            significant_snps_df=_report_sig_table,
            unblocked_snps_df=_report_unblocked,
            isolated_intervals_df=_report_isolated,
            snp_boxplots=_snp_boxplots or None,
            sig_label=_sig_rule_obj.label,
            n_significant_override=(
                len(_report_sig_table) if _report_sig_table is not None else None),
            pc_selection_df=(_pc_diag["spectrum"] if _pc_diag is not None else None),
            lambda_gc=lambda_gc,
            n_samples=int(geno_df.shape[0]),
            n_snps=int(geno_df.shape[1]),
            info_field=info_field,
        )

    # ── ZIP archive (single output file) ──────────────────
    log.info("Saving results to %s", output_dir)
    from gwas.plotting import _build_gwas_results_zip

    zip_name, zip_buf = _build_gwas_results_zip(
        trait_col=args.trait,
        gwas_df=gwas_df,
        figures_dict=figures if not args.no_plots else None,
        pheno_label=pheno_path.stem,
        extra_model_dfs=extra_model_dfs or None,
        extra_tables=extra_csvs or None,
        report_html=report_html,
        metadata=meta,
    )
    zip_path = output_dir / zip_name
    zip_path.write_bytes(zip_buf.getvalue())
    log.info("ZIP archive saved: %s", zip_path)

    # Free figure memory
    if not args.no_plots:
        import matplotlib.pyplot as plt
        for fig in figures.values():
            if hasattr(fig, "savefig"):
                plt.close(fig)

    log.info("GWAS complete! All results in: %s", zip_path)

    return {
        "gwas_df": gwas_df,
        "geno_imputed": geno_imputed,
        "y": y,
        "meff_val": _meff_val,
        "isolated_counts": _iso_counts,
    }


def main():
    parser = _build_parser()

    # Interactive wizard mode: bypass normal argument parsing
    if "--interactive" in sys.argv:
        args = _interactive_wizard(parser)
    else:
        args = parser.parse_args()

    # Validate required args
    for req in ("vcf", "pheno", "trait", "output"):
        if getattr(args, req) is None:
            parser.error(f"--{req} is required")

    run_pipeline(args)


if __name__ == "__main__":
    main()
