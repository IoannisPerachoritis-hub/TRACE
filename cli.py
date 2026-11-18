"""
Command-line interface for TRACE (Trait Resolution and Candidate Evaluation).

Runs the full GWAS pipeline headlessly — suitable for HPC clusters,
batch processing, and reproducible scripted analyses.

Usage:
    trace-gwas --vcf input.vcf.gz --pheno pheno.csv --trait Yield --output results/
"""
import argparse
import logging
import os
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
            "  python cli.py --vcf data.vcf.gz --pheno pheno.csv --trait Yield Height --output results/\n"
        ),
    )

    # ── GWAS required arguments ─────────────────────────
    parser.add_argument("--vcf", help="Path to VCF file (.vcf or .vcf.gz)")
    parser.add_argument("--pheno", help="Path to phenotype CSV/TSV")
    parser.add_argument("--trait", nargs="+",
                        help="Trait column name(s) in phenotype file. Several names run one "
                             "after another, each on its own phenotyped samples: one ZIP per "
                             "trait plus a combined ZIP")
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
    # Heterozygosity screen (one-sided heterozygote EXCESS; NOT two-sided HWE).
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
                           help="RNG seed for the subsampling draws (default: 42). Haplotype "
                                "permutation p-values do not use it: each block's permutations "
                                "are seeded per block from its position and the trait, so they "
                                "are reproducible whatever the seed.")

    # LD + Haplotype + Annotation
    ld_grp = parser.add_argument_group("LD & post-GWAS")
    ld_grp.add_argument("--ld-r2", type=float, default=0.6,
                         help="LD r^2 threshold for block detection (default: 0.6)")
    ld_grp.add_argument("--ld-flank-kb", type=int, default=None,
                         help="LD flank window in kb, also MLMM's cofactor window "
                              "(default: auto, 2 x the LD decay, or 300 kb)")
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
    ld_grp.add_argument("--hap-test", default="kinship", choices=["kinship", "pcs"],
                         help="Haplotype block test. 'kinship' (default): a mixed model with the "
                              "scan's own kinship kernel for the block's chromosome (LOCO, or "
                              "the global kernel with --no-loco), its null model fitted by REML; "
                              "the F test and the Freedman-Lane permutations are made on the "
                              "whitened data. 'pcs': the test with the PCs and covariates alone "
                              "(no kinship). The old test from version 1.1.0. It ignores "
                              "relatedness and can give false positives in related panels. Use it "
                              "only to reproduce old results.")
    ld_grp.add_argument("--no-annotation", action="store_true",
                         help="Skip gene annotation")
    ld_grp.add_argument("--genome-build", default="SL3", choices=["SL3", "SL4"],
                         help="Tomato gene-model assembly: SL3 = SL3.1 (default), SL4 = ITAG4.0. "
                              "Note: Varitome SNPs are in SL2.5, so gene coordinates in either build are "
                              "offset from the SNP positions (SL3 by ~0.5 Mb, SL4 more) and annotation is "
                              "positional, not coordinate-exact.")
    ld_grp.add_argument("--species", default="tomato", choices=["tomato", "custom"],
                         help="Species for annotation files (default: tomato)")
    ld_grp.add_argument("--gene-model",
                         help="Gene file for annotation (required if --species custom): one row per "
                              "gene with chromosome, start, end and gene ID columns, or a GFF3 "
                              "(plain or gzip); see docs/gene_model_upload.md")
    ld_grp.add_argument("--gene-desc",
                         help="Optional gene descriptions: one gene per line, the gene ID, a TAB, then "
                              "the description; see docs/gene_model_upload.md")
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
    # ── Isolated-SNP rescue ──
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
    # --drop-alt was removed. Unplaced/scaffold (ALT) markers carry no valid genomic
    # position, so no position-dependent step can use them and they must never enter
    # the significance divisor; TRACE now always excludes them (no flag).
    parser.add_argument("--n-chromosomes", type=int, default=None,
                         help="Number of chromosomes (default: auto-detect from VCF). "
                              "When set, only chromosomes 1..N are kept.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--interactive", action="store_true",
                         help="Interactive wizard: prompts for all options step by step")

    # Large datasets: where the genotype table is kept
    big_grp = parser.add_argument_group("Large datasets")
    big_grp.add_argument("--geno-store", default="auto", choices=["auto", "memory", "disk"],
                         help="Where the genotype table is kept. 'auto' (default): in memory when "
                              "an in-memory run fits in the free memory, otherwise on disk; the log "
                              "says which and why. 'memory': in memory (the run stops before "
                              "reading if it will not fit). 'disk': the VCF is read in pieces into "
                              "a file of one byte per genotype call, and every step reads the table "
                              "from that file in pieces. Results are the same either way "
                              "(docs/large_datasets.md).")
    big_grp.add_argument("--geno-store-dir", default=None,
                         help="Folder in which a run with the table on disk makes its own new "
                              "subfolder for the table file, removed when the run ends "
                              "(default: the --output folder).")

    return parser


def _new_geno_store_dir(args, output_dir):
    """A new, empty folder for this run's genotype table: inside --geno-store-dir (created if
    missing), else inside the output folder. The exclusive create of a uniquely named folder
    proves that the folder can be written and keeps runs that share one folder apart."""
    from gwas import genostore

    base = Path(args.geno_store_dir) if getattr(args, "geno_store_dir", None) else output_dir
    try:
        return genostore.new_store_dir(base)
    except OSError as exc:
        sys.exit(f"Error: cannot create a folder for the genotype table in {base} "
                 f"({exc.strerror or exc}). Choose a folder you can write to with --geno-store-dir.")


def _finish_geno_store(parsed, store_dir, log):
    """Close and delete this run's genotype table (after the run or an early stop). With
    TRACE_STORE_READLOG=<file> (a diagnostic) the store's read log is written there first."""
    import json

    from gwas import genostore

    _out = os.environ.get("TRACE_STORE_READLOG", "")
    if _out and parsed is not None:
        _rl = genostore.read_log()
        _rl["ingest"] = dict(getattr(parsed, "ingest", {}))
        _rl["shape"] = [int(parsed.store.N), int(parsed.store.M)]
        Path(_out).write_text(json.dumps(_rl, indent=1, default=str))
    genostore.remove_store(store_dir, parsed)
    log.info("  genotype table on disk removed: %s", store_dir)


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
    parts.append("--trait " + " ".join(_trait_list(args.trait)))
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
    print(f"  Trait:      {', '.join(_trait_list(args.trait))}")
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


def _ensure_writable_output_dir(output_dir):
    """Create ``output_dir`` and prove it is writable before any analysis runs.

    ``mkdir(exist_ok=True)`` on an existing folder never tests writability, and the
    results ZIP is the only write into ``--output`` -- so without this probe an
    unwritable mount (typically a Docker bind mount owned by a different uid) failed
    with a PermissionError traceback only after the whole analysis had run.

    The probe is a single exclusive create of a uniquely named file (pid + random
    hex), so concurrent runs sharing one output folder never touch each other's
    probe. It is deliberately not ``tempfile.mkstemp``: on Windows, mkstemp takes a
    PermissionError for a name clash and retries up to TMP_MAX (2**31 - 1) names, so
    a folder that denies file creation would spin for hours instead of failing here.
    Only the create decides writability: a folder that accepts new files but refuses
    deletes can still receive the ZIP, so a probe that cannot be removed is logged
    and the run continues.
    """
    import uuid

    probe = output_dir / f".trace_write_test_{os.getpid()}_{uuid.uuid4().hex}"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as exc:
        _uid = getattr(os, "getuid", lambda: None)()
        sys.exit(
            f"Error: cannot write to output directory {output_dir} "
            f"({exc.strerror or exc}). "
            + (f"TRACE is running as uid {_uid}. " if _uid is not None else "")
            + 'If you run TRACE in Docker, add --user "$(id -u):$(id -g)" and mount a '
            "writable folder at /data (README, Getting started)."
        )
    try:
        os.close(fd)
        os.remove(probe)
    except OSError as exc:
        logging.getLogger("cli").warning(
            "Could not remove the write-test file %s (%s); the run continues. "
            "The file is empty and safe to delete.", probe, exc.strerror or exc,
        )


def _trait_list(trait):
    """``--trait`` as a list of names (nargs="+"). A plain string, as the
    interactive wizard and programmatic callers pass it, is one trait."""
    if trait is None:
        return []
    return [trait] if isinstance(trait, str) else [str(t) for t in trait]


def _run_traits(parsed, pheno, traits, args, output_dir, pheno_path, log):
    """Several traits in sequence (gwas.pipeline.run_traits).

    Each completed trait writes its own ZIP to --output, exactly as a single-trait
    run would. A trait that fails is logged and the others still run. The combined
    ZIP (gwas.plotting._combine_gwas_zips) holds every completed trait in its own
    folder, plus failed_traits.txt when a trait failed; the run then exits non-zero.
    Only each trait's ZIP bytes are kept until the combined ZIP is written.
    """
    import gc
    import traceback

    import matplotlib.pyplot as plt

    from gwas import stagelog as _stagelog   # per-stage marks (TRACE_STAGE_LOG; off by default)
    from gwas.pipeline import failed_traits_report, run_traits
    from gwas.plotting import _combine_gwas_zips

    done, failed = {}, {}

    def _announce(i, trait):
        _stagelog.mark(f"S04_trait_start:{trait}")
        log.info("-- Trait %d/%d: %s --", i, len(traits), trait)

    for trait, res, err in run_traits(parsed, pheno, traits, args, on_start=_announce):
        if res is not None:
            _stagelog.mark("S41_zip_write")
            zip_path = output_dir / res.zip_name
            try:
                zip_path.write_bytes(res.zip_bytes)
            except OSError as exc:  # e.g. a trait name the file system rejects
                log.error("Trait %s: could not write %s: %s", trait, zip_path, exc)
                err = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            else:
                log.info("ZIP archive saved: %s", zip_path)
                done[trait] = res.zip_bytes
        if trait not in done:
            failed[trait] = err
            plt.close("all")  # a trait that failed part-way may leave figures open
        del res
        gc.collect()

    extra = ({"failed_traits.txt": failed_traits_report(traits, done, failed)}
             if failed else None)
    zip_name, zip_buf = _combine_gwas_zips(done, pheno_label=pheno_path.stem,
                                           extra_members=extra)
    combined_path = output_dir / zip_name
    combined_path.write_bytes(zip_buf.getvalue())
    log.info("Combined ZIP (%d of %d traits) saved: %s", len(done), len(traits), combined_path)
    if failed:
        sys.exit(f"Error: {len(failed)} of {len(traits)} traits failed ({', '.join(failed)})"
                 + ("; the others completed" if done else "")
                 + f". Details: failed_traits.txt in {combined_path}")
    log.info("GWAS complete! All results in: %s", output_dir)
    return {"traits": list(done), "combined_zip": str(combined_path)}


def run_pipeline(args):
    """Run the GWAS pipeline for the trait(s) in ``args.trait``.

    One trait writes one results ZIP (unchanged). Several traits run one after
    another through the same per-trait code (gwas.pipeline.run_trait_pipeline), each
    on its own phenotyped samples, sharing only the parsed VCF and the gene models.
    """
    import pandas as pd

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("cli")
    from gwas import stagelog as _stagelog   # per-stage marks (TRACE_STAGE_LOG; off by default)
    _stagelog.mark("S00_startup")

    # ── Validate inputs ──────────────────────────────────
    # --norm is a slug (none/zscore/log/yeojohnson/int) passed straight to
    # normalise_phenotype -- no display-string mapping.

    # MLMM/FarmCPU require MLM as the primary model
    if "mlm" not in args.model and any(
        m in args.model for m in ("mlmm", "farmcpu")
    ):
        sys.exit("Error: mlmm and farmcpu require 'mlm' as the primary model. "
                 "Add 'mlm' to --model.")

    traits = _trait_list(args.trait)
    if not traits:
        sys.exit("Error: --trait is required")
    _dups = sorted({t for t in traits if traits.count(t) > 1})
    if _dups:
        sys.exit(f"Error: trait(s) listed more than once in --trait: {', '.join(_dups)}")
    if len({t.casefold() for t in traits}) < len(traits):
        sys.exit("Error: --trait names that differ only in letter case would overwrite each "
                 "other's ZIP on Windows and macOS; rename one of the columns.")

    vcf_path = Path(args.vcf)
    pheno_path = Path(args.pheno)
    output_dir = Path(args.output)

    if not vcf_path.exists():
        sys.exit(f"Error: VCF file not found: {vcf_path}")
    if not pheno_path.exists():
        sys.exit(f"Error: Phenotype file not found: {pheno_path}")

    _ensure_writable_output_dir(output_dir)
    log.info("Output directory: %s", output_dir)

    # ── Where the genotype table is kept: decided before anything is read ──
    from gwas import storeplan as _sp

    _store_base = Path(args.geno_store_dir) if getattr(args, "geno_store_dir", None) else output_dir
    _plan = _sp.plan(vcf_path, choice=getattr(args, "geno_store", "auto"), store_base=_store_base,
                     models=args.model, subsampling=getattr(args, "subsampling", False),
                     boot_jobs=getattr(args, "boot_jobs", 1), export_qc=getattr(args, "export_qc", False),
                     impute=getattr(args, "impute", "mean"))
    log.info(_plan.message)
    if _plan.stop:
        sys.exit("Error: " + _plan.stop)
    for _old_dir, _old_bytes in _sp.stale_store_folders(_store_base) if _store_base.exists() else []:
        log.warning("A genotype table folder of an earlier run that did not finish is still here: "
                    "%s (%.2f GB). Delete it to free the space.", _old_dir, _old_bytes / 1e9)

    # ── Load data ────────────────────────────────────────
    from gwas.io import load_vcf_cached

    _stagelog.mark("S01_vcf_read")
    log.info("Loading VCF: %s", vcf_path)
    _on_disk = _plan.mode == "disk"
    parsed = None
    _store_dir = None
    _deferred_stop = None
    if _on_disk:
        # The VCF is read in pieces into a file of one byte per call (gwas/genostore.py).
        import shutil as _sh

        from gwas.genostore import DeferredParseError, ingest_vcf
        _store_dir = _new_geno_store_dir(args, output_dir)
        try:
            parsed = ingest_vcf(vcf_path, _store_dir, log=log)
        except DeferredParseError as _e:
            # a stop the in-memory path raises when it parses, after the phenotype checks below
            _sh.rmtree(_store_dir, ignore_errors=True)
            parsed, _deferred_stop = None, _e.original
        except BaseException:
            _sh.rmtree(_store_dir, ignore_errors=True)       # no partial table left behind
            raise
    else:
        vcf_bytes = vcf_path.read_bytes()

        # load_vcf_cached uses @st.cache_data decorator but works as regular function
        # when Streamlit is not running (decorator becomes a no-op passthrough)
        callset = load_vcf_cached(vcf_bytes)

    try:   # a stop before the trait runs must not leave the table on disk behind
        _stagelog.mark("S02_pheno_read")
        log.info("Loading phenotype: %s", pheno_path)
        sep = "\t" if pheno_path.suffix in (".tsv", ".txt") else ","
        # encoding="utf-8-sig" strips BOM if present; default cp1252 on Windows
        # would fail on UTF-8 files with non-ASCII trait names.
        pheno = pd.read_csv(pheno_path, sep=sep, index_col=0, encoding="utf-8-sig")

        for _t in traits:
            if _t not in pheno.columns:
                sys.exit(f"Error: Trait '{_t}' not found in phenotype columns: {list(pheno.columns)}")

        # ── Pipeline stages ──────────────────────────────────
        _stagelog.mark("S03_parse")
        log.info("Parsing VCF (biallelic filter)…")
        from gwas.pipeline import (ParsedGenotypes, TraitPipelineError, is_phenotype_qc_stop,
                                   run_trait_pipeline)
        if not _on_disk:
            parsed = ParsedGenotypes.from_callset(callset)
            del callset, vcf_bytes
        elif _deferred_stop is not None:
            raise _deferred_stop
    except BaseException:
        if _on_disk and parsed is not None:
            _finish_geno_store(parsed, _store_dir, log)
        raise
    log.info("  %d samples, %d variants", len(parsed.samples), parsed.genotypes.shape[1])
    # where the table was kept, for run_metadata.json only; from the genotypes as read, so two
    # runs of the same analysis record the same (the free memory is in the plan's log line only)
    args._geno_store_meta = _plan.metadata(len(parsed.samples), parsed.genotypes.shape[1])

    if parsed.info_field:
        log.info("  Imputation quality field detected: %s", parsed.info_field)

    try:
        if len(traits) == 1:
            try:
                _stagelog.mark(f"S04_trait_start:{traits[0]}")
                res = run_trait_pipeline(parsed, pheno, traits[0], args)
            except TraitPipelineError as exc:
                sys.exit(exc.exit_arg)
            except ValueError as exc:
                # phenotype QC refused the trait: one line, exit code 1, no traceback
                if not is_phenotype_qc_stop(exc):
                    raise
                sys.exit(f"Error: {exc}")
            _stagelog.mark("S41_zip_write")
            zip_path = output_dir / res.zip_name
            zip_path.write_bytes(res.zip_bytes)
            log.info("ZIP archive saved: %s", zip_path)
            log.info("GWAS complete! All results in: %s", zip_path)
            _stagelog.mark("S42_done")
            return res.context

        return _run_traits(parsed, pheno, traits, args, output_dir, pheno_path, log)
    finally:
        if _on_disk:
            _finish_geno_store(parsed, _store_dir, log)


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
