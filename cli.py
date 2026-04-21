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


def _build_parser():
    parser = argparse.ArgumentParser(
        description="TRACE — Trait Resolution and Candidate Evaluation (CLI)",
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

    # Normalization (short slugs for shell-friendliness)
    parser.add_argument(
        "--norm", default="none",
        choices=["none", "zscore", "log", "yeojohnson", "int"],
        help="Phenotype normalization: none, zscore, log, yeojohnson, int (default: none)",
    )

    # Model selection
    parser.add_argument(
        "--model", nargs="+", default=["mlm"],
        choices=["mlm", "mlmm", "farmcpu"],
        help="GWAS models to run (default: mlm). mlmm/farmcpu require mlm.",
    )
    parser.add_argument("--n-pcs", type=int, default=4, help="Number of PCs as covariates (default: 4)")
    parser.add_argument("--no-loco", action="store_true",
                        help="Use global kinship instead of LOCO (for benchmarking)")
    parser.add_argument("--n-pcs-mlm", type=int, default=None, help="PCs for MLM (overrides --n-pcs)")
    parser.add_argument("--n-pcs-mlmm", type=int, default=None, help="PCs for MLMM (overrides --n-pcs)")
    parser.add_argument("--n-pcs-farmcpu", type=int, default=None, help="PCs for FarmCPU (overrides --n-pcs)")
    parser.add_argument(
        "--farmcpu-final-scan", default="mlm", choices=["ols", "mlm"],
        help="FarmCPU final scan: mlm (LOCO-corrected, default) or ols (standard)",
    )

    # Significance threshold
    parser.add_argument(
        "--sig-thresh", default="meff",
        choices=["meff", "bonferroni", "fdr"],
        help="Significance threshold: meff (LD-aware, default), bonferroni, fdr (q<0.05)",
    )

    # Auto PC selection
    pc_grp = parser.add_argument_group("Auto PC selection")
    pc_grp.add_argument("--auto-pcs", action="store_true",
                         help="Auto-select PCs via lambda scan (overrides --n-pcs/--n-pcs-*)")
    pc_grp.add_argument("--pc-strategy", default="band",
                         choices=["band", "closest_to_1"],
                         help="Auto PC strategy (default: band)")
    pc_grp.add_argument("--max-pcs", type=int, default=10,
                         help="Max PCs to scan in auto mode (default: 10)")
    pc_grp.add_argument("--pc-band-lo", type=float, default=0.95,
                         help="Lower lambda_GC bound for band strategy (default: 0.95)")
    pc_grp.add_argument("--pc-band-hi", type=float, default=1.05,
                         help="Upper lambda_GC bound for band strategy (default: 1.05)")
    pc_grp.add_argument("--pc-parsimony-tol", type=float, default=0.02,
                         help="Parsimony tolerance for band fallback (default: 0.02)")

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
    ld_grp.add_argument("--ld-seed-p", type=float, default=1e-5,
                         help="Seed SNP p-threshold for LD blocks (default: 1e-5)")
    ld_grp.add_argument("--ld-top-n", type=int, default=10,
                         help="Also seed top-N SNPs (default: 10)")
    ld_grp.add_argument("--hap-perms", type=int, default=1000,
                         help="Haplotype permutations (default: 1000)")
    ld_grp.add_argument("--no-annotation", action="store_true",
                         help="Skip gene annotation")
    ld_grp.add_argument("--genome-build", default="SL3", choices=["SL3", "SL4"],
                         help="Genome build for tomato gene annotation: SL3 (matches Varitome "
                              "(matches Varitome SNP coords, default) or SL4 = ITAG4.0. Ignored for pepper.")
    ld_grp.add_argument("--species", default="tomato", choices=["tomato", "pepper", "custom"],
                         help="Species for annotation files (default: tomato)")
    ld_grp.add_argument("--gene-model", help="Gene coordinate CSV (required if --species custom)")

    # ── Output options ────────────────────────────────────
    parser.add_argument("--no-report", action="store_true", help="Skip HTML report generation")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--export-qc", action="store_true",
                         help="Export post-QC genotype matrix, SNP map, and phenotype for benchmarking")
    parser.add_argument("--drop-alt", action="store_true", help="Drop ALT chromosomes")
    parser.add_argument("--n-chromosomes", type=int, default=None,
                         help="Number of chromosomes (default: auto-detect from VCF). "
                              "When set, only chromosomes 1..N are kept.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--interactive", action="store_true",
                         help="Interactive wizard — prompts for all options step by step")

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
    if args.auto_pcs:
        parts.append("--auto-pcs")
        if args.pc_strategy != "band":
            parts.append(f"--pc-strategy {args.pc_strategy}")
        if args.max_pcs != 10:
            parts.append(f"--max-pcs {args.max_pcs}")
    elif args.n_pcs != 4:
        parts.append(f"--n-pcs {args.n_pcs}")
    if args.subsampling:
        parts.append(f"--subsampling --boot-reps {args.boot_reps}")
        if args.boot_jobs != 1:
            parts.append(f"--boot-jobs {args.boot_jobs}")
    if getattr(args, "sig_thresh", "meff") != "meff":
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
    if args.auto_pcs:
        print(f"  PCs:        Auto ({args.pc_strategy}, max={args.max_pcs})")
    else:
        print(f"  PCs:        {args.n_pcs}")
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
    auto = click.confirm("Auto-select PCs via lambda scan? (recommended)", default=True)
    args.auto_pcs = auto
    if auto:
        args.pc_strategy = click.prompt(
            "  Strategy", type=click.Choice(["band", "closest_to_1"]),
            default="band",
        )
        args.max_pcs = click.prompt("  Max PCs to scan", type=int, default=10)
    else:
        args.n_pcs = click.prompt("  Number of PCs", type=int, default=4)

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
            type=click.Choice(["tomato", "pepper", "custom"]),
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

    # Apply norm mapping (same as non-interactive path)
    norm_map = {
        "none": "None (raw values)",
        "zscore": "Z-score (mean=0, sd=1)",
        "log": "Log transform (if positive)",
        "yeojohnson": "Yeo-Johnson (robust Box-Cox)",
        "int": "Rank-based inverse normal (INT)",
    }
    args.norm = norm_map.get(args.norm, args.norm)

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
    # Map norm slug to display string expected by _pipeline_phenotype_qc
    norm_map = {
        "none": "None (raw values)",
        "zscore": "Z-score (mean=0, sd=1)",
        "log": "Log transform (if positive)",
        "yeojohnson": "Yeo-Johnson (robust Box-Cox)",
        "int": "Rank-based inverse normal (INT)",
    }
    args.norm = norm_map.get(args.norm, args.norm)

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

    log.info("SNP QC (MAF=%.3f, miss=%.2f, MAC=%d, INFO=%.2f)…",
             args.maf, args.miss, args.mac, args.info_thresh)
    canonical = (tuple(str(i) for i in range(1, args.n_chromosomes + 1))
                 if args.n_chromosomes else None)
    geno_df, chroms, chroms_num, positions, sid, n_raw, qc_snp, info_scores = \
        _pipeline_snp_qc(geno_df, chroms, positions, sid,
                         args.maf, args.miss, args.mac, args.drop_alt,
                         info_scores, args.info_thresh, canonical=canonical)
    log.info("  QC: %s", qc_snp)

    log.info("Building genotype matrices…")
    geno_dosage_raw, snp_imputation_rate, geno_imputed, iid = \
        _pipeline_build_geno_matrices(geno_df)

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

    def _make_covar(k):
        if pcs_full is None or k <= 0:
            return None
        return CovarData(iid, pcs_full[:, :k])

    extra_csvs = {}  # additional CSVs to include in ZIP
    figures = {}     # figures to include in ZIP

    if getattr(args, "export_qc", False):
        log.info("Preparing QC'd genotype export for benchmarking…")
        geno_export = pd.DataFrame(
            geno_imputed, index=list(geno_df.index), columns=sid,
        )
        geno_export.index.name = "SampleID"
        extra_csvs["QC_genotype_matrix.csv"] = geno_export.reset_index()
        ref_vals = [allele_map.get(s, (".", "."))[0] for s in sid]
        alt_vals = [allele_map.get(s, (".", "."))[1] for s in sid]
        extra_csvs["QC_snp_map.csv"] = pd.DataFrame({
            "SNP_ID": sid, "Chr": chroms, "Pos": positions,
            "Ref": ref_vals, "Alt": alt_vals,
        })
        extra_csvs["QC_phenotype.csv"] = pd.DataFrame({
            "SampleID": list(geno_df.index), args.trait: y.ravel(),
        })
        log.info("  QC export: %d samples x %d SNPs", geno_imputed.shape[0], geno_imputed.shape[1])

    # Per-model PC counts (fall back to --n-pcs when not specified)
    n_pcs_mlm = min(args.n_pcs_mlm if args.n_pcs_mlm is not None else n_pcs, _max_avail)
    n_pcs_mlmm = min(args.n_pcs_mlmm if args.n_pcs_mlmm is not None else n_pcs_mlm, _max_avail)
    n_pcs_fc = min(args.n_pcs_farmcpu if args.n_pcs_farmcpu is not None else n_pcs, _max_avail)

    # ── Auto PC selection (overrides manual per-model counts) ──
    if args.auto_pcs:
        from gwas.models import auto_select_pcs
        log.info("Auto-selecting PCs (strategy=%s, max=%d)…", args.pc_strategy, args.max_pcs)

        pc_df = auto_select_pcs(
            geno_imputed, y, sid, chroms, chroms_num, positions,
            iid, Z_for_pca, chroms_grm, K, pcs_full,
            max_pcs=min(args.max_pcs, _max_avail),
            strategy=args.pc_strategy,
            use_loco=not getattr(args, "no_loco", False),
            band_lo=getattr(args, "pc_band_lo", 0.95),
            band_hi=getattr(args, "pc_band_hi", 1.05),
            parsimony_tolerance=getattr(args, "pc_parsimony_tol", 0.02),
        )
        extra_csvs["PC_selection_lambda.csv"] = pc_df

        # MLM best k
        best_mlm = pc_df.loc[pc_df["recommended"] == "★"]
        if not best_mlm.empty:
            n_pcs_mlm = int(best_mlm.iloc[0]["n_pcs"])
        log.info("  MLM auto-selected: %d PCs", n_pcs_mlm)

        # MLMM inherits MLM
        n_pcs_mlmm = n_pcs_mlm

        # FarmCPU independent scan (if selected)
        if "farmcpu" in args.model:
            from gwas.models import run_farmcpu as _run_fc_scan
            log.info("  Scanning PCs for FarmCPU…")
            fc_lambdas = []
            _scan_max = min(args.max_pcs, _max_avail)
            for k in range(0, _scan_max + 1):
                try:
                    _fc_covar = _make_covar(k)
                    _fc_df, _, _ = _run_fc_scan(
                        geno_imputed, sid, chroms, chroms_num, positions, iid,
                        pheno_reader, K0, _fc_covar,
                        final_scan=args.farmcpu_final_scan, verbose=False,
                        use_loco=not getattr(args, "no_loco", False),
                    )
                    fc_lambdas.append(
                        compute_lambda_gc(_fc_df["PValue"].values, trim=False)
                    )
                except Exception:
                    fc_lambdas.append(np.nan)

            # Pick PC count using the same band strategy as MLM
            from gwas.models import select_best_pc_from_lambdas
            fc_valid = [np.isfinite(v) for v in fc_lambdas]
            if any(fc_valid):
                n_pcs_fc = select_best_pc_from_lambdas(
                    fc_lambdas, strategy=args.pc_strategy,
                )
                fc_delta = abs(fc_lambdas[n_pcs_fc] - 1.0)
                if fc_delta > 0.5:
                    log.warning("  FarmCPU lambda scan: best lambda=%.3f (delta=%.3f); results may be unreliable",
                                fc_lambdas[n_pcs_fc], fc_delta)
            else:
                log.warning("  FarmCPU lambda scan failed entirely; falling back to MLM PC count (%d)", n_pcs_mlm)
                n_pcs_fc = n_pcs_mlm  # fallback to MLM
            n_pcs_fc = min(n_pcs_fc, _max_avail)
            log.info("  FarmCPU auto-selected: %d PCs", n_pcs_fc)
            # Add FarmCPU lambdas to PC selection table
            if len(fc_lambdas) == len(pc_df):
                pc_df["lambda_gc_FarmCPU"] = fc_lambdas

        # PC selection lambda curve plot
        if not args.no_plots:
            try:
                import matplotlib.pyplot as plt
                _fig_pc, _ax_pc = plt.subplots(figsize=(6, 3.5))
                _ax_pc.plot(pc_df["n_pcs"], pc_df["lambda_gc"],
                            "o-", color="#0072B2", label="MLM")
                if "lambda_gc_FarmCPU" in pc_df.columns:
                    _ax_pc.plot(pc_df["n_pcs"], pc_df["lambda_gc_FarmCPU"],
                                "s-", color="#D55E00", label="FarmCPU")
                _ax_pc.axhline(1.0, ls="--", color="#999", alpha=0.7)
                _ax_pc.axvline(n_pcs_mlm, ls=":", color="#0072B2", alpha=0.6, lw=1.5,
                               label=f"MLM best: {n_pcs_mlm}")
                if "farmcpu" in args.model and n_pcs_fc != n_pcs_mlm:
                    _ax_pc.axvline(n_pcs_fc, ls=":", color="#D55E00", alpha=0.6, lw=1.5,
                                   label=f"FarmCPU best: {n_pcs_fc}")
                _ax_pc.legend(fontsize=8)
                _ax_pc.set_xlabel("Number of PCs")
                _ax_pc.set_ylabel("lambda_GC")
                _ax_pc.set_title("PC Selection: lambda_GC by PC Count")
                _fig_pc.tight_layout()
                figures["PC_selection_lambda.png"] = _fig_pc
                log.info("  PC selection lambda curve saved")
            except Exception as e:
                log.warning("  PC selection lambda plot failed: %s", e)

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
    sig_rule = getattr(args, "sig_thresh", "meff")
    if sig_rule == "bonferroni":
        primary_thresh = bonf_thresh_naive
        log.info("Significance: Bonferroni %.2e (%d SNPs)", primary_thresh, geno_df.shape[1])
    elif sig_rule == "fdr":
        primary_thresh = None  # FDR uses per-SNP q-values
        log.info("Significance: FDR q < 0.05")
    else:  # meff (default)
        primary_thresh = meff_thresh
        log.info("Significance: M_eff %.2e (M_eff=%d)", primary_thresh, _meff_val)

    if "mlm" in args.model:
        log.info("Running MLM (LOCO)…")
        gwas_df = _run_gwas_impl(
            geno_imputed, y, pcs_full, n_pcs_mlm, sid, positions,
            chroms, chroms_num, iid, K0, K_by_chr, pheno_reader, args.trait,
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
            final_scan=args.farmcpu_final_scan, verbose=False,
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
                if sig_rule == "fdr" and "FDR" in _mdf.columns:
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

    if ld_flank_kb is None:
        log.info("Estimating LD decay…")
        ld_distances = []
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
            dk, _, _ = ld.ld_decay(pos_ch, r2_mat, ld_threshold=0.2, max_dist_kb=5000)
            if np.isfinite(dk):
                ld_distances.append(dk)
        if ld_distances:
            ld_decay_kb = float(np.median(ld_distances))
            ld_flank_kb = int(2 * ld_decay_kb)
            log.info("  LD decay ~ %.0f kb -> flank = %d kb", ld_decay_kb, ld_flank_kb)
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

    for model_name, model_df in post_gwas_models:
        log.info("Post-GWAS: %s", model_name)
        m_hap_gwas = None
        m_ld_annotated = None

        # LD block detection
        has_seeds = (model_df["PValue"] < args.ld_seed_p).any()
        if not has_seeds:
            # Also try top-N seeding
            if args.ld_top_n > 0:
                log.info("  No seed SNPs at p < %.1e; using top-%d seeding.",
                         args.ld_seed_p, args.ld_top_n)
            else:
                log.info("  No seed SNPs (p < %.1e) — skipping.", args.ld_seed_p)
                continue

        try:
            m_ld_blocks = ld.find_ld_clusters_genomewide(
                gwas_df=model_df, chroms=chroms, positions=positions,
                geno_imputed=_geno_float, sid=sid,
                ld_threshold=args.ld_r2, flank_kb=ld_flank_kb,
                ld_decay_kb=ld_decay_kb, min_snps=3,
                top_n=args.ld_top_n, sig_thresh=args.ld_seed_p,
            )
            m_ld_blocks, _ = ld.filter_contained_blocks(m_ld_blocks, min_contained=2)
        except Exception as e:
            log.warning("  LD block detection failed for %s: %s", model_name, e)
            continue

        log.info("  %d LD blocks detected", len(m_ld_blocks))

        if model_name == "MLM":
            ld_blocks_mlm = m_ld_blocks

        if m_ld_blocks.empty:
            continue

        # LD heatmaps for blocks with significant lead SNPs
        if not args.no_plots:
            try:
                import seaborn as sns
                import matplotlib.pyplot as plt
                _sig_col_name = (
                    "Significant_Meff" if sig_rule == "meff"
                    else ("Significant_FDR" if sig_rule == "fdr" else "Significant_Bonf")
                )
                _sig_snps = set()
                if _sig_col_name in model_df.columns:
                    _sig_snps = set(model_df.loc[model_df[_sig_col_name], "SNP"].astype(str))
                _n_hm = 0
                for _, _brow in m_ld_blocks.iterrows():
                    _lead_snps = str(_brow.get("Lead SNP", "")).split(";")
                    if not any(s.strip() in _sig_snps for s in _lead_snps):
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
                    "pepper": {
                        "gene_model": _data_dir / "cann_gene_model.csv",
                        "gene_desc": _data_dir / "cann_gene_annotation.txt",
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
                        f"Subsampling GWAS: SNP discovery frequency — {args.trait}"
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
                                        f"Manhattan — MLM — {args.trait}")
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
                    f"Manhattan — {_m_name} — {args.trait}",
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
                title=f"PCA — {args.trait}",
                eigenvalues=pca_eigenvalues,
            )
            figures["PCA_scatter.png"] = fig_pca

    # ── Build HTML report (in-memory) ────────────────────
    report_html = None
    if not args.no_report:
        from gwas.reports import generate_gwas_report

        meta = {
            "VCF": str(vcf_path),
            "Phenotype": str(pheno_path),
            "Trait": args.trait,
            "Models": args.model,
            "MAF threshold": args.maf,
            "MAC threshold": args.mac,
            "Missingness threshold": args.miss,
            "Info threshold": args.info_thresh,
            "Normalization": args.norm,
            "Significance": sig_rule,
            "PCs_MLM": n_pcs_mlm,
            "PCs_MLMM": n_pcs_mlmm if "mlmm" in args.model else "N/A",
            "PCs_FarmCPU": n_pcs_fc if "farmcpu" in args.model else "N/A",
            "Samples": int(geno_df.shape[0]),
            "SNPs (post-QC)": int(geno_df.shape[1]),
            "SNPs (raw)": int(n_raw),
            "Lambda GC": round(lambda_gc, 4),
            "Kinship model": kinship_model,
            "LD decay (kb)": round(ld_decay_kb, 1) if ld_decay_kb else "N/A",
            "LD flank (kb)": ld_flank_kb if ld_flank_kb else "N/A",
            "LD blocks (MLM)": len(ld_blocks_mlm) if ld_blocks_mlm is not None else "N/A",
            "Subsampling reps": args.boot_reps if args.subsampling else "N/A",
            "Auto PCs": "Yes" if args.auto_pcs else "No",
        }

        report_html = generate_gwas_report(
            trait_col=args.trait,
            qc_snp=qc_snp,
            gwas_df=gwas_df,
            figures=figures if not args.no_plots else None,
            metadata=meta,
            mlmm_df=cofactor_tables.get("MLMM"),
            farmcpu_df=cofactor_tables.get("FarmCPU"),
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
