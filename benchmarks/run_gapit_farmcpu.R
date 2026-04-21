#!/usr/bin/env Rscript
# ──────────────────────────────────────────────────────────────
# GAPIT3 FarmCPU benchmark on pre-QC'd data from TRACE
# Reads the identical QC_genotype_matrix.csv and QC_phenotype.csv
# produced by --export-qc so the ONLY variable is the association engine.
# ──────────────────────────────────────────────────────────────

# Ensure user library is on path (cross-platform: Linux/macOS/Windows)
user_lib <- Sys.getenv("R_LIBS_USER")
if (user_lib == "") {
  user_lib <- file.path(tools::R_user_dir("R", which = "data"), "library",
                         paste0(R.version$major, ".",
                                sub("\\..*", "", R.version$minor)))
}
.libPaths(c(user_lib, .libPaths()))

suppressPackageStartupMessages({
  library(GAPIT)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  cat("Usage: Rscript run_gapit_farmcpu.R <run_name> [n_pcs]\n")
  cat("  run_name: pepper_FWe | pepper_BX | tomato_weight_g | tomato_locule_number\n")
  cat("  n_pcs:    number of PCs (default: from platform auto-PC)\n")
  quit(status = 1)
}

run_name <- args[1]
# PC counts matching platform auto-PC selections
default_pcs <- list(pepper_FWe = 2, pepper_BX = 0,
                     tomato_weight_g = 0, tomato_locule_number = 0)
n_pcs <- if (length(args) >= 2) as.integer(args[2]) else default_pcs[[run_name]]
if (is.null(n_pcs)) n_pcs <- 3  # fallback

cat(sprintf("=== GAPIT3 FarmCPU benchmark: %s (PCs=%d) ===\n", run_name, n_pcs))

# ── Paths ────────────────────────────────────────────────────
base_dir <- getwd()
qc_dir <- file.path(base_dir, "benchmarks", "qc_data", run_name)
out_dir <- file.path(base_dir, "benchmarks", "results", "gapit_farmcpu", run_name)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("QC data dir: %s\n", qc_dir))
cat(sprintf("Output dir:  %s\n", out_dir))

# ── Load QC data ─────────────────────────────────────────────
cat("Loading genotype matrix...\n")
geno_raw <- fread(file.path(qc_dir, "QC_genotype_matrix.csv"), header = TRUE)
snp_map  <- fread(file.path(qc_dir, "QC_snp_map.csv"), header = TRUE)
pheno    <- fread(file.path(qc_dir, "QC_phenotype.csv"), header = TRUE)

cat(sprintf("  Genotype: %d samples x %d SNPs\n",
            nrow(geno_raw), ncol(geno_raw) - 1))
cat(sprintf("  Phenotype: %d samples\n", nrow(pheno)))

# ── Filter non-numeric chromosomes ───────────────────────────
# GAPIT requires integer chromosomes; drop scaffolds/unplaced contigs
numeric_mask <- grepl("^[0-9]+$", snp_map$Chr)
n_dropped <- sum(!numeric_mask)
if (n_dropped > 0) {
  cat(sprintf("  Dropping %d SNPs on non-numeric chromosomes: %s\n",
              n_dropped, paste(unique(snp_map$Chr[!numeric_mask]), collapse=", ")))
  keep_snps <- snp_map$SNP_ID[numeric_mask]
  snp_map <- snp_map[numeric_mask, ]
  geno_raw <- geno_raw[, c("SampleID", keep_snps), with = FALSE]
}

# ── Convert to GAPIT format ──────────────────────────────────
# Phenotype
trait_col <- setdiff(names(pheno), "SampleID")
myY <- data.frame(Taxa = pheno$SampleID, pheno[[trait_col[1]]])
names(myY)[2] <- trait_col[1]

# Genotype (numeric matrix with Taxa column)
myGD <- data.frame(Taxa = geno_raw$SampleID, geno_raw[, -1, with = FALSE])

# Genetic map
myGM <- data.frame(
  Name = snp_map$SNP_ID,
  Chromosome = as.integer(snp_map$Chr),
  Position = as.integer(snp_map$Pos)
)

cat(sprintf("  GAPIT GD: %d x %d\n", nrow(myGD), ncol(myGD)))
cat(sprintf("  GAPIT GM: %d markers\n", nrow(myGM)))
cat(sprintf("  GAPIT Y:  %d samples, trait=%s\n", nrow(myY), trait_col[1]))

# ── Run GAPIT FarmCPU ──────────────────────────────────────────
cat("\nRunning GAPIT FarmCPU...\n")
start_time <- proc.time()

# Change to output directory so GAPIT writes files there
old_wd <- getwd()
setwd(out_dir)

tryCatch({
  result <- GAPIT(
    Y = myY,
    GD = myGD,
    GM = myGM,
    model = "FarmCPU",
    PCA.total = n_pcs,
    SNP.MAF = 0,  # Already QC'd — no additional MAF filter
    file.output = TRUE
  )
  elapsed <- (proc.time() - start_time)[["elapsed"]]
  cat(sprintf("\nGAPIT FarmCPU completed in %.1f seconds\n", elapsed))

  # ── Save timing ──────────────────────────────────────────
  timing <- data.frame(
    run = run_name,
    tool = "GAPIT3_FarmCPU",
    model = "FarmCPU",
    n_pcs = n_pcs,
    elapsed_sec = round(elapsed, 1),
    n_samples = nrow(myY),
    n_snps = nrow(myGM)
  )
  write.csv(timing, file.path(out_dir, "timing.csv"), row.names = FALSE)

  # ── Extract and standardize results ─────────────────────
  gwas_files <- list.files(out_dir, pattern = "GWAS_Results.*\\.csv$",
                            full.names = TRUE)
  gwas_files <- gwas_files[!grepl("StdErr|Filter", gwas_files)]
  if (length(gwas_files) > 0) {
    cat(sprintf("Reading: %s\n", basename(gwas_files[1])))
    gwas <- fread(gwas_files[1])
    cat(sprintf("GWAS results: %d rows\n", nrow(gwas)))
    cat(sprintf("Columns: %s\n", paste(names(gwas), collapse = ", ")))

    # Standardize column names for comparison
    std <- data.frame(
      SNP = gwas$SNP,
      Chr = gwas$Chr,
      Pos = gwas$Pos,
      PValue = gwas$P.value,
      Effect = if ("Effect" %in% names(gwas)) gwas$Effect else NA
    )
    write.csv(std, file.path(out_dir, "gapit_farmcpu_results_standardized.csv"),
              row.names = FALSE)

    # Summary statistics
    sig_bonf <- sum(std$PValue < 0.05 / nrow(std), na.rm = TRUE)
    cat(sprintf("Significant SNPs (Bonferroni): %d\n", sig_bonf))
    cat(sprintf("Min p-value: %.2e\n", min(std$PValue, na.rm = TRUE)))

    # Lambda GC
    chisq <- qchisq(1 - std$PValue[!is.na(std$PValue)], df = 1)
    lambda_gc <- median(chisq) / qchisq(0.5, df = 1)
    cat(sprintf("Lambda GC: %.3f\n", lambda_gc))

    summary_df <- data.frame(
      run = run_name,
      tool = "GAPIT3_FarmCPU",
      model = "FarmCPU",
      n_snps_tested = nrow(std),
      lambda_gc = round(lambda_gc, 4),
      sig_bonferroni = sig_bonf,
      min_pvalue = min(std$PValue, na.rm = TRUE),
      elapsed_sec = round(elapsed, 1)
    )
    write.csv(summary_df, file.path(out_dir, "summary.csv"),
              row.names = FALSE)
  }
}, error = function(e) {
  cat(sprintf("ERROR: %s\n", conditionMessage(e)))
  writeLines(conditionMessage(e), file.path(out_dir, "error.txt"))
}, finally = {
  setwd(old_wd)
})

cat(sprintf("\n=== GAPIT3 FarmCPU %s complete ===\n", run_name))
