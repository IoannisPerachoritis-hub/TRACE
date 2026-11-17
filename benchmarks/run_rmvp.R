#!/usr/bin/env Rscript
# rMVP MLM on TRACE's QC data: the rMVP column of Tables S4 and S11.
# It reads QC_genotype_matrix.csv and QC_phenotype.csv (TRACE's --export-qc), so the
# association engine is the only difference.
#
#   Rscript benchmarks/run_rmvp.R <run_name> [n_pcs] [pcs_twice]
#
# Run it from the data root: it reads benchmarks/qc_data/<run_name>/ and writes
# benchmarks/results/rmvp/<run_name>/ there, with rmvp_results_standardized.csv, sessionInfo.txt
# and packages.txt. The output folder must be empty. n_pcs defaults to the published count.
# The PCs are passed once, as CV.MLM. The published runs passed them twice (CV.MLM, and
# nPC.MLM = n_pcs, so rMVP added its own PCs again); pcs_twice repeats that, into
# benchmarks/results/rmvp_pcs_twice/<run_name>/, for the values the tables give in brackets.
# At n_pcs = 0 the two are the same. rMVP 1.4.6 (commit 48b54836) made the published tables.

# Ensure user library is on path (cross-platform: Linux/macOS/Windows)
user_lib <- Sys.getenv("R_LIBS_USER")
if (user_lib == "") {
  user_lib <- file.path(tools::R_user_dir("R", which = "data"), "library",
                         paste0(R.version$major, ".",
                                sub("\\..*", "", R.version$minor)))
}
.libPaths(c(user_lib, .libPaths()))

suppressPackageStartupMessages({
  library(rMVP)
  library(data.table)
})
# the rMVP that ran, by version and commit (the published tables: 1.4.6 at 48b54836)
rmvp_desc <- packageDescription("rMVP")
cat(sprintf("rMVP %s, commit %s, from %s\n", rmvp_desc$Version,
            if (is.null(rmvp_desc$RemoteSha)) "unknown" else rmvp_desc$RemoteSha, find.package("rMVP")))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  cat("Usage: Rscript run_rmvp.R <run_name> [n_pcs]\n")
  cat("  run_name: pepper_FWe | pepper_BX | tomato_weight_g | tomato_locule_number\n")
  cat("  n_pcs:    number of PCs (default: the published count)\n")
  cat("  pcs_twice: pass the PCs twice, as the published runs did\n")
  quit(status = 1)
}

run_name <- args[1]
# the published PC counts (published_pc_counts.PUBLISHED_K)
default_pcs <- list(pepper_FWe = 2, pepper_BX = 0,
                     tomato_weight_g = 0, tomato_locule_number = 0)
n_pcs <- if (length(args) >= 2) as.integer(args[2]) else default_pcs[[run_name]]
if (is.null(n_pcs) || is.na(n_pcs)) stop(paste("Unknown run_name:", run_name, "-- pass n_pcs explicitly"))
cat(sprintf("n_pcs = %d (the default for %s: %s)\n", n_pcs, run_name,
            if (is.null(default_pcs[[run_name]])) "none" else default_pcs[[run_name]]))
pcs_twice <- length(args) >= 3 && args[3] == "pcs_twice"
if (length(args) >= 3 && !pcs_twice) stop("the third argument, when given, is pcs_twice")

cat(sprintf("=== rMVP MLM benchmark: %s (PCs=%d) ===\n", run_name, n_pcs))

# ── Paths ────────────────────────────────────────────────────
base_dir <- getwd()
qc_dir <- file.path(base_dir, "benchmarks", "qc_data", run_name)
out_dir <- file.path(base_dir, "benchmarks", "results", if (pcs_twice) "rmvp_pcs_twice" else "rmvp", run_name)
# a fresh output folder: the result file is found by listing it
if (dir.exists(out_dir) && length(list.files(out_dir, all.files = TRUE, no.. = TRUE)) > 0)
  stop(paste("The output folder is not empty:", out_dir, "-- remove it to run again"))
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
numeric_mask <- grepl("^[0-9]+$", snp_map$Chr)
n_dropped <- sum(!numeric_mask)
if (n_dropped > 0) {
  cat(sprintf("  Dropping %d SNPs on non-numeric chromosomes: %s\n",
              n_dropped, paste(unique(snp_map$Chr[!numeric_mask]), collapse=", ")))
  keep_snps <- snp_map$SNP_ID[numeric_mask]
  snp_map <- snp_map[numeric_mask, ]
  geno_raw <- geno_raw[, c("SampleID", keep_snps), with = FALSE]
}

# ── Prepare rMVP inputs ─────────────────────────────────────
# rMVP needs numeric genotype matrix (samples x SNPs), map, and phenotype

# Genotype matrix (no Taxa column)
geno_mat <- as.matrix(geno_raw[, -1, with = FALSE])
rownames(geno_mat) <- geno_raw$SampleID

# Map: SNP, Chr, Pos
map_df <- data.frame(
  SNP = snp_map$SNP_ID,
  CHROM = as.integer(snp_map$Chr),
  POS = as.integer(snp_map$Pos)
)

# Phenotype
trait_col <- setdiff(names(pheno), "SampleID")
pheno_df <- data.frame(
  Taxa = pheno$SampleID,
  pheno[[trait_col[1]]]
)
names(pheno_df)[2] <- trait_col[1]

# ── Compute kinship + PCA via rMVP ──────────────────────────
cat("Computing kinship matrix...\n")
K <- MVP.K.VanRaden(bigmemory::as.big.matrix(geno_mat))

# PCA
if (n_pcs > 0) {
  cat(sprintf("Computing PCA (%d PCs)...\n", n_pcs))
  pca <- MVP.PCA(M = bigmemory::as.big.matrix(geno_mat), pcs.keep = n_pcs)
} else {
  pca <- NULL
}

# ── Run rMVP MLM ────────────────────────────────────────────
cat("\nRunning rMVP MLM...\n")
start_time <- proc.time()

old_wd <- getwd()
setwd(out_dir)

tryCatch({
  result <- MVP(
    phe = pheno_df,
    geno = bigmemory::as.big.matrix(geno_mat),
    map = map_df,
    K = K,
    CV.MLM = pca,
    # the PCs once, as CV.MLM; with pcs_twice also nPC.MLM = n_pcs, so rMVP adds its own PCs
    # again, as in the published runs
    nPC.MLM = if (pcs_twice) n_pcs else 0,
    method = "MLM",
    file.output = FALSE,
    verbose = TRUE
  )

  elapsed <- (proc.time() - start_time)[["elapsed"]]
  cat(sprintf("\nrMVP MLM completed in %.1f seconds\n", elapsed))

  # ── Extract and standardize results ─────────────────────
  # rMVP returns list: $map, $glm.results, $mlm.results, $farmcpu.results
  # $mlm.results is a matrix with columns: Effect, SE, <trait>.MLM
  mlm_res <- result$mlm.results
  map_res <- result$map  # has SNP, CHROM, POS, MAF

  if (!is.null(mlm_res)) {
    # The p-value column is the last one (named <trait>.MLM)
    pval_col <- ncol(mlm_res)
    cat(sprintf("MLM result columns: %s\n", paste(colnames(mlm_res), collapse=", ")))

    std <- data.frame(
      SNP = map_res$SNP,
      Chr = map_res$CHROM,
      Pos = map_res$POS,
      PValue = as.numeric(mlm_res[, pval_col]),
      Effect = as.numeric(mlm_res[, 1])
    )

    write.csv(std, file.path(out_dir, "rmvp_results_standardized.csv"),
              row.names = FALSE)

    # Summary statistics
    sig_bonf <- sum(std$PValue < 0.05 / nrow(std), na.rm = TRUE)
    cat(sprintf("Significant SNPs (Bonferroni): %d\n", sig_bonf))
    cat(sprintf("Min p-value: %.2e\n", min(std$PValue, na.rm = TRUE)))

    # Lambda GC
    chisq <- qchisq(1 - std$PValue[!is.na(std$PValue)], df = 1)
    lambda_gc <- median(chisq, na.rm = TRUE) / qchisq(0.5, df = 1)
    cat(sprintf("Lambda GC: %.3f\n", lambda_gc))

    # Save timing
    timing <- data.frame(
      run = run_name,
      tool = "rMVP",
      model = "MLM",
      n_pcs = n_pcs,
      elapsed_sec = round(elapsed, 1),
      n_samples = nrow(pheno_df),
      n_snps = nrow(map_df)
    )
    write.csv(timing, file.path(out_dir, "timing.csv"), row.names = FALSE)

    # Save summary
    summary_df <- data.frame(
      run = run_name,
      tool = "rMVP",
      model = "MLM",
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

cat(sprintf("\n=== rMVP %s complete ===\n", run_name))

# the session and the package versions, written with the outputs
writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))
desc_field <- function(v) if (is.null(v)) "" else v
package_line <- function(p) { d <- packageDescription(p)
  sprintf("%s %s RemoteSha=%s Packaged=%s Built=%s path=%s", p, desc_field(d$Version),
          desc_field(d$RemoteSha), desc_field(d$Packaged), desc_field(d$Built), find.package(p)) }
writeLines(vapply(c("rMVP", "data.table", "bigmemory"), package_line, ""), file.path(out_dir, "packages.txt"))
