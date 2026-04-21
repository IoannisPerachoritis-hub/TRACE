#!/usr/bin/env Rscript
# Run rMVP MLM on a single simulated phenotype.
# Usage: Rscript run_simulation_rmvp.R <scenario_dir> [n_pcs]

# Cross-platform library path: Linux/macOS/Windows
user_lib <- Sys.getenv("R_LIBS_USER")
if (user_lib == "") {
  user_lib <- file.path(tools::R_user_dir("R", which = "data"), "library",
                        paste0(R.version$major, ".",
                               sub("\\..*", "", R.version$minor)))
}
.libPaths(c(user_lib, .libPaths()))

for (pkg in c("rMVP", "data.table", "bigmemory")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf("Required package '%s' is not installed.", pkg), call. = FALSE)
  }
}

suppressPackageStartupMessages({
  library(rMVP)
  library(data.table)
  library(bigmemory)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  cat("Usage: Rscript run_simulation_rmvp.R <scenario_dir> [n_pcs]\n")
  quit(status = 1)
}

scenario_dir <- normalizePath(args[1], mustWork = TRUE)
n_pcs <- if (length(args) >= 2) as.integer(args[2]) else 2

# ── Load fixed genotype data ─────────────────────────────
base_dir <- getwd()  # Expect to run from project root
qc_dir <- file.path(base_dir, "benchmarks", "qc_data", "tomato_locule_number")

geno_raw <- fread(file.path(qc_dir, "QC_genotype_matrix.csv"), header = TRUE)
snp_map  <- fread(file.path(qc_dir, "QC_snp_map.csv"), header = TRUE)

# Filter non-numeric chromosomes
numeric_mask <- grepl("^[0-9]+$", snp_map$Chr)
if (any(!numeric_mask)) {
  keep_snps <- snp_map$SNP_ID[numeric_mask]
  snp_map <- snp_map[numeric_mask, ]
  geno_raw <- geno_raw[, c("SampleID", keep_snps), with = FALSE]
}

# ── Load simulated phenotype ─────────────────────────────
pheno <- fread(file.path(scenario_dir, "phenotype.csv"), header = TRUE)

# ── Align samples: keep intersection in matching order ───
shared_ids <- intersect(geno_raw$SampleID, pheno$SampleID)
if (length(shared_ids) == 0) stop("No shared SampleIDs between genotype and phenotype")
if (length(shared_ids) < nrow(pheno)) {
  cat(sprintf("  Warning: %d/%d phenotype samples matched genotype\n",
              length(shared_ids), nrow(pheno)))
}
geno_raw <- geno_raw[match(shared_ids, geno_raw$SampleID), ]
pheno <- pheno[match(shared_ids, pheno$SampleID), ]

# ── Set up rMVP ──────────────────────────────────────────
out_dir <- file.path(scenario_dir, "rmvp")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Prepare numeric genotype matrix (bigmemory)
geno_mat <- as.matrix(geno_raw[, -1, with = FALSE])
n_samples <- nrow(geno_mat)
n_snps <- ncol(geno_mat)

# Write to bigmemory-backed file
bm_path <- file.path(out_dir, "geno.bin")
bm_desc <- file.path(out_dir, "geno.desc")
bm <- filebacked.big.matrix(
  nrow = n_samples, ncol = n_snps,
  type = "double",
  backingfile = basename(bm_path),
  backingpath = out_dir,
  descriptorfile = basename(bm_desc)
)
bm[,] <- geno_mat

# Kinship (VanRaden)
K <- MVP.K.VanRaden(bm, verbose = FALSE)

# PCA
pca <- NULL
if (n_pcs > 0) {
  pca <- MVP.PCA(bm, pcs.keep = n_pcs, verbose = FALSE)
}

# Phenotype vector
y_df <- data.frame(Taxa = pheno$SampleID, SimTrait = pheno$SimTrait)

# Map
map_df <- data.frame(
  SNP = snp_map$SNP_ID,
  CHROM = as.integer(snp_map$Chr),
  POS = as.integer(snp_map$Pos)
)

# ── Run rMVP MLM ─────────────────────────────────────────
start_time <- proc.time()

tryCatch({
  res <- MVP(
    phe = y_df,
    geno = bm,
    map = map_df,
    K = K,
    CV.MLM = if (!is.null(pca)) pca else NULL,
    nPC.MLM = n_pcs,
    method = "MLM",
    file.output = FALSE,
    verbose = FALSE
  )

  elapsed <- (proc.time() - start_time)[["elapsed"]]

  # Standardize results
  pvals <- res$mlm.results
  # Try to extract p-value by column name first, fall back to last column
  pval_col <- grep("p\\.value|pvalue|p_value", colnames(pvals),
                   ignore.case = TRUE, value = TRUE)
  if (length(pval_col) > 0) {
    pval_vec <- pvals[, pval_col[1]]
  } else {
    pval_vec <- pvals[, ncol(pvals)]
    cat(sprintf("  rMVP: p-value extracted from last column (%s)\n",
                colnames(pvals)[ncol(pvals)]))
  }
  std <- data.frame(
    SNP = map_df$SNP,
    Chr = map_df$CHROM,
    Pos = map_df$POS,
    PValue = pval_vec
  )
  write.csv(std, file.path(out_dir, "results.csv"), row.names = FALSE)

  timing <- data.frame(elapsed_sec = round(elapsed, 2))
  write.csv(timing, file.path(out_dir, "timing.csv"), row.names = FALSE)

}, error = function(e) {
  writeLines(conditionMessage(e), file.path(out_dir, "error.txt"))
})

# Cleanup bigmemory files
unlink(c(bm_path, bm_desc), force = TRUE)
