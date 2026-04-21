#!/usr/bin/env Rscript
# Run GAPIT3 FarmCPU on a single simulated phenotype.
# Usage: Rscript run_simulation_gapit_farmcpu.R <scenario_dir> [n_pcs]
# Where <scenario_dir> contains phenotype.csv with columns SampleID, SimTrait.
# Genotype data is loaded from the fixed tomato QC data.

# Cross-platform library path: Linux/macOS/Windows
user_lib <- Sys.getenv("R_LIBS_USER")
if (user_lib == "") {
  user_lib <- file.path(tools::R_user_dir("R", which = "data"), "library",
                        paste0(R.version$major, ".",
                               sub("\\..*", "", R.version$minor)))
}
.libPaths(c(user_lib, .libPaths()))

for (pkg in c("GAPIT", "data.table")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf("Required package '%s' is not installed.", pkg), call. = FALSE)
  }
}

suppressPackageStartupMessages({
  library(GAPIT)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  cat("Usage: Rscript run_simulation_gapit_farmcpu.R <scenario_dir> [n_pcs]\n")
  quit(status = 1)
}

# Make all paths absolute immediately
scenario_dir <- normalizePath(args[1], mustWork = TRUE)
n_pcs <- if (length(args) >= 2) as.integer(args[2]) else 2

# ── Load fixed genotype data ──────────────────────────────
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

# ── Convert to GAPIT format ──────────────────────────────
myY <- data.frame(Taxa = pheno$SampleID, SimTrait = pheno$SimTrait)
myGD <- data.frame(Taxa = geno_raw$SampleID, geno_raw[, -1, with = FALSE])
myGM <- data.frame(
  Name = snp_map$SNP_ID,
  Chromosome = as.integer(snp_map$Chr),
  Position = as.integer(snp_map$Pos)
)

# ── Run GAPIT FarmCPU ─────────────────────────────────────
out_dir <- file.path(scenario_dir, "gapit_farmcpu")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Use a temp directory for GAPIT working files
tmp_wd <- tempdir()
old_wd <- getwd()
setwd(tmp_wd)

start_time <- proc.time()

tryCatch({
  sink(tempfile())  # Suppress GAPIT's verbose output
  result <- GAPIT(
    Y = myY, GD = myGD, GM = myGM,
    model = "FarmCPU",
    PCA.total = n_pcs,
    SNP.MAF = 0,
    file.output = FALSE  # No plots/files for speed
  )
  sink()

  elapsed <- (proc.time() - start_time)[["elapsed"]]

  # Extract results from GAPIT result object (file.output=FALSE)
  if (!is.null(result$GWAS)) {
    gwas <- result$GWAS
    std <- data.frame(
      SNP = gwas$SNP,
      Chr = gwas$Chr,
      Pos = gwas$Pos,
      PValue = gwas$P.value
    )
  } else {
    stop("No GWAS results found in GAPIT output")
  }

  write.csv(std, file.path(out_dir, "results.csv"), row.names = FALSE)

  timing <- data.frame(elapsed_sec = round(elapsed, 2))
  write.csv(timing, file.path(out_dir, "timing.csv"), row.names = FALSE)

}, error = function(e) {
  try(sink(), silent = TRUE)  # Restore output safely
  try(
    writeLines(conditionMessage(e), file.path(out_dir, "error.txt")),
    silent = TRUE
  )
}, finally = {
  setwd(old_wd)
})
