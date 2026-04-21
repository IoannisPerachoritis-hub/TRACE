#!/usr/bin/env Rscript
# ──────────────────────────────────────────────────────────────
# Master benchmark runner: GAPIT3 + rMVP on all 4 dataset×trait combos
# ──────────────────────────────────────────────────────────────

# Ensure user library is on path (cross-platform: Linux/macOS/Windows)
user_lib <- Sys.getenv("R_LIBS_USER")
if (user_lib == "") {
  user_lib <- file.path(tools::R_user_dir("R", which = "data"), "library",
                         paste0(R.version$major, ".",
                                sub("\\..*", "", R.version$minor)))
}
.libPaths(c(user_lib, .libPaths()))

runs <- c("pepper_FWe", "pepper_BX", "tomato_weight_g", "tomato_locule_number")

cat("=" |> strrep(60), "\n")
cat("  GWAS Benchmarking: GAPIT3 + rMVP\n")
cat("=" |> strrep(60), "\n\n")

# ── Run GAPIT3 on all datasets ──────────────────────────────
cat("\n>>> GAPIT3 BENCHMARKS <<<\n\n")
for (run in runs) {
  cat(sprintf("\n--- GAPIT3: %s ---\n", run))
  tryCatch({
    source("benchmarks/run_gapit.R", local = new.env())
  }, error = function(e) {
    # Source doesn't work well with args, use system call
  })
  cmd <- sprintf('Rscript benchmarks/run_gapit.R "%s"', run)
  cat(sprintf("Running: %s\n", cmd))
  ret <- system(cmd, intern = FALSE)
  if (ret != 0) cat(sprintf("  WARNING: %s returned code %d\n", run, ret))
}

# ── Run rMVP on all datasets ───────────────────────────────
cat("\n>>> rMVP BENCHMARKS <<<\n\n")
for (run in runs) {
  cat(sprintf("\n--- rMVP: %s ---\n", run))
  cmd <- sprintf('Rscript benchmarks/run_rmvp.R "%s"', run)
  cat(sprintf("Running: %s\n", cmd))
  ret <- system(cmd, intern = FALSE)
  if (ret != 0) cat(sprintf("  WARNING: %s returned code %d\n", run, ret))
}

# ── Collect summaries ───────────────────────────────────────
cat("\n\n>>> COLLECTING RESULTS <<<\n\n")
library(data.table)

all_summaries <- list()
for (tool in c("gapit", "rmvp")) {
  for (run in runs) {
    sfile <- file.path("benchmarks", "results", tool, run, "summary.csv")
    if (file.exists(sfile)) {
      all_summaries[[length(all_summaries) + 1]] <- fread(sfile)
    }
  }
}

if (length(all_summaries) > 0) {
  combined <- rbindlist(all_summaries)
  write.csv(combined, "benchmarks/results/all_summaries.csv", row.names = FALSE)
  cat("Combined summary:\n")
  print(combined)
}

cat("\n\nAll benchmarks complete.\n")
