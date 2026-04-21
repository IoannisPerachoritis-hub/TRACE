# Install R packages required for GWAS benchmarking
# GAPIT3 + rMVP + dependencies

cat("Installing R packages for GWAS benchmarking...\n")

# Set up user library (cross-platform: Linux/macOS/Windows)
user_lib <- Sys.getenv("R_LIBS_USER")
if (user_lib == "") {
  # tools::R_user_dir() resolves the OS-appropriate per-user directory
  # (~/.local/share/R on Linux, ~/Library/R on macOS, %APPDATA%/R on Windows)
  user_lib <- file.path(tools::R_user_dir("R", which = "data"), "library",
                         paste0(R.version$major, ".",
                                sub("\\..*", "", R.version$minor)))
}
dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(user_lib, .libPaths()))
cat(sprintf("User library: %s\n", user_lib))

# Set CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Install BiocManager if needed
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", lib = user_lib)
}

# Install core dependencies
pkgs <- c("data.table", "bigmemory", "Rcpp", "RcppArmadillo",
           "RcppProgress", "BH", "R6", "remotes")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) {
    cat(sprintf("Installing %s...\n", p))
    install.packages(p, lib = user_lib)
  } else {
    cat(sprintf("  %s already installed\n", p))
  }
}

# Install GAPIT3 from GitHub
cat("Installing GAPIT3 from GitHub...\n")
tryCatch({
  remotes::install_github("jiabowang/GAPIT3", lib = user_lib,
                           quiet = FALSE, upgrade = "never")
  cat("  GAPIT3 installed successfully\n")
}, error = function(e) {
  cat(sprintf("  GAPIT3 install error: %s\n", conditionMessage(e)))
  cat("  Trying alternative: GAPIT from CRAN dependencies...\n")
})

# Install rMVP from CRAN
if (!requireNamespace("rMVP", quietly = TRUE)) {
  cat("Installing rMVP...\n")
  install.packages("rMVP", lib = user_lib)
} else {
  cat("  rMVP already installed\n")
}

cat("\nVerifying installations...\n")
gapit_ok <- requireNamespace("GAPIT", quietly = TRUE)
rmvp_ok <- requireNamespace("rMVP", quietly = TRUE)
cat(sprintf("  GAPIT3: %s\n", gapit_ok))
cat(sprintf("  rMVP:   %s\n", rmvp_ok))

if (!gapit_ok || !rmvp_ok) {
  cat("\nWARNING: Some packages failed to install.\n")
  quit(status = 1)
}
cat("All packages installed successfully.\n")
