#!/usr/bin/env python3
"""Generate the test-suite summary table in ``docs/development.md``.

The table between the ``BEGIN/END GENERATED: test-table`` markers in ``docs/development.md``
is auto-generated from the live suite -- **do not edit it by hand**. Re-run::

    python -m pytest tests/ --cov=gwas --cov=utils --cov-report=xml -q   # -> coverage.xml
    python scripts/gen_test_table.py

after adding or removing tests. The generator collects the suite with
``pytest --collect-only -q``, groups every ``tests/test_*.py`` into the thematic
rows in ``AREAS`` below, and asserts the per-row counts sum to the collected
total -- so the table can never silently drift from the suite (the counts
must sum to the total reported in the manuscript).

Hard rule: a collected test file that is not mapped in ``AREAS`` makes this
script exit non-zero and names the file. Silently dropping a file is exactly
what let the hand-maintained table drift (378 rows vs a 739-test suite).

Only the content between the two markers is rewritten; the rest of the
file is left byte-for-byte unchanged (LF endings are preserved).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_README = _ROOT / "docs" / "development.md"

_BEGIN = "<!-- BEGIN GENERATED: test-table (scripts/gen_test_table.py -- do not edit by hand) -->"
_END = "<!-- END GENERATED: test-table -->"

# Ordered thematic grouping. Every collected tests/test_*.py MUST appear in
# exactly one area (an unmapped file aborts the generator). Format:
#   (row label, "what it exercises", [test-file stems without the .py])
AREAS: list[tuple[str, str, list[str]]] = [
    ("Genotype I/O & parsing", "VCF/dosage parsing, INFO scores, upload edge cases",
        ["test_io", "test_info_scores", "test_upload_edge_cases", "test_quality_field_number_a",
         "test_duplicate_collapse_called"]),
    ("Quality control", "MAF/missingness/MAC/heterozygosity filters, per-group QC, QC report",
        ["test_qc", "test_qc_report", "test_qc_group_b", "test_qc_group_c",
         "test_qc_group_d", "test_qc_group_e"]),
    ("Imputation", "mean and LD-kNNi imputation and its cache",
        ["test_impute", "test_impute_cache"]),
    ("Phenotype QC & transforms", "normality diagnostics, transformations, embedded QC panel, applied transformations",
        ["test_pheno_normality", "test_pheno_qc_embed_apptest", "test_pheno_transform_apply_apptest",
         "test_pheno_id_blanks", "test_infinite_trait"]),
    ("Kinship (GRM / LOCO)", "VanRaden GRM and leave-one-chromosome-out kernels",
        ["test_kinship"]),
    ("Association models", "MLM / MLMM / FarmCPU scans and FarmCPU pseudo-QTN selection",
        ["test_models", "test_farmcpu_step5", "test_farmcpu_selection_guard",
         "test_mlm_per_copy", "test_mlmm_window", "test_mlmm_cofactors"]),
    ("Significance & multiple testing", "M_eff / Bonferroni / FDR / custom-threshold rules and every model's significance columns",
        ["test_significance", "test_model_significance", "test_manhattan_tie", "test_meff_exact"]),
    ("Covariates & PC diagnostics", "user covariates, covariates from the phenotype file and the PC eigenvalue-spectrum diagnostics",
        ["test_covariates", "test_covar_cli_e2e", "test_pc_diagnostics", "test_covariate_drop",
         "test_covariates_from_pheno"]),
    ("LD block detection", "peak-centric LD block detection and merging",
        ["test_ld", "test_ld_merge", "test_ld_window"]),
    ("Haplotype testing", "block haplotype effects, the kinship-aware test, effect sizes, compact letter display, sample-to-haplotype assignments",
        ["test_haplotype", "test_haplotype_effects", "test_cld", "test_haplotype_assignments",
         "test_block_table", "test_haplotype_kinship", "test_hap_tab_eta2"]),
    ("LD triage", "coherence/haplotype triage layers, router and eta-squared comparability",
        ["test_ld_triage_layer1", "test_ld_triage_layer2", "test_ld_triage_invariants",
         "test_triage_router", "test_triage_eta2", "test_triage_cli"]),
    ("Isolated-SNP rescue & significant-SNP table", "unblocked-SNP intervals and the significant-SNP table",
        ["test_isolated", "test_isolated_cli", "test_sigtable", "test_tab_significant_snps",
         "test_candidate_intervals"]),
    ("Regional & per-SNP visualisation", "regional association plots, LD-block heatmaps, per-SNP boxplots and their haplotype colours, plotting stats, sample views",
        ["test_regional_plot", "test_regional_tab_apptest", "test_ld_block_heatmap", "test_snpview",
         "test_snpplots", "test_plotting_stats", "test_views", "test_snp_boxplot_haplotypes",
         "test_snp_boxplot_haplotypes_cli", "test_snp_boxplot_haplotypes_apptest",
         "test_lead_snp_plot_data", "test_hap_boxplot", "test_regional_gene_track", "test_snpview_collapse",
         "test_manhattan_thinning"]),
    ("Gene annotation", "LD-block gene annotation, gene-model summaries and resolution",
        ["test_annotation", "test_gene_model_summary", "test_gene_model_resolution",
         "test_gene_files", "test_gff3"]),
    ("Subsampling stability", "bootstrap subsampling and stability metrics",
        ["test_subsampling", "test_stability"]),
    ("HTML report", "run-report assembly and section rendering",
        ["test_reports", "test_report_sections"]),
    ("Command-line interface", "CLI parsing, end-to-end and multi-trait runs, doc-to-parser flag parity",
        ["test_cli", "test_cli_e2e", "test_multitrait_cli", "test_doc_flag_parity", "test_example_blocks", "test_zip_patterns",
         "test_numeric_sig_thresh_cli", "test_cli_phenotype_stop", "test_cli_warnings_logged", "test_seed_help"]),
    ("Web UI (Streamlit)", "app-test coverage of the GWAS, Post-GWAS, help and landing pages",
        ["test_gwas_analysis_apptest", "test_help_page_apptest", "test_zz_app_landing_apptest",
         "test_local_ld_star_apptest", "test_block_tables_apptest",
         "test_hap_sample_alignment_apptest", "test_one_click_multitrait_apptest",
         "test_ld_heatmap_same_window_apptest", "test_detailed_view_significance_apptest",
         "test_post_gwas_fdr_apptest", "test_haplotype_assignments_apptest",
         "test_plot_display_guard", "test_show_plot", "test_tab_settings_panel",
         "test_gui_cache_keys", "test_gui_clear", "test_gui_inputs", "test_gui_messages",
         "test_gui_run_keys", "test_guigwas", "test_guipost", "test_guitable",
         "test_post_gwas_haplotype_key", "test_post_gwas_r2_key", "test_post_gwas_run_identity"]),
    ("Large datasets (genotype table on disk)", "the table file, its ingest and reads, the memory plan, the models and the Post-GWAS steps on disk, memory and disk giving the same results",
        ["test_genostore", "test_genostore_ingest", "test_storepipe", "test_storeplan",
         "test_store_models", "test_store_post_gwas", "test_store_shared_use",
         "test_slow_store_equivalence", "test_cli_geno_store", "test_compare_result_zips",
         "test_post_gwas_closed_reads"]),
    ("Pipeline integration", "end-to-end GWAS pipeline and stage wiring",
        ["test_gwas_integration", "test_pipeline_stages", "test_trait_pipeline"]),
    ("Golden regression & pinned defaults", "byte-stable golden fixtures, the golden lock, and pinned signatures/defaults",
        ["test_golden_harness", "test_golden_blocks", "test_golden_published",
         "test_golden_lock", "test_defaults_pinned", "test_golden_qc_copy"]),
    ("Calibration & reproducibility", "null-phenotype calibration, LOCO reproducibility and the thread counts",
        ["test_null_calibration", "test_loco_reproducibility", "test_thread_provenance",
         "test_subsampling_worker_threads", "test_gui_thread_limits"]),
    ("Utilities", "shared helpers and the data behind the GUI's figures",
        ["test_utils", "test_plot_data", "test_offline_html"]),
]

_NODE_RE = re.compile(r"^(?:.*[/\\])?tests[/\\](test_[a-z0-9_]+)\.py::")
_SUMMARY_RE = re.compile(r"(\d+)\s+tests?\s+collected")


def collect_counts() -> tuple[dict[str, int], int]:
    """Return (per-file collected-test counts, authoritative collected total)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=_ROOT, capture_output=True, text=True,
    )
    out = proc.stdout
    counts: dict[str, int] = {}
    for line in out.splitlines():
        m = _NODE_RE.match(line.strip())
        if m:
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    if not counts:
        sys.exit(
            "gen_test_table: `pytest --collect-only -q` yielded no test node ids.\n"
            f"exit={proc.returncode}\n--- stdout tail ---\n{out[-2000:]}\n"
            f"--- stderr tail ---\n{proc.stderr[-2000:]}"
        )
    summ = _SUMMARY_RE.search(out)
    total = int(summ.group(1)) if summ else sum(counts.values())
    if total != sum(counts.values()):
        sys.exit(
            f"gen_test_table: collected summary ({total}) != sum of per-file counts "
            f"({sum(counts.values())}); collection output is not parseable as expected."
        )
    return counts, total


def coverage_pct(override: str | None) -> int:
    if override is not None:
        return round(float(override))
    xml = _ROOT / "coverage.xml"
    if xml.exists():
        rate = ET.parse(xml).getroot().get("line-rate")
        if rate is None:
            sys.exit("gen_test_table: coverage.xml has no line-rate attribute.")
        return round(float(rate) * 100)
    sys.exit(
        "gen_test_table: no coverage.xml and no --coverage PCT.\n"
        "Run: python -m pytest tests/ --cov=gwas --cov=utils --cov-report=xml -q"
    )


def build_block(counts: dict[str, int], total: int, pct: int, scope: str) -> str:
    mapped = [s for _, _, stems in AREAS for s in stems]
    mapped_set = set(mapped)
    if len(mapped) != len(mapped_set):
        dupes = sorted({s for s in mapped if mapped.count(s) > 1})
        sys.exit(f"gen_test_table: test file mapped to more than one area: {dupes}")

    unmapped = sorted(set(counts) - mapped_set)
    if unmapped:
        sys.exit(
            "gen_test_table: unmapped test file(s) -- add each to AREAS in "
            "scripts/gen_test_table.py:\n  " + "\n  ".join(unmapped)
        )

    rows = ["| Module | Tests | What it exercises |", "|--------|-------|-------------------|"]
    grouped_total = 0
    for label, desc, stems in AREAS:
        n = sum(counts.get(s, 0) for s in stems)
        grouped_total += n
        rows.append(f"| {label} | {n} | {desc} |")

    if grouped_total != total:
        sys.exit(
            f"gen_test_table: grouped row counts sum to {grouped_total} but the suite "
            f"collected {total}. Fix the AREAS mapping."
        )

    body = "\n".join(rows)
    footer = f"_Total: {total} tests. Line coverage: {pct}% ({scope})._"
    return f"{_BEGIN}\n\n{body}\n\n{footer}\n\n{_END}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate the README test-table block.")
    ap.add_argument("--coverage", default=None,
                    help="line-coverage percent to print (default: read coverage.xml)")
    ap.add_argument("--scope", default="gwas + utils",
                    help="coverage scope label (default: 'gwas + utils')")
    args = ap.parse_args()

    counts, total = collect_counts()
    pct = coverage_pct(args.coverage)
    block = build_block(counts, total, pct, args.scope)

    # Read with LF enforced (the block content is LF; normalise any stray CRLF so
    # the whole file is written back with Unix line endings).
    text = _README.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    pattern = re.compile(re.escape(_BEGIN) + r".*?" + re.escape(_END), re.DOTALL)
    if not pattern.search(text):
        sys.exit(f"gen_test_table: markers not found in {_README}")
    new_text = pattern.sub(lambda _m: block, text, count=1)
    _README.write_text(new_text, encoding="utf-8", newline="\n")

    print(f"gen_test_table: {total} tests across {len(counts)} files, "
          f"{len(AREAS)} rows, line coverage {pct}% ({args.scope}); wrote {_README}")


if __name__ == "__main__":
    main()
