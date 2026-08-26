# -*- coding: utf-8 -*-
"""Whole-pipeline peak RSS of a full cli.py run (R1.12 / R3.3) -- MEMORY point, not runtime.

The published ~430 MB (docs/revision/measurements/T88_scalability.md) covers ONLY the three
post-GWAS stages (LD blocks + haplotype + annotation). This measures the WHOLE pipeline peak
and attributes it to stage GROUPS via a decomposition of flag-gated cli.py runs, each spawned
as a subprocess with the process-tree RSS polled on a daemon thread (parent + joblib/fastlmm
children):

  A  MLM only, no auto-PC, no plots/report   -> baseline (VCF load + QC + GRM/LOCO + MLM)
  B  MLM + auto-PC, no plots/report          -> + auto-PC scan
  C  3 models + auto-PC, no plots/report     -> full ANALYSIS (all S9b compute stages)
  D  3 models + auto-PC + plots + report      -> full CLI run
  E  D + subsampling (100 LOCO resamples)     -> + subsampling
Peak deltas attribute memory: (C-A)=extra models, (D-C)=plots+report, (E-D)=subsampling.

Why a group decomposition rather than per-S9b-stage RSS: cli.py logs each stage to stderr, but
streamlit's cache manager RECONFIGURES logging mid-run (after auto-PC the "[INFO] cli:" markers
stop reaching stderr and are replaced by streamlit's "No runtime found ..." handler), so parsing
stderr for the 9 individual stage boundaries is unreliable. The peak RSS itself is always valid
(the poll thread is independent of logging). The group decomposition is the robust attribution.

--memory-point mode (M4): one lean MLM+LOCO run on an arbitrary VCF + a geno/kernel decomposition.

Run from TRACE-release (shipping tree). The raw tomato VCF + phenotype are NOT committed -- pass
their paths (e.g. a scratch copy) via --vcf / --pheno.
"""
import argparse
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent  # TRACE-release root
POLL_S = 0.05  # RSS sampling interval (s)


def _tree_rss_mb(proc):
    """RSS of the process tree (parent + all children) in MB."""
    try:
        total = proc.memory_info().rss
    except psutil.Error:
        return float("nan")
    try:
        for c in proc.children(recursive=True):
            try:
                total += c.memory_info().rss
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    return total / 1e6


def run_once(cli_args, label):
    """Run cli.py once; return its peak process-tree RSS (MB) + wall time."""
    t0 = time.perf_counter()
    p = subprocess.Popen(
        [sys.executable, "-u", "cli.py", *cli_args],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(ROOT),
    )
    ps = psutil.Process(p.pid)
    peak = [0.0]
    stop = threading.Event()

    def _poll():
        while not stop.is_set():
            r = _tree_rss_mb(ps)
            if r == r and r > peak[0]:
                peak[0] = r
            time.sleep(POLL_S)

    th = threading.Thread(target=_poll, daemon=True)
    th.start()
    p.wait()
    stop.set()
    th.join(timeout=1.0)
    return dict(label=label, rc=p.returncode, wall=time.perf_counter() - t0, peak_rss=peak[0])


def _provenance():
    def _sha():
        try:
            return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                  capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
        except Exception:
            return "unknown"
    print("=" * 96)
    print("  Whole-pipeline peak RSS (MEMORY point -- not a runtime/scalability benchmark)")
    print("=" * 96)
    print(f"  machine  : {platform.processor() or platform.machine()} | "
          f"{psutil.cpu_count(logical=False)} physical / {os.cpu_count()} logical | "
          f"RAM {psutil.virtual_memory().total / 1e9:.1f} GB | {platform.system()} {platform.release()}")
    print(f"  tree     : TRACE-release, git HEAD = {_sha()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vcf", required=True)
    ap.add_argument("--pheno", required=True)
    ap.add_argument("--trait", default="locule_number")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--boot-reps", type=int, default=100, help="resamples for run E (published S9b=100)")
    ap.add_argument("--memory-point", action="store_true",
                    help="M4 mode: ONE lean MLM+LOCO run + geno/kernel decomposition")
    ap.add_argument("--n", type=int, help="samples (memory-point decomposition)")
    ap.add_argument("--m", type=int, help="SNPs (memory-point decomposition)")
    ap.add_argument("--n-chroms", type=int, default=12, help="chromosomes (kernel-term decomposition)")
    args = ap.parse_args()

    inp = ["--vcf", args.vcf, "--pheno", args.pheno, "--trait", args.trait]
    _provenance()

    if args.memory_point:
        lean = inp + ["--model", "mlm", "--no-plots", "--no-report",
                      "--output", str(Path(args.out_dir) / "mempoint")]
        r = run_once(lean, f"MEMORY POINT n={args.n} m={args.m} ({args.n_chroms} chroms)")
        print(f"\n  {r['label']}  (rc={r['rc']}, wall={r['wall']:.1f}s)")
        print(f"  measured peak RSS = {r['peak_rss']:.0f} MB")
        if args.n and args.m:
            print(f"  predicted geno matrix  (n*m*8)            = {args.n * args.m * 8 / 1e6:.0f} MB")
            print(f"  predicted LOCO kernels ({args.n_chroms}*n^2*4, float32) = {args.n_chroms * args.n * args.n * 4 / 1e6:.0f} MB")
        print("  (memory point -- NOT a runtime/scalability benchmark)")
        return

    full = ["--model", "mlm", "mlmm", "farmcpu", "--auto-pcs"]
    variants = [
        ("A: MLM only, no auto-PC, no plots/report", ["--model", "mlm", "--no-plots", "--no-report"]),
        ("B: MLM + auto-PC, no plots/report",        ["--model", "mlm", "--auto-pcs", "--no-plots", "--no-report"]),
        ("C: 3 models + auto-PC (full ANALYSIS), no plots/report", full + ["--no-plots", "--no-report"]),
        ("D: full CLI run (3 models + auto-PC + plots + report)", full),
        (f"E: D + subsampling ({args.boot_reps} LOCO resamples)", full + ["--subsampling", "--boot-reps", str(args.boot_reps), "--boot-jobs", "1"]),
    ]
    res = {}
    print(f"\n  {'variant':<58}{'peak(MB)':>10}{'wall(s)':>9}")
    for i, (label, extra) in enumerate(variants):
        r = run_once(inp + extra + ["--output", str(Path(args.out_dir) / f"v{i}")], label)
        res[label[0]] = r["peak_rss"]
        print(f"  {label:<58}{r['peak_rss']:>10.0f}{r['wall']:>9.1f}")

    print(f"\n  === attribution (peak deltas) ===")
    print(f"  baseline (VCF+QC+GRM/LOCO+MLM)      A          = {res['A']:.0f} MB")
    print(f"  extra models (MLMM+FarmCPU)         C - A      = {res['C'] - res['A']:+.0f} MB")
    print(f"  plots + report                      D - C      = {res['D'] - res['C']:+.0f} MB   <-- driver")
    print(f"  subsampling                         E - D      = {res['E'] - res['D']:+.0f} MB")
    print(f"  full analysis peak (no plots)       C          = {res['C']:.0f} MB")
    print(f"  WHOLE-PIPELINE peak, no subsampling  D          = {res['D']:.0f} MB")
    print(f"  WHOLE-PIPELINE peak, with subsampling E         = {res['E']:.0f} MB")
    print(f"  published post-GWAS-only peak (T88_scalability.md) = 430 MB")


if __name__ == "__main__":
    main()
