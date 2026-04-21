"""Run GAPIT3 MLM on simulated phenotypes (batch).

Runs the R script on each scenario directory. Can be limited to a subset
for practical runtime.
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SIM_DIR = ROOT / "benchmarks" / "simulation" / "sim_data"
R_SCRIPT = ROOT / "benchmarks" / "simulation" / "run_simulation_gapit.R"
RSCRIPT_EXE = (
    shutil.which("Rscript")
    or r"C:\Program Files\R\R-4.5.2\bin\Rscript.exe"
)


def run_gapit_batch(max_scenarios=None, max_reps=None):
    """Run GAPIT3 on simulation scenarios.

    Args:
        max_scenarios: Limit number of scenarios (None = all 9)
        max_reps: Limit reps per scenario (default 5 for practical runtime)
    """
    scenarios = sorted([d for d in SIM_DIR.iterdir()
                       if d.is_dir() and d.name.startswith("h2_")])

    if max_scenarios:
        scenarios = scenarios[:max_scenarios]

    total = 0
    success = 0
    failed = 0
    start = time.time()

    for scenario_dir in scenarios:
        reps = sorted([d for d in scenario_dir.iterdir()
                      if d.is_dir() and d.name.startswith("rep_")])
        reps = reps[:max_reps]

        for rep_dir in reps:
            total += 1
            gapit_dir = rep_dir / "gapit"

            # Skip if already done
            if (gapit_dir / "results.csv").exists():
                success += 1
                continue

            try:
                result = subprocess.run(
                    [RSCRIPT_EXE, str(R_SCRIPT), str(rep_dir)],
                    capture_output=True, text=True, timeout=120,
                    cwd=str(ROOT),
                )

                if (gapit_dir / "results.csv").exists():
                    success += 1
                elif (gapit_dir / "error.txt").exists():
                    failed += 1
                    err = (gapit_dir / "error.txt").read_text().strip()
                    print(f"  GAPIT error in {rep_dir.name}: {err[:100]}")
                else:
                    failed += 1
                    print(f"  No output for {rep_dir.name}")

            except subprocess.TimeoutExpired:
                failed += 1
                print(f"  Timeout for {rep_dir.name}")
            except Exception as e:
                failed += 1
                print(f"  Exception for {rep_dir.name}: {e}")

            if total % 5 == 0:
                elapsed = time.time() - start
                print(f"  GAPIT progress: {total} runs "
                      f"({success} ok, {failed} fail, {elapsed:.0f}s)")

    elapsed = time.time() - start
    print(f"\nGAPIT batch complete: {success}/{total} successful "
          f"in {elapsed:.0f}s ({failed} failed)")
    return success, total


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--max-reps", type=int, default=None)
    args = parser.parse_args()

    run_gapit_batch(
        max_scenarios=args.max_scenarios,
        max_reps=args.max_reps,
    )
