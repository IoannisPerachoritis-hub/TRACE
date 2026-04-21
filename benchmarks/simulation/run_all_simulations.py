"""Run the full simulation benchmark end-to-end.

Usage:
    python benchmarks/simulation/run_all_simulations.py              # Full run
    python benchmarks/simulation/run_all_simulations.py --quick      # Quick test (5 reps, 10 perms)
    python benchmarks/simulation/run_all_simulations.py --phase sim  # Sim only
    python benchmarks/simulation/run_all_simulations.py --phase null # Null only
    python benchmarks/simulation/run_all_simulations.py --phase eval # Evaluate only
    python benchmarks/simulation/run_all_simulations.py --phase plot # Plot only
    python benchmarks/simulation/run_all_simulations.py --phase r    # R tools only
"""
import argparse
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SIM_DIR = ROOT / "benchmarks" / "simulation"


def run_cmd(cmd, description, timeout=None):
    """Run a command with logging."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd, cwd=str(ROOT), timeout=timeout,
        capture_output=False,
    )
    elapsed = time.perf_counter() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (code={result.returncode})"
    print(f"  [{status}] {elapsed:.1f}s")
    return result.returncode == 0


def phase_generate(n_reps=50, n_perms=100, seed=2026):
    """Phase 1: Generate simulated and null phenotypes."""
    return run_cmd(
        [sys.executable, str(SIM_DIR / "simulate_phenotypes.py"),
         "--mode", "both", "--n-reps", str(n_reps),
         "--n-perms", str(n_perms), "--seed", str(seed)],
        f"Generating {n_reps} reps x 9 scenarios + {n_perms} null phenotypes",
    )


def phase_platform(mode="both", n_pcs=2):
    """Phase 2: Run TRACE on all simulated + null phenotypes."""
    return run_cmd(
        [sys.executable, str(SIM_DIR / "run_simulation_platform.py"),
         "--mode", mode, "--n-pcs", str(n_pcs)],
        f"Running TRACE MLM ({mode}) on simulated phenotypes",
        timeout=72000,  # 20 hours max
    )


def _run_single_r_tool(tool_name, r_script, scenario_dir, n_pcs):
    """Run a single R tool on one scenario. Returns (tool, scenario, ok, elapsed, error)."""
    rscript_exe = shutil.which("Rscript") or "Rscript"
    out_dir = scenario_dir / tool_name
    result_file = out_dir / "results.csv"

    # Skip if already done
    if result_file.exists():
        return (tool_name, str(scenario_dir), True, 0.0, "skipped")

    t0 = time.perf_counter()
    try:
        subprocess.run(
            [rscript_exe, str(r_script), str(scenario_dir), str(n_pcs)],
            capture_output=True, text=True, timeout=300,
            cwd=str(ROOT),
        )
        elapsed = time.perf_counter() - t0

        if result_file.exists():
            return (tool_name, str(scenario_dir), True, elapsed, None)

        # Check for error file
        err_file = out_dir / "error.txt"
        err_msg = err_file.read_text().strip()[:200] if err_file.exists() else "no output"
        return (tool_name, str(scenario_dir), False, elapsed, err_msg)

    except subprocess.TimeoutExpired:
        return (tool_name, str(scenario_dir), False, 300.0, "timeout")
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return (tool_name, str(scenario_dir), False, elapsed, str(e)[:200])


def phase_r_tools(n_pcs=2, n_workers=4):
    """Phase 3: Run GAPIT3 and rMVP on all simulated phenotypes (parallel)."""
    sim_data = SIM_DIR / "sim_data"
    scenario_dirs = sorted([
        d for d in sim_data.glob("h2_*/rep_*")
        if d.is_dir() and (d / "phenotype.csv").exists()
    ])

    print(f"\n  Found {len(scenario_dirs)} scenario/rep combos for R tools")
    print(f"  Using {n_workers} parallel workers")

    gapit_script = SIM_DIR / "run_simulation_gapit.R"
    rmvp_script = SIM_DIR / "run_simulation_rmvp.R"
    gapit_farmcpu_script = SIM_DIR / "run_simulation_gapit_farmcpu.R"

    # Build work items: (tool_name, r_script, scenario_dir, n_pcs)
    work = []
    for sd in scenario_dirs:
        work.append(("gapit", gapit_script, sd, n_pcs))
        work.append(("rmvp", rmvp_script, sd, n_pcs))
        work.append(("gapit_farmcpu", gapit_farmcpu_script, sd, n_pcs))

    total = len(work)
    done = 0
    ok_count = {"gapit": 0, "rmvp": 0, "gapit_farmcpu": 0}
    fail_count = {"gapit": 0, "rmvp": 0, "gapit_farmcpu": 0}
    failures = []
    t_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_run_single_r_tool, tool, script, sd, pcs): (tool, sd)
            for tool, script, sd, pcs in work
        }

        for future in as_completed(futures):
            tool_name, scenario_str, success, elapsed, error = future.result()
            done += 1

            if success:
                ok_count[tool_name] += 1
            else:
                fail_count[tool_name] += 1
                sd_path = Path(scenario_str)
                failures.append(
                    f"  {tool_name} {sd_path.parent.name}/{sd_path.name}: {error}"
                )

            if done % 50 == 0 or done == total:
                wall = time.perf_counter() - t_start
                print(f"  Progress: {done}/{total} "
                      f"(GAPIT {ok_count['gapit']}ok/{fail_count['gapit']}fail, "
                      f"rMVP {ok_count['rmvp']}ok/{fail_count['rmvp']}fail, "
                      f"GAPIT-FC {ok_count['gapit_farmcpu']}ok/"
                      f"{fail_count['gapit_farmcpu']}fail) "
                      f"[{wall:.0f}s]")

    n_scenarios = len(scenario_dirs)
    print(f"\n  R tools complete: "
          f"GAPIT3 {ok_count['gapit']}/{n_scenarios}, "
          f"rMVP {ok_count['rmvp']}/{n_scenarios}, "
          f"GAPIT3-FC {ok_count['gapit_farmcpu']}/{n_scenarios}")

    if failures:
        print(f"\n  {len(failures)} failures:")
        for f in failures[:20]:
            print(f)
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")


def phase_evaluate(mode="both"):
    """Phase 4: Evaluate simulation results."""
    return run_cmd(
        [sys.executable, str(SIM_DIR / "evaluate_simulation.py"),
         "--mode", mode],
        "Evaluating simulation and null results",
    )


def phase_plot():
    """Phase 5: Generate plots."""
    return run_cmd(
        [sys.executable, str(SIM_DIR / "plot_simulation.py")],
        "Generating simulation plots",
    )


def phase_scatter():
    """Phase 6: Generate p-value scatter plots (existing real data)."""
    return run_cmd(
        [sys.executable, str(ROOT / "benchmarks" / "plots" / "pvalue_scatter.py")],
        "Generating p-value scatter plots",
    )


def main():
    parser = argparse.ArgumentParser(description="Run full simulation benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 5 reps, 10 perms")
    parser.add_argument("--phase",
                        choices=["gen", "platform", "r", "eval", "plot",
                                 "scatter", "sim", "null", "all"],
                        default="all", help="Run specific phase only")
    parser.add_argument("--n-reps", type=int, default=100)
    parser.add_argument("--n-perms", type=int, default=100)
    parser.add_argument("--n-pcs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-workers", type=int, default=4,
                        help="Parallel workers for R tools (default 4)")
    args = parser.parse_args()

    if args.quick:
        args.n_reps = 5
        args.n_perms = 10

    t_total = time.perf_counter()

    if args.phase in ("gen", "all"):
        phase_generate(args.n_reps, args.n_perms, args.seed)

    if args.phase in ("platform", "sim", "all"):
        phase_platform("sim", args.n_pcs)

    if args.phase in ("platform", "null", "all"):
        phase_platform("null", args.n_pcs)

    if args.phase in ("r", "all"):
        phase_r_tools(args.n_pcs, n_workers=args.n_workers)

    if args.phase in ("eval", "all"):
        phase_evaluate()

    if args.phase in ("plot", "all"):
        phase_plot()

    if args.phase in ("scatter", "all"):
        phase_scatter()

    elapsed = time.perf_counter() - t_total
    print(f"\n{'='*60}")
    print(f"  TOTAL ELAPSED: {elapsed/3600:.1f} hours ({elapsed:.0f}s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
