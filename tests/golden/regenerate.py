"""Guarded regeneration driver for the golden block-table fixtures (T-95 §1.4).

A regeneration path that is easy defeats the goldens, so the CLI here refuses to
run unless the operator has clearly declared intent: all of ``--case``,
``--reason`` (>= 30 chars), ``--i-am-changing-pinned-behaviour``, and the
environment variable ``TRACE_GOLDEN_REGEN=1``; and it refuses on a dirty tree or
detached HEAD so the ``trace_commit`` recorded in ``meta.json`` is meaningful.

Every deliberate regeneration rewrites ``expected_*.csv``, ``meta.json`` and
``GOLDEN_LOCK``, and appends one entry to ``REGENERATION_LOG.md`` — the file whose
only purpose is to be noticed in code review.

The regeneration *logic* is exposed as importable functions (``regen_case``,
``rewrite_lock``) so the initial fixtures can be bootstrapped without the
interactive guards; the guards live only in the ``__main__`` CLI.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _canon  # noqa: E402  (sibling module, script-dir on sys.path)
import _cases  # noqa: E402

LOG = HERE / "REGENERATION_LOG.md"


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=str(REPO)).decode().strip()


def _now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_input_npz(case):
    inp = _cases.build_case_input(case)
    d = HERE / case
    d.mkdir(exist_ok=True)
    np.savez_compressed(d / "input.npz", **inp)
    return d / "input.npz"


def capture_expected(case, params):
    """Run the *current* detector on the case's input and return
    ``(blocks_df, filtered_df)`` — the regression contract is whatever the code
    emits today, never a hand-derived table."""
    import pandas as pd
    from gwas.ld import filter_contained_blocks, find_ld_clusters_genomewide

    inp = dict(np.load(HERE / case / "input.npz", allow_pickle=False))
    gwas_df = pd.DataFrame({
        "SNP": inp["sid"].astype(str),
        "Chr": inp["chroms"].astype(str),
        "Pos": inp["positions"].astype(int),
        "PValue": inp["pvalue"].astype(float),
    })
    blocks = find_ld_clusters_genomewide(
        gwas_df=gwas_df,
        chroms=inp["chroms"], positions=inp["positions"],
        geno_imputed=inp["geno"].astype(float), sid=inp["sid"],
        ld_threshold=params["ld_threshold"], flank_kb=params["flank_kb"],
        min_snps=params["min_snps"], top_n=params["top_n"],
        sig_thresh=params["sig_thresh"], adj_r2_min=params["adj_r2_min"],
        min_pair_n=params["min_pair_n"], merge_iou=params["merge_iou"],
        gap_factor=params["gap_factor"],
    )
    filtered, _n_removed = filter_contained_blocks(blocks.copy(), min_contained=2)
    return blocks, filtered


def regen_case(case, reason, trace_commit=None):
    """(Re)build one case: input.npz, expected_*.csv (canonicalised), meta.json.
    Returns a small diff summary for the log."""
    params = dict(_cases.STD_PARAMS)
    npz = write_input_npz(case)
    blocks, filtered = capture_expected(case, params)
    d = HERE / case
    (d / "expected_blocks.csv").write_text(_canon.canon_block_table(blocks), encoding="utf-8")
    (d / "expected_blocks_filtered.csv").write_text(_canon.canon_block_table(filtered), encoding="utf-8")

    meta = {
        "case": case,
        "tier": "A",
        "generator_seed": _cases.CASE_SEED[case],
        "input_sha256": _canon.sha256_file(npz),
        "trace_commit": trace_commit or _git("rev-parse", "HEAD"),
        "generated_utc": _now_utc(),
        "reason": reason,
        "params": params,
        "comparison": {"mode": "exact", "float_tolerance": None},
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {"case": case, "n_blocks": int(len(blocks)), "n_filtered": int(len(filtered))}


def rewrite_lock():
    (HERE / "GOLDEN_LOCK").write_text(_canon.golden_lock_digest() + "\n", encoding="utf-8")


def append_log(summary, reason, commit):
    header = "" if LOG.exists() else (
        "# Golden regeneration log\n\n"
        "Append-only. One entry per deliberate regeneration of a pinned golden.\n\n"
    )
    entry = (
        f"## {_now_utc()} — {summary['case']}\n"
        f"- commit: `{commit}`\n"
        f"- blocks: {summary['n_blocks']} (filtered: {summary['n_filtered']})\n"
        f"- reason: {reason}\n\n"
    )
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(header + entry)


# ---------------------------------------------------------------------------
# guarded CLI
# ---------------------------------------------------------------------------
def _main(argv=None):
    assert "PYTEST_CURRENT_TEST" not in os.environ, "regenerate.py must not run under pytest"

    ap = argparse.ArgumentParser(description="Deliberately regenerate a pinned golden.")
    ap.add_argument("--case", required=True, choices=_cases.TIER_A_CASES)
    ap.add_argument("--reason", required=True, help="why (>= 30 chars); recorded in the log")
    ap.add_argument("--i-am-changing-pinned-behaviour", action="store_true")
    args = ap.parse_args(argv)

    if not args.i_am_changing_pinned_behaviour:
        sys.exit("refusing: pass --i-am-changing-pinned-behaviour to confirm intent")
    if os.environ.get("TRACE_GOLDEN_REGEN") != "1":
        sys.exit("refusing: set TRACE_GOLDEN_REGEN=1 in the environment")
    if len(args.reason.strip()) < 30:
        sys.exit("refusing: --reason must be at least 30 characters")
    if _git("status", "--porcelain").strip():
        sys.exit("refusing: working tree is dirty — commit or stash first")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":
        sys.exit("refusing: detached HEAD — check out a branch first")

    commit = _git("rev-parse", "HEAD")
    summary = regen_case(args.case, args.reason.strip(), trace_commit=commit)
    rewrite_lock()
    append_log(summary, args.reason.strip(), commit)
    print(
        f"REGENERATED {summary['case']}: {summary['n_blocks']} blocks "
        f"({summary['n_filtered']} after containment filter).\n"
        "Review the one-line GOLDEN_LOCK diff and the REGENERATION_LOG.md entry "
        "before committing — these are the artefacts reviewers read."
    )


if __name__ == "__main__":
    _main()
