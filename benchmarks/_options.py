"""Options shared by the benchmark scripts.

Each script takes the options it uses out of its command line before reading its own:

    --data-root DIR    the folder that holds benchmarks/qc_data and benchmarks/results (default:
                       this repository). A side analysis runs in a folder of its own, with copies
                       of the QC data it needs; benchmarks/README.md says which.
    --k-set band       TRACE's PC counts from the strict band rule on the corrected data
                       (published_pc_counts.BAND_RULE_K) instead of the published counts.
    --rmvp-pcs-twice   read the rMVP results made with the PCs passed twice (run_rmvp.R with
                       pcs_twice), as in the published runs, and add "_pcs_twice" to the names of
                       the files written.

The values are read once per process, so a script and the scripts it imports agree.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
_PARSED = None


def _take_value(argv, flag, default):
    if flag not in argv:
        return default
    i = argv.index(flag)
    if i + 1 >= len(argv):
        raise SystemExit(f"{flag} needs a value")
    value = argv[i + 1]
    del argv[i:i + 2]
    return value


def options():
    """The shared options: root (the data root), k_set ("published" or "band"), rmvp (the rMVP
    results folder) and suffix ("" or "_pcs_twice", for the names of the files written)."""
    global _PARSED
    if _PARSED is None:
        argv = sys.argv
        root = Path(_take_value(argv, "--data-root", str(REPO))).resolve()
        k_set = _take_value(argv, "--k-set", "published")
        if k_set not in ("published", "band"):
            raise SystemExit("--k-set is published or band")
        twice = "--rmvp-pcs-twice" in argv
        if twice:
            argv.remove("--rmvp-pcs-twice")
        _PARSED = SimpleNamespace(root=root, k_set=k_set,
                                  rmvp="rmvp_pcs_twice" if twice else "rmvp",
                                  suffix="_pcs_twice" if twice else "")
    return _PARSED


def runs_in_root(runs):
    """The runs of ``runs`` (a dict keyed by run name) whose QC folder is in the data root."""
    qc = options().root / "benchmarks" / "qc_data"
    return {name: value for name, value in runs.items() if (qc / name).is_dir()}
