"""Per-stage time and memory marks (a diagnostic for measurements; off by default).

With TRACE_STAGE_LOG=<csv file> in the environment, every stage boundary of the CLI and of
gwas.pipeline._run_trait writes one row: seconds since the first mark, the stage that starts,
the process working set and private bytes now, the operating system's peak counters (peak
working set, peak commit) and, between two marks, the largest working set and private bytes a
background sampler saw (every TRACE_STAGE_SAMPLE_MS milliseconds, default 20). A stage's peak
is exact when the OS peak counter rose while it ran; otherwise the sampler's value is a lower
bound. The on-disk table's read counters (bytes read, file reads, largest block) are recorded
too. Rows are flushed as they are written, so a killed run keeps its trail. Without the
variable nothing happens.
"""
import atexit
import os
import threading
import time

_S = {"fh": None, "t0": None, "proc": None, "max_rss": 0, "max_priv": 0, "lock": threading.Lock(),
      "stop": False, "path": None}
MB = 2.0 ** 20


def _mem():
    mi = _S["proc"].memory_info()
    return (mi.rss, getattr(mi, "private", getattr(mi, "vms", 0)), getattr(mi, "peak_wset", 0),
            getattr(mi, "peak_pagefile", 0))


def _sampler(period):
    while not _S["stop"]:
        try:
            rss, priv, _pw, _pp = _mem()
        except Exception:
            return
        with _S["lock"]:
            if rss > _S["max_rss"]:
                _S["max_rss"] = rss
            if priv > _S["max_priv"]:
                _S["max_priv"] = priv
        time.sleep(period)


def _open(path):
    import psutil

    _S["proc"] = psutil.Process()
    _S["t0"] = time.perf_counter()
    _S["path"] = path
    _S["fh"] = open(path, "w", encoding="utf-8")
    _S["fh"].write("t_s,stage,ws_mb,private_mb,os_peak_ws_mb,os_peak_commit_mb,sampled_max_ws_mb,"
                   "sampled_max_private_mb,store_bytes_read,store_file_reads,store_max_block_cells\n")
    period = float(os.environ.get("TRACE_STAGE_SAMPLE_MS", "20") or 20) / 1000.0
    if period > 0:
        threading.Thread(target=_sampler, args=(period,), daemon=True).start()
    atexit.register(lambda: mark("END"))


def mark(stage):
    """A stage boundary: ``stage`` starts now (no-op unless TRACE_STAGE_LOG is set)."""
    path = os.environ.get("TRACE_STAGE_LOG", "")
    if not path or _S["stop"]:
        return
    if _S["fh"] is None:
        _open(path)
    rss, priv, pw, pp = _mem()
    with _S["lock"]:
        mx_r, mx_p = max(_S["max_rss"], rss), max(_S["max_priv"], priv)
        _S["max_rss"], _S["max_priv"] = rss, priv
    try:
        from gwas import genostore

        rl = genostore._LOG
        rd = (rl["file_bytes"], rl["file_reads"], rl["max_block_cells"])
    except Exception:
        rd = (0, 0, 0)
    _S["fh"].write(f"{time.perf_counter() - _S['t0']:.4f},{stage},{rss / MB:.2f},{priv / MB:.2f},"
                   f"{pw / MB:.2f},{pp / MB:.2f},{mx_r / MB:.2f},{mx_p / MB:.2f},{rd[0]},{rd[1]},{rd[2]}\n")
    _S["fh"].flush()
    if stage == "END":
        _S["stop"] = True
        _S["fh"].close()
