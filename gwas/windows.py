"""Default windows shared by the CLI and the GUI.

No block-detector import (gwas.isolated uses it, and must never import gwas.ld).
"""
import math

LD_WINDOW_NO_DECAY_KB = 300   # the window when the LD decay cannot be estimated


def ld_window_kb(ld_decay_kb):
    """The LD window (kb) around each lead SNP, by the CLI's rule: int(2 x the median LD
    decay), or 300 kb when the decay could not be estimated (None or not finite). Every
    default window uses it: the CLI's and One-Click's block detection, the GUI's
    Settings default, its auto-built blocks, edge flanks and the Regional Plot."""
    if ld_decay_kb is None:
        return LD_WINDOW_NO_DECAY_KB
    x = float(ld_decay_kb)
    if not math.isfinite(x):
        return LD_WINDOW_NO_DECAY_KB
    return int(2 * x)
