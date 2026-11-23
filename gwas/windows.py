"""Default windows shared by the CLI and the GUI.

No block-detector import (gwas.isolated uses it, and must never import gwas.ld).
"""
import math

LD_WINDOW_NO_DECAY_KB = 300   # the window when the LD decay cannot be estimated


def ld_window_kb(ld_decay_kb):
    """The LD window (kb) around each lead SNP, by the CLI's rule: int(2 x the median LD
    decay), or 300 kb when the decay could not be estimated (None or not finite). It sets
    the candidate intervals' edge flank, MLMM's cofactor window and the Regional Plot and
    Local LD windows. Default block detection uses twice it (block_window_kb)."""
    if ld_decay_kb is None:
        return LD_WINDOW_NO_DECAY_KB
    x = float(ld_decay_kb)
    if not math.isfinite(x):
        return LD_WINDOW_NO_DECAY_KB
    return int(2 * x)


def block_window_kb(ld_decay_kb):
    """Default LD-block window (kb): twice the LD window, 2 x int(2 x decay); 300 kb
    when the decay could not be estimated. Block detection only: edge flanks, MLMM's
    cofactor window and the plot windows keep ld_window_kb."""
    w = ld_window_kb(ld_decay_kb)
    if ld_decay_kb is None or not math.isfinite(float(ld_decay_kb)):
        return w
    return 2 * w
