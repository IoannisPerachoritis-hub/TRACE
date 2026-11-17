"""The Post-GWAS page's genotype table, read in windows and blocks when they are needed.

The page and its tabs never hold a whole copy of the trait's table. They read it through a
PostTable, built from the session's gwas.guitable.SessionTable:

    ld          the mean-imputed table as the LD code reads it: today's
                ``np.asarray(geno_imputed, float)``. On disk it is the view's LazyColumns in
                float64; in memory an ArrayColumns over the GWAS page's own array. Either gives
                every window as the float64 array the page's whole copy gave, with the same
                values and layout, and neither reads the table whole.
    ld_aligned  its phenotype-aligned rows: today's ``geno_ld[keep_mask, :]`` (C order).
    hard        hard calls, read a column or a block at a time: today's ``np.rint(geno_ld)``
                for dosage012 genotypes, else a copy of the dosages.
    raw         the raw dosage table (missing calls NaN), read the same way.
    view        the GWAS page's table as it is (the haplotype PCA panel's fallback reads it
                in memory).
    samples     a frame holding only the sample index: the shared functions read nothing
                but the index of the genotype frame they are given.

ArrayColumns    an in-memory array read the way gwas.genostore.LazyColumns reads the table on
                disk: each read returns the block today's whole-array expression gave (number
                format, layout, and a strided view where today's slice of a C-ordered array
                was strided). In memory the page therefore takes the same code path as on
                disk, without the float64 copies.
HardCalls       hard calls of a table, one column or block at a time.

Keys and caches: every Post-GWAS cache names the table's identity (SessionTable.identity) and
the window, and keeps a fixed number of entries (r2_key, haplotype_key, detection_key,
remember, keep_recent, remember_family).

A table file closed after the GWAS page left it in the session (Clear session data, or the
server's registry making room) ends the page with a plain message (RELEASED_MESSAGE,
table_released, released_guard).

The GWAS results and M_eff in the session are taken only for the table they were computed on
(run_table_changed, meff_for_table: the identities the GWAS page records with them), and the
results only for the trait of the last finished Run GWAS (run_trait); otherwise the page says
what happened (results_message: the last Run GWAS did not finish, or the settings changed since
it). The PCs come only from what the GWAS page writes with the table (run_pcs, pcs_of_samples),
the user covariates likewise (run_covariates), and the kinship-aware haplotype test's kernels
only from the record the GWAS page's Run writes with its results (run_kernels). The page
records which trait and table its final LD blocks are of (BLOCKS_FINAL_KEY), for the GWAS
page's block stability.

SharedDetection: the page's automatic LD blocks and the LD Blocks tab's detection share one
result when their inputs are provably the same (the tab at its defaults, every sample
phenotype-aligned), so a first visit detects once. sigtable_key: the Significant SNPs tab
builds its table once per input and serves reruns from a bounded session cache.

POST_GWAS_SESSION_KEYS / POST_GWAS_SESSION_PREFIXES list every session key the page and its
tabs write (widget keys excluded), for Clear (gwas.guitable).
"""
import contextlib
import hashlib
import inspect
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gwas.genostore import LazyColumns, _note_block, is_lazy, skip_in_call_site

# ------------------------------------------------------------------- reading the table
def _layout_of(a):
    """The layout numpy keeps for ``a`` under np.asarray(a, dtype): F for an F-ordered array,
    else C (a one-row or one-column array is both)."""
    return "F" if (a.flags.f_contiguous and not a.flags.c_contiguous) else "C"


class _ArrayShape:
    """The two numbers gwas.genostore.LazyColumns asks its store for (N samples, M SNPs)."""

    def __init__(self, n, m):
        self.N, self.M = int(n), int(m)


class ArrayColumns(LazyColumns):
    """An in-memory samples x SNPs array, read like a gwas.genostore.LazyColumns.

    ``arr`` is kept as it is (never copied whole). ``dtype`` is the number format blocks come
    back in (the array's own by default; float64 stands in for ``np.asarray(arr, float)``), and
    ``layout`` the layout that whole-array expression would have (the array's own). Row and
    column selections stay lazy (lazy_rows, lazy_cols, ``[rows, :]``), and every read returns a
    new array with numpy's layout for the same expression on the whole array: ``[:, index]``
    F-ordered, a column slice of a C-ordered table strided, a single column 1-D. So the code
    that receives a block does today's arithmetic on today's operand (scaling audit, section
    8.6), and a whole-table read happens only if someone asks for it (np.asarray), logged and
    refusable as on disk (TRACE_STORE_FORBID_MATERIALIZE). ``kind`` says which table the array
    is, "imputed" or "raw" (the in-memory array has no fill to tell them apart, as a table on
    disk does), or None when not given."""

    def __init__(self, arr, dtype=None, layout=None, kind=None):
        a = arr if isinstance(arr, np.ndarray) else np.asarray(arr)
        if a.ndim != 2:
            raise ValueError("ArrayColumns takes a samples x SNPs array")
        self.arr = a
        self.store = _ArrayShape(a.shape[0], a.shape[1])
        self.cols = np.arange(a.shape[1], dtype=np.int64)
        self.rows = None
        self.fill = None
        self.dtype = np.dtype(a.dtype if dtype is None else dtype)
        self.layout = layout or _layout_of(a)
        self.std = None
        self.kind = kind

    def __repr__(self):
        return f"ArrayColumns(shape={self.shape}, dtype={self.dtype}, layout={self.layout})"

    def _derive(self, **kw):
        # as LazyColumns._derive, keeping this class (its block reads the array)
        new = type(self).__new__(type(self))
        new.__dict__.update(self.__dict__)
        new.__dict__.update(kw)
        return new

    def standardized(self, block=2000):
        raise NotImplementedError("ArrayColumns does not standardise (the Post-GWAS page never asks)")

    def block(self, sel, layout=None, _whole=False, _strided=False):
        """ndarray (n x k): the local columns ``sel`` of this table in its dtype, laid out as
        LazyColumns.block lays them out (``layout`` F or C, default this object's layout;
        ``_strided``: a C block as a view of a buffer one column wider)."""
        sel = self._col_index(sel)
        w = self.arr[:, self.cols[sel]]              # the window's columns, all rows
        if self.rows is not None:
            w = w[self.rows, :]
        lay = layout or self.layout
        if lay == "C" and _strided:
            buf = np.empty((w.shape[0], w.shape[1] + 1), dtype=self.dtype, order="C")
            buf[:, :w.shape[1]] = w
            out = buf[:, :w.shape[1]]
        else:
            out = np.empty(w.shape, dtype=self.dtype, order="C" if lay == "C" else "F")
            out[...] = w
        _note_block(out.shape, whole=_whole, table_cols=self.shape[1])
        return out


# the read log names who asked for a block, as for a table on disk, not ArrayColumns itself
skip_in_call_site(*(f for f in vars(ArrayColumns).values() if inspect.isfunction(f)))


def reader(table, dtype=None, kind=None):
    """The table read in windows: a LazyColumns as it is (in ``dtype`` if given), an array
    wrapped in an ArrayColumns of that ``kind`` ("imputed" or "raw"); None stays None."""
    if table is None:
        return None
    if is_lazy(table):
        return table if dtype is None else table.astype(dtype)
    return ArrayColumns(table, dtype=dtype, kind=kind)


def table_kind(table):
    """Which table a reader reads: "imputed" or "raw" (an ArrayColumns says so itself; a
    LazyColumns on disk is imputed when it has a fill), or "unknown"."""
    if isinstance(table, ArrayColumns):
        return table.kind or "unknown"
    if is_lazy(table):
        return "imputed" if getattr(table, "fill", None) is not None else "raw"
    return "unknown"


class HardCalls:
    """Hard calls of an imputed table, read a column or a block at a time: today's
    ``np.rint(geno_ld)[key]`` for dosage012 genotypes, else today's ``geno_ld.copy()[key]``
    (the same values: rint is elementwise, and the copy only changes the layout)."""

    ndim = 2

    def __init__(self, table, geno_encoding="dosage012"):
        self.table = table
        self.geno_encoding = geno_encoding

    @property
    def shape(self):
        return tuple(int(x) for x in self.table.shape)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        part = np.asarray(self.table[key])
        return np.rint(part) if self.geno_encoding == "dosage012" else np.array(part, copy=True)


@dataclass
class PostTable:
    """What the Post-GWAS page and its tabs read the trait's table through (module docstring)."""

    identity: str
    mode: str
    ld: object
    ld_aligned: object
    hard: object
    raw: object
    view: object
    samples: pd.DataFrame
    keep_mask: np.ndarray

    @property
    def shape(self):
        return tuple(int(x) for x in self.ld.shape)

    @property
    def keep_key(self):
        return digest(self.keep_mask)


def sample_index(iid=None, geno_df=None, geno_index=None):
    """The trait's sample index, from the first source that has it: the table's own ``iid`` (its
    first column: the samples of the table's rows, the genotype frame's index as strings), else
    the genotype frame's index (the GWAS page's frame in memory, the store frame on disk), else
    the GWAS context's ``geno_index``. The table comes first: the GWAS page writes geno_df
    before the table, so a run of that page that stopped between the two leaves a geno_df that
    is not the table's."""
    if iid is not None:
        return pd.Index(np.asarray(iid)[:, 0])
    if isinstance(geno_df, pd.DataFrame):
        return geno_df.index
    if geno_index is not None:
        return pd.Index(np.asarray(geno_index))
    raise ValueError("The trait's sample IDs are not in the session. Run the GWAS page first.")


def post_table(session_table, keep_mask, geno_encoding="dosage012", samples=None):
    """The PostTable of a gwas.guitable.SessionTable: ``keep_mask`` the phenotype-aligned rows,
    ``geno_encoding`` the hard-call rule, ``samples`` the sample index (sample_index)."""
    view = session_table.view
    ld = reader(view, np.float64, kind="imputed")
    keep = np.asarray(keep_mask, dtype=bool)
    if samples is None:
        samples = sample_index(iid=session_table.iid)
    return PostTable(
        identity=str(session_table.identity), mode=str(session_table.mode),
        ld=ld, ld_aligned=ld.lazy_rows(keep), hard=HardCalls(ld, geno_encoding),
        raw=reader(session_table.raw, kind="raw"), view=view,
        samples=pd.DataFrame(index=samples), keep_mask=keep)


# ------------------------------------------------------------ a table released meanwhile
RELEASED_MESSAGE = ("This run's genotype table was released: Clear session data was used, or "
                    "another dataset took its place on this server. Run the GWAS page again.")


def table_released(session_table):
    """True when the session's table is a table file that was closed after the GWAS page left
    it in the session: Clear session data closes the server's tables, and the server's
    registry closes its least recently used table when another dataset needs the room
    (gwas.guitable.TableRegistry). A table in memory stays readable while the session holds
    it."""
    if session_table is None or str(getattr(session_table, "mode", "")) != "disk":
        return False
    for holder in (getattr(session_table, "parsed", None), getattr(session_table, "view", None)):
        store = getattr(holder, "store", None)
        if store is not None and bool(getattr(store, "closed", False)):
            return True
    return False


# ------------------------------------------- the GWAS page's results and the table they are of
# Session keys of the contract between the two pages (batch 13): the GWAS page writes the first
# two, this page the third.
RUN_IDENTITY = "gwas_run_identity"        # with gwas_results: the identity of its Run's table
MEFF_IDENTITY = "meff_identity"           # with meff_val: the identity of the table of M_eff
BLOCKS_FINAL_KEY = "ld_blocks_final_key"  # with ld_blocks_final: (trait, table identity)
# with RUN_IDENTITY: the kernels the Run's scan used, {identity, trait, use_loco, K0, K_by_chr}
# (references to the GWAS page's own kernel objects, not copies): the haplotype test's
RUN_KERNELS = "gwas_run_kernels"

# The haplotype test's choice in the GUI (the LD Blocks tab's Settings and One-Click): label ->
# run_haplotype_block_gwas's hap_test (the CLI's --hap-test); the first is the default
HAP_TEST_CHOICES = {
    "Kinship-aware (default)": "kinship",
    "PCs and covariates only (the earlier test)": "pcs",
}
HAP_TEST_HELP = (
    "Kinship-aware (the default; the CLI's --hap-test kinship): a mixed model with the GWAS "
    "run's own kinship kernel for the block's chromosome (its LOCO kernel, or the whole-genome "
    "kernel when LOCO is off), whose null model is fitted by REML; the F test and the "
    "Freedman-Lane permutations are made on the data whitened by it. PCs and covariates only "
    "(--hap-test pcs): the earlier test, without kinship.")

NO_RUN_KERNELS_MESSAGE = (
    "The kinship-aware haplotype test needs the kinship kernels of the GWAS page's Run GWAS on "
    "this genotype table, and this session has none for it. Re-run the GWAS on `{trait}`, or "
    "choose the PCs-only test under Settings, Haplotype test.")

RUN_CHANGED_MESSAGE = ("The genotype table's settings changed since the last Run GWAS on the "
                       "GWAS page: run GWAS again (or set the settings back) to use this page.")
RUN_NOT_FINISHED_MESSAGE = ("The last Run GWAS on the GWAS page did not finish: run GWAS again "
                            "to use this page.")


def run_table_changed(state, session_table):
    """True when the session's GWAS results are not results of this table: the GWAS page records
    the identity of the table its Run GWAS used (RUN_IDENTITY) with gwas_results, and rebuilds
    the session's table on every rerun, so another identity means the table's settings changed
    since that run. Results without a recorded identity are not taken for this table either:
    gwas_results, or the frame gwas_df alone, which the GWAS page writes during a Run before
    its results (the MLM frame), so a first Run that did not finish leaves it without one."""
    ran_on = state.get(RUN_IDENTITY)
    if ran_on is None:
        return "gwas_results" in state or state.get("gwas_df") is not None
    return str(ran_on) != str(getattr(session_table, "identity", ""))


def run_unfinished(state):
    """True when the GWAS page's last Run GWAS is known not to have finished. During a Run that
    page writes trait_col_locked (when it locks the Run's trait) and then gwas_df (the MLM
    frame); only when the Run ends does it write gwas_results[trait], RUN_IDENTITY, gwas_df again
    (a copy of the same frame as gwas_results[trait]) and active_trait. So without RUN_IDENTITY
    (no Run finished in this session) a gwas_df frame is a Run's that did not finish; with it, a
    trait_col_locked that is not active_trait, or a gwas_df that is not gwas_results[active_trait],
    was left by a later Run that did not finish. A later Run of the same trait that stopped before
    its MLM frame leaves no such mark."""
    frame = state.get("gwas_df")
    if state.get(RUN_IDENTITY) is None:
        return frame is not None
    trait = state.get("active_trait")
    locked = state.get("trait_col_locked")
    if locked is not None and str(locked) != str(trait):
        return True
    if frame is None:
        return False
    done = (state.get("gwas_results") or {}).get(trait)
    return not (isinstance(frame, pd.DataFrame) and isinstance(done, pd.DataFrame)
                and frame.equals(done))


def results_message(state, session_table):
    """The plain message the Post-GWAS page stops with when the session's GWAS results are not
    this table's (run_table_changed), chosen by what happened, or None when they are: when the
    last Run GWAS is known not to have finished (run_unfinished), RUN_NOT_FINISHED_MESSAGE (no
    setting needs to have changed then, and without a finished Run setting the settings back
    cannot help); else RUN_CHANGED_MESSAGE, the table's settings changed since the last Run GWAS
    (another identity, or results without one)."""
    if not run_table_changed(state, session_table):
        return None
    return RUN_NOT_FINISHED_MESSAGE if run_unfinished(state) else RUN_CHANGED_MESSAGE


def run_trait(state):
    """The trait whose GWAS results the Post-GWAS page takes: the last finished Run GWAS's
    (active_trait). The GWAS page writes it when a Run ends, with gwas_results[trait] and
    RUN_IDENTITY, and that identity names the trait, so while the session's table is that Run's
    (run_table_changed is False) these are the trait's own results on it. A Run writes
    trait_col_locked earlier, when it locks its trait, so a later Run that did not finish left
    its own trait there, and gwas_results may hold an earlier Run's results of that trait, on
    another table. Without a finished Run: the locked trait, or the chosen one (the page then
    finds no results for it)."""
    return state.get("active_trait") or state.get("trait_col_locked") or state.get("trait_col")


def meff_for_table(state, session_table):
    """The session's M_eff (meff_val) when it was computed on this table (MEFF_IDENTITY, which
    the GWAS page writes with it), else None: the page then takes M_eff as absent, as when it
    was never computed."""
    of = state.get(MEFF_IDENTITY)
    if of is None or str(of) != str(getattr(session_table, "identity", "")):
        return None
    return state.get("meff_val")


def run_pcs(state):
    """(pcs, pcs_full): the PCs of the GWAS page's run that left the session's table, the PCs of
    the model and the run's 20 PCs (the haplotype PCA panel), from GWAS_CTX; (None, None)
    without it. The GWAS page writes GWAS_CTX in the same step as geno_table. It writes the
    session's own "pcs" and "pcs_full" earlier in its run, so a run of that page that stopped
    before the table (another trait's samples, at its sample overlap warning) left another
    sample set's PCs there, beside the earlier table: those keys are not read."""
    gwas_ctx = state.get("GWAS_CTX")
    if not isinstance(gwas_ctx, dict):
        return None, None
    return gwas_ctx.get("pcs"), gwas_ctx.get("pcs_full")


def run_kernels(state, identity, samples=None):
    """(K0, K_by_chr, use_loco): the kernels of the GWAS page's Run GWAS on the session's
    table, or None. The GWAS page records them when a Run ends (RUN_KERNELS, with
    RUN_IDENTITY): references to the kernels its scan used, the table's identity, the trait
    and the LOCO choice. They are taken only for the table they were built on (``identity``,
    the session table's: another identity gives None, as run_table_changed), and only when
    every kernel is of ``samples`` (the table's sample IDs, in its row order), as
    pcs_of_samples checks the PCs. K_by_chr maps each chromosome to its LOCO kernel, or every
    one to K0 when the Run did not use LOCO."""
    rec = state.get(RUN_KERNELS)
    if not isinstance(rec, dict) or str(rec.get("identity")) != str(identity):
        return None
    K0, K_by_chr = rec.get("K0"), rec.get("K_by_chr")
    if K0 is None or not isinstance(K_by_chr, dict) or not K_by_chr:
        return None
    if samples is not None:
        ids = np.asarray(samples).astype(str)
        for K in (K0, *K_by_chr.values()):
            kid = np.asarray(getattr(K, "iid", np.empty((0, 2))))
            if kid.ndim != 2 or kid.shape[0] != ids.size \
                    or not np.array_equal(kid[:, 1].astype(str), ids):
                return None
    return K0, K_by_chr, bool(rec.get("use_loco", True))


def run_covariates(state):
    """(matrix, names): the user covariates of the GWAS page's run that left the session's
    table, one row per table sample in its row order, and their names, from GWAS_CTX (written
    with the table, like the PCs: run_pcs); (None, None) without covariates or without
    GWAS_CTX. The haplotype test adjusts for them, as the CLI's does; the table's identity
    names the covariate table, so the haplotype key needs nothing more for them."""
    gwas_ctx = state.get("GWAS_CTX")
    if not isinstance(gwas_ctx, dict) or gwas_ctx.get("user_covar") is None:
        return None, None
    return gwas_ctx.get("user_covar"), gwas_ctx.get("user_covar_names")


def pcs_of_samples(state, samples, *matrices):
    """True when every PC matrix of ``matrices`` (None is skipped) is of these ``samples``, the
    table's sample IDs in its row order: one row per sample, and GWAS_CTX's geno_index, which
    names those rows, the same IDs in the same order. Always true for run_pcs's PCs in a session
    the GWAS page left; the LD Blocks tab stops plainly otherwise."""
    ids = np.asarray(samples).astype(str)
    present = [np.asarray(m) for m in matrices if m is not None]
    if not present:
        return True
    if any(m.ndim != 2 or m.shape[0] != ids.size for m in present):
        return False
    gwas_ctx = state.get("GWAS_CTX")
    index = gwas_ctx.get("geno_index") if isinstance(gwas_ctx, dict) else None
    return index is None or np.array_equal(np.asarray(index).astype(str), ids)


@contextlib.contextmanager
def released_guard(session_table, on_released):
    """Run a block that reads the table. If a read fails because the table file was closed
    meanwhile (gwas.genostore raises ValueError for a closed table), call ``on_released``
    (the page shows RELEASED_MESSAGE and stops); any other error propagates unchanged.
    ``session_table`` is the session's table, or a function returning it when the error
    happens."""
    try:
        yield
    except ValueError:
        table = session_table() if callable(session_table) else session_table
        if table_released(table):
            on_released()
        raise


# ---------------------------------------------------------------------------------- keys
def digest(*parts):
    """A short blake2b of arrays, strings and numbers (an array by its dtype, shape and bytes)."""
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        if p is None:
            h.update(b"<None>")
        elif isinstance(p, np.ndarray) or isinstance(p, (list, tuple, pd.Index)):
            a = np.asarray(p)
            if a.dtype == object:
                a = a.astype(str)
            h.update(f"{a.dtype.str}{a.shape}".encode())
            h.update(np.ascontiguousarray(a).tobytes())
        else:
            h.update(repr(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def frame_key(df):
    """A data frame's content key: its columns and the hash of every value and index entry
    (pandas.util.hash_pandas_object over all rows: never a sample)."""
    if df is None:
        return "none"
    if not isinstance(df, pd.DataFrame) or df.shape[1] == 0:
        return digest(type(df).__name__, getattr(df, "shape", None))
    return digest([str(c) for c in df.columns],
                  pd.util.hash_pandas_object(df, index=True).to_numpy())


def table_key(ctx):
    """The table's name in cache keys: the SessionTable's identity carried by the context, or,
    for a context built with a plain array (tests), a digest of the array's content."""
    ident = str(getattr(ctx, "table_identity", "") or "")
    if ident:
        return ident
    g = getattr(ctx, "geno_ld", None)
    if isinstance(g, np.ndarray):
        return "array:" + digest(g)
    return f"object:{id(g)}"


def labels_key(ctx):
    """The SNP labels' name in cache keys: the table's identity (it names the post-QC SNP
    set), or, without one, a digest of the context's chroms, positions and SNP IDs."""
    ident = str(getattr(ctx, "table_identity", "") or "")
    if ident:
        return ident
    return "labels:" + digest(np.asarray(ctx.chroms).astype(str), np.asarray(ctx.positions),
                              np.asarray(ctx.sid).astype(str))


def rows_key(ctx):
    """The phenotype-aligned rows' name in cache keys."""
    keep = getattr(ctx, "keep_mask", None)
    return "all" if keep is None else digest(np.asarray(keep, dtype=bool))


def r2_key(ctx, kind, chrom, start_bp, end_bp, lead_snp, sids, min_pair_n=20):
    """An r2 matrix's cache key: the table, its rows, and the window's SNPs in their order."""
    sids = np.asarray(sids).astype(str)
    return (f"r2::{kind}::{table_key(ctx)}::{rows_key(ctx)}::{chrom}:{int(start_bp)}-{int(end_bp)}"
            f"::{lead_snp}::{digest(sids)}::{sids.size}snps::mp{int(min_pair_n)}")


def window_r2_key(G, min_pair_n=20):
    """The key of an r2 matrix computed from the window ``G`` itself (its values)."""
    G = np.asarray(G)
    return f"r2::window::{digest(G)}::mp{int(min_pair_n)}"


def haplotype_key(trait_col, min_hap_count, min_group_size, pcs, table, n_perm, blocks_hash,
                  pheno_hash, *, hap_test, use_loco):
    """The session key of one haplotype test: the trait, its settings, the PCs, the table's
    identity (``table``, from table_key), the permutations, the blocks and the phenotype, and
    the test (``hap_test``) with its kernel. The kernels are built from the table (its
    identity names the VCF, the QC, the imputation, the samples and the covariates), so the
    Run's LOCO choice (``use_loco``) is the one other thing they depend on: K{loco|global}
    for the kinship-aware test, Knone for the PCs-only test, which takes no kernel."""
    pcs_sig = "nopcs"
    if pcs is not None:
        pcs_arr = np.asarray(pcs)
        pcs_sig = f"pcs{pcs_arr.shape[0]}x{pcs_arr.shape[1]}"
    kern = "none" if hap_test == "pcs" else ("loco" if use_loco else "global")
    return (f"HAPGWAS::{trait_col}"
            f"::mhc{int(min_hap_count)}::mgs{int(min_group_size)}"
            f"::{pcs_sig}"
            f"::T{table}"
            f"::nperm{int(n_perm)}"
            f"::B{blocks_hash}::P{pheno_hash}"
            f"::test{hap_test}::K{kern}")


def detection_key(trait_col, ld_r2, flank_kb, top_n, seed_p, ld_decay_kb, gwas_sig, table):
    """The session key of the LD Blocks tab's detection: every detection parameter, the GWAS
    results and the table's identity."""
    return (f"LDDETECT::{trait_col}"
            f"::r2{ld_r2}::flank{flank_kb}::topn{int(top_n)}"
            f"::sig{seed_p}::decay{ld_decay_kb}::ms2::adj0.2::gap10"
            f"::G{gwas_sig}::T{table}")


# ------------------------------------------------------------------------ bounded caches
R2_CACHE_MAX = 10           # r2 matrices (the page's MAX_R2_CACHE)
FIGURE_CACHE_MAX = 8        # rendered figures per cache (PNG at 600 dpi, SVG and PDF each)
RESULT_CACHE_MAX = 8        # haplotype tests / LD-block detections kept in the session
TRAIT_CACHE_MAX = 8         # traits whose automatic LD blocks are kept


def bounded(state, name):
    """The session's OrderedDict under ``name`` (made, or converted from a plain dict)."""
    store = state.get(name)
    if not isinstance(store, OrderedDict):
        store = OrderedDict(store or {})
        state[name] = store
    return store


def remember(store, key, value, max_entries):
    """Put ``value`` under ``key`` in the OrderedDict ``store`` as its newest entry; drop the
    oldest entries beyond ``max_entries``."""
    store[key] = value
    store.move_to_end(key)
    while len(store) > int(max_entries):
        store.popitem(last=False)
    return value


def recall(store, key):
    """The entry under ``key`` (made the newest), or None."""
    if key in store:
        store.move_to_end(key)
        return store[key]
    return None


def keep_recent(d, key, value, max_entries):
    """Set ``d[key] = value`` as the newest key of the plain dict ``d`` and drop its oldest
    keys beyond ``max_entries`` (dicts keep insertion order)."""
    d.pop(key, None)
    d[key] = value
    while len(d) > int(max_entries):
        del d[next(iter(d))]
    return value


def remember_family(state, order_key, key, max_entries, suffixes=()):
    """Record session key ``key`` (one of a family such as HAPGWAS::...) as the newest; delete
    the oldest keys of the family beyond ``max_entries``, with their ``key + suffix`` keys."""
    order = [k for k in state.get(order_key, []) if k != key and k in state]
    order.append(key)
    while len(order) > int(max_entries):
        old = order.pop(0)
        for k in (old,) + tuple(old + s for s in suffixes):
            if k in state:
                del state[k]
    state[order_key] = order


# ------------------------------------------------------- one LD-block detection, shared
DETECTION_CACHE = "_ld_detect_cache"
_TABLE_ARGS = ("gwas_df", "chroms", "positions", "sid", "geno_imputed")


def _value(v):
    """An argument as a key part: None, a bool, a whole number, a float or a string by kind
    and value. Anything else is never shared (a key part equal to nothing)."""
    if v is None or isinstance(v, str):
        return (type(v).__name__, v)
    if isinstance(v, (bool, np.bool_)):
        return ("bool", bool(v))
    if isinstance(v, (int, np.integer)):
        return ("int", int(v))
    if isinstance(v, (float, np.floating)):
        return ("float", float(v))
    return ("unshared", object())


def _reads_of(table, identity=""):
    """What reading ``table`` gives besides the table it names: a reader's class, which table
    it reads (imputed or raw: table_kind) and in which number format, its rows (all, or a
    digest of the selected rows), and the table's ``identity`` (without one, the reader's array
    or file object); a plain array by its content. So a raw and an imputed reader of one
    identity, or two number formats of one table, never give the same key."""
    if table is None:
        return "none"
    if is_lazy(table):
        rows = getattr(table, "rows", None)
        n = int(getattr(getattr(table, "store", None), "N", -1))
        if rows is None or (rows.size == n and np.array_equal(rows, np.arange(n))):
            r = "all"
        else:
            r = digest(np.asarray(rows))
        source = table.arr if isinstance(table, ArrayColumns) else getattr(table, "store", None)
        return (type(table).__name__, table_kind(table), str(np.dtype(table.dtype)), r,
                str(identity) if identity else f"object:{id(source)}")
    return ("array", digest(np.asarray(table)))


def _labels_of(chroms, positions, sid):
    """The SNP labels a detection reads, by their values (positions with their number
    format)."""
    def hashed(a):
        return None if a is None else pd.util.hash_array(np.asarray(a, dtype=object))
    return digest(None if positions is None else np.asarray(positions), hashed(chroms), hashed(sid))


class SharedDetection:
    """gwas.ld.find_ld_clusters_genomewide, its result shared between calls whose inputs are
    provably the same.

    The page's automatic blocks and the LD Blocks tab's detection at its defaults are one
    computation when every sample is phenotype-aligned: the same function, the same table
    (``table``, its identity) read on the same rows, the same SNP labels and GWAS results (by
    their values), and every other argument equal, with the function's defaults filled in and
    flank_kb taken as the base pairs it becomes (int(round(float(flank_kb) * 1000.0)), the
    function's only use of it). The function reads the table only as ``[:, columns]`` windows
    of as_columns(table, float), which a reader gives in the same values and layout whatever
    its own layout, so a row selection of all rows reads what the table reads. Results are
    kept in a bounded session cache (DETECTION_CACHE); each caller gets its own copy."""

    def __init__(self, state, find, table):
        self.state, self.find, self.table = state, find, str(table)

    def key(self, kwargs):
        params = {k: v for k, v in kwargs.items() if k not in _TABLE_ARGS}
        try:
            signature = inspect.signature(self.find)
        except (TypeError, ValueError):          # a callable without a signature
            signature = None
        if signature is not None:
            for name, p in signature.parameters.items():
                if name not in _TABLE_ARGS and p.default is not inspect.Parameter.empty:
                    params.setdefault(name, p.default)
        if params.get("flank_kb") is not None:
            try:
                params["flank_bp"] = int(round(float(params["flank_kb"]) * 1000.0))
                del params["flank_kb"]
            except (TypeError, ValueError):
                pass
        f = self.find
        return ("LDBLOCKS", (getattr(f, "__module__", None), getattr(f, "__qualname__", None), id(f)),
                self.table, _reads_of(kwargs.get("geno_imputed"), self.table),
                _labels_of(kwargs.get("chroms"), kwargs.get("positions"), kwargs.get("sid")),
                frame_key(kwargs.get("gwas_df")),
                tuple(sorted((k, _value(v)) for k, v in params.items())))

    def find_ld_clusters_genomewide(self, **kwargs):
        key = self.key(kwargs)
        cache = bounded(self.state, DETECTION_CACHE)
        hit = recall(cache, key)
        if hit is not None:
            return hit.copy()
        res = self.find(**kwargs)
        if isinstance(res, pd.DataFrame):
            remember(cache, key, res.copy(), RESULT_CACHE_MAX)
        return res


# --------------------------------------------------- the significant-SNP table, kept
SIGTABLE_CACHE = "_sigtable_cache"


def gwas_key(ctx):
    """The GWAS results' key (frame_key of ctx.gwas_df), computed once per page run and kept on
    the context."""
    if not getattr(ctx, "gwas_key", ""):
        ctx.gwas_key = frame_key(ctx.gwas_df)
    return ctx.gwas_key


def sigtable_key(ctx, rule, edge_flank_bp, ld_decay_bp, geno_encoding, seed_p_used, top_n_used):
    """The Significant SNPs tab's cache key: the raw table it reads (the table's identity and
    the reader's rows, or a plain array's content), the SNP labels, the GWAS results, the
    blocks, the rule (every field of the gwas.significance.SignificanceRule) and every other
    argument the tab gives gwas.sigtable.build_significant_snp_table."""
    return ("SIGTAB",
            _reads_of(getattr(ctx, "geno_dosage_raw", None), getattr(ctx, "table_identity", "")),
            labels_key(ctx), gwas_key(ctx), frame_key(getattr(ctx, "haplo_df_auto", None)),
            repr(rule), _value(edge_flank_bp), _value(ld_decay_bp), _value(geno_encoding),
            _value(seed_p_used), _value(top_n_used))


# ------------------------------------------------------------ the haplotype tab's effects
def block_qc_effects(hap_gwas_df, chroms, positions, geno_imputed, geno_df, pheno_for_hap,
                     trait_col, min_hap_count, min_group_size, compute_maf=True,
                     geno_encoding=None, pcs=None, strict_selfing=False):
    """Add publication-oriented QC/effect metrics to hap_gwas_df (the Post-GWAS page's
    compute_block_qc_effects, unchanged apart from where its inputs come from):
      - Block_length_kb
      - n_snps_total (in physical window)
      - n_snps_poly
      - maf_median, maf_mean (optional)
      - EtaSq (eta^2) for MLG ANOVA groups actually tested
      - n_groups_tested, n_samples_used
      - Frac_other (based on tested vs total haplotypes, if available)

    ``geno_imputed`` is the imputed table as the LD code reads it (an array or a reader of
    this module); each block's window is read from it, phenotype-aligned rows first, as the
    page always cut it. ``pcs`` are the PCs of the GWAS page's run that left the session's
    table, from GWAS_CTX (run_pcs), the PCs the haplotype test gets, and ``strict_selfing`` is
    the session's selfing_mode. The page passes both; ff56e5c's page read them from the session
    inside this function."""
    from gwas.haplotype import mlg_labels_for_block as _mlg_labels_for_block
    from gwas.ld import anova_eta_sq_from_labels as _anova_eta_sq_from_labels
    from gwas.ld import maf_from_matrix as _maf_from_matrix
    from gwas.utils import align_pheno_to_geno, resolve_trait_column

    if hap_gwas_df is None or hap_gwas_df.empty:
        return hap_gwas_df

    out = hap_gwas_df.copy()
    out["Block_length_kb"] = (out["End"].astype(int) - out["Start"].astype(int)) / 1000.0

    # "Other" fraction (only if run_haplotype_block_gwas provides these columns)
    if ("n_haplotypes" in out.columns) and ("n_tested_haplotypes" in out.columns):
        denom = out["n_haplotypes"].replace(0, np.nan).astype(float)
        out["Frac_other"] = 1.0 - (out["n_tested_haplotypes"].astype(float) / denom)
    else:
        out["Frac_other"] = np.nan

    chroms_np = np.asarray(chroms).astype(str)
    pos_np = np.asarray(positions).astype(int)

    trait_col_qc = resolve_trait_column(trait_col, pheno_for_hap)

    # enforce genotype sample order alignment (prevents genotype-phenotype mismatches)
    ph_aligned, y_vec, keep_mask = align_pheno_to_geno(
        pheno_for_hap,
        geno_df,
        trait_col_qc,
    )

    n_snps_total = []
    n_snps_poly = []
    maf_median = []
    maf_mean = []
    eta_sq_list = []
    n_groups_list = []
    n_used_list = []

    for _, row in out.iterrows():
        ch = str(row["Chr"])
        s = int(row["Start"])
        e = int(row["End"])

        mask = (chroms_np == ch) & (pos_np >= s) & (pos_np <= e)
        G = np.asarray(geno_imputed[keep_mask, :][:, mask], float)

        n_snps_total.append(int(G.shape[1]))

        if G.shape[1] < 1:
            n_snps_poly.append(0)
            maf_median.append(np.nan)
            maf_mean.append(np.nan)
            eta_sq_list.append(np.nan)
            n_groups_list.append(0)
            n_used_list.append(int(np.isfinite(y_vec).sum()))
            continue

        var = np.nanvar(G, axis=0)
        poly = var > 0
        n_snps_poly.append(int(np.sum(poly)))

        # MAF summaries
        if compute_maf and np.any(poly):
            enc = geno_encoding
            maf = _maf_from_matrix(G[:, poly], enc)
            maf_median.append(float(np.nanmedian(maf)))
            maf_mean.append(float(np.nanmean(maf)))
        else:
            maf_median.append(np.nan)
            maf_mean.append(np.nan)

        # Effect size eta^2 using the same MLG grouping logic as permutation test
        # (collapse rare haplotypes and drop too-small groups + "Other")
        keep_y = np.isfinite(y_vec)
        if keep_y.sum() < 5:
            eta_sq_list.append(np.nan)
            n_groups_list.append(0)
            n_used_list.append(int(keep_y.sum()))
            continue

        G2 = G[keep_y, :]
        y2 = y_vec[keep_y]

        # ---- PC-residualize phenotype for eta^2 (match haplotype analysis test) ----
        pcs_for_eta = pcs
        if pcs_for_eta is not None:
            try:
                pcs_arr = np.asarray(pcs_for_eta, float)
                # align to same keep_mask used above
                pcs_sub = pcs_arr[keep_mask, :][keep_y, :]
                X = np.column_stack([np.ones(pcs_sub.shape[0]), pcs_sub])
                beta, *_ = np.linalg.lstsq(X, y2, rcond=None)
                y2 = y2 - X @ beta
            except Exception:
                pass  # fall back to unadjusted eta^2 if alignment fails

        if G2.shape[1] < 2:
            eta_sq_list.append(np.nan)
            n_groups_list.append(0)
            n_used_list.append(int(len(y2)))
            continue

        mlg, keep_mlg = _mlg_labels_for_block(
            G2,
            min_hap_count=min_hap_count,
            return_mask=True,
            strict_selfing=strict_selfing,
            geno_encoding=geno_encoding,
        )

        y2 = y2[keep_mlg]

        df = pd.DataFrame({
            "MLG": mlg,
            "y": y2
        })

        counts = df["MLG"].value_counts()
        valid = [g for g in counts.index if (counts[g] >= int(min_group_size) and g != "Other")]

        if len(valid) < 2:
            eta_sq_list.append(np.nan)
            n_groups_list.append(int(len(valid)))
            n_used_list.append(int(len(y2)))
            continue

        df = df[df["MLG"].isin(valid)].copy()
        eta_sq, n_groups, n_used = _anova_eta_sq_from_labels(df["y"].values, df["MLG"].values)

        eta_sq_list.append(eta_sq)
        n_groups_list.append(n_groups)
        n_used_list.append(n_used)

    out["n_snps_total"] = n_snps_total
    out["n_snps_poly"] = n_snps_poly
    out["maf_median"] = maf_median
    out["maf_mean"] = maf_mean
    out["EtaSq"] = eta_sq_list
    out["n_groups_tested_effect"] = n_groups_list
    out["n_samples_used_effect"] = n_used_list

    return out


def run_haplotype_tests(haplo_df, chroms, positions, geno_imputed, sid, geno_df, pheno_df,
                        trait_col, pcs, min_hap_count, min_group_size, n_perm, geno_encoding,
                        n_pcs_used, *, user_covar=None, hap_test="kinship", kernels=None):
    """gwas.haplotype.run_haplotype_block_gwas with the LD Blocks tab's positional arguments
    (the call it makes to run_haplotype_block_gwas_cached_fn), the run's user covariates
    (run_covariates), the test chosen in the tab's Settings and, for the kinship-aware test
    (the default, as in the CLI), the Run's kernels (run_kernels). Not wrapped in
    st.cache_data: that would hash the whole table, and the tab keeps each result in the
    session under its haplotype_key (a bounded family)."""
    from gwas.haplotype import run_haplotype_block_gwas

    return run_haplotype_block_gwas(
        haplo_df=haplo_df, chroms=chroms, positions=positions, geno_imputed=geno_imputed,
        sid=sid, geno_df=geno_df, pheno_df=pheno_df, trait_col=trait_col, pcs=pcs,
        min_hap_count=min_hap_count, min_group_size=min_group_size, n_perm=n_perm,
        geno_encoding=geno_encoding, n_pcs_used=n_pcs_used, user_covar=user_covar,
        hap_test=hap_test, kernels=kernels)


# ----------------------------------------------------------- the session keys (for Clear)
# Every session key pages/Post_GWAS_Analysis.py and pages/_ld_tabs/* write, widget keys
# excluded. Some are also the GWAS page's keys (chroms, positions, sid, trait_col, ...).
POST_GWAS_SESSION_KEYS = (
    "chroms", "positions", "sid",                         # the page re-stores the synced labels
    "ld_chroms", "ld_positions", "ld_sid",
    "r2_cache", "_ld_prev_trait",
    "haplo_df_auto", "haplo_df_auto_hash",
    "ld_blocks_final", BLOCKS_FINAL_KEY, "ld_blocks", "ld_block_metadata_auto",
    "adj_r2_min",
    "pheno_used_for_hap", "pheno_used_for_hap_label", "trait_col", "trait_col_locked",
    "haplo_blocks_to_use", "pcs_for_hap",
    "hap_tables", "hap_gwas_df", "hap_gwas_df_enriched", "hap_tables_vis",
    "pheno_vis", "sample_vis",
    "_local_ld_cache", "_hap_pca_cache", "_hap_box_cache",
    "_hapgwas_order", "_lddetect_order",
    DETECTION_CACHE,                                      # SharedDetection's results
    SIGTABLE_CACHE,                                       # the Significant SNPs tab's tables
)
# Families of keys named by their settings: each haplotype test (HAPGWAS::..., with its
# "::tables" and "::assign" keys) and each LD-block detection (LDDETECT::...).
POST_GWAS_SESSION_PREFIXES = ("HAPGWAS::", "LDDETECT::")
