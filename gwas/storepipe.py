"""The per-trait stages of the pipeline with the genotype table on disk (gwas.genostore).

One TraitStore per trait run. Its methods take the place of gwas.qc's
_pipeline_harmonize_ids, _pipeline_phenotype_qc, _pipeline_snp_qc,
_pipeline_build_geno_matrices and _pipeline_build_kinship, of
gwas.qc_report.compute_qc_report and of gwas.plotting.compute_meff_li_ji, with the same
arguments and return values, so gwas.pipeline._run_trait calls them at the same places.
How they differ from the in-memory functions:

* ``geno_df`` is a one-column frame: the sample index (as in memory) and ROW_COL, the store
  row of each sample. Every row operation the pipeline does on the genotype frame is done on
  it unchanged. The SNP columns live in ``self.cols`` (store variant indices, pipeline order).
* Per-SNP and per-sample statistics come from exact integer call counts (the ingest's counts,
  or GenotypeStore.count_pass), then the in-memory expressions. Integer counts are exact, so
  the statistics are the in-memory values bit for bit.
* The raw and imputed tables are LazyColumns; the mean fill is the per-SNP mean from the counts,
  float32(float64 count / called), which is SimpleImputer's value bit for bit.

Every line marked "today:" is the in-memory code it replaces (gwas/qc.py, gwas/kinship.py,
gwas/qc_report.py, gwas/plotting.py).
"""
import logging

import numpy as np
import pandas as pd

from gwas.genostore import LazyColumns
from gwas.genostore import piece as _piece
from gwas.io import _clean_chr_series
from gwas.qc import _log_or_warn, normalise_phenotype

log = logging.getLogger("gwas.qc")
ROW_COL = "__trace_store_row__"
PRUNE_BUFFER_COLS = 8192     # columns of one chromosome held at a time while LD-pruning

# The one-piece rule: a stage that sums across SNPs in floating point reads the table whole and
# runs the in-memory function when the table has at most this many cells. A FIXED number, never
# derived from free memory, so results cannot differ between machines.
ONE_PIECE_CELLS = 100_000_000
# The QC report's sample concordance sums 0/1 products in float32: exact below 2**24 markers.
CONCORDANCE_MAX_MARKERS = 2 ** 24


def one_piece_cells():
    import os

    v = os.environ.get("TRACE_STORE_ONE_PIECE_CELLS", "")      # tests only
    return int(v) if v else ONE_PIECE_CELLS


def is_store_frame(geno_df):
    return getattr(geno_df, "shape", (0, 0))[1] == 1 and ROW_COL in getattr(geno_df, "columns", ())


class TraitStore:
    def __init__(self, parsed):
        self.parsed = parsed
        self.store = parsed.store
        self.cols = np.arange(self.store.M, dtype=np.int64)
        self.all_cols = True          # no variant removed yet
        self.counts = None            # n1, n2, nmiss aligned with self.cols, for self.count_rows
        self.count_rows = None
        self.sid = None
        from gwas import genostore as _gs
        self.read_notes = _gs._LOG.setdefault("notes", [])   # shown in the read log

    # ------------------------------------------------------------------ helpers
    def _rows(self, geno_df):
        return geno_df[ROW_COL].to_numpy(dtype=np.int64)

    def _all_rows(self, rows):
        return rows.size == self.store.N and bool((rows == np.arange(self.store.N)).all())

    def _counts_for(self, rows):
        """n1, n2, nmiss (int64, aligned with self.cols) and the largest value over ``rows``."""
        st = self.parsed.stats
        if self._all_rows(rows):
            return {"n1": st["n1"][self.cols], "n2": st["n2"][self.cols], "nmiss": st["nmiss"][self.cols],
                    # the largest value among the CURRENT variants (today: of the current table)
                    "gmax": int(st["vmax"][self.cols].max()) if self.cols.size else -1, "n": int(rows.size)}
        c = self.store.count_pass(self.cols, rows)
        self.read_notes.append(("count_pass_snps", int(rows.size), int(self.cols.size)))
        return c

    def _row_nmiss_for(self, rows):
        """Missing calls of each row over the current columns (int64)."""
        if self.all_cols:
            return self.parsed.stats["row_nmiss"][rows]
        c = self.store.count_pass(self.cols, rows)
        self.read_notes.append(("count_pass_rows", int(rows.size), int(self.cols.size)))
        return c["row_nmiss"]

    def frame(self, geno_df, sid=None):
        """The in-memory genotype frame of today's code (samples x SNPs float32, NaN missing),
        read whole. Only for the stages not yet converted to pieces."""
        rows = self._rows(geno_df)
        raw = np.asarray(LazyColumns(self.store, self.cols, rows, fill=None, layout="F"))
        return pd.DataFrame(raw, index=geno_df.index, columns=self.sid if sid is None else sid)

    # ------------------------------------------------- gwas.qc._pipeline_harmonize_ids
    def harmonize_ids(self, genotypes, samples, chroms, positions, vid, ref, alt, pheno, info_scores=None):
        # today: the SNP-ID loop (qc.py, _pipeline_harmonize_ids), unchanged
        sid = []
        _ref_list = []
        _alt_list = []
        for i, (c, p) in enumerate(zip(chroms, positions)):
            rs = None
            if vid is not None:
                v = vid[i]
                if isinstance(v, (bytes, bytearray)):
                    v = v.decode("utf-8", errors="ignore")
                if v is not None:
                    v = str(v).strip()
                    if v != "" and v != "." and v.lower() != "none":
                        rs = v

            _r = "."
            _a = "."
            if ref is not None:
                rr = ref[i]
                if isinstance(rr, (bytes, bytearray)):
                    rr = rr.decode("utf-8", errors="ignore")
                _r_tmp = str(rr).strip()
                if _r_tmp not in ("", ".", "None", "none"):
                    _r = _r_tmp
            if alt is not None:
                aa = alt[i]
                for x in np.atleast_1d(aa):
                    if x is None:
                        continue
                    if isinstance(x, (bytes, bytearray)):
                        if len(x) == 0:
                            continue
                        x = x.decode("utf-8", errors="ignore")
                    s_alt = str(x).strip()
                    if s_alt not in ("", ".", "None", "none", "nan"):
                        _a = s_alt
                        break

            if rs is not None:
                sid.append(rs)
            else:
                r = _r if _r != "." else "N"
                a = _a if _a != "." else "N"
                sid.append(f"{c}_{p}_{r}_{a}_v{i}")
            _ref_list.append(_r)
            _alt_list.append(_a)

        sid = np.asarray(sid, dtype=str)
        if len(np.unique(sid)) != len(sid):
            sid = np.array([f"{s}_{i}" for i, s in enumerate(sid)], dtype=str)
            if pd.Series(sid).duplicated().any():
                sid = np.asarray([f"{s}__v{i}" for i, s in enumerate(sid)], dtype=str)

        # today: geno_df = pd.DataFrame(genotypes, index=samples, columns=sid)
        geno_df = pd.DataFrame({ROW_COL: np.arange(len(samples), dtype=np.int64)}, index=samples)

        # Filter split multi-allelic records (keep highest MAF per chr:pos)
        chrom_pos_key = pd.Series([f"{c}:{p}" for c, p in zip(chroms, positions)])
        if chrom_pos_key.duplicated().any():
            _n_before_multi = len(chrom_pos_key)
            # today: n_called / ac / an / af / maf_tmp from the whole table (all VCF samples);
            # here from the ingest-time counts: the same exact integers, the same expressions
            st = self.parsed.stats
            n_called = (self.store.N - st["nmiss"]).astype(float)
            ac = (st["n1"] + 2 * st["n2"]).astype(float)
            an = 2.0 * n_called
            af = np.divide(ac, an, out=np.full_like(ac, np.nan, dtype=float), where=(an > 0))
            maf_tmp = np.minimum(af, 1.0 - af)

            # today: for each unique key in order of first appearance, the single index or
            # mask[np.argmax(maf_tmp[mask])]; then np.sort. The same choice per key, found by
            # grouping (today's loop compares every key with all keys).
            dup = chrom_pos_key.duplicated(keep=False).to_numpy()
            keep_idx = list(np.flatnonzero(~dup))            # keys that occur once
            d_idx = np.flatnonzero(dup)
            codes, _uniq = pd.factorize(chrom_pos_key.to_numpy()[d_idx])
            order = np.argsort(codes, kind="stable")
            bounds = np.flatnonzero(np.diff(codes[order])) + 1
            for grp in np.split(d_idx[order], bounds):       # each: one key's records, ascending
                keep_idx.append(grp[np.argmax(maf_tmp[grp])])

            keep_idx = np.sort(keep_idx)
            _n_multi_removed = _n_before_multi - len(keep_idx)
            if _n_multi_removed > 0:
                _log_or_warn(
                    f"{_n_multi_removed} multi-allelic records collapsed "
                    f"(kept highest-MAF allele per chr:pos)."
                )
            self.cols = self.cols[keep_idx]
            self.all_cols = False
            chroms = chroms[keep_idx]
            positions = positions[keep_idx]
            sid = sid[keep_idx]
            _ref_list = [_ref_list[j] for j in keep_idx]
            _alt_list = [_alt_list[j] for j in keep_idx]
            if info_scores is not None:
                info_scores = info_scores[keep_idx]

        # today: Harmonize sample IDs (unchanged)
        geno_ids = pd.Series(geno_df.index.astype(str)).str.strip()
        pheno_ids = pd.Series(pheno.index.astype(str)).str.strip()

        if geno_ids.duplicated().any():
            geno_df = geno_df.loc[(~geno_ids.duplicated()).to_numpy()].copy()
            geno_ids = pd.Series(geno_df.index.astype(str)).str.strip()

        if pheno_ids.duplicated().any():
            pheno = pheno.loc[(~pheno_ids.duplicated()).to_numpy()].copy()
            pheno_ids = pd.Series(pheno.index.astype(str)).str.strip()

        # Match on the IDs without blanks around them (the duplicate checks above use them too):
        # with the raw IDs, a padded phenotype row never matched and was left out without a word.
        _n_blank = int((pd.Series(pheno.index.astype(str)) != pheno_ids.reset_index(drop=True)).sum())
        if _n_blank:
            pheno = pheno.copy()
            pheno.index = pd.Index(np.asarray(pheno_ids.to_numpy(dtype=object), dtype=str), name=pheno.index.name)
            _log_or_warn(f"{_n_blank} phenotype sample IDs had blanks around them; the blanks were "
                         "removed before matching the IDs to the VCF.")

        geno_df.index = np.asarray(geno_ids.to_numpy(dtype=object), dtype=str)
        pheno_orig = pheno.copy()  # keep before intersection for fallback matching

        common_ids = geno_df.index.intersection(pheno.index)

        if len(common_ids) == 0:
            geno_norm = geno_ids.str.lstrip("0").replace("", "0")
            pheno_norm = pheno_ids.str.lstrip("0").replace("", "0")
            geno_df_try = geno_df.copy()
            geno_df_try.index = np.asarray(geno_norm.to_numpy(dtype=object), dtype=str)
            pheno_try = pheno_orig.copy()
            pheno_try.index = np.asarray(pheno_norm.to_numpy(dtype=object), dtype=str)
            common_norm = geno_df_try.index.intersection(pheno_try.index)
            if len(common_norm) > 0:
                _log_or_warn(
                    "Sample IDs matched after stripping leading zeros "
                    f"(e.g., VCF '{geno_ids.iloc[0]}' -> phenotype "
                    f"'{pheno_ids.iloc[0]}'). "
                    f"Proceeding with {len(common_norm)} matched samples."
                )
                geno_df = geno_df_try.loc[common_norm]
                pheno = pheno_try.loc[common_norm]
                common_ids = common_norm
            else:
                raise ValueError(
                    "No overlapping sample IDs between genotype and phenotype files. "
                    "Check that accession names match exactly (case-sensitive). "
                    f"VCF samples (first 5): {geno_ids.head().tolist()} | "
                    f"Phenotype IDs (first 5): {pheno_ids.head().tolist()}"
                )
        else:
            geno_df = geno_df.loc[common_ids]
            pheno = pheno.loc[common_ids]

        n_geno = len(geno_ids)
        n_pheno = len(pheno_ids)
        n_common = len(common_ids)
        if n_common < min(n_geno, n_pheno):
            pct = n_common / max(n_geno, n_pheno) * 100
            _log_or_warn(
                f"Partial sample overlap: {n_common} of {n_geno} VCF samples matched "
                f"{n_common} of {n_pheno} phenotype samples ({pct:.0f}% overlap)."
            )

        allele_map = {s: (_ref_list[j], _alt_list[j]) for j, s in enumerate(sid)}
        self.sid = sid
        return geno_df, pheno, chroms, positions, sid, info_scores, allele_map

    # -------------------------------------------------- gwas.qc._pipeline_phenotype_qc
    def phenotype_qc(self, geno_df, pheno, trait_col, norm_option, ind_miss_thresh):
        raw_values = pheno[trait_col]
        _n_numeric = pd.to_numeric(raw_values, errors="coerce").notna().sum()
        if _n_numeric == 0:
            raise ValueError(
                f"Trait column '{trait_col}' contains no numeric values "
                f"(first 5: {raw_values.head().tolist()}). "
                "Ensure the trait column contains numeric measurements, not categories."
            )
        if _n_numeric < len(raw_values):
            _n_non = len(raw_values) - _n_numeric
            _log_or_warn(
                f"{_n_non} non-numeric values in trait '{trait_col}' will be treated as missing."
            )

        y = pd.to_numeric(pheno[trait_col], errors="coerce").values.reshape(-1, 1)

        keep_non_nan = ~np.isnan(y).ravel()
        y = y[keep_non_nan]
        geno_df = geno_df.iloc[keep_non_nan, :]
        pheno = pheno.iloc[keep_non_nan, :]

        if np.nanstd(y) < 1e-8:
            raise ValueError(
                f"Trait '{trait_col}' has near-zero variance after filtering. "
                "GWAS is not meaningful."
            )

        pheno = pheno.reindex(geno_df.index)

        # today: sample_missing = geno_df.isna().mean(axis=1)   (missing calls / SNPs, float64)
        _rows = self._rows(geno_df)
        with np.errstate(divide="ignore", invalid="ignore"):   # no SNP left: NaN, as pandas gives silently
            sample_missing = pd.Series(self._row_nmiss_for(_rows) / float(self.cols.size), index=geno_df.index)
        keep_samples = sample_missing < ind_miss_thresh

        geno_df = geno_df.loc[keep_samples]
        pheno = pheno.loc[keep_samples]

        if geno_df.shape[0] == 0:
            raise ValueError(
                f"All samples removed by missingness filter (threshold={ind_miss_thresh}). "
                "Try a less strict threshold."
            )

        y = pd.to_numeric(pheno[trait_col], errors="coerce").to_numpy().reshape(-1, 1)

        keep_y = ~np.isnan(y).ravel()
        geno_df = geno_df.iloc[keep_y, :]
        pheno = pheno.iloc[keep_y, :]
        y = y[keep_y, :]

        if not geno_df.index.equals(pheno.index):
            raise RuntimeError("Genotype and phenotype indices misaligned after filtering.")
        if geno_df.shape[0] != y.shape[0]:
            raise RuntimeError("Genotype and phenotype lengths differ after filtering.")

        y_flat, _norm_note = normalise_phenotype(y.ravel(), norm_option)
        y = y_flat.reshape(-1, 1)
        if _norm_note and norm_option not in ("none", "", None):
            log.warning("Phenotype '%s': %s", trait_col, _norm_note)

        return geno_df, pheno, y

    # -------------------------------------------------------- gwas.qc._pipeline_snp_qc
    def snp_qc(self, geno_df, chroms, positions, sid,
               maf_thresh, miss_thresh, mac_thresh, drop_alt,
               info_scores=None, info_thresh=0.0,
               canonical=None, max_het=None, het_excess_p=None):
        n_initial_snps = int(self.cols.size)
        rows = self._rows(geno_df)
        c = self._counts_for(rows)
        n = int(rows.size)

        # today: missing_rate = geno_df.isna().mean(axis=0)
        missing_rate = c["nmiss"] / float(n)

        # today: _dmax from the float table; the store holds the same integers
        _dmax = float(c["gmax"]) if (c["nmiss"] < n).any() else 0.0
        if _dmax > 2.5:
            raise ValueError(
                f"Genotype dosage > 2 detected (maximum observed: {_dmax:g}). "
                "TRACE models diploid panels; polyploid dosage coding is not "
                "supported. Disomic allopolyploid panels coded 0/1/2 per subgenome "
                "are supported -- see Limitations."
            )
        # today: AC = np.nansum(dos, axis=0); n_called = np.sum(~np.isnan(dos), axis=0)
        AC = (c["n1"] + 2 * c["n2"]).astype(np.float64)
        n_called = n - c["nmiss"]
        AN = 2.0 * n_called
        AF = np.divide(AC, AN, out=np.full_like(AC, np.nan, dtype=float), where=(AN > 0))
        maf = np.minimum(AF, 1.0 - AF)
        mac = np.minimum(AC, AN - AC)

        n_het = c["n1"]
        het_rate = np.divide(n_het, n_called,
                             out=np.full_like(AC, np.nan, dtype=float),
                             where=(n_called > 0))
        het_median = float(np.nanmedian(het_rate)) if np.isfinite(het_rate).any() else 0.0

        het_fail = np.zeros(len(maf), dtype=bool)
        if max_het is not None:
            het_fail |= np.isfinite(het_rate) & (het_rate > float(max_het))
        if het_excess_p is not None:
            from scipy.special import erfc
            He = 2.0 * AF * (1.0 - AF)
            exp_het = n_called * He
            var_het = n_called * He * (1.0 - He)
            with np.errstate(invalid="ignore", divide="ignore"):
                z = np.divide(n_het - exp_het, np.sqrt(var_het),
                              out=np.full_like(AC, np.nan, dtype=float),
                              where=(var_het > 0))
            p_excess = 0.5 * erfc(z / np.sqrt(2.0))
            het_fail |= np.isfinite(p_excess) & (p_excess < float(het_excess_p))

        has_info = (info_scores is not None and info_thresh > 0
                    and len(info_scores) == len(maf))
        if has_info:
            info_fail = ~np.isfinite(info_scores) | (info_scores < info_thresh)
        else:
            info_fail = np.zeros(len(maf), dtype=bool)

        qc_snp = {
            "Total SNPs": int(len(maf)),
            "Fail MAF": int((maf < maf_thresh).sum()),
            "Fail Missingness": int((missing_rate >= miss_thresh).sum()),
            "Fail MAC": int((mac < mac_thresh).sum()),
            "Fail INFO": int(info_fail.sum()),
            "Median het": het_median,
            "Fail HetExcess": int(het_fail.sum()),
        }

        keep_mask = (
                (maf >= maf_thresh) &
                (missing_rate < miss_thresh) &
                (mac >= mac_thresh) &
                (~info_fail) &
                (~het_fail)
        )
        keep_mask = np.asarray(keep_mask, dtype=bool)
        qc_snp["Fail ANY"] = int((~keep_mask).sum())
        qc_snp["Pass ALL"] = int(keep_mask.sum())

        if keep_mask.sum() == 0:
            raise ValueError("All SNPs failed QC with current thresholds.")

        cols = self.cols[keep_mask]
        cnt = {k: c[k][keep_mask] for k in ("n1", "n2", "nmiss")}
        chroms = chroms[keep_mask]
        positions = positions[keep_mask]
        sid = sid[keep_mask]
        if info_scores is not None:
            info_scores = info_scores[keep_mask]

        chroms_raw = chroms.copy()
        chr_label, chr_num, chr_order_labels, chr_order_map = _clean_chr_series(chroms, canonical=canonical)
        chroms = chr_label.astype(str)
        chroms_num = chr_num.astype(int)

        n_alt = int((chroms == "ALT").sum())
        qc_snp["ALT chromosomes"] = n_alt

        _alt_raw = pd.Series(chroms_raw).astype(str)[np.asarray(chroms) == "ALT"]
        _nonauto = _alt_raw.str.fullmatch(
            r"(?i)(chr)?(x|y|z|w|mt|m|pt|pltd|mtdna|chloroplast|mitochondri\w*)")
        n_nonauto = int(_nonauto.fillna(False).sum()) if len(_alt_raw) else 0
        qc_snp["Non-autosomal markers"] = n_nonauto
        qc_snp["Unplaced/scaffold markers"] = int(len(_alt_raw) - n_nonauto)
        if n_nonauto > 0:
            _log_or_warn(
                f"{n_nonauto} markers on non-autosomal sequences (X, Y, MT) excluded; "
                "TRACE models autosomes only.")

        if n_alt > 0 and n_alt == len(chroms):
            sample = list(pd.Series(chroms_raw).unique()[:10])
            raise ValueError(
                "None of the chromosome names in your VCF were recognized as "
                "numeric chromosomes. TRACE strips common species prefixes and "
                "keeps entries that resolve to integers (e.g., 'chr1' -> '1', "
                "'SL4.0ch01' -> '1').\n\n"
                f"Your VCF contains these CHROM values: {sample}\n\n"
                "If your species uses non-numeric chromosome names (e.g., wheat "
                "'1A'/'1B'/'1D'), recode them to integers before loading."
            )

        if drop_alt:
            keep = chroms != "ALT"
            cols = cols[keep]
            cnt = {k: v[keep] for k, v in cnt.items()}
            chroms = chroms[keep]
            chroms_num = chroms_num[keep]
            positions = positions[keep]
            sid = sid[keep]
            if info_scores is not None:
                info_scores = info_scores[keep]

        ord_idx = np.lexsort((positions.astype(int), chroms_num.astype(int)))
        cols = cols[ord_idx]
        cnt = {k: v[ord_idx] for k, v in cnt.items()}
        chroms = chroms[ord_idx]
        chroms_num = chroms_num[ord_idx]
        positions = positions[ord_idx]
        sid = sid[ord_idx]
        if info_scores is not None:
            info_scores = info_scores[ord_idx]

        self.cols = cols
        self.all_cols = False
        self.counts = cnt
        self.count_rows = rows
        self.sid = sid
        return geno_df, chroms, chroms_num, positions, sid, n_initial_snps, qc_snp, info_scores

    # ------------------------------------------- gwas.qc._pipeline_build_geno_matrices
    def build_geno_matrices(self, geno_df, method="mean", impute_k=5, impute_l=20):
        if method == "ldknni":
            from gwas.pipeline import TraitPipelineError
            raise TraitPipelineError(
                "--impute ldknni ranks every SNP against all others and needs the whole genotype "
                "table in memory; it is not available with --geno-store disk.")
        rows = self._rows(geno_df)
        if self.count_rows is None or rows.shape != self.count_rows.shape or not (rows == self.count_rows).all():
            raise RuntimeError("the sample set changed between SNP QC and the genotype tables")
        c = self.counts
        n = int(rows.size)
        # today: snp_imputation_rate = geno_df.isna().mean(axis=0).values.astype(np.float32)
        snp_imputation_rate = (c["nmiss"] / float(n)).astype(np.float32)
        # today: SimpleImputer(strategy="mean").fit_transform(geno_df).astype("float32"):
        # a missing call becomes the SNP's mean over its called samples
        fill = ((c["n1"] + 2 * c["n2"]).astype(np.float64) / (n - c["nmiss"])).astype(np.float32)
        self.fill = fill
        geno_dosage_raw = LazyColumns(self.store, self.cols, rows, fill=None, dtype=np.float32, layout="F")
        geno_imputed = LazyColumns(self.store, self.cols, rows, fill=fill, dtype=np.float32, layout="F")
        iid = np.c_[geno_df.index.values, geno_df.index.values].astype(str)
        return geno_dosage_raw, snp_imputation_rate, geno_imputed, iid

    # ------------------------------------------------- gwas.qc._pipeline_build_kinship
    def build_kinship(self, geno_df, geno_imputed, chroms, positions,
                      r2_thresh=0.2, window_bp=500_000, step_bp=100_000):
        """Kinship without the whole table: the allele frequencies come from the counts; the
        standardised columns are made per block with today's elementwise expressions; the LD
        pruning runs today's loop over one chromosome at a time, reading the chromosome's
        columns through a sliding buffer; the pruned set is assembled in memory and today's
        GRM function runs on it unchanged."""
        from gwas.kinship import _build_grm_from_Z
        from gwas.utils import _mean_impute_cols

        c = self.counts
        n = int(geno_imputed.shape[0])
        # today: p = allele_freq_from_called_dosage(G_called); valid_af = valid_af_mask_...(G_called)
        n_called = (n - c["nmiss"]).astype(float)
        AC = (c["n1"] + 2 * c["n2"]).astype(np.float64)
        AN = 2.0 * n_called
        p = np.divide(AC, AN, out=np.full_like(AC, np.nan, dtype=float), where=(AN > 0))
        valid_af = np.isfinite(p) & (p > 0.0) & (p < 1.0) & (AN >= 4.0)

        vloc = np.flatnonzero(valid_af)                 # local columns of geno_for_grm
        p_grm = p[valid_af]
        denom = np.sqrt(2.0 * p_grm * (1.0 - p_grm))
        denom[~np.isfinite(denom) | (denom == 0)] = 1.0

        def z32(local):
            """float32 F block: today's Z[:, local] (local indexes the valid-AF columns)."""
            g = geno_imputed.block(vloc[local], layout="F")
            Z = (g - 2.0 * p_grm[local]) / denom[local]
            return np.nan_to_num(Z, nan=0.0).astype(np.float32, copy=False)

        chroms_for_grm = np.asarray(chroms)[valid_af]
        positions_for_grm = np.asarray(positions)[valid_af]

        # today: _ld_prune_for_grm_by_chr_bp(chroms_for_grm, positions_for_grm, Z, ...): the same
        # loop; Zc[:, loc] comes from a sliding buffer of the chromosome's columns
        chroms_s = np.asarray(chroms_for_grm).astype(str)
        positions_i = np.asarray(positions_for_grm).astype(int)
        m_valid = int(vloc.size)
        keep_global = np.zeros(m_valid, dtype=bool)
        max_buf = 0

        for ch in np.unique(chroms_s):
            if str(ch) == "ALT":
                continue

            idx_chr = np.where(chroms_s == ch)[0]
            if idx_chr.size < 2:
                keep_global[idx_chr] = True
                continue

            idx_chr = idx_chr[np.argsort(positions_i[idx_chr])]
            pos_chr = positions_i[idx_chr]
            m = idx_chr.size
            keep = np.ones(m, dtype=bool)

            start_pos = int(pos_chr.min())
            end_pos = int(pos_chr.max())
            buf_lo = buf_hi = 0
            buf = None

            for w_start in range(start_pos, end_pos + 1, int(step_bp)):
                w_end = w_start + int(window_bp)
                loc = np.where((pos_chr >= w_start) & (pos_chr <= w_end) & keep)[0]
                if loc.size < 2:
                    continue

                MAX_LOC_SNPS = 800
                if loc.size > MAX_LOC_SNPS:
                    step = max(1, loc.size // MAX_LOC_SNPS)
                    loc = loc[::step]

                # today: Zw = Zc[:, loc]  (Zc = np.asarray(Z, float)[:, idx_chr], F-ordered float64)
                a, b = int(loc[0]), int(loc[-1]) + 1
                if buf is None or a < buf_lo or b > buf_hi:
                    buf_lo = a
                    buf_hi = min(m, max(b, a + _piece(PRUNE_BUFFER_COLS)))
                    buf = np.asfortranarray(z32(idx_chr[buf_lo:buf_hi]), dtype=float)
                    max_buf = max(max_buf, buf.shape[1])
                Zw = buf[:, loc - buf_lo]
                Zw = _mean_impute_cols(Zw)
                v = np.var(Zw, axis=0)
                okv = v > 1e-12
                if okv.sum() < 2:
                    continue

                Zw = Zw[:, okv]
                loc2 = loc[okv]

                Zw = Zw - Zw.mean(axis=0, keepdims=True)
                sd = Zw.std(axis=0, keepdims=True)
                sd[sd == 0] = 1.0
                Zw = Zw / sd

                C = (Zw.T @ Zw) / (Zw.shape[0] - 1)
                C = np.clip(C, -1.0, 1.0)

                if not np.isfinite(C).all():
                    continue
                R2 = C * C

                for i in range(R2.shape[0]):
                    if not keep[loc2[i]]:
                        continue
                    high = np.where(R2[i, :] > r2_thresh)[0]
                    high = high[high > i]
                    keep[loc2[high]] = False

            keep_global[idx_chr[keep]] = True
            del buf

        if keep_global.sum() < 10:
            keep_prune = np.ones(m_valid, dtype=bool)      # today: the unpruned Z comes back
        else:
            keep_prune = keep_global

        # today: var_ok = np.nanvar(Z_pruned, axis=0) > 0; Z_grm = Z_pruned[:, var_ok].astype(float32)
        kept = np.flatnonzero(keep_prune)
        var_ok = np.empty(kept.size, dtype=bool)
        blocks = []
        step = _piece(4096)
        for k in range(0, kept.size, step):
            zb = z32(kept[k:k + step])
            vk = np.nanvar(np.asfortranarray(zb, dtype=float), axis=0) > 0
            var_ok[k:k + step] = vk
            blocks.append(zb[:, vk])
        Z_grm = np.empty((n, int(var_ok.sum())), dtype=np.float32, order="F")
        k = 0
        for zb in blocks:
            Z_grm[:, k:k + zb.shape[1]] = zb
            k += zb.shape[1]
        del blocks

        if not np.isfinite(Z_grm).all():
            Z_grm = np.nan_to_num(Z_grm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        chroms_pruned = np.asarray(chroms_for_grm)[keep_prune]
        positions_pruned = np.asarray(positions_for_grm)[keep_prune]
        chroms_grm = chroms_pruned[var_ok]
        positions_grm = positions_pruned[var_ok]

        m = int(Z_grm.shape[1])
        if m < 10 or not np.isfinite(m):
            K = np.eye(Z_grm.shape[0], dtype=np.float32)
            kinship_model = "identity_GRM_low_snp_count"
        else:
            K = _build_grm_from_Z(Z_grm)
            kinship_model = "LD_pruned_Z_over_m"

        diag_mean = float(np.nanmean(np.diag(K)))
        if np.isfinite(diag_mean) and diag_mean > 1e-8:
            K /= diag_mean

        self.read_notes.append(("kinship", {"n": n, "m_post_qc": int(self.cols.size), "m_valid_af": int(m_valid),
                                            "m_pruned": int(Z_grm.shape[1]), "prune_buffer_cols_max": int(max_buf),
                                            "kinship_model": kinship_model}))
        return K, Z_grm, chroms_grm, positions_grm, kinship_model

    # ---------------------------------------------- gwas.qc_report.compute_qc_report
    def qc_report(self, geno_df, trait_values, trait_col, qc_snp=None, n_original_samples=None,
                  het_z_thresh=3.0, dup_conc_thresh=0.99, dup_sample_cap=2000,
                  geno_dosage_raw=None, geno_imputed=None, impute_method="mean",
                  impute_k=5, impute_l=20, piece=4096):
        """The report dict of gwas.qc_report.compute_qc_report from one pass over the store.
        Per-sample and per-marker statistics come from exact counts; the concordance products
        are accumulated in float32 over column blocks: sums of 0/1 products, exact integers
        while there are fewer than 2**24 markers. At 2**24 markers or more the report is not
        made (the in-memory function loses exactness there without saying so)."""
        from gwas import qc_report as _qcr

        rows = self._rows(geno_df)
        n, m = int(rows.size), int(self.cols.size)
        samples = geno_df.index.to_numpy().astype(str)
        c = self.counts

        # today (duplicate_pairs): above sample_cap samples, 5,000 markers drawn with seed 42
        marker_subsample = None
        subsampled = False
        if n > dup_sample_cap and marker_subsample is None:
            marker_subsample = min(m, 5000)
        dup_cols = None
        if marker_subsample is not None and marker_subsample < m:
            rng = np.random.default_rng(42)
            dup_cols = np.sort(rng.choice(m, size=marker_subsample, replace=False))
            subsampled = True
        used_m = m if dup_cols is None else int(dup_cols.size)
        if used_m >= CONCORDANCE_MAX_MARKERS:
            raise ValueError(
                f"the sample concordance counts need fewer than {CONCORDANCE_MAX_MARKERS:,} markers "
                f"with the genotype table on disk ({used_m:,} markers)")

        piece = _piece(piece)
        row_called = np.zeros(n, dtype=np.int64)
        row_het = np.zeros(n, dtype=np.int64)
        n_both = np.zeros((n, n), dtype=np.float32)
        n_equal = np.zeros((n, n), dtype=np.float32)

        def _acc(g):
            called = (g >= 0).astype(np.float32)
            nb = called @ called.T
            ne = np.zeros((n, n), dtype=np.float32)
            for d in (0, 1, 2):
                ind = (g == d).astype(np.float32)
                ne += ind @ ind.T
            return nb, ne

        for k in range(0, m, piece):
            g = self.store.gather(self.cols[k:k + piece], rows, table_cols=m)
            row_called += (g >= 0).sum(axis=1, dtype=np.int64)
            row_het += (g == 1).sum(axis=1, dtype=np.int64)
            if dup_cols is None:
                nb, ne = _acc(g)
                n_both += nb
                n_equal += ne
        if dup_cols is not None:
            for k in range(0, dup_cols.size, piece):
                nb, ne = _acc(self.store.gather(self.cols[dup_cols[k:k + piece]], rows, table_cols=m))
                n_both += nb
                n_equal += ne

        # ---- per_sample_heterozygosity (today: the same lines, on the counts)
        n_called_r, n_het_r = row_called, row_het
        with np.errstate(invalid="ignore", divide="ignore"):
            het = np.where(n_called_r > 0, n_het_r / n_called_r, np.nan)
        mean = float(np.nanmean(het))
        sd = float(np.nanstd(het, ddof=1))
        z = (het - mean) / sd if sd > 0 else np.zeros_like(het)
        flagged = np.abs(z) > het_z_thresh
        per_sample = pd.DataFrame({"sample": samples, "heterozygosity": het, "z": z, "flagged": flagged})
        flag_idx = [i for i in np.argsort(-np.abs(z)) if flagged[i]]
        sample_het = {
            "mean": mean, "sd": sd, "median": float(np.nanmedian(het)), "z_thresh": het_z_thresh,
            "n_flagged": int(flagged.sum()), "per_sample": per_sample,
            "flagged_samples": [{"sample": samples[i], "heterozygosity": float(het[i]), "z": float(z[i])}
                                for i in flag_idx],
        }

        # ---- duplicate_pairs (today: the lines after the products)
        with np.errstate(invalid="ignore", divide="ignore"):
            conc = np.where(n_both > 0, n_equal / n_both, 0.0)
        iu, ju = np.triu_indices(n, k=1)
        keep = (conc[iu, ju] >= dup_conc_thresh) & (n_both[iu, ju] > 0)
        pairs = []
        for i, j in zip(iu[keep], ju[keep]):
            nb = int(round(float(n_both[i, j])))
            n_diff = nb - int(round(float(n_equal[i, j])))
            pairs.append({"sample_a": samples[i], "sample_b": samples[j],
                          "concordance": round(float(conc[i, j]), 4),
                          "n_markers_compared": nb, "n_markers_differ": n_diff})
        pairs.sort(key=lambda p: (-p["concordance"], p["n_markers_differ"]))
        dup_pairs = {"conc_thresh": dup_conc_thresh, "n_pairs_scanned": n * (n - 1) // 2,
                     "n_markers_used": int(used_m), "marker_subsampled": subsampled,
                     "n_pairs_flagged": len(pairs), "pairs": pairs}

        # ---- per_variant_fis (today: the same lines, on the counts)
        he_min = 0.05
        n_called = n - c["nmiss"]
        with np.errstate(invalid="ignore", divide="ignore"):
            ho = np.where(n_called > 0, c["n1"] / n_called, np.nan)
            ac = (c["n1"] + 2 * c["n2"]).astype(np.float64)
            an = 2.0 * n_called
            p = np.where(an > 0, ac / an, np.nan)
            he = 2.0 * p * (1.0 - p)
            fis = np.where(he > 0, 1.0 - ho / he, np.nan)
        mask = (he > he_min) & np.isfinite(fis)
        sub = fis[mask]
        variant_fis = {
            "he_min": he_min, "n_markers": int(len(ho)), "n_he_gt_min": int(mask.sum()),
            "median_fis": float(np.median(sub)) if sub.size else float("nan"),
            "pct5_fis": float(np.percentile(sub, 5)) if sub.size else float("nan"),
            "pct95_fis": float(np.percentile(sub, 95)) if sub.size else float("nan"),
            "ho": ho, "fis": fis,
        }

        # ---- _imputation_summary (mean fill: the counts give every reported number)
        n_cells = n * m
        n_missing = int(c["nmiss"].sum())
        per_marker_missing = c["nmiss"] > 0
        imputation = {"method": impute_method}
        imputation.update(
            missing_cells=n_missing,
            missing_pct=round(100.0 * n_missing / n_cells, 4) if n_cells else 0.0,
            markers_with_missing=int(per_marker_missing.sum()),
            markers_zero_missing=int((~per_marker_missing).sum()),
            cells_filled=n_missing,
            noop=(n_missing == 0),
        )

        return {
            "sample_het": sample_het,
            "dup_pairs": dup_pairs,
            "variant_fis": variant_fis,
            "trait": _qcr.trait_summary(trait_values, trait_col, n_original=n_original_samples),
            "qc_snp": dict(qc_snp) if qc_snp else {},
            "n_samples": n,
            "n_markers": m,
            "imputation": imputation,
        }

    # ------------------------------------------------ gwas.plotting.compute_meff_li_ji
    def meff(self, geno_imputed):
        """M_eff (Li and Ji): the same function as with the table in memory
        (gwas.plotting._compute_meff_li_ji_impl), which sums its Gram matrix over blocks of SNP
        columns and gives the whole number, so both modes give the same M_eff at every size."""
        from gwas.plotting import _compute_meff_li_ji_impl

        return _compute_meff_li_ji_impl(geno_imputed)
