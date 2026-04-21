from fastlmm.association import single_snp
from pysnptools.snpreader import SnpData
import logging
import numpy as np
import pandas as pd
try:
    import streamlit as st
except ImportError:
    st = None
from scipy import stats
from gwas.utils import PhenoData, CovarData

_log = logging.getLogger(__name__)


def _run_gwas_impl(
    geno_imputed,
    y,
    pcs_full,
    n_pcs,
    sid,
    positions,
    chroms,
    chroms_num,
    iid,
    _K0,
    _K_by_chr,
    pheno_reader,
    trait_name=None,
):
    """Pure computation for LOCO GWAS. No Streamlit dependency."""
    geno_imputed = np.asarray(geno_imputed, dtype=np.float32, order="C")
    y = np.asarray(y, dtype=np.float32)

    sid = np.asarray(sid).astype(str)
    positions = np.asarray(positions).astype(int)
    chroms = np.asarray(chroms).astype(str)
    chroms_num = np.asarray(chroms_num).astype(int)
    iid = np.asarray(iid).astype(str)

    # ---- Ensure iid is n×2 for FastLMM ----
    if iid.ndim == 1:
        iid = np.c_[iid, iid]
    if iid.ndim != 2 or iid.shape[1] != 2:
        raise ValueError(f"iid must be (n,2). Got shape {iid.shape}")

    # Canonical IID representation for fast comparisons
    iid_str = np.asarray(iid, dtype=str)

    # ===== PCA alignment safety check =====
    if pcs_full is not None:
        pcs_full = np.asarray(pcs_full, dtype=np.float32)
        if pcs_full.shape[0] != iid.shape[0]:
            raise ValueError(
                f"PCA rows ({pcs_full.shape[0]}) do not match sample count ({iid.shape[0]})."
            )

    # Build covariates
    if int(n_pcs) > 0 and pcs_full is not None:
        covar_reader = CovarData(
            iid=iid_str,
            val=pcs_full[:, :int(n_pcs)],
            names=[f"PC{i + 1}" for i in range(int(n_pcs))],
        )
    else:
        covar_reader = None



    # ============================================================
    # 1) Run LOCO per-chromosome GWAS
    # ============================================================
    results_all = []
    chroms_unique = [c for c in np.unique(chroms) if str(c) != "ALT"]

    # Stable chromosome → numeric mapping
    chrom_to_num = {}
    for c in np.unique(chroms):
        idx = np.where(chroms == c)[0]
        if idx.size > 0:
            chrom_to_num[str(c)] = int(chroms_num[idx[0]])

    # Helper: normalize SNP column in FastLMM output
    def _normalize_fastlmm_snp_col(df: pd.DataFrame) -> pd.DataFrame:
        possible = ["SNP", "sid", "rsid", "rs", "Id", "MarkerName"]
        found = next((c for c in possible if c in df.columns), None)
        if found is None:
            raise ValueError(f"No SNP column found in FastLMM output. Columns: {list(df.columns)}")
        if found != "SNP":
            df = df.rename(columns={found: "SNP"})
        df["SNP"] = df["SNP"].astype(str)
        return df

    for ch in chroms_unique:
        mask_ch = (chroms == ch)
        if mask_ch.sum() == 0:
            continue

        sid_ch = sid[mask_ch]
        pos_ch = positions[mask_ch]
        ch_num = chrom_to_num.get(str(ch), int(chroms_num[mask_ch][0]))

        geno_ch = geno_imputed[:, mask_ch]

        pos_arr_ch = np.c_[
            np.full(len(pos_ch), float(ch_num)),
            np.zeros(len(pos_ch)),
            pos_ch.astype(float),
        ]

        geno_reader = SnpData(
            iid=iid_str,
            sid=sid_ch,
            val=geno_ch,
            pos=pos_arr_ch,
        )

        K_use = _K_by_chr.get(str(ch), None)

        # Basic finite guard
        if K_use is None or getattr(K_use, "val", None) is None:
            K_use = _K0
        else:
            Kv = np.asarray(K_use.val)
            if (Kv.ndim != 2) or (Kv.shape[0] != iid_str.shape[0]) or (Kv.shape[1] != iid_str.shape[0]) or (not np.isfinite(Kv).all()):
                K_use = _K0

        # HARD iid alignment guard
        try:
            if not hasattr(K_use, "iid") or K_use.iid is None:
                K_use = _K0
            else:
                kid = np.asarray(K_use.iid, dtype=str)
                if kid.shape != iid_str.shape or not np.array_equal(kid, iid_str):
                    K_use = _K0
        except (np.linalg.LinAlgError, ValueError, AttributeError):
            K_use = _K0

        res = single_snp(
            test_snps=geno_reader,
            pheno=pheno_reader,
            K0=K_use,
            covar=covar_reader,
            leave_out_one_chrom=False,
        ).copy()

        # normalize SNP column NOW (before concat)
        res = _normalize_fastlmm_snp_col(res)

        # Attach chr/pos from this chromosome context
        res["Chr"] = str(ch)
        res["Pos"] = pos_ch.astype(int).tolist()

        results_all.append(res)

    if len(results_all) == 0:
        raise RuntimeError("No chromosomes produced GWAS results (check chrom labels and ALT filtering).")

    results = pd.concat(results_all, ignore_index=True)

    # Ensure numeric p-values and de-duplicate on SNP
    if "PValue" not in results.columns:
        raise RuntimeError(f"FastLMM output missing PValue column. Columns: {list(results.columns)}")

    results["PValue"] = pd.to_numeric(results["PValue"], errors="coerce")
    results = results.dropna(subset=["SNP", "PValue"]).copy()
    results = results.sort_values("PValue").drop_duplicates(subset=["SNP"], keep="first")

    # Metadata frame (make sure SNP ids are unique)
    meta = pd.DataFrame({
        "SNP": sid.astype(str),
        "Chr_meta": chroms.astype(str),
        "ChrNum_meta": chroms_num.astype(int),
        "Pos_meta": positions.astype(int),
    }).drop_duplicates(subset=["SNP"], keep="first")

    # Merge results + metadata
    gwas_df = results.merge(meta, on="SNP", how="left")

    # Prefer metadata values (canonical)
    gwas_df["Chr"] = gwas_df["Chr_meta"].fillna(gwas_df.get("Chr", np.nan))
    gwas_df["Pos"] = gwas_df["Pos_meta"].fillna(gwas_df.get("Pos", np.nan))
    gwas_df["ChrNum"] = gwas_df["ChrNum_meta"]

    # Drop helper cols
    gwas_df = gwas_df[[c for c in gwas_df.columns if not c.endswith("_meta")]]

    # Final cleaning
    required = ["Chr", "Pos", "ChrNum", "PValue"]
    for col in required:
        if col not in gwas_df.columns:
            raise KeyError(f"Missing required column '{col}' after merge. Columns: {list(gwas_df.columns)}")

    gwas_df = gwas_df.dropna(subset=["Chr", "Pos", "PValue"]).copy()
    gwas_df["Chr"] = gwas_df["Chr"].astype(str)
    _chrnum_raw = pd.to_numeric(gwas_df["ChrNum"], errors="coerce")
    _n_chrnum_na = int(_chrnum_raw.isna().sum())
    if _n_chrnum_na:
        logging.warning("%d SNPs have missing ChrNum — assigned to chromosome 0.", _n_chrnum_na)
    gwas_df["ChrNum"] = _chrnum_raw.fillna(0).astype(int)
    gwas_df["Pos"] = pd.to_numeric(gwas_df["Pos"], errors="coerce")
    gwas_df = gwas_df.dropna(subset=["Pos"])
    gwas_df["Pos"] = gwas_df["Pos"].astype(int)
    gwas_df["PValue"] = pd.to_numeric(gwas_df["PValue"], errors="coerce").astype(float)

    # Preserve MLM effect size from FastLMM if available
    # FastLMM typically outputs "SnpWeight" as the MLM beta estimate
    if "SnpWeight" in gwas_df.columns:
        gwas_df = gwas_df.rename(columns={"SnpWeight": "Beta_MLM"})
        gwas_df["Beta_MLM"] = pd.to_numeric(gwas_df["Beta_MLM"], errors="coerce")

    return gwas_df


def run_gwas_cached(
    geno_imputed,
    y,
    pcs_full,
    n_pcs,
    sid,
    positions,
    chroms,
    chroms_num,
    iid,
    _K0,
    _K_by_chr,
    _pheno_reader_key,
    trait_name=None,
):
    """Cached wrapper: resolves session-state keys, delegates to _run_gwas_impl."""
    if isinstance(geno_imputed, str) and geno_imputed.startswith("ARRKEY::"):
        arr = st.session_state.get(geno_imputed)
        if arr is None:
            raise RuntimeError(
                "Cached genotype array missing from session_state. Re-run preprocessing."
            )
        geno_imputed = arr

    pheno_reader = st.session_state.get(_pheno_reader_key)
    if pheno_reader is None:
        raise RuntimeError("Cached phenotype reader missing from session_state.")

    return _run_gwas_impl(
        geno_imputed, y, pcs_full, n_pcs, sid, positions, chroms,
        chroms_num, iid, _K0, _K_by_chr, pheno_reader, trait_name,
    )

if st is not None:
    run_gwas_cached = st.cache_data(
        show_spinner="Running LOCO GWAS (cached)…", max_entries=8
    )(run_gwas_cached)


def run_mlmm_core_cached(
    geno_key: str, y_key: str,
    iid, sid, chroms, chroms_num, positions,
    K0_key, pheno_reader_key, covar_reader_key,
    gwas_df_mlm,
    p_enter=1e-4, max_cof=10, window_kb=500
):
    pheno_reader = st.session_state.get(pheno_reader_key)
    if pheno_reader is None:
        raise RuntimeError("Missing pheno_reader in session_state.")
    covar_reader = st.session_state.get(covar_reader_key)
    G = st.session_state.get(geno_key)
    Y = st.session_state.get(y_key)

    K0 = st.session_state.get(K0_key)
    if K0 is None:
        raise RuntimeError("Kinship matrix K0 missing from session_state.")
    if G is None or Y is None:
        raise RuntimeError("Missing cached arrays in session_state.")

    return run_mlmm_research_grade_fast(
        geno_imputed=np.asarray(G, dtype=np.float32, order="C"),
        sid=np.asarray(sid).astype(str),
        chroms=np.asarray(chroms).astype(str),
        chroms_num=np.asarray(chroms_num).astype(int),
        positions=np.asarray(positions).astype(int),
        iid=np.asarray(iid).astype(str),
        pheno_reader=pheno_reader,
        K0=K0,
        covar_reader=covar_reader,
        p_enter=float(p_enter),
        max_cof=int(max_cof),
        window_kb=int(window_kb),
        verbose=False,
    )

if st is not None:
    run_mlmm_core_cached = st.cache_data(
        show_spinner="Running MLMM (cached)…", max_entries=6
    )(run_mlmm_core_cached)


def run_mlmm_research_grade_fast(
    geno_imputed, sid, chroms, chroms_num, positions, iid, pheno_reader, K0, covar_reader,
    p_enter=1e-4, max_cof=10, window_kb=250, verbose=True
):
    """
    Optimized MLMM:
      ✅ one FastLMM scan per step
      ✅ cached genotype standardization
      ✅ optional BIC-proxy stopping without extra scans
      ✅ SAFE: no Streamlit calls unless verbose=True
    """

    # ---------------------------
    # Progress UI (SAFE)
    # ---------------------------
    progress_placeholder = None
    progress_bar = None
    if verbose:
        import streamlit as _st
        progress_placeholder = _st.empty()
        progress_bar = _st.progress(0)

    # ---------------------------
    # Standardize genotype ONCE (cached)
    # ---------------------------
    geno_std = _standardize_genotypes_impl(geno_imputed)

    # ---------------------------
    # SnpData for FastLMM (cached builder)
    # ---------------------------
    chrom_numeric = np.asarray(chroms_num, int)
    pos_arr = np.c_[
        chrom_numeric.astype(float),
        np.zeros_like(positions, float),
        np.asarray(positions, int).astype(float)
    ]
    all_snps = _build_mlmm_snpdata_impl(iid, sid, geno_std, pos_arr)

    # ---------------------------
    # Base covariates
    # ---------------------------
    if covar_reader is not None:
        covar_base = np.asarray(covar_reader.val, float)
        covar_names = list(getattr(covar_reader, "sid", []))
    else:
        covar_base = None
        covar_names = []

    def build_covar(selected_idxs: list[int]):
        if not selected_idxs:
            return covar_reader

        Xsel = geno_std[:, selected_idxs]  # (n, k)
        names = [f"cof_{i + 1}" for i in range(len(selected_idxs))]

        if covar_base is None:
            val, nm = Xsel, names
        else:
            val = np.c_[covar_base, Xsel]
            nm = covar_names + names

        return CovarData(iid=iid, val=val, names=nm)

    # ---------------------------
    # MLMM forward search
    # ---------------------------
    selected = []
    models = []
    bic_values = []

    # For BIC proxy (cheap) build y + base X once
    y_vec = np.asarray(pheno_reader.val, float).reshape(-1)
    X_base = np.ones((y_vec.size, 1), float)
    if covar_base is not None:
        X_base = np.c_[X_base, covar_base]

    for step in range(int(max_cof)):

        if verbose:
            progress_placeholder.info(f"MLMM iteration {step + 1}/{max_cof}")
            progress_bar.progress((step + 1) / max_cof)

        covar_current = build_covar(selected)

        res = single_snp(
            test_snps=all_snps,
            pheno=pheno_reader,
            K0=K0,
            covar=covar_current,
            leave_out_one_chrom=False,
        ).copy()

        if "PValue" not in res.columns:
            raise RuntimeError("FastLMM output missing PValue column.")

        pvals = pd.to_numeric(res["PValue"], errors="coerce").to_numpy(dtype=float)
        pvals = np.nan_to_num(pvals, nan=1.0, posinf=1.0, neginf=1.0)

        # ---- LD exclusion around cofactors ----
        exclude = np.zeros(sid.size, dtype=bool)
        if selected:
            for idx in selected:
                same_chr = (chroms == chroms[idx])
                close = (np.abs(positions - positions[idx]) <= int(window_kb) * 1000)
                exclude |= (same_chr & close)
            exclude[np.asarray(selected, int)] = True

        if exclude.all():
            break

        p_work = pvals.copy()
        p_work[exclude] = 1.0

        best_idx = int(np.argmin(p_work))
        best_p = float(p_work[best_idx])

        if verbose:
            import streamlit as _st
            _st.write(f"Forward step {step + 1}: SNP={sid[best_idx]} p={best_p:.2e}")

        # stopping rule: no SNP enters
        if best_p > float(p_enter):
            break

        # accept cofactor
        selected.append(best_idx)

        # ---- BIC proxy stopping (NO extra FastLMM call) ----
        X = X_base
        if selected:
            X = np.c_[X_base, geno_std[:, selected]]
        bic = bic_proxy_from_design(y_vec, X)

        bic_values.append(bic)
        models.append(selected.copy())

        if len(bic_values) >= 2 and (bic_values[-1] > bic_values[-2]):
            selected = models[-2]
            break

    # ---------------------------
    # Final GWAS model (one last scan)
    # ---------------------------
    covar_final = build_covar(selected)

    res_final = single_snp(
        test_snps=all_snps,
        pheno=pheno_reader,
        K0=K0,
        covar=covar_final,
        leave_out_one_chrom=False,
    ).copy()

    # normalize SNP column to "SNP"
    if "SNP" not in res_final.columns:
        if "sid" in res_final.columns:
            res_final = res_final.rename(columns={"sid": "SNP"})
    if "SNP" not in res_final.columns:
        res_final["SNP"] = sid.astype(str)

    mlmm_df = pd.DataFrame({
        "SNP": res_final["SNP"].astype(str).values,
        "PValue": pd.to_numeric(res_final["PValue"], errors="coerce").astype(float).values,
        "Model": "MLMM"
    })

    meta = pd.DataFrame({
        "SNP": sid.astype(str),
        "Chr": chroms.astype(str),
        "Pos": positions.astype(int)
    })
    mlmm_df = mlmm_df.merge(meta, on="SNP", how="left")

    if verbose and progress_placeholder is not None:
        progress_placeholder.success("MLMM finished.")
        progress_bar.empty()

    cof_table = pd.DataFrame({
        "Step": range(1, len(selected) + 1),
        "SNP": [sid[i] for i in selected],
        "Chr": [chroms[i] for i in selected],
        "Pos": [positions[i] for i in selected],
    })

    return mlmm_df, cof_table

def _build_mlmm_snpdata_impl(iid, sid, geno_std, pos_arr):
    """Build SnpData object for MLMM. Pure computation, no Streamlit."""
    from pysnptools.snpreader import SnpData
    return SnpData(
        iid=iid,
        sid=sid,
        val=geno_std.astype("float32"),
        pos=pos_arr,
    )


def build_mlmm_snpdata(iid, sid, geno_std, pos_arr):
    """Cached wrapper for _build_mlmm_snpdata_impl."""
    return _build_mlmm_snpdata_impl(iid, sid, geno_std, pos_arr)

if st is not None:
    build_mlmm_snpdata = st.cache_resource(
        show_spinner=False
    )(build_mlmm_snpdata)


def _standardize_genotypes_impl(geno_imputed: np.ndarray) -> np.ndarray:
    """
    Standardize genotype matrix. Pure computation, no Streamlit.
    geno_imputed: (n_samples, n_snps) float32/float64
    Returns float32 standardized matrix with NaNs -> 0.
    """
    G = np.asarray(geno_imputed, dtype=np.float32, order="C")
    mu = np.nanmean(G, axis=0)
    sd = np.nanstd(G, axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    Z = (G - mu) / sd
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return Z.astype(np.float32, copy=False)


def standardize_genotypes_for_mlmm_cached(geno_imputed: np.ndarray) -> np.ndarray:
    """Cached wrapper for _standardize_genotypes_impl."""
    return _standardize_genotypes_impl(geno_imputed)

if st is not None:
    standardize_genotypes_for_mlmm_cached = st.cache_data(
        show_spinner=False, max_entries=6
    )(standardize_genotypes_for_mlmm_cached)


def subset_snps_for_mlmm(
    geno_imputed: np.ndarray,
    sid: np.ndarray,
    chroms: np.ndarray,
    chroms_num: np.ndarray,
    positions: np.ndarray,
    gwas_df_mlm: pd.DataFrame | None,
    max_snps: int = 20000
):
    """
    Preferentially keep SNPs with smallest MLM p-values (much better than taking first N SNPs).
    Falls back to first N if MLM results missing.
    """
    sid = np.asarray(sid).astype(str)
    m = sid.size
    if m <= max_snps:
        return geno_imputed, sid, chroms, chroms_num, positions

    idx_keep = None

    if gwas_df_mlm is not None and isinstance(gwas_df_mlm, pd.DataFrame):
        if ("SNP" in gwas_df_mlm.columns) and ("PValue" in gwas_df_mlm.columns):
            tmp = gwas_df_mlm[["SNP", "PValue"]].dropna()
            tmp["SNP"] = tmp["SNP"].astype(str)
            tmp["PValue"] = pd.to_numeric(tmp["PValue"], errors="coerce")
            tmp = tmp.dropna().drop_duplicates("SNP")
            pmap = dict(zip(tmp["SNP"].values, tmp["PValue"].values))

            p = np.array([pmap.get(s, np.nan) for s in sid], dtype=float)
            # put missing p-values at the end
            p[~np.isfinite(p)] = np.inf
            idx_keep = np.argsort(p)[:max_snps]

    if idx_keep is None:
        idx_keep = np.arange(max_snps, dtype=int)

    idx_keep = np.asarray(idx_keep, dtype=int)
    return (
        np.asarray(geno_imputed, dtype=np.float32, order="C")[:, idx_keep],
        sid[idx_keep],
        np.asarray(chroms)[idx_keep],
        np.asarray(chroms_num)[idx_keep],
        np.asarray(positions)[idx_keep],
    )

def bic_proxy_from_design(y_vec: np.ndarray, X: np.ndarray) -> float:
    """
    Cheap BIC proxy using OLS RSS (ignores kinship covariance).
    Used only for relative stopping in MLMM.

    May underperform with strong population structure where OLS and MLM
    RSS diverge substantially.
    """
    y = np.asarray(y_vec, float).reshape(-1)
    X = np.asarray(X, float)
    # OLS solve
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(np.sum(resid * resid))
    n = y.size
    k = X.shape[1]
    rss = max(rss, 1e-12)
    return n * np.log(rss / n) + k * np.log(n)

def add_ols_effects_to_gwas(gwas_df, geno_imputed, y, covar_reader, sid, chunk_size=2000):
    """
    Compute per-SNP OLS effect size (Beta_OLS), standard error (SE_OLS), and t-statistic (Tstat_OLS)
    using the SAME phenotype and covariates used in the MLM.

    Chunked to avoid allocating an n×m residual matrix.
    """
    G = np.asarray(geno_imputed, float)            # (n, m)
    n, m = G.shape
    y = np.asarray(y, float).reshape(-1, 1)        # (n, 1)

    # Build covariate matrix C: intercept + PCs (if any)
    if covar_reader is not None and getattr(covar_reader, "val", None) is not None:
        C_cov = np.asarray(covar_reader.val, float)
        if C_cov.shape[0] != n:
            raise ValueError(
                f"Covariate matrix rows ({C_cov.shape[0]}) do not match genotype rows ({n})."
            )
        C = np.column_stack([np.ones((n, 1)), C_cov])
    else:
        C = np.ones((n, 1), float)

    p = C.shape[1]

    # Compute (C'C)^-1 C'
    CtC = C.T @ C
    try:
        CtC_inv = np.linalg.inv(CtC)
    except np.linalg.LinAlgError:
        CtC_inv = np.linalg.pinv(CtC)
    C_pinv = CtC_inv @ C.T

    # Rank deficiency: use pseudoinverse instead of dropping all covariates
    if np.linalg.matrix_rank(C) < C.shape[1]:
        logging.warning(
            "Covariate matrix is rank-deficient (%d cols, rank %d); "
            "using pseudoinverse for OLS effects.",
            C.shape[1], np.linalg.matrix_rank(C),
        )
        CtC_inv = np.linalg.pinv(CtC)
        C_pinv = CtC_inv @ C.T

    # Residualize phenotype once
    y_r = y - C @ (C_pinv @ y)
    sum_y2 = float(np.nansum(y_r ** 2))

    df = max(n - p - 1, 1)

    beta_all = np.full(m, np.nan, float)
    se_all = np.full(m, np.nan, float)
    t_all = np.full(m, np.nan, float)

    for start in range(0, m, int(chunk_size)):
        end = min(m, start + int(chunk_size))
        Gc = G[:, start:end]

        # Residualize this chunk
        Gc_r = Gc - C @ (C_pinv @ Gc)

        numer = (Gc_r * y_r).sum(axis=0)
        denom = (Gc_r ** 2).sum(axis=0)

        safe = denom > 0
        beta = np.full(end - start, np.nan, float)
        beta[safe] = numer[safe] / denom[safe]

        rss = np.full(end - start, np.nan, float)
        rss[safe] = sum_y2 - 2.0 * beta[safe] * numer[safe] + (beta[safe] ** 2) * denom[safe]
        rss = np.maximum(rss, 0.0)

        sigma2 = rss / df
        se = np.full(end - start, np.nan, float)
        se[safe] = np.sqrt(sigma2[safe] / denom[safe])

        tstat = np.full(end - start, np.nan, float)
        ok = np.isfinite(beta) & np.isfinite(se) & (se > 0)
        tstat[ok] = beta[ok] / se[ok]

        beta_all[start:end] = beta
        se_all[start:end] = se
        t_all[start:end] = tstat

    # Map back onto gwas_df by SNP id
    eff = pd.DataFrame({
        "SNP": np.asarray(sid).astype(str),
        "Beta_OLS": beta_all,
        "SE_OLS": se_all,
        "Tstat_OLS": t_all,
    })
    out = gwas_df.merge(eff, on="SNP", how="left")
    return out

def _ols_scan_pvals_fast(G, y, covar_reader=None, chunk_size=2000):
    """
    Fast OLS scan (chunked, memory-safe-ish):
    residualize y and SNPs against covariates (PCs), then compute
    t-test p-values for SNP slope.

    IMPORTANT: used only for stability/sensitivity screens.
    """
    G = np.asarray(G, float)                    # (n, m)
    y = np.asarray(y, float).reshape(-1, 1)     # (n, 1)
    n, m = G.shape

    # C = intercept + PCs
    if covar_reader is not None and getattr(covar_reader, "val", None) is not None:
        C_cov = np.asarray(covar_reader.val, float)
        if C_cov.shape[0] != n:
            raise ValueError(f"Covariates rows {C_cov.shape[0]} != n {n}")
        C = np.column_stack([np.ones((n, 1)), C_cov])
    else:
        C = np.ones((n, 1), float)

    p = C.shape[1]
    rank = np.linalg.matrix_rank(C)

    if rank < p:
        # Rank-deficient covariates (e.g., collinear pseudo-QTNs in LD).
        # Use pinv(C) directly — np.linalg.inv(C'C) may silently return
        # garbage for near-singular matrices without raising LinAlgError.
        _log.debug(
            "_ols_scan_pvals_fast: covariate matrix rank %d < %d cols; "
            "using pseudoinverse", rank, p,
        )
        C_pinv = np.linalg.pinv(C)        # (p, n)
        p = rank                           # adjust df for actual rank
    else:
        CtC_inv = np.linalg.inv(C.T @ C)
        C_pinv = CtC_inv @ C.T

    # Residualize y once
    y_r = y - C @ (C_pinv @ y)          # (n, 1)
    sum_y2 = float(np.nansum(y_r ** 2))

    df = max(n - p - 1, 1)

    pvals = np.ones(m, dtype=float)

    for start in range(0, m, int(chunk_size)):
        end = min(m, start + int(chunk_size))
        Gc = G[:, start:end]  # (n, chunk)

        # Residualize this chunk only
        Gc_r = Gc - C @ (C_pinv @ Gc)   # (n, chunk)

        numer = (Gc_r * y_r).sum(axis=0)      # (chunk,)
        denom = (Gc_r ** 2).sum(axis=0)       # (chunk,)

        safe = denom > 0
        beta = np.full(end - start, np.nan, float)
        beta[safe] = numer[safe] / denom[safe]

        rss = np.full(end - start, np.nan, float)
        rss[safe] = sum_y2 - 2.0 * beta[safe] * numer[safe] + (beta[safe] ** 2) * denom[safe]
        n_clamped = int(np.sum(rss[safe] < 0))
        if n_clamped > 0:
            _log.debug(
                "_ols_scan_pvals_fast: %d SNP(s) had negative RSS (catastrophic "
                "cancellation) in chunk [%d:%d] and were clamped to 0.",
                n_clamped, start, end,
            )
        rss = np.maximum(rss, 0.0)

        sigma2 = rss / df
        se = np.full(end - start, np.nan, float)
        se[safe] = np.sqrt(sigma2[safe] / denom[safe])

        t = np.full(end - start, np.nan, float)
        ok = np.isfinite(beta) & np.isfinite(se) & (se > 0)
        t[ok] = beta[ok] / se[ok]

        pv = np.ones(end - start, dtype=float)
        ok2 = np.isfinite(t)
        pv[ok2] = 2.0 * stats.t.sf(np.abs(t[ok2]), df=df)
        pvals[start:end] = np.clip(pv, 1e-300, 1.0)

    return pvals

def _ols_fit(y, X):
    """
    OLS (X already includes intercept if desired).
    Returns beta, residuals, SSE, df_resid
    """
    y = np.asarray(y, float).reshape(-1)
    X = np.asarray(X, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sse = float(np.sum(resid ** 2))
    df_resid = int(max(1, X.shape[0] - X.shape[1]))
    return beta, resid, sse, df_resid

def _one_hot_drop_first(groups: np.ndarray):
    """
    One-hot encode categorical groups and drop the first level (baseline).
    Returns: (H, levels_kept, levels_all)
    """
    g = pd.Series(groups).astype(str)
    levels_all = list(pd.unique(g))
    if len(levels_all) < 2:
        return None, [], levels_all
    keep_levels = levels_all[1:]
    H = np.zeros((len(g), len(keep_levels)), float)
    for j, lev in enumerate(keep_levels):
        H[:, j] = (g.values == lev).astype(float)
    return H, keep_levels, levels_all

def _nested_f_test(sse0, sse1, df0, df1):
    """
    Nested model F test: reduced(0) vs full(1).
    df0, df1 are residual degrees of freedom.
    """
    from scipy.stats import f as fdist

    df_num = int(df0 - df1)  # added parameters in full model
    if df_num <= 0 or df1 <= 0:
        return np.nan, np.nan

    if not (np.isfinite(sse0) and np.isfinite(sse1)) or sse1 <= 0:
        return np.nan, np.nan

    F = ((sse0 - sse1) / df_num) / (sse1 / df1)
    if not np.isfinite(F) or F < 0:
        return np.nan, np.nan

    p = float(fdist.sf(F, df_num, df1))
    return float(F), float(p)


# ============================================================
# FarmCPU — Fixed and Random Model Circulating Probability
# Unification (Liu et al. 2016, PLOS Genetics)
#
# Iterates between:
#   1. OLS scan (fixed-effect linear model) with pseudo-QTNs as covariates
#   2. MLM-based optimization of pseudo-QTN set
# ============================================================


def _bin_select_pseudo_qtns(
    pvals, chroms, positions, bin_sizes, p_threshold,
    exclude_idxs=None,
):
    """
    Select candidate pseudo-QTNs by binning significant SNPs.

    For each bin size, group significant SNPs into non-overlapping
    genomic bins and keep the best (smallest p-value) per bin.
    Union across all bin sizes, deduplicate.
    """
    chroms = np.asarray(chroms, str)
    positions = np.asarray(positions, int)
    pvals = np.asarray(pvals, float)

    sig_mask = np.isfinite(pvals) & (pvals < float(p_threshold))
    if exclude_idxs:
        sig_mask[np.array(exclude_idxs, int)] = False

    sig_idx = np.where(sig_mask)[0]
    if sig_idx.size == 0:
        return []

    selected = set()
    for bin_size in bin_sizes:
        for ch in np.unique(chroms[sig_idx]):
            ch_mask = (chroms[sig_idx] == ch)
            ch_idx = sig_idx[ch_mask]
            ch_pos = positions[ch_idx]
            ch_pvals = pvals[ch_idx]

            # Assign bins
            bins = ch_pos // int(bin_size)
            for b in np.unique(bins):
                in_bin = (bins == b)
                best_local = int(ch_idx[in_bin][np.argmin(ch_pvals[in_bin])])
                selected.add(best_local)

    return sorted(selected)


def _optimize_pseudo_qtns_mlm(
    candidate_idxs, geno_imputed, y_vec, iid, sid,
    chroms_num, positions, covar_reader, K0,
    p_threshold,
    max_pseudo_qtns=15,
    pvals=None,
):
    """
    Validate pseudo-QTN candidates via MLM forward selection.

    Candidates are sorted by OLS p-value (best first) and tested
    sequentially with ``single_snp``.  Each accepted candidate is added
    to the covariate set for subsequent tests, so later candidates must
    be significant *after* accounting for PCs + kinship + all previously
    accepted pseudo-QTNs.  This approximates GAPIT3's joint REML
    validation and prevents noise accumulation.
    """
    candidate_idxs = list(candidate_idxs)
    if not candidate_idxs:
        return []

    # Sort candidates by OLS p-value (best first) and cap.
    if pvals is not None:
        candidate_idxs = sorted(
            candidate_idxs,
            key=lambda i: float(pvals[i]) if np.isfinite(pvals[i]) else 1.0,
        )
    if len(candidate_idxs) > max_pseudo_qtns * 2:
        candidate_idxs = candidate_idxs[:max_pseudo_qtns * 2]

    pheno_val = np.asarray(y_vec, float)
    if pheno_val.ndim == 1:
        pheno_val = pheno_val.reshape(-1, 1)
    pheno = PhenoData(iid=iid, val=pheno_val)

    # Extract base PCs from covar_reader
    base_cov = None
    base_names = []
    if covar_reader is not None and getattr(covar_reader, "val", None) is not None:
        base_cov = np.asarray(covar_reader.val, float)
        base_names = list(getattr(covar_reader, "sid", []))

    # Forward selection: accepted pseudo-QTNs accumulate into covariates
    accepted = []
    geno_std = _standardize_genotypes_impl(geno_imputed)

    for test_idx in candidate_idxs:
        # Build covariates: PCs + already-accepted pseudo-QTN genotypes
        parts = []
        names = list(base_names)
        if base_cov is not None:
            parts.append(base_cov)
        if accepted:
            parts.append(geno_std[:, accepted])
            names += [f"pqtn_{k+1}" for k in range(len(accepted))]

        if parts:
            covar_test = CovarData(
                iid=iid, val=np.column_stack(parts), names=names,
            )
        else:
            covar_test = None

        test_geno = np.ascontiguousarray(
            geno_imputed[:, test_idx:test_idx + 1],
        )
        pos_arr = np.array([[float(chroms_num[test_idx]), 0.0,
                             float(positions[test_idx])]])
        snp_data = SnpData(
            iid=iid, sid=np.array([sid[test_idx]]),
            val=test_geno, pos=pos_arr,
        )

        try:
            res = single_snp(
                test_snps=snp_data, pheno=pheno,
                K0=K0, covar=covar_test,
                leave_out_one_chrom=False,
            )
            p = float(pd.to_numeric(res["PValue"], errors="coerce").iloc[0])
            if np.isfinite(p) and p < float(p_threshold):
                accepted.append(test_idx)
        except Exception as exc:
            import logging
            logging.debug("pseudo-QTN %s skipped: %s", sid[test_idx], exc)
            continue

        if len(accepted) >= max_pseudo_qtns:
            break

    return accepted


def run_farmcpu(
    geno_imputed, sid, chroms, chroms_num, positions, iid,
    pheno_reader, K0, covar_reader,
    p_threshold=0.01,
    bin_sizes=(500_000, 5_000_000, 50_000_000),
    max_iterations=10,
    max_pseudo_qtns=15,
    final_scan="mlm",
    verbose=True,
    use_loco=True,
):
    """
    FarmCPU: Fixed and Random Model Circulating Probability Unification.

    Iterates between:
      1. GLM scan with pseudo-QTNs as fixed covariates (fast OLS)
      2. Bin-based pseudo-QTN selection (matching GAPIT3's FarmCPU.BIN)

    Parameters
    ----------
    geno_imputed : ndarray (n, m)
        Imputed genotype matrix.
    sid, chroms, chroms_num, positions : arrays
        SNP metadata.
    iid : ndarray (n, 2)
        Sample IDs.
    pheno_reader : PhenoData
        Phenotype.
    K0 : KernelData
        Whole-genome kinship matrix.
    covar_reader : CovarData or None
        PCA covariates.
    p_threshold : float
        P-value threshold for pseudo-QTN selection (default 0.01,
        matching GAPIT3).
    bin_sizes : tuple of int
        Genomic bin sizes (bp) for candidate selection.
    max_iterations : int
        Maximum FarmCPU iterations.
    max_pseudo_qtns : int
        Maximum number of pseudo-QTNs.
    final_scan : str
        Final scan model: ``"mlm"`` (default, LOCO kinship with LD-pruned
        GRM) or ``"ols"`` (matching standard FarmCPU).
    verbose : bool
        Show Streamlit progress.
    use_loco : bool
        Use LOCO kinship for the MLM final scan.  When False, uses
        global K0 for all chromosomes.

    Returns
    -------
    farmcpu_df : DataFrame
        GWAS results with columns [SNP, Chr, Pos, PValue, Model].
    pseudo_qtn_table : DataFrame
        Selected pseudo-QTNs with p-values.
    convergence_info : dict
        Iteration count, convergence status.

    Notes
    -----
    Iterative scans use fast OLS for pseudo-QTN selection.  The default
    final scan (``final_scan="mlm"``) uses MLM with LOCO kinship for
    structure correction.  The ``"ols"`` option matches the standard
    FarmCPU algorithm (Liu et al. 2016).
    """
    geno_imputed = np.asarray(geno_imputed, dtype=np.float32, order="C")
    sid = np.asarray(sid, str)
    chroms = np.asarray(chroms, str)
    chroms_num = np.asarray(chroms_num, int)
    positions = np.asarray(positions, int)
    iid = np.asarray(iid, str)
    if iid.ndim == 1:
        iid = np.c_[iid, iid]

    y_vec = np.asarray(pheno_reader.val, float).reshape(-1)
    geno_std = _standardize_genotypes_impl(geno_imputed)
    n, m = geno_imputed.shape

    # Progress UI
    progress_placeholder = None
    progress_bar = None
    if verbose:
        import streamlit as _st
        progress_placeholder = _st.empty()
        progress_bar = _st.progress(0)

    # Base covariate matrix (PCs)
    base_cov = None
    base_names = []
    if covar_reader is not None and getattr(covar_reader, "val", None) is not None:
        base_cov = np.asarray(covar_reader.val, float)
        base_names = list(getattr(covar_reader, "sid", []))

    def _build_covar(pseudo_qtn_idxs):
        """Build CovarData with PCs + pseudo-QTN genotypes."""
        parts = []
        names = list(base_names)
        if base_cov is not None:
            parts.append(base_cov)
        if pseudo_qtn_idxs:
            parts.append(geno_std[:, pseudo_qtn_idxs])
            names += [f"pqtn_{i+1}" for i in range(len(pseudo_qtn_idxs))]
        if not parts:
            return covar_reader
        return CovarData(
            iid=iid,
            val=np.column_stack(parts),
            names=names,
        )

    pseudo_qtns = []
    pvals = None  # last OLS iteration p-values (used for OLS final scan)
    converged = False
    iteration_log = []

    for iteration in range(int(max_iterations)):
        if verbose:
            progress_placeholder.info(
                f"FarmCPU iteration {iteration + 1}/{max_iterations} "
                f"({len(pseudo_qtns)} pseudo-QTNs)"
            )
            progress_bar.progress((iteration + 1) / max_iterations)

        # Step 1: GLM scan with current pseudo-QTNs as covariates
        covar_current = _build_covar(pseudo_qtns)
        pvals = _ols_scan_pvals_fast(geno_imputed, y_vec,
                                     covar_reader=covar_current)

        # Step 2: Bin-select candidate pseudo-QTNs
        candidates = _bin_select_pseudo_qtns(
            pvals, chroms, positions, bin_sizes, p_threshold,
            exclude_idxs=pseudo_qtns,
        )

        if not candidates:
            converged = True
            break

        # Step 3: Merge candidates with existing pseudo-QTNs, then
        # validate the FULL set via MLM forward selection.  Each
        # candidate must be significant under kinship + PCs + all
        # previously accepted pseudo-QTNs (approximates GAPIT3's
        # joint REML pruning).
        all_candidates = sorted(set(candidates + pseudo_qtns))
        new_pseudo_qtns = _optimize_pseudo_qtns_mlm(
            all_candidates, geno_imputed, y_vec, iid, sid,
            chroms_num, positions, covar_reader, K0,
            p_threshold, max_pseudo_qtns=max_pseudo_qtns,
            pvals=pvals,
        )

        if not new_pseudo_qtns:
            converged = True
            break

        iteration_log.append({
            "iteration": iteration + 1,
            "n_candidates": len(candidates),
            "n_pseudo_qtns": len(new_pseudo_qtns),
        })

        # Step 4: Check convergence (exact match or Jaccard > 0.8)
        if set(new_pseudo_qtns) == set(pseudo_qtns):
            converged = True
            break
        if pseudo_qtns:
            _inter = len(set(new_pseudo_qtns) & set(pseudo_qtns))
            _union = len(set(new_pseudo_qtns) | set(pseudo_qtns))
            if _union > 0 and _inter / _union > 0.8:
                converged = True
                break

        pseudo_qtns = new_pseudo_qtns

    # ── Prune collinear pseudo-QTNs ─────────────────────────────────
    # High-LD pairs (r² > 0.5) degrade the final scan as near-
    # collinear covariates.  Drop the weaker member of each pair.
    if len(pseudo_qtns) >= 2:
        _pq = np.array(pseudo_qtns)
        _G_pq = geno_std[:, _pq]
        _C_pq = np.corrcoef(_G_pq, rowvar=False)
        _r2_pq = _C_pq ** 2
        np.fill_diagonal(_r2_pq, 0.0)
        _drop = set()
        for i in range(len(_pq)):
            if i in _drop:
                continue
            for j in range(i + 1, len(_pq)):
                if j in _drop:
                    continue
                if _r2_pq[i, j] > 0.5:
                    # Drop the one with higher (worse) OLS p-value
                    pi = pvals[_pq[i]] if np.isfinite(pvals[_pq[i]]) else 1.0
                    pj = pvals[_pq[j]] if np.isfinite(pvals[_pq[j]]) else 1.0
                    _drop.add(j if pj >= pi else i)
        if _drop:
            kept = [idx for k_idx, idx in enumerate(_pq) if k_idx not in _drop]
            logging.warning(
                "FarmCPU: dropped %d collinear pseudo-QTN(s) (r2>0.5); "
                "%d -> %d pseudo-QTNs.", len(_drop), len(pseudo_qtns), len(kept),
            )
            pseudo_qtns = kept

    # ── Final scan ───────────────────────────────────────────────────
    covar_final = _build_covar(pseudo_qtns)

    if final_scan == "mlm":
        # MLM final scan with LOCO kinship + pseudo-QTNs as covariates.
        # Uses raw (0/1/2) genotypes for both SnpData and GRM construction,
        # and LD-pruned genotypes for LOCO kernels (matching main pipeline).
        from gwas.kinship import (
            _build_loco_kernels_impl, _standardize_geno_for_grm,
            _ld_prune_for_grm_by_chr_bp,
        )
        Z_pruned, prune_mask = _ld_prune_for_grm_by_chr_bp(
            chroms, positions, geno_imputed, return_mask=True,
        )
        chroms_pruned = chroms[prune_mask]
        Z_grm = _standardize_geno_for_grm(Z_pruned)
        _, K_by_chr, _ = _build_loco_kernels_impl(iid, Z_grm, chroms_pruned)
        if not use_loco:
            K_by_chr = {ch: K0 for ch in K_by_chr}

        results_all = []
        for ch in np.unique(chroms):
            mask_ch = chroms == ch
            sid_ch = sid[mask_ch]
            pos_ch = positions[mask_ch]
            geno_ch = geno_imputed[:, mask_ch]
            ch_num = int(chroms_num[mask_ch][0])

            pos_arr_ch = np.c_[
                np.full(mask_ch.sum(), float(ch_num)),
                np.zeros(mask_ch.sum()),
                pos_ch.astype(float),
            ]
            snps_ch = _build_mlmm_snpdata_impl(
                iid, sid_ch, geno_ch, pos_arr_ch,
            )
            K_use = K_by_chr.get(str(ch), K0)

            res_ch = single_snp(
                test_snps=snps_ch,
                pheno=pheno_reader,
                K0=K_use,
                covar=covar_final,
                leave_out_one_chrom=False,
            ).copy()

            if "SNP" not in res_ch.columns:
                for col in ("sid", "rsid", "rs", "Id", "MarkerName"):
                    if col in res_ch.columns:
                        res_ch = res_ch.rename(columns={col: "SNP"})
                        break
            if "SNP" not in res_ch.columns:
                res_ch["SNP"] = sid_ch.astype(str)
            res_ch["SNP"] = res_ch["SNP"].astype(str)
            results_all.append(res_ch)

        res_final = pd.concat(results_all, ignore_index=True)
        pval_map = dict(zip(
            res_final["SNP"].values,
            pd.to_numeric(res_final["PValue"], errors="coerce").values,
        ))
        pvals_final = np.array([pval_map.get(s, np.nan) for s in sid])

        # Re-test pseudo-QTN SNPs individually (near-collinear with own
        # covariate → unreliable p-values in the joint scan).
        if pseudo_qtns:
            for idx in pseudo_qtns:
                others = [j for j in pseudo_qtns if j != idx]
                covar_no_self = _build_covar(others)
                ch = chroms[idx]
                ch_num_val = int(chroms_num[idx])
                K_use = K_by_chr.get(str(ch), K0)
                test_geno = geno_imputed[:, idx:idx + 1]
                pos_arr = np.array([[float(ch_num_val), 0.0,
                                     float(positions[idx])]])
                snp_data = _build_mlmm_snpdata_impl(
                    iid, np.array([sid[idx]]), test_geno, pos_arr,
                )
                try:
                    res = single_snp(
                        test_snps=snp_data, pheno=pheno_reader,
                        K0=K_use, covar=covar_no_self,
                        leave_out_one_chrom=False,
                    )
                    p = float(
                        pd.to_numeric(res["PValue"], errors="coerce").iloc[0]
                    )
                    if np.isfinite(p):
                        pvals_final[idx] = p
                except Exception:
                    pass
    else:
        # OLS final scan — standard FarmCPU (Liu et al. 2016).
        pvals_final = _ols_scan_pvals_fast(
            geno_imputed, y_vec, covar_reader=covar_final,
        )
        # Pseudo-QTN SNPs are collinear with their own covariate in the
        # final OLS scan, giving unreliable p-values.  Restore from the
        # last iterative OLS scan where they were tested without being
        # included as covariates.
        if pseudo_qtns and pvals is not None:
            for idx in pseudo_qtns:
                pvals_final[idx] = pvals[idx]

    farmcpu_df = pd.DataFrame({
        "SNP": sid,
        "Chr": chroms,
        "Pos": positions,
        "PValue": pvals_final,
        "Model": "FarmCPU",
    })

    # Pseudo-QTN summary table
    if pseudo_qtns:
        pseudo_qtn_table = pd.DataFrame({
            "Step": range(1, len(pseudo_qtns) + 1),
            "SNP": [sid[i] for i in pseudo_qtns],
            "Chr": [chroms[i] for i in pseudo_qtns],
            "Pos": [positions[i] for i in pseudo_qtns],
            "PValue_final": [float(pvals_final[i]) for i in pseudo_qtns],
        })
    else:
        pseudo_qtn_table = pd.DataFrame(
            columns=["Step", "SNP", "Chr", "Pos", "PValue_final"]
        )

    convergence_info = {
        "n_iterations": len(iteration_log),
        "converged": converged,
        "n_pseudo_qtns": len(pseudo_qtns),
        "log": iteration_log,
    }

    if verbose and progress_placeholder is not None:
        status = "converged" if converged else "stopped at max iterations"
        progress_placeholder.success(
            f"FarmCPU {status}: {len(pseudo_qtns)} pseudo-QTNs "
            f"in {len(iteration_log)} iterations."
        )
        progress_bar.empty()

    return farmcpu_df, pseudo_qtn_table, convergence_info


def run_farmcpu_cached(
    geno_key, y_key, iid, sid, chroms, chroms_num, positions,
    K0_key, pheno_reader_key, covar_reader_key,
    p_threshold=0.01, max_iterations=10, max_pseudo_qtns=15,
    final_scan="mlm", use_loco=True,
):
    """Cached wrapper: resolves session-state keys, delegates to run_farmcpu."""
    G = st.session_state.get(geno_key)
    Y = st.session_state.get(y_key)
    K0 = st.session_state.get(K0_key)
    pheno_reader = st.session_state.get(pheno_reader_key)
    covar_reader = st.session_state.get(covar_reader_key)
    if G is None or Y is None or K0 is None or pheno_reader is None:
        raise RuntimeError("Missing cached data in session_state.")
    return run_farmcpu(
        geno_imputed=np.asarray(G, dtype=np.float32, order="C"),
        sid=np.asarray(sid).astype(str),
        chroms=np.asarray(chroms).astype(str),
        chroms_num=np.asarray(chroms_num).astype(int),
        positions=np.asarray(positions).astype(int),
        iid=np.asarray(iid).astype(str),
        pheno_reader=pheno_reader,
        K0=K0,
        covar_reader=covar_reader,
        p_threshold=float(p_threshold),
        max_iterations=int(max_iterations),
        max_pseudo_qtns=int(max_pseudo_qtns),
        final_scan=str(final_scan),
        verbose=False,
        use_loco=bool(use_loco),
    )

if st is not None:
    run_farmcpu_cached = st.cache_data(
        show_spinner="Running FarmCPU (cached)…", max_entries=4
    )(run_farmcpu_cached)


# ============================================================
# Auto PC Selection via Lambda GC Scan
# ============================================================

def auto_select_pcs(
    geno_imputed, y, sid, chroms, chroms_num, positions,
    iid, Z_grm, chroms_grm, K_base, pcs_full,
    max_pcs=10, progress_callback=None, strategy="band",
    use_loco=True, band_lo=0.95, band_hi=1.05,
    parsimony_tolerance=0.02,
    model="mlm",
    farmcpu_p_threshold=0.01,
    farmcpu_max_iterations=10,
    farmcpu_max_pseudo_qtns=15,
    farmcpu_final_scan="mlm",
    mlmm_p_enter=1e-4,
    mlmm_max_cof=10,
):
    """
    Run a GWAS model at each PC count 0..max_pcs and compute λGC.

    Returns a DataFrame with columns:
        n_pcs, lambda_gc, delta_from_1, recommended

    Parameters
    ----------
    model : {'mlm', 'mlmm', 'farmcpu'}, default 'mlm'
        Which association model to scan. All three share the same band /
        closest_to_1 selection logic and the directional deflation guard.

        - ``'mlm'``: LOCO MLM via :func:`_run_gwas_impl`. The ``use_loco``
          flag controls whether per-chromosome kinships or the global GRM
          are used for the final scan.
        - ``'mlmm'``: MLMM via :func:`run_mlmm_research_grade_fast` with
          the global kinship (``K0``). ``use_loco`` is ignored since MLMM
          uses ``K0`` directly.
        - ``'farmcpu'``: FarmCPU via :func:`run_farmcpu`. ``use_loco``
          controls LOCO kinship in FarmCPU's MLM final scan.
    strategy : str, 'band' or 'closest_to_1'
        'band' (default): pick the smallest k where band_lo <= λGC <= band_hi.
            If no k falls within the band, falls back to the closest-to-1.0
            with adaptive parsimony tolerance: the smallest k whose
            delta_from_1 is within max(parsimony_tolerance, best_delta * 0.15)
            of the best delta. This avoids wasting PCs when all λGC values
            are far from 1.0 (e.g., strong QTL deflation).
        'closest_to_1': pick the PC count with λGC nearest to 1.0.
    band_lo : float
        Lower bound of the acceptable λGC band (default 0.95).
    band_hi : float
        Upper bound of the acceptable λGC band (default 1.05).
    parsimony_tolerance : float
        Minimum absolute tolerance for the band fallback (default 0.02).
        The actual tolerance used is max(parsimony_tolerance, best_delta * 0.15).
    farmcpu_p_threshold, farmcpu_max_iterations, farmcpu_max_pseudo_qtns,
    farmcpu_final_scan : FarmCPU tuning parameters used only when
        ``model='farmcpu'``. Defaults mirror :func:`run_farmcpu`.
    mlmm_p_enter, mlmm_max_cof : MLMM tuning parameters used only when
        ``model='mlmm'``. Defaults mirror :func:`run_mlmm_research_grade_fast`.
    """
    from gwas.plotting import compute_lambda_gc
    from gwas.kinship import _build_loco_kernels_impl

    model = str(model).lower()
    if model not in {"mlm", "mlmm", "farmcpu"}:
        raise ValueError(
            f"auto_select_pcs: model must be one of 'mlm', 'mlmm', "
            f"'farmcpu'; got {model!r}."
        )

    max_pcs = min(max_pcs, pcs_full.shape[1] if pcs_full is not None else 0)
    pheno_reader = PhenoData(iid=iid, val=y)

    rows = []
    for k in range(0, max_pcs + 1):
        if progress_callback:
            progress_callback(k, max_pcs + 1)

        # Build LOCO kernels (reused across PCs since kinship doesn't change)
        if k == 0:
            K0, K_by_chr, _ = _build_loco_kernels_impl(
                iid=iid, Z_grm=Z_grm, chroms_grm=chroms_grm, K_base=K_base,
            )
            if not use_loco:
                K_by_chr = {ch: K0 for ch in K_by_chr}

        # Build per-k PC covariates once (MLMM and FarmCPU expect a
        # CovarData, or None when k == 0).
        if k > 0 and pcs_full is not None:
            _pcs_k = pcs_full[:, :k]
            _covar_k = CovarData(iid=iid, val=_pcs_k)
        else:
            _covar_k = None

        try:
            if model == "mlm":
                gwas_df = _run_gwas_impl(
                    geno_imputed=geno_imputed, y=y, pcs_full=pcs_full,
                    n_pcs=k, sid=sid, positions=positions, chroms=chroms,
                    chroms_num=chroms_num, iid=iid,
                    _K0=K0, _K_by_chr=K_by_chr, pheno_reader=pheno_reader,
                )
                lam = compute_lambda_gc(gwas_df["PValue"].values, trim=False)
            elif model == "mlmm":
                mlmm_df, _ = run_mlmm_research_grade_fast(
                    geno_imputed=geno_imputed, sid=sid, chroms=chroms,
                    chroms_num=chroms_num, positions=positions, iid=iid,
                    pheno_reader=pheno_reader, K0=K0,
                    covar_reader=_covar_k,
                    p_enter=float(mlmm_p_enter),
                    max_cof=int(mlmm_max_cof),
                    verbose=False,
                )
                lam = compute_lambda_gc(mlmm_df["PValue"].values, trim=False)
            else:  # farmcpu
                fc_df, _, _ = run_farmcpu(
                    geno_imputed=geno_imputed, sid=sid, chroms=chroms,
                    chroms_num=chroms_num, positions=positions, iid=iid,
                    pheno_reader=pheno_reader, K0=K0,
                    covar_reader=_covar_k,
                    p_threshold=float(farmcpu_p_threshold),
                    max_iterations=int(farmcpu_max_iterations),
                    max_pseudo_qtns=int(farmcpu_max_pseudo_qtns),
                    final_scan=str(farmcpu_final_scan),
                    verbose=False,
                    use_loco=bool(use_loco),
                )
                lam = compute_lambda_gc(fc_df["PValue"].values, trim=False)
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            logging.exception("%s GWAS failed at k=%d PCs", model.upper(), k)
            lam = np.nan

        rows.append({
            "n_pcs": k,
            "lambda_gc": round(lam, 4) if np.isfinite(lam) else np.nan,
            "delta_from_1": round(abs(lam - 1.0), 4) if np.isfinite(lam) else np.nan,
        })

    df = pd.DataFrame(rows)

    # Select best PC count based on strategy
    df["recommended"] = ""
    valid = df["lambda_gc"].notna()
    if valid.any():
        if strategy == "band":
            # Smallest k where λGC falls within [band_lo, band_hi].
            # If nothing is in-band, fall back to closest-to-1.0 with
            # adaptive parsimony tolerance so we don't waste PCs when
            # all λGC values are far from 1.0 (e.g., strong QTL).
            in_band = df[valid & (df["lambda_gc"] >= band_lo)
                         & (df["lambda_gc"] <= band_hi)]
            if len(in_band) > 0:
                best_idx = in_band["n_pcs"].idxmin()
            else:
                best_delta = df.loc[valid, "delta_from_1"].min()
                tol = max(parsimony_tolerance, best_delta * 0.15)
                near_best = df[valid & (df["delta_from_1"]
                                        <= best_delta + tol)]
                best_idx = near_best["n_pcs"].idxmin()
            df.loc[best_idx, "recommended"] = "★"

            # --- Directional guard: deflated baseline ---
            # If λ(0) < band_lo the kinship is already over-correcting
            # on its own. Adding PCs cannot fix that — any "recovery"
            # back into band at k > 0 is either coincidence on an
            # oscillating λ curve or artificial inflation from
            # discarding signal. Force k=0 so the user sees (and
            # reports) the deflation instead of burying it under
            # spurious PCs.
            lam0 = df.loc[df["n_pcs"] == 0, "lambda_gc"].iloc[0]
            if pd.notna(lam0) and lam0 < 0.80:
                logging.warning(
                    "Severe lambda_GC deflation (%.3f at k=0). The "
                    "kinship model may be absorbing substantial trait "
                    "signal. Results are valid but interpret cautiously.",
                    lam0,
                )
            if pd.notna(lam0) and lam0 < band_lo:
                df["recommended"] = ""
                df.loc[df["n_pcs"] == 0, "recommended"] = "★"
                logging.warning(
                    "Deflated baseline lambda_GC=%.3f (< %.2f): forced "
                    "recommendation to k=0 (kinship is over-correcting; "
                    "adding PCs cannot repair it).",
                    lam0, band_lo,
                )
        else:
            # closest_to_1: pick PC count with λGC nearest to 1.0
            best_idx = df["delta_from_1"].idxmin()
            df.loc[best_idx, "recommended"] = "★"

    return df


def select_best_pc_from_lambdas(
    lambdas, strategy="band", band_lo=0.95, band_hi=1.05,
    parsimony_tolerance=0.02,
):
    """Pick best PC count from a list of lambda_GC values.

    Applies the same band-then-fallback logic as :func:`auto_select_pcs`,
    plus the directional deflation guard: if ``lambdas[0] < band_lo``
    the kinship is already over-correcting and adding PCs cannot repair
    it; selection is forced to ``k=0`` so the deflation is surfaced
    rather than buried under spurious PCs that artificially pull
    lambda_GC back into band. This keeps FarmCPU's auto-PC behaviour
    consistent with MLM's on deflated datasets.

    Parameters
    ----------
    lambdas : list[float]
        Lambda GC value for each k = 0, 1, ..., len(lambdas)-1.
    strategy : str
        'band' or 'closest_to_1'.

    Returns
    -------
    int
        Best PC count (index into *lambdas*).
    """
    n = len(lambdas)
    deltas = [abs(v - 1.0) if np.isfinite(v) else np.inf for v in lambdas]
    valid = [np.isfinite(v) for v in lambdas]

    if not any(valid):
        return 0

    if strategy != "band":
        # closest_to_1: pick PC count with lambda nearest to 1.0
        # (guard does not apply, matching auto_select_pcs)
        return int(np.argmin(deltas))

    def _pick(indices):
        """Band-then-fallback within a set of candidate indices."""
        in_band = [
            i for i in indices
            if valid[i] and band_lo <= lambdas[i] <= band_hi
        ]
        if in_band:
            return min(in_band)
        valid_in = [i for i in indices if valid[i]]
        if not valid_in:
            return indices[0] if indices else 0
        best_delta = min(deltas[i] for i in valid_in)
        tol = max(parsimony_tolerance, best_delta * 0.15)
        near = [i for i in valid_in if deltas[i] <= best_delta + tol]
        return min(near) if near else min(valid_in, key=lambda i: deltas[i])

    best = _pick(list(range(n)))

    # --- Directional deflation guard (mirrors auto_select_pcs) ---
    # If lambda(0) < band_lo, the kinship is already over-correcting.
    # Any in-band recovery at k > 0 on an oscillating lambda curve is
    # noise, not a genuine fix. Force k=0 unconditionally so deflation
    # is surfaced rather than buried under spurious PCs.
    if valid[0] and lambdas[0] < band_lo:
        if best != 0:
            best = 0
            logging.warning(
                "FarmCPU auto-PC: deflated baseline lambda_GC=%.3f "
                "(< %.2f); forced k=0 (adding PCs cannot repair an "
                "over-correcting kinship).",
                lambdas[0], band_lo,
            )

    return int(best)
