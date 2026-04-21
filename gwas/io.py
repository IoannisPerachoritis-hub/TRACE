import os
import tempfile
import allel
import numpy as np
import pandas as pd
try:
    import streamlit as st
except ImportError:
    st = None


def load_vcf_cached(vcf_bytes: bytes):
    """
    Fully robust VCF loader that:
    - detects gzip via magic bytes
    - sanitizes text VCFs
    - strips Excel/Notepad quotes
    - checks #CHROM exists pre-allel
    - guarantees allel.read_vcf compatibility
    """


    # ─────────────────────────────────────────────
    # 1. Fast gzip detection
    # ─────────────────────────────────────────────
    gz = vcf_bytes.startswith(b"\x1f\x8b")

    # ─────────────────────────────────────────────
    # 2. Sanitizer for plain-text VCFs
    # ─────────────────────────────────────────────
    def sanitize_vcf_text(text: str) -> str:
        cleaned = []
        for line in text.splitlines():
            l = line.strip()

            # Remove wrapping quotes
            if l.startswith('"') and l.endswith('"'):
                l = l[1:-1]

            # Replace doubled Excel quotes
            l = l.replace('""', '"')

            cleaned.append(l)
        return "\n".join(cleaned)

    # ─────────────────────────────────────────────
    # 3. Optional validation
    # ─────────────────────────────────────────────
    def assert_has_chrom_line(text: str):
        if "#CHROM" not in text:
            raise RuntimeError(
                "VCF file is missing the mandatory '#CHROM' header line. "
                "The file may be truncated or corrupted."
            )

    # ─────────────────────────────────────────────
    # 4. Sanitize only if plain text
    # ─────────────────────────────────────────────
    if not gz:
        txt = vcf_bytes.decode("utf-8-sig", errors="ignore")  # utf-8-sig auto-strips BOM
        txt = sanitize_vcf_text(txt)
        assert_has_chrom_line(txt)       # <── NEW: early clear error
        vcf_bytes = txt.encode("utf-8")
        suffix = ".vcf"
    else:
        suffix = ".vcf.gz"

    # ─────────────────────────────────────────────
    # 5. Optional debug preview
    # ─────────────────────────────────────────────
    if st is not None and st.session_state.get("vcf_debug", False) and not gz:
        st.info("VCF preview (sanitized, first 40 lines):")
        st.code("\n".join(txt.splitlines()[:40]), language="text")

    # ─────────────────────────────────────────────
    # 6. Write temp file
    # ─────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(vcf_bytes)
        temp_path = tmp.name

    # ─────────────────────────────────────────────
    # 7. Read with scikit-allel
    # ─────────────────────────────────────────────
    try:
        callset = allel.read_vcf(
            temp_path,
            fields=[
                "samples",
                "calldata/GT",
                "variants/CHROM",
                "variants/POS",
                "variants/ALT",  # needed for biallelic filter
                "variants/REF",  # useful for SNP IDs / sanity
                "variants/ID",  # if present in the VCF, use as rsID-like identifier
                # Imputation quality fields (silently skipped if absent)
                "variants/DR2",       # Beagle
                "variants/R2",        # minimac4
                "variants/INFO",      # IMPUTE2
                "variants/AR2",       # alternate
                "variants/IMP_QUAL",  # other tools
            ],
        )
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    return callset

if st is not None:
    load_vcf_cached = st.cache_data(
        show_spinner="Reading VCF file…"
    )(load_vcf_cached)


# ── Imputation quality score extraction ─────────────────────
_INFO_QUALITY_FIELDS = ("DR2", "R2", "INFO", "AR2", "IMP_QUAL")


def _extract_info_scores(callset):
    """
    Extract per-variant imputation quality scores from VCF INFO fields.

    Checks common field names used by Beagle (DR2), minimac4 (R2),
    IMPUTE2 (INFO), and others (AR2, IMP_QUAL).

    Returns
    -------
    scores : ndarray of float32 or None
    field_name : str or None  (the INFO key that was found)
    """
    for field in _INFO_QUALITY_FIELDS:
        key = f"variants/{field}"
        arr = callset.get(key, None)
        if arr is not None:
            raw = np.asarray(arr).ravel()
            scores = pd.to_numeric(raw, errors="coerce").astype(np.float32)
            if scores.size > 0 and np.any(np.isfinite(scores)):
                return scores, field
    return None, None


def _clean_chr_series(chr_array_like, canonical=None):
    """
    Chromosome normalizer for diploid species.

    Strips assembly prefixes and normalizes to bare integers.
    Supports any diploid species — chromosome count is auto-detected
    from the VCF (any value that parses to a positive integer is kept).

    Recognized prefix formats (case-insensitive):
      - Plain numeric: ``1``, ``01``, ``001``
      - Generic: ``chr1``, ``Chr01``, ``chromosome_1``, ``Chromosome01``
      - Tomato: ``SL4.0ch01``, ``SL2.50ch01``, ``ITAG4.0ch01``
      - Pepper: ``Ca01``, ``Ca_01``, ``Ca_chr01``, ``CaP_Chr01``
      - Potato: ``ST4.03ch01``
      - Rice: ``Os01``, ``Os_01``
      - Soybean: ``Gm01``, ``Gm_01``
      - Arabidopsis: ``At1``, ``At01``
      - Maize: ``Zm01`` (note: plain ``chr1`` also common)
      - Common bean: ``Pv01``
      - Pea: ``Ps01``
      - Peanut: ``Ah01``
      - Barley: ``1H``, ``chr1H`` (trailing H stripped)
    Everything else (scaffolds, MT, Pt, unplaced contigs) -> ``'ALT'``

    Parameters
    ----------
    canonical : tuple of str or None
        If None (default), auto-detect: all positive-integer chromosome
        labels are canonical. If provided, only those values are kept
        (e.g. ``("1","2","3")`` to restrict to 3 chromosomes).
    """
    raw = pd.Series(chr_array_like).astype(str).str.strip()

    # ── Prefix stripping (order matters) ──────────────────────
    # 1. NCBI-style: chromosome_1, Chromosome01 (must come before chr)
    s = raw.str.replace(r"(?i)^chromosome[_\-]?", "", regex=True)
    # 2. Generic chr prefix: chr1, Chr01
    s = s.str.replace(r"(?i)^chr", "", regex=True)
    # 3. Assembly-versioned ch suffix: SL4.0ch01, ST4.03ch01, ITAG4.0ch01
    s = s.str.replace(r"(?i)^[a-z]+[\d\.]*ch(?=\d)", "", regex=True)
    # 4. Genus/species prefixes (2-3 letters + optional separator before digit)
    #    Os (rice), Gm (soybean), At (arabidopsis), Zm (maize),
    #    Pv (common bean), Ps (pea), Ah (peanut)
    s = s.str.replace(r"(?i)^(?:os|gm|at|zm|pv|ps|ah)[_\-]?(?=\d)", "", regex=True)
    # 5. Pepper: Ca01, Ca_01, CaP_Chr01
    s = s.str.replace(r"(?i)^ca[p]?[_\-]?(?:chr)?(?=\d)", "", regex=True)
    # 6. Barley trailing H: 1H → 1, 02H → 02
    s = s.str.replace(r"(?i)^(\d+)h$", r"\1", regex=True)
    # 7. Strip leading zeros: 01 → 1, 001 → 1
    s = s.str.replace(r"^0+", "", regex=True)

    # keep only pure digits now
    is_digit = s.str.fullmatch(r"\d+")
    num = pd.Series(np.where(is_digit, s, np.nan))

    if canonical is None:
        # Auto-detect: any positive integer is a valid chromosome
        lab = num.where(is_digit, other="ALT").astype(str)
    else:
        lab = num.where(num.isin(canonical), other="ALT").astype(str)
    numcode = lab.where(lab != "ALT", other="0").astype(int)

    uniq = pd.Index(sorted({int(x) for x in lab[lab!="ALT"].unique()}, key=int)).astype(str).tolist()
    if "ALT" in lab.values:
        uniq = uniq + ["ALT"]

    order_map = {k: i for i, k in enumerate(uniq)}
    # Force numpy return type — `.values` on a PyArrow-backed Series returns
    # ArrowStringArray (pandas>=2.0 with pyarrow installed), which Streamlit's
    # @st.cache_data cannot hash. The two-step asarray is robust across the
    # pandas 2.x range we declare in pyproject.toml.
    return (
        np.asarray(lab.to_numpy(dtype=object), dtype=str),
        np.asarray(numcode.to_numpy()),
        uniq,
        order_map,
    )