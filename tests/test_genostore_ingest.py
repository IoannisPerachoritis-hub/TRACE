"""The chunked VCF ingest of the on-disk table (gwas.genostore.ingest_vcf) and the streaming
clean-up of plain-text VCFs (gwas.io.sanitize_plain_vcf_stream).

The reference is today's in-memory read: load_vcf_cached, then ParsedGenotypes.from_callset.
The ingest must give the same parsed calls and metadata at every chunk length, and the
streaming clean-up must write exactly the bytes load_vcf_cached writes to its temporary file.
"""
import gzip
import io as _io
from pathlib import Path

import numpy as np
import pytest

import gwas.io as gio
from gwas import genostore as gs
from gwas.genostore import DeferredParseError, LazyColumns, ingest_vcf
from gwas.pipeline import ParsedGenotypes

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = [REPO / "tests" / "data" / "example_v1" / "example.vcf.gz", REPO / "examples" / "example.vcf.gz"]

_load = getattr(gio.load_vcf_cached, "__wrapped__", gio.load_vcf_cached)   # not the Streamlit cache


def _today(vcf_path):
    return ParsedGenotypes.from_callset(_load(Path(vcf_path).read_bytes()))


def _vcf_text(n=13, m=61, seed=3, quality=True, crlf=False, quotes=False):
    """A plain-text VCF with the cases the parse filters: multi-allelic and missing-ALT
    records, missing calls, IDs '.', a quality field, uneven lines."""
    rng = np.random.default_rng(seed)
    lines = ["##fileformat=VCFv4.2",
             '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">']
    if quality:
        lines.append('##INFO=<ID=DR2,Number=1,Type=Float,Description="Dosage R2">')
    lines.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                 + "\t".join(f"S{i:02d}" for i in range(n)))
    calls = ["0/0", "0/1", "1/1", "./.", "1/0", "0|1"]
    for j in range(m):
        alt = "T" if j % 11 else ("T,G" if j % 2 else ".")
        vid = "." if j % 3 else f"rs{j}"
        info = f"DR2={rng.random():.3f}" if quality else "."
        gt = "\t".join(calls[int(k)] for k in rng.integers(0, len(calls), n))
        lines.append(f"chr{1 + j // 25}\t{1000 + 37 * j}\t{vid}\tA\t{alt}\t.\tPASS\t{info}\tGT\t{gt}")
    if quotes:
        lines = [f'"{l}"' if i % 7 == 3 else l.replace("PASS", 'PA""SS') for i, l in enumerate(lines)]
    sep = "\r\n" if crlf else "\n"
    return sep.join(lines) + sep


def _assert_same_parse(parsed, ref):
    G = np.asarray(LazyColumns(parsed.store, np.arange(parsed.store.M)))
    assert G.dtype == np.float32
    assert np.ascontiguousarray(G).tobytes() == np.ascontiguousarray(ref.genotypes).tobytes()
    for name in ("samples", "chroms", "positions", "vid", "ref", "alt"):
        a, b = getattr(parsed, name), getattr(ref, name)
        if b is None:
            assert a is None, name
        else:
            assert a.dtype == b.dtype and np.array_equal(a, b), name
    if ref.info_scores is None:
        assert parsed.info_scores is None
    else:
        assert parsed.info_scores.dtype == ref.info_scores.dtype
        assert np.ascontiguousarray(parsed.info_scores).tobytes() == np.ascontiguousarray(ref.info_scores).tobytes()
    assert parsed.info_field == ref.info_field
    g = np.asarray(ref.genotypes)
    st = parsed.stats
    assert np.array_equal(st["n1"], (g == 1).sum(axis=0))
    assert np.array_equal(st["n2"], (g == 2).sum(axis=0))
    assert np.array_equal(st["nmiss"], np.isnan(g).sum(axis=0))
    assert np.array_equal(st["row_nmiss"], np.isnan(g).sum(axis=1))
    assert st["gmax"] == int(np.nanmax(g))


@pytest.mark.parametrize("chunk", [1, 97, 997, None])
@pytest.mark.parametrize("which", [0, 1])
def test_ingest_equals_todays_parse_examples(tmp_path, which, chunk):
    vcf = EXAMPLES[which]
    parsed = ingest_vcf(vcf, tmp_path / "s", chunk_records=chunk)
    try:
        _assert_same_parse(parsed, _today(vcf))
        assert parsed.ingest["gz"] is True
    finally:
        parsed.store.close()


def test_the_shared_parse_is_read_only(tmp_path):
    """The GUI's sessions share one StoreParsed: its metadata and its count arrays cannot be
    changed in place."""
    parsed = ingest_vcf(EXAMPLES[0], tmp_path / "s")
    try:
        arrays = {name: getattr(parsed, name) for name in ("samples", "chroms", "positions")}
        arrays.update({f"stats[{k}]": v for k, v in parsed.stats.items() if isinstance(v, np.ndarray)})
        assert {"stats[n1]", "stats[n2]", "stats[nmiss]", "stats[row_nmiss]", "stats[vmax]"} <= set(arrays)
        for name, arr in arrays.items():
            assert not arr.flags.writeable, name
        with pytest.raises(ValueError):
            parsed.stats["row_nmiss"][0] = 0
    finally:
        parsed.store.close()


@pytest.mark.parametrize("chunk", [1, 7, 97, None])
@pytest.mark.parametrize("variant", ["plain", "crlf_quotes", "gzip", "no_quality"])
def test_ingest_equals_todays_parse_edge_cases(tmp_path, variant, chunk):
    text = _vcf_text(crlf=variant == "crlf_quotes", quotes=variant == "crlf_quotes",
                     quality=variant != "no_quality")
    vcf = tmp_path / ("in.vcf.gz" if variant == "gzip" else "in.vcf")
    if variant == "gzip":
        vcf.write_bytes(gzip.compress(text.encode()))
    else:
        vcf.write_bytes(text.encode())
    parsed = ingest_vcf(vcf, tmp_path / "s", chunk_records=chunk)
    try:
        _assert_same_parse(parsed, _today(vcf))
        assert not (tmp_path / "s" / "sanitized.vcf").exists()      # the cleaned copy is removed
    finally:
        parsed.store.close()


def test_ingest_stops_like_todays_parse(tmp_path):
    header_only = tmp_path / "h.vcf"
    header_only.write_text(_vcf_text(m=0))
    with pytest.raises(DeferredParseError) as e:
        ingest_vcf(header_only, tmp_path / "s1")
    with pytest.raises(ValueError) as t:
        _today(header_only)
    assert str(e.value.original) == str(t.value) and isinstance(e.value.original, ValueError)

    no_chrom = tmp_path / "n.vcf"
    no_chrom.write_text("##fileformat=VCFv4.2\n1\t100\t.\tA\tT\t.\t.\t.\tGT\t0/1\n")
    with pytest.raises(RuntimeError, match="#CHROM"):
        ingest_vcf(no_chrom, tmp_path / "s2")
    with pytest.raises(RuntimeError, match="#CHROM"):
        _load(no_chrom.read_bytes())

    with pytest.raises(FileExistsError):                       # never into an existing folder
        ingest_vcf(EXAMPLES[0], tmp_path / "s1")


# ------------------------------------------------------------------ the streaming clean-up
def _todays_bytes(data, monkeypatch):
    """The bytes load_vcf_cached writes to its temporary file for ``data`` (its own code runs;
    only the final allel.read_vcf is replaced)."""
    written = []

    class _Tmp(_io.BytesIO):
        name = "unused.vcf"

        def __init__(self, *a, **k):
            super().__init__()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            written.append(self.getvalue())

    monkeypatch.setattr(gio.tempfile, "NamedTemporaryFile", _Tmp)
    monkeypatch.setattr(gio.allel, "read_vcf", lambda *a, **k: {})
    monkeypatch.setattr(gio.os, "remove", lambda p: None)
    try:
        _load(data)
    except RuntimeError:
        return None                                            # no '#CHROM' line: today stops
    return written[0]


def _streamed(data, tmp_path, read_bytes):
    src, dst = tmp_path / "src.vcf", tmp_path / "dst.vcf"
    src.write_bytes(data)
    ok = gio.sanitize_plain_vcf_stream(src, dst, read_bytes=read_bytes)
    return dst.read_bytes() if ok else None


def _variants_of(text):
    """The fixture's first lines rewritten with every line ending and quoting case."""
    head = "\n".join(text.splitlines()[:12])
    yield head.encode()
    yield head.replace("\n", "\r\n").encode()
    yield head.replace("\n", "\r").encode()
    yield ("﻿" + head).encode()
    yield ('"' + head.replace("\n", '"\n"') + '"').encode()
    yield head.replace("\n", "\r\n", 3).replace("\n", " ", 1).encode()
    yield (head + "\n\n\r\n").encode()
    yield head.replace("PASS", 'PA""SS').encode() + b"\xff\xfe broken \xc3"


@pytest.mark.parametrize("read_bytes", [1, 2, 3, 7, 64, 4096, 8 * 2**20])
def test_streaming_cleanup_equals_todays_bytes(tmp_path, monkeypatch, read_bytes):
    datas = list(_variants_of(_vcf_text(crlf=False)))
    datas.append(gzip.decompress(EXAMPLES[0].read_bytes())[:20000])
    rng = np.random.default_rng(read_bytes)
    alphabet = np.frombuffer(b'ab\t"\r\n#CHROM \xef\xbb\xbf\xc3\xa9\xff', dtype=np.uint8)
    for _ in range(150):
        k = int(rng.integers(0, 60))
        datas.append(bytes(rng.choice(alphabet, size=k)))
    for d in datas:
        assert _streamed(d, tmp_path, read_bytes) == _todays_bytes(d, monkeypatch), d[:80]


def test_store_module_is_unused_by_default():
    """The default (in-memory) path never imports the store's ingest."""
    assert gs.StoreParsed.__dataclass_fields__["is_store"].default is True
    assert not getattr(_today(EXAMPLES[0]), "is_store", False)
