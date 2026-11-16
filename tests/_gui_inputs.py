"""Inputs for the GUI tests that start the GWAS page from persisted inputs.

The GWAS page reads its VCF from a file (the location typed in its location box, or the upload
it saved once in its work folder); the session holds only the path, ``_persist_vcf_path``, never
the bytes. Tests whose VCF is text in memory write it to a file here first. The file's content
hash is the page's ``vcf_hash``, so the caches and keys built on it are those of the same bytes
uploaded."""
import atexit
import hashlib
import shutil
import tempfile
from pathlib import Path

_DIR = Path(tempfile.mkdtemp(prefix="trace_gui_tests_"))
atexit.register(shutil.rmtree, _DIR, True)


def vcf_file(text_or_bytes, name="genotypes.vcf"):
    """The path of a file holding ``text_or_bytes`` (written once per content)."""
    data = text_or_bytes.encode("utf-8") if isinstance(text_or_bytes, str) else bytes(text_or_bytes)
    path = _DIR / f"{hashlib.blake2b(data, digest_size=8).hexdigest()}_{name}"
    if not path.exists():
        path.write_bytes(data)
    return str(path)
