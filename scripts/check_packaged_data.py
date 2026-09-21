"""Verify an INSTALLED TRACE can still find its gene models.

Run this against a wheel installed somewhere other than the source tree:

    python -m build --wheel
    pip install --no-deps --target /tmp/wheel-install dist/*.whl
    cd /tmp && PYTHONPATH=/tmp/wheel-install python <repo>/scripts/check_packaged_data.py

The guard is deliberately stronger than "the files are in the wheel". The defect
it exists for was not "files absent from the archive" but "the path the code
looks at does not exist" -- and those come apart. Ship data/ under a name or a
nesting the resolver never reaches and a members-of-the-wheel check passes green
with the bug fully intact. So this imports ``annotation`` from whatever is on
sys.path, refuses to proceed if that turns out to be the source tree, and then
asks the resolver itself -- the same function the CLI calls -- for every build,
checking each returned file exists AND is non-empty.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_failures = []


def _fail(msg):
    _failures.append(msg)
    print(f"FAIL  {msg}")


def _ok(msg):
    print(f"ok    {msg}")


def main():
    import annotation

    module = Path(annotation.__file__).resolve()
    print(f"annotation imported from: {module}")

    # An import that resolved to the source tree tells us nothing about the
    # wheel -- the source tree always has data/ beside it.
    if module == (_REPO_ROOT / "annotation.py") or _REPO_ROOT in module.parents:
        _fail(
            f"imported annotation from the SOURCE TREE ({module}); this check must "
            f"run against an install. Set PYTHONPATH to the install directory and "
            f"run from a directory outside {_REPO_ROOT}."
        )
        return 1
    _ok("annotation came from an install, not the source tree")

    if not hasattr(annotation, "resolve_gene_model"):
        _fail("the installed annotation module has no resolve_gene_model(); this "
              "install predates WO-PKGDATA-01 or the resolver was removed")
        return 1

    for build in ("SL3", "SL4"):
        res = annotation.resolve_gene_model("tomato", genome_build=build)
        if not res.found:
            _fail(f"tomato/{build}: gene model not found ({res.status})")
            continue

        for label, path in (("gene model", res.gene_model),
                            ("descriptions", res.descriptions)):
            if path is None:
                _fail(f"tomato/{build}: {label} resolved to None")
                continue
            if not path.exists():
                _fail(f"tomato/{build}: {label} {path} does not exist")
                continue
            size = path.stat().st_size
            if size == 0:
                _fail(f"tomato/{build}: {label} {path} is empty")
                continue
            _ok(f"tomato/{build}: {label} {path.name} present ({size:,} bytes)")

    if _failures:
        print(f"\n{len(_failures)} check(s) failed -- the installed package cannot "
              f"annotate. See WO-PKGDATA-01.")
        return 1
    print("\nAll packaged-data checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
