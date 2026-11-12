"""Compare two TRACE results ZIPs file by file.

    python scripts/compare_result_zips.py RUN_A.zip RUN_B.zip

Use it to check that a run with the genotype table in memory and a run with it on disk gave
the same results (docs/large_datasets.md). Every file of the two ZIPs is compared byte for
byte, with three exceptions:

* in run_metadata.json, the three keys that say where the genotype table was kept and why
  (they differ between the two modes; everything else in the file is compared);
* in report.html, the line with the time the report was made;
* the ZIP file names themselves, which hold the time of the run.

Exit code 0 when everything else is identical, 1 when a file differs or is missing.
"""
import json
import re
import sys
import zipfile

WHERE_KEYS = (
    "Genotype table",
    "Genotype table choice",
    "Genotype table reason",
)
_GENERATED = re.compile(rb"<strong>Generated:</strong>[^\n]*")


def _normalise(name, data):
    if name.endswith("run_metadata.json"):
        meta = json.loads(data.decode("utf-8"))
        for k in WHERE_KEYS:
            meta.pop(k, None)
        return json.dumps(meta, indent=2, sort_keys=True).encode("utf-8")
    if name.endswith("report.html"):
        return _GENERATED.sub(b"", data)
    return data


def compare(path_a, path_b, out=sys.stdout):
    with zipfile.ZipFile(path_a) as za, zipfile.ZipFile(path_b) as zb:
        names_a, names_b = set(za.namelist()), set(zb.namelist())
        bad = 0
        for name in sorted(names_a | names_b):
            if name not in names_b:
                print(f"only in {path_a}: {name}", file=out)
                bad += 1
            elif name not in names_a:
                print(f"only in {path_b}: {name}", file=out)
                bad += 1
            elif _normalise(name, za.read(name)) != _normalise(name, zb.read(name)):
                print(f"differs: {name}", file=out)
                bad += 1
        n = len(names_a & names_b)
    print(f"{n} files compared; {'all identical' if not bad else f'{bad} differ'}"
          " (where the genotype table was kept is not compared)", file=out)
    return bad


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__)
        return 2
    return 1 if compare(argv[0], argv[1]) else 0


if __name__ == "__main__":
    sys.exit(main())
