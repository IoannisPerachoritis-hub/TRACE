"""C1 T-49 — isolated-SNP rescue negative-acceptance suite (AC-N1/N4/N5, CLI).

Proves the rescue is strictly additive at the CLI surface: it never perturbs a
pre-existing bundle member, its escape hatch is byte-clean, it never enters the
block detector, and it added no code to gwas/ld.py. (AC-N2/N7 — block byte-
identity and detector defaults — are covered by tests/test_golden_* and
tests/test_defaults_pinned; AC-N8 — the CLI min_snps=3 literal — by
tests/test_defaults_pinned.)
"""
import pathlib
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

from cli import _build_parser, run_pipeline
from tests.test_cli_e2e import _write_test_data

_SKIP_NONDETERMINISTIC = ("report.html", "run_metadata.json")


def _run_cli(data_dir, out_dir, extra):
    data_dir.mkdir(parents=True, exist_ok=True)
    vcf, pheno = _write_test_data(data_dir)
    args = _build_parser().parse_args([
        "--vcf", str(vcf), "--pheno", str(pheno), "--trait", "TestTrait",
        "--output", str(out_dir), "--model", "mlm", "--no-report", "--no-plots",
        "--maf", "0.01", "--mac", "1", "--n-pcs", "2", *extra,
    ])
    run_pipeline(args)
    zp = list(out_dir.glob("*.zip"))[0]
    with zipfile.ZipFile(zp) as zf:
        return {n: zf.read(n) for n in zf.namelist()}


# ---- AC-N5: additive at the bundle level ----
def test_rescue_is_additive_and_escape_hatch_is_clean(tmp_path):
    """rescue ON vs OFF: every member present in both is byte-identical, and the
    only new members are the post-GWAS-visibility tables (Isolated_SNP_* /
    Significant_SNPs_* / Unblocked_SNPs_*), all gated by --no-isolated-rescue."""
    d = tmp_path / "data"
    on = _run_cli(d, tmp_path / "on", [])                          # rescue ON (default)
    off = _run_cli(d, tmp_path / "off", ["--no-isolated-rescue"])  # rescue OFF

    _allowed = ("Isolated_SNP", "Significant_SNPs", "Unblocked_SNPs", "SNP_view_index")
    new = set(on) - set(off)
    assert all(any(p in n for p in _allowed) for n in new), f"unexpected new members: {new}"

    for n in set(on) & set(off):
        if any(s in n for s in _SKIP_NONDETERMINISTIC):
            continue
        assert on[n] == off[n], f"pre-existing bundle member perturbed by the rescue: {n}"


# ---- AC-N4: the rescue never enters the block detector ----
def test_rescue_never_calls_the_detector(monkeypatch):
    import gwas.ld as _ld
    from gwas.isolated import run_isolated_snp_rescue
    from gwas.significance import rule_from_cli_args

    def _boom(*a, **k):
        raise AssertionError("rescue must not call the block detector")

    monkeypatch.setattr(_ld, "find_ld_clusters_genomewide", _boom)
    monkeypatch.setattr(_ld, "find_ld_blocks_graph", _boom)
    monkeypatch.setattr(_ld, "find_ld_blocks_from_genotypes", _boom)
    monkeypatch.setattr(_ld, "filter_contained_blocks", _boom)

    gwas = pd.DataFrame({"SNP": ["a", "b"], "Chr": ["1", "2"], "Pos": [100, 200],
                         "PValue": [1e-8, 1e-8]})
    rule = rule_from_cli_args(SimpleNamespace(sig_thresh="bonferroni"), 100, None)
    res = run_isolated_snp_rescue(
        gwas, pd.DataFrame(), rule,
        np.array(["1", "2"]), np.array([100, 200]), np.array(["a", "b"]),
        genes=None, seed_p_used=1e-5, top_n_used=0, edge_flank_bp=1000,
    )
    assert res.n_uncovered == 2  # completed with every detector entry point monkeypatched to raise


# ---- AC-N1 (structural): the detector file carries no rescue code ----
def test_ld_py_has_no_rescue_code():
    src = pathlib.Path("gwas/ld.py").read_text(encoding="utf-8")
    for token in ("find_unblocked_significant_snps", "run_isolated_snp_rescue",
                  "build_flanking_intervals", "annotate_isolated_intervals"):
        assert token not in src, f"rescue code leaked into gwas/ld.py: {token}"


# ---- AC-N8 (reinforce here too): the CLI passes min_snps=2 to detection ----
def test_cli_still_passes_min_snps_2():
    import ast
    tree = ast.parse(pathlib.Path("cli.py").read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if fn == "find_ld_clusters_genomewide":
                for kw in node.keywords:
                    if kw.arg == "min_snps":
                        found.append(ast.literal_eval(kw.value) if isinstance(kw.value, ast.Constant) else None)
    assert found and all(v == 2 for v in found), f"CLI min_snps at detection != 2: {found}"
