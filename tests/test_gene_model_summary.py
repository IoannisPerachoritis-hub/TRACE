"""summarize_gene_model — per-chromosome upload-time validation summary."""
import pandas as pd

from annotation import summarize_gene_model

_COLS = ["Chr", "n_genes", "min_start", "max_end", "span_Mb"]


def _gm():
    # realistic Mb-scale gene coordinates
    return pd.DataFrame({
        "Chr": ["1", "1", "2", "2", "2"],
        "Start": [1_000_000, 5_000_000, 100_000, 2_000_000, 500_000],
        "End": [1_002_000, 9_000_000, 600_000, 3_000_000, 800_000],
        "Strand": ["+", "-", "+", "+", "-"],
        "Gene_ID": ["g1", "g2", "g3", "g4", "g5"],
    })


def test_per_chr_counts_and_ranges():
    s = summarize_gene_model(_gm())
    assert list(s.columns) == _COLS
    r = s.set_index("Chr")
    assert int(r.loc["1", "n_genes"]) == 2
    assert int(r.loc["1", "min_start"]) == 1_000_000 and int(r.loc["1", "max_end"]) == 9_000_000
    assert int(r.loc["2", "n_genes"]) == 3
    assert int(r.loc["2", "min_start"]) == 100_000 and int(r.loc["2", "max_end"]) == 3_000_000
    # span_Mb = round((max_end - min_start) / 1e6, 2)
    assert abs(float(r.loc["1", "span_Mb"]) - 8.0) < 1e-9
    assert abs(float(r.loc["2", "span_Mb"]) - 2.9) < 1e-9


def test_chromosome_order_is_numeric():
    # rows sorted by chromosome number, not lexicographically
    gm = pd.DataFrame({
        "Chr": ["10", "2", "1"],
        "Start": [1, 1, 1], "End": [2, 2, 2],
        "Gene_ID": ["a", "b", "c"],
    })
    s = summarize_gene_model(gm)
    assert list(s["Chr"]) == ["1", "2", "10"]


def test_empty_and_none_return_stable_schema():
    s = summarize_gene_model(pd.DataFrame(columns=["Chr", "Start", "End", "Gene_ID"]))
    assert len(s) == 0 and list(s.columns) == _COLS
    s2 = summarize_gene_model(None)
    assert len(s2) == 0 and list(s2.columns) == _COLS
