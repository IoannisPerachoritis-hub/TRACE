"""An infinite trait value (for example from a log of zero taken before the file was written;
TRACE's own log transform refuses values at or below zero) stops the run before the scan, with
a message naming the count and the first samples; it used to crash inside FaST-LMM. A missing
value is still dropped by phenotype QC, as before. The transforms keep an infinite value
infinite (the rank transform included), so it stops whatever --norm says. Both modes."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gwas.io as gio
from gwas import qc as gqc
from gwas.genostore import ingest_vcf
from gwas.pipeline import ParsedGenotypes
from gwas.storepipe import TraitStore

_load = getattr(gio.load_vcf_cached, "__wrapped__", gio.load_vcf_cached)


def _files(tmp_path, values, n=30, m=40, seed=6):
    rng = np.random.default_rng(seed)
    samples = [f"S{i:02d}" for i in range(n)]
    lines = ["##fileformat=VCFv4.2", '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">',
             "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples)]
    for j in range(m):
        g = "\t".join(["0/0", "0/1", "1/1"][k] for k in rng.integers(0, 3, n))
        lines.append(f"1\t{100 + 50 * j}\t.\tA\tT\t.\tPASS\t.\tGT\t{g}")
    vcf = tmp_path / "t.vcf"
    vcf.write_text("\n".join(lines) + "\n")
    trait = rng.uniform(1.0, 5.0, size=n).round(3).astype(object)
    for i, v in values.items():
        trait[i] = v
    pd.DataFrame({"T": trait}, index=pd.Index(samples, name="id")).to_csv(tmp_path / "t_pheno.csv")
    return vcf, tmp_path / "t_pheno.csv"


def _phenotype_qc(tmp_path, mode, values, norm):
    vcf, pheno_path = _files(tmp_path, values)
    pheno = pd.read_csv(pheno_path, index_col=0)
    if mode == "memory":
        parsed = ParsedGenotypes.from_callset(_load(Path(vcf).read_bytes()))
        fns = gqc._pipeline_harmonize_ids, gqc._pipeline_phenotype_qc
    else:
        parsed = ingest_vcf(vcf, tmp_path / "s")
        st = TraitStore(parsed)
        fns = st.harmonize_ids, st.phenotype_qc
    try:
        geno_df, ph, *_ = fns[0](parsed.genotypes, parsed.samples, parsed.chroms, parsed.positions,
                                 parsed.vid, parsed.ref, parsed.alt, pheno, parsed.info_scores)
        return fns[1](geno_df, ph, "T", norm, 0.5)
    finally:
        if mode == "disk":
            parsed.store.close()


@pytest.mark.parametrize("mode", ["memory", "disk"])
def test_log_of_zero_stops_before_the_scan(tmp_path, mode):
    """A trait column the user took the log of, with a zero in it: -inf in the file."""
    with pytest.raises(ValueError) as e:
        _phenotype_qc(tmp_path, mode, {3: "-inf", 11: "-inf"}, "none")
    msg = str(e.value)
    assert "2 infinite trait values" in msg and "log of zero" in msg and "fix or remove them" in msg
    assert "S03" in msg and "S11" in msg


@pytest.mark.parametrize("mode", ["memory", "disk"])
def test_infinite_raw_values_are_named_up_to_ten(tmp_path, mode):
    with pytest.raises(ValueError) as e:
        _phenotype_qc(tmp_path, mode, {i: "inf" for i in range(12)}, "none")
    msg = str(e.value)
    assert "12 infinite trait values" in msg
    assert "S09" in msg and "S10" not in msg and "S11" not in msg


@pytest.mark.parametrize("mode", ["memory", "disk"])
def test_missing_values_are_still_dropped(tmp_path, mode):
    geno_df, ph, y = _phenotype_qc(tmp_path, mode, {2: np.nan, 5: np.nan}, "none")
    assert len(y) == 28 and np.isfinite(y).all()


@pytest.mark.parametrize("mode", ["memory", "disk"])
def test_rank_transform_leaves_it_infinite_so_it_stops_too(tmp_path, mode):
    with pytest.raises(ValueError, match="1 infinite trait values"):
        _phenotype_qc(tmp_path, mode, {4: "inf"}, "int")
