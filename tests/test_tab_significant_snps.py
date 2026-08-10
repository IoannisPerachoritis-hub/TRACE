"""2c — significant-SNP LD tab: structure + the build-path contract it relies on."""
import dataclasses
import inspect

import numpy as np
import pandas as pd

from gwas.significance import rule_from_streamlit
from gwas.sigtable import build_significant_snp_table, project_unblocked
from pages._ld_tabs import LDContext
from pages._ld_tabs import tab_significant_snps as T


def test_ldcontext_has_sig_tab_fields():
    fields = {f.name for f in dataclasses.fields(LDContext)}
    assert {"geno_dosage_raw", "meff_val", "sig_rule_label"} <= fields


def test_render_takes_ctx_only():
    assert list(inspect.signature(T.render).parameters) == ["ctx"]


def test_tab_build_path_complete_and_projects_unblocked():
    """Mirror exactly what render() does (minus the st.* calls): resolve the rule the
    same way, build the complete table, project the unblocked rows."""
    n = 30
    rng = np.random.default_rng(0)
    sid = np.array([f"s{i}" for i in range(n)])
    chroms = np.array(["1"] * n)
    positions = np.arange(1000, 1000 + n * 100, 100)
    pvals = np.full(n, 0.5)
    pvals[[3, 7]] = 1e-9                                   # two genome-wide-significant SNPs
    gwas = pd.DataFrame({"SNP": sid, "Chr": chroms, "Pos": positions, "PValue": pvals,
                         "Beta_OLS": 0.3, "SE_OLS": 0.1, "Significant_Meff": pvals < 1e-6})
    geno = rng.integers(0, 3, size=(20, n)).astype(float)

    rule = rule_from_streamlit("M_eff — Li & Ji (LD-aware Bonferroni)", n, n)
    sig = build_significant_snp_table(
        gwas, pd.DataFrame(), rule, chroms, positions, sid,
        geno_dosage_raw=geno, genes=None, seed_p_used=1e-5, top_n_used=0,
        edge_flank_bp=300_000)

    assert len(sig) == 2                                   # complete: both sig SNPs present
    assert (sig["Block_Status"] != "in_block").all()       # no blocks -> every row unblocked
    unb = project_unblocked(sig)
    assert set(unb["SNP"].astype(str)) == {"s3", "s7"}     # unblocked projection = the sig set
