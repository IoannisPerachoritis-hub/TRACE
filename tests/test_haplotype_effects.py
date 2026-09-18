"""Effect-size helpers for the haplotype block GWAS: eta2 CI (noncentral-F
inversion), omega2, and the lead-SNP additive partial R^2.

The helpers are pure functions of (F, df1, df2) / (dosage, residualised phenotype),
so they are exercised with literal statistics rather than a genotype fixture. The
CI's real acceptance is the coverage check (test_eta2_ci_coverage).
"""
import numpy as np
import pytest
from scipy.stats import ncf

from gwas.haplotype import _eta2_confint, _omega2_from_f, _lead_snp_partial_r2


# ── _eta2_confint ─────────────────────────────────────────────

@pytest.mark.parametrize("F,df1,df2", [(19.86, 3, 140), (8.5, 1, 153),
                                       (2.0, 5, 135), (1.2, 2, 100)])
def test_eta2_ci_contains_point_estimate(F, df1, df2):
    """The eta2 point estimate df1*F/(df1*F+df2) lies inside its 95% CI."""
    lo, hi = _eta2_confint(F, df1, df2)
    eta2 = df1 * F / (df1 * F + df2)
    assert lo <= eta2 <= hi
    assert 0.0 <= lo < hi <= 1.0


def test_eta2_ci_back_substitution():
    """Inversion self-consistency: the returned bounds map to lambdas whose
    noncentral-F cdf at F is 0.975 (lower) and 0.025 (upper)."""
    F, df1, df2 = 19.86, 3, 140
    M = df1 + df2 + 1
    lo, hi = _eta2_confint(F, df1, df2)
    lam_lo = lo * M / (1 - lo)
    lam_hi = hi * M / (1 - hi)
    assert ncf.cdf(F, df1, df2, lam_lo) == pytest.approx(0.975, abs=1e-3)
    assert ncf.cdf(F, df1, df2, lam_hi) == pytest.approx(0.025, abs=1e-3)


def test_eta2_ci_monotonic_in_F():
    """A larger F shifts both CI bounds up."""
    lo1, hi1 = _eta2_confint(5.0, 2, 100)
    lo2, hi2 = _eta2_confint(25.0, 2, 100)
    assert lo2 > lo1 and hi2 > hi1


def test_eta2_ci_weak_effect_lower_bound_zero():
    """When even nc=0 puts F below the 0.975 quantile, the lower bound is 0."""
    # F just above 1 -> central-F cdf well below 0.975 -> lam_lo == 0 -> lo == 0
    lo, hi = _eta2_confint(1.05, 1, 150)
    assert lo == 0.0
    assert hi > 0.0


@pytest.mark.parametrize("F,df1,df2", [(np.nan, 3, 140), (0.0, 3, 140),
                                       (-1.0, 3, 140), (5.0, 0, 140), (5.0, 3, 0)])
def test_eta2_ci_undefined_returns_nan(F, df1, df2):
    lo, hi = _eta2_confint(F, df1, df2)
    assert np.isnan(lo) and np.isnan(hi)


def test_eta2_ci_coverage():
    """The 95% CI has nominal coverage. FIXED effect: draw F from its exact
    noncentral-F distribution for a fixed true eta2 = lam/(lam+M); a random-effect
    sim (redrawing group means) would wrongly fail. ~95% expected, 93-97% enforced."""
    rng = np.random.default_rng(2024)
    for df1, df2, te in [(3, 140, 0.15), (1, 150, 0.05), (5, 135, 0.30)]:
        M = df1 + df2 + 1
        lam = te * M / (1 - te)
        Fs = ncf.rvs(df1, df2, lam, size=2500, random_state=rng)
        cov = np.mean([lo <= te <= hi for lo, hi in (_eta2_confint(F, df1, df2) for F in Fs)])
        assert 0.93 <= cov <= 0.97, f"coverage {cov:.3f} for (df1={df1},df2={df2},eta2={te})"


# ── _omega2_from_f ────────────────────────────────────────────

def test_omega2_formula():
    F, df1, df2 = 19.86, 3, 140
    assert _omega2_from_f(F, df1, df2) == pytest.approx(df1 * (F - 1) / (df1 * F + df2 + 1))


def test_omega2_below_eta2_for_positive_effect():
    """omega2 is less upward-biased than eta2 (strictly smaller when F > 1)."""
    F, df1, df2 = 17.62, 4, 155
    eta2 = df1 * F / (df1 * F + df2)
    assert _omega2_from_f(F, df1, df2) < eta2


def test_omega2_negative_when_F_below_one():
    """F < 1 legitimately yields a negative omega2 (NOT clamped)."""
    assert _omega2_from_f(0.5, 3, 140) < 0.0


def test_omega2_zero_at_F_one():
    assert _omega2_from_f(1.0, 3, 140) == pytest.approx(0.0)


@pytest.mark.parametrize("F,df1,df2", [(np.nan, 3, 140), (5.0, 0, 140), (5.0, 3, 0)])
def test_omega2_undefined_returns_nan(F, df1, df2):
    assert np.isnan(_omega2_from_f(F, df1, df2))


# ── _lead_snp_partial_r2 ──────────────────────────────────────

def _setup(n=100, seed=0):
    rng = np.random.default_rng(seed)
    sid = np.array([f"s{i}" for i in range(5)])
    sample_ids = np.array([f"S{i}" for i in range(n)])
    test_samples = sample_ids                       # all retained
    geno = rng.integers(0, 3, (n, 5)).astype(float)
    return rng, sid, sample_ids, test_samples, geno


def test_lead_r2_perfect_correlation():
    rng, sid, sample_ids, test_samples, geno = _setup()
    dose = geno[:, 2]
    y = 3.0 * dose                                  # y perfectly determined by lead dosage
    r2 = _lead_snp_partial_r2("s2", sid, geno, sample_ids, test_samples, y, None)
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_lead_r2_uncorrelated_near_zero():
    rng, sid, sample_ids, test_samples, geno = _setup(n=500)
    y = rng.normal(0, 1, 500)                        # independent of any SNP
    r2 = _lead_snp_partial_r2("s2", sid, geno, sample_ids, test_samples, y, None)
    assert r2 < 0.05


def test_lead_r2_lead_absent_from_sid():
    rng, sid, sample_ids, test_samples, geno = _setup()
    y = rng.normal(0, 1, 100)
    assert np.isnan(_lead_snp_partial_r2("not_a_snp", sid, geno, sample_ids, test_samples, y, None))
    assert np.isnan(_lead_snp_partial_r2("", sid, geno, sample_ids, test_samples, y, None))


def test_lead_r2_monomorphic_dose_nan():
    rng, sid, sample_ids, test_samples, geno = _setup()
    geno[:, 2] = 1.0                                 # constant dosage
    y = rng.normal(0, 1, 100)
    assert np.isnan(_lead_snp_partial_r2("s2", sid, geno, sample_ids, test_samples, y, None))


def test_lead_r2_partials_out_pcs():
    """With X_pcs given, structure carried by a PC is removed before the R^2."""
    rng, sid, sample_ids, test_samples, geno = _setup(n=300, seed=1)
    pc = rng.normal(0, 1, 300)
    geno[:, 2] = (pc > 0).astype(float) + (pc > 1).astype(float)   # dose tracks the PC
    y_resid = rng.normal(0, 1, 300)                                # phenotype independent of the PC
    X_pcs = np.column_stack([np.ones(300), pc])
    r2 = _lead_snp_partial_r2("s2", sid, geno, sample_ids, test_samples, y_resid, X_pcs)
    assert r2 < 0.05                                # PC-driven dose explains ~nothing after partialling
