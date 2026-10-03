"""Multivariate Spearman's rho, Kendall's tau and Blomqvist's beta."""

import numpy as np
import pytest
from scipy import stats

import copul as cp
from copul.multivariate import (
    ClaytonND,
    FunctionalCopulaND,
    GaussianND,
    GumbelND,
    IndependenceND,
    MixtureND,
    StudentTND,
    UpperFrechetND,
    blomqvists_beta_nd,
    kendalls_tau_nd,
    sample_blomqvists_beta_nd,
    sample_kendalls_tau_nd,
    sample_spearmans_rho_nd,
    spearmans_rho_h,
    spearmans_rho_nd,
)

R3 = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])


def test_normalizing_constant():
    assert spearmans_rho_h(2) == pytest.approx(3.0)
    assert spearmans_rho_h(3) == pytest.approx(1.0)
    assert spearmans_rho_h(4) == pytest.approx(5 / 11)


@pytest.mark.parametrize("d", [2, 3, 5])
def test_independence_and_comonotonicity(d):
    P, M = IndependenceND(d), UpperFrechetND(d)
    for kind in (1, 2, 3):
        assert spearmans_rho_nd(P, kind=kind) == 0.0
        assert spearmans_rho_nd(M, kind=kind) == 1.0
    assert kendalls_tau_nd(P) == 0.0 and kendalls_tau_nd(M) == 1.0
    assert blomqvists_beta_nd(P) == 0.0 and blomqvists_beta_nd(M) == 1.0


@pytest.mark.parametrize("d", [3, 4])
def test_generic_routes_at_independence_and_comonotonicity(d):
    """The numerical routes (no class shortcuts) give 0 at Pi_d and 1 at M_d."""
    Pi = FunctionalCopulaND(lambda U: U.prod(axis=1), d, rvs=lambda n, r: r.random((n, d)))
    M = FunctionalCopulaND(
        lambda U: U.min(axis=1), d, rvs=lambda n, r: np.repeat(r.random((n, 1)), d, axis=1)
    )
    assert blomqvists_beta_nd(Pi) == pytest.approx(0.0, abs=1e-14)
    assert blomqvists_beta_nd(M) == pytest.approx(1.0, abs=1e-14)
    for kind in (1, 2):
        assert spearmans_rho_nd(Pi, kind=kind, method="qmc") == pytest.approx(0.0, abs=2e-3)
        assert spearmans_rho_nd(M, kind=kind, method="qmc") == pytest.approx(1.0, abs=2e-3)
        assert spearmans_rho_nd(Pi, kind=kind, method="mc") == pytest.approx(0.0, abs=0.02)
    assert kendalls_tau_nd(Pi, method="mc") == pytest.approx(0.0, abs=0.01)
    assert kendalls_tau_nd(M, method="mc") == pytest.approx(1.0, abs=0.01)


def test_bivariate_cases_reduce_to_copul_measures():
    for B in (cp.Clayton(2.0), cp.Frank(-3.0), cp.GumbelHougaard(1.7)):
        assert spearmans_rho_nd(B, kind=1) == pytest.approx(B.spearmans_rho())
        assert spearmans_rho_nd(B, kind=2) == pytest.approx(B.spearmans_rho())
        assert kendalls_tau_nd(B) == pytest.approx(B.kendalls_tau())
        assert blomqvists_beta_nd(B) == pytest.approx(B.blomqvists_beta(), abs=1e-12)
    G2 = GaussianND([[1, 0.6], [0.6, 1]])
    assert G2.spearmans_rho() == pytest.approx(6 / np.pi * np.arcsin(0.3))
    assert G2.kendalls_tau() == pytest.approx(2 / np.pi * np.arcsin(0.6))
    assert G2.blomqvists_beta() == pytest.approx(2 / np.pi * np.arcsin(0.6), abs=1e-12)
    # the generic d-dimensional integrals agree with the bivariate values
    C2 = ClaytonND(2.0, 2)
    assert spearmans_rho_nd(C2, kind=1, method="qmc", n_samples=2**16) == pytest.approx(
        cp.Clayton(2.0).spearmans_rho(), abs=1e-4
    )
    assert spearmans_rho_nd(C2, kind=2, method="qmc", n_samples=2**16) == pytest.approx(
        cp.Clayton(2.0).spearmans_rho(), abs=1e-4
    )


def test_gaussian_three_dims_closed_forms():
    G = GaussianND(R3)
    r = np.array([0.5, 0.3, 0.4])
    # Blomqvist: orthant probability 1/8 + sum(arcsin r)/(4 pi), radial symmetry
    assert G.blomqvists_beta() == pytest.approx(2 * np.arcsin(r).sum() / (3 * np.pi), abs=2e-5)
    # Kendall: average of pairwise 2/pi arcsin r (Nelsen, 1996)
    assert G.kendalls_tau() == pytest.approx(np.mean(2 / np.pi * np.arcsin(r)))
    # Spearman: rho_1 = rho_2 = rho_3 by radial symmetry
    rho3 = np.mean(6 / np.pi * np.arcsin(r / 2))
    for kind in (1, 2, 3):
        assert G.spearmans_rho(kind) == pytest.approx(rho3)
    val, se = G.spearmans_rho(1, method="mc", n_samples=100_000, return_se=True)
    assert abs(val - rho3) < 4 * se


def test_student_t_three_dims():
    T = StudentTND(R3, nu=4)
    taus = 2 / np.pi * np.arcsin(np.array([0.5, 0.3, 0.4]))
    assert T.kendalls_tau() == pytest.approx(taus.mean(), abs=1e-6)
    # Blomqvist of elliptical copulas: same orthant probabilities as the Gaussian
    assert T.blomqvists_beta() == pytest.approx(GaussianND(R3).blomqvists_beta(), abs=2e-3)


def test_three_dim_identities_for_asymmetric_copula():
    """d = 3: (rho_1 + rho_2)/2 = rho_3, tau_3 and beta_3 are pairwise averages."""
    C = MixtureND([ClaytonND(3.0, 3), GaussianND(R3)], [0.5, 0.5])
    C._cheap_cdf = True
    r1 = spearmans_rho_nd(C, kind=1, method="mc", n_samples=200_000)
    r2 = spearmans_rho_nd(C, kind=2, method="mc", n_samples=200_000)
    r3 = spearmans_rho_nd(C, kind=3)
    assert (r1 + r2) / 2 == pytest.approx(r3, abs=0.01)
    pairs = [(0, 1), (0, 2), (1, 2)]
    beta_pairs = np.mean([C.margin(i, j).blomqvists_beta() for i, j in pairs])
    assert C.blomqvists_beta() == pytest.approx(beta_pairs, abs=1e-5)


def test_clayton_multivariate_measures():
    C = ClaytonND(2.0, 3)
    assert C.kendalls_tau() == pytest.approx(0.5, abs=1e-8)  # = theta/(theta+2) for d = 3
    beta3 = C.blomqvists_beta()
    b2 = cp.Clayton(2.0).blomqvists_beta()
    assert beta3 == pytest.approx(b2, abs=1e-12)  # beta_3 = average pairwise beta
    r1 = C.spearmans_rho(1)  # quasi-Monte Carlo on the cdf
    r2 = C.spearmans_rho(2)
    assert (r1 + r2) / 2 == pytest.approx(cp.Clayton(2.0).spearmans_rho(), abs=2e-4)
    assert r1 == pytest.approx(C.survival_copula().spearmans_rho(2, method="qmc"), abs=2e-4)
    val, se = C.spearmans_rho(1, method="qmc", return_se=True)
    assert abs(val - r1) < 1e-3 and se < 1e-3
    mc, se = C.spearmans_rho(1, method="mc", return_se=True)
    assert abs(mc - r1) < 4 * se


def test_four_dims_monte_carlo_vs_exact():
    C = GumbelND(2.0, 4)
    exact = C.kendalls_tau()  # Kendall distribution
    mc, se = kendalls_tau_nd(C, method="mc", return_se=True)
    assert abs(exact - mc) < 4 * se
    G = GaussianND(np.full((4, 4), 0.5) + 0.5 * np.eye(4))  # expensive cdf: paired sampling
    t, se = kendalls_tau_nd(G, n_samples=100_000, return_se=True)
    X = G.rvs(1500, random_state=3)
    assert abs(t - sample_kendalls_tau_nd(X)) < 0.05
    assert se > 0


def test_method_validation():
    with pytest.raises(ValueError):
        spearmans_rho_nd(ClaytonND(2.0, 3), method="bad")
    with pytest.raises(ValueError):
        spearmans_rho_nd(ClaytonND(2.0, 3), kind=4)
    with pytest.raises(ValueError):
        kendalls_tau_nd(ClaytonND(2.0, 4), method="qmc")
    with pytest.raises(ValueError):
        spearmans_rho_nd(ClaytonND(2.0, 4), method="exact")


# ---------------------------------------------------------------------------
# sample versions
# ---------------------------------------------------------------------------


def test_sample_versions_at_comonotone_data():
    x = np.arange(20, dtype=float)
    X = np.column_stack([x, 2 * x + 1, np.exp(x / 5)])
    assert sample_kendalls_tau_nd(X) == pytest.approx(1.0)
    assert sample_blomqvists_beta_nd(X) == pytest.approx(1.0)
    assert sample_spearmans_rho_nd(X, kind=3) == pytest.approx(1.0)
    n = 20
    expect = spearmans_rho_h(3) * (8 * np.mean((1 - x / n - 1 / n) ** 3) - 1)
    assert sample_spearmans_rho_nd(X, kind=1) == pytest.approx(expect)


def test_sample_versions_bivariate_agree_with_classical():
    X = cp.Clayton(2.0).rvs(400, random_state=0)
    assert sample_kendalls_tau_nd(X) == pytest.approx(stats.kendalltau(*X.T).statistic)
    rho = stats.spearmanr(X).statistic
    assert sample_spearmans_rho_nd(X, kind=3) == pytest.approx(rho)
    assert sample_spearmans_rho_nd(X, kind=1) == pytest.approx(rho, abs=0.02)


def test_sample_versions_consistent_with_copula_values():
    C = ClaytonND(2.0, 3)
    X = C.rvs(3000, random_state=6)
    assert sample_kendalls_tau_nd(X) == pytest.approx(0.5, abs=0.03)
    assert sample_blomqvists_beta_nd(X) == pytest.approx(C.blomqvists_beta(), abs=0.04)
    for kind in (1, 2, 3):
        assert sample_spearmans_rho_nd(X, kind=kind) == pytest.approx(
            C.spearmans_rho(kind), abs=0.04
        )


def test_data_dispatch():
    X = IndependenceND(3).rvs(2000, random_state=1)
    assert spearmans_rho_nd(X) == pytest.approx(sample_spearmans_rho_nd(X))
    assert kendalls_tau_nd(X) == pytest.approx(sample_kendalls_tau_nd(X))
    assert blomqvists_beta_nd(X) == pytest.approx(sample_blomqvists_beta_nd(X))
    assert abs(kendalls_tau_nd(X)) < 0.05
    from copul.stats import EmpiricalCopula

    ec = EmpiricalCopula(X)
    assert kendalls_tau_nd(ec) == pytest.approx(sample_kendalls_tau_nd(X))
    import pandas as pd

    assert blomqvists_beta_nd(pd.DataFrame(X)) == pytest.approx(sample_blomqvists_beta_nd(X))
