"""d-dimensional Archimedean copulas."""

import numpy as np
import pytest
import sympy as sp
from scipy import integrate, stats

import copul as cp
from copul.multivariate import (
    AliMikhailHaqND,
    ArchimedeanCopulaND,
    ClaytonND,
    FrankND,
    GumbelND,
    JoeND,
    is_d_monotone,
)
from copul.multivariate.archimedean import _resolve_generator

FAMILIES = [
    ("clayton", 2.0),
    ("clayton", -0.4),
    ("gumbel", 2.5),
    ("frank", 4.0),
    ("joe", 2.0),
    ("amh", 0.6),
]

_t = sp.symbols("t", positive=True)
_PSI = {
    "clayton": lambda th: (1 + th * _t) ** (-1 / sp.nsimplify(th)),
    "gumbel": lambda th: sp.exp(-(_t ** (1 / sp.nsimplify(th)))),
    "frank": lambda th: (
        -sp.log(1 - (1 - sp.exp(-sp.nsimplify(th))) * sp.exp(-_t)) / sp.nsimplify(th)
    ),
    "joe": lambda th: 1 - (1 - sp.exp(-_t)) ** (1 / sp.nsimplify(th)),
    "amh": lambda th: (1 - sp.nsimplify(th)) / (sp.exp(_t) - sp.nsimplify(th)),
}


def _mixed_partial_fd(C, u, h=2e-3):
    """d-th mixed central difference of the cdf at u."""
    d = len(u)
    from itertools import product

    total = 0.0
    for signs in product([-1, 1], repeat=d):
        pt = np.asarray(u) + h * np.asarray(signs)
        total += np.prod(signs) * C.cdf(pt)
    return total / (2 * h) ** d


@pytest.mark.parametrize(
    "fam, theta",
    [*FAMILIES, ("frank", -3.0), ("amh", -0.5), ("gumbel", 1.0), ("joe", 5.5), ("frank", 30.0)],
)
def test_generator_derivatives_match_sympy(fam, theta):
    """log|psi^(k)| (Hofert-Maechler-McNeil closed forms) vs. symbolic derivatives."""
    import mpmath

    g = _resolve_generator(fam, theta)
    expr = _PSI[fam](theta)
    s = np.array([1e-3, 0.3, 1.0, 2.5, 7.0, 25.0])
    if fam == "clayton" and theta < 0:
        s = s[s < -1 / theta]
    kmax = 3 if (fam == "clayton" and theta < 0) else 6  # d-monotone up to d = 3
    with mpmath.workdps(80):
        for k in range(kmax + 1):
            f = sp.lambdify(_t, sp.diff(expr, _t, k), "mpmath")
            ref = np.array([abs(float(mpmath.re(f(mpmath.mpf(x))))) for x in s])
            got = np.exp(g.log_abs_dpsi(k, s))
            np.testing.assert_allclose(got, ref, rtol=1e-9)


@pytest.mark.parametrize("fam, theta", FAMILIES)
def test_density_matches_mixed_partial_of_cdf(fam, theta):
    C = ArchimedeanCopulaND(fam, 3, theta=theta)
    for u in ([0.3, 0.5, 0.7], [0.6, 0.4, 0.8], [0.2, 0.25, 0.3]):
        assert C.pdf(u) == pytest.approx(_mixed_partial_fd(C, u), rel=1e-4)


@pytest.mark.parametrize("fam, theta", FAMILIES)
def test_density_integrates_to_one(fam, theta):
    C = ArchimedeanCopulaND(fam, 3, theta=theta)
    P = stats.qmc.Sobol(3, scramble=True, seed=1).random_base2(14)
    # probability of the box [0.1, 0.9]^3 by density vs. by the C-volume
    a, b = 0.1, 0.9
    mass = np.mean(C.pdf(a + (b - a) * P)) * (b - a) ** 3
    # (Clayton with theta < 0 has an integrable singularity of the density)
    rel = 1e-2 if theta < 0 else 2e-3
    assert mass == pytest.approx(C.h_volume([a] * 3, [b] * 3), rel=rel)


@pytest.mark.parametrize("fam, theta", FAMILIES)
def test_sampling_margins_and_tau(fam, theta):
    C = ArchimedeanCopulaND(fam, 3, theta=theta)
    X = C.rvs(6000, random_state=42)
    assert X.shape == (6000, 3)
    assert all(stats.kstest(X[:, j], "uniform").pvalue > 1e-4 for j in range(3))
    tau = stats.kendalltau(X[:, 0], X[:, 2]).statistic
    assert tau == pytest.approx(C.pairwise_kendalls_tau(), abs=0.03)
    # the box probability of the sample agrees with the C-volume
    emp = np.mean(np.all(X <= 0.5, axis=1))
    assert emp == pytest.approx(C.cdf([0.5, 0.5, 0.5]), abs=0.02)


def test_clayton_closed_forms():
    th = 2.0
    C = ClaytonND(th, 4)
    u = np.array([0.3, 0.4, 0.5, 0.8])
    assert C.cdf(u) == pytest.approx((np.sum(u**-th) - 3) ** (-1 / th))
    dens = np.prod([1 + j * th for j in range(4)]) * np.prod(u ** (-th - 1))
    dens *= (np.sum(u**-th) - 3) ** (-1 / th - 4)
    assert C.pdf(u) == pytest.approx(dens, rel=1e-12)
    # bivariate margins: copul Clayton with tau = theta / (theta + 2)
    m = C.margin(1, 3)
    assert isinstance(m, cp.Clayton)
    assert m.kendalls_tau() == pytest.approx(th / (th + 2))
    assert C.pairwise_kendalls_tau() == pytest.approx(0.5)


def test_legacy_multivariate_clayton_agrees():
    from copul.family.archimedean.multivariate_clayton import MultivariateClayton

    legacy = MultivariateClayton(dimension=3, theta=1.5)
    C = ClaytonND(1.5, 3)
    P = np.random.default_rng(3).random((10, 3))
    np.testing.assert_allclose(C.cdf(P), legacy.cdf_vectorized(*P.T), rtol=1e-12)


def test_legacy_gumbel_agrees():
    from copul.family.extreme_value.multivariate_gumbel_hougaard import (
        MultivariateGumbelHougaard,
    )

    legacy = MultivariateGumbelHougaard(dimension=3, theta=2.0)
    P = np.random.default_rng(3).random((10, 3))
    np.testing.assert_allclose(GumbelND(2.0, 3).cdf(P), legacy.cdf_vectorized(*P.T), rtol=1e-12)


@pytest.mark.parametrize("fam, theta", FAMILIES)
def test_tau_three_dims_equals_pairwise(fam, theta):
    """Kendall-distribution tau_3 equals the average pairwise tau (Nelsen, 1996)."""
    C = ArchimedeanCopulaND(fam, 3, theta=theta)
    assert C._exact_measure("tau") == pytest.approx(C.pairwise_kendalls_tau(), abs=1e-8)
    assert C.margin(0, 1).kendalls_tau() == pytest.approx(C.pairwise_kendalls_tau(), abs=1e-6)


def test_tau_four_dims_kendall_distribution_vs_monte_carlo():
    C = ClaytonND(2.0, 4)
    exact = C.kendalls_tau()
    mc, se = C.kendalls_tau(method="mc", n_samples=200_000, return_se=True)
    assert abs(exact - mc) < 4 * se + 1e-3
    assert exact == pytest.approx(27 / 56, abs=1e-8)  # differs from theta/(theta+2)


def test_kendall_distribution():
    C = GumbelND(2.0, 3)
    X = C.rvs(20_000, random_state=8)
    W = C.cdf(X)
    for t in (0.05, 0.2, 0.5, 0.8):
        assert C.kendall_distribution(t) == pytest.approx(np.mean(t >= W), abs=0.012)
    # bivariate: K(t) = t - phi(t)/phi'(t) (Genest & Rivest, 1993); Clayton: t + t(1 - t^th)/th
    K2 = ClaytonND(2.0, 2)
    t = np.array([0.1, 0.4, 0.9])
    np.testing.assert_allclose(K2.kendall_distribution(t), t + t * (1 - t**2.0) / 2.0)
    assert C.kendall_distribution(1.0) == 1.0 and C.kendall_distribution(-0.5) == 0.0


def test_radial_representation():
    """F_R from the inverse Williamson transform drives the McNeil-Neslehova sampler."""
    C = ClaytonND(-0.3, 3)
    assert C.radial_cdf(np.array([0.0]))[0] == 0.0
    assert C.radial_cdf(np.array([1 / 0.3 + 1e-9]))[0] == pytest.approx(1.0)
    X = C.rvs(20_000, random_state=1)
    assert np.mean(np.all(X <= 0.6, axis=1)) == pytest.approx(C.cdf([0.6] * 3), abs=0.01)
    assert C.kendalls_tau() == pytest.approx(-0.3 / 1.7, abs=1e-8)


def test_rosenblatt_transform():
    C = FrankND(5.0, 3)
    X = C.rvs(5000, random_state=4)
    Z = C.rosenblatt(X)
    assert np.max(np.abs(np.corrcoef(Z, rowvar=False) - np.eye(3))) < 0.05
    assert all(stats.kstest(Z[:, j], "uniform").pvalue > 1e-3 for j in range(3))


def test_d_monotonicity():
    assert is_d_monotone("clayton", 3, theta=-0.5)
    assert not is_d_monotone("clayton", 3, theta=-0.51)
    assert is_d_monotone("clayton", 10, theta=0.5)
    assert is_d_monotone("frank", 2, theta=-2.0) and not is_d_monotone("frank", 3, theta=-2.0)
    assert not is_d_monotone("amh", 3, theta=-0.3)
    # SymPy generators are checked numerically
    assert is_d_monotone("(1 + t)**(-1/2)", 6)
    assert not is_d_monotone("Max(1 - t, 0)", 3)  # kink at t = 1
    assert is_d_monotone("Max(1 - t, 0)", 2)
    assert is_d_monotone("Max(1 - t, 0)**2", 3) and not is_d_monotone("Max(1 - t, 0)**2", 4)
    assert not is_d_monotone("exp(-t) * (1 + sin(5*t) / 10)", 2)
    with pytest.raises(ValueError):
        ArchimedeanCopulaND("clayton", 3, theta=-0.7)
    with pytest.raises(ValueError):
        ArchimedeanCopulaND("gumbel", 3, theta=0.5)
    with pytest.raises(ValueError):
        ArchimedeanCopulaND("clayton", 3)


def test_singular_clayton_boundary_case():
    C = ClaytonND(-0.5, 3)
    assert not C.is_absolutely_continuous
    X = C.rvs(5000, random_state=2)
    # all mass on the surface sum u_i^{1/2} = 2
    np.testing.assert_allclose(np.sum(np.sqrt(X), axis=1), 2.0, atol=1e-6)


def test_symbolic_generator_equals_family():
    th = 1.5
    sym = ArchimedeanCopulaND(f"(1 + {th}*t)**(-1/{th})", 3)
    fam = ClaytonND(th, 3)
    P = np.random.default_rng(5).random((8, 3))
    np.testing.assert_allclose(sym.cdf(P), fam.cdf(P), rtol=1e-8)
    np.testing.assert_allclose(sym.pdf(P), fam.pdf(P), rtol=1e-7)
    assert sym.pairwise_kendalls_tau() == pytest.approx(th / (th + 2), abs=1e-7)
    X = sym.rvs(4000, random_state=0)  # radial sampler (no frailty known)
    assert stats.kendalltau(X[:, 0], X[:, 1]).statistic == pytest.approx(th / (th + 2), abs=0.03)
    m = sym.margin(0, 2)
    assert m.cdf(0.3, 0.6) == pytest.approx(cp.Clayton(th).cdf(0.3, 0.6), rel=1e-8)
    assert m.cond_distr_1(0.3, 0.6) == pytest.approx(cp.Clayton(th).cond_distr_1(0.3, 0.6))


def test_from_copul_families():
    C = ArchimedeanCopulaND(cp.Clayton(theta=2.0), 3)
    assert C.family == "clayton" and C.theta == 2.0
    G = ArchimedeanCopulaND(cp.GumbelHougaard(3.0), 4)
    assert G.family == "gumbel"
    N12 = ArchimedeanCopulaND(cp.Nelsen12(theta=2.0), 3)  # generic SymPy generator
    biv = cp.Nelsen12(theta=2.0)
    assert N12.margin(0, 1).cdf(0.3, 0.7) == pytest.approx(biv.cdf(0.3, 0.7), rel=1e-8)
    assert ArchimedeanCopulaND(N12, 3).family == N12.family
    with pytest.raises(ValueError):
        ArchimedeanCopulaND(cp.Clayton(), 3)  # free parameter


def test_independence_limits():
    P = np.random.default_rng(6).random((5, 3))
    for C in (ClaytonND(0.0, 3), GumbelND(1.0, 3), FrankND(0.0, 3), JoeND(1.0, 3)):
        np.testing.assert_allclose(C.cdf(P), P.prod(axis=1), rtol=1e-10)
        np.testing.assert_allclose(C.pdf(P), 1.0, rtol=1e-9)
    np.testing.assert_allclose(AliMikhailHaqND(0.0, 3).cdf(P), P.prod(axis=1), rtol=1e-10)


def test_bivariate_tau_closed_forms_agree_with_quadrature():
    for fam, th in [("frank", 4.0), ("amh", 0.6), ("joe", 2.0), ("frank", -3.0)]:
        C = ArchimedeanCopulaND(fam, 2, theta=th)
        assert C.pairwise_kendalls_tau() == pytest.approx(C._tau_from_kendall(2), abs=1e-8)
        assert C.pairwise_kendalls_tau() == pytest.approx(C.margin(0, 1).kendalls_tau(), abs=1e-6)


@pytest.mark.parametrize("fam, theta", [("clayton", 2.0), ("gumbel", 2.0), ("frank", 5.0)])
def test_fit(fam, theta):
    X = ArchimedeanCopulaND(fam, 3, theta=theta).rvs(1500, random_state=17)
    F = ArchimedeanCopulaND.fit(X, fam)
    assert F.theta == pytest.approx(theta, rel=0.12)
    F2 = ArchimedeanCopulaND.fit(X, fam, method="itau")
    assert F2.theta == pytest.approx(theta, rel=0.15)


def test_fit_errors():
    X = ClaytonND(2.0, 3).rvs(100, random_state=0)
    with pytest.raises(ValueError):
        ArchimedeanCopulaND.fit(X, "nope")
    with pytest.raises(ValueError):
        ArchimedeanCopulaND.fit(X, "clayton", method="bad")


def test_psi_phi_inverse_and_derivative_signs():
    for fam, th in FAMILIES:
        C = ArchimedeanCopulaND(fam, 3, theta=th)
        u = np.array([0.05, 0.3, 0.7, 0.95])
        np.testing.assert_allclose(C.psi(C.phi(u)), u, rtol=1e-10)
        s = np.array([0.2, 1.0])
        if fam == "clayton" and th < 0:
            s = s * 0.5
        for k in range(4):
            assert np.all(np.sign(C.psi_derivative(k, s)) == (-1) ** k)
        # numerical derivative of psi
        d1 = (C.psi(s + 1e-6) - C.psi(s - 1e-6)) / 2e-6
        np.testing.assert_allclose(C.psi_derivative(1, s), d1, rtol=1e-5)


def test_kendall_tau_bivariate_formula():
    """tau = 1 + 4 int phi/phi' (Genest & MacKay, 1986) for the Joe family."""
    C = JoeND(2.0, 2)
    val = 1 + 4 * integrate.quad(lambda t: -C.phi(t) * np.exp(-C.generator.log_mdphi(t)), 0, 1)[0]
    assert C.pairwise_kendalls_tau() == pytest.approx(val, abs=1e-8)
