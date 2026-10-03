"""Tests for copul.theory.archimedean and the numerical Archimedean copulas."""

import math
import warnings

import numpy as np
import pytest

import copul as cp
from copul.family.archimedean.numeric_archimedean import NumericArchimedeanCopula
from copul.measures.quadrature import gauss_legendre_nodes
from copul.theory.archimedean import (
    archimedean_from_generator,
    archimedean_from_kendall_distribution,
    archimedean_generator,
    associativity_defect,
    check_generator,
    from_laplace_transform,
    generator_properties,
    is_archimedean,
    kendall_distribution,
    kendall_distribution_inverse,
    max_dimension,
    zero_curve,
)
from tests.family.numeric_copula_checks import (
    check_axioms,
    check_conditionals,
    check_density,
    check_sampling,
)

T = np.array([0.0, 1e-6, 0.01, 0.1, 0.3, 0.5, 0.77, 0.95, 1.0])


def _integral_of_K(C, method="auto"):
    """int_0^1 K_C(t) dt by composite Gauss-Legendre (graded towards 0)."""
    edges = np.concatenate([[0.0], np.geomspace(1e-7, 1e-1, 7), np.linspace(0.2, 1.0, 9)])
    x, w = gauss_legendre_nodes(10)
    h = np.diff(edges)
    nodes = (edges[:-1, None] + h[:, None] * x[None, :]).ravel()
    weights = (h[:, None] * w[None, :]).ravel()
    return float(np.sum(weights * kendall_distribution(C, nodes, method=method)))


# ---------------------------------------------------------------------------
# Kendall distribution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "C, K",
    [
        (cp.Clayton(2), lambda t: t + t * (1 - t**2) / 2),
        (cp.Clayton(0.5), lambda t: t + t * (1 - t**0.5) / 0.5),
        (cp.GumbelHougaard(3), lambda t: t - t * np.log(t) / 3),
        (cp.BivIndependenceCopula(), lambda t: t - t * np.log(t)),
        (cp.UpperFrechet(), lambda t: t),
        (cp.LowerFrechet(), lambda t: np.ones_like(t)),
    ],
    ids=["clayton2", "clayton05", "gumbel3", "pi", "M", "W"],
)
def test_kendall_closed_forms(C, K):
    with np.errstate(all="ignore"):
        ref = np.where(T > 0, K(np.where(T > 0, T, 0.5)), K(np.array([1e-300]))[0])
    ref[-1] = 1.0
    got = kendall_distribution(C, T)
    np.testing.assert_allclose(got, ref, atol=1e-14)
    # the general integral representation agrees with the closed form
    num = kendall_distribution(C, T, method="numeric")
    np.testing.assert_allclose(num, ref, atol=1e-9)


@pytest.mark.parametrize(
    "C",
    [
        cp.Frank(3),
        cp.Frank(-4),
        cp.Joe(2),
        cp.AliMikhailHaq(0.5),
        cp.Nelsen2(1.5),
        cp.BB1(0.7, 1.4),
    ],
    ids=["frank3", "frank-4", "joe2", "amh", "nelsen2", "bb1"],
)
def test_kendall_closed_vs_numeric(C):
    t = np.linspace(0.0, 1.0, 21)
    np.testing.assert_allclose(
        kendall_distribution(C, t, method="closed"),
        kendall_distribution(C, t, method="numeric"),
        atol=1e-9,
    )


def test_kendall_scalar_and_bounds():
    C = cp.Clayton(2)
    assert isinstance(kendall_distribution(C, 0.3), float)
    assert kendall_distribution(C, 0.3) == pytest.approx(0.4365, abs=1e-14)
    assert C.kendall_distribution(0.3) == pytest.approx(0.4365, abs=1e-14)
    t = np.linspace(0, 1, 101)
    k = kendall_distribution(cp.Gaussian(-0.4), t)
    assert np.all(k >= t - 1e-12) and np.all(k <= 1 + 1e-12)
    assert np.all(np.diff(k) >= -1e-12)
    with pytest.raises(NotImplementedError):
        kendall_distribution(cp.Gaussian(0.3), 0.5, method="closed")
    with pytest.raises(ValueError):
        kendall_distribution(cp.Clayton(), 0.5)  # free parameter


def test_kendall_numeric_vs_monte_carlo():
    C = cp.Gaussian(0.5)
    t = np.array([0.05, 0.2, 0.4, 0.6, 0.8])
    num = kendall_distribution(C, t)
    mc = kendall_distribution(C, t, method="mc", n_samples=100_000, random_state=3)
    np.testing.assert_allclose(num, mc, atol=0.006)


@pytest.mark.parametrize(
    "C, tau",
    [
        (cp.Clayton(2), 0.5),
        (cp.Frank(5), None),
        (cp.Nelsen2(2.0), None),
        (cp.Gaussian(0.5), 1 / 3),
        (cp.StudentT(-0.3, 4), 2 / np.pi * np.arcsin(-0.3)),
        (cp.BivCheckPi([[2, 1, 0], [0, 1, 2], [1, 1, 1]]), None),
        (cp.FarlieGumbelMorgenstern(0.6), 2 * 0.6 / 9),
    ],
    ids=["clayton", "frank", "nelsen2", "gauss", "student", "checkpi", "fgm"],
)
def test_tau_from_kendall_distribution(C, tau):
    """Kendall's tau = 3 - 4 int K_C (also for non-Archimedean copulas)."""
    tau = C.kendalls_tau() if tau is None else tau
    # K_C of a checkerboard has kinks: the fixed Gauss-Legendre rule is only O(h^2)
    tol = 1e-4 if isinstance(C, cp.BivCheckPi) else 1e-7
    assert 3 - 4 * _integral_of_K(C) == pytest.approx(tau, abs=tol)


def test_kendall_inverse():
    p = np.array([0.0, 0.01, 0.3, 0.7, 0.999, 1.0])
    for C in (cp.Clayton(2), cp.GumbelHougaard(1.5)):
        t = kendall_distribution_inverse(C, p)
        np.testing.assert_allclose(kendall_distribution(C, t), p, atol=1e-11)
    # non-strict generator: K(0) = 1/theta is the mass of the zero curve
    C = cp.Nelsen2(1.5)
    t = kendall_distribution_inverse(C, np.array([0.2, 0.6, 0.8]))
    np.testing.assert_allclose(t[:2], 0.0)
    assert kendall_distribution(C, t[2]) == pytest.approx(0.8, abs=1e-11)
    # numerical Kendall distribution of a non-Archimedean copula
    G = cp.Gaussian(0.3)
    q = kendall_distribution_inverse(G, 0.5)
    assert isinstance(q, float)
    assert kendall_distribution(G, q) == pytest.approx(0.5, abs=1e-8)


def test_kendall_sampling_algorithm():
    """S = phi(U)/(phi(U)+phi(V)) ~ U(0,1) is independent of T = C(U,V) ~ K_C;
    (psi(S phi(T)), psi((1-S) phi(T))) samples C (Genest & Rivest 1993)."""
    C = cp.Frank(4)
    gen = archimedean_generator(C)
    rng = np.random.default_rng(11)
    n = 40_000
    s, q = rng.random(n), rng.random(n)
    t = kendall_distribution_inverse(C, q)
    phit = gen.phi(t)
    x = np.column_stack([gen.psi(s * phit), gen.psi((1 - s) * phit)])
    for a, b in [(0.2, 0.3), (0.5, 0.5), (0.8, 0.6)]:
        emp = np.mean((x[:, 0] <= a) & (x[:, 1] <= b))
        assert emp == pytest.approx(C.cdf(a, b), abs=0.012)


# ---------------------------------------------------------------------------
# generator theory
# ---------------------------------------------------------------------------


def test_all_archimedean_representatives_have_valid_generators():
    from tests.family_representatives import archimedean_representatives as reps

    for name, theta in reps.items():
        C = getattr(cp, name)(theta)
        rep = generator_properties(C, d_max=4)
        assert rep.valid, (name, rep)
        assert rep.max_dimension >= 2
        assert rep.strict == (not math.isfinite(rep.phi0))


def test_check_generator_invalid():
    for phi in ("t * (1 - t)", "1 - t**2", "2 - t", "log(t)"):
        rep = check_generator(phi, d_max=3)
        assert not rep.valid
        assert not bool(rep)
        assert rep.max_dimension == 1
    assert not check_generator(lambda t: 1 - t**2).valid
    assert not check_generator(lambda t: np.sqrt(1 - t)).valid


def test_strict_non_strict_and_zero_curve_mass():
    rep = check_generator("-log(t)")
    assert rep.strict and rep.zero_curve_mass == 0 and math.isinf(rep.phi0)
    rep = check_generator("(1 - t)**3")  # Nelsen 2, theta = 3
    assert not rep.strict and rep.phi0 == pytest.approx(1.0)
    assert rep.zero_curve_mass == pytest.approx(1 / 3)
    th = 0.5  # Nelsen 7: phi(0) = -log(1-th), phi'(0) = -th/(1-th)
    rep = generator_properties(cp.Nelsen7(th))
    assert rep.zero_curve_mass == pytest.approx(-np.log(1 - th) * (1 - th) / th)
    assert rep.max_dimension == 2  # positive singular mass: only bivariate
    assert generator_properties(cp.LowerFrechet()).zero_curve_mass == pytest.approx(1.0)


@pytest.mark.parametrize("d", [2, 3, 4, 6])
def test_clayton_negative_max_dimension(d):
    """Clayton with theta < 0 is d-monotone iff theta >= -1/(d-1)."""
    theta = -1.0 / (d - 1)
    assert max_dimension(cp.Clayton(theta)) == d
    # exact rational parameter in the symbolic generator: high-precision check
    expr = f"(t**(1/{d - 1}) - 1) * (-{d - 1})"
    assert check_generator(expr, d_max=8).max_dimension == d
    # a slightly smaller theta loses the dimension d
    if d > 2:
        assert max_dimension(cp.Clayton(theta * 1.05)) == d - 1


def test_max_dimension_laplace_transforms_and_negative_parameters():
    for C in (cp.Clayton(1), cp.GumbelHougaard(2), cp.Frank(2), cp.Joe(3), cp.AliMikhailHaq(0.4)):
        rep = generator_properties(C)
        assert rep.completely_monotone is True and rep.max_dimension == math.inf
    assert cp.BB1(0.7, 1.4).max_dimension() == math.inf
    for C in (cp.AliMikhailHaq(-0.5), cp.Frank(-3), cp.LowerFrechet()):
        d = max_dimension(C, d_max=8)
        assert 2 <= d < 8, (C, d)
        assert generator_properties(C, d_max=8).completely_monotone is False
    # weaker negative dependence -> higher admissible dimension
    assert max_dimension(cp.AliMikhailHaq(-0.05)) > max_dimension(cp.AliMikhailHaq(-0.9))


def test_d_monotone_numeric_generators():
    # completely monotone: psi(s) = 1/(1+s) (Clayton(1)) as numerical generator
    rep = check_generator(lambda t: 1 / t - 1, d_max=6)
    assert rep.valid and rep.max_dimension == 6 and rep.completely_monotone is None
    # psi(s) = (1 - s/2)_+^2 (Clayton(-1/2)) is 3- but not 4-monotone
    rep = check_generator(lambda t: 2 * (1 - np.sqrt(t)), d_max=6)
    assert rep.valid and rep.max_dimension == 3
    assert rep.d_monotone == {2: True, 3: True, 4: False, 5: False, 6: False}


def test_generator_methods_on_classes():
    C = cp.Clayton(-0.3)
    assert C.max_dimension() == 4
    assert C.generator_properties().valid
    assert cp.Nelsen2(2).zero_curve().mass == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# zero curve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "C, mass",
    [
        (cp.Nelsen2(1.5), 1 / 1.5),
        (cp.Nelsen7(0.5), np.log(2)),
        (cp.Nelsen8(2), 0.5),
        (cp.LowerFrechet(), 1.0),
        (cp.Clayton(-0.5), 0.0),
    ],
    ids=["nelsen2", "nelsen7", "nelsen8", "W", "clayton-0.5"],
)
def test_zero_curve(C, mass):
    zc = zero_curve(C)
    assert zc.mass == pytest.approx(mass, abs=1e-12)
    assert not zc.empty
    u = np.linspace(0.02, 0.98, 25)
    z = zc(u)
    np.testing.assert_allclose(C.cdf(u, z), 0.0, atol=1e-12)
    inside = z < 1 - 1e-5
    assert np.all(C.cdf(u[inside], z[inside] + 1e-5) > 0)
    assert zc.contains(0.5, zc(0.5) / 2) and not zc.contains(0.5, min(zc(0.5) + 0.01, 1))
    # K_C(0) = P(C(U, V) = 0) is the mass of the zero curve
    assert kendall_distribution(C, 0.0, method="numeric") == pytest.approx(mass, abs=1e-8)
    # the area of the zero set: int_0^1 z(u) du
    g, w = gauss_legendre_nodes(200)
    assert zc.area == pytest.approx(float(np.sum(w * zc(g))), abs=1e-6)


def test_zero_curve_mass_by_sampling():
    C = NumericArchimedeanCopula(phi=lambda t: (1 - t) ** 1.5)  # Nelsen 2
    x = C.rvs(40_000, random_state=5)
    frac = np.mean(cp.Nelsen2(1.5).cdf(x) <= 1e-9)
    assert frac == pytest.approx(1 / 1.5, abs=0.012)


def test_zero_curve_strict_and_generic():
    assert zero_curve(cp.Clayton(2)).empty
    assert zero_curve(cp.Clayton(2)).mass == 0
    # generic copula: checkerboard with an empty lower-left block
    zc = zero_curve(cp.BivCheckPi([[0, 1], [1, 0]]))
    np.testing.assert_allclose(zc(np.array([0.25, 0.5, 0.75])), [0.5, 0.5, 0.0], atol=1e-12)
    assert zc.area == pytest.approx(0.25, abs=1e-9)
    assert zc.mass == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# associativity and the characterisation of Archimedean copulas
# ---------------------------------------------------------------------------

ARCHIMEDEAN = [
    cp.Clayton(2),
    cp.Clayton(-0.4),
    cp.Frank(-3),
    cp.GumbelHougaard(2.5),
    cp.Joe(2),
    cp.AliMikhailHaq(0.5),
    cp.Nelsen2(2),
    cp.Nelsen21(3),
    cp.BB1(0.7, 1.4),
    cp.BivIndependenceCopula(),
    cp.LowerFrechet(),
    cp.GumbelHougaardEV(2),  # Gumbel is Archimedean and extreme-value
]
NOT_ARCHIMEDEAN = [
    cp.UpperFrechet(),
    cp.Gaussian(0.5),
    cp.StudentT(0.3, 5),
    cp.Plackett(3),
    cp.FarlieGumbelMorgenstern(0.5),
    cp.Galambos(1.5),
    cp.ordinal_sum([(0.0, 0.6, cp.GumbelHougaard(3)), (0.6, 1.0, cp.Clayton(2))]),
    cp.BivCheckPi([[1, 0], [0, 1]]),
]


@pytest.mark.parametrize("C", ARCHIMEDEAN, ids=lambda C: type(C).__name__)
def test_archimedean_copulas_are_recognised(C):
    assert associativity_defect(C) < 1e-12
    assert is_archimedean(C)


@pytest.mark.parametrize("C", NOT_ARCHIMEDEAN, ids=lambda C: type(C).__name__)
def test_non_archimedean_copulas_are_rejected(C):
    ok, info = is_archimedean(C, return_details=True)
    assert not ok
    if isinstance(
        C, (cp.Gaussian, cp.StudentT, cp.Plackett, cp.FarlieGumbelMorgenstern, cp.Galambos)
    ):
        assert info["associativity_defect"] > 1e-4
    else:  # associative, but with idempotents delta(t) = t
        assert info["diagonal_gap"] < 1e-6


def test_associativity_argmax():
    val, (u, v, w) = associativity_defect(cp.Gaussian(0.7), n=8, return_argmax=True)
    C = cp.Gaussian(0.7)
    assert val == pytest.approx(abs(C.cdf(C.cdf(u, v), w) - C.cdf(u, C.cdf(v, w))))


# ---------------------------------------------------------------------------
# constructions of numerical Archimedean copulas
# ---------------------------------------------------------------------------

P = np.random.default_rng(0).random((400, 2))


@pytest.mark.parametrize(
    "family, K",
    [
        (cp.Clayton(2), lambda t: t + t * (1 - t**2) / 2),
        (cp.GumbelHougaard(2), lambda t: t - t * np.log(t) / 2),
        (cp.Frank(-3), None),
        (cp.Nelsen2(1.5), None),
    ],
    ids=["clayton", "gumbel", "frank-3", "nelsen2"],
)
def test_round_trip_from_kendall_distribution(family, K):
    if K is None:
        K = family.kendall_distribution  # closed form of the family
    C = archimedean_from_kendall_distribution(K)
    np.testing.assert_allclose(C.cdf(P), family.cdf(P), atol=1e-9)
    # reference h1 = phi'(u) / phi'(C(u, v)) from the family's symbolic generator
    # (tests/schur_order replaces Nelsen2.cond_distr_1 by a mock at class level)
    gen = archimedean_generator(family)
    c = family.cdf(P)
    with np.errstate(all="ignore"):
        h_ref = np.where(c > 0, gen.dphi(P[:, 0]) / gen.dphi(np.maximum(c, 1e-300)), 0.0)
    np.testing.assert_allclose(C.cond_distr_1(P), h_ref, atol=1e-7)
    np.testing.assert_allclose(C.kendall_distribution(T), family.kendall_distribution(T), atol=1e-9)
    assert C.kendalls_tau() == pytest.approx(family.kendalls_tau(), abs=1e-8)
    assert C.is_strict == math.isinf(archimedean_generator(family).phi0)


def test_kendall_round_trip_is_fully_usable():
    C = archimedean_from_kendall_distribution(lambda t: t - t * np.log(t) / 2)  # Gumbel(2)
    G = cp.GumbelHougaard(2)
    assert C.spearmans_rho() == pytest.approx(G.spearmans_rho(), abs=1e-8)
    assert C.blomqvists_beta() == pytest.approx(G.blomqvists_beta(), abs=1e-10)
    np.testing.assert_allclose(C.pdf(P[:50]), G.pdf(P[:50]), rtol=1e-6)
    x = C.rvs(5000, random_state=0)
    from scipy.stats import kendalltau

    assert kendalltau(x[:, 0], x[:, 1])[0] == pytest.approx(0.5, abs=0.02)


def test_from_kendall_distribution_rejects_invalid_K():
    with pytest.raises(ValueError):
        archimedean_from_kendall_distribution(lambda t: t)  # K_M is not Archimedean


@pytest.mark.parametrize(
    "psi, family",
    [
        (lambda s: (1 + s) ** -0.5, cp.Clayton(2)),
        (lambda s: np.exp(-np.sqrt(s)), cp.GumbelHougaard(2)),
        (lambda s: -np.log1p(-(1 - np.exp(-3.0)) * np.exp(-s)) / 3.0, cp.Frank(3)),
    ],
    ids=["gamma", "stable", "log-series"],
)
def test_from_laplace_transform(psi, family):
    C = from_laplace_transform(psi)
    np.testing.assert_allclose(C.cdf(P), family.cdf(P), atol=1e-12)
    np.testing.assert_allclose(C.cond_distr_1(P), family.cond_distr_1(P), atol=1e-8)
    np.testing.assert_allclose(
        C.cond_distr_1_inv(P[:, 0], P[:, 1]), family.cond_distr_1_inv(P[:, 0], P[:, 1]), atol=1e-8
    )
    assert C.kendalls_tau() == pytest.approx(family.kendalls_tau(), abs=1e-8)
    assert C.max_dimension() >= 6


def test_laplace_transform_copula_without_frailty_sampler():
    # LT of a two-point frailty V in {1, 3}: not one of the standard families
    C = from_laplace_transform(lambda s: 0.5 * np.exp(-s) + 0.5 * np.exp(-3 * s))
    check_axioms(C)
    check_conditionals(C)
    check_density(C, total=False)
    check_sampling(C)
    # tau from the generator formula vs. the 2D integral formula
    assert C.kendalls_tau() == pytest.approx(C.kendalls_tau(method="numeric"), abs=1e-6)


def test_from_laplace_transform_checks():
    with pytest.raises(ValueError):
        from_laplace_transform(lambda s: 0.5 * np.exp(-s))  # psi(0) != 1
    with pytest.raises(ValueError):
        from_laplace_transform(lambda s: np.clip(1 - s**2, 0, None))  # concave
    with pytest.warns(UserWarning, match="Laplace transform"):
        from_laplace_transform(lambda s: np.clip(1 - s / 2, 0, None) ** 2)  # Clayton(-1/2)


def test_archimedean_from_generator_non_strict():
    C = archimedean_from_generator(lambda t: (1 - t) ** 2, dphi=lambda t: -2 * (1 - t))
    N2 = cp.Nelsen2(2)
    np.testing.assert_allclose(C.cdf(P), N2.cdf(P), atol=1e-12)
    assert not C.is_absolutely_continuous and C.zero_curve_mass == pytest.approx(0.5)
    check_axioms(C)
    check_sampling(C)
    with pytest.raises(ValueError):
        archimedean_from_generator(lambda t: 1 - t**2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert repr(C).startswith("NumericArchimedeanCopula")
