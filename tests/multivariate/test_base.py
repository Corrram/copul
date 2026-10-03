"""Basic d-dimensional copulas, generic API, margins, validation and conversions."""

import numpy as np
import pytest
from scipy import stats

import copul as cp
from copul.exceptions import PropertyUnavailableException
from copul.multivariate import (
    BivariateCopulaND,
    ClaytonND,
    CopulaND,
    FunctionalCopulaND,
    GaussianND,
    IndependenceND,
    MixtureND,
    UpperFrechetND,
    as_copula_nd,
    frechet_hoeffding_bounds,
    is_copula_nd,
    margin,
)

RNG = np.random.default_rng(12345)


# ---------------------------------------------------------------------------
# call conventions
# ---------------------------------------------------------------------------


def test_call_conventions_independence():
    C = IndependenceND(3)
    assert C.cdf([0.5, 0.5, 0.5]) == pytest.approx(0.125)
    assert isinstance(C.cdf([0.5, 0.5, 0.5]), float)
    P = RNG.random((7, 3))
    np.testing.assert_allclose(C.cdf(P), P.prod(axis=1))
    # separate broadcastable coordinates
    out = C.cdf(P[:, 0], P[:, 1], 0.5)
    np.testing.assert_allclose(out, P[:, 0] * P[:, 1] * 0.5)
    # (..., d) arrays keep their leading shape
    assert C.cdf(RNG.random((2, 4, 3))).shape == (2, 4)
    assert C(P).shape == (7,)
    with pytest.raises(ValueError):
        C.cdf(RNG.random((4, 2)))
    with pytest.raises(TypeError):
        C.cdf(0.1, 0.2)


def test_boundary_conditions_are_exact():
    C = ClaytonND(2.0, 4)
    u = RNG.random(5)
    for k in range(4):
        P = np.ones((5, 4))
        P[:, k] = u
        np.testing.assert_allclose(C.cdf(P), u, atol=1e-15)
        Z = RNG.random((5, 4))
        Z[:, k] = 0.0
        assert np.all(C.cdf(Z) == 0.0)


def test_dimension_validation():
    with pytest.raises(ValueError):
        IndependenceND(1)


# ---------------------------------------------------------------------------
# independence and comonotonicity
# ---------------------------------------------------------------------------


def test_independence_nd():
    C = IndependenceND(4)
    P = RNG.random((10, 4))
    np.testing.assert_allclose(C.pdf(P), 1.0)
    np.testing.assert_allclose(C.logpdf(P), 0.0)
    np.testing.assert_allclose(C.survival_function(P), np.prod(1 - P, axis=1))
    X = C.rvs(2000, random_state=0)
    assert X.shape == (2000, 4)
    assert all(stats.kstest(X[:, j], "uniform").pvalue > 1e-3 for j in range(4))
    assert isinstance(C.margin(0, 2), cp.BivIndependenceCopula)
    assert isinstance(C.margin(0, 1, 3), IndependenceND)
    assert C.pdf([0.3, 0.2, 1.5, 0.4]) == 0.0  # outside the cube


def test_upper_frechet_nd():
    C = UpperFrechetND(3)
    P = RNG.random((10, 3))
    np.testing.assert_allclose(C.cdf(P), P.min(axis=1))
    np.testing.assert_allclose(C.survival_function(P), 1 - P.max(axis=1))
    X = C.rvs(100, random_state=1)
    assert np.all(X[:, 0] == X[:, 1]) and np.all(X[:, 1] == X[:, 2])
    with pytest.raises(PropertyUnavailableException):
        C.pdf([0.2, 0.3, 0.4])
    assert type(C.margin(1, 2)).__name__ == "UpperFrechet"


def test_frechet_hoeffding_bounds_hold():
    P = RNG.random((200, 3))
    lo, hi = frechet_hoeffding_bounds(P)
    for C in (ClaytonND(1.5, 3), GaussianND(np.eye(3) * 0.6 + 0.4), IndependenceND(3)):
        val = C.cdf(P)
        assert np.all(val >= lo - 1e-12) and np.all(val <= hi + 1e-12)


# ---------------------------------------------------------------------------
# survival function, survival copula, volumes
# ---------------------------------------------------------------------------


def test_survival_function_inclusion_exclusion_matches_samples():
    C = ClaytonND(2.0, 3)
    u = np.array([0.3, 0.4, 0.5])
    X = C.rvs(200_000, random_state=3)
    emp = np.mean(np.all(u < X, axis=1))
    assert C.survival_function(u) == pytest.approx(emp, abs=4e-3)


def test_survival_copula():
    C = ClaytonND(2.0, 3)
    S = C.survival_copula()
    P = RNG.random((20, 3))
    np.testing.assert_allclose(S.cdf(P), C.survival_function(1 - P), atol=1e-14)
    np.testing.assert_allclose(S.pdf(P), C.pdf(1 - P), rtol=1e-12)
    assert S.survival_copula() is C
    X = S.rvs(50_000, random_state=4)
    emp = np.mean(np.all(X <= 0.3, axis=1))
    assert S.cdf([0.3, 0.3, 0.3]) == pytest.approx(emp, abs=5e-3)
    # bivariate margin is copul's survival construction
    m = S.margin(0, 1)
    assert m.cdf(0.3, 0.6) == pytest.approx(cp.survival(cp.Clayton(2.0)).cdf(0.3, 0.6), abs=1e-12)


def test_h_volume():
    C = IndependenceND(3)
    a, b = np.array([0.1, 0.2, 0.3]), np.array([0.5, 0.9, 0.4])
    assert C.h_volume(a, b) == pytest.approx(np.prod(b - a))
    assert C.h_volume(np.zeros(3), np.ones(3)) == pytest.approx(1.0)
    G = ClaytonND(3.0, 3)
    A = RNG.random((5, 3)) * 0.5
    B = A + 0.3
    vols = G.h_volume(A, B)
    assert vols.shape == (5,) and np.all(vols >= 0)
    with pytest.raises(ValueError):
        C.h_volume(b, a)


# ---------------------------------------------------------------------------
# margins
# ---------------------------------------------------------------------------


def test_generic_margins():
    base = MixtureND([UpperFrechetND(4), IndependenceND(4)], [0.3, 0.7])
    m3 = margin(FunctionalCopulaND(base.cdf, 4), [0, 2, 3])
    assert isinstance(m3, CopulaND) and m3.dim == 3
    P = RNG.random((6, 3))
    expect = 0.3 * P.min(axis=1) + 0.7 * P.prod(axis=1)
    np.testing.assert_allclose(m3.cdf(P), expect, atol=1e-14)
    m2 = margin(FunctionalCopulaND(base.cdf, 4), [3, 1])
    assert m2.cdf(0.3, 0.6) == pytest.approx(0.3 * 0.3 + 0.7 * 0.18)
    # margins of margins compose
    assert m3.margin(0, 2).cdf(0.3, 0.6) == pytest.approx(0.3 * 0.3 + 0.7 * 0.18)
    with pytest.raises(ValueError):
        base.margin(0, 0)
    with pytest.raises(IndexError):
        base.margin(0, 7)


def test_bivariate_margin_has_measures_and_sampling():
    C = ClaytonND(2.0, 3)
    generic = margin(FunctionalCopulaND(C.cdf, 3, rvs=C._rvs), [0, 1])
    assert generic.kendalls_tau() == pytest.approx(0.5, abs=2e-3)
    X = generic.rvs(500, random_state=0)
    assert X.shape == (500, 2)


def test_pairwise_matrix():
    R = np.array([[1, 0.5, 0.2], [0.5, 1, -0.3], [0.2, -0.3, 1]])
    G = GaussianND(R)
    np.testing.assert_allclose(G.pairwise_matrix("tau"), 2 / np.pi * np.arcsin(R), atol=1e-10)
    np.testing.assert_allclose(G.spearmans_rho_matrix(), 6 / np.pi * np.arcsin(R / 2), atol=1e-10)
    T = ClaytonND(2.0, 4).pairwise_matrix("beta")
    assert T.shape == (4, 4)
    assert np.allclose(T[np.triu_indices(4, 1)], cp.Clayton(2.0).blomqvists_beta())


# ---------------------------------------------------------------------------
# mixtures and functional copulas
# ---------------------------------------------------------------------------


def test_mixture_nd():
    C = MixtureND([UpperFrechetND(3), IndependenceND(3)], [0.4, 0.6])
    assert C.cdf([0.5, 0.5, 0.5]) == pytest.approx(0.4 * 0.5 + 0.6 * 0.125)
    assert not C.is_absolutely_continuous
    X = C.rvs(20_000, random_state=0)
    assert np.mean(np.all(X <= 0.5, axis=1)) == pytest.approx(0.275, abs=0.01)
    D = MixtureND([ClaytonND(2.0, 3), IndependenceND(3)], [0.5, 0.5])
    p = np.array([0.3, 0.6, 0.8])
    assert D.pdf(p) == pytest.approx(0.5 * ClaytonND(2.0, 3).pdf(p) + 0.5)
    with pytest.raises(ValueError):
        MixtureND([IndependenceND(2), IndependenceND(3)])
    with pytest.raises(ValueError):
        MixtureND([IndependenceND(2), IndependenceND(2)], [0.7, 0.7])


def test_functional_copula():
    C = FunctionalCopulaND(
        lambda U: U.prod(axis=1),
        3,
        pdf=lambda U: np.ones(len(U)),
        rvs=lambda n, rng: rng.random((n, 3)),
    )
    assert C.is_absolutely_continuous
    assert C.logpdf([0.2, 0.3, 0.4]) == 0.0
    assert C.rvs(5, random_state=0).shape == (5, 3)
    assert C.kendalls_tau(method="mc", n_samples=4000) == pytest.approx(0.0, abs=0.05)
    with pytest.raises(NotImplementedError):
        FunctionalCopulaND(lambda U: U.prod(axis=1), 3).rvs(3)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_is_copula_nd_accepts_copulas():
    assert is_copula_nd(lambda U: np.prod(U, axis=1), d=3)
    assert is_copula_nd(lambda U: np.min(U, axis=1), d=4, grid=5)
    assert ClaytonND(2.0, 3).is_copula(grid=9)
    ok, det = ClaytonND(-0.4, 3).is_copula(grid=9, return_details=True)
    assert ok and det["total_mass"] == pytest.approx(1.0) and det["min_volume"] >= -1e-12


def test_is_copula_nd_rejects_non_copulas():
    def w3(U):
        return np.maximum(U.sum(axis=1) - 2.0, 0.0)

    ok, det = is_copula_nd(w3, d=3, return_details=True)
    assert not ok and not det["d_increasing"] and det["min_volume"] < 0
    # not grounded
    assert not is_copula_nd(lambda U: np.prod(U, axis=1) + 0.01, d=3)
    # wrong margins
    assert not is_copula_nd(lambda U: np.prod(U**2, axis=1), d=2)
    # Clayton beyond its d-monotone range is not a copula
    with pytest.raises(ValueError):
        ClaytonND(-0.6, 3)

    def clayton_bad(U, th=-0.6):
        return np.maximum(np.sum(U**-th, axis=1) - 2.0, 0.0) ** (-1 / th)

    assert not is_copula_nd(clayton_bad, d=3, grid=21)


def test_is_copula_nd_grid_options():
    ok, det = is_copula_nd(lambda U: np.prod(U, axis=1), d=2, grid=[0.25, 0.5], return_details=True)
    assert ok and det["n_knots"] == 4
    with pytest.raises(ValueError):
        is_copula_nd(lambda U: U[:, 0], d=12, grid=11)
    with pytest.raises(ValueError):
        is_copula_nd(lambda U: U[:, 0])


# ---------------------------------------------------------------------------
# conversions of copul objects
# ---------------------------------------------------------------------------


def test_as_copula_nd_bivariate():
    B = cp.Clayton(2.0)
    C = as_copula_nd(B)
    assert isinstance(C, BivariateCopulaND) and C.dim == 2
    assert C.margin(0, 1) is B
    P = RNG.random((5, 2))
    np.testing.assert_allclose(C.cdf(P), B.cdf(P))
    np.testing.assert_allclose(C.logpdf(P), B.logpdf(P))
    np.testing.assert_allclose(C.survival_function(P), B.survival_function(P))
    assert C.kendalls_tau() == pytest.approx(0.5)
    t = C.margin(1, 0)
    assert t.cdf(0.2, 0.7) == pytest.approx(B.cdf(0.7, 0.2))
    assert as_copula_nd(C) is C


def test_as_copula_nd_legacy_classes():
    from copul.family.archimedean.multivariate_clayton import MultivariateClayton
    from copul.family.frechet.frechet_multi import MVFrechet

    mc = MultivariateClayton(dimension=3, theta=2.0)
    C = as_copula_nd(mc)
    assert C.family == "clayton" and C.dim == 3
    p = [0.3, 0.5, 0.7]
    assert C.cdf(p) == pytest.approx(float(mc.cdf(*p)), rel=1e-12)

    ind = as_copula_nd(cp.IndependenceCopula(dimension=4))
    assert isinstance(ind, IndependenceND) and ind.dim == 4

    fr = as_copula_nd(MVFrechet(dimension=3, alpha=0.4))
    assert fr.cdf(p) == pytest.approx(0.4 * 0.3 + 0.6 * 0.105)

    from copul.family.elliptical.multivar_gaussian import MultivariateGaussian

    R = [[1, 0.3, 0.2], [0.3, 1, 0.1], [0.2, 0.1, 1]]
    g = as_copula_nd(MultivariateGaussian(3, corr_matrix=R))
    assert isinstance(g, GaussianND)
    np.testing.assert_allclose(g.corr, R)


def test_as_copula_nd_checkerboard_and_callable():
    rng = np.random.default_rng(0)
    m = rng.random((3, 3, 3))
    ch = cp.CheckPi(m)
    C = as_copula_nd(ch)
    P = rng.random((5, 3))
    np.testing.assert_allclose(C.cdf(P), ch.cdf(P), atol=1e-12)
    assert C.rvs(10, random_state=0).shape == (10, 3)
    F = as_copula_nd(lambda U: np.prod(U, axis=1), dim=3)
    assert F.cdf([0.5, 0.5, 0.5]) == pytest.approx(0.125)
    with pytest.raises(ValueError):
        as_copula_nd(lambda U: np.prod(U, axis=1))
    with pytest.raises(TypeError):
        as_copula_nd(object())


def test_legacy_classes_to_nd():
    from copul.family.archimedean.multivariate_clayton import MultivariateClayton
    from copul.family.elliptical.multivar_gaussian import MultivariateGaussian
    from copul.family.frechet.frechet_multi import MVFrechet

    C = MultivariateClayton(dimension=4, theta=1.5).to_nd()
    assert C.family == "clayton" and C.dim == 4 and C.theta == 1.5
    R = [[1, 0.3, 0.2], [0.3, 1, 0.1], [0.2, 0.1, 1]]
    G = MultivariateGaussian(3, corr_matrix=R).to_nd()
    assert isinstance(G, GaussianND) and G.margin(0, 1).rho == pytest.approx(0.3)
    assert isinstance(cp.IndependenceCopula(dimension=5).to_nd(), IndependenceND)
    F = MVFrechet(dimension=2, alpha=0.3, beta=0.2).to_nd()
    assert F.cdf([0.4, 0.7]) == pytest.approx(0.3 * 0.4 + 0.5 * 0.28 + 0.2 * 0.1)
    B = cp.Gaussian(0.5).to_nd()
    assert isinstance(B, BivariateCopulaND) and B.dim == 2
