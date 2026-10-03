"""Tests for copul.theory.symmetry (exchangeability and radial symmetry)."""

import numpy as np
import pytest

import copul as cp
from copul.theory.symmetry import (
    SymmetryTestResult,
    exchangeability_test,
    is_exchangeable,
    is_radially_symmetric,
    maximally_nonexchangeable_copula,
    nonexchangeability,
    radial_asymmetry,
    radial_symmetrize,
    radial_symmetry_test,
    symmetrize,
)

THIRD = 1.0 / 3.0
G = np.linspace(0.0, 1.0, 41)
U, V = np.meshgrid(G, G, indexing="ij")


@pytest.fixture(scope="module")
def checkerboards():
    pis = cp.BivCheckPi.generate_diverse(n_samples=24, grid_size=(2, 12), rng=13)
    return (
        pis + [cp.BivCheckMin(C.matr) for C in pis[:12]] + [cp.BivCheckW(C.matr) for C in pis[12:]]
    )


def _khoudraji():
    return cp.khoudraji(cp.BivIndependenceCopula(), cp.GumbelHougaard(3), 0.3, 0.9)


# ---------------------------------------------------------------------------
# non-exchangeability
# ---------------------------------------------------------------------------


def test_maximally_nonexchangeable_copula():
    """sup |C(u,v) - C(v,u)| = 1/3 is attained (Klement & Mesiar 2006; Nelsen 2007)."""
    C = maximally_nonexchangeable_copula()
    assert C.cdf(THIRD, 2 * THIRD) == pytest.approx(THIRD)
    assert C.cdf(2 * THIRD, THIRD) == pytest.approx(0.0, abs=1e-15)
    mu, loc = nonexchangeability(C, return_location=True)
    assert mu == pytest.approx(1.0, abs=1e-12)
    assert loc == pytest.approx((THIRD, 2 * THIRD), abs=1e-9)
    CT = maximally_nonexchangeable_copula(transpose=True)
    assert np.allclose(CT.cdf(U, V), C.cdf(V, U), atol=1e-14)
    assert nonexchangeability(CT) == pytest.approx(1.0, abs=1e-12)
    # closed form of the extremal copula
    expected = np.maximum(0, np.minimum(U, V - THIRD)) + np.maximum(0, np.minimum(U - 2 * THIRD, V))
    assert np.allclose(C.cdf(U, V), expected, atol=1e-14)


def test_pointwise_nonexchangeability_bound(checkerboards):
    """|C(u,v) - C(v,u)| <= min(u, 1 - v, v - u) for u <= v."""
    up = U <= V
    bound = np.minimum(np.minimum(U, 1 - V), V - U)
    for C in [*checkerboards, maximally_nonexchangeable_copula()]:
        diff = np.abs(C.cdf_vectorized(U, V) - C.cdf_vectorized(V, U))
        assert np.all(diff[up] <= bound[up] + 1e-12)


def test_nonexchangeability_of_random_copulas_is_at_most_one(checkerboards):
    vals = [nonexchangeability(C) for C in checkerboards]
    assert max(vals) <= 1.0
    assert max(vals) > 0.2  # the sample contains clearly asymmetric copulas
    for C in checkerboards[:4]:
        assert nonexchangeability(C) == pytest.approx(nonexchangeability(C.transpose()), abs=1e-12)


@pytest.mark.parametrize(
    "C",
    [
        cp.Clayton(2),
        cp.Frank(-4),
        cp.GumbelHougaard(2),
        cp.Gaussian(0.6),
        cp.FarlieGumbelMorgenstern(0.7),
        cp.BivCheckPi([[1, 2, 0], [2, 0, 1], [0, 1, 2]]),
    ],
    ids=lambda C: type(C).__name__,
)
def test_exchangeable_copulas(C):
    assert nonexchangeability(C) < 1e-10
    assert nonexchangeability(C, p=2) < 1e-10
    assert is_exchangeable(C)


def test_khoudraji_is_not_exchangeable():
    K = _khoudraji()
    mu, (u, v) = nonexchangeability(K, return_location=True)
    assert 0.05 < mu <= 1.0
    assert 3 * abs(K.cdf(u, v) - K.cdf(v, u)) == pytest.approx(mu, abs=1e-12)
    l2 = nonexchangeability(K, p=2)
    assert 0.0 < l2 <= mu / 3
    assert not is_exchangeable(K)
    with pytest.raises(ValueError):
        nonexchangeability(K, p=0.5)


def test_symmetrize():
    K = _khoudraji()
    S = symmetrize(K)
    assert nonexchangeability(S) < 1e-10
    assert np.allclose(S.cdf(U, V), 0.5 * (K.cdf(U, V) + K.cdf(V, U)), atol=1e-12)
    C = maximally_nonexchangeable_copula()
    assert nonexchangeability(symmetrize(C)) < 1e-12


# ---------------------------------------------------------------------------
# radial asymmetry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "C",
    [
        cp.Gaussian(0.5),
        cp.Frank(4),
        cp.FarlieGumbelMorgenstern(-0.6),
        cp.Plackett(3),
        cp.StudentT(0.3, 4),
        cp.BivIndependenceCopula(),
    ],
    ids=lambda C: type(C).__name__,
)
def test_radially_symmetric_families(C):
    assert radial_asymmetry(C) < 1e-8
    assert radial_asymmetry(C, p=2) < 1e-9
    assert is_radially_symmetric(C, tol=1e-8)


@pytest.mark.parametrize(
    "C", [cp.Clayton(3), cp.GumbelHougaard(2), cp.Joe(2)], ids=lambda C: type(C).__name__
)
def test_radially_asymmetric_families(C):
    nu = radial_asymmetry(C)
    assert 0.015 < nu <= 1.0
    assert not is_radially_symmetric(C)
    # nu(C) = nu(survival copula)
    assert radial_asymmetry(cp.survival(C)) == pytest.approx(nu, abs=1e-7)
    assert radial_asymmetry(radial_symmetrize(C)) < 1e-10


def test_radial_asymmetry_of_random_copulas(checkerboards):
    vals = [radial_asymmetry(C) for C in checkerboards]
    assert min(vals) >= 0.0
    assert max(vals) > 0.05
    for C in checkerboards[:3]:
        assert radial_asymmetry(radial_symmetrize(C)) < 1e-10


# ---------------------------------------------------------------------------
# tests from data
# ---------------------------------------------------------------------------


def test_exchangeability_test_keeps_exchangeable_data():
    X = cp.Clayton(2).rvs(250, random_state=1)
    res = exchangeability_test(X, n_boot=200, random_state=0)
    assert isinstance(res, SymmetryTestResult)
    assert res.n == 250 and res.n_boot == 200
    assert 0 < res.pvalue <= 1
    assert res.pvalue > 0.05 and not res.reject()
    res_r = exchangeability_test(X, n_boot=200, statistic="Rn", random_state=0)
    assert res_r.pvalue > 0.05
    # reproducible
    assert exchangeability_test(X, n_boot=200, random_state=0).pvalue == res.pvalue


def test_exchangeability_test_rejects_asymmetric_data():
    C = cp.khoudraji(cp.BivIndependenceCopula(), cp.GumbelHougaard(5), 0.3, 0.95)
    X = C.rvs(250, random_state=1)
    res = exchangeability_test(X, n_boot=200, random_state=0)
    assert res.pvalue < 0.05 and res.reject()
    assert exchangeability_test(X, n_boot=200, statistic="Rn", random_state=0).reject()


def test_radial_symmetry_test():
    X = cp.Frank(5).rvs(250, random_state=2)
    assert radial_symmetry_test(X, n_boot=200, random_state=0).pvalue > 0.05
    Y = cp.Clayton(4).rvs(250, random_state=2)
    res = radial_symmetry_test(Y, n_boot=200, random_state=0)
    assert res.pvalue < 0.05
    assert "radial" in res.method


def test_symmetry_tests_validate_input():
    with pytest.raises(ValueError):
        exchangeability_test(np.zeros((10, 1)), n_boot=10)
    with pytest.raises(ValueError):
        exchangeability_test(cp.Clayton(2).rvs(20, random_state=0), statistic="Tn", n_boot=10)
