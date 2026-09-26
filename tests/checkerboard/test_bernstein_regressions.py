"""Regression tests for the Bernstein copula fixes."""

import numpy as np
import pytest
from scipy.stats import kstest

from copul.checkerboard.bernstein import BernsteinCopula
from copul.checkerboard.biv_bernstein import BivBernsteinCopula


def _theta(m, n, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.random((m, n)) ** 3 + 1e-3
    for _ in range(3000):
        A = A / A.sum(1, keepdims=True) / m
        A = A / A.sum(0, keepdims=True) / n
    return A


def test_xi_condition_on_y_is_honoured():
    theta = _theta(3, 5)
    cop = BivBernsteinCopula(theta)
    xi_x = cop.chatterjees_xi()
    xi_y = cop.chatterjees_xi(condition_on_y=True)
    assert not np.isclose(xi_x, xi_y)
    assert np.isclose(xi_y, BivBernsteinCopula(theta.T).chatterjees_xi())
    # numerical reference 6 int int (d2 C)^2 - 2
    N = 400
    g = (np.arange(N) + 0.5) / N
    U, V = np.meshgrid(g, g, indexing="ij")
    ref = 6 * np.mean(cop.cond_distr_2(U, V) ** 2) - 2
    assert np.isclose(xi_y, ref, atol=1e-4)


def test_pdf_and_cond_distr_at_the_boundary():
    theta = _theta(3, 3, seed=1)
    cop = BivBernsteinCopula(theta)
    h = 1e-6
    p = cop.pdf(0.5, 1.0)
    num = (cop.cond_distr_1(0.5, 1.0) - cop.cond_distr_1(0.5, 1.0 - h)) / h
    assert np.isfinite(p) and np.isclose(p, num, atol=1e-4)
    c0 = cop.cond_distr_1(0.0, 0.5)
    assert c0 > 0 and np.isclose(c0, (cop.cdf(h, 0.5) - cop.cdf(0.0, 0.5)) / h, atol=1e-4)
    assert np.isclose(cop.cond_distr_1(1.0, 1.0), 1.0)
    assert np.isclose(cop.cond_distr_2(1.0, 1.0), 1.0)


def test_constructor_does_not_mutate_caller_array():
    theta = np.array([[1.0, 2.0], [2.0, 1.0]])
    BernsteinCopula(theta)
    BivBernsteinCopula(theta)
    BernsteinCopula(np.ones((2, 2, 2)) * 3)
    assert np.array_equal(theta, [[1.0, 2.0], [2.0, 1.0]])


def test_vectorised_cdf_conventions():
    cop = BivBernsteinCopula(_theta(4, 3, seed=2))
    rng = np.random.default_rng(0)
    pts = rng.random((50, 2))
    vec = cop.cdf(pts)
    assert np.allclose(vec, cop.cdf(pts[:, 0], pts[:, 1]))
    assert np.allclose(vec, [cop.cdf(a, b) for a, b in pts])
    assert np.isclose(cop.cdf(u=0.3, v=0.4), cop.cdf(0.3, 0.4))
    assert isinstance(cop.cdf(0.3, 0.4), float)


@pytest.mark.parametrize("shape", [(3, 3), (3, 5), (5, 2)])
def test_bernstein_measures_match_numerical(shape):
    cop = BivBernsteinCopula(_theta(*shape, seed=sum(shape)))
    N = 500
    g = (np.arange(N) + 0.5) / N
    U, V = np.meshgrid(g, g, indexing="ij")
    C = cop.cdf(U, V)
    assert np.isclose(cop.blests_nu(), 24 * np.mean((1 - U) * C) - 2, atol=1e-4)
    assert np.isclose(cop.spearmans_rho(), 12 * np.mean(C) - 3, atol=1e-4)
    t = (np.arange(100_000) + 0.5) / 100_000
    d, a = np.mean(cop.cdf(t, t)), np.mean(cop.cdf(t, 1 - t))
    assert np.isclose(cop.spearmans_footrule(), 6 * d - 2, atol=1e-6)
    assert np.isclose(cop.ginis_gamma(), 4 * (d + a) - 2, atol=1e-6)
    assert np.isclose(cop.blomqvists_beta(), 4 * cop.cdf(0.5, 0.5) - 1)


def test_bernstein_exact_sampler():
    cop = BivBernsteinCopula(_theta(3, 4, seed=4))
    X = cop.rvs(20_000, random_state=0)
    assert kstest(X[:, 0], "uniform").pvalue > 1e-3
    assert kstest(X[:, 1], "uniform").pvalue > 1e-3
    for u, v in [(0.3, 0.4), (0.7, 0.2), (0.5, 0.9)]:
        emp = np.mean((X[:, 0] <= u) & (X[:, 1] <= v))
        assert abs(emp - cop.cdf(u, v)) < 0.015
    assert np.allclose(X, cop.rvs(20_000, random_state=0))
