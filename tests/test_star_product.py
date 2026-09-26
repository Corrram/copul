"""Tests for the Markov (*-) product."""

import numpy as np
import pytest

import copul
from copul.checkerboard.biv_check_min import BivCheckMin
from copul.checkerboard.biv_check_pi import BivCheckPi
from copul.checkerboard.biv_check_w import BivCheckW
from copul.star_product import markov_product


def _cb(m, n, seed):
    rng = np.random.default_rng(seed)
    A = rng.random((m, n)) ** 2 + 1e-3
    for _ in range(3000):
        A = A / A.sum(1, keepdims=True) / m
        A = A / A.sum(0, keepdims=True) / n
    return BivCheckPi(A)


def _definition(A, B, u, v, n_quad=60_000):
    """Direct quadrature of int d2A(u,t) d1B(t,v) dt."""
    t = (np.arange(n_quad) + 0.5) / n_quad
    return np.mean(A.cond_distr_2(np.full_like(t, u), t) * B.cond_distr_1(t, np.full_like(t, v)))


PTS = [(0.2, 0.3), (0.45, 0.8), (0.9, 0.1), (0.6, 0.6)]


def test_exact_checkerboard_product_matches_definition_and_formula():
    A, B = _cb(4, 4, 0), _cb(4, 4, 1)
    S = markov_product(A, B)
    assert isinstance(S, BivCheckPi)
    assert np.allclose(S.matr, 4 * A.matr @ B.matr)
    for u, v in PTS:
        assert np.isclose(S.cdf(u, v), _definition(A, B, u, v), atol=1e-6)


def test_exact_product_for_incompatible_grids():
    A, B = _cb(2, 3, 2), _cb(4, 5, 3)
    S = markov_product(A, B)
    assert S.matr.shape == (2, 5)
    for u, v in PTS:
        assert np.isclose(S.cdf(u, v), _definition(A, B, u, v), atol=1e-6)


def test_checkerboard_product_is_associative():
    A, B, C = _cb(3, 3, 4), _cb(3, 3, 5), _cb(3, 3, 6)
    left = markov_product(markov_product(A, B), C)
    right = markov_product(A, markov_product(B, C))
    assert np.allclose(left.matr, right.matr)


def test_independence_is_absorbing():
    C = _cb(4, 3, 7)
    Pi = BivCheckPi(np.ones((3, 3)))
    assert np.allclose(markov_product(Pi, _cb(3, 3, 8)).matr, 1 / 9)
    assert np.allclose(markov_product(C, Pi).matr, 1 / 12)
    # symbolic independence copula (general numerical path)
    S = markov_product(copul.BivIndependenceCopula(), C, n_grid=20, n_quad=400)
    uu = np.linspace(0, 1, 11)
    U, V = np.meshgrid(uu, uu, indexing="ij")
    assert np.allclose(S.cdf(U, V), U * V, atol=1e-10)


def test_upper_frechet_is_neutral():
    C = _cb(4, 4, 9)
    M = BivCheckMin(np.eye(4))  # exactly the upper Frechet bound
    for S in (
        markov_product(M, C, n_grid=40, n_quad=800),
        markov_product(C, M, n_grid=40, n_quad=800),
    ):
        for u, v in PTS:
            assert np.isclose(S.cdf(u, v), C.cdf(u, v), atol=2e-3)


def test_w_times_w_is_m():
    W = BivCheckW(np.fliplr(np.eye(3)))  # exactly the lower Frechet bound
    S = markov_product(W, W, n_grid=60, n_quad=1200)
    for u, v in PTS:
        assert np.isclose(S.cdf(u, v), min(u, v), atol=1 / (4 * 60) + 1e-9)
    assert S.spearmans_rho() > 0.99


def test_symbolic_families():
    frank = copul.Frank(theta=3)
    S = markov_product(copul.UpperFrechet(), frank, n_grid=40, n_quad=800)
    for u, v in PTS:
        assert np.isclose(S.cdf(u, v), float(frank.cdf(u, v)), atol=5e-3)


def test_checkerboard_keyword_is_deprecated_not_recursive():
    A = BivCheckMin(np.eye(2))
    with pytest.warns(DeprecationWarning):
        S = markov_product(A, A, checkerboard=True, n_grid=10, n_quad=200)
    assert np.isfinite(S.pdf(0.4, 0.7))
