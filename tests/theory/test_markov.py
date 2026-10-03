"""Tests for copul.theory.markov (Markov product, invertibility, Markov operators)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

import copul as cp
from copul.family.constructions import mixture
from copul.theory import markov as mk

M, W, PI = cp.UpperFrechet(), cp.LowerFrechet(), cp.BivIndependenceCopula()
X = np.array([0.07, 0.3, 0.55, 0.92])
U, V = np.meshgrid(X, X, indexing="ij")


# ---------------------------------------------------------------------------
# Markov product
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "C",
    [cp.Clayton(2), cp.MarshallOlkin(0.3, 0.7), cp.BivCheckMin([[0.2, 0.3], [0.3, 0.2]])],
    ids=["Clayton", "MarshallOlkin", "CheckMin"],
)
def test_neutral_null_and_reflection_elements(C):
    c = C.cdf(U, V)
    prod = mk.markov_product_cdf
    assert np.allclose(prod(PI, C, U, V), U * V, atol=1e-9)
    assert np.allclose(prod(C, PI, U, V), U * V, atol=1e-9)
    assert np.allclose(prod(M, C, U, V), c, atol=1e-8)
    assert np.allclose(prod(C, M, U, V), c, atol=1e-8)
    # W * C (u, v) = v - C(1-u, v),  C * W (u, v) = u - C(u, 1-v)
    assert np.allclose(prod(W, C, U, V), V - C.cdf(1 - U, V), atol=1e-8)
    assert np.allclose(prod(C, W, U, V), U - C.cdf(U, 1 - V), atol=1e-8)


def test_w_is_an_involution_and_fgm_products():
    assert np.allclose(mk.markov_product_cdf(W, W, U, V), np.minimum(U, V), atol=1e-9)
    # FGM_a * FGM_b = FGM_{ab/3}
    a, b = 0.9, -0.6
    lhs = mk.markov_product_cdf(cp.FarlieGumbelMorgenstern(a), cp.FarlieGumbelMorgenstern(b), U, V)
    assert np.allclose(lhs, cp.FarlieGumbelMorgenstern(a * b / 3).cdf(U, V), atol=1e-10)


def test_checkerboard_products_are_exact_and_agree_with_quadrature():
    A = cp.BivCheckPi([[2, 1], [1, 2]])
    B = cp.BivCheckPi([[1, 0, 2], [1, 2, 0], [1, 1, 1]])
    exact = mk.markov_product_cdf(A, B, U, V)
    numeric = mk.markov_product_cdf(mixture([A], [1.0]), B, U, V)
    assert np.allclose(exact, numeric, atol=1e-8)
    # associativity and (A * B)^T = B^T * A^T
    Cc = cp.BivCheckPi([[3, 1, 1], [1, 3, 1], [1, 1, 3]])
    left = mk.markov_product(mk.markov_product(A, B), Cc).matr
    right = mk.markov_product(A, mk.markov_product(B, Cc)).matr
    assert np.allclose(left / left.sum(), right / right.sum())
    t1 = mk.transpose(mk.markov_product(A, B)).matr
    t2 = mk.markov_product(mk.transpose(B), mk.transpose(A)).matr
    assert np.allclose(t1 / t1.sum(), t2 / t2.sum())


def test_markov_power():
    A = cp.BivCheckPi([[2, 1], [1, 2]])
    assert isinstance(mk.markov_power(A, 0), cp.UpperFrechet)
    assert mk.markov_power(A, 1) is A
    p3 = mk.markov_power(A, 3).matr
    ref = mk.markov_product(mk.markov_product(A, A), A).matr
    assert np.allclose(p3 / p3.sum(), ref / ref.sum())
    # powers of an independence-kernel checkerboard converge to Pi
    p20 = mk.markov_power(A, 20).matr
    assert np.allclose(p20 / p20.sum(), 0.25, atol=1e-9)
    with pytest.raises(ValueError):
        mk.markov_power(A, -1)


# ---------------------------------------------------------------------------
# idempotents and invertibility
# ---------------------------------------------------------------------------


def test_idempotents():
    assert mk.is_idempotent(PI).holds and mk.is_idempotent(PI).method == "exact"
    assert mk.is_idempotent(M).holds
    # ordinal sum of two copies of Pi
    assert mk.is_idempotent(cp.BivCheckPi([[1, 0], [0, 1]])).holds
    assert not mk.is_idempotent(W)  # W * W = M
    r = mk.is_idempotent(cp.Clayton(2))
    assert not r and r.worst_violation > 1e-2
    assert not mk.is_idempotent(cp.BivCheckPi([[1, 2], [2, 1]]))


TENT = cp.BivCheckMixed([[0.5], [0.5]], sign=[[1], [-1]])  # V = 1 - |2U - 1|


def test_complete_dependence_and_invertibility():
    for C in (M, W, cp.ShuffleOfMin([2, 4, 1, 3]), cp.ShuffleOfMin([3, 1, 2])):
        r = mk.is_invertible(C)
        assert r.holds and r.method == "exact"
        assert mk.is_mutually_completely_dependent(C)
    # V is a (two-to-one) function of U: left but not right invertible
    assert mk.is_left_invertible(TENT).holds
    assert mk.is_completely_dependent(TENT, i=1)
    r = mk.is_right_invertible(TENT)
    assert not r and r.worst_violation > 0.1
    assert not mk.is_mutually_completely_dependent(TENT)
    for C in (cp.Clayton(2), PI, cp.Gaussian(0.9)):
        assert not mk.is_left_invertible(C) and not mk.is_right_invertible(C)


def test_invertibility_product_route_agrees():
    S = cp.ShuffleOfMin([2, 4, 1, 3])
    assert mk.is_left_invertible(S, method="product").holds
    assert mk.is_right_invertible(S, method="product").holds
    assert mk.is_left_invertible(TENT, method="product").holds
    assert not mk.is_right_invertible(TENT, method="product").holds
    r = mk.is_left_invertible(cp.Clayton(2), method="product")
    assert not r and r.worst_violation > 0.05


def test_transpose():
    S = cp.ShuffleOfMin([2, 4, 1, 3])
    T = mk.transpose(S)
    assert np.allclose(T.cdf(U, V), S.cdf(V, U))
    assert np.allclose(mk.transpose(TENT).cdf(U, V), TENT.cdf(V, U))
    C = cp.MarshallOlkin(0.2, 0.9)
    assert np.allclose(mk.transpose(C).cdf(U, V), C.cdf(V, U))
    assert mk.transpose(M) is M


# ---------------------------------------------------------------------------
# Markov operator
# ---------------------------------------------------------------------------


def test_markov_operator_basic_identities():
    x = np.linspace(0.01, 0.99, 7)
    f = np.cos
    assert np.allclose(mk.markov_operator(M)(f)(x), f(x), atol=1e-12)
    assert np.allclose(mk.markov_operator(W)(f)(x), f(1 - x), atol=1e-12)
    assert np.allclose(mk.markov_operator(PI)(f)(x), np.sin(1.0), atol=1e-12)
    T = mk.markov_operator(cp.Clayton(2))
    assert np.allclose(T(np.ones_like)(x), 1.0)
    # T preserves the Lebesgue integral: int_0^1 E[f(V) | U = x] dx = int_0^1 f
    nodes, weights = np.polynomial.legendre.leggauss(64)
    xs, ws = 0.5 * (nodes + 1), 0.5 * weights
    assert np.dot(ws, T(f)(xs)) == pytest.approx(np.sin(1.0), abs=1e-6)


def test_markov_operator_closed_forms():
    x = np.array([0.05, 0.3, 0.5, 0.8])
    rho = 0.6
    # Gaussian copula: E[Phi^{-1}(V) | U = x] = rho Phi^{-1}(x)
    T = mk.markov_operator(cp.Gaussian(rho))
    assert np.allclose(T(norm.ppf)(x), rho * norm.ppf(x), atol=1e-8)
    # FGM: E[V | U = x] = 1/2 - theta (1 - 2x) / 6
    th = 0.6
    reg = mk.regression_function(cp.FarlieGumbelMorgenstern(th), x)
    assert np.allclose(reg, 0.5 - th * (1 - 2 * x) / 6, atol=1e-12)
    # checkerboard with comonotone cells (atoms of the conditional laws)
    assert np.allclose(mk.regression_function(cp.BivCheckMin(np.eye(2)), x), x, atol=1e-9)


def test_operator_of_product_is_composition():
    a, b = 0.8, -0.9
    A, B = cp.FarlieGumbelMorgenstern(a), cp.FarlieGumbelMorgenstern(b)
    AB = cp.FarlieGumbelMorgenstern(a * b / 3)  # = A * B
    x = np.linspace(0.02, 0.98, 6)
    f = np.exp
    lhs = mk.markov_operator(AB)(f)(x)
    rhs = mk.markov_operator(A)(mk.markov_operator(B)(f))(x)
    assert np.allclose(lhs, rhs, atol=1e-10)


def test_adjoint_and_conditional_expectation_given_v():
    C = cp.MarshallOlkin(0.3, 0.8)
    T = mk.markov_operator(C)
    nodes, weights = np.polynomial.legendre.leggauss(48)
    xs, ws = 0.5 * (nodes + 1), 0.5 * weights
    f, g = np.sin, np.exp
    lhs = np.dot(ws, g(xs) * T(f)(xs))
    rhs = np.dot(ws, f(xs) * T.adjoint(g)(xs))
    assert lhs == pytest.approx(rhs, abs=2e-3)  # Gauss-Legendre on kinked integrands
    y = np.array([0.2, 0.7])
    direct = mk.conditional_expectation(C, f, y, given=2)
    assert np.allclose(direct, mk.markov_operator(mk.transpose(C)).apply(f, y))
    with pytest.raises(ValueError):
        mk.conditional_expectation(C, f, y, given=3)


def test_markov_kernel():
    C = cp.Clayton(2)
    K = mk.markov_kernel(C)
    assert np.allclose(K(U, V), C.cond_distr_1(U, V))
    assert K(0.3, 0.0) == 0.0 and K(0.3, 1.0) == 1.0
