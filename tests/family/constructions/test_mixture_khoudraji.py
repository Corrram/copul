import numpy as np
import pytest

import copul as cp
from copul.family.constructions import KhoudrajiCopula, MixtureCopula, khoudraji, mixture, rotate
from tests.family.numeric_copula_checks import (
    check_axioms,
    check_conditionals,
    check_density,
    check_sampling,
    closed_vs_numeric,
)

U = np.array([0.1, 0.3, 0.5, 0.72, 0.9])
V = np.array([0.8, 0.2, 0.5, 0.35, 0.95])


def _cdf(C, u, v):
    return np.array([float(C.cdf(a, b)) for a, b in zip(u, v)])


# ------------------------------------------------------------------ mixture


def test_mixture_basic_properties():
    C = mixture([cp.Clayton(2), cp.Frank(-3), rotate(cp.Clayton(2), 180)], [0.5, 0.2, 0.3])
    assert isinstance(C, MixtureCopula)
    assert C.is_absolutely_continuous
    check_axioms(C)
    check_conditionals(C)
    check_density(C)
    check_sampling(C)


def test_mixture_cdf_is_convex_combination():
    A, B = cp.Clayton(3), cp.Frank(2)
    C = mixture([A, B], [0.25, 0.75])
    np.testing.assert_allclose(C.cdf(U, V), 0.25 * _cdf(A, U, V) + 0.75 * _cdf(B, U, V), atol=1e-12)


def test_mixture_degenerate_weights():
    A, B = cp.Clayton(3), cp.Frank(2)
    np.testing.assert_allclose(mixture([A, B], [1, 0]).cdf(U, V), _cdf(A, U, V), atol=1e-12)
    np.testing.assert_allclose(mixture([A, B], [0, 1]).cdf(U, V), _cdf(B, U, V), atol=1e-12)
    # weights are normalised, default is equal weights
    np.testing.assert_allclose(mixture([A, B], [2, 2]).cdf(U, V), mixture([A, B]).cdf(U, V))


def test_mixture_frechet_family():
    # Frechet copula alpha M + beta W + (1 - alpha - beta) Pi
    a, b = 0.3, 0.2
    C = mixture(
        [cp.UpperFrechet(), cp.LowerFrechet(), cp.BivIndependenceCopula()], [a, b, 1 - a - b]
    )
    assert not C.is_absolutely_continuous
    check_axioms(C)
    assert C.spearmans_rho() == pytest.approx(a - b, abs=1e-12)
    assert C.lambda_L() == pytest.approx(a, abs=1e-12)
    assert C.blomqvists_beta() == pytest.approx(a - b, abs=1e-12)
    check_sampling(C)


def test_mixture_linear_measures_against_numerics():
    C = mixture([cp.Clayton(2), cp.GumbelHougaard(2)], [0.6, 0.4])
    keys = ["rho", "tau", "footrule", "gamma", "beta", "nu", "xi", "lambda_l", "lambda_u"]
    closed = closed_vs_numeric(C, keys, tol=1e-7)
    assert set(closed) == {"rho", "footrule", "gamma", "beta", "nu", "lambda_l", "lambda_u"}
    assert C.lambda_L() == pytest.approx(0.6 * 2**-0.5, abs=1e-10)
    assert C.lambda_U() == pytest.approx(0.4 * (2 - 2**0.5), abs=1e-10)


def test_mixture_errors():
    with pytest.raises(ValueError):
        mixture([])
    with pytest.raises(ValueError):
        mixture([cp.Clayton(2)], [-1])
    with pytest.raises(ValueError):
        mixture([cp.Clayton(2), cp.Frank(1)], [1])
    with pytest.raises(ValueError):
        mixture([cp.Clayton()])


# ------------------------------------------------------------------ Khoudraji


def _k():
    return khoudraji(cp.BivIndependenceCopula(), cp.GumbelHougaard(3), 0.3, 0.9)


def test_khoudraji_formula():
    G = cp.GumbelHougaard(3)
    K = _k()
    assert isinstance(K, KhoudrajiCopula)
    ref = U**0.7 * V**0.1 * _cdf(G, U**0.3, V**0.9)
    np.testing.assert_allclose(K.cdf(U, V), ref, atol=1e-12)


def test_khoudraji_copula_properties():
    K = khoudraji(cp.Frank(3), cp.Clayton(4), 0.4, 0.8)
    assert K.is_absolutely_continuous
    check_axioms(K)
    check_conditionals(K)
    check_density(K)
    check_sampling(K)


def test_khoudraji_singular_component_sampling():
    # Pi and M give a Marshall-Olkin type copula with a singular part
    K = khoudraji(cp.BivIndependenceCopula(), cp.UpperFrechet(), 0.5, 0.25)
    assert not K.is_absolutely_continuous
    check_axioms(K)
    check_sampling(K)
    # C(u,v) = u^{1-a} v^{1-b} min(u^a, v^b)
    np.testing.assert_allclose(
        K.cdf(U, V), U**0.5 * V**0.75 * np.minimum(U**0.5, V**0.25), atol=1e-12
    )


def test_khoudraji_special_cases():
    A, B = cp.Clayton(2), cp.Frank(5)
    np.testing.assert_allclose(khoudraji(A, B, 0, 0).cdf(U, V), _cdf(A, U, V), atol=1e-12)
    np.testing.assert_allclose(khoudraji(A, B, 1, 1).cdf(U, V), _cdf(B, U, V), atol=1e-12)
    x = khoudraji(A, B, 0, 0).rvs(2000, random_state=1)
    assert x.shape == (2000, 2)


def test_khoudraji_asymmetry_and_measures():
    K = _k()
    assert not K.is_symmetric
    assert K.chatterjees_xi() != pytest.approx(K.chatterjees_xi(condition_on_y=True), abs=1e-3)
    closed = closed_vs_numeric(K, ["beta", "rho"])
    assert closed == ["beta"]


def test_khoudraji_errors():
    with pytest.raises(ValueError):
        khoudraji(cp.BivIndependenceCopula(), cp.Clayton(2), 1.5, 0.2)
