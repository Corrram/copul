"""Tests for copul.theory.extreme_value and numerical extreme-value copulas."""

import numpy as np
import pytest

import copul as cp
from copul.family.extreme_value.numeric_extreme_value import NumericExtremeValueCopula
from copul.theory.extreme_value import (
    check_pickands,
    ev_attractor,
    ev_copula_from_pickands,
    extremal_coefficient,
    is_extreme_value,
    is_max_stable,
    max_stability_defect,
    pickands_estimator,
    pickands_function,
    stable_tail_dependence,
    tail_copula,
)
from tests.family.numeric_copula_checks import (
    check_axioms,
    check_conditionals,
    check_density,
    check_sampling,
)

T = np.linspace(0.0, 1.0, 11)


def gumbel_A(theta):
    return lambda t: (t**theta + (1 - t) ** theta) ** (1 / theta)


def galambos_A(delta):
    def A(t):
        t = np.asarray(t, float)
        with np.errstate(all="ignore"):
            a = 1 - (t ** (-delta) + (1 - t) ** (-delta)) ** (-1 / delta)
        return np.where((t <= 0) | (t >= 1), 1.0, a)

    return A


EV_FAMILIES = [
    cp.JoeEV(0.5, 0.5, 2),
    cp.BB5(2, 2),
    cp.CuadrasAuge(0.5),
    cp.Galambos(0.5),
    cp.GumbelHougaardEV(3),
    cp.HueslerReiss(2),
    cp.Tawn(0.5, 0.5, 2),
    cp.tEV(2, 0.5),
    cp.MarshallOlkin(0.8, 0.5),
]


# ---------------------------------------------------------------------------
# characterisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("C", EV_FAMILIES, ids=lambda C: type(C).__name__)
def test_ev_families_are_max_stable_with_valid_pickands(C):
    assert max_stability_defect(C) < 1e-12
    assert is_extreme_value(C)
    rep = check_pickands(C)
    assert rep.valid, rep
    assert C.check_pickands().valid
    theta = extremal_coefficient(C)
    assert 1.0 <= theta <= 2.0
    assert theta == pytest.approx(2 - C.lambda_U(), abs=1e-12)
    assert rep.extremal_coefficient == pytest.approx(theta, abs=1e-12)
    u = np.array([0.1, 0.45, 0.8])
    np.testing.assert_allclose(C.cdf(u, u), u**theta, rtol=1e-10)  # C(u, u) = u^theta


@pytest.mark.parametrize(
    "C",
    [cp.Clayton(2), cp.Frank(3), cp.Gaussian(0.5), cp.Plackett(3), cp.FarlieGumbelMorgenstern(0.7)],
    ids=lambda C: type(C).__name__,
)
def test_non_ev_copulas_are_not_max_stable(C):
    assert max_stability_defect(C) > 1e-3
    assert not is_max_stable(C)
    assert not is_extreme_value(C)


@pytest.mark.parametrize(
    "C",
    [cp.GumbelHougaard(2.5), cp.BivIndependenceCopula(), cp.UpperFrechet()],
    ids=["gumbel-archimedean", "pi", "M"],
)
def test_other_ev_copulas(C):
    assert is_extreme_value(C)


def test_check_pickands_detects_violations():
    assert check_pickands(gumbel_A(2.5)).valid
    assert check_pickands(lambda t: np.ones_like(t)).valid  # independence
    assert check_pickands(lambda t: np.maximum(t, 1 - t)).valid  # M
    rep = check_pickands(lambda t: 1 - 0.3 * np.sin(np.pi * t) ** 4)  # not convex near 0, 1
    assert not rep.valid and rep.bounds_ok and not rep.convex_ok
    rep = check_pickands(lambda t: 1 - 1.5 * t * (1 - t))  # below max(t, 1 - t)
    assert not rep.valid and not rep.bounds_ok and rep.convex_ok
    rep = check_pickands(lambda t: 0.9 + 0.0 * t)  # wrong endpoints
    assert not rep.endpoints_ok and not bool(rep)
    with pytest.raises(TypeError):
        check_pickands(cp.Clayton(2))


# ---------------------------------------------------------------------------
# stable tail dependence, extremal coefficient, tail copulas
# ---------------------------------------------------------------------------


def test_stable_tail_dependence_function():
    C = cp.Galambos(2)
    x = np.array([0.0, 1.0, 2.0, 0.3])
    y = np.array([1.0, 1.0, 0.5, 0.0])
    ell = stable_tail_dependence(C, x, y)
    np.testing.assert_allclose(ell, (x + y) * galambos_A(2)(np.where(x + y > 0, y / (x + y), 0)))
    assert np.all(ell >= np.maximum(x, y) - 1e-14) and np.all(ell <= x + y + 1e-14)
    np.testing.assert_allclose(stable_tail_dependence(C, 2 * x, 2 * y), 2 * ell)
    assert C.stable_tail_dependence(1.0, 1.0) == pytest.approx(extremal_coefficient(C))
    # any copula: the stable tail dependence function of its attractor
    S = cp.survival(cp.Clayton(2))
    assert stable_tail_dependence(S, 1.0, 2.0) == pytest.approx(
        ell[2] * 0 + 3 * galambos_A(2)(2 / 3)
    )


def test_extremal_coefficient():
    assert extremal_coefficient(cp.UpperFrechet()) == pytest.approx(1.0)
    assert extremal_coefficient(cp.BivIndependenceCopula()) == pytest.approx(2.0)
    assert extremal_coefficient(cp.GumbelHougaardEV(3)) == pytest.approx(2 ** (1 / 3))
    assert extremal_coefficient(cp.Clayton(2)) == pytest.approx(2.0, abs=1e-9)
    T4 = cp.StudentT(0.5, 4)
    assert extremal_coefficient(T4) == pytest.approx(2 - T4.lambda_U(), abs=1e-6)


def test_tail_copula_lower_clayton_closed_form():
    th = 2.0
    x = np.array([1.0, 0.5, 2.0, 0.0])
    y = np.array([1.0, 1.0, 3.0, 1.0])
    lam = tail_copula(cp.Clayton(th), x, y)
    with np.errstate(divide="ignore"):
        ref = np.where(x > 0, (x**-th + y**-th) ** (-1 / th), 0.0)
    np.testing.assert_allclose(lam, ref, atol=1e-10)
    val, err = tail_copula(cp.Clayton(th), 1.0, 1.0, return_error=True)
    assert isinstance(val, float) and err < 1e-8


@pytest.mark.parametrize(
    "C, lower",
    [
        (cp.Clayton(1.5), True),
        (cp.BB1(0.7, 1.4), True),
        (cp.BB1(0.7, 1.4), False),
        (cp.GumbelHougaard(3), False),
        (cp.StudentT(0.5, 4), True),
        (cp.Joe(2), False),
    ],
    ids=["clayton-L", "bb1-L", "bb1-U", "gumbel-U", "t-L", "joe-U"],
)
def test_tail_copula_at_one_is_tail_dependence_coefficient(C, lower):
    lam = tail_copula(C, 1.0, 1.0, lower=lower)
    ref = C.lambda_L() if lower else C.lambda_U()
    assert lam == pytest.approx(ref, abs=1e-6)
    # homogeneity of order one
    assert tail_copula(C, 0.6, 1.8, lower=lower) == pytest.approx(
        3 * tail_copula(C, 0.2, 0.6, lower=lower), abs=1e-6
    )


def test_tail_copula_ev_exact_vs_numeric():
    from copul.theory.extreme_value import _tail_limits

    C = cp.Galambos(1.5)
    x = np.array([1.0, 0.4, 2.0])
    y = np.array([1.0, 1.3, 0.7])
    exact = tail_copula(C, x, y, lower=False)
    num, _ = _tail_limits(C, x, y, upper=True)
    np.testing.assert_allclose(num, exact, atol=1e-7)
    assert tail_copula(cp.Gaussian(0.5), 1.0, 1.0, lower=False) == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# extreme-value attractors
# ---------------------------------------------------------------------------


def test_attractor_of_ev_copula_is_itself():
    C = cp.Galambos(2)
    att = ev_attractor(C)
    assert isinstance(att, NumericExtremeValueCopula)
    np.testing.assert_allclose(att.pickands(T), galambos_A(2)(T), atol=1e-14)


@pytest.mark.parametrize("C", [cp.Clayton(2), cp.Frank(3), cp.Clayton(-0.5)], ids=str)
def test_attractor_independence(C):
    att = ev_attractor(C)
    np.testing.assert_allclose(att.pickands(T), 1.0, atol=0)
    assert extremal_coefficient(C) == pytest.approx(2.0, abs=1e-9)


@pytest.mark.parametrize(
    "C, A, tol",
    [
        (cp.GumbelHougaard(2), gumbel_A(2), 1e-6),  # Gumbel is its own attractor
        (cp.survival(cp.Clayton(2)), galambos_A(2), 1e-8),  # survival Clayton -> Galambos
        (cp.Joe(2), gumbel_A(2), 1e-8),  # phi(1 - s) ~ s^2: Gumbel(2)
        (cp.BB1(0.7, 1.4), gumbel_A(1.4), 1e-6),  # phi(1 - s) ~ (0.7 s)^1.4: Gumbel(1.4)
    ],
    ids=["gumbel", "survival-clayton", "joe", "bb1"],
)
def test_known_attractors(C, A, tol):
    np.testing.assert_allclose(pickands_function(C, T), A(T), atol=tol)
    att = ev_attractor(C, deg=32)
    np.testing.assert_allclose(att.pickands(T), A(T), atol=5e-5)
    assert att.pickands_report.valid


def test_attractor_gaussian_is_independence():
    a = pickands_function(cp.Gaussian(0.5), np.array([0.2, 0.5, 0.9]))
    np.testing.assert_allclose(a, 1.0, atol=1e-4)


def test_attractor_student_t_is_tev():
    t = np.array([0.1, 0.3, 0.5, 0.8])
    A_tev = cp.tEV(4, 0.5)._pickands_numpy()[0]
    np.testing.assert_allclose(pickands_function(cp.StudentT(0.5, 4), t), A_tev(t), atol=1e-5)


@pytest.mark.slow
def test_attractor_student_t_full():
    att = ev_attractor(cp.StudentT(0.5, 4))
    tev = cp.tEV(4, 0.5)
    np.testing.assert_allclose(att.pickands(T), tev._pickands_numpy()[0](T), atol=5e-5)
    assert att.spearmans_rho() == pytest.approx(tev.spearmans_rho(), abs=1e-4)


def test_attractor_is_a_usable_copula():
    att = ev_attractor(cp.survival(cp.Clayton(2)), deg=32)
    G = cp.Galambos(2)
    P = np.random.default_rng(1).random((200, 2))
    np.testing.assert_allclose(att.cdf(P), G.cdf(P), atol=1e-6)
    assert att.spearmans_rho() == pytest.approx(G.spearmans_rho(), abs=1e-5)
    assert att.kendalls_tau() == pytest.approx(G.kendalls_tau(), abs=1e-5)
    assert att.lambda_U() == pytest.approx(G.lambda_U(), abs=1e-6)
    check_sampling(att, n=10_000)


# ---------------------------------------------------------------------------
# numerical extreme-value copulas
# ---------------------------------------------------------------------------


def test_ev_copula_from_pickands_matches_family():
    C = ev_copula_from_pickands(gumbel_A(2))
    G = cp.GumbelHougaardEV(2)
    P = np.random.default_rng(2).random((300, 2))
    np.testing.assert_allclose(C.cdf(P), G.cdf(P), atol=1e-14)
    np.testing.assert_allclose(C.cond_distr_1(P), G.cond_distr_1(P), atol=1e-9)
    np.testing.assert_allclose(C.pdf(P), G.pdf(P), rtol=1e-4)  # A'' by finite differences
    assert C.spearmans_rho() == pytest.approx(G.spearmans_rho(), abs=1e-9)
    assert C.kendalls_tau() == pytest.approx(0.5, abs=1e-9)
    assert C.blomqvists_beta() == pytest.approx(G.blomqvists_beta(), abs=1e-12)
    assert C.lambda_U() == pytest.approx(2 - 2**0.5, abs=1e-12)
    assert C.lambda_L() == 0.0
    assert C.is_absolutely_continuous and C.is_symmetric
    assert C.extremal_coefficient() == pytest.approx(2**0.5)


def test_numeric_ev_copula_axioms_and_sampling():
    C = ev_copula_from_pickands(lambda t: 1 - 0.5 * t * (1 - t))  # Tawn's mixed model
    check_axioms(C)
    check_conditionals(C)
    check_density(C, total=False)
    check_sampling(C)
    assert C.lambda_U() == pytest.approx(0.25)
    with pytest.raises(TypeError):
        C.cdf()
    assert C() is C


def test_kinked_pickands_marshall_olkin():
    a1, a2 = 0.8, 0.5

    def A(t):
        return np.maximum(1 - a1 * (1 - t), 1 - a2 * t)

    def dA(t):
        t = np.asarray(t, float)
        return np.where(1 - a1 * (1 - t) >= 1 - a2 * t, a1, -a2)

    C = ev_copula_from_pickands(A, dA, lambda t: 0.0 * np.asarray(t, float))
    MO = cp.MarshallOlkin(a1, a2)
    assert not C.is_absolutely_continuous
    assert not C.is_symmetric
    P = np.random.default_rng(3).random((200, 2))
    np.testing.assert_allclose(C.cdf(P), MO.cdf(P), atol=1e-14)
    assert C.kendalls_tau() == pytest.approx(a1 * a2 / (a1 + a2 - a1 * a2), abs=1e-9)
    assert C.spearmans_rho() == pytest.approx(3 * a1 * a2 / (2 * a1 + 2 * a2 - a1 * a2), abs=1e-9)
    check_sampling(C, n=10_000)


def test_ev_copula_from_pickands_rejects_invalid():
    with pytest.raises(ValueError):
        ev_copula_from_pickands(lambda t: 1 - 0.3 * np.sin(np.pi * t) ** 4)
    C = ev_copula_from_pickands(lambda t: 1 - 0.3 * np.sin(np.pi * t) ** 4, check=False)
    assert not C.check_pickands().valid


# ---------------------------------------------------------------------------
# Pickands estimators
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gumbel_sample():
    return cp.GumbelHougaardEV(2).rvs(5000, random_state=0)


@pytest.mark.parametrize("method", ["cfg", "pickands"])
def test_pickands_estimators_recover_gumbel(gumbel_sample, method):
    t = np.linspace(0, 1, 41)
    est = pickands_estimator(gumbel_sample, t, method=method)
    np.testing.assert_allclose(est, gumbel_A(2)(t), atol=0.02)
    assert est[0] == pytest.approx(1.0) and est[-1] == pytest.approx(1.0)


def test_pickands_estimator_options(gumbel_sample):
    t = np.linspace(0, 1, 41)
    raw = pickands_estimator(gumbel_sample, t, endpoint_correction=False)
    np.testing.assert_allclose(raw, gumbel_A(2)(t), atol=0.03)
    cvx = pickands_estimator(gumbel_sample, t, convexify=True)
    assert check_pickands(lambda s: np.interp(s, t, cvx)).valid
    np.testing.assert_allclose(cvx, gumbel_A(2)(t), atol=0.02)
    # invariant under monotone transformations of the margins (rank-based)
    X = np.column_stack([np.exp(gumbel_sample[:, 0]), -np.log1p(-gumbel_sample[:, 1])])
    np.testing.assert_allclose(pickands_estimator(X, t), pickands_estimator(gumbel_sample, t))
    assert isinstance(pickands_estimator(gumbel_sample, 0.5), float)
    with pytest.raises(ValueError):
        pickands_estimator(gumbel_sample, t, method="foo")
    with pytest.raises(ValueError):
        pickands_estimator(gumbel_sample[:, 0], t)
    # an estimated Pickands function defines a usable EV copula
    C = ev_copula_from_pickands(lambda s: np.interp(s, t, cvx), check=False)
    assert C.kendalls_tau() == pytest.approx(0.5, abs=0.05)
