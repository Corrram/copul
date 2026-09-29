"""Tests of the numerical evaluation API beyond the universal properties."""

import time

import numpy as np
import pytest
import sympy as sp

import copul as cp
from copul.family.archimedean import _frailty
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper

P = np.random.default_rng(0).random((200, 2))


def test_symbolic_behaviour_without_arguments():
    assert isinstance(cp.Clayton().cdf(), SymPyFuncWrapper)
    assert isinstance(cp.Clayton(2).cdf(), SymPyFuncWrapper)
    assert isinstance(cp.Clayton(2).cdf(v=0.5), SymPyFuncWrapper)  # partial
    assert isinstance(cp.FarlieGumbelMorgenstern().cdf, SymPyFuncWrapper)  # property
    # free parameter: families with a symbolic evaluation keep it ...
    expr = cp.FarlieGumbelMorgenstern().cdf(u=0.3, v=0.2)
    assert "theta" in {str(s) for s in getattr(expr, "func", expr).free_symbols}
    # ... all others explain what is missing
    with pytest.raises(ValueError, match="free"):
        cp.Clayton().cdf(u=0.3, v=0.2)


def test_free_parameters():
    with pytest.raises(ValueError, match="free"):
        cp.Clayton().cdf(P)
    assert np.allclose(cp.Clayton().cdf(P, theta=2), cp.Clayton(2).cdf(P))
    assert np.allclose(cp.Frank().pdf(P[:, 0], P[:, 1], theta=3), cp.Frank(3).pdf(P))
    with pytest.raises(ValueError, match="free"):
        cp.Gaussian().rvs(5)


@pytest.mark.parametrize(
    "cop",
    [cp.Clayton(3), cp.Gaussian(0.95), cp.StudentT(0.9, 3), cp.GumbelHougaard(5), cp.Frank(20)],
    ids=repr,
)
def test_logpdf_is_stable_in_the_tails(cop):
    pts = np.array([[1e-12, 2e-12], [1 - 1e-12, 1 - 2e-12], [1e-9, 0.5], [0.5, 1 - 1e-9]])
    lp = cop.logpdf(pts)
    assert np.all(np.isfinite(lp))


def test_scalar_and_array_types():
    c = cp.Joe(2)
    assert isinstance(c.cdf(0.3, 0.4), float)
    assert isinstance(c.pdf([0.3, 0.4]), float)
    assert c.cdf(P).shape == (200,)
    assert c.cond_distr_1(np.array([0.3, 0.6]), 0.5).shape == (2,)
    assert isinstance(c.cond_distr_1_inv(0.3, 0.4), float)


def test_rvs_approximate_still_supported():
    s = cp.Clayton(2).rvs(100, random_state=1, approximate=True)
    assert s.shape == (100, 2)


def test_rvs_zero_and_legacy_size():
    assert cp.Frank(2).rvs(0).shape == (0, 2)
    assert cp.Frank(2).rvs(size=4, random_state=0).shape == (4, 2)


def test_rvs_does_not_touch_global_state():
    np.random.seed(123)
    expected = np.random.random(3)
    np.random.seed(123)
    cp.Gaussian(0.3).rvs(10, random_state=5)
    cp.IndependenceCopula().rvs(10, random_state=5)
    cp.Clayton(2).rvs(10, random_state=5)
    assert np.array_equal(np.random.random(3), expected)


@pytest.mark.parametrize(
    "sampler, lt",
    [
        (lambda n, r: _frailty.gamma_frailty(n, r, 0.5), lambda t: (1 + t) ** -0.5),
        (lambda n, r: _frailty.positive_stable(n, r, 0.4), lambda t: np.exp(-(t**0.4))),
        (
            lambda n, r: _frailty.logarithmic_frailty(n, r, 0.7),
            lambda t: np.log(1 - 0.7 * np.exp(-t)) / np.log(0.3),
        ),
        (lambda n, r: _frailty.sibuya(n, r, 0.6), lambda t: 1 - (1 - np.exp(-t)) ** 0.6),
        (
            lambda n, r: _frailty.geometric_frailty(n, r, 0.4),
            lambda t: 0.4 * np.exp(-t) / (1 - 0.6 * np.exp(-t)),
        ),
    ],
    ids=["gamma", "stable", "logarithmic", "sibuya", "geometric"],
)
def test_frailty_laplace_transforms(sampler, lt):
    rng = np.random.default_rng(3)
    x = sampler(200_000, rng)
    for t in (0.2, 1.0, 3.0):
        emp = np.exp(-t * x)
        assert abs(emp.mean() - lt(t)) < 5 * emp.std() / np.sqrt(x.size) + 1e-4


def test_closed_form_inverses_match_generic_solver():
    from copul.measures.backend import invert_h, numeric_backend

    u = np.linspace(0.05, 0.95, 7)[:, None] * np.ones((1, 7))
    w = np.linspace(0.05, 0.95, 7)[None, :] * np.ones((7, 1))
    for cop in (cp.Clayton(1.5), cp.Frank(-3), cp.Plackett(4), cp.Gaussian(0.4)):
        be = numeric_backend(cop)
        generic = invert_h(be.h1, be.pdf, u, w)
        assert np.allclose(cop.cond_distr_1_inv(u, w), generic, atol=1e-9)


def test_student_t_cdf_integer_and_fractional_nu():
    from scipy import integrate
    from scipy.special import stdtr, stdtrit

    for nu in (1, 4, 4.5):
        cop = cp.StudentT(0.6, nu)
        u, v = 0.3, 0.8
        x, y = stdtrit(nu, u), stdtrit(nu, v)

        def h(s, x=x, y=y, nu=nu):
            xs = stdtrit(nu, s)
            return stdtr(nu + 1, np.sqrt(nu + 1) * (y - 0.6 * xs) / (0.8 * np.sqrt(nu + xs**2)))

        ref = integrate.quad(h, 0, u, epsabs=1e-13, epsrel=1e-12, limit=200)[0]
        assert cop.cdf(u, v) == pytest.approx(ref, abs=1e-9)
        assert np.isfinite(x) and np.isfinite(y)


def test_symbolic_expression_values_unchanged():
    c = cp.Clayton(2)
    expr = c.cdf().func
    assert float(expr.subs({c.u: 0.3, c.v: 0.7})) == pytest.approx(c.cdf(0.3, 0.7))
    assert isinstance(sp.sympify(c.cdf().func), sp.Expr)


@pytest.mark.slow
@pytest.mark.parametrize(
    "cop",
    [cp.Gaussian(0.5), cp.Clayton(2), cp.Frank(3), cp.GumbelHougaard(2), cp.Joe(2)],
    ids=repr,
)
def test_rvs_speed(cop):
    cop.rvs(10, random_state=0)  # build caches
    t = time.perf_counter()
    cop.rvs(100_000, random_state=0)
    assert time.perf_counter() - t < 1.0
