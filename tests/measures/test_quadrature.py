import numpy as np
import pytest

from copul.measures.quadrature import (
    gauss_legendre_1d,
    gauss_legendre_2d,
    integrate_1d,
    integrate_1d_batch,
    integrate_2d,
)


def test_integrate_1d_smooth_and_singular():
    val, err = integrate_1d(np.sin, 0, np.pi)
    assert val == pytest.approx(2.0, abs=1e-12)
    assert err < 1e-8
    val, err = integrate_1d(lambda x: 1 / np.sqrt(x))
    assert val == pytest.approx(2.0, abs=1e-6)


def test_integrate_1d_never_evaluates_endpoints():
    seen = []

    def f(x):
        seen.append(np.asarray(x).copy())
        return np.log(x) + np.log1p(-x)

    val, _ = integrate_1d(f)
    allx = np.concatenate(seen)
    assert np.all((allx > 0) & (allx < 1))
    assert val == pytest.approx(-2.0, abs=1e-7)


def test_batch_steps_including_near_breakpoints():
    # jumps close to the graded breakpoints and to the domain ends
    v = np.concatenate(
        [np.random.default_rng(1).uniform(0, 1, 400), [1e-5, 0.49999, 0.2499, 1 - 1e-5]]
    )
    vals, errs = integrate_1d_batch(
        lambda u, r: (u < v[r]).astype(float),
        np.zeros(v.size),
        np.ones(v.size),
        atol=1e-13,
        rtol=1e-11,
    )
    assert np.max(np.abs(vals - v)) < 1e-9
    assert np.all(errs < 1e-8)


def test_batch_row_limits():
    a = np.array([0.0, 0.2, 0.5])
    b = np.array([1.0, 0.7, 0.5])
    vals, _ = integrate_1d_batch(lambda x, r: 2 * x, a, b)
    np.testing.assert_allclose(vals, b**2 - a**2, atol=1e-13)


def test_integrate_2d():
    val, err = integrate_2d(lambda u, v: np.minimum(u, v), rtol=1e-11, atol=1e-13)
    assert val == pytest.approx(1 / 3, abs=1e-10)
    val, _ = integrate_2d(lambda u, v: (u < v).astype(float), rtol=1e-11, atol=1e-13)
    assert val == pytest.approx(0.5, abs=1e-9)
    # triangle via inner limits
    val, _ = integrate_2d(lambda u, v: np.ones_like(u), inner_limits=lambda v: (0 * v, v))
    assert val == pytest.approx(0.5, abs=1e-12)


def test_gauss_legendre_doubling():
    val, err = gauss_legendre_1d(np.exp, rtol=1e-13)
    assert val == pytest.approx(np.e - 1, abs=1e-13)
    val, err = gauss_legendre_2d(lambda u, v: u * v**2)
    assert val == pytest.approx(1 / 6, abs=1e-13)
