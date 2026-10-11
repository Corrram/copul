"""Independent precision checks for the xi--nu boundary and its inversion."""

import mpmath as mp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from copul.family.other.clamped_parabola_copula import XiNuBoundaryCopula
from copul.regions.catalog import _b_of_xi_nu, nu_max_given_xi, xi_nu_parametric


def _high_precision_boundary(b):
    # Evaluate the original closed forms, not the rationalized implementation.
    # 100 digits retain ample precision after the O(b^-3) cancellation at 1e20.
    with mp.workdps(100):
        b = mp.mpf(str(b))
        s2 = 1 / b
        if b <= 1:
            xi = 8 * (7 * s2 - 3) / (105 * s2**3)
            nu = 4 * (28 * s2 - 9) / (105 * s2**2)
        else:
            t = mp.sqrt(1 - s2)
            A = mp.asinh(t / mp.sqrt(s2))
            xi = (
                -105 * s2**4 * A
                + 183 * s2**3 * t
                - 38 * s2**2 * t
                - 88 * s2 * t
                + 112 * s2
                + 48 * t
                - 48
            ) / (210 * s2**3)
            nu = (
                -105 * s2**4 * A
                + 87 * s2**3 * t
                + 250 * s2**2 * t
                - 376 * s2 * t
                + 448 * s2
                + 144 * t
                - 144
            ) / (420 * s2**2)
        return float(xi), float(nu)


@pytest.mark.parametrize(
    "b", [1e-150, 1e-10, 0.2, 0.99999, 1.0, 1.00001, 2.0, 10.0, 1e4, 1e8, 1e12, 1e20]
)
def test_boundary_matches_high_precision_original_formulas(b):
    expected = _high_precision_boundary(b)
    assert_allclose(xi_nu_parametric(b), expected, rtol=2e-14, atol=0)
    for sign in (-1, 1):
        copula = XiNuBoundaryCopula(b=sign * b)
        assert_allclose(copula.chatterjees_xi(method="closed"), expected[0], rtol=2e-14, atol=0)
        assert_allclose(copula.blests_nu(method="closed"), sign * expected[1], rtol=2e-14, atol=0)


def test_vectorized_boundary_shape_range_and_monotonicity_in_double_precision():
    b = np.geomspace(1e-150, 1e300, 600).reshape(20, 30)
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        xi, nu = xi_nu_parametric(b)
    assert xi.shape == nu.shape == b.shape
    for values in (xi, nu):
        assert values.dtype == np.float64
        assert np.all(np.isfinite(values))
        assert np.all((values >= 0) & (values <= 1))
        assert np.all(np.diff(values.ravel()) >= -5 * np.finfo(float).eps)
    assert xi_nu_parametric([])[0].shape == (0,)


def test_inversion_roundtrip_including_values_near_endpoints():
    x = np.r_[
        1e-100, 1e-30, 1e-16, np.linspace(0.001, 0.999, 65), 1 - 1e-10, np.nextafter(1.0, 0.0)
    ]
    b = _b_of_xi_nu(x)
    assert_allclose(xi_nu_parametric(b)[0], x, rtol=1e-12, atol=0)
    assert np.all(np.diff(b) > 0)
    upper = nu_max_given_xi(x)
    assert np.all(np.diff(upper) >= -5 * np.finfo(float).eps)
    assert_allclose(nu_max_given_xi([0, 1]), [0, 1], rtol=0, atol=0)


@pytest.mark.parametrize("b", [0, -1, np.inf, -np.inf, np.nan, [1, 0]])
def test_invalid_boundary_parameters_are_rejected(b):
    with pytest.raises(ValueError, match="positive finite"):
        xi_nu_parametric(b)
