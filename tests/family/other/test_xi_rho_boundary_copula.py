# tests/family/other/test_xi_rho_boundary_copula.py
import numpy as np
import pytest

from copul.family.other.xi_rho_boundary_copula import XiRhoBoundaryCopula
from copul.family.frechet.upper_frechet import UpperFrechet
from copul.family.frechet.lower_frechet import LowerFrechet
from copul.family.frechet.biv_independence_copula import BivIndependenceCopula
from scipy.integrate import dblquad

RTOL = 1e-9
ATOL = 1e-9


def grid(n=11):
    u = np.linspace(0.0, 1.0, n)
    v = np.linspace(0.0, 1.0, n)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    return uu, vv


def xi_from_bnew(b):
    """Chatterjee's xi expressed in terms of b_new (the class parameter)."""
    ab = abs(float(b))
    if ab >= 1.0:
        # |b_new| >= 1  <=> |b_old| <= 1
        return (1.0 / (10.0 * ab**2)) * (5.0 - 2.0 / ab)
    else:
        # |b_new| < 1  <=> |b_old| >= 1
        return 1.0 - ab + 0.3 * ab**2


@pytest.mark.parametrize("b", [0.5, 1.0, 2.0, -1.5])
def test_cdf_boundary_conditions(b):
    C = XiRhoBoundaryCopula(b=b)

    u = np.linspace(0, 1, 51)
    v = np.linspace(0, 1, 51)

    # C(u,0) = 0
    assert np.allclose(C.cdf_vectorized(u, 0.0), 0.0, rtol=RTOL, atol=ATOL)

    # C(0,v) = 0
    assert np.allclose(C.cdf_vectorized(0.0, v), 0.0, rtol=RTOL, atol=ATOL)

    # C(u,1) = u
    Cu1 = C.cdf_vectorized(u, 1.0)
    assert np.allclose(Cu1, u, rtol=RTOL, atol=ATOL)

    # C(1,v) = v
    C1v = C.cdf_vectorized(1.0, v)
    assert np.allclose(C1v, v, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("b", [0.4, 1.2, -0.9, -2.5])
def test_frechet_bounds(b):
    C = XiRhoBoundaryCopula(b=b)
    uu, vv = grid(n=21)
    Cuv = C.cdf_vectorized(uu, vv)

    W = np.maximum(uu + vv - 1.0, 0.0)  # Lower Fréchet bound
    M = np.minimum(uu, vv)  # Upper Fréchet bound

    assert np.all(Cuv >= W - 1e-12)
    assert np.all(Cuv <= M + 1e-12)


def test_special_case_b_zero_is_independence():
    C = XiRhoBoundaryCopula(0.0)
    assert isinstance(C, BivIndependenceCopula)


@pytest.mark.parametrize("bpos", [0.5, 1.0, 2.0])
def test_reflection_identity_for_negative_b(bpos):
    Cpos = XiRhoBoundaryCopula(b=bpos)
    Cneg = XiRhoBoundaryCopula(b=-bpos)

    uu, vv = grid(n=25)

    left = Cneg.cdf_vectorized(uu, vv)
    right = vv - Cpos.cdf_vectorized(1.0 - uu, vv)
    assert np.allclose(left, right, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("b", [0.5, 0.9, 1.1, 2.0, -0.7, -3.0])
def test_monotonicity_in_u_and_v(b):
    C = XiRhoBoundaryCopula(b=b)
    u = np.linspace(0, 1, 101)
    v = np.linspace(0, 1, 101)

    # Nondecreasing in u for fixed v
    for vv in [0.0, 0.1, 0.5, 0.9, 1.0]:
        vals = C.cdf_vectorized(u, vv)
        diffs = np.diff(vals)
        assert np.all(diffs >= -1e-10)

    # Nondecreasing in v for fixed u
    for uu in [0.0, 0.1, 0.5, 0.9, 1.0]:
        vals = C.cdf_vectorized(uu, v)
        diffs = np.diff(vals)
        assert np.all(diffs >= -1e-10)


@pytest.mark.parametrize("b", [0.4, 0.8, 1.2, 2.5])
def test_vectorized_shape_and_scalar_consistency(b):
    C = XiRhoBoundaryCopula(b=b)
    uu, vv = grid(n=13)
    Cuv = C.cdf_vectorized(uu, vv)
    assert Cuv.shape == uu.shape == vv.shape

    # Compare a few scalar points with the vectorized outputs
    for u, v in [(0.2, 0.3), (0.7, 0.1), (0.55, 0.8)]:
        scalar = float(C.cdf_vectorized(u, v))
        vect = float(C.cdf_vectorized(np.array([u]), np.array([v]))[0])
        assert np.isclose(scalar, vect, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("b", [0.5, 1.5, 2.0, -0.5, -1.5])
def test_measure_sign_matches_parameter_sign(b):
    C = XiRhoBoundaryCopula(b=b)
    rho = float(C.spearmans_rho())
    tau = float(C.kendalls_tau())
    # The formulas imply sign(rho) = sign(b) and sign(tau) = sign(b)
    if b > 0:
        assert rho > 0 and tau > 0
    else:
        assert rho < 0 and tau < 0


def test_from_xi_endpoints_and_sign():
    assert isinstance(XiRhoBoundaryCopula.from_xi(1), UpperFrechet)
    assert isinstance(XiRhoBoundaryCopula.from_xi(-1), LowerFrechet)


# --- NEW TEST CASE ---
@pytest.mark.parametrize("b_pos", [0.4, 1.0, 2.5])
def test_measures_are_odd_functions(b_pos):
    """
    Tests that rho, tau, and nu are odd functions of b,
    i.e., f(-b) = -f(b). This is a direct consequence of the
    C_(-b) = v - C_b(1-u, v) reflection and the sign(b) * g(|b|)
    structure of the implemented formulas.
    """
    # Create copula for positive parameter
    C_pos = XiRhoBoundaryCopula(b=b_pos)
    # Create copula for negative parameter
    C_neg = XiRhoBoundaryCopula(b=-b_pos)

    # Get measures for positive b
    rho_pos = float(C_pos.spearmans_rho())
    tau_pos = float(C_pos.kendalls_tau())
    nu_pos = float(C_pos.blests_nu())
    xi_pos = float(C_pos.chatterjees_xi())

    # Get measures for negative b
    rho_neg = float(C_neg.spearmans_rho())
    tau_neg = float(C_neg.kendalls_tau())
    nu_neg = float(C_neg.blests_nu())
    xi_neg = float(C_neg.chatterjees_xi())

    # Assert f(-b) == -f(b)
    assert np.isclose(rho_neg, -rho_pos, rtol=RTOL, atol=ATOL)
    assert np.isclose(tau_neg, -tau_pos, rtol=RTOL, atol=ATOL)
    assert np.isclose(nu_neg, -nu_pos, rtol=RTOL, atol=ATOL)
    assert np.isclose(xi_neg, xi_pos, rtol=RTOL, atol=ATOL)

    rho_neg_ccop = C_neg.to_checkerboard(grid_size=80).spearmans_rho()
    rho_pos_ccop = C_pos.to_checkerboard(grid_size=80).spearmans_rho()
    assert np.isclose(rho_neg_ccop, rho_neg, atol=1e-2)
    assert np.isclose(rho_pos_ccop, rho_pos, atol=1e-2)

    tau_neg_ccop = C_neg.to_checkerboard(grid_size=80).kendalls_tau()
    tau_pos_ccop = C_pos.to_checkerboard(grid_size=80).kendalls_tau()
    assert np.isclose(tau_neg_ccop, tau_neg, atol=1e-2)
    assert np.isclose(tau_pos_ccop, tau_pos, atol=1e-2)

    xi_neg_ccop = C_neg.to_checkerboard(grid_size=80).chatterjees_xi()
    xi_pos_ccop = C_pos.to_checkerboard(grid_size=80).chatterjees_xi()
    assert np.isclose(xi_neg_ccop, xi_neg, atol=1e-2)
    assert np.isclose(xi_pos_ccop, xi_pos, atol=1e-2)

    nu_pos_ccop = C_pos.to_checkerboard(grid_size=80).blests_nu()
    nu_neg_ccop = C_neg.to_checkerboard(grid_size=80).blests_nu()
    assert np.isclose(nu_neg_ccop, nu_neg, atol=1e-2)
    assert np.isclose(nu_pos_ccop, nu_pos, atol=1e-2)


@pytest.mark.parametrize("b", [0.4, 1.0, 3.0])
def test_sign_behavior(b):
    Cpos, Cneg = XiRhoBoundaryCopula(b=b), XiRhoBoundaryCopula(b=-b)
    assert np.isclose(float(Cpos.chatterjees_xi()), float(Cneg.chatterjees_xi()))
    for f in (Cpos.spearmans_rho, Cpos.kendalls_tau, Cpos.blests_nu):
        fp, fn = float(f()), float(getattr(Cneg, f.__name__)())
        assert np.isclose(fn, -fp)


@pytest.mark.parametrize("b", [5.0, 20.0])
def test_tau_limits(b):
    C = XiRhoBoundaryCopula(b=b)
    assert float(C.kendalls_tau()) > 0.8
    Cn = XiRhoBoundaryCopula(b=-b)
    assert float(Cn.kendalls_tau()) < -0.8


def test_order_of_params():
    ccop1 = XiRhoBoundaryCopula(b=0.5).to_checkerboard()
    ccop2 = XiRhoBoundaryCopula(b=2).to_checkerboard()
    assert ccop1.chatterjees_xi() < ccop2.chatterjees_xi()
    assert ccop1.spearmans_rho() < ccop2.spearmans_rho()


# @pytest.mark.slow
# @pytest.mark.parametrize("b", [0.5, 2.0])
# def test_pdf_integration_is_one(b):
#     """
#     Integrate the explicit PDF over the unit square [0,1]^2.
#     Must equal 1.0.
#     """
#     C = XiRhoBoundaryCopula(b=b)
#     pdf_func = C.pdf  # This uses the sympy-compiled lambda
#
#     # We use a slight epsilon to avoid singularities at boundary if any
#     res, err = dblquad(
#         lambda v, u: float(pdf_func(u, v)), 0, 1, 0, 1, epsabs=1e-5, epsrel=1e-5
#     )
#
#     assert np.isclose(res, 1.0, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("b", [2.0])
def test_pdf_value_in_center_large_b(b):
    """
    For |b| >= 1, the density in the 'middle' band (near v=0.5)
    should be exactly |b|.
    """
    C = XiRhoBoundaryCopula(b=b)
    # v=0.5 is definitely in the middle linear section for |b| >= 1
    # Check u=0.5 (which is on the diagonal u=v approx)
    val = float(C.pdf(0.5, 0.5))
    assert np.isclose(val, abs(b), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("b", [0.5])
def test_pdf_value_in_center_small_b(b):
    """
    For |b| < 1, the density in the 'middle' band (near v=0.5)
    should be exactly 1.
    """
    C = XiRhoBoundaryCopula(b=b)
    # v=0.5 is in the middle section for |b| < 1
    val = float(C.pdf(0.5, 0.5))
    assert np.isclose(val, 1.0, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("b", [1.5])
def test_pdf_symmetry_explicit(b):
    """
    Check that pdf_neg(u, v) == pdf_pos(1-u, v).
    """
    Cpos = XiRhoBoundaryCopula(b=b)
    Cneg = XiRhoBoundaryCopula(b=-b)

    # Check a few points
    pts = [(0.2, 0.3), (0.8, 0.8), (0.4, 0.6)]
    for u, v in pts:
        val_pos = float(Cpos.pdf(1 - u, v))
        val_neg = float(Cneg.pdf(u, v))
        assert np.isclose(val_pos, val_neg, rtol=RTOL, atol=ATOL)


def test_pdf_zero_outside_band():
    """
    Check that the PDF is zero outside the diagonal band.
    Using b=2, s_v approx v + 0.25 in middle.
    Band is [s - 0.5, s].
    """
    b = 2.0
    C = XiRhoBoundaryCopula(b=b)

    # at v=0.5, s_v = 0.5 + 1/4 = 0.75.
    # Band is roughly [0.25, 0.75].
    # u=0.1 is outside. u=0.9 is outside.

    assert np.isclose(float(C.pdf(0.1, 0.5)), 0.0, atol=ATOL)
    assert np.isclose(float(C.pdf(0.9, 0.5)), 0.0, atol=ATOL)
    # Inside
    assert float(C.pdf(0.5, 0.5)) > 0
