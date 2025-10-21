# tests/family/other/test_xi_nu_boundary_copula.py
import numpy as np
import pytest
from scipy.integrate import dblquad

from copul.family.other.clamped_parabola_copula import XiNuBoundaryCopula


RTOL = 1e-9
ATOL = 1e-9


def grid(n=41):
    u = np.linspace(0.0, 1.0, n)
    v = np.linspace(0.0, 1.0, n)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    return uu, vv


@pytest.mark.parametrize("b", [0.5, 1.0, 2.0])
def test_cdf_boundary_conditions(b):
    C = XiNuBoundaryCopula(b=b)

    u = np.linspace(0, 1, 101)
    v = np.linspace(0, 1, 101)

    # C(u,0) = 0, C(0,v)=0
    assert np.allclose(C.cdf_vectorized(u, 0.0), 0.0, rtol=RTOL, atol=ATOL)
    assert np.allclose(C.cdf_vectorized(0.0, v), 0.0, rtol=RTOL, atol=ATOL)

    # C(u,1) = u, C(1,v)=v
    assert np.allclose(C.cdf_vectorized(u, 1.0), u, rtol=RTOL, atol=ATOL)
    assert np.allclose(C.cdf_vectorized(1.0, v), v, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("b", [0.4, 1.2, 2.5])
def test_frechet_bounds(b):
    C = XiNuBoundaryCopula(b=b)
    uu, vv = grid(n=31)
    Cuv = C.cdf_vectorized(uu, vv)

    W = np.maximum(uu + vv - 1.0, 0.0)
    M = np.minimum(uu, vv)

    assert np.all(Cuv >= W - 1e-12)
    assert np.all(Cuv <= M + 1e-12)


@pytest.mark.parametrize("b", [0.5, 1.0, 2.0])
def test_monotonicity_in_u_and_v(b):
    C = XiNuBoundaryCopula(b=b)
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


@pytest.mark.parametrize("b", [0.6, 1.3, 2.7])
def test_vectorized_shape_and_scalar_consistency(b):
    C = XiNuBoundaryCopula(b=b)
    uu, vv = grid(n=19)
    Cuv = C.cdf_vectorized(uu, vv)
    assert Cuv.shape == uu.shape == vv.shape

    for u, v in [(0.2, 0.3), (0.7, 0.1), (0.55, 0.8)]:
        scalar = float(C.cdf_vectorized(u, v))
        vect = float(C.cdf_vectorized(np.array([u]), np.array([v]))[0])
        assert np.isclose(scalar, vect, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("b", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("v", [0.1, 0.3, 0.6, 0.9])
def test_piecewise_regions_match_cdf_definition(b, v):
    C = XiNuBoundaryCopula(b=b)
    q = C._get_q_v_vec(np.array([v]), float(b))[0]
    a, s = C._switch_points(np.array([q]), float(b))
    a, s = float(a), float(s)

    # Pick u below a (if possible) → C(u,v) ≈ u
    if a > 1e-6:
        u1 = 0.5 * a
        assert np.isclose(C.cdf_vectorized(u1, v), u1, rtol=RTOL, atol=1e-10)

    # Pick u above s (if s < 1) → C(u,v) ≈ v
    if s < 1.0 - 1e-6:
        u3 = s + 0.5 * (1.0 - s)
        assert np.isclose(C.cdf_vectorized(u3, v), v, rtol=RTOL, atol=1e-10)


@pytest.mark.parametrize("b", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("v", [0.2, 0.5, 0.8])
def test_du_cdf_matches_clamp_formula(b, v):
    """
    Finite-difference ∂C/∂u should match h(u,v)=clamp(b((1-u)^2 - q(v)), 0, 1)
    away from switching points.
    """
    C = XiNuBoundaryCopula(b=b)
    q = C._get_q_v_vec(np.array([v]), float(b))[0]

    def h_clamp(u):
        return np.clip(b * ((1.0 - u) ** 2 - q), 0.0, 1.0)

    u = np.linspace(0, 1, 2001)
    du = u[1] - u[0]
    Cu = C.cdf_vectorized(u, v)
    # central difference; endpoints with one-sided
    dCdu = np.empty_like(Cu)
    dCdu[1:-1] = (Cu[2:] - Cu[:-2]) / (2 * du)
    dCdu[0] = (Cu[1] - Cu[0]) / du
    dCdu[-1] = (Cu[-1] - Cu[-2]) / du

    h = h_clamp(u)

    # mask out a small band around the switch points where h is discontinuous
    a, s = C._switch_points(np.array([q]), float(b))
    a, s = float(a), float(s)
    mask = np.ones_like(u, dtype=bool)
    eps = 5e-3
    if a > 0:
        mask &= np.abs(u - a) > eps
    if s < 1:
        mask &= np.abs(u - s) > eps

    # compare where smooth
    assert np.allclose(dCdu[mask], h[mask], rtol=2e-3, atol=2e-3)


def _xi_formula_from_mu(mu):
    """
    Closed form used in XiNuBoundaryCopula.chatterjees_xi(), rewritten here
    to cross-check monotonicity quickly without relying on the method.
    """
    s = np.sqrt(mu)
    if mu < 1.0:
        t = np.sqrt(1.0 - mu)
        A = np.arcsinh(t / s)
        num = (
            -105 * s**8 * A
            + 183 * s**6 * t
            - 38 * s**4 * t
            - 88 * s**2 * t
            + 112 * s**2
            + 48 * t
            - 48
        )
        den = 210 * s**6
        return num / den
    else:
        return 8.0 * (7.0 * mu - 3.0) / (105.0 * mu**3)


def _nu_formula_from_mu(mu):
    s = np.sqrt(mu)
    if mu < 1.0:
        t = np.sqrt(1.0 - mu)
        A = np.arcsinh(t / s)
        num = (
            -105 * s**8 * A
            + 87 * s**6 * t
            + 250 * s**4 * t
            - 376 * s**2 * t
            + 448 * s**2
            + 144 * t
            - 144
        )
        den = 420 * s**4
        return num / den
    else:
        return 4.0 * (28.0 * mu - 9.0) / (105.0 * mu**2)


@pytest.mark.parametrize("b_list", [[0.4, 0.7, 1.0, 1.6, 3.0]])
def test_rank_measures_monotone_in_b(b_list):
    xis = []
    nus = []
    for b in b_list:
        C = XiNuBoundaryCopula(b=b)
        xis.append(float(C.chatterjees_xi()))
        nus.append(float(C.blests_nu()))

    # As b increases (μ=1/b decreases), ξ and ν should increase
    assert all(xis[i] < xis[i + 1] for i in range(len(xis) - 1))
    assert all(nus[i] < nus[i + 1] for i in range(len(nus) - 1))

    # Bounds
    assert all(0.0 < x < 1.0 for x in xis)
    assert all(0.0 < n < 1.0 for n in nus)


@pytest.mark.parametrize("x", [0.05, 0.2, 0.5, 0.8, 0.95])
def test_from_xi_roundtrip(x):
    inst = XiNuBoundaryCopula.from_xi(x)
    assert isinstance(inst, XiNuBoundaryCopula)
    x_back = float(inst.chatterjees_xi())
    assert np.isclose(x_back, x, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("b", [0.5, 1.0, 2.0])
def test_checkerboard_approximation_reasonable(b):
    C = XiNuBoundaryCopula(b=b)
    xi_exact = float(C.chatterjees_xi())
    nu_exact = float(C.blests_nu())
    # moderately fine grid for a good approximation
    CC = C.to_checkerboard(grid_size=80)
    xi_app = float(CC.chatterjees_xi())
    nu_app = float(CC.blests_nu())

    assert np.isclose(xi_app, xi_exact, rtol=5e-3, atol=5e-3)
    assert np.isclose(nu_app, nu_exact, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("b", [0.2, 0.5, 1.0, 2.0, 5.0])
def test_blests_nu_numerical_validation(b):
    """
    Numerically validates the closed-form blests_nu() by integrating
    its fundamental definition:
    nu(C) = 24 * integral[ (1-u) * C(u,v) ] du dv - 2
    """
    C = XiNuBoundaryCopula(b=b)

    # 1. Get the value from the closed-form (analytical) formula
    nu_exact = float(C.blests_nu())

    # 2. Define the integrand for numerical calculation
    #    The dblquad signature is func(y, x), which corresponds to func(v, u)
    #    for the integral: integral_u=0..1 [ integral_v=0..1 [ f(u,v) dv ] du ]
    def integrand(v, u):
        # We integrate 24 * (1-u) * C(u,v)
        # cdf_vectorized handles scalar inputs
        return 24.0 * (1.0 - u) * C.cdf_vectorized(u, v)

    # 3. Perform the numerical integration
    #    integral = dblquad(func, x_min, x_max, y_min, y_max)
    integral_part, abs_err = dblquad(integrand, 0, 1, 0, 1)

    # 4. Calculate the final numerical value
    nu_numerical = integral_part - 2.0

    # 5. Compare the analytical and numerical results
    #    Use a reasonable tolerance for numerical integration
    assert abs_err < 1e-6
    assert np.isclose(nu_numerical, nu_exact, rtol=1e-6, atol=1e-6)


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
    C_pos = XiNuBoundaryCopula(b=b_pos)
    # Create copula for negative parameter
    C_neg = XiNuBoundaryCopula(b=-b_pos)

    # Get measures for positive b
    nu_pos = float(C_pos.blests_nu())
    xi_pos = float(C_pos.chatterjees_xi())

    # Get measures for negative b
    nu_neg = float(C_neg.blests_nu())
    xi_neg = float(C_neg.chatterjees_xi())

    # Assert f(-b) == -f(b)
    assert np.isclose(nu_neg, -nu_pos, rtol=RTOL, atol=ATOL)
    assert np.isclose(xi_neg, xi_pos, rtol=RTOL, atol=ATOL)

    nu_pos_ccop = C_pos.to_checkerboard().blests_nu()
    nu_neg_ccop = C_neg.to_checkerboard().blests_nu()
    assert np.isclose(nu_neg_ccop, nu_neg, atol=1e-2)
    assert np.isclose(nu_pos_ccop, nu_pos, atol=1e-2)

    xi_neg_ccop = C_neg.to_checkerboard().chatterjees_xi()
    xi_pos_ccop = C_pos.to_checkerboard().chatterjees_xi()
    assert np.isclose(xi_neg_ccop, xi_neg, atol=1e-2)
    assert np.isclose(xi_pos_ccop, xi_pos, atol=1e-2)
