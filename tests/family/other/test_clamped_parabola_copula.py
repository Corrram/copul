# tests/test_clamped_parabola_copula.py
import numpy as np
import pytest

# The class under test
from copul.family.other.clamped_parabola_copula import ClampedParabolaCopula


@pytest.mark.parametrize("b", [0.1, 0.5, 1.0, 2.5])
def test_copula_boundaries(b):
    cp = ClampedParabolaCopula(b=b)

    # sample grid
    u = np.linspace(0.0, 1.0, 11)
    v = np.linspace(0.0, 1.0, 11)

    # C(0,v) = 0 and C(u,0) = 0
    assert np.allclose(cp.cdf_vectorized(0.0, v), 0.0, atol=1e-10)
    assert np.allclose(cp.cdf_vectorized(u, 0.0), 0.0, atol=1e-10)

    # C(1,v) = v and C(u,1) = u
    assert np.allclose(cp.cdf_vectorized(1.0, v), v, atol=5e-7)
    assert np.allclose(cp.cdf_vectorized(u, 1.0), u, atol=5e-7)


@pytest.mark.parametrize("b", [0.2, 0.7, 1.5])
def test_two_increasing(b, rng_seed=1234):
    """Random rectangles must have nonnegative C-measure."""
    rng = np.random.default_rng(rng_seed)
    cp = ClampedParabolaCopula(b=b)

    for _ in range(200):
        u1, u2 = np.sort(rng.uniform(0, 1, size=2))
        v1, v2 = np.sort(rng.uniform(0, 1, size=2))
        C11 = cp.cdf_vectorized(u1, v1)
        C12 = cp.cdf_vectorized(u1, v2)
        C21 = cp.cdf_vectorized(u2, v1)
        C22 = cp.cdf_vectorized(u2, v2)
        rect_mass = C22 - C21 - C12 + C11
        # allow tiny negative numerical noise
        assert rect_mass >= -1e-9


@pytest.mark.parametrize("b_small, b_large", [(0.2, 3.0), (0.4, 5.0)])
def test_xi_and_nu_monotone_in_b(b_small, b_large):
    cp_small = ClampedParabolaCopula(b=b_small)
    cp_large = ClampedParabolaCopula(b=b_large)

    xi_small = cp_small.chatterjees_xi()
    xi_large = cp_large.chatterjees_xi()
    nu_small = cp_small.blests_nu()
    nu_large = cp_large.blests_nu()

    # As b increases (μ decreases), both xi and nu should increase (towards M).
    assert xi_small < xi_large
    assert nu_small < nu_large

    # sanity ranges
    assert 0.0 <= xi_small <= 1.0 and 0.0 <= xi_large <= 1.0
    assert 0.0 <= nu_small <= 1.0 and 0.0 <= nu_large <= 1.0


@pytest.mark.parametrize("b", [0.3, 1.0, 2.0])
def test_h_matches_du_of_C(b):
    """Check that the analytical h_v(t) matches finite-difference ∂_u C."""
    cp = ClampedParabolaCopula(b=b)
    v = 0.37  # any interior v
    q = cp._get_q_v(v, float(b))

    # expected h(u,v) from the definition (with b = 1/μ here as the model parameter)
    def h_expected(u):
        return np.clip(b * ((1.0 - u) ** 2 - q), 0.0, 1.0)

    # finite-difference du of C
    u_grid = np.linspace(0.001, 0.999, 200)
    eps = 1e-5
    C_plus = cp.cdf_vectorized(u_grid + eps, v)
    C_minus = cp.cdf_vectorized(u_grid - eps, v)
    h_fd = (C_plus - C_minus) / (2 * eps)

    err = np.mean(np.abs(h_fd - h_expected(u_grid)))
    assert err < 2e-3  # fairly tight tolerance given nested numerics


@pytest.mark.parametrize("x_target", [0.15, 0.35, 0.6, 0.85])
def test_from_xi_hits_target(x_target):
    cp = ClampedParabolaCopula.from_xi(x_target)
    x_val = cp.chatterjees_xi()
    assert abs(x_val - x_target) < 3e-3  # root-finding + quadrature tolerances


def test_limit_independence():
    """Small b should approach independence: xi≈0, nu≈0, C(1,v)=v."""
    cp = ClampedParabolaCopula(b=1e-2)  # b -> 0+ ⇒ Π
    xi = cp.chatterjees_xi()
    nu = cp.blests_nu()
    assert xi < 3e-2
    assert nu < 3e-2

    v = np.linspace(0, 1, 6)
    assert np.allclose(cp.cdf_vectorized(1.0, v), v, atol=1e-2)


@pytest.mark.parametrize("b", [0.2, 0.8, 2.0])
def test_basic_ranges(b):
    cp = ClampedParabolaCopula(b=b)
    xi = cp.chatterjees_xi()
    nu = cp.blests_nu()
    assert 0.0 <= xi <= 1.0
    assert 0.0 <= nu <= 1.0
    xi_ccop = cp.to_checkerboard(100).chatterjees_xi()
    nu_ccop = cp.to_checkerboard(100).blests_nu()
    # checkerboard should be very close to original for smooth copula
    assert abs(xi - xi_ccop) < 1e-3
    assert abs(nu - nu_ccop) < 1e-3


@pytest.mark.parametrize("b", [0.4, 1.2])
def test_q_cache_consistency(b):
    """_get_q_v should be stable and cache-consistent."""
    cp = ClampedParabolaCopula(b=b)
    v_vals = [0.2, 0.5, 0.8]
    q_first = [cp._get_q_v(v, float(b)) for v in v_vals]
    q_second = [cp._get_q_v(v, float(b)) for v in v_vals]  # served from cache
    assert np.allclose(q_first, q_second, rtol=0, atol=0)


@pytest.mark.parametrize("b", [0.5, 1.5])
def test_pdf_nonnegative(b):
    """pdf_vectorized (∂_v h) should be nonnegative almost everywhere."""
    cp = ClampedParabolaCopula(b=b)
    u = np.linspace(0.05, 0.95, 25)
    v = np.linspace(0.05, 0.95, 11)
    U, V = np.meshgrid(u, v)
    pdf_vals = cp.pdf_vectorized(U, V)
    # allow tiny numerical negatives from finite differencing
    assert np.min(pdf_vals) > -5e-4
