# tests/test_v_threshold_copula.py
import numpy as np
import pytest

from copul.family.other.v_threshold_copula import VThresholdCopula


# -------------------------------
# Helpers
# -------------------------------


def grid(n=41):
    u = np.linspace(0.0, 1.0, n)
    v = np.linspace(0.0, 1.0, n)
    U, V = np.meshgrid(u, v, indexing="ij")
    return u, v, U, V


def frechet_upper(U, V):
    # M(u,v) = min(u, v)
    return np.minimum(U, V)


def frechet_lower(U, V):
    # W(u,v) = max(u+v-1, 0)
    return np.maximum(U + V - 1.0, 0.0)


def rho_closed_form(mu):
    mu = float(mu)
    if mu <= 1.0:
        return 1.0 - mu**3
    return -(mu**3) + 6.0 * mu**2 - 12.0 * mu + 7.0


def nu_closed_form(mu):
    mu = float(mu)
    if mu <= 1.0:
        return 1.0 - 0.75 * mu**4
    return -0.75 * mu**4 + 4.0 * mu**3 - 6.0 * mu**2 + 3.0


# -------------------------------
# Basic copula axioms on a grid
# -------------------------------


@pytest.mark.parametrize("mu", [0.0, 0.2, 0.5, 0.9, 1.0, 1.2, 1.6, 2.0])
def test_margins_and_bounds(mu):
    cop = VThresholdCopula(mu=mu)
    _, _, U, V = grid(51)
    C = cop.cdf_vectorized(U, V)

    # Margins
    # C(u,0)=0, C(0,v)=0
    assert np.allclose(C[:, 0], 0.0, atol=1e-12)
    assert np.allclose(C[0, :], 0.0, atol=1e-12)

    # C(u,1) = u, C(1,v) = v
    assert np.allclose(C[:, -1], U[:, -1], atol=1e-12)
    assert np.allclose(C[-1, :], V[-1, :], atol=1e-12)

    # Frechet bounds: W <= C <= M
    W = frechet_lower(U, V)
    M = frechet_upper(U, V)
    assert np.all(C >= W - 1e-12)
    assert np.all(C <= M + 1e-12)

    # Monotone nondecreasing in each coordinate (discrete check)
    dU = np.diff(C, axis=0)
    dV = np.diff(C, axis=1)
    assert np.all(dU >= -1e-12)
    assert np.all(dV >= -1e-12)


@pytest.mark.parametrize("mu", [0.0, 0.25, 0.8, 1.0, 1.2, 1.7, 2.0])
def test_two_increasing_volume_nonnegative(mu):
    cop = VThresholdCopula(mu=mu)
    rng = np.random.default_rng(12345)

    # Sample random rectangles and check 2-increasing volume >= 0
    for _ in range(300):
        u1, u2 = np.sort(rng.random(2))
        v1, v2 = np.sort(rng.random(2))

        C11 = cop.cdf_vectorized(u1, v1)
        C12 = cop.cdf_vectorized(u1, v2)
        C21 = cop.cdf_vectorized(u2, v1)
        C22 = cop.cdf_vectorized(u2, v2)

        vol = C22 - C12 - C21 + C11
        assert vol >= -1e-12


# -------------------------------
# Closed-form measures
# -------------------------------


@pytest.mark.parametrize("mu", [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
def test_closed_form_rho_nu(mu):
    cop = VThresholdCopula(mu=mu)
    rho = cop.spearmans_rho()
    nu = cop.blests_nu()

    assert np.isclose(rho, rho_closed_form(mu), atol=1e-12)
    assert np.isclose(nu, nu_closed_form(mu), atol=1e-12)

    # Boundary law for rho >= 0: nu = 1 - 3/4 * (1 - rho)^(4/3)
    if rho >= -1e-14:  # allow tiny numeric
        nu_boundary = 1.0 - 0.75 * (1.0 - rho) ** (4.0 / 3.0)
        assert np.isclose(nu, nu_boundary, atol=5e-12)


@pytest.mark.parametrize("rho_target", [-1.0, -0.75, -0.3, 0.0, 0.2, 0.8, 0.99, 1.0])
def test_from_rho_inverse(rho_target):
    cop = VThresholdCopula.from_rho(rho_target)
    mu = float(cop.mu)
    assert 0.0 <= mu <= 2.0

    rho = cop.spearmans_rho()
    assert np.isclose(rho, rho_target, atol=5e-12)


def _rho_nu_from_cdf_grid(C, u, v):
    """
    Compute rho and nu from a sampled CDF grid via trapezoidal integration.
    """
    U, V = np.meshgrid(u, v, indexing="ij")
    I1 = np.trapz(np.trapz(C, v, axis=1), u, axis=0)  # ∬ C
    I2 = np.trapz(np.trapz((1.0 - U) * C, v, axis=1), u, axis=0)  # ∬ (1-u) C
    rho = 12.0 * I1 - 3.0
    nu = 24.0 * I2 - 2.0
    return rho, nu


@pytest.mark.parametrize("mu", [0.0, 0.3, 0.7, 1.3, 1.7, 2.0])
def test_survival_relation_rho_nu(mu):
    """
    Survival transform Ĉ(u,v) = u + v - 1 + C(1-u, 1-v) satisfies:
      rho(Ĉ) = rho(C)
      nu(Ĉ)  = 2*rho(C) - nu(C)
    """
    cop = VThresholdCopula(mu=mu)
    n = 801
    u = np.linspace(0.0, 1.0, n)
    v = np.linspace(0.0, 1.0, n)
    U, V = np.meshgrid(u, v, indexing="ij")

    C = cop.cdf_vectorized(U, V)
    C_rev = cop.cdf_vectorized(1.0 - U, 1.0 - V)
    C_surv = (
        U + V - 1.0 + C_rev
    )  # no clipping; the expression is already in [0,1] for copulas

    rho, nu = _rho_nu_from_cdf_grid(C, u, v)
    rho_s, nu_s = _rho_nu_from_cdf_grid(C_surv, u, v)

    assert np.isclose(rho_s, rho, atol=5e-5)
    assert np.isclose(nu_s, 2.0 * rho - nu, atol=7e-5)


def test_survival_relation_endpoints():
    for mu, expected in [(0.0, (1.0, 1.0)), (2.0, (-1.0, -1.0))]:
        cop = VThresholdCopula(mu=mu)
        n = 401
        u = np.linspace(0, 1, n)
        v = np.linspace(0, 1, n)
        U, V = np.meshgrid(u, v, indexing="ij")
        C = cop.cdf_vectorized(U, V)
        C_rev = cop.cdf_vectorized(1 - U, 1 - V)
        C_surv = U + V - 1 + C_rev
        rho, nu = _rho_nu_from_cdf_grid(C, u, v)
        rho_s, nu_s = _rho_nu_from_cdf_grid(C_surv, u, v)
        # M and W are fixed by survival, so (rho_s, nu_s) == (rho, nu)
        assert np.isclose(rho_s, rho, atol=1e-5)
        assert np.isclose(nu_s, nu, atol=1e-5)
        assert np.isclose(rho, expected[0], atol=1e-5)
        assert np.isclose(nu, expected[1], atol=1e-5)


@pytest.mark.parametrize("mu", [0.0, 0.3, 0.7, 1.3, 1.7, 2.0])
def test_vertical_reflection_antisymmetry(mu):
    """
    Vertical reflection C^↓(u,v) = u - C(u, 1-v) flips concordance:
    (rho, nu) -> ( -rho, -nu ).
    """
    cop = VThresholdCopula(mu=mu)
    n = 801
    u = np.linspace(0.0, 1.0, n)
    v = np.linspace(0.0, 1.0, n)
    U, V = np.meshgrid(u, v, indexing="ij")

    C = cop.cdf_vectorized(U, V)
    C_flip = np.clip(U - cop.cdf_vectorized(U, 1.0 - V), 0.0, 1.0)

    rho, nu = _rho_nu_from_cdf_grid(C, u, v)
    rho_f, nu_f = _rho_nu_from_cdf_grid(C_flip, u, v)

    # Single-axis reflection should negate rho and nu
    assert np.isclose(rho_f, -rho, atol=5e-5)
    assert np.isclose(nu_f, -nu, atol=5e-5)


# -------------------------------
# Endpoints == Frechet copulas
# -------------------------------


@pytest.mark.parametrize("mu", [0.0, 2.0])
def test_endpoints_equal_frechet(mu):
    cop = VThresholdCopula(mu=mu)
    _, _, U, V = grid(61)
    C = cop.cdf_vectorized(U, V)

    if mu == 0.0:  # M(u,v) = min(u,v)
        M = frechet_upper(U, V)
        assert np.allclose(C, M, atol=1e-12)
    else:  # W(u,v) = max(u+v-1, 0)
        W = frechet_lower(U, V)
        assert np.allclose(C, W, atol=1e-12)


# -------------------------------
# Regime threshold continuity
# -------------------------------


@pytest.mark.parametrize("mu", [0.2, 0.6, 0.9, 1.1, 1.6, 1.8])
def test_continuity_across_v0(mu):
    cop = VThresholdCopula(mu=mu)
    eps = 1e-8
    t_star = 1.0 - mu / 2.0
    # threshold v0 = |2 t* - 1|
    v0 = abs(2.0 * t_star - 1.0)

    # Avoid exactly at the kinks in u to keep this a genuine continuity test
    u_vals = np.linspace(0.03, 0.97, 25)
    for u in u_vals:
        C_minus = cop.cdf_vectorized(u, max(v0 - eps, 0.0))
        C_plus = cop.cdf_vectorized(u, min(v0 + eps, 1.0))
        assert abs(C_plus - C_minus) <= 5e-7  # generous tol due to regime switch


# -------------------------------
# ∂2 C checks on one-tail regimes
# -------------------------------


@pytest.mark.parametrize("mu", [0.2, 0.8, 1.2, 1.8])
def test_partial_v_in_one_tail_regimes(mu):
    cop = VThresholdCopula(mu=mu)
    1.0 - mu / 2.0

    if mu <= 1.0:
        # Left-only regime for v ∈ [0, 1-μ]
        v = 0.5 * (1.0 - mu)
        assert 0.0 <= v <= 1.0
        # Expect ∂_2 C(u,v) = 1{u ≥ v}
        u_test = np.array([v - 0.1, v + 0.1])
        u_test = np.clip(u_test, 0.0, 1.0)
        dC = np.array([cop.cond_distr_2(u=uu, v=v) for uu in u_test], dtype=float)
        expected = (u_test >= v).astype(float)
        assert np.allclose(dC, expected, atol=1e-12)

    else:
        # Right-only regime for v ∈ [0, μ-1]
        v = 0.5 * (mu - 1.0)
        assert 0.0 <= v <= 1.0
        # Expect ∂_2 C(u,v) = 1{u ≥ 1 − v}
        thresh = 1.0 - v
        u_test = np.array([thresh - 0.1, thresh + 0.1])
        u_test = np.clip(u_test, 0.0, 1.0)
        dC = np.array([cop.cond_distr_2(u=uu, v=v) for uu in u_test], dtype=float)
        expected = (u_test >= thresh).astype(float)
        assert np.allclose(dC, expected, atol=1e-12)
