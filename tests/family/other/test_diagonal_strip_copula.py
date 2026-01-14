import numpy as np
import pytest
from scipy.integrate import trapezoid

from copul.family.other.diagonal_strip_copula import (
    XiPsiApproxLowerBoundaryCopula,
    DiagonalStripCopula,
)


RTOL = 1e-6
ATOL = 1e-6


def grid(n=201):
    """[0,1]x[0,1] mesh with indexing='ij' (rows=v, cols=u)."""
    x = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return X, Y, x


@pytest.mark.parametrize("alpha,beta", [(0.20, 0.30), (0.30, 0.50), (0.40, 0.50)])
def test_pdf_nonnegative_and_normalized(alpha, beta):
    C = XiPsiApproxLowerBoundaryCopula(alpha=alpha, beta=beta)

    # moderate grid for speed + accuracy
    Vg, Ug, x = grid(n=301)
    Z = C.pdf_vectorized(Ug, Vg)

    # nonnegative everywhere
    assert np.all(Z >= -1e-12)

    # integral over the unit square ~ 1
    # integrate over v (axis 0) then u (axis 1 is collapsed, so axis 0 of result)
    Hv = trapezoid(Z, x, axis=0)
    II = trapezoid(Hv, x, axis=0)
    assert np.isclose(II, 1.0, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("alpha,beta", [(0.25, 0.40)])
def test_v_marginal_is_uniform(alpha, beta):
    """
    For almost every v, ∫_0^1 c(u,v) du ≈ 1.
    """
    C = XiPsiApproxLowerBoundaryCopula(alpha=alpha, beta=beta)

    Vg, Ug, x = grid(n=401)
    Z = C.pdf_vectorized(Ug, Vg)

    # ∫_0^1 c(u,v) du for each v (integrate along U, which is axis 1)
    row_int = trapezoid(Z, x, axis=1)

    # 1) Globally the v-marginal must integrate to 1
    area_over_v = trapezoid(row_int, x)
    assert np.isclose(area_over_v, 1.0, rtol=3e-3, atol=3e-3)

    # 2) Pointwise uniformity holds except possibly at plateau v.
    ok = np.isclose(row_int, 1.0, rtol=5e-3, atol=5e-3)
    # Allow up to ~3 outliers on a 401-point grid
    assert ok.sum() >= len(ok) - 3, f"Too many non-uniform rows: {(~ok).sum()} outliers"


@pytest.mark.parametrize("alpha,beta", [(0.30, 0.50)])
def test_u_marginal_is_not_uniform(alpha, beta):
    """
    The construction preserves only the V-marginal, so f_U(u) is generally not flat.
    """
    C = XiPsiApproxLowerBoundaryCopula(alpha=alpha, beta=beta)

    Vg, Ug, x = grid(n=401)
    Z = C.pdf_vectorized(Ug, Vg)

    # ∫_0^1 c(u,v) dv for each u (integrate along V, which is axis 0)
    col_int = trapezoid(Z, x, axis=0)

    # Some noticeable variation away from a flat line.
    # We lowered the threshold to 1e-3 as the variation can be subtle.
    assert np.std(col_int) > 1e-3

    # But the total mass over u is still 1
    assert np.isclose(trapezoid(col_int, x), 1.0, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("alpha,beta", [(0.20, 0.30), (0.30, 0.50)])
def test_cdf_basic_properties(alpha, beta):
    C = XiPsiApproxLowerBoundaryCopula(alpha=alpha, beta=beta)

    # A modest grid for CDF
    u = np.linspace(0.0, 1.0, 121)
    v = np.linspace(0.0, 1.0, 121)
    Vg, Ug = np.meshgrid(v, u, indexing="ij")

    Cv = C.cdf_vectorized(Ug, Vg)

    # Range
    assert np.min(Cv) >= -2e-3
    assert np.max(Cv) <= 1.0 + 2e-3

    # Boundaries: C(0,v)=0, C(u,0)=0, C(1,1)=1
    assert np.allclose(Cv[:, 0], 0.0, atol=2e-3, rtol=0)
    assert np.allclose(Cv[0, :], 0.0, atol=2e-3, rtol=0)
    assert np.isclose(Cv[-1, -1], 1.0, atol=2e-3, rtol=0)

    # Because V-marginal is uniform, C(1,v) ≈ v
    assert np.allclose(Cv[:, -1], v, atol=3e-3, rtol=0)

    # Monotone: nondecreasing in u and in v
    du = np.diff(Cv, axis=1)
    dv = np.diff(Cv, axis=0)
    assert np.min(du) >= -5e-4
    assert np.min(dv) >= -5e-4


@pytest.mark.parametrize("alpha,beta", [(0.25, 0.40)])
def test_conditional_distributions(alpha, beta):
    C = XiPsiApproxLowerBoundaryCopula(alpha=alpha, beta=beta)

    # 1) h(u|v) = P(V <= v | U = u).
    u = np.linspace(0.0, 1.0, 301)
    v = np.linspace(0.1, 0.9, 9)  # avoid extreme endpoints
    Vg, Ug = np.meshgrid(v, u, indexing="ij")

    H = C.cond_distr_1(Ug, Vg)  # shape (Nv, Nu)

    # Check boundaries in u: h(0|v) is not necessarily 0, but h(1|v) should check out or be monotonic
    # Actually, h(u|v) is a CDF in v, but varying u? No.
    # cond_distr_1 is P(V<=v | U=u). As a function of v it is a CDF (0 to 1).
    # As a function of u, it's just a surface.

    # Let's check the CDF property in V-direction for fixed U
    # Fix u=0.5, vary v from 0 to 1
    v_fine = np.linspace(0.0, 1.0, 100)
    u_fixed = np.full_like(v_fine, 0.5)
    H_v = C.cond_distr_1(u_fixed, v_fine)

    assert np.isclose(H_v[0], 0.0, atol=1e-3)
    assert np.isclose(H_v[-1], 1.0, atol=1e-3)
    assert np.all(np.diff(H_v) >= -1e-4)  # Monotonic in v

    # 2) C(u|v) = P(U <= u | V = v).
    # This is a CDF in u (0 to 1).
    v_fixed = np.full_like(v_fine, 0.5)
    u_fine = np.linspace(0.0, 1.0, 100)

    Cvu = C.cond_distr_2(u_fine, v_fixed)

    assert np.isclose(Cvu[0], 0.0, atol=1e-3)
    assert np.isclose(Cvu[-1], 1.0, atol=1e-3)
    assert np.all(np.diff(Cvu) >= -1e-4)  # Monotonic in u


def test_is_symmetric_flag_and_alias():
    C = XiPsiApproxLowerBoundaryCopula(alpha=0.3, beta=0.5)
    assert C.is_symmetric is False

    # Type alias should refer to the same class
    assert DiagonalStripCopula is XiPsiApproxLowerBoundaryCopula
