import numpy as np
import pytest

from copul.family.other.diagonal_strip_copula import (
    XiPsiApproxLowerBoundaryCopula,
    DiagonalStripCopula,
)
from copul.family.frechet.biv_independence_copula import BivIndependenceCopula


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
    # integrate over v then u (consistent with class' usage)
    from numpy import trapz

    Hv = trapz(Z, x, axis=0)  # ∫ c dv (over rows)
    II = trapz(Hv, x, axis=0)  # ∫∫ c dv du
    assert np.isclose(II, 1.0, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("alpha,beta", [(0.25, 0.40), (0.30, 0.60)])
def test_v_marginal_is_uniform(alpha, beta):
    """
    For almost every v, ∫_0^1 c(u,v) du ≈ 1.
    When β ≥ 0.5 the diagonal strip can cover the entire u-axis for a
    measure-zero set of heights, creating a single v where the row integral
    is ~0 due to F_T plateaus. We allow a tiny number of such outliers.
    """
    C = XiPsiApproxLowerBoundaryCopula(alpha=alpha, beta=beta)

    Vg, Ug, x = grid(n=401)
    Z = C.pdf_vectorized(Ug, Vg)

    from numpy import trapz

    row_int = trapz(Z, x, axis=1)  # ∫_0^1 c(u,v) du for each v

    # 1) Globally the v-marginal must integrate to 1
    area_over_v = trapz(row_int, x)  # ∫_0^1 [∫ c du] dv = 1
    assert np.isclose(area_over_v, 1.0, rtol=3e-3, atol=3e-3)

    # 2) Pointwise uniformity holds except possibly at plateau v (measure zero).
    ok = np.isclose(row_int, 1.0, rtol=5e-3, atol=5e-3)
    # Allow up to ~3 outliers on a 401-point grid
    assert ok.sum() >= len(ok) - 3, f"Too many non-uniform rows: {(~ok).sum()} outliers"


@pytest.mark.parametrize("alpha,beta", [(0.30, 0.50)])
def test_u_marginal_is_not_uniform(alpha, beta):
    """
    The construction preserves only the V-marginal, so f_U(u) is generally not flat.
    Check that its variation is clearly nonzero (but still integrates to 1).
    """
    C = XiPsiApproxLowerBoundaryCopula(alpha=alpha, beta=beta)

    Vg, Ug, x = grid(n=401)
    Z = C.pdf_vectorized(Ug, Vg)

    from numpy import trapz

    # ∫_0^1 c(u,v) dv for each u
    col_int = trapz(Z, x, axis=0)  # shape (Nu,)

    # Some noticeable variation away from a flat line
    assert np.std(col_int) > 3e-3

    # But the total mass over u is still 1
    assert np.isclose(trapz(col_int, x), 1.0, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("alpha,beta", [(0.20, 0.30), (0.30, 0.50)])
def test_cdf_basic_properties(alpha, beta):
    C = XiPsiApproxLowerBoundaryCopula(alpha=alpha, beta=beta)

    # A modest grid for CDF
    u = np.linspace(0.0, 1.0, 121)
    v = np.linspace(0.0, 1.0, 121)
    Vg, Ug = np.meshgrid(v, u, indexing="ij")

    Cv = C.cdf_vectorized(Ug, Vg, grid_n=300)

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
    # differences along axis-1 (u) and axis-0 (v) should be ≥ ~0 up to tiny noise
    du = np.diff(Cv, axis=1)
    dv = np.diff(Cv, axis=0)
    assert np.min(du) >= -5e-4
    assert np.min(dv) >= -5e-4


@pytest.mark.parametrize("alpha,beta", [(0.25, 0.40)])
def test_conditional_distributions(alpha, beta):
    C = XiPsiApproxLowerBoundaryCopula(alpha=alpha, beta=beta)

    # 1) h(u|v) increases in u, with h(0|v)=0 and h(1|v)=1
    u = np.linspace(0.0, 1.0, 301)
    v = np.linspace(0.1, 0.9, 9)  # avoid extreme endpoints
    Vg, Ug = np.meshgrid(v, u, indexing="ij")
    H = C.cond_distr_1_vectorized(Ug, Vg)  # shape (Nv, Nu)

    assert np.all(H[:, 0] == pytest.approx(0.0, abs=3e-3))
    assert np.all(H[:, -1] == pytest.approx(1.0, abs=3e-3))
    assert np.min(np.diff(H, axis=1)) >= -5e-4  # monotone in u

    # 2) C(v|u) increases in v, with C(0|u)=0 and C(1|u)=1
    v2 = np.linspace(0.0, 1.0, 301)
    u2 = np.linspace(0.1, 0.9, 7)
    Vg2, Ug2 = np.meshgrid(v2, u2, indexing="ij")
    Cvu = C.cond_distr_2_vectorized(Ug2, Vg2)

    assert np.all(Cvu[0, :] == pytest.approx(0.0, abs=3e-3))
    assert np.all(Cvu[-1, :] == pytest.approx(1.0, abs=3e-3))
    assert np.min(np.diff(Cvu, axis=0)) >= -5e-4  # monotone in v


def test_is_symmetric_flag_and_alias():
    C = XiPsiApproxLowerBoundaryCopula(alpha=0.3, beta=0.5)
    assert C.is_symmetric is False

    # Type alias should refer to the same class
    assert DiagonalStripCopula is XiPsiApproxLowerBoundaryCopula


def test_special_case_alpha_zero_returns_independence():
    inst = XiPsiApproxLowerBoundaryCopula(alpha=0.0, beta=0.5)
    assert isinstance(inst, BivIndependenceCopula)
