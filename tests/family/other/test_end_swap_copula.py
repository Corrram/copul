# tests/test_end_swap_copula.py
import numpy as np
import pytest

from copul.family.other.end_swap_copula import EndSwapCopula


def frechet_M(u, v):
    return np.minimum(u, v)


def frechet_W(u, v):
    return np.maximum(u + v - 1.0, 0.0)


@pytest.mark.parametrize("d", [0.0, 0.05, 0.2, 0.35, 0.5])
def test_margins_uniform(d):
    cop = EndSwapCopula(d=d)
    v = np.linspace(0, 1, 201)
    u = np.linspace(0, 1, 201)

    # C(1,v) = v
    Cv = cop.cdf_vectorized(1.0, v)
    assert np.allclose(Cv, v, atol=1e-12)

    # C(u,1) = u
    Cu = cop.cdf_vectorized(u, 1.0)
    assert np.allclose(Cu, u, atol=1e-12)

    # C(0,v) = C(u,0) = 0
    assert np.allclose(cop.cdf_vectorized(0.0, v), 0.0, atol=1e-12)
    assert np.allclose(cop.cdf_vectorized(u, 0.0), 0.0, atol=1e-12)


@pytest.mark.parametrize("d", [0.0, 0.05, 0.2, 0.35, 0.5])
def test_two_increasing(d):
    cop = EndSwapCopula(d=d)
    # coarse grid to keep test fast
    U = np.linspace(0, 1, 41)
    V = np.linspace(0, 1, 41)
    C = cop.cdf_vectorized(U[:, None], V[None, :])

    # Rectangle volumes must be >= 0
    vols = C[1:, 1:] - C[1:, :-1] - C[:-1, 1:] + C[:-1, :-1]
    assert np.all(vols >= -1e-12)  # small numerical tolerance


@pytest.mark.parametrize("d", [0.0, 0.05, 0.2, 0.35, 0.5])
def test_closed_forms_psi_nu(d):
    cop = EndSwapCopula(d=d)
    psi_expected = 1.0 - 6.0 * (d - d * d)
    nu_expected = 1.0 - 12.0 * d + 24.0 * d * d - 16.0 * d**3
    assert np.isclose(cop.spearmans_footrule(), psi_expected, atol=1e-12)
    assert np.isclose(cop.blests_nu(), nu_expected, atol=1e-12)


@pytest.mark.parametrize("psi", [-0.5, -0.2, 0.0, 0.7, 1.0])
def test_from_psi_roundtrip(psi):
    cop = EndSwapCopula.from_psi(psi)

    # d should be (1 - sqrt((1 + 2 ψ)/3))/2, clamped to [0, 1/2]
    val = (1.0 + 2.0 * psi) / 3.0
    val = max(0.0, min(1.0, val))
    d_expected = (1.0 - np.sqrt(val)) / 2.0
    d_expected = max(0.0, min(0.5, d_expected))
    assert np.isclose(float(cop.d), d_expected, atol=1e-12)

    # and the copula should report the same ψ (within tolerance)
    assert np.isclose(cop.spearmans_footrule(), psi, atol=1e-12)


@pytest.mark.parametrize("d", [0.0, 0.1, 0.25, 0.5])
def test_cdf_monotone_in_arguments(d):
    """
    C is non-decreasing in each argument; discrete check along grid lines.
    """
    cop = EndSwapCopula(d=d)
    grid = np.linspace(0, 1, 51)

    # monotone in u for fixed v
    for v in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
        C = cop.cdf_vectorized(grid, v)
        assert np.all(np.diff(C) >= -1e-12)

    # monotone in v for fixed u
    for u in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
        C = cop.cdf_vectorized(u, grid)
        assert np.all(np.diff(C) >= -1e-12)


def test_special_cases_frechet():
    """
    d=0 -> M, d=1/2 -> W
    """
    U = np.linspace(0, 1, 41)
    V = np.linspace(0, 1, 41)
    Ug, Vg = np.meshgrid(U, V, indexing="ij")

    cop_M = EndSwapCopula(d=0.0)
    C_M = cop_M.cdf_vectorized(Ug, Vg)
    assert np.allclose(C_M, frechet_M(Ug, Vg), atol=1e-12)

    cop_W = EndSwapCopula(d=0.5)
    C_W = cop_W.cdf_vectorized(Ug, Vg)
    assert np.allclose(C_W, frechet_W(Ug, Vg), atol=1e-12)
