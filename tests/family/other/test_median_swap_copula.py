# tests/test_median_swap_copula.py
import numpy as np
import pytest

from copul.family.other.median_swap_copula import MedianSwapCopula


def frechet_M(u, v):
    return np.minimum(u, v)


def frechet_W(u, v):
    return np.maximum(u + v - 1.0, 0.0)


@pytest.mark.parametrize("delta", [0.0, 0.05, 0.2, 0.35, 0.5])
def test_margins_uniform(delta):
    cop = MedianSwapCopula(delta=delta)
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


@pytest.mark.parametrize("delta", [0.0, 0.05, 0.2, 0.35, 0.5])
def test_two_increasing(delta):
    cop = MedianSwapCopula(delta=delta)
    # coarse grid to keep test fast
    U = np.linspace(0, 1, 41)
    V = np.linspace(0, 1, 41)
    C = cop.cdf_vectorized(U[:, None], V[None, :])

    # Rectangle volumes must be >= 0
    vols = C[1:, 1:] - C[1:, :-1] - C[:-1, 1:] + C[:-1, :-1]
    assert np.all(vols >= -1e-12)  # small numerical tolerance


@pytest.mark.parametrize("delta", [0.0, 0.05, 0.2, 0.35, 0.5])
def test_closed_forms_beta_nu(delta):
    cop = MedianSwapCopula(delta=delta)
    beta_expected = 1.0 - 4.0 * delta
    nu_expected = 1.0 - 6.0 * delta * delta - 8.0 * delta**4
    assert np.isclose(cop.blomqvists_beta(), beta_expected, atol=1e-12)
    assert np.isclose(cop.blests_nu(), nu_expected, atol=1e-12)


@pytest.mark.parametrize("beta", [-1.0, -0.4, 0.0, 0.7, 1.0])
def test_from_beta_roundtrip(beta):
    cop = MedianSwapCopula.from_beta(beta)
    # delta should be (1 - beta)/4, clamped
    delta_expected = (1.0 - beta) / 4.0
    delta_expected = max(0.0, min(0.5, delta_expected))
    assert np.isclose(float(cop.delta), delta_expected, atol=1e-12)

    # and the copula should report the same beta (within tolerance)
    assert np.isclose(cop.blomqvists_beta(), beta, atol=1e-12)


@pytest.mark.parametrize("delta", [0.0, 0.1, 0.25, 0.5])
def test_cdf_monotone_in_arguments(delta):
    """
    C is non-decreasing in each argument; discrete check along grid lines.
    """
    cop = MedianSwapCopula(delta=delta)
    grid = np.linspace(0, 1, 51)

    # monotone in u for fixed v
    for v in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
        C = cop.cdf_vectorized(grid, v)
        assert np.all(np.diff(C) >= -1e-12)

    # monotone in v for fixed u
    for u in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
        C = cop.cdf_vectorized(u, grid)
        assert np.all(np.diff(C) >= -1e-12)
