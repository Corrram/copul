"""Tests for copul.theory.quasi (quasi-copulas, 2-increasing defect, lattice operations)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import copul as cp
from copul.theory.bounds import ShuffleOfM
from copul.theory.quasi import (
    FunctionCopula,
    NumericQuasiCopula,
    check_quasi_copula,
    copula_max,
    copula_min,
    is_copula,
    is_quasi_copula,
    quasi_copula_volume,
    two_increasing_defect,
)
from copul.theory.symmetry import maximally_nonexchangeable_copula

THIRD = 1.0 / 3.0
CENTER = (THIRD, 2 * THIRD, THIRD, 2 * THIRD)


@pytest.fixture(scope="module")
def checkerboards():
    pis = cp.BivCheckPi.generate_diverse(n_samples=12, grid_size=(2, 9), rng=7)
    return pis + [cp.BivCheckMin(C.matr) for C in pis[:6]] + [cp.BivCheckW(C.matr) for C in pis[6:]]


# ---------------------------------------------------------------------------
# copulas are quasi-copulas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "C",
    [
        cp.Clayton(2),
        cp.Frank(-4),
        cp.GumbelHougaard(1.7),
        cp.LowerFrechet(),
        cp.UpperFrechet(),
        cp.BivIndependenceCopula(),
        maximally_nonexchangeable_copula(),
    ],
    ids=repr,
)
def test_copulas_are_quasi_copulas(C):
    chk = check_quasi_copula(C, m=60)
    assert chk.is_quasi_copula and chk.frechet_ok
    assert is_copula(C, m=60)
    assert two_increasing_defect(C, m=60).is_two_increasing(1e-10)


def test_checkerboards_are_copulas(checkerboards):
    for C in checkerboards:
        assert is_quasi_copula(C, m=36)
        d = two_increasing_defect(C, m=36)
        assert d.volume >= -1e-12


# ---------------------------------------------------------------------------
# pointwise max / min: quasi-copulas that are not copulas
# ---------------------------------------------------------------------------


def test_max_of_shuffle_and_its_transpose_is_a_proper_quasi_copula():
    """Q = max(C*, C*^T) has V_Q([1/3, 2/3]^2) = -1/3 (the minimal possible volume)."""
    C = maximally_nonexchangeable_copula()
    Q = copula_max(C, maximally_nonexchangeable_copula(transpose=True))
    assert isinstance(Q, NumericQuasiCopula)
    chk = Q.check(m=60)
    assert chk.is_quasi_copula and chk.frechet_ok
    d = Q.two_increasing_defect(m=60)
    assert d.volume == pytest.approx(-THIRD, abs=1e-12)
    assert d.rectangle == pytest.approx(CENTER, abs=1e-12)
    assert not Q.is_copula(m=60)
    assert quasi_copula_volume(Q, CENTER) == pytest.approx(-THIRD, abs=1e-12)
    # corner values
    assert Q.cdf(THIRD, THIRD) == pytest.approx(0.0, abs=1e-15)
    assert Q.cdf(2 * THIRD, 2 * THIRD) == pytest.approx(THIRD)
    assert Q.cdf(THIRD, 2 * THIRD) == pytest.approx(THIRD)


def test_min_of_two_shuffles_is_a_proper_quasi_copula():
    A = ShuffleOfM.from_permutation([THIRD] * 3, [0, 2, 1])  # strips 1->1, 2->3, 3->2
    B = ShuffleOfM.from_permutation([THIRD] * 3, [1, 0, 2])
    Q = copula_min(A, B)
    assert Q.is_quasi_copula(m=60)
    d = two_increasing_defect(Q, m=60)
    assert d.volume == pytest.approx(-THIRD, abs=1e-12)
    assert d.rectangle == pytest.approx(CENTER, abs=1e-12)
    assert not is_copula(Q, m=60)


def test_lattice_operations_of_random_copulas_respect_volume_bound(checkerboards):
    """Every quasi-copula has V_Q(R) >= -1/3 (Nelsen et al. 2002)."""
    rng = np.random.default_rng(3)
    for _ in range(8):
        i, j = rng.choice(len(checkerboards), size=2, replace=False)
        for op in (copula_max, copula_min):
            Q = op(checkerboards[i], checkerboards[j])
            assert Q.is_quasi_copula(m=36)
            assert Q.two_increasing_defect(m=36).volume >= -THIRD - 1e-12


def test_max_of_identical_copulas_is_the_copula():
    C = cp.Clayton(2)
    Q = copula_max(C, C)
    g = np.linspace(0, 1, 11)
    U, V = np.meshgrid(g, g)
    assert np.allclose(Q.cdf(U, V), C.cdf(U, V), atol=1e-13)
    D = Q.to_copula(m=30)
    assert isinstance(D, FunctionCopula)
    assert D.spearmans_rho() == pytest.approx(C.spearmans_rho(), abs=1e-6)


def test_to_copula_rejects_proper_quasi_copulas():
    C = maximally_nonexchangeable_copula()
    Q = copula_max(C, lambda u, v: C.cdf_vectorized(v, u))
    with pytest.raises(ValueError, match="not a copula"):
        Q.to_copula(m=30)


def test_lattice_operations_need_arguments():
    with pytest.raises(ValueError):
        copula_max()
    with pytest.raises(ValueError):
        copula_min()


# ---------------------------------------------------------------------------
# axiom violations are detected
# ---------------------------------------------------------------------------


def test_fgm_outside_parameter_range_is_not_increasing():
    th = 3.0
    chk = check_quasi_copula(lambda u, v: u * v * (1 + th * (1 - u) * (1 - v)), m=40)
    assert chk.boundary_ok
    assert not chk.increasing_ok
    assert not chk.is_quasi_copula
    assert "decrease_at" in chk.details


def test_lipschitz_violation_is_detected():
    # boundary conditions, monotone and within the Frechet bounds, but slope 2
    f = lambda u, v: np.minimum(np.minimum(u, v), 2 * np.maximum(u + v - 1, 0))
    chk = check_quasi_copula(f, m=40)
    assert chk.boundary_ok and chk.increasing_ok and chk.frechet_ok
    assert not chk.lipschitz_ok
    assert chk.max_lipschitz_excess == pytest.approx(1 / 40, abs=1e-12)
    assert not is_quasi_copula(f, m=40)
    assert not is_copula(f, m=40)


def test_boundary_violation_is_detected():
    chk = check_quasi_copula(lambda u, v: np.minimum(u * v + 0.05, 1.0), m=20)
    assert not chk.boundary_ok
    assert chk.max_boundary_error == pytest.approx(0.05)
    assert not bool(chk)


# ---------------------------------------------------------------------------
# volumes and the light quasi-copula object
# ---------------------------------------------------------------------------


def test_volume_vectorized():
    Pi = cp.BivIndependenceCopula()
    rects = np.array([[0.1, 0.4, 0.2, 0.9], [0.0, 1.0, 0.0, 1.0], [0.5, 0.5, 0.1, 0.3]])
    vol = quasi_copula_volume(Pi, rects)
    assert vol == pytest.approx([0.3 * 0.7, 1.0, 0.0])
    assert isinstance(quasi_copula_volume(Pi, rects[0]), float)
    with pytest.raises(ValueError):
        quasi_copula_volume(Pi, [0.1, 0.2, 0.3])


def test_numeric_quasi_copula_call_conventions():
    Q = NumericQuasiCopula(lambda u, v: np.minimum(u, v), name="M")
    assert repr(Q) == "M"
    assert isinstance(Q.cdf(0.3, 0.6), float)
    assert Q.cdf(0.3, 0.6) == pytest.approx(0.3)
    assert Q.cdf(u=0.7, v=0.2) == pytest.approx(0.2)
    pts = np.array([[0.1, 0.2], [0.5, 0.4]])
    assert Q.cdf(pts) == pytest.approx([0.1, 0.4])
    assert Q.cdf([0.2, 0.9], 0.5).shape == (2,)
    assert Q.diagonal(0.4) == pytest.approx(0.4)
    assert Q.cdf(1.5, 0.3) == pytest.approx(0.3)  # clipped
    assert Q.volume(CENTER) == pytest.approx(THIRD)


def test_function_copula():
    C = FunctionCopula(lambda u, v: u * v, h1=lambda u, v: v, h2=lambda u, v: u, name="Pi")
    assert C.spearmans_rho() == pytest.approx(0.0, abs=1e-10)
    assert C.kendalls_tau() == pytest.approx(0.0, abs=1e-8)
    assert C.cond_distr_1(0.3, 0.7) == pytest.approx(0.7)
    x = C.rvs(4000, random_state=0)
    assert x.shape == (4000, 2)
    from scipy.stats import kstest

    assert kstest(x[:, 0], "uniform").pvalue > 0.01
    assert kstest(x[:, 1], "uniform").pvalue > 0.01
    # finite-difference h-functions when none are given
    D = FunctionCopula(lambda u, v: np.minimum(u, v))
    assert D.cond_distr_1(0.3, 0.7) == pytest.approx(1.0)
    assert D.cond_distr_1(0.7, 0.3) == pytest.approx(0.0)
    y = D.rvs(1000, random_state=1)
    assert np.allclose(y[:, 0], y[:, 1], atol=1e-5)


@pytest.mark.parametrize("kind", ["contour", "surface", "mass"])
def test_plot(kind):
    C = maximally_nonexchangeable_copula()
    Q = copula_max(C, maximally_nonexchangeable_copula(transpose=True))
    ax = Q.plot(kind=kind, m=30)
    assert ax is not None
    plt.close("all")
    with pytest.raises(ValueError):
        Q.plot(kind="nope", m=10)
    plt.close("all")
