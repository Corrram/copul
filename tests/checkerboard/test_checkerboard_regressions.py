"""Regression tests for bugs fixed in the bivariate checkerboard engine.

Reference values are obtained numerically from the (independently verified)
distribution function, never from the closed forms under test.
"""

import time

import numpy as np
import pytest

from copul.checkerboard.biv_check_min import BivCheckMin
from copul.checkerboard.biv_check_mixed import BivCheckMixed
from copul.checkerboard.biv_check_pi import BivCheckPi
from copul.checkerboard.biv_check_w import BivCheckW
from copul.checkerboard.check_min import CheckMin
from copul.checkerboard.check_pi import CheckPi


def _copula_matrix(m, n, seed):
    rng = np.random.default_rng(seed)
    A = rng.random((m, n)) ** 3 + 1e-3
    for _ in range(3000):
        A = A / A.sum(1, keepdims=True) / m
        A = A / A.sum(0, keepdims=True) / n
    return A


def _cell_cdf(D, s, u, v):
    """Independent reference cdf (explicit loop over cells)."""
    m, n = D.shape
    out = np.zeros(np.broadcast(u, v).shape)
    for i in range(m):
        a = np.clip(m * u - i, 0, 1)
        for j in range(n):
            b = np.clip(n * v - j, 0, 1)
            if s == 0:
                k = a * b
            elif s == 1:
                k = np.minimum(a, b)
            else:
                k = np.maximum(a + b - 1, 0)
            out = out + D[i, j] * k
    return out


def _numeric_nu(cop, N=800):
    g = (np.arange(N) + 0.5) / N
    U, V = np.meshgrid(g, g, indexing="ij")
    return 24 * np.mean((1 - U) * cop.cdf(U, V)) - 2


def _numeric_diag(cop, N=200_000):
    t = (np.arange(N) + 0.5) / N
    return np.mean(cop.cdf(t, t)), np.mean(cop.cdf(t, 1 - t))


SHAPES = [(2, 1), (3, 3), (4, 4), (3, 5), (5, 2)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("cls", [BivCheckPi, BivCheckMin, BivCheckW])
def test_blests_nu_matches_numerical(shape, cls):
    D = _copula_matrix(*shape, seed=sum(shape))
    cop = cls(D)
    assert np.isclose(cop.blests_nu(), _numeric_nu(cop), atol=2e-5)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("cls", [BivCheckPi, BivCheckMin, BivCheckW])
def test_footrule_and_gini_match_numerical(shape, cls):
    D = _copula_matrix(*shape, seed=10 + sum(shape))
    cop = cls(D)
    i_diag, i_anti = _numeric_diag(cop)
    assert np.isclose(cop.spearmans_footrule(), 6 * i_diag - 2, atol=1e-6)
    assert np.isclose(cop.ginis_gamma(), 4 * (i_diag + i_anti) - 2, atol=1e-6)


@pytest.mark.parametrize("cls,s", [(BivCheckPi, 0), (BivCheckMin, 1), (BivCheckW, -1)])
def test_cdf_matches_independent_reference(cls, s):
    D = _copula_matrix(3, 5, seed=3)
    cop = cls(D)
    rng = np.random.default_rng(0)
    pts = rng.random((300, 2))
    ref = _cell_cdf(D, s, pts[:, 0], pts[:, 1])
    assert np.allclose(cop.cdf(pts), ref, atol=1e-14)
    assert np.allclose(cop.cdf(pts[:, 0], pts[:, 1]), ref, atol=1e-14)
    assert np.isclose(cop.cdf(float(pts[0, 0]), float(pts[0, 1])), ref[0])
    assert np.isclose(cop.cdf(u=float(pts[0, 0]), v=float(pts[0, 1])), ref[0])


@pytest.mark.parametrize("shape", [(3, 3), (3, 5), (5, 2)])
def test_min_tail_dependence(shape):
    D = _copula_matrix(*shape, seed=7)
    cop = BivCheckMin(D)
    eps = 1e-9
    assert np.isclose(cop.lambda_L(), D[0, 0] * min(shape))
    assert np.isclose(cop.lambda_U(), D[-1, -1] * min(shape))
    assert np.isclose(cop.lambda_L(), cop.cdf(eps, eps) / eps, atol=1e-6)
    upper = (1 - 2 * (1 - eps) + cop.cdf(1 - eps, 1 - eps)) / eps
    assert np.isclose(cop.lambda_U(), upper, atol=1e-5)
    assert BivCheckW(D).lambda_L() == 0 and BivCheckPi(D).lambda_U() == 0


def test_checkpi_vectorized_pdf_is_not_zero():
    rng = np.random.default_rng(1)
    cop = CheckPi(rng.random((2, 3, 2)))
    pts = rng.random((20, 3))
    vec = cop.pdf(pts)
    single = np.array([cop.pdf(p) for p in pts])
    assert np.all(vec > 0)
    assert np.allclose(vec, single)


@pytest.mark.parametrize("cls", [BivCheckPi, BivCheckMin, BivCheckW])
def test_cond_distr_vectorized_matches_scalar_and_derivative(cls):
    D = _copula_matrix(4, 3, seed=5)
    cop = cls(D)
    rng = np.random.default_rng(2)
    pts = rng.random((200, 2))
    for i in (1, 2):
        vec = cop.cond_distr(i, pts)
        vec2 = cop.cond_distr(i, pts[:, 0], pts[:, 1])
        sc = np.array([cop.cond_distr(i, a, b) for a, b in pts])
        assert np.allclose(vec, sc) and np.allclose(vec, vec2)
        h = 1e-7
        e = np.array([h, 0]) if i == 1 else np.array([0, h])
        num = (cop.cdf(pts + e) - cop.cdf(pts - e)) / (2 * h)
        # away from the singular lines the derivative is well defined
        good = np.abs(num - np.round(num * 1e6) / 1e6) < 1
        assert np.allclose(vec[good], num[good], atol=1e-5)
    assert cop.cond_distr_1(0.3, 0.6) == cop.cond_distr(1, [0.3, 0.6])


def test_checkerboard_cdf_is_fast_for_large_grids():
    cop = BivCheckPi.generate_diverse(1, grid_size=50, rng=0)
    P = np.random.default_rng(0).random((20_000, 2))
    t = time.perf_counter()
    cop.cdf(P)
    cop.cond_distr_1(P)
    BivCheckMin(cop.matr).cdf(P)
    BivCheckMin(cop.matr).rvs(100_000, random_state=0)
    assert time.perf_counter() - t < 1.0


def test_generate_randomly_sizes_rng_and_no_global_seed():
    state = np.random.get_state()[1].copy()
    cops = BivCheckPi.generate_randomly(grid_size=[2, 4], n=60, rng=1)
    sizes = {c.m for c in cops}
    assert sizes == {2, 3, 4}  # inclusive upper end, per-sample sizes
    again = BivCheckPi.generate_randomly(grid_size=(2, 4), n=60, random_state=1)
    assert all(np.allclose(a.matr, b.matr) for a, b in zip(cops, again))
    assert np.array_equal(np.random.get_state()[1], state)
    assert isinstance(BivCheckMin.generate_randomly(3, rng=0), BivCheckMin)
    assert isinstance(BivCheckW.generate_randomly(3, n=2, rng=0)[1], BivCheckW)


@pytest.mark.parametrize("cls", [BivCheckPi, BivCheckMin, BivCheckW])
def test_rvs_random_state_reproducible_without_global_seeding(cls):
    cop = cls(_copula_matrix(3, 4, seed=0))
    state = np.random.get_state()[1].copy()
    a = cop.rvs(500, random_state=3)
    b = cop.rvs(500, random_state=np.random.default_rng(3))
    assert np.allclose(a, b)
    assert np.array_equal(np.random.get_state()[1], state)


def test_constructors_do_not_mutate_input():
    D = np.array([[1.0, 2.0], [2.0, 1.0]])
    for cls in [BivCheckPi, BivCheckMin, BivCheckW, CheckPi, CheckMin]:
        cls(D)
        assert np.array_equal(D, [[1.0, 2.0], [2.0, 1.0]])
    BivCheckMixed(D, sign=[[1, 0], [0, -1]])
    assert np.array_equal(D, [[1.0, 2.0], [2.0, 1.0]])


@pytest.mark.parametrize("cls", [BivCheckPi, BivCheckMin, BivCheckW, BivCheckMixed])
def test_condition_on_y_is_keyword_and_honoured(cls):
    D = _copula_matrix(3, 4, seed=11)
    cop = cls(D)
    xt = cls(D.T).chatterjees_xi()
    assert np.isclose(cop.chatterjees_xi(condition_on_y=True), xt)
    # a positional boolean is interpreted as ``condition_on_y``
    assert np.isclose(cop.chatterjees_xi(True), xt)
    with pytest.raises(TypeError):
        cop.chatterjees_xi(condition_on_x=True)


def test_approximate_shuffle_of_min_from_copula():
    from copul.checkerboard.checkerboarder import Checkerboarder
    from copul.checkerboard.shuffle_min import ShuffleOfMin

    som = Checkerboarder(4).approximate_shuffle_of_min(copula=BivCheckMin(np.eye(4)))
    assert isinstance(som, ShuffleOfMin)
    assert som.pi.tolist() == [1, 2, 3, 4]


def test_fast_rank_vectorised():
    from copul.checkerboard.checkerboarder import _fast_rank

    x = np.array([0.3, -1.0, 2.5, 0.0])
    assert np.allclose(_fast_rank(x), [3 / 4, 1 / 4, 1.0, 2 / 4])


def test_checkerboarder_roundtrip_uses_vectorised_cdf():
    from copul.checkerboard.checkerboarder import Checkerboarder

    D = _copula_matrix(4, 4, seed=21)
    approx = Checkerboarder(4).get_checkerboard_copula(BivCheckMin(D))
    assert np.allclose(approx.matr, D)
