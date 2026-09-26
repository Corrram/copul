"""Consistency of all closed-form measures with numerical integration.

For random (seeded) copula matrices and every checkerboard-type class the
closed-form dependence measures are compared with independent numerical
evaluations computed from the class's own ``cdf`` / ``cond_distr`` on fine
midpoint grids:

* rho      = 12 E[C] - 3
* tau      = 1 - 4 E[d1C d2C]
* xi       = 6 E[(d1C)^2] - 2        (and d2C for ``condition_on_y=True``)
* nu       = 24 E[(1-u) C] - 2
* footrule = 6 int C(t,t) - 2
* gini     = 4 (int C(t,t) + int C(t,1-t)) - 2
* beta     = 4 C(1/2,1/2) - 1
* lambda_L = lim C(e,e)/e,  lambda_U = lim (1 - 2(1-e) + C(1-e,1-e))/e

For the checkerboards the cdf itself is additionally compared with an
explicit cell-by-cell reference, and ``rvs`` margins are tested for
uniformity (KS).
"""

import numpy as np
import pytest
from scipy.stats import kstest

from copul.checkerboard.biv_bernstein import BivBernsteinCopula
from copul.checkerboard.biv_block_diag_mixed import BivBlockDiagMixed
from copul.checkerboard.biv_check_min import BivCheckMin
from copul.checkerboard.biv_check_mixed import BivCheckMixed
from copul.checkerboard.biv_check_pi import BivCheckPi
from copul.checkerboard.biv_check_w import BivCheckW
from copul.checkerboard.shuffle_min import ShuffleOfMin

SHAPES = [(3, 3), (4, 4), (3, 5), (5, 2)]
N_GRID = 600  # divisible by every grid size used (cell edges align)
N_DIAG = 120_000
TOL = 1e-3

MEASURES = [
    "spearmans_rho",
    "kendalls_tau",
    "chatterjees_xi",
    "xi_y",
    "blests_nu",
    "spearmans_footrule",
    "ginis_gamma",
    "blomqvists_beta",
    "lambda_L",
    "lambda_U",
]


def _matrix(m, n, seed):
    rng = np.random.default_rng(seed)
    A = rng.random((m, n)) ** 3 + 1e-3
    # make the corner cells heavy sometimes (non-trivial tail dependence)
    A[0, 0] += rng.random()
    A[-1, -1] += rng.random()
    for _ in range(3000):
        A = A / A.sum(1, keepdims=True) / m
        A = A / A.sum(0, keepdims=True) / n
    return A


def _build(kind, shape, seed):
    m, n = shape
    D = _matrix(m, n, seed)
    if kind == "pi":
        return BivCheckPi(D), D, 0
    if kind == "min":
        return BivCheckMin(D), D, 1
    if kind == "w":
        return BivCheckW(D), D, -1
    if kind == "mixed":
        S = np.random.default_rng(seed + 100).integers(-1, 2, size=shape)
        S[0, 0] = 1
        return BivCheckMixed(D, sign=S), D, S
    if kind == "bernstein":
        return BivBernsteinCopula(D), D, None
    if kind == "shuffle":
        perm = np.random.default_rng(seed).permutation(m)
        perm[[0, np.argmin(perm)]] = perm[[np.argmin(perm), 0]]  # pi(1) = 1
        return ShuffleOfMin(perm + 1), None, None
    raise ValueError(kind)


CASES = [(kind, shape) for shape in SHAPES for kind in ["pi", "min", "w", "mixed", "bernstein"]] + [
    ("shuffle", (3, 3)),
    ("shuffle", (4, 4)),
]

_CACHE = {}


def _numerical(cop):
    key = id(cop)
    if key in _CACHE:
        return _CACHE[key]
    # Midpoint rule in v; in u the midpoints are shifted by +-1/4 of a step
    # and both results averaged.  On the aligned grid the singular diagonals
    # of Min/W cells pass exactly through grid points, and the symmetric
    # shift cancels the resulting O(1/N) bias of the step functions.
    g = (np.arange(N_GRID) + 0.5) / N_GRID
    parts = []
    for shift in (0.25, -0.25):
        U, V = np.meshgrid(g + shift / N_GRID, g, indexing="ij")
        C = cop.cdf(U, V)
        C1 = cop.cond_distr_1(U, V)
        C2 = cop.cond_distr_2(U, V)
        parts.append(
            (
                12 * C.mean() - 3,
                1 - 4 * np.mean(C1 * C2),
                6 * np.mean(C1**2) - 2,
                6 * np.mean(C2**2) - 2,
                24 * np.mean((1 - U) * C) - 2,
            )
        )
    rho, tau, xi, xi_y, nu = np.mean(parts, axis=0)
    t = (np.arange(N_DIAG) + 0.5) / N_DIAG
    diag = np.mean(cop.cdf(t, t))
    anti = np.mean(cop.cdf(t, 1 - t))
    eps = 1e-7
    res = {
        "spearmans_rho": rho,
        "kendalls_tau": tau,
        "chatterjees_xi": xi,
        "xi_y": xi_y,
        "blests_nu": nu,
        "spearmans_footrule": 6 * diag - 2,
        "ginis_gamma": 4 * (diag + anti) - 2,
        "blomqvists_beta": 4 * cop.cdf(0.5, 0.5) - 1,
        "lambda_L": cop.cdf(eps, eps) / eps,
        "lambda_U": (1 - 2 * (1 - eps) + cop.cdf(1 - eps, 1 - eps)) / eps,
    }
    _CACHE[key] = res
    return res


_COPULAS = {}


def _copula(kind, shape):
    key = (kind, shape)
    if key not in _COPULAS:
        _COPULAS[key] = _build(kind, shape, seed=7 * shape[0] + shape[1])
    return _COPULAS[key]


def _closed_form(cop, name):
    if name == "xi_y":
        return cop.chatterjees_xi(condition_on_y=True)
    return getattr(cop, name)()


@pytest.mark.parametrize("kind,shape", CASES, ids=[f"{k}-{s[0]}x{s[1]}" for k, s in CASES])
def test_closed_forms_match_numerical_integration(kind, shape):
    cop, _, _ = _copula(kind, shape)
    num = _numerical(cop)
    errors = {}
    for name in MEASURES:
        val = _closed_form(cop, name)
        assert isinstance(val, (float, int, np.floating)), name
        if abs(float(val) - num[name]) > TOL:
            errors[name] = (float(val), float(num[name]))
    assert not errors, errors


@pytest.mark.parametrize("kind", ["pi", "min", "w", "mixed"])
@pytest.mark.parametrize("shape", SHAPES)
def test_checkerboard_cdf_matches_cell_reference(kind, shape):
    cop, D, S = _copula(kind, shape)
    m, n = D.shape
    S = np.broadcast_to(S, D.shape)
    rng = np.random.default_rng(1)
    u, v = rng.random(400), rng.random(400)
    ref = np.zeros_like(u)
    for i in range(m):
        a = np.clip(m * u - i, 0, 1)
        for j in range(n):
            b = np.clip(n * v - j, 0, 1)
            k = {0: a * b, 1: np.minimum(a, b), -1: np.maximum(a + b - 1, 0)}[S[i, j]]
            ref += D[i, j] * k
    assert np.allclose(cop.cdf(u, v), ref, atol=1e-13)


@pytest.mark.parametrize("kind,shape", CASES, ids=[f"{k}-{s[0]}x{s[1]}" for k, s in CASES])
def test_rvs_margins_uniform_and_call_conventions(kind, shape):
    cop, _, _ = _copula(kind, shape)
    X = cop.rvs(4000, random_state=11)
    assert X.shape == (4000, 2)
    assert kstest(X[:, 0], "uniform").pvalue > 1e-3
    assert kstest(X[:, 1], "uniform").pvalue > 1e-3
    # empirical cdf agrees with the cdf
    for a, b in [(0.3, 0.6), (0.7, 0.2), (0.5, 0.5)]:
        assert abs(np.mean((X[:, 0] <= a) & (X[:, 1] <= b)) - cop.cdf(a, b)) < 0.03
    # call conventions: scalars -> float, arrays -> ndarray of the same shape
    for f in (cop.cdf, cop.cond_distr_1, cop.cond_distr_2):
        assert isinstance(f(0.3, 0.6), float)
        uu = np.array([[0.1, 0.4], [0.6, 0.9]])
        out = f(uu, uu.T)
        assert isinstance(out, np.ndarray) and out.shape == (2, 2)
        pts = np.column_stack([uu.ravel(), uu.T.ravel()])
        assert np.allclose(f(pts), out.ravel())
        assert np.isclose(f(u=0.3, v=0.6), f(0.3, 0.6))


def test_block_diag_mixed_closed_forms_match_engine():
    sizes = [1, 2, 3]
    d = sum(sizes)
    S = np.random.default_rng(0).integers(-1, 2, size=(d, d))
    blk = BivBlockDiagMixed(sizes, sign=S)
    general = BivCheckMixed(blk.matr, sign=S)
    for name in ["kendalls_tau", "spearmans_rho", "chatterjees_xi"]:
        assert np.isclose(getattr(blk, name)(), getattr(general, name)())
    assert np.isclose(
        blk.chatterjees_xi(condition_on_y=True),
        general.chatterjees_xi(condition_on_y=True),
    )
