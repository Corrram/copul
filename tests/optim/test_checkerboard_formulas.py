"""Exact checkerboard measure formulas vs. independent numerical integration."""

import numpy as np
import pytest

from copul.checkerboard.biv_check_min import BivCheckMin
from copul.checkerboard.biv_check_pi import BivCheckPi
from copul.checkerboard.biv_check_w import BivCheckW
from copul.optim.checkerboard_formulas import (
    Bilinear,
    QuadraticForm,
    RowGram,
    _chatterjee_gram,
    checkerboard_copula,
    is_feasible_mass_matrix,
    measure_form,
    measure_values,
)
from copul.optim.problem import balance

KEYS = ("xi", "rho", "tau", "footrule", "gamma", "beta", "nu")


def _local(kind, s, t):
    """Local copula D(s,t) and its partial derivatives (ties split evenly)."""
    if kind == "pi":
        return s * t, t + 0 * s, s + 0 * t
    if kind == "min":
        ds = np.where(s < t, 1.0, np.where(s == t, 0.5, 0.0))
        return np.minimum(s, t), ds, 1 - ds
    ds = np.where(s + t > 1, 1.0, np.where(s + t == 1, 0.5, 0.0))
    return np.maximum(s + t - 1, 0), ds, ds


def _brute(P, kind, K):
    """Midpoint-rule integration of the defining integrals (independent code)."""
    m, n = P.shape
    u = (np.arange(m * K) + 0.5) / (m * K)
    v = (np.arange(n * K) + 0.5) / (n * K)
    U, V = np.meshgrid(u, v, indexing="ij")
    I = np.minimum((U * m).astype(int), m - 1)
    J = np.minimum((V * n).astype(int), n - 1)
    S, T = U * m - I, V * n - J
    Dv, Ds, Dt = _local(kind, S, T)
    G = np.zeros((m + 1, n + 1))
    G[1:, 1:] = P.cumsum(0).cumsum(1)
    rc = np.zeros((m, n + 1))
    rc[:, 1:] = P.cumsum(1)
    cc = np.zeros((m + 1, n))
    cc[1:, :] = P.cumsum(0)
    Pij = P[I, J]
    C = G[I, J] + S * rc[I, J] + T * cc[I, J] + Pij * Dv
    h1 = m * (rc[I, J] + Pij * Ds)
    h2 = n * (cc[I, J] + Pij * Dt)

    def Cf(a, b):
        i = np.minimum((a * m).astype(int), m - 1)
        j = np.minimum((b * n).astype(int), n - 1)
        s, t = a * m - i, b * n - j
        d, _, _ = _local(kind, s, t)
        return G[i, j] + s * rc[i, j] + t * cc[i, j] + P[i, j] * d

    out = {
        "rho": 12 * C.mean() - 3,
        "nu": 24 * ((1 - U) * C).mean() - 2,
        "xi": 6 * (h1**2).mean() - 2,
        "tau": 1 - 4 * (h1 * h2).mean(),
        "beta": 4 * float(Cf(np.array(0.5), np.array(0.5))) - 1,
    }
    if m == n:
        tt = (np.arange(4 * n * K) + 0.5) / (4 * n * K)
        out["footrule"] = 6 * Cf(tt, tt).mean() - 2
        out["gamma"] = 4 * (Cf(tt, tt).mean() + Cf(tt, 1 - tt).mean()) - 2
    return out


def _random_P(m, n, rng):
    return balance(rng.random((m, n)) ** 3 + 1e-3)


@pytest.mark.parametrize("kind", ["pi", "min", "w"])
@pytest.mark.parametrize("shape", [(3, 3), (2, 3), (4, 4), (3, 5)])
def test_formulas_match_numerical_integration(kind, shape):
    rng = np.random.default_rng({"pi": 0, "min": 1, "w": 2}[kind] * 100 + 10 * shape[0] + shape[1])
    P = _random_P(*shape, rng)
    exact = measure_values(P, kind)
    # The midpoint rule converges like O(1/K) for the indicator integrands of
    # M/W cells and like O(1/K^2) otherwise: errors must shrink accordingly,
    # which would fail if a formula were off by a constant.
    b1, b2 = _brute(P, kind, 60), _brute(P, kind, 120)
    for k in b1:
        e1, e2 = exact[k] - b1[k], exact[k] - b2[k]
        assert abs(e2) <= 2e-3, (k, e2)
        if abs(e1) > 1e-4:
            assert 1.6 <= e1 / e2 <= 4.5, (k, e1, e2)
        else:
            assert abs(e2) <= 1e-4, (k, e2)
    if shape[0] != shape[1]:
        assert np.isnan(exact["footrule"]) and np.isnan(exact["gamma"])


def test_pi_formulas_match_package_methods():
    cops = BivCheckPi.generate_diverse(40, grid_size=(2, 12), rng=3)
    for c in cops:
        vals = measure_values(c.matr, "pi")
        assert vals["xi"] == pytest.approx(c.chatterjees_xi(), abs=1e-12)
        assert vals["rho"] == pytest.approx(c.spearmans_rho(), abs=1e-12)
        assert vals["tau"] == pytest.approx(c.kendalls_tau(), abs=1e-12)
        assert vals["nu"] == pytest.approx(c.blests_nu(), abs=1e-12)
        assert vals["footrule"] == pytest.approx(c.spearmans_footrule(), abs=1e-12)
        assert vals["gamma"] == pytest.approx(c.ginis_gamma(), abs=1e-12)
        assert vals["beta"] == pytest.approx(c.blomqvists_beta(), abs=1e-12)


@pytest.mark.parametrize("cls,kind", [(BivCheckMin, "min"), (BivCheckW, "w")])
def test_min_w_formulas_match_uncontroversial_package_methods(cls, kind):
    rng = np.random.default_rng(5)
    for n in (2, 3, 5):
        P = _random_P(n, n, rng)
        c = cls(P)
        vals = measure_values(P, kind)
        assert vals["xi"] == pytest.approx(c.chatterjees_xi(), abs=1e-12)
        assert vals["rho"] == pytest.approx(c.spearmans_rho(), abs=1e-12)
        assert vals["tau"] == pytest.approx(c.kendalls_tau(), abs=1e-12)


@pytest.mark.parametrize("n", [1, 2, 5, 8])
def test_frechet_bounds(n):
    I = np.eye(n) / n
    A = I[:, ::-1]
    vm = measure_values(I, "min")
    for k in KEYS:
        assert vm[k] == pytest.approx(1.0, abs=1e-12), k
    vw = measure_values(A, "w")
    expected = {"xi": 1, "rho": -1, "tau": -1, "footrule": -0.5, "gamma": -1}
    expected.update({"beta": -1, "nu": -1})
    for k, val in expected.items():
        assert vw[k] == pytest.approx(val, abs=1e-12), k
    vp = measure_values(I, "pi")
    assert vp["rho"] == pytest.approx(1 - 1 / n**2, abs=1e-12)
    vpi = measure_values(np.full((n, n), 1 / n**2), "pi")
    for k in KEYS:
        assert vpi[k] == pytest.approx(0.0, abs=1e-12), k


@pytest.mark.parametrize("n", [1, 2, 3, 10, 40])
def test_chatterjee_gram_positive_definite(n):
    assert np.linalg.eigvalsh(_chatterjee_gram(n)).min() > 0.08


def test_curvature_and_arithmetic():
    rho, xi, tau = (measure_form(k, 5) for k in ("rho", "xi", "tau"))
    assert rho.curvature() == "affine"
    assert xi.curvature() == "convex"
    assert (-xi).curvature() == "concave"
    assert (rho - 0.5 * xi).curvature() == "concave"
    assert tau.curvature() == "indefinite"
    assert measure_form("tau", 5, kind="min").curvature() == "indefinite"
    P = _random_P(5, 5, np.random.default_rng(0))
    f = 2 * rho - xi / 4 + 1.0
    assert f.value(P) == pytest.approx(2 * rho(P) - xi(P) / 4 + 1.0)
    assert (1.0 - rho).value(P) == pytest.approx(1.0 - rho(P))
    assert isinstance(f, QuadraticForm)


@pytest.mark.parametrize("key", KEYS)
def test_gradients_by_finite_differences(key):
    rng = np.random.default_rng(1)
    P = _random_P(4, 4, rng)
    f = measure_form(key, 4, 4, "min")
    D = rng.standard_normal(P.shape)
    eps = 1e-6
    fd = (f.value(P + eps * D) - f.value(P - eps * D)) / (2 * eps)
    assert np.sum(f.grad(P) * D) == pytest.approx(fd, rel=1e-6, abs=1e-8)


def test_terms_dense_representation():
    rng = np.random.default_rng(2)
    P = rng.random((3, 4))
    G = _chatterjee_gram(4)
    A, B = rng.random((3, 3)), rng.random((4, 4))
    x = P.reshape(-1)
    assert x @ RowGram(G).dense(3, 4) @ x == pytest.approx(RowGram(G).value(P))
    assert x @ Bilinear(A, B).dense(3, 4) @ x == pytest.approx(Bilinear(A, B).value(P))
    R = RowGram(G).factor
    assert np.allclose(R @ R.T, G)


def test_errors_and_helpers():
    with pytest.raises(ValueError):
        measure_form("footrule", 3, 4)
    with pytest.raises(KeyError):
        measure_form("kappa", 3)
    with pytest.raises(ValueError):
        measure_form("rho", 3, kind="bernstein")
    P = _random_P(3, 3, np.random.default_rng(0))
    assert is_feasible_mass_matrix(P)
    assert not is_feasible_mass_matrix(P + 0.01)
    assert isinstance(checkerboard_copula(P, "min"), BivCheckMin)
    assert isinstance(checkerboard_copula(P, "w"), BivCheckW)
    assert type(checkerboard_copula(P)) is BivCheckPi
