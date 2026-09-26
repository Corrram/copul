"""Exact linear shape constraints vs. brute-force checks and package verifiers."""

import numpy as np
import pytest

from copul.checkerboard.biv_check_min import BivCheckMin
from copul.checkerboard.biv_check_pi import BivCheckPi
from copul.optim import CheckerboardProblem, available_shapes, shape_violation
from copul.optim.problem import balance
from copul.optim.shapes import cvxpy_shape_constraints, resolve_shape, satisfies_shape
from copul.schur_order.cis_verifier import CISVerifier
from copul.schur_order.ltd_verifier import LTDVerifier
from copul.schur_order.plod_verifier import PLODVerifier
from copul.search import si_rearrangement

pytest.importorskip("cvxpy")


def _grid(k, per_cell):
    """Points inside every cell plus points just left/right of each cell boundary."""
    inner = (np.arange(k * per_cell) + 0.5) / (k * per_cell)
    edges = np.arange(1, k) / k
    pts = np.concatenate([inner, edges - 1e-7, edges + 1e-7, [1e-6, 1 - 1e-6]])
    return np.unique(pts)


def _cdf(P, kind, u, v):
    U, V = np.meshgrid(u, v, indexing="ij")
    cop = BivCheckPi(P) if kind == "pi" else BivCheckMin(P)
    pts = np.column_stack([U.ravel(), V.ravel()])
    return np.asarray(cop.cdf(pts), dtype=float).reshape(U.shape), U, V


def _brute(P, name, kind="pi"):
    """Brute-force violation of a shape property from the package CDF (0 iff it holds)."""
    m, n = P.shape
    if name in ("si", "sd"):
        # d1 C at u-points that are cell midpoints (C is piecewise linear in u)
        v = _grid(n, 6)
        u_lo = np.arange(m) / m + 0.3 / m
        u_hi = u_lo + 0.2 / m
        C1, _, _ = _cdf(P, kind, u_lo, v)
        C2, _, _ = _cdf(P, kind, u_hi, v)
        h = (C2 - C1) / (0.2 / m)
        d = h[:-1] - h[1:]  # >= 0 for SI
        return max(0.0, -float((d if name == "si" else -d).min()))
    if name in ("ltd", "lti", "rti", "rtd"):
        u = _grid(m, 6)
        v = _grid(n, 3)
        C, U, V = _cdf(P, kind, u, v)
        if name in ("ltd", "lti"):
            r = C / U
        else:
            r = (1 - U - V + C) / (1 - U)
        d = np.diff(r, axis=0)  # along u
        sign = {"ltd": -1, "lti": 1, "rti": 1, "rtd": -1}[name]
        return max(0.0, -float((sign * d).min()))
    if name in ("pqd", "nqd"):
        u = _grid(m, 4)
        v = _grid(n, 4)
        C, U, V = _cdf(P, kind, u, v)
        d = C - U * V
        return max(0.0, -float((d if name == "pqd" else -d).min()))
    raise AssertionError(name)


def _samples(n_samples, rng):
    out = []
    for _ in range(n_samples):
        m = int(rng.integers(2, 5))
        n = int(rng.integers(2, 5))
        A = balance(rng.random((m, n)) ** 3 + 1e-3)
        base = si_rearrangement(A)
        t = rng.choice([0.0, 0.02, 0.1, 0.5])
        out.append(balance((1 - t) * base + t * A))
    return out


@pytest.mark.parametrize("name", ["si", "sd", "ltd", "lti", "rti", "rtd", "pqd", "nqd"])
def test_exact_conditions_match_bruteforce(name):
    rng = np.random.default_rng(["si", "sd", "ltd", "lti", "rti", "rtd", "pqd", "nqd"].index(name))
    agree = {True: 0, False: 0}
    for P in _samples(120, rng):
        if name in ("sd", "lti", "rtd", "nqd"):
            P = P[::-1].copy()  # negatively dependent samples
        brute = _brute(P, name)
        if 1e-11 < brute < 1e-8:
            continue  # numerically ambiguous, both answers acceptable
        exact = satisfies_shape(P, name, tol=1e-12)
        assert exact == (brute <= 1e-11), (name, P, brute, shape_violation(P, name))
        agree[exact] += 1
    assert agree[True] >= 2 and agree[False] >= 2, agree


def test_min_si_and_pqd_bruteforce():
    rng = np.random.default_rng(11)
    agree = {True: 0, False: 0}
    for _ in range(40):
        n = int(rng.integers(2, 5))
        base = np.eye(n) / n
        if rng.random() < 0.5:
            base = si_rearrangement(balance(rng.random((n, n)) ** 6))
        P = balance(base + rng.choice([0, 1e-3, 0.05]) * rng.random((n, n)))
        for name in ("pqd",):
            b = _brute(P, name, "min")
            if not 1e-11 < b < 1e-8:
                assert satisfies_shape(P, name, "min", tol=1e-12) == (b <= 1e-11)
        # SI of M-checkerboards: check d1 C on both sides of each row boundary
        m = n
        v = _grid(n, 7)
        cop = BivCheckMin(P)
        us = np.concatenate([np.arange(m) / m + 1e-7, np.arange(1, m + 1) / m - 1e-7])
        us = np.sort(us)
        hs = []
        for u in us:
            pts = np.column_stack([np.full(v.size, u + 1e-9), v])
            pts2 = np.column_stack([np.full(v.size, u - 1e-9), v])
            hs.append((np.asarray(cop.cdf(pts)) - np.asarray(cop.cdf(pts2))) / 2e-9)
        hs = np.array(hs)
        violation = max(0.0, -float((hs[:-1] - hs[1:]).min()))
        if not 1e-5 < violation < 1e-3:  # finite differences are accurate to ~1e-6
            exact = satisfies_shape(P, "si", "min", tol=1e-12)
            assert exact == (violation <= 1e-5), (P, violation)
            agree[exact] += 1
    assert agree[True] > 0 and agree[False] > 0, agree


VERIFIERS = {
    "si": lambda c: CISVerifier(1).cis_direction(c)[0],
    "sd": lambda c: CISVerifier(1).cis_direction(c)[1],
    "si2": lambda c: CISVerifier(2).cis_direction(c)[0],
    "ltd": lambda c: LTDVerifier().is_ltd(c),
    "lti": lambda c: LTDVerifier().is_lti(c),
    "rti": lambda c: LTDVerifier().is_rti(c),
    "rtd": lambda c: LTDVerifier().is_rtd(c),
    "pqd": lambda c: PLODVerifier().is_plod(c),
}


@pytest.mark.parametrize("name", sorted(VERIFIERS))
def test_optimal_outputs_pass_package_verifiers(name):
    rng = np.random.default_rng(len(name))
    prob = CheckerboardProblem(n=6, constraints=[name])
    W = rng.standard_normal((6, 6))
    from copul.optim.checkerboard_formulas import QuadraticForm

    res = prob.maximize(QuadraticForm(W, 0.0, [], "random"))
    assert shape_violation(res.P, name) < 1e-8
    assert VERIFIERS[name](res.copula)


def test_symmetries_and_registry():
    prob = CheckerboardProblem(n=5, constraints=["exchangeable", "radially_symmetric"])
    res = prob.maximize("nu", subject_to={"rho": ("==", 0.3)})
    assert np.allclose(res.P, res.P.T, atol=1e-8)
    assert np.allclose(res.P, res.P[::-1, ::-1], atol=1e-8)
    assert res.copula.is_symmetric
    assert "si" in available_shapes("pi") and "si" in available_shapes("min")
    assert "nqd" in available_shapes("w") and "si" not in available_shapes("w")
    assert resolve_shape("CI") == "si" and resolve_shape("plod") == "pqd"
    with pytest.raises(KeyError):
        resolve_shape("tp2")
    with pytest.raises(NotImplementedError):
        CheckerboardProblem(n=3, kind="w").add_constraint("ltd")
    with pytest.raises(ValueError):
        cvxpy_shape_constraints(np.ones((2, 3)) / 6, "exchangeable")


def test_si_is_the_rearrangement_fixed_point():
    rng = np.random.default_rng(0)
    for _ in range(20):
        P = balance(rng.random((5, 5)))
        S = si_rearrangement(P)
        assert satisfies_shape(S, "si", tol=1e-12)
        assert np.allclose(si_rearrangement(S), S)
