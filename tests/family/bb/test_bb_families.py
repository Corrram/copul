import mpmath
import numpy as np
import pytest
import sympy as sp

import copul as cp
from copul.family.bb import BB1, BB2, BB3, BB6, BB7, BB8, BB9, BB10, LTArchimedeanCopula
from copul.family.bb._frailty import (
    log_gamma_rv,
    log_positive_stable_rv,
    log_tilted_stable_rv,
    sibuya_rv,
)
from tests.family.numeric_copula_checks import (
    check_axioms,
    check_conditionals,
    check_density,
    check_sampling,
)

MEMBERS = [
    (BB1, (0.7, 1.4)),
    (BB1, (2.5, 3.0)),
    (BB2, (1.2, 1.5)),
    (BB2, (2.0, 1.0)),
    (BB3, (1.5, 0.8)),
    (BB3, (1.0, 2.0)),
    (BB6, (1.5, 2.0)),
    (BB7, (1.5, 0.8)),
    (BB7, (2.5, 2.0)),
    (BB8, (2.5, 0.7)),
    (BB8, (3.0, 1.0)),
    (BB9, (2.0, 0.5)),
    (BB10, (1.5, 0.6)),
]
IDS = [f"{c.__name__}{p}" for c, p in MEMBERS]

U = np.array([0.05, 0.2, 0.5, 0.7, 0.93, 0.4])
V = np.array([0.3, 0.9, 0.5, 0.1, 0.96, 0.4])


def _cdf(C, u, v):
    return np.array([float(C.cdf(a, b)) for a, b in zip(u, v)])


@pytest.mark.parametrize(("cls", "params"), MEMBERS, ids=IDS)
def test_copula_properties(cls, params):
    C = cls(*params)
    check_axioms(C)
    check_conditionals(C)
    check_density(C)
    check_sampling(C)


@pytest.mark.parametrize(("cls", "params"), MEMBERS, ids=IDS)
def test_numeric_matches_symbolic_cdf(cls, params):
    """The log-scale numerics agree with the SymPy cdf and its derivatives (mpmath)."""
    C = cls(*params)
    expr = C._cdf_expr
    u, v = C.u, C.v
    mpmath.mp.dps = 30
    funcs = {
        "cdf": (expr, C.cdf_vectorized),
        "h1": (sp.diff(expr, u), C.cond_distr_1_vectorized),
        "pdf": (sp.diff(expr, u, v), C.pdf_vectorized),
    }
    for name, (e, fn) in funcs.items():
        f = sp.lambdify((u, v), e, "mpmath")
        ref = np.array([float(f(mpmath.mpf(a), mpmath.mpf(b))) for a, b in zip(U, V)])
        np.testing.assert_allclose(fn(U, V), ref, rtol=1e-8, atol=1e-12, err_msg=name)


def test_generator_round_trip():
    for cls, params in MEMBERS:
        C = cls(*params)
        for t in (0.1, 0.5, 0.9):
            y = float(C.generator(t=t))
            if not np.isfinite(y):  # BB2 generators exceed double range
                continue
            assert float(C.inv_generator(y=y)) == pytest.approx(t, abs=1e-10)
            assert np.exp(C._log_phi(np.array(t), *C._pv)) == pytest.approx(y, rel=1e-10)


# ------------------------------------------------------------------ closed forms


def _numeric(C, key):
    return C.measure(key, method="numeric", full_output=True)


@pytest.mark.parametrize(
    ("C", "value"),
    [
        (BB1(0.7, 1.4), 1 - 2 / (1.4 * 2.7)),
        (BB1(2.5, 3.0), 1 - 2 / (3.0 * 4.5)),
        (BB7(1.5, 0.8), None),
        (BB7(1.2, 3.0), None),
    ],
)
def test_kendalls_tau_closed_forms(C, value):
    res = C.measure("tau", full_output=True)
    assert res.method == "closed"
    if value is not None:
        assert res.value == pytest.approx(value, abs=1e-14)
    assert res.value == pytest.approx(_numeric(C, "tau").value, abs=1e-8)


def test_bb7_tau_falls_back_for_large_theta():
    C = BB7(2.5, 2.0)
    assert C.measure("tau", full_output=True).method == "numeric"


TAIL_CASES = [
    (BB1(0.7, 1.4), 2 ** (-1 / 0.98), 2 - 2 ** (1 / 1.4)),
    (BB1(2.5, 3.0), 2 ** (-1 / 7.5), 2 - 2 ** (1 / 3)),
    (BB2(1.2, 1.5), 1.0, 0.0),
    (BB3(1.5, 0.8), 1.0, 2 - 2 ** (1 / 1.5)),
    (BB3(1.0, 2.0), 2**-0.5, 0.0),
    (BB6(1.5, 2.0), 0.0, 2 - 2 ** (1 / 3)),
    (BB7(1.5, 0.8), 2 ** (-1 / 0.8), 2 - 2 ** (1 / 1.5)),
    (BB8(2.5, 0.7), 0.0, 0.0),
    (BB8(3.0, 1.0), 0.0, 2 - 2 ** (1 / 3)),
    (BB9(2.0, 0.5), 0.0, 0.0),
    (BB10(1.5, 0.6), 0.0, 0.0),
]


@pytest.mark.parametrize(("C", "lam_l", "lam_u"), TAIL_CASES, ids=[repr(t[0]) for t in TAIL_CASES])
def test_tail_coefficients(C, lam_l, lam_u):
    assert C.lambda_L() == pytest.approx(lam_l, abs=1e-14)
    assert C.lambda_U() == pytest.approx(lam_u, abs=1e-14)
    for key, val in (("lambda_l", lam_l), ("lambda_u", lam_u)):
        num = _numeric(C, key)
        # slowly varying generators (lambda_L = 1 of BB2, BB3 with theta > 1)
        # defeat the numerical extrapolation; everything else must agree
        if num.error is not None and num.error < 1e-6:
            assert num.value == pytest.approx(val, abs=1e-5), key


@pytest.mark.parametrize(("cls", "params"), MEMBERS[::3], ids=IDS[::3])
def test_blomqvist_beta_exact(cls, params):
    C = cls(*params)
    res = C.measure("beta", full_output=True)
    assert res.method == "closed"
    assert res.value == pytest.approx(4 * C.cdf(0.5, 0.5) - 1, abs=1e-15)


# ------------------------------------------------------------------ special cases


def _close_cdf(A, B, atol=1e-9):
    np.testing.assert_allclose(A.cdf_vectorized(U, V), _cdf(B, U, V), atol=atol)


def test_special_cases():
    _close_cdf(BB1(2.0, 1.0), cp.Clayton(2))
    _close_cdf(BB1(1e-9, 2.0), cp.GumbelHougaard(2), atol=1e-7)
    _close_cdf(BB2(1.5, 1e-9), cp.Clayton(1.5), atol=1e-7)
    _close_cdf(BB3(1.0, 2.0), cp.Clayton(2))
    _close_cdf(BB6(1.0, 2.0), cp.GumbelHougaard(2))
    _close_cdf(BB6(3.0, 1.0), cp.Joe(3))
    _close_cdf(BB7(1.0, 2.0), cp.Clayton(2))
    _close_cdf(BB7(3.0, 1e-9), cp.Joe(3), atol=1e-7)
    _close_cdf(BB8(3.0, 1.0), cp.Joe(3))
    np.testing.assert_allclose(BB8(1.0, 0.6).cdf(U, V), U * V, atol=1e-12)
    np.testing.assert_allclose(BB9(1.0, 0.6).cdf(U, V), U * V, atol=1e-12)
    _close_cdf(BB9(2.0, 1e7), cp.GumbelHougaard(2), atol=1e-6)
    np.testing.assert_allclose(BB10(2.0, 0.0).cdf(U, V), U * V, atol=1e-12)
    _close_cdf(BB10(1.0, 0.5), cp.AliMikhailHaq(0.5))


# ------------------------------------------------------------------ API


def test_symbolic_api():
    C = BB1()
    assert isinstance(C, LTArchimedeanCopula)
    assert isinstance(C.cdf().func, sp.Expr)
    assert sp.simplify(C.kendalls_tau() - (1 - 2 / (C.delta * (C.theta + 2)))) == 0
    tau = BB7().kendalls_tau()
    assert isinstance(tau, sp.Expr)
    with pytest.raises(ValueError):
        C.rvs(5)


def test_parameter_handling():
    C = BB1(theta=2, delta=1.5)
    assert C.params == []
    assert C.cdf(0.3, 0.4) == C.cdf(u=0.3, v=0.4)
    assert isinstance(C.cdf(0.3, 0.4), float)
    np.testing.assert_allclose(
        C.cdf(np.array([[0.3, 0.4], [0.5, 0.6]])), C.cdf([0.3, 0.5], [0.4, 0.6])
    )
    D = BB1(delta=2)
    assert [str(p) for p in D.params] == ["theta"]
    assert D(theta=3).kendalls_tau() == pytest.approx(1 - 2 / (2 * 5))
    with pytest.raises(ValueError):
        BB1(-1, 2)
    with pytest.raises(ValueError):
        BB1(2, 0.5)
    with pytest.raises(ValueError):
        BB8(2, 1.5)
    with pytest.raises(ValueError):
        BB1(2, 3)(delta=0.2)


def test_from_measure_and_curve():
    C = BB1(delta=1.5).from_measure("tau", 0.6)
    assert C.kendalls_tau() == pytest.approx(0.6, abs=1e-10)
    curve = BB7(theta=1.5).measure_curve(["tau", "beta"], values=[0.5, 1.0, 2.0])
    assert np.all(np.diff(curve["tau"]) > 0)


def test_rvs_reproducible():
    C = BB6(2.0, 1.5)
    np.testing.assert_array_equal(C.rvs(10, random_state=3), C.rvs(10, random_state=3))
    assert C.rvs(0).shape == (0, 2)


def test_constructions_accept_bb_families():
    from copul.family.constructions import rotate

    R = rotate(BB7(1.5, 0.8), 180)
    assert R.lambda_L() == pytest.approx(2 - 2 ** (1 / 1.5), abs=1e-14)
    assert R.kendalls_tau() == pytest.approx(BB7(1.5, 0.8).kendalls_tau(), abs=1e-14)


# ------------------------------------------------------------------ frailty samplers


def test_frailty_laplace_transforms():
    rng = np.random.default_rng(0)
    n = 200_000
    se = 4.0 / np.sqrt(n)
    for alpha in (0.3, 0.7):
        S = np.exp(log_positive_stable_rv(alpha, n, rng))
        N = sibuya_rv(alpha, n, rng)
        for s in (0.5, 2.0):
            assert np.mean(np.exp(-s * S)) == pytest.approx(np.exp(-(s**alpha)), abs=se)
            assert np.mean(np.exp(-s * N)) == pytest.approx(1 - (1 - np.exp(-s)) ** alpha, abs=se)
    V = np.exp(log_tilted_stable_rv(0.5, 4.0, n, rng))
    for s in (0.5, 2.0):
        assert np.mean(np.exp(-s * V)) == pytest.approx(np.exp(-((4 + s) ** 0.5 - 2)), abs=se)
    G = np.exp(log_gamma_rv(np.full(n, 0.05), rng))
    assert np.mean(np.exp(-G)) == pytest.approx(2**-0.05, abs=se)


def test_registered_in_family_list():
    from copul.family_list import Families, families

    for name in ("BB1", "BB2", "BB3", "BB6", "BB7", "BB8", "BB9", "BB10"):
        assert name in families
        assert Families[name].cls.__name__ == name
    C = Families.create("BB7", 1.5, 2.0)
    assert isinstance(C, BB7) and C.params == []
