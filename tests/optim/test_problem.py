"""Tests for copul.optim.CheckerboardProblem (convex path and CCP)."""

import sys

import numpy as np
import pytest

cp = pytest.importorskip("cvxpy")

from copul.checkerboard.biv_check_pi import BivCheckPi
from copul.optim import (
    CheckerboardProblem,
    NonConvexError,
    measure_form,
    shape_violation,
)
from copul.optim._backend import require_cvxpy
from copul.regions.catalog import rho_max_given_xi


@pytest.mark.parametrize("n", [4, 7, 12])
def test_rho_extremes_pi(n):
    prob = CheckerboardProblem(n=n)
    assert prob.maximize("rho").value == pytest.approx(1 - 1 / n**2, abs=1e-9)
    assert prob.minimize("rho").value == pytest.approx(-(1 - 1 / n**2), abs=1e-9)


def test_rho_extremes_min_w():
    assert CheckerboardProblem(n=5, kind="min").maximize("rho").value == pytest.approx(
        1.0, abs=1e-9
    )
    assert CheckerboardProblem(n=5, kind="w").minimize("rho").value == pytest.approx(-1.0, abs=1e-9)


def test_result_object_consistency():
    prob = CheckerboardProblem(n=8)
    res = prob.maximize(prob.expr("rho") - 0.5 * prob.expr("xi"))
    assert res.status in ("optimal", "optimal_inaccurate")
    assert set(res.values) == {"xi", "rho", "tau", "footrule", "gamma", "beta", "nu"}
    c = res.copula
    assert isinstance(c, BivCheckPi)
    assert c.spearmans_rho() == pytest.approx(res["rho"], abs=1e-10)
    assert c.chatterjees_xi() == pytest.approx(res["Chatterjee"], abs=1e-10)
    assert res.value == pytest.approx(res["rho"] - 0.5 * res["xi"], abs=1e-6)
    assert res.residual < 1e-6
    np.testing.assert_allclose(res.P.sum(axis=0), 1 / 8, atol=1e-12)
    np.testing.assert_allclose(res.P.sum(axis=1), 1 / 8, atol=1e-12)


def test_equivalent_objective_spellings():
    prob = CheckerboardProblem(n=6)
    a = prob.maximize(prob.form("rho") - 0.3 * prob.form("xi")).value
    b = prob.maximize({"rho": 1.0, "xi": -0.3}).value
    c = prob.maximize(prob.expr("rho") - 0.3 * prob.expr("xi")).value
    assert a == pytest.approx(b, abs=1e-6) and a == pytest.approx(c, abs=1e-6)


def test_scalarised_values_increase_with_n_and_stay_below_exact_supremum():
    mu = 0.5
    xs = np.linspace(0, 1, 200001)
    exact = float(np.max(rho_max_given_xi(xs) - mu * xs))
    vals = []
    for n in (6, 12, 24):
        prob = CheckerboardProblem(n=n)
        res = prob.maximize(prob.form("rho") - mu * prob.form("xi"))
        vals.append(res.value)
        assert res["rho"] <= rho_max_given_xi(res["xi"]) + 1e-9
    assert vals[0] < vals[1] < vals[2] <= exact + 1e-9
    assert exact - vals[2] < 5e-3


def test_subject_to_variants():
    prob = CheckerboardProblem(n=8)
    r1 = prob.maximize("rho", subject_to={"xi": ("<=", 0.3)})
    r2 = prob.maximize("rho", subject_to=("xi", "<=", 0.3))
    r3 = prob.maximize("rho", subject_to=[("xi", "<=", 0.3)])
    assert r1.value == pytest.approx(r2.value, abs=1e-7)
    assert r1.value == pytest.approx(r3.value, abs=1e-7)
    assert r1["xi"] <= 0.3 + 1e-6
    assert r1["rho"] <= 0.7 + 1e-9  # exact bound at xi = 3/10
    r4 = prob.minimize("xi", subject_to={"beta": 0.5})
    assert r4["beta"] == pytest.approx(0.5, abs=1e-7)
    r5 = prob.maximize("nu", subject_to=[prob.P[0, 0] >= 0.05])
    assert r5.P[0, 0] >= 0.05 - 1e-7
    prob.add_constraint("footrule", ">=", 0.2).add_constraint("si")
    r6 = prob.minimize("rho")
    assert r6["footrule"] >= 0.2 - 1e-7
    assert shape_violation(r6.P, "si") < 1e-8
    assert set(prob.check(r6.P)) == {"si", "footrule >= 0.2"}
    assert "si" in repr(prob)


def test_rectangular_grid():
    prob = CheckerboardProblem(n=6, m=4)
    res = prob.maximize("rho", subject_to={"xi": ("<=", 0.4)})
    c = res.copula
    assert res.P.shape == (4, 6)
    assert c.spearmans_rho() == pytest.approx(res["rho"], abs=1e-10)
    assert c.chatterjees_xi() == pytest.approx(res["xi"], abs=1e-10)
    with pytest.raises(ValueError):
        prob.expr("footrule")


def test_nonconvex_requests_raise():
    prob = CheckerboardProblem(n=5)
    with pytest.raises(NonConvexError):
        prob.maximize("xi")
    with pytest.raises(NonConvexError):
        prob.maximize("rho", subject_to={"xi": (">=", 0.3)})
    with pytest.raises(NonConvexError):
        prob.minimize("rho", subject_to={"xi": ("==", 0.3)})
    with pytest.raises(NonConvexError):
        prob.expr("tau")
    with pytest.raises(NonConvexError):
        prob.maximize("tau")
    with pytest.raises(NonConvexError):
        prob.maximize(prob.expr("xi"))
    with pytest.raises(ValueError):
        prob.maximize("rho", method="magic")
    with pytest.raises(ValueError):
        prob.add_constraint("rho")


def test_ccp_maximise_xi_reaches_permutation_value():
    n = 6
    prob = CheckerboardProblem(n=n)
    res = prob.maximize("xi", method="ccp", n_starts=2)
    # the maximum of xi over n x n BivCheckPi is attained at permutation matrices
    xi_perm = measure_form("xi", n).value(np.eye(n) / n)
    assert res.value == pytest.approx(xi_perm, abs=1e-6)
    assert res.method == "ccp" and res.iterations >= 1


def test_ccp_nonconvex_constraint_feasible_and_inside_region():
    prob = CheckerboardProblem(n=8)
    res = prob.minimize("rho", subject_to={"xi": (">=", 0.5)}, method="ccp")
    assert res["xi"] >= 0.5 - 1e-6
    assert res["rho"] >= -rho_max_given_xi(res["xi"]) - 1e-9
    assert res.history[-1] <= res.history[1] + 1e-9  # minimisation made progress


def test_ccp_indefinite_tau_objective():
    """Kendall's tau (indefinite quadratic) can be used as a CCP objective."""
    prob = CheckerboardProblem(n=6)
    res = prob.maximize("tau", subject_to={"xi": ("<=", 0.3)}, method="ccp", n_starts=2)
    assert res["xi"] <= 0.3 + 1e-6
    assert res["tau"] > 0.3
    assert res.copula.kendalls_tau() == pytest.approx(res["tau"], abs=1e-10)


def test_missing_cvxpy_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "cvxpy", None)
    with pytest.raises(ImportError, match=r"pip install copul\[optim\]"):
        require_cvxpy()
    with pytest.raises(ImportError, match="cvxpy"):
        CheckerboardProblem(n=3)
