"""Dispatch semantics of the measure methods (method=..., fallbacks, symbolic
route, deprecations)."""

import time
import warnings

import numpy as np
import pytest
import sympy as sp

import copul as cp
from copul.measures import MeasureResult, compute

# ---------------------------------------------------------------------------
# Previously failing / slow cases: must now return correct floats quickly
# ---------------------------------------------------------------------------

PROBLEM_CASES = [
    # (id, factory, method name, reference value or None, reference tolerance)
    ("clayton-rho", lambda: cp.Clayton(2), "spearmans_rho", 0.6822338332806566, 1e-8),
    ("joe-tau", lambda: cp.Joe(2), "kendalls_tau", 2 - np.pi**2 / 6, 1e-9),
    ("joe-nu", lambda: cp.Joe(2), "blests_nu", None, None),
    ("gh-rho", lambda: cp.GumbelHougaard(2), "spearmans_rho", 0.6822338332806566, 1e-8),
    ("joe-rho", lambda: cp.Joe(2), "spearmans_rho", None, None),
    ("clayton-xi", lambda: cp.Clayton(2), "chatterjees_xi", None, None),
    ("frank-xi", lambda: cp.Frank(2), "chatterjees_xi", None, None),
    ("gh-xi", lambda: cp.GumbelHougaard(2), "chatterjees_xi", None, None),
    ("plackett-xi", lambda: cp.Plackett(3), "chatterjees_xi", None, None),
    ("joe-xi", lambda: cp.Joe(2), "chatterjees_xi", None, None),
    ("galambos-xi", lambda: cp.Galambos(1), "chatterjees_xi", None, None),
    ("clayton-nu", lambda: cp.Clayton(2), "blests_nu", None, None),
    ("plackett-tau", lambda: cp.Plackett(3), "kendalls_tau", None, None),
    ("galambos-tau", lambda: cp.Galambos(1), "kendalls_tau", None, None),
    ("galambos-nu", lambda: cp.Galambos(1), "blests_nu", None, None),
    ("clayton-gamma", lambda: cp.Clayton(2), "ginis_gamma", None, None),
    ("gh-gamma", lambda: cp.GumbelHougaard(2), "ginis_gamma", None, None),
]


@pytest.mark.parametrize(
    "cid, factory, name, ref, tol", PROBLEM_CASES, ids=[c[0] for c in PROBLEM_CASES]
)
def test_previously_failing_cases(cid, factory, name, ref, tol):
    cop = factory()
    t0 = time.perf_counter()
    val = getattr(cop, name)()
    elapsed = time.perf_counter() - t0
    assert type(val) is float
    assert np.isfinite(val)
    assert elapsed < 3.0, f"{cid} took {elapsed:.2f}s"
    if ref is not None:
        assert val == pytest.approx(ref, abs=tol)
    # auto and numeric agree
    key_val = getattr(factory(), name)(method="numeric")
    assert val == pytest.approx(key_val, abs=1e-7)


def test_gh_archimedean_and_ev_agree():
    a = cp.GumbelHougaard(2).spearmans_rho()
    b = cp.GumbelHougaardEV(2).spearmans_rho()
    assert a == pytest.approx(b, abs=1e-9)


def test_mc_estimates():
    c = cp.Clayton(2)
    rho = c.spearmans_rho()
    est = c.spearmans_rho(method="mc", n_samples=20_000, random_state=3)
    assert est == pytest.approx(rho, abs=0.03)
    r = compute(c, "tau", method="mc", n_samples=20_000, random_state=3, full_output=True)
    assert r.method == "mc" and r.value == pytest.approx(0.5, abs=0.03)


# ---------------------------------------------------------------------------
# Symbolic behaviour with free parameters
# ---------------------------------------------------------------------------


def test_free_parameters_stay_symbolic():
    fgm = cp.FarlieGumbelMorgenstern()
    theta = sp.Symbol("theta")
    rho = fgm.spearmans_rho()
    assert isinstance(rho, sp.Basic)
    assert sp.simplify(rho - fgm.theta / 3) == 0
    assert sp.simplify(fgm.chatterjees_xi() - fgm.theta**2 / 15) == 0
    assert str(fgm.kendalls_tau()) in ("2*theta/9",)
    del theta


def test_symbolic_base_route_with_free_parameter():
    fgm = cp.FarlieGumbelMorgenstern()
    # generic SymPy integration (base class) instead of the family formula
    rho = fgm.spearmans_rho(method="symbolic")
    assert sp.simplify(rho - fgm.theta / 3) == 0
    beta = fgm.blomqvists_beta(method="symbolic")
    assert sp.simplify(beta - fgm.theta / 4) == 0


def test_symbolic_route_for_fully_specified_copula():
    val = cp.FarlieGumbelMorgenstern(0.5).spearmans_rho(method="symbolic")
    assert float(val) == pytest.approx(0.5 / 3)


def test_numeric_with_free_parameters_raises():
    with pytest.raises(ValueError, match="free parameters"):
        cp.Clayton().spearmans_rho(method="numeric")


def test_parameters_passed_to_measure_method():
    c = cp.FarlieGumbelMorgenstern()
    assert c.spearmans_rho(0.6) == pytest.approx(0.2)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        cp.Clayton(2).spearmans_rho(method="magic")


# ---------------------------------------------------------------------------
# closed / auto / fallback
# ---------------------------------------------------------------------------


def test_closed_requires_closed_form():
    assert cp.Clayton(2).kendalls_tau(method="closed") == pytest.approx(0.5)
    with pytest.raises(NotImplementedError):
        cp.Clayton(2).spearmans_rho(method="closed")


def test_full_output_reports_method():
    r = compute(cp.Clayton(2), "tau", full_output=True)
    assert isinstance(r, MeasureResult) and r.method == "closed" and r.error == 0.0
    r = compute(cp.Clayton(2), "rho", full_output=True)
    assert r.method == "numeric" and r.error < 1e-7
    out = compute(cp.Clayton(2), ["tau", "spearman", "Chatterjee"])
    assert set(out) == {"tau", "rho", "xi"}


def test_failing_closed_form_falls_back(monkeypatch):
    from copul.family.other.plackett import Plackett

    def broken(self, *args, **kwargs):
        raise RuntimeError("boom")

    # a subclass override is wrapped at class creation
    class BrokenPlackett(Plackett):
        def spearmans_rho(self, *args, **kwargs):
            return broken(self)

    c = BrokenPlackett(3)
    r = compute(c, "rho", full_output=True)
    assert r.method == "numeric" and "boom" in r.info["fallback_reason"]
    assert r.value == pytest.approx(Plackett(3).spearmans_rho(), abs=1e-9)
    with pytest.raises(RuntimeError):
        c.spearmans_rho(method="closed")


def test_closed_form_returning_integral_falls_back():
    from copul.family.other.plackett import Plackett

    u = sp.Symbol("u")

    class IntegralPlackett(Plackett):
        def kendalls_tau(self, *args, **kwargs):
            return sp.Integral(u, (u, 0, 1)) * 0 + sp.Integral(
                sp.exp(-(u**2)) * sp.sin(u) ** 7, (u, 0, 1)
            )

    val = IntegralPlackett(3).kendalls_tau()
    assert type(val) is float
    assert val == pytest.approx(Plackett(3).kendalls_tau(method="numeric"), abs=1e-9)


def test_nan_closed_form_falls_back():
    from copul.family.other.plackett import Plackett

    class NanPlackett(Plackett):
        def blests_nu(self, *args, **kwargs):
            return float("nan")

    assert np.isfinite(NanPlackett(3).blests_nu())


def test_nested_closed_forms_do_not_fall_back_individually():
    # BivCheckMin.spearmans_footrule = BivCheckPi.spearmans_footrule(self) + ...;
    # for rectangular matrices the Pi part warns and returns nan, which must
    # make the *outer* call fall back (not the inner one with the wrong copula)
    c = cp.BivCheckMin(np.ones((2, 3)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        val = c.spearmans_footrule()
    assert val == pytest.approx(c.spearmans_footrule(method="numeric"), abs=1e-10)


def test_xi_condition_on_y():
    c = cp.MarshallOlkin(0.8, 0.3)  # not exchangeable
    x1 = c.chatterjees_xi()
    x2 = c.chatterjees_xi(condition_on_y=True)
    assert x1 != pytest.approx(x2, abs=1e-3)
    assert x2 == pytest.approx(compute(c, "xi_2"), abs=1e-12)
    # transposed copula: xi_2(C) == xi(C^T)
    t = cp.MarshallOlkin(0.3, 0.8)
    assert x2 == pytest.approx(t.chatterjees_xi(), abs=1e-6)


def test_lp_distance_and_aliases():
    c = cp.FarlieGumbelMorgenstern(0.6)
    assert c.lp_distance(2) == pytest.approx(c.hoeffdings_d(), abs=1e-12)
    assert c.lp_distance(1, method="numeric") == pytest.approx(c.schweizer_wolff_sigma(), abs=1e-9)
    assert c.lp_distance(p=2.5) == pytest.approx(c.lp_distance(p=2.5, method="numeric"), abs=1e-8)
    assert c.hoeffdings_phi_square() == pytest.approx(c.hoeffdings_d())


def test_generic_measure_accessors():
    c = cp.Clayton(2)
    assert c.measure("kendall") == pytest.approx(0.5)
    d = c.measures()
    assert set(d) == {"xi", "rho", "tau", "footrule", "gamma", "beta", "nu"}
    assert all(type(v) is float for v in d.values())
    assert c.measures(["rho"]) == {"rho": pytest.approx(d["rho"])}


def test_new_measures():
    c = cp.Gaussian(0.5)
    assert 0 < c.uniform_distance() < 1
    assert c.blum_kiefer_rosenblatt() > 0
    mi = c.mutual_information()
    assert mi == pytest.approx(-0.5 * np.log(1 - 0.25), abs=1e-7)
    assert cp.BivIndependenceCopula().mutual_information(method="numeric") == pytest.approx(
        0, abs=1e-12
    )


# ---------------------------------------------------------------------------
# deprecations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "old, new, args",
    [
        ("gini_gamma", "ginis_gamma", ()),
        ("spearman_footrule", "spearmans_footrule", ()),
        ("lp_concordance", "lp_distance", (2,)),
    ],
)
def test_deprecated_aliases(old, new, args):
    c = cp.FarlieGumbelMorgenstern(0.4)
    with pytest.warns(DeprecationWarning):
        v_old = getattr(c, old)(*args)
    assert v_old == pytest.approx(getattr(c, new)(*args))
    d = cp.Clayton(2)
    with pytest.warns(DeprecationWarning):
        assert getattr(d, old)(*args) == pytest.approx(getattr(d, new)(*args))


def test_legacy_override_name_in_user_subclass():
    from copul.family.other.plackett import Plackett

    class Legacy(Plackett):
        def gini_gamma(self, *args, **kwargs):  # old name
            return 0.123

    c = Legacy(3)
    assert c.ginis_gamma() == 0.123
    with pytest.warns(DeprecationWarning):
        assert c.gini_gamma() == 0.123


def test_no_deprecation_warning_from_new_names():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        cp.Clayton(2).ginis_gamma()
        cp.Clayton(2).spearmans_footrule()
