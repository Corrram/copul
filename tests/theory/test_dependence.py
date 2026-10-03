"""Tests for copul.theory.dependence (dependence concepts and their hierarchy)."""

from __future__ import annotations

import numpy as np
import pytest

import copul as cp
from copul.family.constructions import mixture, reflect, rotate, survival, transpose
from copul.theory import dependence as dep
from copul.theory.dependence import (
    IMPLICATIONS,
    PROPERTIES,
    check_property,
    dependence_profile,
    exact_facts,
    implication_violations,
    resolve_property,
)
from tests.properties.representatives import IDS, instance

POSITIVE = [k for k, p in PROPERTIES.items() if p.sign > 0]
NEGATIVE = [k for k, p in PROPERTIES.items() if p.sign < 0]


# ---------------------------------------------------------------------------
# catalogue
# ---------------------------------------------------------------------------


def test_catalogue_and_aliases():
    assert len(PROPERTIES) == 18
    assert resolve_property("si") == "SI(V|U)"
    assert resolve_property("SI", i=2) == "SI(U|V)"
    assert resolve_property("cis(u|v)") == "SI(U|V)"
    assert resolve_property("TP2_cdf") == "LCSD"
    assert resolve_property("tp2_survival") == "RCSI"
    assert resolve_property("PLOD") == "PQD"
    assert resolve_property("plr") == "TP2"
    with pytest.raises(KeyError):
        resolve_property("XYZ")
    with pytest.raises(ValueError):
        resolve_property("SI(V|U)", i=2)


def test_implications_are_between_known_properties_and_acyclic():
    for p, q in IMPLICATIONS:
        assert p in PROPERTIES and q in PROPERTIES
        # weak-to-strong evaluation order: implied properties come first
        assert dep._ORDER.index(q) < dep._ORDER.index(p)
    assert implication_violations({"SI(V|U)": True, "LTD(V|U)": False}) == [("SI(V|U)", "LTD(V|U)")]


# ---------------------------------------------------------------------------
# exact characterizations agree with dense grid checks
# ---------------------------------------------------------------------------

AGREEMENT_CASES = {
    "Gaussian(0.6)": lambda: cp.Gaussian(0.6),
    "Gaussian(-0.4)": lambda: cp.Gaussian(-0.4),
    "FGM(0.7)": lambda: cp.FarlieGumbelMorgenstern(0.7),
    "FGM(-0.9)": lambda: cp.FarlieGumbelMorgenstern(-0.9),
    "StudentT(-0.5,3)": lambda: cp.StudentT(-0.5, 3),
    "Clayton(3)": lambda: cp.Clayton(3),
    "Clayton(-0.3)": lambda: cp.Clayton(-0.3),
    "Clayton(-0.7)": lambda: cp.Clayton(-0.7),
    "Frank(5)": lambda: cp.Frank(5),
    "Frank(-6)": lambda: cp.Frank(-6),
    "GumbelHougaard(1.7)": lambda: cp.GumbelHougaard(1.7),
    "Joe(2.5)": lambda: cp.Joe(2.5),
    "AMH(0.8)": lambda: cp.AliMikhailHaq(0.8),
    "AMH(-0.9)": lambda: cp.AliMikhailHaq(-0.9),
    "Nelsen2(2.5)": lambda: cp.Nelsen2(2.5),
    "GumbelBarnett(0.7)": lambda: cp.GumbelBarnett(0.7),
    "Nelsen10(0.8)": lambda: cp.Nelsen10(0.8),
    "Nelsen12(1.5)": lambda: cp.Nelsen12(1.5),
    "Nelsen16(0.5)": lambda: cp.Nelsen16(0.5),
    "Nelsen17(-2)": lambda: cp.Nelsen17(-2),
    "Nelsen20(0.5)": lambda: cp.Nelsen20(0.5),
    "Galambos(1.3)": lambda: cp.Galambos(1.3),
    "HueslerReiss(1.2)": lambda: cp.HueslerReiss(1.2),
    "Tawn(0.3,0.9,2)": lambda: cp.Tawn(0.3, 0.9, 2),
    "MarshallOlkin(0.2,0.8)": lambda: cp.MarshallOlkin(0.2, 0.8),
    "BB1(0.5,1.5)": lambda: cp.BB1(0.5, 1.5),
    "BB7(1.4,0.6)": lambda: cp.BB7(1.4, 0.6),
    "rot90(Clayton(2))": lambda: rotate(cp.Clayton(2), 90),
    "survival(MarshallOlkin)": lambda: survival(cp.MarshallOlkin(0.2, 0.8)),
    "reflect_u(Tawn)": lambda: reflect(cp.Tawn(0.3, 0.9, 2), "u"),
    "transpose(MarshallOlkin)": lambda: transpose(cp.MarshallOlkin(0.3, 0.9)),
    "mixture(Clayton,Gumbel)": lambda: mixture([cp.Clayton(2), cp.GumbelHougaard(2)], [0.4, 0.6]),
}


@pytest.mark.parametrize("name", list(AGREEMENT_CASES))
def test_exact_characterizations_agree_with_grid(name):
    C = AGREEMENT_CASES[name]()
    facts = exact_facts(C)
    assert facts, "expected at least one exact fact"
    grid = dependence_profile(C, method="grid", propagate=False, n_grid=49)
    for key, fact in facts.items():
        assert fact.method in ("exact", "symbolic")
        assert grid[key].holds == fact.holds, (key, fact, grid[key], grid[key].where)


@pytest.mark.parametrize("seed", range(4))
def test_checkerboard_exact_facts_agree_with_grid(seed):
    rng = np.random.default_rng(seed)
    m = int(rng.integers(2, 5))
    if seed % 2:
        x = np.arange(m)[:, None] / (m - 1)
        P = np.exp(-rng.uniform(1, 6) * (x - x.T) ** 2)  # TP2-like (Gaussian kernel)
    else:
        P = rng.random((m, m)) + 0.05
    for _ in range(500):
        P = P / P.sum(1, keepdims=True)
        P = P / P.sum(0, keepdims=True)
    C = cp.BivCheckPi(P)
    facts = exact_facts(C)
    assert {"LCSD", "RCSI", "TP2", "RR2"} <= set(facts)
    grid = dependence_profile(C, method="grid", propagate=False, n_grid=49)
    for key, fact in facts.items():
        assert grid[key].holds == fact.holds, (key, P)


def test_grid_check_is_numerically_sound_on_nelsen19_exception():
    # The symbolic (generator) characterization is right; the family's double
    # precision cdf underflows to 0 for u < theta/709, which a grid check sees.
    C = cp.Nelsen19(0.5)
    assert exact_facts(C)["TP2"].holds
    assert exact_facts(C)["TP2"].method == "symbolic"


# ---------------------------------------------------------------------------
# hierarchy never violated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", IDS)
def test_profiles_never_violate_the_hierarchy(name):
    C = instance(name)
    raw = dependence_profile(C, propagate=False, n_grid=33)
    assert raw.consistent, raw.violations
    prop = dependence_profile(C, n_grid=33)
    assert prop.consistent
    # propagation never contradicts an independent check
    for key in raw.results:
        if raw[key].method in ("exact", "symbolic"):
            assert prop[key].holds == raw[key].holds


def test_exact_facts_are_closed_under_implications():
    for C in (cp.Clayton(2), cp.Frank(-3), cp.GumbelHougaardEV(2), cp.StudentT(-0.4, 4)):
        facts = exact_facts(C)
        assert not implication_violations(facts)
        for p, q in IMPLICATIONS:
            if p in facts and facts[p].holds:
                assert q in facts and facts[q].holds
            if q in facts and not facts[q].holds:
                assert p in facts and not facts[p].holds


# ---------------------------------------------------------------------------
# specific published facts
# ---------------------------------------------------------------------------


def test_frechet_bounds_and_independence():
    M, W, Pi = cp.UpperFrechet(), cp.LowerFrechet(), cp.BivIndependenceCopula()
    pM, pW, pPi = dependence_profile(M), dependence_profile(W), dependence_profile(Pi)
    assert all(pPi.as_dict().values())
    assert all(pM[k].holds for k in POSITIVE if k != "TP2") and not pM["TP2"].holds
    assert not any(pM[k].holds for k in NEGATIVE)
    assert all(pW[k].holds for k in NEGATIVE if k != "RR2") and not pW["RR2"].holds
    assert not any(pW[k].holds for k in POSITIVE)


def test_gaussian_and_fgm_sign_characterizations():
    for rho in (-0.8, -0.2, 0.3, 0.9):
        p = dependence_profile(cp.Gaussian(rho))
        assert p["PQD"].holds == (rho > 0) and p["NQD"].holds == (rho < 0)
        assert p["TP2"].holds == (rho > 0) and p["RR2"].holds == (rho < 0)
        assert p["TP2" if rho > 0 else "RR2"].method == "exact"
    for th in (-1.0, -0.5, 0.5, 1.0):
        p = dependence_profile(cp.FarlieGumbelMorgenstern(th))
        assert p["SI(V|U)"].holds == (th > 0) and p["SD(U|V)"].holds == (th < 0)


def test_student_t_is_not_pqd_for_small_positive_rho():
    # tau > 0 but C - Pi changes sign in the off-diagonal corners
    r = check_property(cp.StudentT(0.5, 2), "PQD")
    assert r.method == "grid" and not r.holds
    u, v = r.where["u"], r.where["v"]
    assert (u - 0.5) * (v - 0.5) < 0
    assert r.worst_violation > 1e-4
    neg = check_property(cp.StudentT(-0.5, 2), "SI")
    assert not neg.holds and neg.method == "exact"


def test_extreme_value_copulas_are_si_and_lcsd():
    for C in (cp.Galambos(0.8), cp.HueslerReiss(1.5), cp.CuadrasAuge(0.4)):
        p = dependence_profile(C)
        for key in ("SI(V|U)", "SI(U|V)", "LCSD", "LTD(V|U)", "RTI(U|V)", "PQD"):
            assert p[key].holds and p[key].method == "exact"
        assert not any(p[k].holds for k in NEGATIVE)


def test_ev_counterexample_rcsi_fails_on_grid():
    # min(u, v, (uv)^0.75): Pickands max(w, 1-w, 0.75); LCSD but not RCSI
    C = cp.from_cdf("min(min(u, v), (u*v)**0.75)")
    assert check_property(C, "LCSD").holds
    r = check_property(C, "RCSI")
    assert not r.holds and r.method == "grid"


def test_clayton_negative_parameter_sd_but_rr2_only_above_minus_half():
    # psi(s) = (1 + theta s)^(-1/theta): log(-psi') is concave for theta < 0, and
    # log psi'' is concave iff theta >= -1/2
    for th, rr2 in ((-0.3, True), (-0.7, False)):
        C = cp.Clayton(th)
        sd = check_property(C, "SD")
        assert sd.holds and sd.method == "symbolic"
        assert check_property(C, "RR2").holds is rr2
        assert check_property(C, "RR2", method="grid").holds is rr2
        assert not check_property(C, "PQD").holds  # non-strict generator


def test_frank_reflection_identity_matches_profiles():
    # C_{-theta}(u, v) = u - C_theta(u, 1 - v) for the Frank family
    P, N = cp.Frank(3), cp.Frank(-3)
    pts = np.random.default_rng(0).random((50, 2))
    lhs = N.cdf(pts[:, 0], pts[:, 1])
    rhs = pts[:, 0] - P.cdf(pts[:, 0], 1 - pts[:, 1])
    assert np.allclose(lhs, rhs, atol=1e-12)
    refl = reflect(P, "v")
    assert dependence_profile(refl).as_dict() == dependence_profile(N).as_dict()


def test_transfer_of_properties_through_symmetries():
    base = cp.MarshallOlkin(0.2, 0.8)
    T = survival(base)
    assert T.dependence_property("RTI", i=1).holds  # LTD of the base
    assert dep._transfer_key("LTD(V|U)", False, True, True) == "RTI(V|U)"
    assert dep._transfer_key("SI(V|U)", True, False, False) == "SI(U|V)"
    assert dep._transfer_key("LTD(V|U)", False, True, False) == "RTD(V|U)"
    assert dep._transfer_key("LTD(V|U)", False, False, True) == "LTI(V|U)"
    assert dep._transfer_key("LCSD", False, True, False) is None
    assert dep._transfer_key("TP2", False, False, True) == "RR2"


def test_mixture_inherits_linear_properties_only():
    C = mixture([cp.Clayton(2), cp.Gaussian(0.5)], [0.5, 0.5])
    facts = exact_facts(C)
    assert facts["SI(V|U)"].holds and facts["RTI(U|V)"].holds
    assert "LCSD" not in facts and "TP2" not in facts


# ---------------------------------------------------------------------------
# results and grid machinery
# ---------------------------------------------------------------------------


def test_property_result_api():
    r = check_property(cp.Plackett(0.3), "PQD")
    assert isinstance(r, dep.PropertyResult)
    assert r.method == "grid" and not r and r.worst_violation > 0
    assert set(r.where) == {"u", "v"}
    assert "tol" in r.info and r.info["n_evals"] > 0
    assert "PQD" in repr(r)
    with pytest.raises(ValueError):
        check_property(cp.Plackett(0.3), "PQD", method="exact")
    with pytest.raises(ValueError):
        check_property(cp.Clayton(), "PQD")  # free parameter


@pytest.mark.parametrize("seed", range(5))
def test_coarse_refined_grid_agrees_with_exact_checkerboard_algorithms(seed):
    rng = np.random.default_rng(100 + seed)
    m = int(rng.integers(3, 6))
    x = np.arange(m)[:, None] / (m - 1)
    P = np.exp(-rng.uniform(0.5, 4) * (x - x.T) ** 2) + rng.uniform(0, 0.3, (m, m))
    for _ in range(1000):
        P = P / P.sum(1, keepdims=True)
        P = P / P.sum(0, keepdims=True)
    C = cp.BivCheckPi(P)
    facts = exact_facts(C)
    for key in ("SI(V|U)", "LTD(V|U)", "RTI(U|V)", "PQD", "LCSD", "TP2"):
        assert check_property(C, key, method="grid", n_grid=9).holds == facts[key].holds


def test_non_copula_matrix_is_reported_as_inconsistent(caplog):
    # margins not uniform: the exact algorithms (which assume a copula) disagree
    P = np.array([[0.30, 0.03, 0.0], [0.03, 0.27, 0.033], [0.0, 0.033, 0.3]])
    exact_facts(cp.BivCheckPi(P))
    assert "inconsistent exact facts" in caplog.text


def test_profile_object():
    p = dependence_profile(cp.Clayton(2))
    assert p.consistent and p.holds("TP2") and "RCSI" in p
    df = p.to_frame()
    assert list(df.index) == list(dep._ORDER)
    assert "DependenceProfile" in repr(p)
    sub = dependence_profile(cp.Frank(2), ["PQD", "SI"])
    assert list(sub.results) == ["PQD", "SI(V|U)"]


# ---------------------------------------------------------------------------
# copula methods (BivCoreCopula wiring)
# ---------------------------------------------------------------------------


def test_copula_methods_use_the_engine():
    C = cp.MarshallOlkin(0.2, 0.8)
    assert C.is_si(1) and C.is_si(2) and not C.is_sd()
    assert C.is_ltd(i=2) and C.is_rti(i=2) and not C.is_lti() and not C.is_rtd()
    assert C.is_lcsd() and C.is_pqd() and not C.is_nqd()
    assert C.dependence_property("SI", i=2).method == "exact"
    assert C.dependence_profile().consistent
    N = cp.Frank(-2)
    assert N.is_sd(2) and N.is_cds() and N.is_lti() and N.is_rtd() and N.is_rr2()
    assert not N.is_tp2() and N.is_nqd()
    assert cp.Gaussian(0.3).is_tp2() and cp.Gaussian(0.3).is_cis(2)


def test_checkerboard_methods_condition_on_either_variable():
    P = np.array([[0.2, 0.1, 0.0333], [0.0, 0.1333, 0.2], [0.1333, 0.1, 0.1]])
    C = cp.BivCheckPi(P / P.sum())
    for kind in ("ltd", "lti", "rti", "rtd"):
        for i in (1, 2):
            expected = check_property(C, kind, i=i)
            assert getattr(C, f"is_{kind}")(i=i) is expected.holds
            assert check_property(C, kind, i=i, method="grid").holds is expected.holds


def test_free_parameter_families_scan_parameter_range():
    assert cp.FarlieGumbelMorgenstern().is_pqd() is False  # theta in [-1, 1]
    with pytest.raises(ValueError):
        cp.Plackett().is_lcsd()
