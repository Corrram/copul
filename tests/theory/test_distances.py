"""Tests for copul.theory.distances (copula metrics and Trutschnig's zeta_1)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

import copul as cp
from copul.family.constructions import mixture
from copul.theory.distances import METRICS, copula_distance, trutschnig_zeta, zeta1

M, W, PI = cp.UpperFrechet(), cp.LowerFrechet(), cp.BivIndependenceCopula()


@pytest.mark.parametrize(
    "A, B, expected",
    [
        # M vs Pi: |min(u,v) - uv|, |1{u<=v} - v|
        (M, PI, {"sup": 0.25, "L1": 1 / 12, "L2": math.sqrt(1 / 90), "D1": 1 / 3,
                 "D2": math.sqrt(1 / 6), "Dinf": 0.5}),
        (W, PI, {"sup": 0.25, "L1": 1 / 12, "L2": math.sqrt(1 / 90), "D1": 1 / 3,
                 "D2": math.sqrt(1 / 6), "Dinf": 0.5}),
        (M, W, {"sup": 0.5, "L1": 1 / 6, "L2": math.sqrt(1 / 24), "D1": 0.5,
                "D2": math.sqrt(0.5), "Dinf": 1.0}),
        # FGM(theta) vs Pi: theta uv(1-u)(1-v) and theta v(1-v)(1-2u)
        (cp.FarlieGumbelMorgenstern(0.5), PI, {"sup": 0.5 / 16, "L1": 0.5 / 36, "L2": 0.5 / 30,
                                               "D1": 0.5 / 12, "D2": 0.5 * math.sqrt(1 / 90),
                                               "Dinf": 0.5 / 8}),
    ],
    ids=["M-Pi", "W-Pi", "M-W", "FGM-Pi"],
)  # fmt: skip
def test_closed_form_distances(A, B, expected):
    for metric, val in expected.items():
        assert copula_distance(A, B, metric) == pytest.approx(val, abs=2e-6), metric


def test_metric_names_and_result_object():
    r = copula_distance(cp.Clayton(2), cp.Frank(3), "D_inf", full_output=True)
    assert r.metric == "Dinf" and r.method == "numeric" and set(r.where) == {"v"}
    assert float(r) == r.value
    assert copula_distance(M, PI, "uniform") == pytest.approx(0.25, abs=1e-9)
    with pytest.raises(ValueError):
        copula_distance(M, PI, "hellinger")


SAMPLE = {
    "Clayton(2)": lambda: cp.Clayton(2),
    "Gaussian(-0.4)": lambda: cp.Gaussian(-0.4),
    "MarshallOlkin": lambda: cp.MarshallOlkin(0.3, 0.7),
    "CheckPi": lambda: cp.BivCheckPi([[2, 1, 0], [1, 1, 1], [0, 1, 2]]),
}


@pytest.fixture(scope="module")
def distance_table():
    cops = {k: f() for k, f in SAMPLE.items()}
    table = {}
    for (na, a), (nb, b) in itertools.product(cops.items(), repeat=2):
        if na <= nb:
            for m in METRICS:
                table[(na, nb, m)] = copula_distance(a, b, m, rtol=1e-6, atol=1e-8)
                table[(nb, na, m)] = copula_distance(b, a, m, rtol=1e-6, atol=1e-8)
    return table


def test_metric_axioms(distance_table):
    names = list(SAMPLE)
    for m in METRICS:
        for a in names:
            assert distance_table[(a, a, m)] == pytest.approx(0.0, abs=1e-6)
        for a, b in itertools.combinations(names, 2):
            dab, dba = distance_table[(a, b, m)], distance_table[(b, a, m)]
            assert dab > 1e-3
            assert dab == pytest.approx(dba, abs=1e-6)
        for a, b, c in itertools.permutations(names, 3):
            assert (
                distance_table[(a, c, m)]
                <= distance_table[(a, b, m)] + distance_table[(b, c, m)] + 1e-6
            )


def test_proven_inequalities(distance_table):
    names = list(SAMPLE)
    for a, b in itertools.combinations(names, 2):
        d = {m: distance_table[(a, b, m)] for m in METRICS}
        tol = 1e-6
        assert d["sup"] <= d["Dinf"] + tol  # |A - B| <= Phi_{A,B}(y)
        assert d["L1"] <= d["D1"] + tol
        assert d["D1"] <= d["Dinf"] + tol
        assert d["D2"] ** 2 <= d["D1"] + tol  # |d1 A - d1 B| <= 1
        assert d["D1"] <= d["D2"] + tol  # Cauchy-Schwarz
        assert d["L2"] <= d["sup"] + tol and d["L1"] <= d["L2"] + tol


def test_exact_checkerboard_distances_match_quadrature():
    A = cp.BivCheckPi([[1, 2], [2, 1]])
    B = cp.BivCheckPi([[2, 1, 0], [1, 1, 1], [0, 1, 2]])
    wrapped = mixture([A], [1.0])  # same copula, generic numerical route
    for m in ("sup", "D1", "D2", "Dinf"):
        ex = copula_distance(A, B, m, full_output=True)
        num = copula_distance(wrapped, B, m, full_output=True)
        assert ex.method == "exact" and num.method == "numeric"
        assert num.value == pytest.approx(ex.value, abs=1e-6), m


def test_zeta1_properties():
    for C in (M, W, cp.ShuffleOfMin([3, 1, 2]), cp.BivCheckMin(np.eye(4))):
        assert trutschnig_zeta(C) == pytest.approx(1.0, abs=1e-12)
    assert trutschnig_zeta(PI) == pytest.approx(0.0, abs=1e-12)
    for th in (-1.0, -0.3, 0.6):
        assert zeta1(cp.FarlieGumbelMorgenstern(th)) == pytest.approx(abs(th) / 4, abs=1e-9)
    for C in (cp.Clayton(2), cp.Gaussian(0.5), cp.BivCheckPi([[1, 2], [2, 1]])):
        z = trutschnig_zeta(C)
        assert 0 < z < 1
        assert z == pytest.approx(3 * copula_distance(C, PI, "D1"), abs=1e-7)
