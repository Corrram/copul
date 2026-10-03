"""Trutschnig's zeta_1 in the measures registry (key "zeta1")."""

from __future__ import annotations

import numpy as np
import pytest

import copul as cp
from copul.measures import compute, get_measure, measures_from_h
from copul.measures.backend import numeric_backend
from copul.measures.engine import MEASURE_METHODS
from copul.measures.numeric import zeta1_checkerboard, zeta1_from_h


def test_registry_entry_and_aliases():
    m = get_measure("zeta")
    assert m.key == "zeta1" and m.method_name == "trutschnig_zeta"
    assert get_measure("trutschnig_zeta").key == "zeta1"
    assert (m.at_M, m.at_W, m.at_Pi, m.range) == (1.0, 1.0, 0.0, (0.0, 1.0))
    assert "trutschnig_zeta" in MEASURE_METHODS


@pytest.mark.parametrize(
    "factory, attr",
    [
        (cp.UpperFrechet, "at_M"),
        (cp.LowerFrechet, "at_W"),
        (cp.BivIndependenceCopula, "at_Pi"),
    ],
)
def test_registry_values_at_bounds(factory, attr):
    expected = getattr(get_measure("zeta1"), attr)
    assert factory().measure("zeta1") == pytest.approx(expected, abs=1e-12)
    assert factory().trutschnig_zeta() == pytest.approx(expected, abs=1e-12)


def test_fgm_closed_form_and_method_dispatch():
    # d1 C - v = theta v (1-v) (1-2u): zeta_1 = 3 |theta| / 6 / 2 = |theta| / 4
    for th in (-0.8, 0.4, 1.0):
        C = cp.FarlieGumbelMorgenstern(th)
        res = compute(C, "zeta1", full_output=True)
        assert res.value == pytest.approx(abs(th) / 4, abs=1e-10)
        assert res.method == "numeric" and res.error < 1e-8

    def h1(u, v):
        return v + 0.4 * v * (1 - v) * (1 - 2 * u)

    assert zeta1_from_h(h1) == pytest.approx(0.1, abs=1e-12)


def test_complete_dependence_has_zeta_one():
    for C in (cp.ShuffleOfMin([2, 4, 1, 3]), cp.BivCheckMin(np.eye(3)), cp.BivCheckW(np.eye(2))):
        assert C.trutschnig_zeta() == pytest.approx(1.0, abs=1e-12)
    # V = f(U) with a two-to-one f: still completely dependent (zeta_1 = 1),
    # while U is not a function of V
    tent = cp.BivCheckMixed([[0.5], [0.5]], sign=[[1], [-1]])
    assert tent.trutschnig_zeta() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("sign", [0, 1, -1])
def test_exact_checkerboard_formula_matches_grid(sign):
    rng = np.random.default_rng(3 + sign)
    P = rng.random((3, 4))
    for _ in range(500):
        P = P / P.sum(1, keepdims=True) / 3
        P = P / P.sum(0, keepdims=True) / 4
    cls = {0: cp.BivCheckPi, 1: cp.BivCheckMin, -1: cp.BivCheckW}[sign]
    C = cls(P)
    exact = zeta1_checkerboard(P, sign)
    assert C.trutschnig_zeta() == pytest.approx(exact, abs=1e-14)
    # midpoint grid of the h-function (error O(N^-1) at the kernel jumps)
    N = 1200
    g = (np.arange(N) + 0.5) / N
    U, V = np.meshgrid(g, g, indexing="ij")
    H = numeric_backend(C).h1(U, V)
    assert measures_from_h(H, "zeta1") == pytest.approx(exact, abs=3e-3 / N * 100)


def test_numeric_value_agrees_with_monte_carlo_grid_estimate():
    C = cp.Clayton(1.5)
    val = C.trutschnig_zeta()
    N = 600
    g = (np.arange(N) + 0.5) / N
    U, V = np.meshgrid(g, g, indexing="ij")
    approx = 3 * np.mean(np.abs(C.cond_distr_1(U, V) - V))
    assert val == pytest.approx(approx, abs=1e-4)
    assert 0 < val < 1


def test_not_symmetric_and_free_parameters():
    tent = cp.BivCheckMixed([[0.5], [0.5]], sign=[[1], [-1]])
    from copul.family.constructions import transpose

    assert transpose(tent).trutschnig_zeta() < 0.99
    with pytest.raises(ValueError):
        cp.Clayton().trutschnig_zeta()
