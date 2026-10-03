"""Tests for copul.theory.orders (concordance and more-SI orderings)."""

from __future__ import annotations

import numpy as np
import pytest

import copul as cp
from copul.theory.dependence import check_property
from copul.theory.orders import (
    CONCORDANCE_MEASURES,
    concordance_order,
    is_concordance_ordered,
    is_more_si,
    measures_along,
)

M, W, PI = cp.UpperFrechet(), cp.LowerFrechet(), cp.BivIndependenceCopula()


def test_frechet_hoeffding_bounds():
    for C in (cp.Clayton(2), cp.Frank(-3), cp.MarshallOlkin(0.2, 0.6), cp.StudentT(0.3, 4)):
        lo, hi = concordance_order(W, C), concordance_order(C, M)
        assert lo.holds and hi.holds and lo.method == hi.method == "exact"
        assert concordance_order(C, W, method="grid").holds is False
        assert concordance_order(M, C, method="grid").holds is False
        assert concordance_order(C, C, method="grid").holds


@pytest.mark.parametrize(
    "C",
    [
        cp.Clayton(2),
        cp.Frank(-3),
        cp.Plackett(3),
        cp.StudentT(0.5, 2),
        cp.FarlieGumbelMorgenstern(-0.4),
    ],
    ids=["Clayton", "Frank-", "Plackett", "StudentT", "FGM-"],
)
def test_independence_below_iff_pqd(C):
    pqd = check_property(C, "PQD").holds
    assert concordance_order(PI, C).holds is pqd
    assert concordance_order(PI, C, method="grid").holds is pqd
    nqd = check_property(C, "NQD").holds
    assert concordance_order(C, PI).holds is nqd


@pytest.mark.parametrize(
    "family, values",
    [
        (cp.Clayton, [-0.5, 0.5, 2, 5]),
        (cp.GumbelHougaard, [1.2, 2, 4]),
        (cp.Frank, [-4, -1, 2, 6]),
        (cp.Gaussian, [-0.5, 0, 0.3, 0.8]),
        (cp.AliMikhailHaq, [-0.8, 0, 0.5, 0.9]),
        (cp.Joe, [1.2, 2, 4]),
        (cp.Plackett, [0.3, 1.5, 4]),
        (cp.FarlieGumbelMorgenstern, [-1, 0, 1]),
        (cp.Galambos, [0.3, 1, 2]),
    ],
    ids=lambda x: getattr(x, "__name__", None),
)
def test_positively_ordered_families_and_monotone_measures(family, values):
    res = is_concordance_ordered(family, values)
    assert res.direction == "increasing" and res.holds
    meas = measures_along(family, values)
    assert set(meas) == set(CONCORDANCE_MEASURES)
    for key, arr in meas.items():
        assert np.all(np.diff(arr) >= -1e-9), (key, arr)


def test_family_given_as_instance_or_callable():
    res = is_concordance_ordered(cp.Clayton(), [3, 1])  # free parameter, unsorted values
    assert res.values == [1.0, 3.0] and res.increasing
    res = is_concordance_ordered(lambda t: cp.Gaussian(-t), [0.1, 0.5])
    assert res.direction == "decreasing"
    with pytest.raises(TypeError):
        is_concordance_ordered(3, [0.1, 0.2])


def test_non_comparable_pair():
    a, b = cp.Clayton(2), cp.Gaussian(0.5)
    ab, ba = concordance_order(a, b), concordance_order(b, a)
    assert not ab and not ba
    assert ab.worst_violation > 1e-3 and ba.worst_violation > 1e-3
    u, v = ab.where["u"], ab.where["v"]
    assert a.cdf(u, v) > b.cdf(u, v)


def test_checkerboard_order_is_exact():
    A = cp.BivCheckPi([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # = Pi
    B = cp.BivCheckPi([[2, 1, 0], [1, 1, 1], [0, 1, 2]])
    Cc = cp.BivCheckPi([[3, 0, 0], [0, 2, 1], [0, 1, 2]])
    for X, Y in ((A, B), (B, Cc), (A, Cc), (B, A)):
        ex, gr = concordance_order(X, Y), concordance_order(X, Y, method="grid")
        assert ex.method == "exact" and ex.holds == gr.holds
    assert concordance_order(B, Cc).holds and not concordance_order(Cc, B).holds
    # different grid sizes
    D = cp.BivCheckPi([[1, 0], [0, 1]])
    assert concordance_order(B, D).holds == concordance_order(B, D, method="grid").holds


def test_more_si_order():
    for C in (PI, cp.Clayton(2), cp.Gaussian(0.5), cp.Frank(-2), cp.MarshallOlkin(0.3, 0.6)):
        assert is_more_si(C, C).holds  # reflexive
        assert is_more_si(C, M).holds  # M is the maximum
        # Pi <=_SI C iff C is SI(V|U)
        assert is_more_si(PI, C).holds is check_property(C, "SI").holds
    # Gaussian: psi = Phi(rho2 x + s2/s1 (y - rho1 x)) increases in x iff
    # rho2 / sqrt(1 - rho2^2) >= rho1 / sqrt(1 - rho1^2)
    assert is_more_si(cp.Gaussian(0.2), cp.Gaussian(0.6)).holds
    r = is_more_si(cp.Gaussian(0.6), cp.Gaussian(0.2))
    assert not r and r.worst_violation > 1e-3
    assert not is_more_si(M, PI)
