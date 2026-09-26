"""Validation (c): optimal n=24 checkerboards lie inside and near each boundary."""

import numpy as np
import pytest

pytest.importorskip("cvxpy")

import copul.regions as cr
from copul.optim import trace_boundary

N = 24
MUS = [0.0, 0.2, 1.0, 5.0]

CASES = [
    # (x, y, class, side, constraints, method, max_gap)
    ("xi", "rho", "all", "upper", (), "mu", 3e-3),
    ("xi", "rho", "all", "lower", (), "mu", 3e-3),
    ("xi", "nu", "all", "upper", (), "mu", 3e-3),
    ("xi", "nu", "all", "lower", (), "mu", 3e-3),
    ("xi", "footrule", "si", "upper", ("si",), "mu", 3e-2),
    ("rho", "nu", "all", "upper", (), "target", 3e-3),
    ("rho", "nu", "all", "lower", (), "target", 3e-3),
]


@pytest.mark.parametrize("x,y,cls,side,cons,method,max_gap", CASES)
def test_traces_inside_and_near_boundary(x, y, cls, side, cons, method, max_gap):
    reg = cr.get(x, y, cls)
    kw = {"mus": MUS} if method == "mu" else {"targets": np.linspace(-0.9, 0.9, 7)}
    tr = trace_boundary(x, y, side=side, n=N, constraints=cons, method=method, **kw)
    assert len(tr) >= 3
    assert np.all(reg.contains(tr.xs, tr.ys, tol=1e-9)), reg.margin(tr.xs, tr.ys).max()
    bound = reg.upper(tr.xs) if side == "upper" else reg.lower(tr.xs)
    gap = np.abs(bound - tr.ys)
    assert gap.max() <= max_gap, gap
