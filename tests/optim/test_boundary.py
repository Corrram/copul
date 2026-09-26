"""Tests for copul.optim.trace_boundary."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

pytest.importorskip("cvxpy")

from copul.optim import BoundaryTrace, NonConvexError, trace_boundary
from copul.regions.catalog import rho_max_given_xi


def test_mu_sweep_xi_rho_below_exact_bound_and_converging():
    mus = [0.0, 0.1, 0.5, 2.0, 10.0]
    gaps = {}
    for n in (8, 16):
        tr = trace_boundary("xi", "rho", side="upper", n=n, mus=mus)
        assert isinstance(tr, BoundaryTrace)
        assert np.all(np.diff(tr.xs) >= 0)
        bound = rho_max_given_xi(tr.xs)
        assert np.all(tr.ys <= bound + 1e-9)
        gaps[n] = float(np.max(bound - tr.ys))
        assert len(tr.copulas) == len(tr)
    assert gaps[16] < gaps[8]
    lo = trace_boundary("xi", "rho", side="lower", n=8, mus=mus)
    assert np.all(lo.ys >= -rho_max_given_xi(lo.xs) - 1e-9)
    np.testing.assert_allclose(lo.ys, -trace_boundary("xi", "rho", n=8, mus=mus).ys, atol=1e-6)


def test_target_sweep_linear_pair():
    tr = trace_boundary("rho", "nu", side="upper", n=8, method="target", n_points=5)
    # rho = +-1 is not attainable on an 8x8 BivCheckPi grid: those targets are skipped
    np.testing.assert_allclose(tr.xs, [-0.5, 0.0, 0.5], atol=1e-6)
    tr2 = trace_boundary("xi", "nu", side="upper", n=8, method="target", targets=[0.1, 0.5])
    assert np.all(tr2.xs <= np.array([0.1, 0.5]) + 1e-7)
    assert tr2.interpolate(0.3) == pytest.approx(np.interp(0.3, tr2.xs, tr2.ys))


def test_affine_x_default_directions_and_constraints():
    tr = trace_boundary("beta", "nu", side="upper", n=6, n_points=7, constraints=("si",))
    assert len(tr) >= 2
    for r in tr.results:
        assert r.values["beta"] >= -1e-7  # SI implies beta >= 0


def test_invalid_sweeps():
    with pytest.raises(NonConvexError):
        trace_boundary("xi", "tau", n=4)
    with pytest.raises(NonConvexError):
        trace_boundary("rho", "xi", side="upper", n=4)
    with pytest.raises(NonConvexError):
        trace_boundary("xi", "rho", n=4, mus=[-1.0])
    with pytest.raises(ValueError):
        trace_boundary("xi", "rho", n=4, side="left")
    with pytest.raises(ValueError):
        trace_boundary("xi", "rho", n=4, method="grid")


def test_plot_runs():
    tr = trace_boundary("xi", "rho", n=6, mus=[0.0, 1.0])
    ax = tr.plot()
    assert ax.get_xlabel().startswith("Chatterjee")
    ax2 = tr.plot(ax=ax, color="red", label="n=6")
    assert ax2 is ax
    plt.close("all")
