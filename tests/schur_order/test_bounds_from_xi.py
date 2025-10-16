# tests/test_bounds_from_xi.py
import math
import numpy as np
import pytest

# Module under test
from copul.schur_order.bounds_from_xi import (
    bounds_from_xi,
    rho_bounds_from_xi,
    tau_bounds_from_xi,
    psi_bounds_from_xi,
    nu_bounds_from_xi,
    # internals (used for ν–ξ parametric consistency checks)
    _Xi_of_b,
    _N_of_b,
)


# ------------------------- Validation -------------------------

@pytest.mark.parametrize("bad_x", [float("nan"), float("inf"), -0.1, 1.1, "0.2"])
def test_validate_xi_rejects_bad_inputs(bad_x):
    with pytest.raises(Exception):
        bounds_from_xi(bad_x, measure="rho")


# ------------------------- Rho / Tau --------------------------

@pytest.mark.parametrize("x", [0.0, 0.05, 0.3, 0.7, 1.0])
def test_rho_symmetry_and_range(x):
    rmin, rmax = rho_bounds_from_xi(x)
    assert np.isfinite(rmin) and np.isfinite(rmax)
    # Symmetry
    assert pytest.approx(rmin, rel=0, abs=1e-12) == -rmax
    # Range
    assert -1.0 <= rmin <= rmax <= 1.0
    # Edge cases
    if x == 0.0:
        assert rmin == 0.0 and rmax == 0.0
    if x == 1.0:
        assert rmin == -1.0 and rmax == 1.0

@pytest.mark.parametrize("x", [0.0, 0.05, 0.3, 0.7, 1.0])
def test_tau_symmetry_and_range(x):
    tmin, tmax = tau_bounds_from_xi(x)
    assert np.isfinite(tmin) and np.isfinite(tmax)
    # Symmetry
    assert pytest.approx(tmin, rel=0, abs=1e-12) == -tmax
    # Range
    assert -1.0 <= tmin <= tmax <= 1.0
    # Edge cases
    if x == 0.0:
        assert tmin == 0.0 and tmax == 0.0
    if x == 1.0:
        assert tmin == -1.0 and tmax == 1.0


# ------------------------- Psi -------------------------------

@pytest.mark.parametrize("x", [0.0, 0.1, 0.25, 0.49, 0.5, 0.8, 1.0])
def test_psi_bounds_all_and_SI(x):
    # "all" class: returns (lower_bound, sqrt(x))
    pmin_all, pmax_all = psi_bounds_from_xi(x, cls="all", return_lower_bound=True)
    assert pytest.approx(pmax_all, rel=0, abs=1e-12) == math.sqrt(x)
    # SI class: exact (x, sqrt(x))
    pmin_si, pmax_si = psi_bounds_from_xi(x, cls="SI")
    assert pytest.approx(pmin_si, rel=0, abs=1e-12) == x
    assert pytest.approx(pmax_si, rel=0, abs=1e-12) == math.sqrt(x)
    # ranges
    assert pmin_all <= pmax_all
    assert pmin_si <= pmax_si
    assert -0.5 <= pmin_all <= pmax_all <= 1.0  # known global range
    # edge-cases
    if x == 0.0:
        assert pmin_all == 0.0 and pmax_all == 0.0
        assert pmin_si == 0.0 and pmax_si == 0.0


# ------------------------- Nu (Blest) ------------------------

@pytest.mark.parametrize("x", [0.0, 0.05, 0.3, 0.7, 0.95, 1.0])
def test_nu_symmetry_and_range(x):
    nmin, nmax = nu_bounds_from_xi(x)
    assert np.isfinite(nmin) and np.isfinite(nmax)
    # Symmetry
    assert pytest.approx(nmin, rel=0, abs=1e-12) == -nmax
    # Range
    assert -1.0 <= nmin <= nmax <= 1.0
    # Edge cases
    if x == 0.0:
        assert nmin == 0.0 and nmax == 0.0
    if x == 1.0:
        assert nmin == -1.0 and nmax == 1.0


@pytest.mark.parametrize("b", [0.05, 0.2, 0.5, 0.9, 1.0, 1.2, 2.0, 5.0])
def test_nu_parametric_consistency_via_b(b):
    """
    For a given b>0, compute x = Xi(b) and N = N(b).
    Then nu_bounds_from_xi(x) should return (-N, N) (up to small numeric tolerance).
    This checks the bisection inversion Xi(b)=x and the boundary evaluation.
    """
    x = _Xi_of_b(b)
    N = _N_of_b(b)
    nmin, nmax = nu_bounds_from_xi(x)
    assert np.isfinite(x) and np.isfinite(N)
    assert pytest.approx(nmin, rel=1e-10, abs=5e-10) == -N
    assert pytest.approx(nmax, rel=1e-10, abs=5e-10) == N
    # region bounds always symmetric and ordered
    assert nmin <= 0.0 <= nmax


# ------------------------- Dispatcher ------------------------

@pytest.mark.parametrize("x", [0.0, 0.2, 0.3, 0.8, 1.0])
def test_dispatcher_matches_direct_calls(x):
    # rho
    assert bounds_from_xi(x, "rho") == rho_bounds_from_xi(x)
    # tau
    assert bounds_from_xi(x, "tau") == tau_bounds_from_xi(x)
    # psi ("all", with lower bound)
    assert bounds_from_xi(x, "psi", return_lower_bound=True) == psi_bounds_from_xi(
        x, cls="all", return_lower_bound=True
    )
    # nu
    assert bounds_from_xi(x, "nu") == nu_bounds_from_xi(x)


def test_dispatcher_rejects_unknown_measure():
    with pytest.raises(ValueError):
        bounds_from_xi(0.3, measure="oops")  # type: ignore[arg-type]


@pytest.mark.parametrize("x", [0,0.1, 0.2, 0.3, 0.8, 0.9, 1])
def test_nu_bound_larger_than_rho(x):
    nmin, nmax = nu_bounds_from_xi(x)
    assert np.isfinite(nmin) and np.isfinite(nmax)
    rho_nmin, rho_nmax = rho_bounds_from_xi(x)
    assert np.isfinite(rho_nmin) and np.isfinite(rho_nmax)
    if x not in [0,1]:
        assert nmin < rho_nmin < rho_nmax < nmax
    elif x == 0.0:
        assert nmin == rho_nmin == rho_nmax == nmax
    else:
        assert nmin == rho_nmin < rho_nmax == nmax
