"""
Unit tests for the mixed checkerboard copula.

We check that                           BivCheckMixed(Δ,S)
reduces to the correct specialised class when the sign-matrix S
is constant, and that it blends the three regimes cell-wise.
"""

import matplotlib
import numpy as np
import pytest

from copul.checkerboard.biv_check_pi import BivCheckPi
from copul.checkerboard.biv_check_min import BivCheckMin
from copul.checkerboard.biv_check_w import BivCheckW
from copul.checkerboard.biv_check_mixed import BivCheckMixed

matplotlib.use("Agg")  # do not open GUI windows


# --------------------------------------------------------------------- #
# 1)  Constant sign-matrix  →  should match the specialised classes
# --------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "sign_cls",
    [
        (0, BivCheckPi),  # independence in every cell
        (+1, BivCheckMin),  # perfect +dep in every cell
        (-1, BivCheckW),  # perfect –dep in every cell
    ],
)
def test_constant_sign_matches_base(sign_cls):
    s_const, base_cls = sign_cls

    Δ = np.array([[1, 1], [1, 1]], dtype=float)
    Δ /= Δ.sum()

    S = np.full_like(Δ, s_const, dtype=int)

    cop_mixed = BivCheckMixed(Δ, sign=S)
    cop_base = base_cls(Δ)

    # pick a few test points
    pts = [(0.25, 0.25), (0.25, 0.75), (0.8, 0.1), (1.0, 1.0)]

    for u, v in pts:
        assert np.isclose(cop_mixed.cdf(u, v), cop_base.cdf(u, v))

    # measures of association should coincide too
    tau_actual = cop_mixed.kendalls_tau()
    tau_expected = cop_base.kendalls_tau()
    assert np.isclose(tau_actual, tau_expected)
    rho_actual = cop_mixed.spearmans_rho()
    rho_expected = cop_base.spearmans_rho()
    assert np.isclose(rho_actual, rho_expected)
    xi_actual = cop_mixed.chatterjees_xi()
    xi_expected = cop_base.chatterjees_xi()
    assert np.isclose(xi_actual, xi_expected)


# --------------------------------------------------------------------- #
# 2)  Mixed pattern: check per–cell delegation
# --------------------------------------------------------------------- #
def test_mixed_cdf_piecewise():
    Δ = np.full((2, 2), 0.25)  # uniform masses
    S = np.array(
        [
            [0, 1],  #  (0,0): Pi    (0,1): Min
            [-1, 0],
        ]
    )  #  (1,0): W     (1,1): Pi

    mixed = BivCheckMixed(Δ, sign=S)

    # pre-build the reference copulas
    pi = BivCheckPi(Δ)
    cm = BivCheckMin(Δ)
    cw = BivCheckW(Δ)

    # points chosen inside each cell
    tests = [
        ((0.25, 0.25), pi),  # cell (0,0)
        ((0.25, 0.75), cm),  # cell (0,1)
        ((0.75, 0.25), cw),  # cell (1,0)
        ((0.75, 0.75), pi),  # cell (1,1)
    ]

    for (u, v), ref in tests:
        assert np.isclose(mixed.cdf(u, v), ref.cdf(u, v))


# --------------------------------------------------------------------- #
# 3)  Measures for the mixed pattern: only sanity (signs & range)
# --------------------------------------------------------------------- #
def test_measures_sanity_mixed():
    Δ = np.array([[0.1, 0.05, 0.05], [0.05, 0.4, 0.05], [0.05, 0.05, 0.2]])
    S = np.array([[0, 1, -1], [1, 1, 0], [-1, 0, -1]])

    cop = BivCheckMixed(Δ, sign=S)

    tau = cop.kendalls_tau()
    rho = cop.spearmans_rho()
    xi = cop.chatterjees_xi()

    # all measures are within [-1,1] and xi ∈ [0,1]
    assert -1 <= tau <= 1
    assert -1 <= rho <= 1
    assert 0 <= xi <= 1

    # the central block is +dep dominated → expect positive tau
    assert tau > 0


# --------------------------------------------------------------------- #
# 4)  Plotting functions (smoke tests)
# --------------------------------------------------------------------- #
@pytest.fixture
def small_mixed():
    Δ = np.array([[1, 1], [1, 1]], dtype=float) / 4
    S = np.array([[0, 1], [-1, 0]])
    return BivCheckMixed(Δ, sign=S)
