"""Regression tests for the schur_order verifiers, rearranger and xi-bounds."""

import numpy as np
import pytest
import sympy

import copul
from copul.checkerboard import _biv_engine as eng
from copul.checkerboard.biv_check_min import BivCheckMin
from copul.checkerboard.biv_check_mixed import BivCheckMixed
from copul.checkerboard.biv_check_pi import BivCheckPi
from copul.checkerboard.biv_check_w import BivCheckW
from copul.schur_order.bounds_from_xi import _N_of_b, _Xi_of_b, nu_bounds_from_xi
from copul.schur_order.cis_rearranger import CISRearranger
from copul.schur_order.cis_verifier import CISVerifier
from copul.schur_order.corner_set_verifier import CornerSetVerifier
from copul.schur_order.ltd_verifier import LTDVerifier


# --------------------------------------------------------------------------
# CornerSetVerifier works for checkerboards (used to crash on cdf(u=, v=))
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cop, lcsd",
    [
        (BivCheckMin(np.eye(3)), True),  # = M
        (BivCheckPi(np.ones((3, 3))), True),  # = Pi
        (BivCheckW(np.fliplr(np.eye(3))), False),  # = W
    ],
)
def test_corner_set_verifier_on_checkerboards(cop, lcsd):
    assert CornerSetVerifier(n_grid=15).is_lcsd(cop) is lcsd
    assert isinstance(CornerSetVerifier(n_grid=15).is_rcsi(cop), bool)


# --------------------------------------------------------------------------
# CIS verifier: bool return, parameter ranges aggregated (not last value)
# --------------------------------------------------------------------------
class _ToyFamily:
    """Parametric toy family: SI for theta > 0, SD for theta < 0.

    Instances (FGM copulas) only provide a numerical ``cond_distr_1``, so the
    numerical grid path of the verifier is exercised.
    """

    def __init__(self, theta=None):
        self.theta = theta
        self.params = [] if theta is not None else [sympy.Symbol("theta")]
        self.intervals = {"theta": sympy.Interval(-1, 1)}

    def __call__(self, theta):
        return _ToyFamily(theta)

    def cond_distr_1(self, u=None, v=None):
        # FGM copula: d1 C = v + theta v (1 - v) (1 - 2u)
        if u is None:
            raise TypeError("numerical only")
        return v + self.theta * v * (1 - v) * (1 - 2 * u)


def test_cis_verifier_returns_bool_and_aggregates_parameter_range():
    fam = _ToyFamily()
    # the old implementation returned the result of the *last* parameter
    assert CISVerifier().cis_direction(fam) == (False, False)
    assert CISVerifier().is_cis(fam) is False
    assert CISVerifier().cis_direction(fam, range_min=0.1) == (True, False)
    assert CISVerifier().is_cis(fam, range_min=0.1) is True
    assert CISVerifier().cis_direction(fam, range_max=-0.1) == (False, True)


def test_cis_verifier_symbolic_family():
    frank = copul.Frank(theta=2)
    assert CISVerifier().cis_direction(frank) == (True, False)


def test_checkerboard_is_si_returns_bool():
    cop = BivCheckPi([[2, 1, 0], [1, 1, 1], [0, 1, 2]])
    assert cop.is_si() is True and cop.is_cis() is True
    assert cop.cis_direction() == (True, False)
    assert CISVerifier(2).is_cis(cop) is True
    assert BivCheckW(np.fliplr(np.eye(3))).cis_direction() == (False, True)


def _copula_matrix(m, n, rng):
    A = rng.random((m, n)) ** 3 + 1e-3
    for _ in range(2000):
        A = A / A.sum(1, keepdims=True) / m
        A = A / A.sum(0, keepdims=True) / n
    return A


def _brute_force(P, S, N=301):
    g = (np.arange(N) + 0.37) / N
    U, V = np.meshgrid(g, g, indexing="ij")
    C = eng.cdf(P, S, U, V)
    H = eng.cond_distr(P, S, 1, U, V)
    tol = 1e-9
    dH = np.diff(H, axis=0)
    dL = np.diff(C / U, axis=0)
    dR = np.diff((1 - U - V + C) / (1 - U), axis=0)
    return {
        "si": np.all(dH <= tol),
        "sd": np.all(dH >= -tol),
        "ltd": np.all(dL <= tol),
        "lti": np.all(dL >= -tol),
        "rti": np.all(dR >= -tol),
        "rtd": np.all(dR <= tol),
        "pqd": np.all(U * V - tol <= C),
        "nqd": np.all(U * V + tol >= C),
    }


@pytest.mark.parametrize("seed", range(6))
def test_exact_dependence_properties_match_brute_force(seed):
    rng = np.random.default_rng(seed)
    for trial in range(8):
        m, n = rng.integers(1, 4, size=2)
        if trial % 2:
            x = np.arange(m)[:, None] / max(m - 1, 1)
            y = np.arange(n)[None, :] / max(n - 1, 1)
            sgn = 1 if trial % 4 == 1 else -1
            A = np.exp(-rng.uniform(1, 8) * (x - y if sgn == 1 else x + y - 1) ** 2)
            for _ in range(2000):
                A = A / A.sum(1, keepdims=True) / m
                A = A / A.sum(0, keepdims=True) / n
            P = A
        else:
            P = _copula_matrix(m, n, rng)
        S = rng.integers(-1, 2, size=(m, n)) if trial % 3 == 0 else rng.integers(-1, 2)
        cop = BivCheckMixed(P, sign=np.broadcast_to(S, (m, n)))
        bf = _brute_force(cop.matr, cop.sign)
        si, sd = cop.cis_direction()
        exact = {
            "si": si,
            "sd": sd,
            "ltd": cop.is_ltd(),
            "lti": cop.is_lti(),
            "rti": cop.is_rti(),
            "rtd": cop.is_rtd(),
            "pqd": cop.is_pqd(),
            "nqd": cop.is_nqd(),
        }
        assert exact == {k: bool(v) for k, v in bf.items()}


def test_ltd_verifier_uses_exact_checkerboard_path():
    cop = BivCheckPi(np.array([[3, 0, 0], [0, 1, 2], [0, 2, 1]]))
    assert LTDVerifier().is_ltd(cop) is True
    assert LTDVerifier().is_ltd(BivCheckPi([[4, 0, 0], [0, 1, 3], [0, 3, 1]])) is False


# --------------------------------------------------------------------------
# CIS rearrangement: normalised numpy output + BivCheckPi convenience
# --------------------------------------------------------------------------
def test_cis_rearrangement_is_normalised_numpy_and_si():
    D = np.array([[1, 2, 0], [0, 1, 2], [2, 0, 1]]) / 9
    R = CISRearranger.rearrange_checkerboard(D)
    assert isinstance(R, np.ndarray)
    assert np.isclose(R.sum(), 1.0)
    assert np.allclose(R.sum(axis=1), D.sum(axis=1))
    assert np.allclose(R.sum(axis=0), D.sum(axis=0))
    cop = BivCheckPi(D).rearrange_cis()
    assert isinstance(cop, BivCheckPi)
    assert cop.is_si()
    assert np.allclose(cop.matr, R)


# --------------------------------------------------------------------------
# bounds_from_xi: stable near xi = 1
# --------------------------------------------------------------------------
def test_xi_nu_boundary_is_stable_and_monotone():
    bs = np.logspace(-3, 13, 500)
    xi = np.array([_Xi_of_b(b) for b in bs])
    nu = np.array([_N_of_b(b) for b in bs])
    assert np.all(np.diff(xi) > -1e-15) and np.all(np.diff(nu) > -1e-15)
    assert np.all((xi >= 0) & (xi <= 1)) and np.all((nu >= 0) & (nu <= 1))
    assert 0.9999 < _Xi_of_b(1e5) <= 1.0
    assert 0.99999 < _Xi_of_b(1e6) <= 1.0
    lo, hi = nu_bounds_from_xi(1 - 1e-9)
    assert abs(hi - 1) < 1e-8 and hi <= 1 and lo == -hi
    # agreement with the published (unstable) formula where it is accurate
    b = 3.0
    s, t = 1 / np.sqrt(b), np.sqrt((b - 1) / b)
    A = np.arcsinh(np.sqrt(b - 1))
    xi_pub = (
        -105 * s**8 * A + 183 * s**6 * t - 38 * s**4 * t - 88 * s**2 * t + 112 * s**2 + 48 * t - 48
    ) / (210 * s**6)
    nu_pub = (
        -105 * s**8 * A
        + 87 * s**6 * t
        + 250 * s**4 * t
        - 376 * s**2 * t
        + 448 * s**2
        + 144 * t
        - 144
    ) / (420 * s**4)
    assert np.isclose(_Xi_of_b(b), xi_pub, atol=1e-13)
    assert np.isclose(_N_of_b(b), nu_pub, atol=1e-13)
