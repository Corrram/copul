"""Checkerboard copulas keep their closed forms; pure numeric API; curves."""

import numpy as np
import pytest

import copul as cp
from copul.measures import (
    compute,
    lp_constant,
    measures_from_cdf,
    measures_from_h,
    numeric_backend,
    rho_from_h,
    tau_from_h,
    xi_from_h,
)

KEYS = ["rho", "tau", "xi", "footrule", "gamma", "beta", "nu"]


def _matrix(n=5, seed=0):
    from copul.checkerboard.biv_check_pi import BivCheckPi

    return BivCheckPi.random_bistochastic_matrix(n, rng=np.random.default_rng(seed))


# closed forms that disagree with the class' own cdf (the numerical value
# agrees with a brute-force evaluation of the class' cdf, see below)
KNOWN_CLOSED_FORM_BUGS: set = set()  # all previously known bugs are fixed


@pytest.mark.parametrize("cls", [cp.BivCheckPi, cp.BivCheckMin, cp.BivCheckW])
@pytest.mark.parametrize("key", [*KEYS, "xi_2"])
def test_checkerboard_closed_forms_used_and_match_numeric(cls, key, request):
    if (cls.__name__, key) in KNOWN_CLOSED_FORM_BUGS:
        request.applymarker(
            pytest.mark.xfail(reason="closed form inconsistent with cdf", strict=False)
        )
    cop = cls(_matrix())
    auto = compute(cop, key, full_output=True)
    assert auto.method == "closed", (key, auto)
    num = compute(cop, key, method="numeric", full_output=True)
    tol = 1e-9 if cls is cp.BivCheckPi else 1e-4  # Min/W are singular
    assert num.value == pytest.approx(auto.value, abs=tol), (cls.__name__, key)


def test_checkerboard_numeric_matches_bruteforce_of_class_cdf():
    cop = cp.BivCheckW(_matrix())
    N = 2000
    t = (np.arange(N) + 0.5) / N
    diag = np.array([cop.cdf(x, x) for x in t])
    num = cop.spearmans_footrule(method="numeric")
    assert num == pytest.approx(6 * diag.mean() - 2, abs=1e-6)


def test_checkerboard_condition_on_y():
    cop = cp.BivCheckPi(_matrix(4, 1))
    closed = cop.chatterjees_xi(condition_on_y=True)
    num = cop.chatterjees_xi(condition_on_y=True, method="numeric")
    assert num == pytest.approx(closed, abs=1e-6)
    # positional flag keeps working
    assert cop.chatterjees_xi(True) == pytest.approx(closed)


def test_checkerboard_backend_is_fast_and_exact():
    cop = cp.BivCheckPi(_matrix())
    be = numeric_backend(cop)
    assert be.source["cdf"] == "checkerboard"
    pts = np.random.default_rng(2).random((50, 2))
    np.testing.assert_allclose(be.cdf(pts[:, 0], pts[:, 1]), cop.cdf(pts), atol=1e-14)
    np.testing.assert_allclose(
        be.get("h1")(pts[:, 0], pts[:, 1]), cop.cond_distr(1, pts), atol=1e-12
    )


# ---------------------------------------------------------------------------
# pure numeric API
# ---------------------------------------------------------------------------

TH = 0.7


def fgm_cdf(u, v):
    return u * v * (1 + TH * (1 - u) * (1 - v))


def fgm_h1(u, v):
    return v + TH * v * (1 - v) * (1 - 2 * u)


def fgm_h2(u, v):
    return u + TH * u * (1 - u) * (1 - 2 * v)


def test_pure_functions():
    assert xi_from_h(fgm_h1) == pytest.approx(TH**2 / 15, abs=1e-12)
    assert rho_from_h(fgm_h1) == pytest.approx(TH / 3, abs=1e-12)
    assert tau_from_h(fgm_h1, fgm_h2) == pytest.approx(2 * TH / 9, abs=1e-12)
    val, err = xi_from_h(fgm_h1, full_output=True)
    assert err < 1e-8


def test_measures_from_cdf_with_fd_derivatives():
    out = measures_from_cdf(fgm_cdf, ["rho", "tau", "xi", "footrule", "gamma", "beta", "nu"])
    exact = {
        "rho": TH / 3,
        "tau": 2 * TH / 9,
        "xi": TH**2 / 15,
        "footrule": TH / 5,
        "gamma": 4 * TH / 15,
        "beta": TH / 4,
        "nu": TH / 3,
    }
    for k, v in exact.items():
        assert out[k] == pytest.approx(v, abs=1e-8), k


def test_measures_from_h_callable():
    out = measures_from_h(fgm_h1, ["xi", "rho", "nu", "footrule", "beta", "tau"], h2=fgm_h2)
    assert out["xi"] == pytest.approx(TH**2 / 15, abs=1e-10)
    assert out["rho"] == pytest.approx(TH / 3, abs=1e-10)
    assert out["nu"] == pytest.approx(TH / 3, abs=1e-10)
    assert out["footrule"] == pytest.approx(TH / 5, abs=1e-8)
    assert out["beta"] == pytest.approx(TH / 4, abs=1e-8)
    assert out["tau"] == pytest.approx(2 * TH / 9, abs=1e-10)
    # without h2: C from integrating h1, h2 by finite differences
    assert measures_from_h(fgm_h1, "tau") == pytest.approx(2 * TH / 9, abs=1e-6)


def test_measures_from_h_grid():
    m = 400
    u = (np.arange(m) + 0.5) / m
    U, V = np.meshgrid(u, u, indexing="ij")
    H = fgm_h1(U, V)
    out = measures_from_h(H, ["xi", "rho", "tau", "nu", "footrule", "gamma", "beta"])
    assert out["xi"] == pytest.approx(TH**2 / 15, abs=1e-5)
    assert out["rho"] == pytest.approx(TH / 3, abs=1e-5)
    assert out["tau"] == pytest.approx(2 * TH / 9, abs=1e-4)
    assert out["nu"] == pytest.approx(TH / 3, abs=1e-4)
    assert out["footrule"] == pytest.approx(TH / 5, abs=1e-4)
    assert out["gamma"] == pytest.approx(4 * TH / 15, abs=1e-4)
    assert out["beta"] == pytest.approx(TH / 4, abs=1e-4)


def test_lp_constant():
    assert lp_constant(1) == 12
    assert lp_constant(2) == 90
    assert lp_constant(3) == 560
    assert lp_constant(5) == 16632
    assert lp_constant(1.5) == pytest.approx(2.5 / (2 * __import__("scipy").special.beta(2.5, 3.5)))


# ---------------------------------------------------------------------------
# curves and calibration
# ---------------------------------------------------------------------------


def test_measure_curve_clayton():
    curve = cp.Clayton().measure_curve(["xi", "tau"], n=6)
    assert curve.param == "theta"
    assert curve["tau"].shape == (6,)
    th = curve.params
    np.testing.assert_allclose(curve["tau"], th / (th + 2), atol=1e-9)
    assert np.all(np.diff(curve["xi"][1:]) > 0)  # increasing for theta >= 0
    assert curve.as_dict()["theta"] is curve.params


def test_measure_curve_explicit_values_and_param():
    curve = cp.Gaussian().measure_curve("rho", values=[-0.5, 0.0, 0.5])
    np.testing.assert_allclose(
        curve["rho"], 6 / np.pi * np.arcsin(np.array([-0.5, 0, 0.5]) / 2), atol=1e-12
    )
    with pytest.raises(ValueError):
        cp.Tawn().measure_curve("rho", n=3)  # several free parameters


@pytest.mark.parametrize(
    "cls, key, target, expected",
    [
        (cp.Clayton, "tau", 0.5, 2.0),
        (cp.GumbelHougaard, "tau", 0.75, 4.0),
        (cp.Gaussian, "tau", 2 / np.pi * np.arcsin(0.3), 0.3),
    ],
)
def test_from_measure_known(cls, key, target, expected):
    c = cls.from_measure(key, target)
    param = str(cls.params[0]) if hasattr(cls, "params") and cls.params else "theta"
    assert float(getattr(c, param)) == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize(
    "factory, key, target",
    [
        (lambda: cp.Frank(), "rho", -0.4),
        (lambda: cp.Clayton(), "xi", 0.3),
        (lambda: cp.Tawn(0.6, 0.9), "tau", 0.3),
    ],
)
def test_from_measure_round_trip(factory, key, target):
    c = factory().from_measure(key, target)
    assert c.measure(key) == pytest.approx(target, abs=1e-8)
