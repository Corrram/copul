"""Accuracy of the numerical engine (``method="numeric"``) against known
closed forms of dependence measures."""

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import spence

import copul as cp


def debye(n, x):
    val, _ = quad(lambda t: t**n / np.expm1(t), 0, x, epsabs=1e-15, epsrel=1e-14)
    return n / x**n * val


def frank_tau(th):
    return 1 - 4 / th + 4 * debye(1, th) / th


def frank_rho(th):
    return 1 - 12 / th * (debye(1, th) - debye(2, th))


def amh_tau(th):
    return 1 - 2 / (3 * th) - 2 * (1 - th) ** 2 * math.log(1 - th) / (3 * th**2)


def amh_rho(th):
    li2 = spence(1 - th)  # dilogarithm Li_2(th)
    return (12 * (1 + th) * li2 - 24 * (1 - th) * math.log(1 - th)) / th**2 - 3 * (th + 12) / th


def plackett_rho(th):
    return (th + 1) / (th - 1) - 2 * th * math.log(th) / (th - 1) ** 2


def gaussian(r):
    a = math.asin
    return {
        "rho": 6 / math.pi * a(r / 2),
        "tau": 2 / math.pi * a(r),
        "xi": 3 / math.pi * a((1 + r * r) / 2) - 0.5,
        "beta": 2 / math.pi * a(r),
        "footrule": 3 / math.pi * a((1 + r) / 2) - 0.5,
        "gamma": 2 / math.pi * (a((1 + r) / 2) - a((1 - r) / 2)),
    }


def frechet(a, b):
    # C = a M + (1-a-b) Pi + b W
    return {
        "rho": a - b,
        "tau": (a - b) * (a + b + 2) / 3,
        "xi": a * a - a * b + b * b,
        "beta": a - b,
        "footrule": a - b / 2,
        "gamma": a - b,
    }


CASES = [
    # (id, copula factory, key, exact value, tolerance)
    ("clayton-tau-2", lambda: cp.Clayton(2), "tau", 0.5, 1e-9),
    ("clayton-tau-0.5", lambda: cp.Clayton(0.5), "tau", 0.2, 1e-9),
    ("clayton-tau-neg", lambda: cp.Clayton(-0.5), "tau", -0.5 / 1.5, 1e-8),
    ("clayton-lambdaL", lambda: cp.Clayton(2), "lambda_l", 2**-0.5, 1e-6),
    ("gh-tau-2", lambda: cp.GumbelHougaard(2), "tau", 0.5, 1e-9),
    ("gh-tau-4", lambda: cp.GumbelHougaard(4), "tau", 0.75, 1e-8),
    ("gh-lambdaU", lambda: cp.GumbelHougaard(3), "lambda_u", 2 - 2 ** (1 / 3), 1e-6),
    ("ghev-tau-3", lambda: cp.GumbelHougaardEV(3), "tau", 2 / 3, 1e-8),
    ("joe-tau-2", lambda: cp.Joe(2), "tau", 2 - math.pi**2 / 6, 1e-9),
    ("frank-tau-2", lambda: cp.Frank(2), "tau", frank_tau(2), 1e-9),
    ("frank-rho-2", lambda: cp.Frank(2), "rho", frank_rho(2), 1e-9),
    ("frank-tau-neg", lambda: cp.Frank(-5), "tau", frank_tau(-5), 1e-9),
    ("frank-rho-8", lambda: cp.Frank(8), "rho", frank_rho(8), 1e-8),
    ("amh-tau", lambda: cp.AliMikhailHaq(0.5), "tau", amh_tau(0.5), 1e-9),
    ("amh-rho", lambda: cp.AliMikhailHaq(0.5), "rho", amh_rho(0.5), 1e-9),
    ("amh-rho-neg", lambda: cp.AliMikhailHaq(-0.7), "rho", amh_rho(-0.7), 1e-9),
    ("plackett-rho", lambda: cp.Plackett(3), "rho", plackett_rho(3), 1e-9),
    ("plackett-rho-small", lambda: cp.Plackett(0.2), "rho", plackett_rho(0.2), 1e-9),
    ("fgm-rho", lambda: cp.FarlieGumbelMorgenstern(0.6), "rho", 0.2, 1e-10),
    ("fgm-tau", lambda: cp.FarlieGumbelMorgenstern(0.6), "tau", 2 * 0.6 / 9, 1e-10),
    ("fgm-xi", lambda: cp.FarlieGumbelMorgenstern(0.6), "xi", 0.36 / 15, 1e-10),
    ("fgm-xi2", lambda: cp.FarlieGumbelMorgenstern(-0.6), "xi_2", 0.36 / 15, 1e-10),
    ("fgm-footrule", lambda: cp.FarlieGumbelMorgenstern(0.6), "footrule", 0.12, 1e-10),
    ("fgm-gamma", lambda: cp.FarlieGumbelMorgenstern(0.6), "gamma", 4 * 0.6 / 15, 1e-10),
    ("fgm-beta", lambda: cp.FarlieGumbelMorgenstern(0.6), "beta", 0.15, 1e-12),
    ("fgm-nu", lambda: cp.FarlieGumbelMorgenstern(0.6), "nu", 0.2, 1e-10),
    ("fgm-phi2", lambda: cp.FarlieGumbelMorgenstern(0.6), "hoeffdings_d", 0.036, 1e-10),
    ("fgm-sigma", lambda: cp.FarlieGumbelMorgenstern(-0.6), "sigma", 0.2, 1e-9),
    ("fgm-bkr", lambda: cp.FarlieGumbelMorgenstern(0.6), "bkr", 0.36 / 30, 1e-10),
    ("fgm-kappa", lambda: cp.FarlieGumbelMorgenstern(0.6), "kappa", 4 * 0.6 / 16, 1e-8),
    ("mo-tau", lambda: cp.MarshallOlkin(0.8, 0.5), "tau", 0.4 / (0.8 + 0.5 - 0.4), 1e-6),
    ("mo-rho", lambda: cp.MarshallOlkin(0.8, 0.5), "rho", 1.2 / (1.6 + 1.0 - 0.4), 1e-6),
    ("mo-lambdaU", lambda: cp.MarshallOlkin(0.8, 0.5), "lambda_u", 0.5, 1e-6),
    ("ca-tau", lambda: cp.CuadrasAuge(0.5), "tau", 0.5 / 1.5, 1e-6),
    ("ca-rho", lambda: cp.CuadrasAuge(0.5), "rho", 1.5 / 3.5, 1e-6),
    ("galambos-lambdaU", lambda: cp.Galambos(1), "lambda_u", 2**-1, 1e-6),
]

for r in (0.5, -0.3, 0.9):
    for k, v in gaussian(r).items():
        CASES.append((f"gauss-{k}-{r}", (lambda r=r: cp.Gaussian(r)), k, v, 1e-9))

for a, b in ((0.5, 0.2), (0.1, 0.6)):
    for k, v in frechet(a, b).items():
        CASES.append((f"frechet-{k}-{a}-{b}", (lambda a=a, b=b: cp.Frechet(a, b)), k, v, 1e-6))

for th in (0.4, -0.7):
    a_, b_ = th**2 * (1 + th) / 2, th**2 * (1 - th) / 2
    CASES.append((f"mardia-rho-{th}", (lambda th=th: cp.Mardia(th)), "rho", th**3, 1e-6))
    CASES.append(
        (
            f"mardia-tau-{th}",
            (lambda th=th: cp.Mardia(th)),
            "tau",
            (a_ - b_) * (a_ + b_ + 2) / 3,
            1e-6,
        )
    )


@pytest.mark.parametrize("cid, factory, key, exact, tol", CASES, ids=[c[0] for c in CASES])
def test_numeric_matches_closed_form(cid, factory, key, exact, tol):
    res = factory().measure(key, method="numeric", full_output=True)
    assert res.method == "numeric"
    assert isinstance(res.value, float)
    assert res.value == pytest.approx(exact, abs=tol), res
    # error estimates are honest (within a generous factor)
    if key not in ("beta", "kappa"):
        assert abs(res.value - exact) <= max(10 * res.error, tol)


@pytest.mark.parametrize(
    "cid, factory, key, exact, tol",
    [c for c in CASES if not c[2].startswith("lambda")],
    ids=[c[0] for c in CASES if not c[2].startswith("lambda")],
)
def test_auto_matches_closed_form(cid, factory, key, exact, tol):
    val = factory().measure(key)
    assert isinstance(val, float)
    assert val == pytest.approx(exact, abs=tol)


BOUNDS = {
    "M": (lambda: cp.UpperFrechet(), "at_M"),
    "W": (lambda: cp.LowerFrechet(), "at_W"),
    "Pi": (lambda: cp.BivIndependenceCopula(), "at_Pi"),
}
BOUND_KEYS = [
    "xi",
    "xi_2",
    "rho",
    "tau",
    "footrule",
    "gamma",
    "beta",
    "nu",
    "hoeffdings_d",
    "sigma",
    "kappa",
    "lp",
    "bkr",
    "lambda_l",
    "lambda_u",
]


@pytest.mark.parametrize("which", list(BOUNDS))
@pytest.mark.parametrize("key", BOUND_KEYS)
def test_registry_values_at_bounds(which, key):
    from copul.measures import get_measure

    factory, attr = BOUNDS[which]
    expected = getattr(get_measure(key), attr)
    val = factory().measure(key, method="numeric")
    assert val == pytest.approx(expected, abs=2e-6)
