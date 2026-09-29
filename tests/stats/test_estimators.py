import itertools

import numpy as np
import pytest
from scipy import stats

import copul as cp
from copul import stats as cs
from copul.chatterjee import xi_ncalculate
from copul.measures import compute


@pytest.fixture(scope="module")
def small():
    rng = np.random.default_rng(42)
    x = rng.normal(size=9)
    y = x + rng.normal(size=9)
    return x, y


def _ranks(x):
    return stats.rankdata(x)


def test_tau_brute_force(small):
    x, y = small
    n = x.size
    s = sum(
        np.sign(x[i] - x[j]) * np.sign(y[i] - y[j]) for i, j in itertools.combinations(range(n), 2)
    )
    tau_a = 2 * s / (n * (n - 1))
    assert cs.sample_tau(x, y) == pytest.approx(tau_a)
    assert cs.sample_tau(x, y, variant="a") == pytest.approx(tau_a)


def test_tau_with_ties():
    x = np.array([1, 2, 2, 3, 4, 4, 5.0])
    y = np.array([1, 3, 2, 2, 5, 4, 4.0])
    n = x.size
    s = sum(
        np.sign(x[i] - x[j]) * np.sign(y[i] - y[j]) for i, j in itertools.combinations(range(n), 2)
    )
    assert cs.sample_tau(x, y, variant="a") == pytest.approx(2 * s / (n * (n - 1)))
    assert cs.sample_tau(x, y) == pytest.approx(stats.kendalltau(x, y)[0])
    assert cs.sample_rho(x, y) == pytest.approx(stats.spearmanr(x, y)[0])


def test_rank_formulas(small):
    x, y = small
    n = x.size
    r, s = _ranks(x), _ranks(y)
    assert cs.sample_rho(x, y) == pytest.approx(1 - 6 * np.sum((r - s) ** 2) / (n * (n * n - 1)))
    assert cs.sample_footrule(x, y) == pytest.approx(1 - 3 * np.sum(np.abs(r - s)) / (n * n - 1))
    assert cs.sample_gamma(x, y) == pytest.approx(
        np.sum(np.abs(r + s - n - 1) - np.abs(r - s)) / np.floor(n * n / 2)
    )
    # Blomqvist: 4 C_n(1/2, 1/2) - 1 with pseudo-observations R/(n+1)
    u, v = r / (n + 1), s / (n + 1)
    assert cs.sample_beta(x, y) == pytest.approx(4 * np.mean((u <= 0.5) & (v <= 0.5)) - 1)
    # Blest (Genest & Plante 2003)
    nu = (2 * n + 1) / (n - 1) - 12 / (n * n - n) * np.sum((1 - r / (n + 1)) ** 2 * s)
    assert cs.sample_nu(x, y) == pytest.approx(nu)
    assert cs.sample_xi(x, y) == xi_ncalculate(x, y)
    assert cs.sample_xi_2(x, y) == xi_ncalculate(y, x)


def test_distance_measures_brute_force(small):
    x, y = small
    n = x.size
    u, v = _ranks(x) / n, _ranks(y) / n

    def cn(a, b):
        return np.mean((u[None, :] <= a[:, None]) & (v[None, :] <= b[:, None]), axis=1)

    # Schweizer-Wolff on the rank grid
    i, j = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1), indexing="ij")
    grid = cn((i / n).ravel(), (j / n).ravel()).reshape(i.shape)
    sigma = 12 / (n * n - 1) * np.sum(np.abs(grid - i * j / n**2))
    assert cs.sample_sigma(x, y) == pytest.approx(sigma)
    lp3 = cp.measures.lp_constant(3) / n**2 * np.sum(np.abs(grid - i * j / n**2) ** 3)
    assert cs.sample_lp(x, y, p=3) == pytest.approx(lp3)
    # Phi^2 and kappa against a fine midpoint grid (C_n is a step function)
    m = 900
    t = (np.arange(m) + 0.5) / m
    a, b = np.meshgrid(t, t, indexing="ij")
    d = cn(a.ravel(), b.ravel()) - a.ravel() * b.ravel()
    assert cs.sample_hoeffdings_d(x, y) == pytest.approx(90 * np.mean(d**2), rel=5e-3)
    kappa = cs.sample_kappa(x, y)
    assert kappa >= 4 * np.max(np.abs(d)) - 1e-12
    assert kappa == pytest.approx(4 * np.max(np.abs(d)), abs=4 * 2.5 / m)
    assert cs.cramer_von_mises_independence(x, y) == pytest.approx(
        n * cs.sample_hoeffdings_d(x, y) / 90
    )


def _psi(a, b, c):
    return float(a >= b) - float(a >= c)


def test_hoeffding_d_matches_u_statistic_definition():
    rng = np.random.default_rng(7)
    x = rng.normal(size=7)
    y = x + rng.normal(size=7)
    n = x.size
    tot = 0.0
    cnt = 0
    for idx in itertools.permutations(range(n), 5):
        i1, i2, i3, i4, i5 = idx
        tot += (
            0.25
            * _psi(x[i1], x[i2], x[i3])
            * _psi(x[i1], x[i4], x[i5])
            * _psi(y[i1], y[i2], y[i3])
            * _psi(y[i1], y[i4], y[i5])
        )
        cnt += 1
    assert cs.sample_bkr(x, y) == pytest.approx(30 * tot / cnt)


def test_comonotone_and_countermonotone_samples():
    x = np.arange(1.0, 51.0)
    for y, sign in [(x, 1), (-x, -1)]:
        for key in ["rho", "tau", "gamma", "beta", "nu"]:
            assert cs.sample_measure(x, y, key) == pytest.approx(sign), key
        for key in ["sigma", "kappa", "bkr"]:
            assert cs.sample_measure(x, y, key) == pytest.approx(1.0), key
    assert cs.sample_footrule(x, x) == pytest.approx(1.0)
    assert cs.sample_footrule(x, -x) == pytest.approx(-0.5, abs=1e-3)
    assert cs.sample_lambda_l(x, x) == 1.0
    assert cs.sample_lambda_u(x, -x) == 0.0


def test_tail_estimators():
    rng = np.random.default_rng(0)
    x, y = rng.random(100), rng.random(100)
    r, s = _ranks(x), _ranks(y)
    assert cs.sample_lambda_l(x, y, k=20) == np.count_nonzero((r <= 20) & (s <= 20)) / 20
    assert cs.sample_lambda_u(x, y, k=20) == np.count_nonzero((r > 80) & (s > 80)) / 20
    with pytest.raises(ValueError):
        cs.sample_lambda_l(x, y, k=0)
    with pytest.raises(ValueError):
        cs.sample_lambda_u(x, y, method="nope")
    # CFG for an extreme-value (Gumbel) sample: lambda_U = 2 - 2^(1/theta)
    X = cs.sample(cp.GumbelHougaard(theta=2), 4000, random_state=1)
    lam = 2 - 2**0.5
    assert cs.sample_lambda_u(X[:, 0], X[:, 1], method="cfg") == pytest.approx(lam, abs=0.03)
    # Schmidt-Stadtmueller for Clayton: lambda_L = 2^(-1/theta)
    X = cs.sample(cp.Clayton(theta=2), 4000, random_state=2)
    assert cs.sample_lambda_l(X[:, 0], X[:, 1], k=150) == pytest.approx(2**-0.5, abs=0.1)


def test_mutual_information_gaussian():
    X = cs.sample(cp.Gaussian(rho=0.7), 3000, random_state=0)
    mi = -0.5 * np.log(1 - 0.7**2)
    assert cs.sample_mutual_information(X[:, 0], X[:, 1], random_state=0) == pytest.approx(
        mi, abs=0.04
    )


def test_input_validation():
    with pytest.raises(ValueError):
        cs.sample_rho([1, 2, 3], [1, 2])
    with pytest.raises(ValueError):
        cs.sample_rho([1.0, np.nan], [1.0, 2.0])
    with pytest.raises(ValueError):
        cs.sample_bkr([1, 2, 3], [3, 2, 1])
    with pytest.raises(KeyError):
        cs.sample_measure([1, 2, 3], [1, 2, 3], "no_such_measure")


_CONSISTENCY_TOL = {
    "xi": 0.04,
    "xi_2": 0.04,
    "rho": 0.03,
    "tau": 0.025,
    "footrule": 0.03,
    "gamma": 0.03,
    "beta": 0.05,
    "nu": 0.035,
    "hoeffdings_d": 0.035,
    "sigma": 0.03,
    "kappa": 0.05,
    "bkr": 0.02,
    "mutual_information": 0.04,
}


@pytest.fixture(scope="module")
def samples():
    cops = {
        "clayton": cp.Clayton(theta=2),
        "gaussian": cp.Gaussian(rho=0.6),
        "frank": cp.Frank(theta=-4),
    }
    return {k: (c, cs.sample(c, 3000, random_state=11)) for k, c in cops.items()}


@pytest.mark.parametrize("family", ["clayton", "gaussian", "frank"])
@pytest.mark.parametrize("key", sorted(_CONSISTENCY_TOL))
def test_consistency(samples, family, key):
    cop, X = samples[family]
    est = cs.sample_measure(X[:, 0], X[:, 1], key, random_state=0)
    true = float(compute(cop, key))
    assert est == pytest.approx(true, abs=_CONSISTENCY_TOL[key])
