import numpy as np
import pytest

import copul as cp
from copul import stats as cs


@pytest.fixture(scope="module")
def clayton_sample():
    return cs.sample(cp.Clayton(theta=2), 500, random_state=0)


def test_estimate_table(clayton_sample):
    df = cs.estimate(clayton_sample)
    assert list(df.columns) == ["estimate", "se", "ci_low", "ci_high", "ci_method", "n"]
    assert list(df.index) == ["xi", "rho", "tau", "footrule", "gamma", "beta", "nu"]
    assert df["se"].isna().all()
    assert (df["n"] == 500).all()
    assert df.loc["tau", "estimate"] == pytest.approx(cs.sample_tau(*clayton_sample.T))


def test_estimate_asymptotic_and_bootstrap(clayton_sample):
    keys = ["tau", "rho", "xi", "beta", "footrule", "lambda_l"]
    df = cs.estimate(clayton_sample, keys, ci="asymptotic", n_boot=60, random_state=1)
    assert list(df["ci_method"]) == ["asymptotic"] * 4 + ["bootstrap"] * 2
    assert (df["ci_low"] <= df["estimate"]).all()
    assert (df["estimate"] <= df["ci_high"]).all()
    # tau: se from the U-statistic variance
    se = np.sqrt(cs.asymptotic_variance(clayton_sample, "tau") / 500)
    assert df.loc["tau", "se"] == pytest.approx(se)
    df2 = cs.estimate(clayton_sample, keys, ci="asymptotic", n_boot=60, random_state=1)
    assert df.equals(df2)  # seeded
    dfb = cs.estimate(clayton_sample, ["tau", "rho"], ci="bootstrap", n_boot=200, random_state=0)
    # bootstrap and asymptotic standard errors agree roughly
    assert dfb.loc["tau", "se"] == pytest.approx(df.loc["tau", "se"], rel=0.3)
    assert dfb.loc["rho", "se"] == pytest.approx(df.loc["rho", "se"], rel=0.3)
    with pytest.raises(ValueError):
        cs.estimate(clayton_sample, ci="jackknife")
    with pytest.raises(ValueError):
        cs.asymptotic_variance(clayton_sample, "gamma")


def test_estimate_accepts_empirical_copula(clayton_sample):
    ec = cs.EmpiricalCopula(clayton_sample)
    assert ec.estimate(["rho"]).loc["rho", "estimate"] == pytest.approx(ec.spearmans_rho())


def test_null_variances():
    X = np.random.default_rng(0).random((50, 2))
    assert cs.asymptotic_variance(X, "tau", null=True) == pytest.approx(2 * 105 / (9 * 49))
    assert cs.asymptotic_variance(X, "rho", null=True) == pytest.approx(50 / 49)
    assert cs.asymptotic_variance(X, "xi", null=True) == pytest.approx(0.4)
    assert cs.asymptotic_variance(X, "beta", null=True) == 1.0


def test_asymptotic_variances_under_independence_are_consistent():
    X = np.random.default_rng(1).random((4000, 2))
    # tau: 4/9, rho: 1, beta: 1
    assert cs.asymptotic_variance(X, "tau") == pytest.approx(4 / 9, rel=0.1)
    assert cs.asymptotic_variance(X, "rho") == pytest.approx(1.0, rel=0.1)
    assert cs.asymptotic_variance(X, "beta") == pytest.approx(1.0, rel=0.15)


@pytest.mark.parametrize("key", ["tau", "rho", "beta"])
def test_asymptotic_ci_coverage(key):
    cop = cp.Frank(theta=4)
    true = float(cp.compute_measures(cop, [key])[key])
    rng = np.random.default_rng(123)
    reps = 150
    hits = 0
    for _ in range(reps):
        X = cs.sample(cop, 200, random_state=rng)
        row = cs.estimate(X, [key], ci="asymptotic").loc[key]
        hits += row["ci_low"] <= true <= row["ci_high"]
    assert 0.87 <= hits / reps <= 0.995


@pytest.mark.parametrize("method", ["tau", "rho", "xi", "hoeffding", "cvm"])
def test_independence_tests(method):
    dep = cs.sample(cp.Gaussian(rho=0.6), 200, random_state=0)
    ind = np.random.default_rng(5).random((150, 2))
    r1 = cs.independence_test(dep, method, n_perm=199, random_state=0)
    r0 = cs.independence_test(ind, method, n_perm=199, random_state=0)
    assert r1.pvalue < 0.01
    assert r1.reject()
    assert r0.pvalue > 0.05
    assert 0 < r0.pvalue <= 1
    assert r1.method == method
    assert method in repr(r1)


def test_independence_test_options():
    X = cs.sample(cp.Clayton(theta=1), 100, random_state=0)
    a = cs.independence_test(X, "tau", null_distribution="permutation", n_perm=99, random_state=1)
    b = cs.independence_test(X, "tau", null_distribution="permutation", n_perm=99, random_state=1)
    assert a.pvalue == b.pvalue == pytest.approx(1 / 100)
    less = cs.independence_test(X, "rho", alternative="less")
    assert less.pvalue > 0.99
    with pytest.raises(ValueError):
        cs.independence_test(X, "cvm", null_distribution="asymptotic")
    with pytest.raises(ValueError):
        cs.independence_test(X, "xi", alternative="less")
    with pytest.raises(ValueError):
        cs.independence_test(X, "pearson")


def test_independence_pvalues_roughly_uniform():
    rng = np.random.default_rng(2)
    p = np.array([cs.independence_test(rng.random((80, 2)), "tau").pvalue for _ in range(200)])
    assert 0.02 <= np.mean(p < 0.05) <= 0.1
