import importlib

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

import copul as cp


def test_bootstrap_moment_estimates_match_independent_paired_resampling():
    X = cp.Gaussian(rho=0.5).rvs(120, random_state=2)
    result = cp.bootstrap_fit(cp.Gaussian, X, fit_method="itau", n_boot=15, random_state=24)
    rng = np.random.default_rng(24)
    expected = []
    for _ in range(15):
        xb = X[rng.integers(0, len(X), len(X))]
        expected.append(np.sin(np.pi * stats.kendalltau(*xb.T).statistic / 2))
    assert_allclose(result.replicates[:, 0], expected, atol=1e-12)
    assert result.n_success == 15
    assert_allclose(result.cov, [[np.var(expected, ddof=1)]])
    assert result.se["rho"] == pytest.approx(np.std(expected, ddof=1))
    assert result.bias["rho"] == pytest.approx(np.mean(expected) - result.fit.params["rho"])
    quantiles = np.quantile(expected, [0.025, 0.975])
    assert_allclose(result.conf_int()["rho"], quantiles)
    assert_allclose(
        result.conf_int(method="basic")["rho"], 2 * result.fit.params["rho"] - quantiles[::-1]
    )
    assert list(result.to_frame().columns) == ["estimate", "se", "bias", "ci_low", "ci_high"]


@pytest.mark.parametrize("pseudo_obs", [True, False])
def test_mle_bootstrap_ranks_each_sample_only_when_requested(pseudo_obs):
    X = cp.Gaussian(rho=0.6).rvs(90, random_state=4)
    if pseudo_obs:
        X = stats.norm.ppf(X)
    result = cp.bootstrap_fit(cp.Gaussian, X, n_boot=6, random_state=33, pseudo_obs=pseudo_obs)
    rng = np.random.default_rng(33)
    for row in result.replicates:
        Xb = X[rng.integers(0, len(X), len(X))]
        expected = cp.fit(
            cp.Gaussian, Xb, pseudo_obs=pseudo_obs, start=result.fit.params, _light=True
        )
        assert_allclose(row, [expected.params["rho"]], atol=1e-10)


def test_parametric_bootstrap_respects_fixed_parameters_and_existing_fit():
    X = cp.StudentT(rho=0.5, nu=4).rvs(200, random_state=8)
    fitted = cp.fit(cp.StudentT(nu=4), X, pseudo_obs=False)
    result = fitted.bootstrap(X, n_boot=8, method="parametric", random_state=18)
    assert result.parameter_names == ("rho",)
    assert result.fit.fixed == {"nu": 4}
    assert result.replicates.shape == (8, 1)
    assert np.all(np.isfinite(result.replicates))
    again = cp.bootstrap_fit(fitted, X, n_boot=8, method="parametric", random_state=18)
    assert_allclose(result.replicates, again.replicates)
    with pytest.raises(ValueError, match="pseudo_obs"):
        fitted.bootstrap(X, n_boot=3, pseudo_obs=True)
    with pytest.raises(ValueError, match="fit_method"):
        fitted.bootstrap(X, n_boot=3, fit_method="itau")


def test_multiple_free_parameters_have_correct_covariance_shape():
    X = cp.StudentT(rho=0.6, nu=3).rvs(350, random_state=19)
    result = cp.bootstrap_fit(cp.StudentT, X, n_boot=5, random_state=11, pseudo_obs=False)
    assert result.cov.shape == (2, 2)
    assert set(result.conf_int()) == {"rho", "nu"}
    assert_allclose(result.cov, np.cov(result.replicates, rowvar=False, ddof=1))


def test_selected_fit_preserves_estimated_margin_convention():
    X = stats.norm.ppf(cp.Gaussian(rho=0.5).rvs(100, random_state=4))
    selected = cp.select(X, [cp.Gaussian]).iloc[0]["fit"]
    result = selected.bootstrap(X, n_boot=3, random_state=7)
    direct = cp.bootstrap_fit(cp.Gaussian, X, n_boot=3, random_state=7)
    assert_allclose(result.replicates, direct.replicates, atol=1e-10)


def test_failed_refits_raise_or_require_explicit_omission(monkeypatch):
    module = importlib.import_module("copul.stats.fit_bootstrap")
    X = cp.Gaussian(rho=0.5).rvs(100, random_state=2)
    original = cp.fit(cp.Gaussian, X, method="itau")
    real_fit = module.fit
    calls = []

    def fail_first(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("deliberate refit failure")
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(module, "fit", fail_first)
    with pytest.raises(RuntimeError, match="replicate 0"):
        cp.bootstrap_fit(original, X, n_boot=4, random_state=4)
    calls.clear()
    with pytest.warns(UserWarning, match="1 of 4"):
        result = cp.bootstrap_fit(original, X, n_boot=4, random_state=4, errors="omit")
    assert result.n_success == 3
    assert np.isnan(result.replicates[0]).all()
    assert "deliberate" in result.failures[0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_boot": 1},
        {"n_boot": 2.5},
        {"level": 1},
        {"ci_method": "bca"},
        {"method": "wild"},
        {"errors": "ignore"},
    ],
)
def test_bootstrap_validation(kwargs):
    with pytest.raises(ValueError):
        cp.bootstrap_fit(cp.Gaussian, np.random.default_rng(2).random((20, 2)), **kwargs)


def test_bootstrap_preserves_global_random_state_and_seed_reproducibility():
    X = cp.Gaussian(rho=0.4).rvs(80, random_state=1)
    np.random.seed(21)
    expected = np.random.random(4)
    np.random.seed(21)
    a = cp.bootstrap_fit(cp.Gaussian, X, fit_method="itau", n_boot=4, random_state=4)
    assert_allclose(np.random.random(4), expected)
    b = cp.bootstrap_fit(cp.Gaussian, X, fit_method="itau", n_boot=4, random_state=4)
    assert_allclose(a.replicates, b.replicates)
