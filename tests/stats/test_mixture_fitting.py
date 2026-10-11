import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

import copul as cp
from copul.stats.fitting import _fit_mle, _fixed_params


def test_fixed_component_weights_match_independent_scalar_optimization():
    components = [cp.Gaussian(rho=0.8), cp.Clayton(theta=0.8)]
    U = cp.mixture(components, [0.65, 0.35]).rvs(1200, random_state=51)
    result = cp.fit_mixture(U, components, pseudo_obs=False, n_init=1, max_iter=500, tol=1e-10)
    lp = np.column_stack([c.logpdf(*U.T) for c in components])
    reference = minimize_scalar(
        lambda w: -logsumexp(lp + np.log([w, 1 - w]), axis=1).sum(),
        bounds=(0.001, 0.999),
        method="bounded",
    )
    assert result.converged
    assert_allclose(result.weights[0], reference.x, atol=3e-5)
    assert_allclose(result.loglik, -reference.fun, atol=1e-6)
    assert np.all(np.diff(result.loglik_path) >= 0)
    assert_allclose(result.loglik, result.copula.logpdf(*U.T).sum(), atol=1e-9)
    posterior = np.exp(
        lp + np.log(result.weights) - logsumexp(lp + np.log(result.weights), axis=1)[:, None]
    )
    assert_allclose(result.predict_proba(U), posterior, atol=1e-12)
    assert result.predict_proba(U[:1]).shape == (1, 2)
    assert result.k == 1
    assert result.component_params == ({}, {})
    assert result.aic == pytest.approx(2 - 2 * result.loglik)
    assert result.bic == pytest.approx(np.log(len(U)) - 2 * result.loglik)


def test_free_gaussian_components_recover_opposite_dependence():
    U = cp.mixture([cp.Gaussian(rho=0.8), cp.Gaussian(rho=-0.7)], [0.6, 0.4]).rvs(
        2000, random_state=38
    )
    result = cp.fit_mixture(
        U,
        [cp.Gaussian, cp.Gaussian],
        pseudo_obs=False,
        start=[{"rho": 0.6}, {"rho": -0.5}],
        n_init=1,
    )
    assert result.converged
    assert result.k == 3
    assert abs(result.component_params[0]["rho"] - 0.8) < 0.05
    assert abs(result.component_params[1]["rho"] + 0.7) < 0.06
    assert_allclose(result.weights, [0.6, 0.4], atol=0.06)
    assert_allclose(result.predict_proba(U).sum(axis=1), 1, atol=1e-14)
    assert np.all(np.diff(result.loglik_path) >= 0)
    sample = result.rvs(4000, random_state=23)
    assert_allclose(sample.mean(axis=0), 0.5, atol=0.025)
    assert "converged=True" in result.summary()


def test_multiple_starts_are_reproducible_and_best_likelihood_is_retained():
    U = cp.mixture([cp.Gaussian(rho=0.85), cp.Gaussian(rho=-0.8)], [0.5, 0.5]).rvs(
        600, random_state=22
    )
    kwargs = dict(pseudo_obs=False, n_init=3, random_state=42, max_iter=100)
    a = cp.fit_mixture(U, [cp.Gaussian, cp.Gaussian], **kwargs)
    b = cp.fit_mixture(U, [cp.Gaussian, cp.Gaussian], **kwargs)
    assert_allclose(a.weights, b.weights)
    assert_allclose(a.loglik_path, b.loglik_path)
    assert a.loglik == max(s["loglik"] for s in a.starts)
    assert a.loglik > cp.fit(cp.Gaussian, U, pseudo_obs=False).loglik + 50
    assert len(a.starts) == 3


def test_one_component_reduces_to_ordinary_fit_and_fixed_weights_are_preserved():
    U = cp.Clayton(theta=2).rvs(500, random_state=12)
    ordinary = cp.fit(cp.Clayton, U, pseudo_obs=False)
    mix = cp.fit_mixture(U, [cp.Clayton], pseudo_obs=False, n_init=1)
    assert mix.converged
    assert mix.k == 1
    assert_allclose(mix.loglik, ordinary.loglik, atol=1e-7)
    result = cp.fit_mixture(
        U,
        [cp.Clayton, "Independence"],
        weights=[3, 1],
        fit_weights=False,
        pseudo_obs=False,
        n_init=2,
        random_state=7,
    )
    assert_allclose(result.weights, [0.75, 0.25])
    assert result.k == 1


def test_reflected_families_partial_fixed_parameters_and_rank_equivalence():
    spec = cp.PairCopulaFamily(cp.Clayton, rotation=90)
    U = spec.wrap(cp.Clayton(theta=2)).rvs(400, random_state=8)
    a = cp.fit_mixture(U, [spec], n_init=1)
    b = cp.fit_mixture(cp.pseudo_obs(U), [spec], pseudo_obs=False, n_init=1)
    assert_allclose(a.loglik, b.loglik, atol=1e-8)
    assert a.component_params[0]["theta"] == pytest.approx(2, abs=0.35)
    T = cp.StudentT(rho=0.5, nu=4).rvs(500, random_state=7)
    fixed = cp.fit_mixture(T, [cp.StudentT(nu=4)], n_init=1, pseudo_obs=False)
    assert fixed.k == 1
    assert float(fixed.copula.copulas[0].nu) == 4


def test_weighted_component_fit_matches_replication_and_ignores_zero_weight_support():
    U = cp.Gaussian(rho=0.4).rvs(80, random_state=5)
    w = np.random.default_rng(16).integers(0, 4, len(U))
    base = cp.Gaussian()
    weighted = _fit_mle(
        base, "Gaussian", U, _fixed_params(base), len(U), None, None, None, light=True, weights=w
    )
    repeated = cp.fit(cp.Gaussian, np.repeat(U, w, axis=0), pseudo_obs=False, _light=True)
    assert_allclose(weighted.params["rho"], repeated.params["rho"], atol=1e-7)
    assert_allclose(weighted.loglik, repeated.loglik, atol=1e-8)
    # A zero-weight point lies outside the support of negative Clayton.
    X = np.vstack(([0.001, 0.001], cp.Clayton(theta=-0.3).rvs(100, random_state=3)))
    cw = np.r_[0, np.ones(100)]
    result = _fit_mle(
        cp.Clayton(), "Clayton", X, {}, len(X), {"theta": -0.25}, None, None, light=True, weights=cw
    )
    assert np.isfinite(result.loglik)


def test_mixture_logpdf_remains_finite_when_component_densities_underflow():
    components = [cp.Gaussian(rho=0.995), cp.Gaussian(rho=0.99)]
    copula = cp.mixture(components, [0.3, 0.7])
    u, v = np.array([1e-10, 1e-8]), np.array([1 - 1e-10, 1 - 1e-8])
    expected = logsumexp(
        np.stack([c.logpdf(u, v) + np.log(w) for c, w in zip(components, [0.3, 0.7])]), axis=0
    )
    assert np.all(np.isfinite(expected))
    assert np.all(copula.pdf(u, v) == 0)
    assert_allclose(copula.logpdf(u, v), expected, atol=1e-8)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weights": [0, 1]},
        {"weights": [1]},
        {"weights": [np.nan, 1]},
        {"n_init": 0},
        {"max_iter": 0},
        {"n_init": 1.5},
        {"tol": 0},
        {"start": [{"bad": 0.5}, None]},
        {"start": [{"rho": 1.5}, None]},
    ],
)
def test_invalid_mixture_options(kwargs):
    with pytest.raises(ValueError):
        cp.fit_mixture(
            np.random.default_rng(2).random((30, 2)), [cp.Gaussian, cp.Gaussian], **kwargs
        )


def test_singular_components_and_iteration_limit():
    U = cp.Clayton(theta=2).rvs(200, random_state=2)
    with pytest.raises(ValueError):
        cp.fit_mixture(U, [cp.UpperFrechet()])
    with pytest.raises(ValueError):
        cp.fit_mixture(U, [])
    result = cp.fit_mixture(U, [cp.Gaussian, cp.Clayton], n_init=1, max_iter=1, tol=1e-14)
    assert not result.converged
    assert result.n_iter == 1
    assert "Maximum" in result.message
