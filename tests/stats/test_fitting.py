import warnings

import numpy as np
import pytest

import copul as cp
from copul import stats as cs
from copul.stats._adapters import ParametricLogDensity, logpdf, sample


@pytest.mark.parametrize(
    "family, params, tol",
    [
        (cp.Clayton, {"theta": 2.0}, {"theta": 0.25}),
        (cp.Gaussian, {"rho": 0.6}, {"rho": 0.04}),
        (cp.GumbelHougaard, {"theta": 1.8}, {"theta": 0.12}),
        (cp.Frank, {"theta": 5.0}, {"theta": 0.5}),
        (cp.StudentT, {"rho": 0.6, "nu": 4.0}, {"rho": 0.05, "nu": 1.6}),
    ],
)
def test_mle_recovers_parameters(family, params, tol):
    X = sample(family(**params), 2000, random_state=2024)
    res = cs.fit(family, X)
    assert res.method == "mle"
    assert res.converged
    assert res.k == len(params)
    for p, v in params.items():
        assert res.params[p] == pytest.approx(v, abs=tol[p])
        # the truth is within ~4 standard errors
        assert abs(res.params[p] - v) < 4.5 * res.se[p]
    assert np.isfinite(res.loglik) and res.loglik > 0
    assert res.aic == pytest.approx(2 * res.k - 2 * res.loglik)
    assert res.bic == pytest.approx(res.k * np.log(2000) - 2 * res.loglik)
    assert res.loglik == pytest.approx(cs.loglik(res.copula, X), rel=1e-8)


def test_mle_with_fixed_parameter():
    X = sample(cp.StudentT(rho=0.5, nu=5), 1000, random_state=5)
    res = cs.fit(cp.StudentT(nu=5), X)
    assert list(res.params) == ["rho"]
    assert res.fixed == {"nu": 5.0}
    assert abs(res.params["rho"] - 0.5) < 4 * res.se["rho"]
    assert float(res.copula.nu) == 5.0


def test_itau_closed_forms():
    X = sample(cp.Clayton(theta=3), 800, random_state=1)
    tau = cs.sample_tau(*X.T)
    rho = cs.sample_rho(*X.T)
    assert cs.fit(cp.Clayton, X, method="itau").params["theta"] == pytest.approx(
        2 * tau / (1 - tau), rel=1e-7
    )
    assert cs.fit(cp.GumbelHougaard, X, method="itau").params["theta"] == pytest.approx(
        1 / (1 - tau), rel=1e-7
    )
    assert cs.fit(cp.Gaussian, X, method="itau").params["rho"] == pytest.approx(
        np.sin(np.pi * tau / 2), rel=1e-7
    )
    assert cs.fit(cp.Gaussian, X, method="irho").params["rho"] == pytest.approx(
        2 * np.sin(np.pi * rho / 6), rel=1e-6
    )
    r = cs.fit("Clayton", X, method="itau")
    assert r.extra["tau"] == pytest.approx(tau)
    assert np.isfinite(r.se["theta"]) and r.se["theta"] > 0
    assert np.isfinite(r.loglik)


def test_moment_methods_ixi_ibeta():
    X = sample(cp.Frank(theta=6), 1500, random_state=4)
    for m in ["ixi", "ibeta"]:
        r = cs.fit(cp.Frank, X, method=m)
        assert r.params["theta"] == pytest.approx(6, abs=1.8), m


def test_ixi_negative_dependence_branch():
    X = sample(cp.Frank(theta=-6), 1500, random_state=4)
    r = cs.fit(cp.Frank, X, method="ixi")
    assert r.params["theta"] == pytest.approx(-6, abs=1.8)


def test_itau_out_of_range_is_clipped():
    X = sample(cp.Clayton(theta=-0.5), 500, random_state=0)  # negative dependence
    with pytest.warns(UserWarning):
        r = cs.fit(cp.GumbelHougaard, X, method="itau")
    assert r.params["theta"] == pytest.approx(1.0, abs=0.05)
    assert r.extra["clipped"]
    assert not r.converged


def test_itau_needs_one_parameter():
    X = sample(cp.Gaussian(rho=0.3), 200, random_state=0)
    with pytest.raises(ValueError, match="exactly one free parameter"):
        cs.fit(cp.StudentT, X, method="itau")
    r = cs.fit(cp.StudentT(nu=4), X, method="itau")
    assert r.params["rho"] == pytest.approx(np.sin(np.pi * cs.sample_tau(*X.T) / 2))


def test_singular_family():
    X = sample(cp.Clayton(theta=2), 300, random_state=0)
    with pytest.raises(cs.SingularFamilyError, match="itau"):
        cs.fit(cp.MarshallOlkin, X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = cs.fit(cp.CuadrasAuge, X, method="itau")
    assert 0 < r.params["delta"] < 1


def test_family_resolution():
    X = sample(cp.Frank(theta=3), 300, random_state=0)
    a = cs.fit("Frank", X).params["theta"]
    b = cs.fit("FRANK", X).params["theta"]
    c = cs.fit(cp.Frank(), X).params["theta"]
    assert a == pytest.approx(b) == pytest.approx(c)
    assert cs.fit("gumbel_hougaard", X).family == "gumbel_hougaard"
    with pytest.raises(ValueError):
        cs.fit("NoSuchFamily", X)
    with pytest.raises(ValueError):
        cs.fit(cp.Frank, X, method="least_squares")


def test_fitresult_output():
    X = sample(cp.Clayton(theta=2), 400, random_state=0)
    r = cs.fit(cp.Clayton, X)
    s = r.summary()
    assert "Clayton" in s and "theta" in s and "AIC" in s
    assert "Clayton" in repr(r)
    df = r.to_frame()
    assert list(df.columns) == ["estimate", "se", "ci_low", "ci_high"]
    lo, hi = r.conf_int()["theta"]
    assert lo < r.params["theta"] < hi
    assert r.rvs(5, random_state=0).shape == (5, 2)


def test_pseudo_obs_false_and_start():
    U = sample(cp.Clayton(theta=2), 500, random_state=0)
    r1 = cs.fit(cp.Clayton, U, pseudo_obs=False)
    r2 = cs.fit(cp.Clayton, U, pseudo_obs=False, start={"theta": 5.0})
    assert r1.params["theta"] == pytest.approx(r2.params["theta"], rel=1e-4)
    with pytest.raises(ValueError):
        cs.fit(cp.Clayton, U * 2, pseudo_obs=False)


def test_parametric_log_density_matches_instances():
    base = cp.GumbelHougaard()
    grid = 1 + np.geomspace(1e-2, 1e2, 25)
    dens = ParametricLogDensity(
        base,
        ["theta"],
        lambda t: base(theta=float(t[0])),
        center=[2.0],
        grids=[grid],
        bounds=[(1.0, np.inf)],
    )
    # with the unified vectorized ``logpdf`` the per-instance path is used;
    # the compiled symbolic density is only a fallback
    if dens.source == "symbolic":
        assert dens.trusted([1.5]) and dens.trusted([7.0])
    u = np.array([0.1, 0.3, 0.5, 0.9])
    v = np.array([0.2, 0.7, 0.5, 0.95])
    for theta in [1.2, 2.5, 7.0]:
        np.testing.assert_allclose(
            dens([theta], u, v), logpdf(cp.GumbelHougaard(theta=theta), u, v), rtol=1e-8
        )


def test_parametric_log_density_trusted_region_clayton():
    base = cp.Clayton()
    dens = ParametricLogDensity(
        base,
        ["theta"],
        lambda t: base(theta=float(t[0])),
        center=[1.0],
        grids=[-1 + np.geomspace(1e-2, 1e2, 25)],
        bounds=[(-1.0, np.inf)],
    )
    if dens.source == "symbolic":
        # theta = 0 is a removable singularity of the closed-form density
        assert not dens.trusted([0.0])
        assert dens.trusted([2.0])
    u = np.array([0.05, 0.5])
    v = np.array([0.08, 0.6])
    # outside the support of Clayton(-0.45) the density vanishes (the closed
    # form is undefined there; only at special values such as -1/2 it is not)
    lp = dens([-0.45], u, v)
    assert lp[0] == -np.inf and np.isfinite(lp[1])


def test_select_picks_true_family():
    fams = ["Clayton", "Frank", "GumbelHougaard", "Gaussian"]
    for true, cop in [
        ("Clayton", cp.Clayton(theta=2)),
        ("GumbelHougaard", cp.GumbelHougaard(theta=2)),
        ("Frank", cp.Frank(theta=6)),
    ]:
        X = sample(cop, 2000, random_state=7)
        df = cs.select(X, fams)
        assert df.loc[0, "family"] == true
        assert list(df.columns) == [
            "family",
            "params",
            "loglik",
            "aic",
            "bic",
            "k",
            "converged",
            "error",
            "fit",
        ]
        assert df["aic"].is_monotonic_increasing
    dfb = cs.select(X, fams, criterion="bic")
    assert dfb.loc[0, "family"] == "Frank"


def test_select_reports_failures():
    X = sample(cp.Clayton(theta=2), 300, random_state=0)
    df = cs.select(X, ["Clayton", "MarshallOlkin", "NoSuchFamily"])
    assert df.loc[0, "family"] == "Clayton"
    errs = df.set_index("family")["error"]
    assert "SingularFamilyError" in errs["MarshallOlkin"]
    assert "Unknown copula family" in errs["NoSuchFamily"]


def test_sampler_reproducible_and_correct():
    a = sample(cp.Gaussian(rho=0.5), 50, random_state=1)
    b = sample(cp.Gaussian(rho=0.5), 50, random_state=1)
    np.testing.assert_array_equal(a, b)
    X = sample(cp.Frank(theta=4), 5000, random_state=3)
    assert cs.sample_tau(*X.T) == pytest.approx(float(cp.Frank(theta=4).kendalls_tau()), abs=0.02)
    Y = sample(cp.Clayton(theta=2), 10, random_state=0, method="rvs")
    assert Y.shape == (10, 2)
