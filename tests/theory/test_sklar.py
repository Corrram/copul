"""Sklar's theorem: JointDistribution and copula_from_joint."""

import numpy as np
import pytest
from scipy import stats

import copul as cp
from copul.exceptions import PropertyUnavailableException
from copul.multivariate import ClaytonND, GaussianND, StudentTND, UpperFrechetND
from copul.sklar import EmpiricalMarginal, JointDistribution, SklarCopula, copula_from_joint

RHO = 0.6
MU = np.array([1.0, -2.0])
SD = np.array([2.0, 0.5])
COV = np.array([[SD[0] ** 2, RHO * SD[0] * SD[1]], [RHO * SD[0] * SD[1], SD[1] ** 2]])


@pytest.fixture(scope="module")
def bvn():
    H = JointDistribution(cp.Gaussian(RHO), [stats.norm(MU[0], SD[0]), stats.norm(MU[1], SD[1])])
    return H, stats.multivariate_normal(MU, COV)


@pytest.fixture(scope="module")
def points():
    return stats.multivariate_normal(MU, 4 * COV).rvs(50, random_state=3)


# ---------------------------------------------------------------------------
# Gaussian copula + normal margins = bivariate normal
# ---------------------------------------------------------------------------


def test_gaussian_copula_with_normal_margins_is_bivariate_normal(bvn, points):
    H, ref = bvn
    np.testing.assert_allclose(H.cdf(points), ref.cdf(points), atol=1e-12)
    np.testing.assert_allclose(H.pdf(points), ref.pdf(points), rtol=1e-8)
    np.testing.assert_allclose(H.logpdf(points), ref.logpdf(points), rtol=1e-10)
    assert isinstance(H.cdf(points[0]), float)
    assert H.cdf(MU[0], MU[1]) == pytest.approx(0.25 + np.arcsin(RHO) / (2 * np.pi))
    # survival function = P(X > x) = cdf of -X at -x
    neg = stats.multivariate_normal(-MU, COV)
    np.testing.assert_allclose(H.sf(points), neg.cdf(-points), atol=1e-12)


def test_pearson_correlation_and_covariance(bvn):
    H, _ = bvn
    np.testing.assert_allclose(H.covariance(), COV, atol=1e-8)
    np.testing.assert_allclose(H.correlation(), [[1, RHO], [RHO, 1]], atol=1e-9)
    np.testing.assert_allclose(H.mean(), MU)
    np.testing.assert_allclose(H.var(), SD**2)
    # rank correlations are those of the copula
    assert H.correlation("kendall")[0, 1] == pytest.approx(2 / np.pi * np.arcsin(RHO))
    assert H.correlation("spearman")[0, 1] == pytest.approx(6 / np.pi * np.arcsin(RHO / 2))


def test_hoeffding_covariance_formula_lognormal():
    """Gaussian copula with lognormal margins (Embrechts, McNeil & Straumann, 2002)."""
    s1, s2 = 0.5, 1.2
    H = JointDistribution(cp.Gaussian(RHO), [stats.lognorm(s1), stats.lognorm(s2)])
    r = (np.exp(RHO * s1 * s2) - 1) / np.sqrt((np.exp(s1**2) - 1) * (np.exp(s2**2) - 1))
    assert H.correlation()[0, 1] == pytest.approx(r, abs=1e-8)


def test_hoeffding_covariance_formula_vs_monte_carlo():
    H = JointDistribution(cp.Clayton(2.0), [stats.expon(scale=2.0), stats.gamma(3.0)])
    cov_h = H.covariance()[0, 1]
    X = H.rvs(400_000, random_state=11)
    cov_mc = np.cov(X, rowvar=False)[0, 1]
    se = np.std((X[:, 0] - X[:, 0].mean()) * (X[:, 1] - X[:, 1].mean())) / np.sqrt(len(X))
    assert abs(cov_h - cov_mc) < 4 * se
    np.testing.assert_allclose(H.covariance(method="mc", n_samples=50_000)[0, 1], cov_h, rtol=0.1)


def test_hoeffding_uniform_margins_gives_spearman():
    """With uniform margins Pearson's correlation is Spearman's rho of the copula."""
    H = JointDistribution(cp.FarlieGumbelMorgenstern(0.7), [stats.uniform(), stats.uniform()])
    assert H.correlation()[0, 1] == pytest.approx(0.7 / 3, abs=1e-6)


def test_rvs_margins_match_scipy():
    margins = [stats.gamma(2.5, scale=1.5), stats.t(5), stats.beta(2, 3)]
    H = JointDistribution(ClaytonND(2.0, 3), margins)
    X = H.rvs(5000, random_state=0)
    assert X.shape == (5000, 3)
    for j, m in enumerate(margins):
        assert stats.kstest(X[:, j], m.cdf).pvalue > 1e-3
    U = H.transform(X)
    np.testing.assert_allclose(H.inverse_transform(U), X, rtol=1e-8, atol=1e-10)
    np.testing.assert_array_equal(X, H.rvs(5000, random_state=0))


def test_regression_of_gaussian_is_linear(bvn):
    H, _ = bvn
    x = np.linspace(-4.0, 6.0, 9)
    line = MU[1] + RHO * SD[1] / SD[0] * (x - MU[0])
    np.testing.assert_allclose(H.regression(x), line, atol=1e-10)
    np.testing.assert_allclose(H.regression(x, "median"), line, atol=1e-10)
    q = 0.9
    sd_c = SD[1] * np.sqrt(1 - RHO**2)
    np.testing.assert_allclose(H.regression(x, q), line + sd_c * stats.norm.ppf(q), atol=1e-9)
    assert isinstance(H.regression(1.0), float)
    with pytest.raises(ValueError):
        H.regression(x, 1.5)
    with pytest.raises(ValueError):
        H.regression(x, "mode")


def test_regression_nonlinear_copula_vs_simulation():
    """Clayton with exponential margins: E[X2 | X1 = x] by quadrature vs. local averages."""
    H = JointDistribution(cp.Clayton(3.0), [stats.expon(), stats.expon()])
    X = H.rvs(400_000, random_state=5)
    for x0 in (0.3, 1.0, 2.0):
        sel = np.abs(X[:, 0] - x0) < 0.02
        assert H.regression(x0) == pytest.approx(X[sel, 1].mean(), abs=0.04)
    # conditional quantiles invert the conditional cdf
    y = H.conditional_ppf(np.array([0.1, 0.5, 0.9]), 1.0)
    np.testing.assert_allclose(H.conditional_cdf(y, 1.0), [0.1, 0.5, 0.9], atol=1e-9)


def test_conditional_distribution(bvn):
    H, _ = bvn
    x, y = 2.0, -1.5
    m = MU[1] + RHO * SD[1] / SD[0] * (x - MU[0])
    s = SD[1] * np.sqrt(1 - RHO**2)
    assert H.conditional_cdf(y, x) == pytest.approx(stats.norm.cdf(y, m, s), abs=1e-12)
    assert H.conditional_pdf(y, x) == pytest.approx(stats.norm.pdf(y, m, s), rel=1e-10)
    # reversed roles: X1 given X2
    m1 = MU[0] + RHO * SD[0] / SD[1] * (y - MU[1])
    s1 = SD[0] * np.sqrt(1 - RHO**2)
    assert H.conditional_cdf(x, y, i=1, j=0) == pytest.approx(stats.norm.cdf(x, m1, s1))
    with pytest.raises(ValueError):
        H.conditional_cdf(y, x, i=0, j=0)


def test_rectangle_probabilities(bvn):
    H, ref = bvn
    a, b = np.array([0.0, -2.5]), np.array([2.5, -1.0])
    expect = ref.cdf(b) - ref.cdf([a[0], b[1]]) - ref.cdf([b[0], a[1]]) + ref.cdf(a)
    assert H.rectangle_probability(a, b) == pytest.approx(expect, abs=1e-12)
    assert H.rectangle_probability([-np.inf, -np.inf], [np.inf, np.inf]) == pytest.approx(1.0)
    A = np.array([a, a - 1])
    B = np.array([b, b + 1])
    assert H.h_volume(A, B).shape == (2,)
    with pytest.raises(ValueError):
        H.rectangle_probability(b, a)


def test_three_dimensional_joint_distribution():
    R = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
    H = JointDistribution(GaussianND(R), stats.norm())
    ref = stats.multivariate_normal(np.zeros(3), R)
    pts = ref.rvs(5, random_state=1)
    np.testing.assert_allclose(H.cdf(pts), ref.cdf(pts), atol=1e-4)
    np.testing.assert_allclose(H.pdf(pts), ref.pdf(pts), rtol=1e-10)
    np.testing.assert_allclose(H.correlation(), R, atol=1e-8)
    # conditional distributions of pairs use the bivariate margins
    assert H.conditional_cdf(0.3, 1.0, i=0, j=2) == pytest.approx(
        stats.norm.cdf(0.3, 0.3 * 1.0, np.sqrt(1 - 0.09))
    )
    # multivariate t = t copula + t margins with the same degrees of freedom
    T = JointDistribution(StudentTND(R, 4), stats.t(4))
    ref_t = stats.multivariate_t(np.zeros(3), R, df=4)
    np.testing.assert_allclose(T.pdf(pts), ref_t.pdf(pts), rtol=1e-10)


def test_bivariate_t_from_t_copula_and_t_margins():
    H = JointDistribution(cp.StudentT(rho=0.5, nu=3), [stats.t(3), stats.t(3)])
    ref = stats.multivariate_t([0, 0], [[1, 0.5], [0.5, 1]], df=3)
    pts = ref.rvs(10, random_state=2)
    np.testing.assert_allclose(H.pdf(pts), ref.pdf(pts), rtol=1e-10)
    np.testing.assert_allclose(H.cdf(pts), ref.cdf(pts, random_state=0), atol=5e-4)


def test_singular_copula_has_no_density():
    H = JointDistribution(UpperFrechetND(2), [stats.norm(), stats.expon()])
    with pytest.raises(PropertyUnavailableException):
        H.pdf([0.0, 1.0])
    # comonotone: H(x, y) = min(F(x), G(y))
    assert H.cdf([0.0, 1.0]) == pytest.approx(min(0.5, stats.expon.cdf(1.0)))


# ---------------------------------------------------------------------------
# discrete margins
# ---------------------------------------------------------------------------


def test_discrete_margins_pmf_cdf_and_sampling():
    H = JointDistribution(cp.Clayton(2.0), [stats.poisson(3), stats.binom(5, 0.4)])
    assert not H.is_continuous
    grid = np.array([[i, j] for i in range(40) for j in range(6)])
    p = H.pmf(grid)
    assert p.sum() == pytest.approx(1.0, abs=1e-12) and np.all(p >= 0)
    sel = (grid[:, 0] <= 2) & (grid[:, 1] <= 3)
    assert H.cdf([2, 3]) == pytest.approx(p[sel].sum(), abs=1e-12)
    # marginal pmfs are recovered
    np.testing.assert_allclose(
        p.reshape(40, 6).sum(axis=1)[:10], stats.poisson(3).pmf(np.arange(10)), atol=1e-12
    )
    X = H.rvs(200_000, random_state=1)
    emp = np.mean((X[:, 0] == 3) & (X[:, 1] == 2))
    assert emp == pytest.approx(H.pmf([3, 2]), abs=4e-3)
    assert H.sf([2, 1]) == pytest.approx(np.mean((X[:, 0] > 2) & (X[:, 1] > 1)), abs=4e-3)
    for f in (H.pdf, H.logpdf):
        with pytest.raises(ValueError):
            f([1, 1])
    with pytest.raises(ValueError):
        H.regression(1.0)
    with pytest.raises(ValueError):
        H.covariance()
    cov = H.covariance(method="mc", n_samples=100_000)
    assert cov[0, 1] > 0
    Hc = JointDistribution(cp.Clayton(2.0), [stats.norm(), stats.norm()])
    with pytest.raises(ValueError):
        Hc.pmf([0.0, 0.0])


def test_empirical_marginal():
    x = np.array([3.0, 1.0, 2.0, 2.0, 5.0])
    m = EmpiricalMarginal(x)
    np.testing.assert_allclose(m.cdf([0.5, 1.0, 2.0, 4.9, 5.0]), [0, 0.2, 0.6, 0.8, 1.0])
    np.testing.assert_allclose(m.ppf([0.1, 0.2, 0.21, 0.6, 0.61, 1.0]), [1, 1, 2, 2, 3, 5])
    np.testing.assert_allclose(m.pmf([2.0, 4.0]), [0.4, 0.0])
    assert m.mean() == pytest.approx(2.6)
    assert set(m.rvs(20, random_state=0)) <= set(x)
    with pytest.raises(ValueError):
        EmpiricalMarginal([])


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------


def test_fit_ifm_and_cml_bivariate():
    true = JointDistribution(cp.Clayton(2.0), [stats.norm(1, 2), stats.expon(scale=3)])
    X = true.rvs(2000, random_state=1)
    ifm = JointDistribution.fit(X, cp.Clayton, [stats.norm, stats.expon], method="ifm")
    assert ifm.fit_info["copula_fit"].params["theta"] == pytest.approx(2.0, abs=0.3)
    loc, scale = ifm.fit_info["marginal_params"][0]
    assert loc == pytest.approx(1.0, abs=0.15) and scale == pytest.approx(2.0, abs=0.15)
    assert np.isfinite(ifm.fit_info["loglik"])
    cml = JointDistribution.fit(X, "Clayton", method="cml")
    assert cml.fit_info["copula_fit"].params["theta"] == pytest.approx(2.0, abs=0.3)
    assert isinstance(cml.marginal(0), EmpiricalMarginal)
    emp = np.mean((X[:, 0] <= 1.0) & (X[:, 1] <= 2.0))
    assert cml.cdf([1.0, 2.0]) == pytest.approx(emp, abs=0.02)
    mom = JointDistribution.fit(
        X, cp.Clayton, [stats.norm(1, 2), "empirical"], method="ifm", copula_method="itau"
    )
    assert mom.fit_info["copula_fit"].method == "itau"
    with pytest.raises(ValueError):
        JointDistribution.fit(X, cp.Clayton, method="bad")


def test_fit_multivariate():
    R = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
    true = JointDistribution(GaussianND(R), [stats.norm(), stats.t(5), stats.gamma(2)])
    X = true.rvs(2000, random_state=2)
    H = JointDistribution.fit(X, GaussianND, [stats.norm, stats.t, stats.gamma])
    np.testing.assert_allclose(H.copula.corr, R, atol=0.06)
    A = JointDistribution.fit(ClaytonND(2.0, 3).rvs(1500, random_state=0), "clayton")
    assert A.copula.theta == pytest.approx(2.0, rel=0.15)
    with pytest.raises(ValueError):
        JointDistribution.fit(X, cp.Clayton)


# ---------------------------------------------------------------------------
# copula_from_joint
# ---------------------------------------------------------------------------


def test_copula_from_bivariate_normal_is_gaussian_copula():
    ref = cp.Gaussian(RHO)
    C = copula_from_joint(
        stats.multivariate_normal(MU, COV),
        marginals=[stats.norm(MU[0], SD[0]), stats.norm(MU[1], SD[1])],
    )
    assert isinstance(C, SklarCopula) and C.is_absolutely_continuous
    U = np.random.default_rng(0).random((40, 2))
    np.testing.assert_allclose(C.cdf(U), ref.cdf(U), atol=1e-12)
    np.testing.assert_allclose(C.pdf(U), ref.pdf(U), rtol=1e-10)
    np.testing.assert_allclose(C.cond_distr_1(U), ref.cond_distr_1(U), atol=1e-9)
    np.testing.assert_allclose(C.cond_distr_2(U), ref.cond_distr_2(U), atol=1e-9)
    assert C.cdf(1.0, 0.3) == pytest.approx(0.3) and C.cdf(0.0, 0.3) == 0.0
    assert C.kendalls_tau() == pytest.approx(2 / np.pi * np.arcsin(RHO), abs=1e-9)
    assert C.spearmans_rho() == pytest.approx(6 / np.pi * np.arcsin(RHO / 2), abs=1e-6)
    assert C.blomqvists_beta() == pytest.approx(2 / np.pi * np.arcsin(RHO), abs=1e-12)
    X = C.rvs(4000, random_state=1)
    assert stats.kendalltau(*X.T).statistic == pytest.approx(2 / np.pi * np.arcsin(RHO), abs=0.03)


def test_copula_from_joint_with_callables():
    """Explicit callables, two-argument signature, no density: cdf and FD h-functions."""
    mvn = stats.multivariate_normal(MU, COV)
    C = copula_from_joint(
        lambda x, y: mvn.cdf(np.dstack([x, y])),
        marginal_cdfs=[stats.norm(MU[0], SD[0]).cdf, stats.norm(MU[1], SD[1]).cdf],
        marginal_ppfs=[stats.norm(MU[0], SD[0]).ppf, stats.norm(MU[1], SD[1]).ppf],
        signature="xy",
    )
    assert not C.is_absolutely_continuous
    ref = cp.Gaussian(RHO)
    assert C.cdf(0.3, 0.7) == pytest.approx(ref.cdf(0.3, 0.7), abs=1e-12)
    assert C.cond_distr_1(0.3, 0.7) == pytest.approx(ref.cond_distr_1(0.3, 0.7), abs=1e-6)
    X = C.rvs(500, random_state=0)  # conditional inversion
    assert X.shape == (500, 2)
    with pytest.raises(ValueError):
        copula_from_joint(mvn.cdf)


def test_copula_from_bivariate_t_is_student_t_copula():
    R = np.array([[1.0, -0.4], [-0.4, 1.0]])
    joint = stats.multivariate_t([0.5, 1.0], 2.0 * R, df=4)
    margins = [stats.t(4, loc=0.5, scale=np.sqrt(2)), stats.t(4, loc=1.0, scale=np.sqrt(2))]
    C = copula_from_joint(joint, marginals=margins)
    ref = cp.StudentT(rho=-0.4, nu=4)
    U = np.random.default_rng(2).random((12, 2))
    np.testing.assert_allclose(C.pdf(U), ref.pdf(U), rtol=1e-9)
    np.testing.assert_allclose(C.cond_distr_1(U), ref.cond_distr_1(U), atol=1e-7)
    # SciPy's bivariate t cdf is computed by randomized quasi-Monte Carlo
    np.testing.assert_allclose(C.cdf(U), ref.cdf(U), atol=5e-4)
    X = C.rvs(3000, random_state=3)  # transforms joint samples
    assert stats.kendalltau(*X.T).statistic == pytest.approx(2 / np.pi * np.arcsin(-0.4), abs=0.04)
