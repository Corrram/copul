"""d-dimensional Gaussian and Student-t copulas."""

import numpy as np
import pytest
from scipy import stats
from scipy.special import ndtri, stdtrit

import copul as cp
from copul.multivariate import GaussianND, StudentTND, nearest_correlation

R3 = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
R4 = np.array(
    [
        [1.0, 0.6, 0.3, 0.2],
        [0.6, 1.0, 0.4, 0.1],
        [0.3, 0.4, 1.0, -0.2],
        [0.2, 0.1, -0.2, 1.0],
    ]
)
U3 = np.random.default_rng(7).random((40, 3))


def test_validation_of_correlation_matrix():
    with pytest.raises(ValueError):
        GaussianND([[1, 0.5], [0.4, 1]])  # not symmetric
    with pytest.raises(ValueError):
        GaussianND([[2, 0.5], [0.5, 1]])  # diagonal
    with pytest.raises(ValueError):
        GaussianND([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]])  # not PD
    with pytest.raises(ValueError):
        StudentTND(R3, nu=0)


def test_gaussian_pdf_matches_scipy():
    G = GaussianND(R3)
    X = ndtri(U3)
    ref = stats.multivariate_normal(np.zeros(3), R3).pdf(X) / np.prod(stats.norm.pdf(X), axis=1)
    np.testing.assert_allclose(G.pdf(U3), ref, rtol=1e-12)
    np.testing.assert_allclose(G.logpdf(U3), np.log(ref), rtol=1e-12, atol=1e-12)


def test_gaussian_cdf_matches_scipy_and_bivariate():
    G = GaussianND(R3)
    X = ndtri(U3[:10])
    ref = stats.multivariate_normal(np.zeros(3), R3).cdf(X)
    np.testing.assert_allclose(G.cdf(U3[:10]), ref, atol=5e-5)
    # orthant probability (Sheppard): 1/8 + sum(arcsin r_ij) / (4 pi)
    p = 0.125 + np.sum(np.arcsin([0.5, 0.3, 0.4])) / (4 * np.pi)
    assert G.cdf([0.5, 0.5, 0.5]) == pytest.approx(p, abs=1e-5)
    G2 = GaussianND([[1, 0.7], [0.7, 1]])
    P = U3[:, :2]
    np.testing.assert_allclose(G2.cdf(P), cp.Gaussian(0.7).cdf(P), atol=1e-14)


def test_gaussian_margins_are_copul_objects():
    G = GaussianND(R4)
    m = G.margin(0, 2)
    assert isinstance(m, cp.Gaussian) and float(m.rho) == pytest.approx(0.3)
    sub = G.margin([3, 0, 1])
    assert isinstance(sub, GaussianND)
    np.testing.assert_allclose(sub.corr, R4[np.ix_([3, 0, 1], [3, 0, 1])])
    # margin via cdf with ones agrees with the closed-form margin
    P = np.random.default_rng(1).random((5, 2))
    full = np.ones((5, 4))
    full[:, [0, 2]] = P
    np.testing.assert_allclose(G.cdf(full), m.cdf(P), atol=5e-5)


def test_gaussian_rvs():
    G = GaussianND(R4)
    X = G.rvs(20_000, random_state=11)
    assert X.shape == (20_000, 4)
    assert all(stats.kstest(X[:, j], "uniform").pvalue > 1e-3 for j in range(4))
    np.testing.assert_allclose(np.corrcoef(ndtri(X), rowvar=False), R4, atol=0.03)
    np.testing.assert_array_equal(X, G.rvs(20_000, random_state=11))


def test_gaussian_rosenblatt_gives_independent_uniforms():
    G = GaussianND(R3)
    X = G.rvs(5000, random_state=2)
    Z = G.rosenblatt(X)
    assert np.max(np.abs(np.corrcoef(Z, rowvar=False) - np.eye(3))) < 0.05
    assert all(stats.kstest(Z[:, j], "uniform").pvalue > 1e-3 for j in range(3))


def test_gaussian_closed_form_matrices():
    G = GaussianND(R3)
    np.testing.assert_allclose(G.kendalls_tau_matrix(), 2 / np.pi * np.arcsin(R3))
    np.testing.assert_allclose(G.spearmans_rho_matrix(), 6 / np.pi * np.arcsin(R3 / 2))
    np.testing.assert_allclose(G.tail_dependence_matrix(), np.eye(3))
    assert G.radially_symmetric and not G.exchangeable
    assert GaussianND(np.full((3, 3), 0.4) + 0.6 * np.eye(3)).exchangeable


def test_gaussian_radial_symmetry_of_survival_function():
    from itertools import product

    G = GaussianND(R3)
    u = np.array([0.2, 0.3, 0.4])
    surv = G.survival_function(u)
    assert surv == pytest.approx(G.cdf(1 - u), abs=1e-12)
    # inclusion-exclusion over the 2^3 vertices (exact up to the QMC error of the cdf)
    total = 0.0
    for mask in product([False, True], repeat=3):
        pt = np.where(mask, u, 1.0)
        total += (-1) ** sum(mask) * G.cdf(pt)
    assert surv == pytest.approx(total, abs=1e-4)


def test_gaussian_is_copula_on_grid():
    ok, det = GaussianND(R3).is_copula(grid=6, tol=1e-4, return_details=True)
    assert ok and det["total_mass"] == pytest.approx(1.0)


def test_student_t_pdf_matches_scipy():
    T = StudentTND(R3, nu=4.5)
    X = stdtrit(4.5, U3)
    ref = stats.multivariate_t(np.zeros(3), R3, df=4.5).pdf(X) / np.prod(
        stats.t.pdf(X, 4.5), axis=1
    )
    np.testing.assert_allclose(T.pdf(U3), ref, rtol=1e-11)


def test_student_t_bivariate_reduction_and_margins():
    T2 = StudentTND([[1, -0.4], [-0.4, 1]], nu=3)
    P = U3[:, :2]
    ref = cp.StudentT(rho=-0.4, nu=3)
    np.testing.assert_allclose(T2.cdf(P), ref.cdf(P), atol=1e-12)
    np.testing.assert_allclose(T2.pdf(P), ref.pdf(P), rtol=1e-10)
    T = StudentTND(R4, nu=5)
    m = T.margin(1, 2)
    assert isinstance(m, cp.StudentT)
    assert float(m.rho) == pytest.approx(0.4) and float(m.nu) == pytest.approx(5)
    assert isinstance(T.margin(0, 1, 2), StudentTND)


def test_student_t_cdf_matches_scipy():
    T = StudentTND(R3, nu=4)
    ref = stats.multivariate_t(np.zeros(3), R3, df=4).cdf(stdtrit(4, U3[:5]), random_state=1)
    np.testing.assert_allclose(T.cdf(U3[:5]), ref, atol=1e-3)


def test_student_t_tail_dependence():
    T = StudentTND(R3, nu=4)
    lam = T.tail_dependence_matrix()
    r = 0.5
    expect = 2 * stats.t.cdf(-np.sqrt(5 * (1 - r) / (1 + r)), 5)
    assert lam[0, 1] == pytest.approx(expect)
    assert lam[0, 1] == pytest.approx(cp.StudentT(rho=0.5, nu=4).lambda_L(), rel=1e-6)


def test_student_t_rvs_and_tau():
    T = StudentTND(R3, nu=3)
    X = T.rvs(4000, random_state=5)
    assert all(stats.kstest(X[:, j], "uniform").pvalue > 1e-3 for j in range(3))
    tau = stats.kendalltau(X[:, 0], X[:, 1]).statistic
    assert tau == pytest.approx(2 / np.pi * np.arcsin(0.5), abs=0.03)


def test_gaussian_fit():
    X = GaussianND(R4).rvs(3000, random_state=9)
    for method in ("itau", "irho", "normal_scores"):
        F = GaussianND.fit(X, method=method)
        np.testing.assert_allclose(F.corr, R4, atol=0.06)
    with pytest.raises(ValueError):
        GaussianND.fit(X, method="nope")


def test_student_t_fit():
    X = StudentTND(R3, nu=4).rvs(3000, random_state=10)
    F = StudentTND.fit(X)
    np.testing.assert_allclose(F.corr, R3, atol=0.06)
    assert 2.5 < F.nu < 7.0
    assert StudentTND.fit(X, nu=6).nu == 6


def test_nearest_correlation():
    A = np.array([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]])
    B = nearest_correlation(A)
    assert np.allclose(np.diag(B), 1)
    assert np.linalg.eigvalsh(B).min() > 0
    GaussianND(B)  # valid
    np.testing.assert_allclose(nearest_correlation(R3), R3)
