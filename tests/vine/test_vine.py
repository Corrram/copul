"""Independent distribution identities, transforms, graph and API checks."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from scipy.special import ndtr, ndtri

import copul as cp
from copul.vine import VineStructure


def general_structure():
    # First tree has both degree 3 and a length-3 path: neither a C nor D vine.
    return VineStructure(
        [
            [(0, 1, ()), (1, 2, ()), (1, 3, ()), (3, 4, ())],
            [(0, 2, (1,)), (2, 3, (1,)), (1, 4, (3,))],
            [(0, 3, (1, 2)), (2, 4, (1, 3))],
            [(0, 4, (1, 2, 3))],
        ]
    )


def correlation(d):
    rng = np.random.default_rng(271)
    A = rng.normal(size=(d, d))
    S = A @ A.T + 2 * np.eye(d)
    return S / np.sqrt(np.outer(np.diag(S), np.diag(S)))


def elliptical_pairs(structure, R, nu=None):
    pairs = []
    for tree in structure.trees:
        pt = []
        for e in tree:
            ab, D = [e.a, e.b], list(e.conditioning)
            S = R[np.ix_(ab, ab)]
            if D:
                S = S - R[np.ix_(ab, D)] @ np.linalg.solve(R[np.ix_(D, D)], R[np.ix_(D, ab)])
            rho = S[0, 1] / np.sqrt(S[0, 0] * S[1, 1])
            pt.append(cp.Gaussian(rho=rho) if nu is None else cp.StudentT(rho=rho, nu=nu + len(D)))
        pairs.append(pt)
    return pairs


@pytest.mark.parametrize(
    "structure",
    [VineStructure.cvine([2, 0, 3, 1]), VineStructure.dvine([3, 1, 0, 2]), general_structure()],
)
@pytest.mark.parametrize("nu", [None, 4.5])
def test_elliptical_density_and_conditional_transforms(structure, nu):
    d = structure.dim
    R = correlation(d)
    model = cp.VineCopula(structure, elliptical_pairs(structure, R, nu))
    U = np.random.default_rng(21).uniform(0.03, 0.97, (80, d))
    if nu is None:
        X = ndtri(U)
        expected = stats.multivariate_normal.logpdf(X, cov=R) - stats.norm.logpdf(X).sum(axis=1)
    else:
        X = stats.t.ppf(U, nu)
        expected = stats.multivariate_t.logpdf(X, shape=R, df=nu) - stats.t.logpdf(X, nu).sum(
            axis=1
        )
    assert_allclose(model.logpdf(U), expected, atol=2e-8)
    W = model.rosenblatt(U)
    assert_allclose(model.inverse_rosenblatt(W), U, atol=2e-9)
    # Independent conditional normal/t distribution formula in sampling order.
    for k, a in enumerate(model.order):
        D = list(model.order[:k])
        if not D:
            assert_allclose(W[:, a], U[:, a])
            continue
        beta = np.linalg.solve(R[np.ix_(D, D)], R[D, a])
        mu = X[:, D] @ beta
        var = 1 - R[a, D] @ beta
        if nu is None:
            value = ndtr((X[:, a] - mu) / np.sqrt(var))
        else:
            q = np.einsum("ni,in->n", X[:, D], np.linalg.solve(R[np.ix_(D, D)], X[:, D].T))
            value = stats.t.cdf((X[:, a] - mu) / np.sqrt(var * (nu + q) / (nu + k)), nu + k)
        assert_allclose(W[:, a], value, atol=3e-9)


@pytest.mark.parametrize("cls", [cp.CVineCopula, cp.DVineCopula])
def test_independence_shapes_and_boundaries(cls):
    model = cls(4, order=[2, 0, 3, 1])
    U = np.random.default_rng(7).uniform(0.01, 0.99, (2, 3, 4))
    assert_allclose(model.pdf(U), np.ones((2, 3)))
    assert_allclose(model.cdf(U, n_points=16), U.prod(axis=-1), atol=1e-14)
    assert_allclose(model.rosenblatt(U), U)
    assert_allclose(model.inverse_rosenblatt(U), U, atol=1e-10)
    assert isinstance(model.pdf(U[0, 0]), float)
    assert_allclose(model.pdf(*U[0, 0]), 1)
    assert model.cdf([1, 0.34, 1, 1]) == 0.34
    assert model.cdf([0, 0.34, 1, 1]) == 0
    assert model.cdf([1, 1, 1, 1]) == 1
    assert model.pdf([-0.1, 0.2, 0.3, 0.4]) == 0
    assert model.rvs(0).shape == (0, 4)
    assert model.rosenblatt(np.empty((0, 4))).shape == (0, 4)
    assert model.cdf(np.empty((0, 4))).shape == (0,)
    assert_allclose(model.rvs(10, 11), model.rvs(10, 11))


def asymmetric_model():
    return cp.DVineCopula(
        4,
        [
            [
                cp.Clayton(theta=1.2),
                cp.rotate(cp.GumbelHougaard(theta=1.4), 90),
                cp.Frank(theta=-2),
            ],
            [cp.rotate(cp.Clayton(theta=1.3), 180), cp.Gaussian(rho=-0.3)],
            [cp.StudentT(rho=0.3, nu=5)],
        ],
        order=[2, 0, 3, 1],
    )


def test_asymmetric_pair_orientation_jacobian_and_margins():
    model = asymmetric_model()
    x = np.array([0.24, 0.53, 0.68, 0.42])
    eps = 2e-6
    jac = np.column_stack(
        [
            (model.rosenblatt(x + eps * e) - model.rosenblatt(x - eps * e)) / (2 * eps)
            for e in np.eye(4)
        ]
    )
    assert_allclose(abs(np.linalg.det(jac)), model.pdf(x), rtol=3e-5)
    W = np.random.default_rng(7).uniform(0.02, 0.98, (100, 4))
    assert_allclose(model.rosenblatt(model.inverse_rosenblatt(W)), W, atol=2e-8)
    U = model.rvs(6000, 39)
    for col in U.T:
        assert stats.kstest(col, "uniform").statistic < 0.025
    # First-tree margins must equal the respective bivariate distributions.
    for e, pair in zip(model.structure.trees[0], model.pair_copulas[0]):
        assert (
            abs(stats.kendalltau(U[:, e.a], U[:, e.b]).statistic - float(pair.kendalls_tau()))
            < 0.035
        )


def test_cdf_against_gaussian_orthant_formula_and_pair_margin():
    R = np.array([[1, 0.5, -0.2], [0.5, 1, 0.3], [-0.2, 0.3, 1]])
    structure = VineStructure.dvine([2, 0, 1])
    model = cp.VineCopula(structure, elliptical_pairs(structure, R))
    exact = 1 / 8 + (np.arcsin(0.5) + np.arcsin(-0.2) + np.arcsin(0.3)) / (4 * np.pi)
    assert_allclose(model.cdf([0.5] * 3, n_points=8192), exact, atol=8e-5)
    assert_allclose(
        model.cdf([0.3, 1, 0.6], n_points=8192),
        float(cp.Gaussian(rho=-0.2).cdf(0.3, 0.6)),
        atol=8e-5,
    )


def test_three_dimensional_clayton_factorization_and_normalization():
    # Clayton(theta) conditional pair has parameter theta/(1+theta).
    theta = 1.7
    model = cp.CVineCopula(
        3, [[cp.Clayton(theta=theta)] * 2, [cp.Clayton(theta=theta / (1 + theta))]]
    )
    U = np.random.default_rng(12).uniform(0.05, 0.95, (100, 3))
    assert_allclose(model.pdf(U), cp.ClaytonND(theta=theta, dim=3).pdf(U), rtol=2e-10)
    assert_allclose(
        model.cdf([0.4, 0.6, 0.7], n_points=8192),
        cp.ClaytonND(theta=theta, dim=3).cdf([0.4, 0.6, 0.7]),
        atol=5e-5,
    )
    # Independent Gauss-Legendre integration for a smooth Gaussian vine.
    g = cp.CVineCopula(3, [[cp.Gaussian(rho=0.2), cp.Gaussian(rho=-0.15)], [cp.Gaussian(rho=0.1)]])
    x, w = np.polynomial.legendre.leggauss(28)
    U = np.stack(np.meshgrid(*([0.5 * (x + 1)] * 3), indexing="ij"), axis=-1)
    mass = np.einsum("i,j,k,ijk->", w / 2, w / 2, w / 2, g.pdf(U))
    assert_allclose(mass, 1, atol=2e-5)


@pytest.mark.parametrize(
    "trees",
    [
        [],
        [[(0, 1, ())], []],
        [[(0, 1, ()), (1, 0, ())], [(0, 2, (1,))]],
        [[(0, 1, ()), (1, 2, ())], [(0, 1, (2,))]],
        [[(0, 1, ()), (1, 3, ())], [(0, 2, (1,))]],
    ],
)
def test_invalid_structures(trees):
    with pytest.raises(ValueError):
        VineStructure(trees)


def test_constructor_and_input_validation():
    with pytest.raises(ValueError):
        cp.VineEdge(0, 1, [1])
    with pytest.raises(ValueError):
        VineStructure(VineStructure.dvine(range(4)).trees, order=[0, 2, 1, 3])
    for pairs in ([[cp.UpperFrechet()]], [[cp.Gaussian()]], [[]]):
        with pytest.raises(ValueError):
            cp.CVineCopula(2, pairs)
    for dim in (1, 3.5, True):
        with pytest.raises(ValueError):
            cp.CVineCopula(dim)
    model = cp.CVineCopula(2)
    for U in ([0, 0.5], [0.5, 1], [np.nan, 0.5], [np.inf, 0.5], [0.1, 0.2, 0.3]):
        with pytest.raises(ValueError):
            model.rosenblatt(U)
    with pytest.raises(ValueError):
        model.rvs(2.5)
    with pytest.raises(ValueError):
        model.cdf([np.nan, 0.5])


def test_bivariate_limit_preserves_asymmetric_orientation():
    pair = cp.reflect(cp.Clayton(theta=2), axis="u")
    model = cp.CVineCopula(2, [[pair]], order=[1, 0])
    u = np.array([[0.2, 0.7], [0.5, 0.3], [0, 0.8], [1, 0.8]])
    assert_allclose(model.cdf(u), pair.cdf(u[:, 1], u[:, 0]), atol=1e-13)
    assert_allclose(model.pdf(u[:2]), pair.pdf(u[:2, 1], u[:2, 0]), atol=1e-13)


def test_components_are_copied_and_seeded_transform_roundtrip():
    pair = cp.Gaussian(rho=0.5)
    model = cp.CVineCopula(2, [[pair]])
    pair.rho = -0.6
    assert float(model.pair_copulas[0][0].rho) == 0.5
    W = np.random.default_rng(183).uniform(0.005, 0.995, (50, 2))
    assert_allclose(model.rosenblatt(model.inverse_rosenblatt(W)), W, atol=1e-10)
