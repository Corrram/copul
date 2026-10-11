"""Fitting recovery, family selection and failure diagnostics."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from scipy.special import ndtri

import copul as cp
from copul.vine import PairCopulaFamily, VineFitResult, VineStructure


@pytest.mark.parametrize(
    "kind, cls", [("cvine", cp.CVineCopula), ("dvine", cp.DVineCopula), ("rvine", cp.VineCopula)]
)
def test_fit_gaussian_recovery_and_result_consistency(kind, cls):
    R = np.array(
        [[1, 0.65, 0.3, -0.2], [0.65, 1, 0.2, -0.1], [0.3, 0.2, 1, 0.25], [-0.2, -0.1, 0.25, 1]]
    )
    U = cp.GaussianND(R).rvs(900, random_state=34)
    result = cp.fit_vine(U, vine_type=kind, families=[cp.Gaussian], pseudo_obs=False)
    assert isinstance(result.copula, cls)
    assert result.converged
    assert result.k == 6
    assert not result.selection_failures
    assert result.copula.fit_result_ is result
    assert_allclose(result.loglik, result.copula.logpdf(U).sum())
    assert_allclose(
        result.loglik, sum(r.loglik for tree in result.edge_fits for r in tree), atol=1e-7
    )
    assert_allclose(result.aic, 12 - 2 * result.loglik)
    assert_allclose(result.bic, 6 * np.log(len(U)) - 2 * result.loglik)
    sample = result.rvs(6000, 19)
    assert_allclose(np.corrcoef(ndtri(sample), rowvar=False), R, atol=0.09)
    W = result.copula.rosenblatt(U)
    assert_allclose(np.corrcoef(ndtri(W), rowvar=False), np.eye(4), atol=0.075)
    assert "Sequential" in result.summary()


def test_conditional_pairs_are_not_ranked_again_and_joint_improves():
    true = cp.DVineCopula(
        3, [[cp.Gaussian(rho=0.7), cp.Gaussian(rho=0.4)], [cp.Gaussian(rho=-0.35)]]
    )
    U = true.rvs(450, 129)
    result = cp.fit_vine(
        U, vine_type="dvine", order=[0, 1, 2], families=[cp.Gaussian], pseudo_obs=False
    )
    a, b = result.copula._backends[0]
    conditional = np.column_stack((a.h2(U[:, 0], U[:, 1]), b.h1(U[:, 1], U[:, 2])))
    direct = cp.fit(cp.Gaussian, conditional, pseudo_obs=False, _light=True)
    assert_allclose(result.edge_fits[1][0].params["rho"], direct.params["rho"], atol=2e-7)
    joint = cp.fit_vine(
        U, vine_type="dvine", order=[0, 1, 2], families=[cp.Gaussian], pseudo_obs=False, joint=True
    )
    assert joint.converged
    assert joint.loglik >= joint.sequential_loglik - 1e-8
    assert_allclose(joint.sequential_loglik, result.loglik, atol=1e-7)
    assert_allclose(sum(r.loglik for t in joint.edge_fits for r in t), joint.loglik, atol=1e-7)
    assert joint.optimizer_result is not None


def test_truncation_and_independence_selection():
    U = np.random.default_rng(42).uniform(0.001, 0.999, (800, 4))
    independent = cp.fit_vine(
        U, families=["Independence", "Gaussian"], criterion="bic", pseudo_obs=False
    )
    assert independent.k == 0
    assert independent.loglik == 0
    R = np.full((4, 4), 0.7)
    np.fill_diagonal(R, 1)
    U = cp.GaussianND(R).rvs(400, 7)
    result = cp.fit_vine(U, families=["Gaussian"], truncation=1, pseudo_obs=False)
    assert result.k == 3
    assert all(r.k == 0 and r.loglik == 0 for t in result.edge_fits[1:] for r in t)
    zero = cp.fit_vine(U, truncation=0)
    assert zero.k == 0
    assert_allclose(zero.copula.pdf(U), 1)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_rotated_clayton_family_selection(rotation):
    spec = PairCopulaFamily(cp.Clayton, rotation)
    original = spec.wrap(cp.Clayton(theta=2))
    U = original.rvs(700, random_state=11)
    candidates = [
        "Independence",
        "Gaussian",
        *(PairCopulaFamily(cp.Clayton, angle) for angle in (0, 90, 180, 270)),
    ]
    result = cp.fit_vine(U, families=candidates, pseudo_obs=False, criterion="bic")
    pair = result.edge_fits[0][0]
    assert pair.family == f"Clayton[{rotation}]"
    assert abs(pair.params["theta"] - 2) < 0.3
    assert result.k == 1


def test_partially_fixed_student_t_and_fixed_pair_candidates():
    U = cp.StudentT(rho=0.55, nu=4).rvs(600, random_state=38)
    result = cp.fit_vine(U, families=[cp.StudentT(nu=4)], pseudo_obs=False)
    edge = result.edge_fits[0][0]
    assert result.k == 1
    assert edge.fixed["nu"] == 4
    assert abs(edge.params["rho"] - 0.55) < 0.08
    fixed = cp.fit_vine(U, families=[cp.StudentT(rho=0.55, nu=4)], pseudo_obs=False, joint=True)
    assert fixed.k == 0
    assert fixed.optimizer_result is None
    assert_allclose(fixed.loglik, cp.StudentT(rho=0.55, nu=4).logpdf(U[:, 0], U[:, 1]).sum())


def test_default_candidates_recover_student_t_two_parameters():
    U = cp.StudentT(rho=0.55, nu=3).rvs(500, random_state=21)
    result = cp.fit_vine(U, pseudo_obs=False)
    edge = result.edge_fits[0][0]
    assert edge.family == "T[0]"
    assert result.k == 2
    assert abs(edge.params["rho"] - 0.55) < 0.06
    assert abs(edge.params["nu"] - 3) < 1
    assert_allclose(cp.stats.loglik(result.copula, U, pseudo_obs=False), result.loglik)


def test_explicit_structure_raw_ranks_and_public_dispatch():
    U = cp.GaussianND([[1, 0.6, -0.3], [0.6, 1, -0.2], [-0.3, -0.2, 1]]).rvs(250, 31)
    X = np.column_stack((stats.norm.ppf(U[:, 0]), stats.expon.ppf(U[:, 1]), U[:, 2] ** 3))
    structure = VineStructure.dvine([2, 0, 1])
    raw = cp.fit_vine(X, structure=structure, families="Gaussian")
    ranked = cp.fit_vine(
        cp.pseudo_obs(X), structure=structure, families="Gaussian", pseudo_obs=False
    )
    assert_allclose(raw.loglik, ranked.loglik, atol=1e-8)
    assert raw.structure.order == (2, 0, 1)
    fitted = cp.CVineCopula.fit(X, families="Gaussian", order=[2, 0, 1])
    assert isinstance(fitted, cp.CVineCopula)
    for family in (cp.DVineCopula, "DVineCopula"):
        fit = cp.fit(family, X, method="sequential", families="Gaussian", order=[2, 0, 1])
        assert isinstance(fit, VineFitResult)
        assert fit.structure.order == (2, 0, 1)
    joint = cp.fit(cp.CVineCopula, U[:, :2], families="Gaussian", pseudo_obs=False)
    assert joint.method == "joint_mle"


def test_rejected_candidates_are_reported_without_hiding_all_failures():
    U = np.random.default_rng(7).uniform(0.1, 0.9, (40, 2))
    result = cp.fit_vine(U, families=[cp.UpperFrechet(), "Independence"], pseudo_obs=False)
    assert result.k == 0
    assert len(result.selection_failures) == 1
    with pytest.raises(ValueError, match="No candidate"):
        cp.fit_vine(U, families=[cp.UpperFrechet()], pseudo_obs=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"families": []},
        {"families": ["no_such_copula"]},
        {"truncation": -1},
        {"truncation": 3},
        {"truncation": 1.5},
        {"criterion": "waic"},
        {"vine_type": "other"},
        {"order": [0, 1]},
        {"vine_type": "dvine", "order": [0, 0]},
        {"ties": "random"},
    ],
)
def test_bad_fitting_options(kwargs):
    with pytest.raises(ValueError):
        cp.fit_vine(np.random.default_rng(1).random((10, 2)), **kwargs)


@pytest.mark.parametrize(
    "U",
    [
        [[0.3, 0.5]],
        [[0.2], [0.5]],
        [[0.3, 0.5], [np.nan, 0.4]],
        [[0.3, 0.5], [np.inf, 0.4]],
        [[0.3, 0.5], [0.3, 0.4]],
        [[0, 0.5], [0.3, 0.4]],
        [[1, 0.5], [0.3, 0.4]],
    ],
)
def test_bad_data(U):
    with pytest.raises(ValueError):
        cp.fit_vine(U, pseudo_obs=False)


def test_optimizer_failure_is_visible_and_seed_does_not_touch_global_rng():
    U = cp.CVineCopula(3, [[cp.Gaussian(rho=0.5)] * 2, [cp.Gaussian(rho=0.3)]]).rvs(100, 1)
    result = cp.fit_vine(
        U, families="Gaussian", joint=True, joint_options={"maxiter": 0}, pseudo_obs=False
    )
    assert not result.converged
    assert result.loglik >= result.sequential_loglik - 1e-8
    np.random.seed(27)
    expected = np.random.random(3)
    np.random.seed(27)
    result.rvs(10, random_state=14)
    assert_allclose(np.random.random(3), expected)
