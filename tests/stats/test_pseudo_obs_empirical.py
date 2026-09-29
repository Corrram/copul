import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import copul as cp
from copul import stats as cs


def test_pseudo_obs_scaling_and_ties():
    X = np.array([[1.0, 10.0], [3.0, 30.0], [2.0, 5.0]])
    np.testing.assert_allclose(cs.pseudo_obs(X), [[0.25, 0.5], [0.75, 0.75], [0.5, 0.25]])
    np.testing.assert_allclose(
        cs.pseudo_obs(X, scale="n"), [[1 / 3, 2 / 3], [1, 1], [2 / 3, 1 / 3]]
    )
    x = np.array([1.0, 2.0, 2.0, 3.0])
    np.testing.assert_allclose(cs.pseudo_obs(x, ties="average", scale="n"), [0.25, 0.625, 0.625, 1])
    np.testing.assert_allclose(cs.pseudo_obs(x, ties="max", scale="n"), [0.25, 0.75, 0.75, 1])
    r1 = cs.pseudo_obs(x, ties="random", random_state=3)
    r2 = cs.pseudo_obs(x, ties="random", random_state=3)
    np.testing.assert_array_equal(r1, r2)
    assert sorted(r1 * 5) == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        cs.pseudo_obs(x, ties="bogus")
    with pytest.raises(ValueError):
        cs.pseudo_obs(x, scale="2n")


def test_pseudo_obs_dataframe():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [3.0, 1.0, 2.0], "b": [1.0, 2.0, 3.0]})
    np.testing.assert_allclose(cs.pseudo_obs(df)[:, 0], [0.75, 0.25, 0.5])


def test_empirical_cdf_matches_brute_force():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(57, 2))
    ec = cs.EmpiricalCopula(X)
    U = ec.U
    q = rng.random((40, 2))
    brute = np.array([np.mean((U[:, 0] <= a) & (U[:, 1] <= b)) for a, b in q])
    np.testing.assert_allclose(ec.cdf(q[:, 0], q[:, 1]), brute)
    np.testing.assert_allclose(ec.cdf(q), brute)
    assert isinstance(ec.cdf(0.5, 0.5), float)
    assert ec.cdf(1.0, 1.0) == 1.0
    grid = ec.cdf(q[:5, 0][:, None], q[:5, 1][None, :])
    assert grid.shape == (5, 5)


def test_empirical_cdf_d_dim():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    ec = cs.EmpiricalCopula(X)
    assert ec.dim == 3
    q = rng.random((10, 3))
    brute = np.array([np.mean(np.all(p >= ec.U, axis=1)) for p in q])
    np.testing.assert_allclose(ec.cdf(q), brute)
    with pytest.raises(ValueError):
        ec.spearmans_rho()


@pytest.mark.parametrize("m", [4, 7, 10])
def test_checkerboard_has_uniform_margins(m):
    X = cs.sample(cp.Clayton(theta=2), 103, random_state=0)
    P = cs.checkerboard_mass(X, m)
    assert P.shape == (m, m)
    np.testing.assert_allclose(P.sum(axis=0), 1 / m, atol=1e-12)
    np.testing.assert_allclose(P.sum(axis=1), 1 / m, atol=1e-12)
    ec = cs.EmpiricalCopula(X)
    cb = ec.to_checkerboard(m)
    assert type(cb).__name__ == "BivCheckPi"
    # rho of the checkerboard is close to the sample rho
    assert abs(cb.spearmans_rho() - ec.spearmans_rho()) < 0.25


def test_checkerboard_of_comonotone_sample():
    x = np.arange(20.0)
    P = cs.checkerboard_mass(np.column_stack([x, x]), 5)
    np.testing.assert_allclose(P, np.eye(5) / 5)
    ec = cs.EmpiricalCopula(np.column_stack([x, x]))
    assert type(ec.to_checkerboard(5, "CheckMin")).__name__ == "BivCheckMin"
    assert ec.to_checkerboard(5, "CheckMin").spearmans_rho() == pytest.approx(1.0)


def test_bernstein():
    X = cs.sample(cp.Frank(theta=5), 400, random_state=2)
    b = cs.EmpiricalCopula(X).to_bernstein(8)
    assert "Bernstein" in type(b).__name__
    assert float(b.cdf(0.5, 1.0)) == pytest.approx(0.5, abs=1e-12)
    assert 0 < b.spearmans_rho() < 1


def test_method_names_and_measures_dict():
    X = cs.sample(cp.Gaussian(rho=0.5), 300, random_state=1)
    ec = cs.EmpiricalCopula(X)
    d = ec.measures()
    assert list(d) == ["xi", "rho", "tau", "footrule", "gamma", "beta", "nu"]
    assert ec.spearmans_rho() == d["rho"]
    assert ec.kendalls_tau() == d["tau"]
    assert ec.measure("spearman") == d["rho"]
    assert ec.chatterjees_xi(condition_on_y=True) == ec.measure("xi_2")
    assert ec.lambda_L(k=10) == ec.measure("lambda_l", k=10)
    for key in ec.available_measures():
        assert np.isfinite(ec.measure(key))
    assert repr(ec) == "EmpiricalCopula(n=300, dim=2)"


def test_plots():
    import matplotlib.pyplot as plt

    X = cs.sample(cp.Clayton(theta=1), 150, random_state=0)
    ec = cs.EmpiricalCopula(X)
    ax = ec.scatter_plot()
    assert ax is not None
    ax2 = ec.plot_contour(grid=21, compare_with=cp.Clayton(theta=1))
    assert ax2 is not None
    plt.close("all")
