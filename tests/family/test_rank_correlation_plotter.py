from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import sympy as sp
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from copul.family.rank_correlation_plotter import (
    RankCorrelationPlotter,
    CorrelationData,
    MEASURES,
    measure,
)

# ----------------------------
# Helpers / fixtures
# ----------------------------


class _BaseDummy:
    params = [sp.symbols("theta")]

    def __init__(
        self, interval=(0.0, 5.0), left_open=False, right_open=False, theta=None
    ):
        self._interval = SimpleNamespace(
            inf=float(interval[0]),
            sup=float(interval[1]),
            left_open=left_open,
            right_open=right_open,
        )
        self.intervals = {"theta": self._interval}
        self.theta = theta

    def __call__(self, **kwargs):
        theta = kwargs.get("theta", self.theta)
        return self.__class__(
            interval=(self._interval.inf, self._interval.sup),
            left_open=self._interval.left_open,
            right_open=self._interval.right_open,
            theta=theta,
        )

    def rvs(self, n_obs, approximate=False):
        t = 0.0 if self.theta is None else float(self.theta)
        rng = np.random.default_rng(1234 + int(round(t * 10)))
        x = rng.normal(loc=t, scale=1.0, size=n_obs)
        y = x * 0.8 + rng.normal(scale=0.2, size=n_obs)
        return np.column_stack([x, y])


class DummyCopulaNoGet(_BaseDummy):
    """No get_params present (hasattr -> False)."""

    pass


class DummyCopulaWithGet(_BaseDummy):
    """Provides get_params; used to test preference branch."""

    def get_params(self, n_params, log_scale=False):
        if log_scale:
            return np.logspace(-1, 1, n_params) + float(self._interval.inf)
        return np.linspace(self._interval.inf, self._interval.sup, n_params)


@pytest.fixture(autouse=True)
def no_show():
    with patch.object(plt, "show", return_value=None):
        yield


@pytest.fixture
def restore_measures():
    snapshot = dict(MEASURES)
    try:
        yield
    finally:
        MEASURES.clear()
        MEASURES.update(snapshot)


# ----------------------------
# param_grid tests
# ----------------------------


def test_param_grid_prefers_family_get_params():
    cop = DummyCopulaWithGet()
    runner = RankCorrelationPlotter(cop)
    grid = runner.param_grid(n_params=7, log_cut_off=None)
    assert len(grid) == 7
    assert np.isclose(grid[0], cop.intervals["theta"].inf)
    assert np.isclose(grid[-1], cop.intervals["theta"].sup)


def test_param_grid_finite_set_fallback():
    cop = DummyCopulaNoGet()
    cop.intervals = {"theta": sp.FiniteSet(1.0, 3.0, 2.0)}
    runner = RankCorrelationPlotter(cop)
    grid = runner.param_grid(n_params=10)
    assert set(grid.tolist()) == {1.0, 2.0, 3.0}


def test_param_grid_linear_with_open_bounds_and_xlim():
    cop = DummyCopulaNoGet(interval=(0.0, 5.0), left_open=True, right_open=True)
    runner = RankCorrelationPlotter(cop)
    grid = runner.param_grid(n_params=5, xlim=(0.5, 4.5))
    assert len(grid) == 5
    assert grid[0] >= 0.5
    assert grid[-1] <= 4.5
    assert np.all(np.diff(grid) > 0)


def test_param_grid_shifted_log_from_bounds_tuple():
    cop = DummyCopulaNoGet(interval=(2.0, 100.0))  # inf = 2.0
    runner = RankCorrelationPlotter(cop)
    a, b = -2.0, 2.0
    grid = runner.param_grid(n_params=9, log_cut_off=(a, b))
    expected = np.logspace(a, b, 9) + 2.0
    assert np.allclose(grid, expected)


# ----------------------------
# compute tests
# ----------------------------


def test_compute_calls_registered_measures_and_handles_exceptions(restore_measures):
    calls = {"good": 0, "maybe_bad": 0}

    @measure("good_measure")
    def good_m(x, y, rx, ry):
        calls["good"] += 1
        return float(np.corrcoef(rx, ry)[0, 1])

    @measure("fragile_measure")
    def fragile_m(x, y, rx, ry):
        calls["maybe_bad"] += 1
        t_est = float(np.mean(x))
        if t_est > 2.5:
            raise ValueError("boom")
        return 42.0

    cop = DummyCopulaNoGet()
    # Put the fragile measure FIRST so if it fails, nothing has been appended yet
    runner = RankCorrelationPlotter(
        cop, measures=["fragile_measure", "good_measure"], save_pickles=False
    )
    params = np.array([0.0, 1.0, 3.0])
    data = runner.compute(params=params, n_obs=5000, approximate=False)

    assert set(data.values.keys()) == {"good_measure", "fragile_measure"}
    # Exactly one value per theta
    assert len(data.values["good_measure"]) == 3
    assert len(data.values["fragile_measure"]) == 3
    # Last theta failed -> NaN for both
    assert np.isnan(data.values["fragile_measure"][-1])
    assert np.isnan(data.values["good_measure"][-1])

    # Call counts: fragile called for all 3 thetas; good only for the first 2
    assert calls["maybe_bad"] == 3
    assert calls["good"] == 2


# ----------------------------
# plot tests
# ----------------------------


def test_plot_returns_splines_and_writes_png(tmp_path):
    cop = DummyCopulaNoGet()
    runner = RankCorrelationPlotter(cop, images_dir=tmp_path, save_pickles=False)

    params = np.array([0.5, 1.0, 1.5])
    values = {
        "m1": np.array([0.1, 0.2, 0.25]),
        "m2": np.array([np.nan, 0.0, 0.05]),
    }
    data = CorrelationData(params=params, values=values)

    splines = runner.plot(
        data,
        title="DummyTitle",
        ylim=(-1, 1),
        log_x=False,
    )
    assert "m1" in splines and "m2" in splines

    out = tmp_path / "DummyTitle_rank_correlations.png"
    assert out.exists()


def test_plot_log_axis_with_shift_and_cutoffs(tmp_path):
    cop = DummyCopulaNoGet(interval=(2.0, 10.0))
    runner = RankCorrelationPlotter(cop, images_dir=tmp_path, save_pickles=False)

    params = np.array([2.1, 3.0, 6.0, 9.0])
    values = {"m": np.array([0.0, 0.2, 0.4, 0.6])}
    data = CorrelationData(params=params, values=values)

    _ = runner.plot(
        data,
        title="Loggy",
        ylim=(-1, 1),
        log_x=True,
        log_cut_off=(-1.0, 1.0),
    )
    out = tmp_path / "Loggy_rank_correlations.png"
    assert out.exists()


# ----------------------------
# save tests
# ----------------------------


def test_save_pickles_toggle(tmp_path):
    cop = DummyCopulaNoGet()
    runner_yes = RankCorrelationPlotter(cop, images_dir=tmp_path, save_pickles=True)
    RankCorrelationPlotter(cop, images_dir=tmp_path, save_pickles=False)

    params = np.array([0.0, 1.0])
    values = {"m": np.array([0.1, 0.2])}
    data = CorrelationData(params=params, values=values)

    from scipy.interpolate import CubicSpline

    cs = CubicSpline(params, values["m"])
    splines = {"m": cs}

    runner_yes.save("Base", data, splines)
    data_pkl = tmp_path / "data" / "Base_data.pkl"
    cs_pkl = tmp_path / "splines" / "Base_splines.pkl"
    assert data_pkl.exists()
    assert cs_pkl.exists()

    tmp2 = tmp_path / "off"
    tmp2.mkdir()
    runner_no2 = RankCorrelationPlotter(cop, images_dir=tmp2, save_pickles=False)
    runner_no2.save("Base", data, splines)
    assert not (tmp2 / "data" / "Base_data.pkl").exists()
    assert not (tmp2 / "splines" / "Base_splines.pkl").exists()
