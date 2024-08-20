import numpy as np
import matplotlib
import pytest
from matplotlib import pyplot as plt

from copul import CheckerboardCopula

matplotlib.use("Agg")  # Use the 'Agg' backend to suppress the pop-up


@pytest.mark.parametrize(
    "matr, point, expected",
    [
        (
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            (0.5, 0.5),
            0.25,
        ),
        (
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            (0.5, 1),
            0.5,
        ),
        (
            [[1, 5, 4], [5, 3, 2], [4, 2, 4]],
            (0.5, 0.5),
            0.225,
        ),
        (
            [[1, 5, 4], [5, 3, 2], [4, 2, 4]],
            (1, 0.5),
            0.5,
        ),
    ],
)
def test_ccop_cdf(matr, point, expected):
    ccop = CheckerboardCopula(matr)
    assert ccop.cdf(*point) == expected


@pytest.fixture
def setup_checkerboard_copula():
    # Setup code for initializing the CheckerboardCopula instance
    matr = [[0, 9, 1], [1, 0, 9], [9, 1, 0]]
    return CheckerboardCopula(matr)


@pytest.mark.parametrize(
    "plotting_method",
    [
        lambda ccop: ccop.scatter_plot(),
        lambda ccop: ccop.plot_cdf(),
        lambda ccop: ccop.plot_pdf(),
        lambda ccop: ccop.plot(cd1=ccop.cond_distr_1, cd2=ccop.cond_distr_2),
    ],
)
def test_ccop_plotting(setup_checkerboard_copula, plotting_method):
    ccop = setup_checkerboard_copula

    plotting_method(ccop)
    try:
        plotting_method(ccop)
    except Exception as e:
        pytest.fail(f"{plotting_method.__name__} raised an exception: {e}")
    finally:
        plt.close("all")


@pytest.mark.parametrize(
    "matr, point, expected",
    [
        (
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            (0.5, 0.5),
            0.0625,
        ),
        (
            [[1, 5, 4], [5, 3, 2], [4, 2, 4]],
            (0.5, 0.5),
            0.1,
        ),
        (
            [[1, 5, 4], [5, 3, 2], [4, 2, 4]],
            (0.5, 1),
            1 / 15,
        ),
    ],
)
def test_ccop_pdf(matr, point, expected):
    ccop = CheckerboardCopula(matr)
    result = ccop.pdf(*point)
    assert result == expected


@pytest.mark.parametrize(
    "matr, expected",
    [
        (
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            0.5,
        ),
        (
            [[1, 5, 4], [5, 3, 2], [4, 2, 4]],
            0.65,
        ),
    ],
)
def test_ccop_cond_distr_1(matr, expected):
    ccop = CheckerboardCopula(matr)
    assert ccop.cond_distr_1(0.5, 0.5) == expected


@pytest.mark.parametrize(
    "matr, expected",
    [
        (
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            0.5,
        ),
        (
            [[1, 2], [2, 1]],
            2 / 3,
        ),
    ],
)
def test_ccop_cond_distr_2(matr, expected):
    ccop = CheckerboardCopula(matr)
    result = ccop.cond_distr_2(0.5, 0.5)
    assert np.isclose(result, expected)


@pytest.mark.parametrize(
    "matr, expected",
    [
        (
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            0,
        ),
    ],
)
def test_ccop_xi(matr, expected):
    ccop = CheckerboardCopula(matr)
    xi_estimate = ccop.chatterjees_xi()
    assert np.abs(xi_estimate - expected) < 0.01
