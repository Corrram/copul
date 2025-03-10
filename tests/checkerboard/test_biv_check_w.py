import random

import numpy as np
import pytest

from copul.checkerboard.biv_check_w import BivCheckW
from copul.exceptions import PropertyUnavailableException


@pytest.mark.parametrize(
    "matr, point, expected",
    [
        ([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], (0.5, 0.5), 0.25),
        ([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], (0.5, 1), 0.5),
        (
            [[1, 5, 4], [5, 3, 2], [4, 2, 4]],
            (0.5, 0.5),
            (1 + 5 / 2 + 5 / 2) / 30,
        ),
        ([[1, 5, 4], [5, 3, 2], [4, 2, 4]], (1, 0.5), 0.5),
    ],
)
def test_ccop_cdf(matr, point, expected):
    ccop = BivCheckW(matr)
    actual = ccop.cdf(*point)
    assert np.isclose(actual, expected)


def test_ccop_pdf():
    ccop = BivCheckW([[1, 0], [0, 1]])
    with pytest.raises(PropertyUnavailableException):
        ccop.pdf()


@pytest.mark.parametrize(
    "matr, point, expected",
    [
        ([[1, 1], [1, 1]], (0.2, 0.1), 0),
        ([[1, 1], [1, 1]], (0.1, 0.3), 0),
        ([[1, 1], [1, 1]], (0.1, 0.4), 0.5),
        ([[1, 1], [1, 1]], (0.1, 0.8), 0.5),
        ([[1, 1], [1, 1]], (0.1, 0.9), 1),
        ([[1, 1], [1, 1]], (0.1, 1), 1),
        ([[1, 1], [1, 1]], (0.7, 0.6), 0.5),
        ([[1, 1], [1, 1]], (0.6, 0.7), 0.5),
        ([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], (0.5, 0.5), 0.5),
        ([[1, 5, 4], [5, 3, 2], [4, 2, 4]], (0.5, 0.5), 0.8),
    ],
)
def test_ccop_cond_distr_1(matr, point, expected):
    ccop = BivCheckW(matr)
    actual = ccop.cond_distr_1(*point)
    assert np.isclose(actual, expected)


@pytest.mark.parametrize(
    "matr, point, expected",
    [
        # (
        #     [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        #     (0.5, 0.5),
        #     0.5,
        # ),
        # (
        #     [[1, 2], [2, 1]],
        #     (0.5, 0.5),
        #     2 / 3,
        # ),
        ([[1, 1], [1, 1]], (0.7, 0.25), 0.5),
        ([[1, 1], [1, 1]], (0.8, 0.25), 1),
    ],
)
def test_ccop_cond_distr_2(matr, point, expected):
    ccop = BivCheckW(matr)
    result = ccop.cond_distr_2(*point)
    assert np.isclose(result, expected)


@pytest.mark.parametrize("matr, expected", [([[0, 1], [1, 0]], 1)])
def test_lower_ccop_xi(matr, expected):
    random.seed(0)
    ccop = BivCheckW(matr)
    xi_estimate = ccop.chatterjees_xi()
    assert np.isclose(xi_estimate, expected, atol=0.02)


@pytest.mark.parametrize("matr", [[[1, 0], [0, 1]], [[1]]])
def test_biv_check_min_rvs(matr):
    ccop = BivCheckW(matr)
    sample_data = ccop.rvs(3)
    for data in sample_data:
        # (data[0] + data[1])*2 should approx be an integer
        check_sum = (data[0] + data[1]) * 2
        rounded_sum = round(check_sum)
        assert np.isclose(check_sum, rounded_sum)
