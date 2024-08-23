import numpy as np
import pytest

from copul.exceptions import PropertyUnavailableException
from copul.families.other.upper_checkerboard import UpperCheckerboardCopula


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
            0.25,
        ),
        (
            [[1, 5, 4], [5, 3, 2], [4, 2, 4]],
            (1, 0.5),
            0.5,
        ),
    ],
)
def test_ccop_cdf(matr, point, expected):
    ccop = UpperCheckerboardCopula(matr)
    assert ccop.cdf(*point) == expected


def test_ccop_pdf():
    ccop = UpperCheckerboardCopula([[1, 0], [0, 1]])
    with pytest.raises(PropertyUnavailableException):
        ccop.pdf()


@pytest.mark.parametrize(
    "matr, point, expected",
    [
        (
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            (0.5, 0.5),
            0.5,
        ),
        (
            [[1, 5, 4], [5, 3, 2], [4, 2, 4]],
            (0.5, 0.5),
            0.8,
        ),
        ([[1, 1], [1, 1]], (0.2, 0.1), 0),
        ([[1, 1], [1, 1]], (0.1, 0.2), 0.5),
        ([[1, 1], [1, 1]], (0.7, 0.6), 0.5),
        ([[1, 1], [1, 1]], (0.6, 0.7), 1),
    ],
)
def test_ccop_cond_distr_1(matr, point, expected):
    ccop = UpperCheckerboardCopula(matr)
    assert ccop.cond_distr_1(*point) == expected


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
        ([[1, 1], [1, 1]], (0.8, 0.25), 1),
        ([[1, 1], [1, 1]], (0.7, 0.25), 0.5),
    ],
)
def test_ccop_cond_distr_2(matr, point, expected):
    ccop = UpperCheckerboardCopula(matr)
    result = ccop.cond_distr_2(*point)
    assert np.isclose(result, expected)


@pytest.mark.parametrize(
    "matr, expected",
    [
        (
            [[1, 0], [0, 1]],
            1,
        ),
    ],
)
def test_ccop_xi(matr, expected):
    ccop = UpperCheckerboardCopula(matr)
    xi_estimate = ccop.chatterjees_xi()
    assert np.abs(xi_estimate - expected) < 0.01


def test_upper_ccop_rvs():
    ccop = UpperCheckerboardCopula([[1]])
    sample_data = ccop.rvs(3)
    for data in sample_data:
        assert np.isclose(data[0], data[1])
