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
    "matr, expected",
    [
        (
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            0.5,
        ),
        (
            [[1, 5, 4], [5, 3, 2], [4, 2, 4]],
            0.8,
        ),
    ],
)
def test_ccop_cond_distr_1(matr, expected):
    ccop = UpperCheckerboardCopula(matr)
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
    ccop = UpperCheckerboardCopula(matr)
    result = ccop.cond_distr_2(0.5, 0.5)
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
