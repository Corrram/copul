import numpy as np
import pytest

from copul.checkerboard.check_min import CheckMin
from copul.exceptions import PropertyUnavailableException


def test_2x2x2_multivar_checkerboard():
    # 3-dim matrix
    matr = np.full((2, 2, 2), 0.5)
    copula = CheckMin(matr)
    u = (0.5, 0.5, 0.5)
    with pytest.raises(PropertyUnavailableException):
        copula.pdf(*u)
    assert copula.cdf(*u) == 0.125
    expected_cd = 0.25  # 1/4 for (0,0,0) block + 0 for all other blocks
    assert copula.cond_distr(1, u) == expected_cd
    assert copula.cond_distr(2, u) == expected_cd
    assert copula.cond_distr(3, u) == expected_cd


def test_3x3x3_multivar_checkerboard():
    # 3-dim matrix
    matr = np.full((3, 3, 3), 1)
    copula = CheckMin(matr)
    u = (0.5, 0.5, 0.5)
    with pytest.raises(PropertyUnavailableException):
        copula.pdf(*u)
    expected_cdf = (
        1 / 6
    )  # 1/27 for (0,0,0) block + 7*1/27 for (0,0,1),...,(1,1,1) blocks
    assert np.isclose(copula.cdf(*u), expected_cdf)
    expected_cd = (
        5 / 18  # 1/9 for (0,0,0) block + 3/2*1/9 for (0,1,1),(1,0,1),(1,1,1) blocks
    )
    actual_cd1 = copula.cond_distr(1, u)
    assert actual_cd1 == expected_cd
    assert copula.cond_distr(2, u) == expected_cd
    assert copula.cond_distr(3, u) == expected_cd


@pytest.mark.parametrize(
    "matr, point, expected",
    [
        ([[1, 1], [1, 1]], (0.2, 0.1), 0),
        ([[1, 1], [1, 1]], (0.1, 0.2), 0.5),
        ([[1, 1], [1, 1]], (0.7, 0.6), 0.5),
        ([[1, 1], [1, 1]], (0.6, 0.7), 1),
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
    ],
)
def test_ccop_cond_distr(matr, point, expected):
    ccop = CheckMin(matr)
    actual = ccop.cond_distr(1, point)
    assert np.isclose(actual, expected)
