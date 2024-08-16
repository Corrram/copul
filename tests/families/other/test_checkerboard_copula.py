import numpy as np
import pytest

from copul import CheckerboardCopula


@pytest.mark.parametrize(
    "matr, expected",
    [
        (
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            0.25,
        ),
        (
            [[1, 5, 4], [5, 3, 2], [4, 2, 4]],
            0.225,
        ),
    ],
)
def test_ccop_cdf(matr, expected):
    ccop = CheckerboardCopula(matr)
    assert ccop.cdf(0.5, 0.5) == expected


@pytest.mark.parametrize(
    "matr, expected",
    [
        (
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            0.0625,
        ),
        (
            [[1, 5, 4], [5, 3, 2], [4, 2, 4]],
            0.1,
        ),
    ],
)
def test_ccop_pdf(matr, expected):
    ccop = CheckerboardCopula(matr)
    assert ccop.pdf(0.5, 0.5) == expected


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
            2/3,
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
    xi_estimate = ccop.xi()
    assert np.abs(xi_estimate - expected) < 0.01
