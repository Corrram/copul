import numpy as np
import pytest
from copul import IndependenceCopula, LowerFrechet, UpperFrechet
from copul.families.elliptical.gaussian import Gaussian


@pytest.mark.parametrize(
    "rho, expected_class",
    [
        (-1, LowerFrechet),
        (0, IndependenceCopula),
        (1, UpperFrechet),
    ],
)
def test_gaussian_edge_cases(rho, expected_class):
    cop = Gaussian()(rho)
    assert isinstance(
        cop, expected_class
    ), f"Expected {expected_class.__name__} for rho={rho}, but got {type(cop).__name__}"


def test_gaussian_rvs():
    cop = Gaussian()(0.5)
    assert cop.rvs(10).shape == (10, 2)


def test_gaussian_cdf():
    cop = Gaussian()(0.5)
    assert np.isclose(cop.cdf(0.5, 0.5), 1 / 3)


@pytest.mark.parametrize(
    "rho, expected",
    [
        (-1, -1),
        (0, 0),
        (1, 1),
    ],
)
def test_gaussian_tau(rho, expected):
    cop = Gaussian()(rho)
    assert cop.kendalls_tau() == expected


@pytest.mark.parametrize(
    "rho, expected",
    [
        (-1, 1),
        (0, 0),
        (1, 1),
    ],
)
def test_gaussian_xi(rho, expected):
    cop = Gaussian()(rho)
    assert cop.chatterjees_xi() == expected
