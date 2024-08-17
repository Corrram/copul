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
