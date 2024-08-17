import pytest
from copul.families.other import LowerFrechet, UpperFrechet
from copul.families.elliptical.laplace import Laplace


@pytest.mark.parametrize(
    "rho, expected_class",
    [
        (-1, LowerFrechet),
        (1, UpperFrechet),
    ],
)
def test_gaussian_edge_cases(rho, expected_class):
    cop = Laplace()(rho)
    assert isinstance(
        cop, expected_class
    ), f"Expected {expected_class.__name__} for rho={rho}, but got {type(cop).__name__}"
