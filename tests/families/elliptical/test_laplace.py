import pytest

from copul.families.elliptical.laplace import Laplace
from copul.families.other import LowerFrechet, UpperFrechet


@pytest.mark.parametrize("rho, expected_class", [(-1, LowerFrechet), (1, UpperFrechet)])
def test_laplace_edge_cases(rho, expected_class):
    cop = Laplace()(rho)
    assert isinstance(cop, expected_class), (
        f"Expected {expected_class.__name__} for rho={rho}, but got {type(cop).__name__}"
    )
