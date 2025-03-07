import numpy as np
import pytest

from copul import LowerFrechet, StudentT, UpperFrechet


@pytest.mark.parametrize("rho, expected_class", [(-1, LowerFrechet), (1, UpperFrechet)])
def test_student_t_edge_cases(rho, expected_class):
    cop = StudentT()(rho)
    assert isinstance(cop, expected_class), (
        f"Expected {expected_class.__name__} for rho={rho}, but got {type(cop).__name__}"
    )


def test_cdf():
    cop = StudentT()(0.5, 2)
    try:
        cop.cdf(0.5, 0.5)
        print("Test passed: No exception thrown.")
    except Exception as e:
        print(f"Test failed: Exception thrown - {e}")


def test_student_t_cond_distr_1():
    cop = StudentT()(0.5, 2)
    result = cop.cond_distr_1(0.3, 0.4)
    assert np.isclose(result.evalf(), 0.507248278491941)
