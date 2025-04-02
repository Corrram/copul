

import numpy as np
from copul.checkerboard.biv_bernstein import BivBernsteinCopula


def test_tau_example():
    """Test tau for the example matrix from the original code."""
    matr = np.array([[1, 5, 4], [5, 3, 2], [4, 2, 4]])
    ccop = BivBernsteinCopula(matr)
    tau = ccop.tau()

    # Check range and expected sign (this matrix has positive dependence)
    assert -1 <= tau <= 1
    assert tau < 0

def test_rho_example():
    """Test tau for the example matrix from the original code."""
    matr = np.array([[1, 5, 4], [5, 3, 2], [4, 2, 4]])
    ccop = BivBernsteinCopula(matr)
    rho = ccop.rho()

    # Check range and expected sign (this matrix has positive dependence)
    assert -1 <= rho <= 1
    assert rho < 0

def test_measures_of_association_with_rectangular_matrix():
    """Test that tau and rho are consistent for a rectangular matrix."""
    matr = [
        [
            0.258794517498538,
            0.3467253550730139,
            0.39100995184938075,
            0.41768373795216235,
        ],
        [
            0.4483122636880096,
            0.3603814261135337,
            0.3160968293371668,
            0.2894230432343852,
        ],
    ]
    ccop = BivBernsteinCopula(matr)
    tau = ccop.tau()
    rho = ccop.rho()
    assert 1 > rho > -1
    assert 1 > tau > -1
    xi1 = ccop.xi(condition_on_y=False)
    xi2 = ccop.xi(condition_on_y=True)
    assert 1 > xi1 > 0
    assert 1 > xi2 > 0