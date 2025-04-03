

import numpy as np
from copul.checkerboard.biv_bernstein import BivBernsteinCopula
from copul.families.other.lower_frechet import LowerFrechet
from copul.families.other.upper_frechet import UpperFrechet


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

def test_tau_independence():
    """Test that rho is close to 0 for independence copula."""
    np.random.seed(42)
    matr = np.ones((4, 4))  # Uniform distribution represents independence
    ccop = BivBernsteinCopula(matr)
    tau = ccop.tau()
    assert np.isclose(tau, 0)

def test_rho_independence():
    """Test that rho is close to 0 for independence copula."""
    np.random.seed(42)
    matr = np.ones((4, 4))  # Uniform distribution represents independence
    ccop = BivBernsteinCopula(matr)
    rho = ccop.rho()
    assert np.isclose(rho, 0)


def test_tau_perfect_dependence():
    """Test tau for perfect positive and negative dependence."""
    # Perfect positive dependence
    matr_pos = np.zeros((2, 2))
    np.fill_diagonal(matr_pos, 1)  # Place 1's on the main diagonal
    ccop_pos = BivBernsteinCopula(matr_pos)
    tau_pos = ccop_pos.tau()

    # Perfect negative dependence
    matr_neg = np.zeros((2, 2))
    for i in range(2):
        matr_neg[i, 1 - i] = 1  # Place 1's on the opposite diagonal
    ccop_neg = BivBernsteinCopula(matr_neg)
    tau_neg = ccop_neg.tau()

    bern_up = UpperFrechet().to_bernstein()
    bern_low = LowerFrechet().to_bernstein()
    tau_up = bern_up.tau()
    tau_low = bern_low.tau()
    bern_up_2 = UpperFrechet().to_bernstein(2)
    bern_low_2 = LowerFrechet().to_bernstein(2)
    tau_up_2 = bern_up_2.tau()
    tau_low_2 = bern_low_2.tau()

    # tau should be positive for positive dependence and negative for negative dependence
    assert np.isclose(tau_pos, 2/9)
    assert np.isclose(tau_neg, -2/9)
    assert np.isclose(tau_up_2, 2/9)
    assert np.isclose(tau_low_2, -2/9)
    assert tau_up > 0.6
    assert tau_low < -0.6

def test_rho_perfect_dependence():
    """Test rho for perfect positive and negative dependence."""
    # Perfect positive dependence
    matr_pos = np.zeros((2, 2))
    np.fill_diagonal(matr_pos, 1)  # Place 1's on the main diagonal
    ccop_pos = BivBernsteinCopula(matr_pos)
    rho_pos = ccop_pos.rho()

    # Perfect negative dependence
    matr_neg = np.zeros((2, 2))
    for i in range(2):
        matr_neg[i, 1 - i] = 1  # Place 1's on the opposite diagonal
    ccop_neg = BivBernsteinCopula(matr_neg)
    rho_neg = ccop_neg.rho()

    # Rho should be positive for positive dependence and negative for negative dependence
    assert np.isclose(rho_pos, 1/3)
    assert np.isclose(rho_neg, -1/3)

    bern_up = UpperFrechet().to_bernstein(5)
    bern_low = LowerFrechet().to_bernstein(5)
    rho_up = bern_up.rho()
    rho_low = bern_low.rho()
    assert rho_up > 0.6
    assert rho_low < -0.6


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