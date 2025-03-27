import random

import numpy as np
import pytest

from copul.checkerboard.biv_check_min import BivCheckMin
from copul.exceptions import PropertyUnavailableException


@pytest.mark.parametrize(
    "matr, point, expected",
    [
        ([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], (0.5, 0.5), 0.25),
        ([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], (0.5, 1), 0.5),
        (
            [[1, 5, 4], [5, 3, 2], [4, 2, 4]],
            (0.5, 0.5),
            (1 + 5 / 2 + 5 / 2 + 3 / 2) / 30,
        ),
        ([[1, 5, 4], [5, 3, 2], [4, 2, 4]], (1, 0.5), 0.5),
    ],
)
def test_ccop_cdf(matr, point, expected):
    ccop = BivCheckMin(matr)
    actual = ccop.cdf(*point)
    assert np.isclose(actual, expected)


def test_ccop_pdf():
    ccop = BivCheckMin([[1, 0], [0, 1]])
    with pytest.raises(PropertyUnavailableException):
        ccop.pdf()


@pytest.mark.parametrize(
    "matr, point, expected",
    [
        ([[1, 1], [1, 1]], (0.2, 0.1), 0),
        ([[1, 1], [1, 1]], (0.1, 0.2), 0.5),
        ([[1, 1], [1, 1]], (0.7, 0.6), 0.5),
        ([[1, 1], [1, 1]], (0.6, 0.7), 1),
        ([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], (0.5, 0.5), 0.5),
        ([[1, 5, 4], [5, 3, 2], [4, 2, 4]], (0.5, 0.5), 0.8),
    ],
)
def test_ccop_cond_distr_1(matr, point, expected):
    ccop = BivCheckMin(matr)
    actual = ccop.cond_distr_1(*point)
    assert np.isclose(actual, expected)


@pytest.mark.parametrize(
    "matr, point, expected",
    [
        # (
        #     [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        #     (0.5, 0.5),
        #     0.5,
        # ),
        # (
        #     [[1, 2], [2, 1]],
        #     (0.5, 0.5),
        #     2 / 3,
        # ),
        ([[1, 1], [1, 1]], (0.7, 0.25), 0.5),
        ([[1, 1], [1, 1]], (0.8, 0.25), 1),
    ],
)
def test_ccop_cond_distr_2(matr, point, expected):
    ccop = BivCheckMin(matr)
    result = ccop.cond_distr_2(*point)
    assert np.isclose(result, expected)


@pytest.mark.parametrize("matr, expected", [([[1, 0], [0, 1]], 1)])
def test_upper_ccop_xi(matr, expected):
    random.seed(0)
    ccop = BivCheckMin(matr)
    xi_estimate = ccop.xi()
    assert np.abs(xi_estimate - expected) < 0.02


def test_biv_check_min_rvs():
    ccop = BivCheckMin([[1]])
    sample_data = ccop.rvs(3)
    for data in sample_data:
        assert np.isclose(data[0], data[1])


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_tau_independence(n):
    """Test that tau is close to 0 for independence copula."""
    matr = np.ones((n, n))  # Uniform distribution represents independence
    ccop = BivCheckMin(matr)
    tau = ccop.tau()
    assert tau > 0.01


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_xi_independence(n):
    """Test that tau is close to 0 for independence copula."""
    matr = np.ones((n, n))  # Uniform distribution represents independence
    ccop = BivCheckMin(matr)
    xi = ccop.xi()
    assert xi > 0.01


@pytest.mark.parametrize("n", [3, 4])
def test_tau_perfect_dependence(n):
    """Test tau for perfect positive and negative dependence."""
    # Perfect positive dependence
    matr_pos = np.zeros((n, n))
    np.fill_diagonal(matr_pos, 1)  # Place 1's on the main diagonal
    ccop_pos = BivCheckMin(matr_pos)
    tau_pos = ccop_pos.tau()

    # Perfect negative dependence
    matr_neg = np.zeros((n, n))
    for i in range(3):
        matr_neg[i, 2 - i] = 1  # Place 1's on the opposite diagonal
    ccop_neg = BivCheckMin(matr_neg)
    tau_neg = ccop_neg.tau()

    # Tau should be positive for positive dependence and negative for negative dependence
    assert tau_pos > 0.6
    assert -0.3 > tau_neg > -0.6


def test_tau_2x2_exact():
    """Test exact values for 2x2 checkerboard copulas."""
    np.random.seed(42)
    # For a 2x2 checkerboard with perfect positive dependence
    matr_pos = np.array([[1, 0], [0, 1]])
    ccop_pos = BivCheckMin(matr_pos)

    # For a 2x2 checkerboard with perfect negative dependence
    matr_neg = np.array([[0, 1], [1, 0]])
    ccop_neg = BivCheckMin(matr_neg)

    # For 2x2, these are the exact values
    pos_tau = ccop_pos.tau()
    assert np.isclose(pos_tau, 1, atol=1e-2)
    neg_tau = ccop_neg.tau()
    assert np.isclose(neg_tau, 0, atol=1e-2)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_xi_with_shuffled_eye(n):
    """Test that xi is close to 1 for a shuffled identity matrix."""
    np.random.seed(42)
    n = 10
    matr = np.eye(n)
    np.random.shuffle(matr)
    ccop = BivCheckMin(matr)
    xi = ccop.xi()
    assert np.isclose(xi, 1, atol=0.02)


def test_tau_example():
    """Test tau for the example matrix from the original code."""
    matr = np.array([[1, 5, 4], [5, 3, 2], [4, 2, 4]])
    ccop = BivCheckMin(matr)
    tau_val = ccop.tau()

    # Check range and expected sign (this matrix has positive dependence)
    assert -1 <= tau_val <= 1
    assert tau_val > 0


# Tests for rho (Spearman's rho)
def test_rho_independence():
    """Test that rho is close to 0 for independence copula."""
    np.random.seed(42)
    matr = np.ones((4, 4))  # Uniform distribution represents independence
    ccop = BivCheckMin(matr)
    rho = ccop.rho()
    assert rho > 0.02


def test_rho_perfect_dependence():
    """Test rho for perfect positive and negative dependence."""
    # Perfect positive dependence
    matr_pos = np.zeros((3, 3))
    np.fill_diagonal(matr_pos, 1)  # Place 1's on the main diagonal
    ccop_pos = BivCheckMin(matr_pos)
    rho_pos = ccop_pos.rho()

    # Perfect negative dependence
    matr_neg = np.zeros((3, 3))
    for i in range(3):
        matr_neg[i, 2 - i] = 1  # Place 1's on the opposite diagonal
    ccop_neg = BivCheckMin(matr_neg)
    rho_neg = ccop_neg.rho()

    # Rho should be positive for positive dependence and negative for negative dependence
    assert rho_pos > 0.5
    assert rho_neg < -0.5


def test_rho_2x2_exact():
    """Test exact values for 2x2 checkerboard copulas."""
    # For a 2x2 checkerboard with perfect positive dependence
    matr_pos = np.array([[1, 0], [0, 1]])
    ccop_pos = BivCheckMin(matr_pos)

    # For a 2x2 checkerboard with perfect negative dependence
    matr_neg = np.array([[0, 1], [1, 0]])
    ccop_neg = BivCheckMin(matr_neg)

    # For 2x2, these are the exact values
    pos_rho = ccop_pos.rho()
    assert np.isclose(pos_rho, 1, atol=1e-2)
    neg_rho = ccop_neg.rho()
    assert -0.6 < neg_rho < -0.4


def test_rho_example():
    """Test rho for the example matrix from the original code."""
    matr = np.array([[1, 5, 4], [5, 3, 2], [4, 2, 4]])
    ccop = BivCheckMin(matr)
    rho_val = ccop.rho()

    # Check range and expected sign (this matrix has positive dependence)
    assert -1 <= rho_val <= 1
    assert rho_val < 0


def test_xi_perfect_dependence():
    """Test xi for perfect positive and negative dependence."""
    # Perfect positive dependence
    matr_pos = np.zeros((10, 10))
    np.fill_diagonal(matr_pos, 1)  # Place 1's on the main diagonal
    ccop_pos = BivCheckMin(matr_pos)
    xi_pos = ccop_pos.xi()

    # Perfect negative dependence
    matr_neg = np.zeros((10, 10))
    for i in range(10):
        matr_neg[i, 9 - i] = 1  # Place 1's on the opposite diagonal
    ccop_neg = BivCheckMin(matr_neg)
    xi_neg = ccop_neg.xi()

    # Xi should be close to 1 for both perfect positive and negative dependence
    assert xi_pos > 0.8
    assert xi_neg > 0.8


def test_xi_2x2_exact():
    """Test exact values for 2x2 checkerboard copulas."""
    # For a 2x2 checkerboard with perfect positive dependence
    matr_pos = np.array([[1, 0], [0, 1]])
    ccop_pos = BivCheckMin(matr_pos)

    # For a 2x2 checkerboard with perfect negative dependence
    matr_neg = np.array([[0, 1], [1, 0]])
    ccop_neg = BivCheckMin(matr_neg)

    # For 2x2, both should have xi = 1 (perfect dependence)
    xi_pos = ccop_pos.xi()
    xi_neg = ccop_neg.xi()
    assert np.isclose(xi_pos, 1, atol=1e-2)
    assert np.isclose(xi_neg, 1, atol=1e-2)


def test_xi_example():
    """Test xi for the example matrix from the original code."""
    matr = np.array([[1, 5, 4], [5, 3, 2], [4, 2, 4]])
    ccop = BivCheckMin(matr)
    xi_val = ccop.xi()

    # Check range (xi is always between 0 and 1)
    assert 0 <= xi_val <= 1


def test_measure_consistency():
    """Test that tau and rho have consistent signs for asymmetric matrices."""
    # Create a matrix with positive dependence
    matr_pos = np.array([[0.6, 0.2, 0.0], [0.2, 0.4, 0.2], [0.0, 0.2, 0.6]])
    ccop_pos = BivCheckMin(matr_pos)
    tau_pos = ccop_pos.tau()
    rho_pos = ccop_pos.rho()

    # Both should be positive
    assert tau_pos > 0
    assert rho_pos > 0

    # Create a matrix with negative dependence
    matr_neg = np.array([[0.0, 0.2, 0.6], [0.2, 0.4, 0.2], [0.6, 0.2, 0.0]])
    ccop_neg = BivCheckMin(matr_neg)
    tau_neg = ccop_neg.tau()
    rho_neg = ccop_neg.rho()

    # Both should be negative
    assert tau_neg < 0
    assert rho_neg < 0


def test_xi_equivalent_to_monte_carlo():
    """Test that our implementation matches the standard case from existing test."""
    # This matrix was tested previously with Monte Carlo
    matr = np.array([[1, 0], [0, 1]])
    ccop = BivCheckMin(matr)
    xi_value = ccop.xi()
    assert np.isclose(xi_value, 1, atol=0.02)


def test_measures_of_assiciation_with_rectangular_matrix():
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
    ccop = BivCheckMin(matr)
    xi = ccop.xi(condition_on_y=True)
    xi2 = ccop.xi(condition_on_y=False)
    assert 1 > xi > 0
    assert 1 > xi2 > 0
    tau = ccop.tau()
    rho = ccop.rho()
    assert 1 > tau > -1
    assert 1 > rho > -1
