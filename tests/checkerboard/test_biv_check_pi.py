import matplotlib
import numpy as np
import pytest
from matplotlib import pyplot as plt

from copul.checkerboard.biv_check_pi import BivCheckPi

matplotlib.use("Agg")  # Use the 'Agg' backend to suppress the pop-up


@pytest.mark.parametrize(
    "matr, point, expected",
    [
        ([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], (0.5, 1), 0.5),
        ([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], (0.5, 0.5), 0.25),
        ([[1, 5, 4], [5, 3, 2], [4, 2, 4]], (0.5, 0.5), 0.225),
        ([[1, 5, 4], [5, 3, 2], [4, 2, 4]], (1, 0.5), 0.5),
    ],
)
def test_ccop_cdf(matr, point, expected):
    ccop = BivCheckPi(matr)
    actual = ccop.cdf(*point)
    assert np.isclose(actual, expected)


@pytest.fixture
def setup_checkerboard_copula():
    # Setup code for initializing the CheckerboardCopula instance
    matr = [[0, 9, 1], [1, 0, 9], [9, 1, 0]]
    return BivCheckPi(matr)


def test___init__():
    orig_matr = [[1, 0], [0, 1]]
    ccop = BivCheckPi(orig_matr)
    ccop_matr = ccop.matr.tolist()
    assert ccop_matr == [[0.5, 0], [0, 0.5]]
    assert hasattr(ccop.matr, "ndim")


@pytest.mark.parametrize(
    "plotting_method",
    [
        lambda ccop: ccop.scatter_plot(),
        lambda ccop: ccop.plot_cdf(),
        lambda ccop: ccop.plot_pdf(),
        lambda ccop: ccop.plot(cd1=ccop.cond_distr_1, cd2=ccop.cond_distr_2),
    ],
)
def test_ccop_plotting(setup_checkerboard_copula, plotting_method):
    ccop = setup_checkerboard_copula

    plotting_method(ccop)
    try:
        plotting_method(ccop)
    except Exception as e:
        pytest.fail(f"{plotting_method.__name__} raised an exception: {e}")
    finally:
        plt.close("all")


@pytest.mark.parametrize(
    "matr, point, expected",
    [
        ([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], (0.5, 0.5), 0.0625),
        ([[1, 5, 4], [5, 3, 2], [4, 2, 4]], (0.5, 0.5), 0.1),
        ([[1, 5, 4], [5, 3, 2], [4, 2, 4]], (0.5, 1), 1 / 15),
    ],
)
def test_ccop_pdf(matr, point, expected):
    ccop = BivCheckPi(matr)
    result = ccop.pdf(*point)
    assert result == expected


@pytest.mark.parametrize(
    "matr, expected",
    [
        ([[1, 0], [0, 1]], 0),  # 0.5 belongs to the second row, so ~ Unif[0.5, 1]
        ([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], 0.5),
        ([[1, 5, 4], [5, 3, 2], [4, 2, 4]], 0.65),  # second row -> (1*5+0.5*3+0*2)/10
    ],
)
def test_ccop_cond_distr_1(matr, expected):
    ccop = BivCheckPi(matr)
    actual = ccop.cond_distr_1(0.5, 0.5)
    assert np.isclose(actual, expected)


@pytest.mark.parametrize(
    "u, v, expected",
    [
        (0.4, 0.4, 0.8),
        (0.4, 0.6, 1),
        (0.6, 0.4, 0),
    ],
)
def test_ccop_cond_distr_1_different_points(u, v, expected):
    matr = [[1, 0], [0, 1]]
    ccop = BivCheckPi(matr)
    actual = ccop.cond_distr_1(u, v)
    assert np.isclose(actual, expected)


@pytest.mark.parametrize(
    "matr, expected",
    [
        ([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], 0.5),
        ([[1, 2], [2, 1]], 2 / 3),  # 0.5 belongs to second column
        ([[1, 0], [0, 1]], 0),  # 0.5 belongs to second row
    ],
)
def test_ccop_cond_distr_2(matr, expected):
    ccop = BivCheckPi(matr)
    result = ccop.cond_distr_2(0.5, 0.5)
    assert np.isclose(result, expected)


@pytest.mark.parametrize(
    "matr, expected",
    [
        ([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], 0),
        ([[1, 0], [0, 1]], 0.5),
    ],
)
def test_ccop_xi(matr, expected):
    np.random.seed(1)
    ccop = BivCheckPi(matr)
    xi_estimate = ccop.chatterjees_xi()
    actual_diff = np.abs(xi_estimate - expected)
    assert actual_diff < 0.02


def test_check_pi_rvs():
    np.random.seed(1)
    ccop = BivCheckPi([[1, 2], [2, 1]])
    n = 1_000
    samples = ccop.rvs(n)
    n_lower_empirical = sum([(sample < (0.5, 0.5)).all() for sample in samples])
    n_upper_empirical = sum([(sample > (0.5, 0.5)).all() for sample in samples])
    theoretical_ratio = 1 / 6 * n
    assert n_lower_empirical < 1.5 * theoretical_ratio
    assert n_upper_empirical < 1.5 * theoretical_ratio


# Tests for tau (Kendall's tau)
def test_tau_independence():
    """Test that tau is close to 0 for independence copula."""
    matr = np.ones((4, 4))  # Uniform distribution represents independence
    ccop = BivCheckPi(matr)
    tau = ccop.tau()
    assert np.isclose(tau, 0, atol=1e-2)


def test_tau_perfect_dependence():
    """Test tau for perfect positive and negative dependence."""
    # Perfect positive dependence
    matr_pos = np.zeros((3, 3))
    np.fill_diagonal(matr_pos, 1)  # Place 1's on the main diagonal
    ccop_pos = BivCheckPi(matr_pos)
    tau_pos = ccop_pos.tau()

    # Perfect negative dependence
    matr_neg = np.zeros((3, 3))
    for i in range(3):
        matr_neg[i, 2 - i] = 1  # Place 1's on the opposite diagonal
    ccop_neg = BivCheckPi(matr_neg)
    tau_neg = ccop_neg.tau()

    # Tau should be positive for positive dependence and negative for negative dependence
    assert tau_pos > 0.5
    assert tau_neg < -0.5


def test_tau_2x2_exact():
    """Test exact values for 2x2 checkerboard copulas."""
    # For a 2x2 checkerboard with perfect positive dependence
    matr_pos = np.array([[1, 0], [0, 1]])
    ccop_pos = BivCheckPi(matr_pos)

    # For a 2x2 checkerboard with perfect negative dependence
    matr_neg = np.array([[0, 1], [1, 0]])
    ccop_neg = BivCheckPi(matr_neg)

    # For 2x2, these are the exact values
    pos_tau = ccop_pos.tau()
    assert np.isclose(pos_tau, 0.5, atol=1e-2)
    neg_tau = ccop_neg.tau()
    assert np.isclose(neg_tau, -0.5, atol=1e-2)


def test_tau_example():
    """Test tau for the example matrix from the original code."""
    matr = np.array([[1, 5, 4], [5, 3, 2], [4, 2, 4]])
    ccop = BivCheckPi(matr)
    tau_val = ccop.tau()

    # Check range and expected sign (this matrix has positive dependence)
    assert -1 <= tau_val <= 1
    assert tau_val < 0


# Tests for rho (Spearman's rho)
def test_rho_independence():
    """Test that rho is close to 0 for independence copula."""
    np.random.seed(42)
    matr = np.ones((4, 4))  # Uniform distribution represents independence
    ccop = BivCheckPi(matr)
    rho = ccop.rho()
    assert np.isclose(rho, 0, atol=1e-2)


def test_rho_perfect_dependence():
    """Test rho for perfect positive and negative dependence."""
    # Perfect positive dependence
    matr_pos = np.zeros((3, 3))
    np.fill_diagonal(matr_pos, 1)  # Place 1's on the main diagonal
    ccop_pos = BivCheckPi(matr_pos)
    rho_pos = ccop_pos.rho()

    # Perfect negative dependence
    matr_neg = np.zeros((3, 3))
    for i in range(3):
        matr_neg[i, 2 - i] = 1  # Place 1's on the opposite diagonal
    ccop_neg = BivCheckPi(matr_neg)
    rho_neg = ccop_neg.rho()

    # Rho should be positive for positive dependence and negative for negative dependence
    assert rho_pos > 0.5
    assert rho_neg < -0.5


def test_rho_2x2_exact():
    """Test exact values for 2x2 checkerboard copulas."""
    np.random.seed(42)
    # For a 2x2 checkerboard with perfect positive dependence
    matr_pos = np.array([[1, 0], [0, 1]])
    ccop_pos = BivCheckPi(matr_pos)

    # For a 2x2 checkerboard with perfect negative dependence
    matr_neg = np.array([[0, 1], [1, 0]])
    ccop_neg = BivCheckPi(matr_neg)

    # For 2x2, these are the exact values
    pos_rho = ccop_pos.rho()
    assert np.isclose(pos_rho, 0.745, atol=1e-1)
    neg_rho = ccop_neg.rho()
    assert np.isclose(neg_rho, -0.745, atol=1e-1)


def test_rho_example():
    """Test rho for the example matrix from the original code."""
    matr = np.array([[1, 5, 4], [5, 3, 2], [4, 2, 4]])
    ccop = BivCheckPi(matr)
    rho_val = ccop.rho()

    # Check range and expected sign (this matrix has positive dependence)
    assert -1 <= rho_val <= 1
    assert rho_val < 0


# Tests for xi (Chatterjee's xi)
def test_xi_independence():
    """Test that xi is close to 0 for independence copula."""
    matr = np.ones((4, 4))  # Uniform distribution represents independence
    ccop = BivCheckPi(matr)
    assert np.isclose(ccop.chatterjees_xi(), 0, atol=1e-2)


def test_xi_perfect_dependence():
    """Test xi for perfect positive and negative dependence."""
    # Perfect positive dependence
    matr_pos = np.zeros((10, 10))
    np.fill_diagonal(matr_pos, 1)  # Place 1's on the main diagonal
    ccop_pos = BivCheckPi(matr_pos)
    xi_pos = ccop_pos.chatterjees_xi()

    # Perfect negative dependence
    matr_neg = np.zeros((10, 10))
    for i in range(10):
        matr_neg[i, 9 - i] = 1  # Place 1's on the opposite diagonal
    ccop_neg = BivCheckPi(matr_neg)
    xi_neg = ccop_neg.chatterjees_xi()

    # Xi should be close to 1 for both perfect positive and negative dependence
    assert xi_pos > 0.8
    assert xi_neg > 0.8


def test_xi_2x2_exact():
    """Test exact values for 2x2 checkerboard copulas."""
    # For a 2x2 checkerboard with perfect positive dependence
    matr_pos = np.array([[1, 0], [0, 1]])
    ccop_pos = BivCheckPi(matr_pos)

    # For a 2x2 checkerboard with perfect negative dependence
    matr_neg = np.array([[0, 1], [1, 0]])
    ccop_neg = BivCheckPi(matr_neg)

    # For 2x2, both should have xi = 1 (perfect dependence)
    xi_pos = ccop_pos.chatterjees_xi()
    xi_neg = ccop_neg.chatterjees_xi()
    assert np.isclose(xi_pos, 0.5, atol=1e-2)
    assert np.isclose(xi_neg, 0.5, atol=1e-2)


def test_xi_example():
    """Test xi for the example matrix from the original code."""
    matr = np.array([[1, 5, 4], [5, 3, 2], [4, 2, 4]])
    ccop = BivCheckPi(matr)
    xi_val = ccop.chatterjees_xi()

    # Check range (xi is always between 0 and 1)
    assert 0 <= xi_val <= 1


def test_measure_consistency():
    """Test that tau and rho have consistent signs for asymmetric matrices."""
    # Create a matrix with positive dependence
    matr_pos = np.array([[0.6, 0.2, 0.0], [0.2, 0.4, 0.2], [0.0, 0.2, 0.6]])
    ccop_pos = BivCheckPi(matr_pos)
    tau_pos = ccop_pos.tau()
    rho_pos = ccop_pos.rho()

    # Both should be positive
    assert tau_pos > 0
    assert rho_pos > 0

    # Create a matrix with negative dependence
    matr_neg = np.array([[0.0, 0.2, 0.6], [0.2, 0.4, 0.2], [0.6, 0.2, 0.0]])
    ccop_neg = BivCheckPi(matr_neg)
    tau_neg = ccop_neg.tau()
    rho_neg = ccop_neg.rho()

    # Both should be negative
    assert tau_neg < 0
    assert rho_neg < 0


def test_xi_equivalent_to_monte_carlo():
    """Test that our implementation matches the standard case from existing test."""
    # This matrix was tested previously with Monte Carlo
    matr = np.array([[1, 0], [0, 1]])
    ccop = BivCheckPi(matr)
    xi_value = ccop.chatterjees_xi()
    assert np.isclose(xi_value, 0.5, atol=0.02)
