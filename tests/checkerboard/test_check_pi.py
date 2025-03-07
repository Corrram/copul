import numpy as np
import pytest
import sympy

from copul.checkerboard.check_pi import CheckPi


def test_multivar_checkerboard():
    # 3-dim matrix
    matr = np.full((2, 2, 2), 0.5)
    copula = CheckPi(matr)
    u = (0.5, 0.5, 0.5)
    assert copula.cdf(*u) == 0.125
    assert copula.pdf(*u) == 0.125
    assert copula.cond_distr(1, u) == 0.25
    assert copula.cond_distr(2, u) == 0.25
    assert copula.cond_distr(3, u) == 0.25


def test_2d_check_pi_rvs():
    np.random.seed(1)
    ccop = CheckPi([[1, 2], [2, 1]])
    n = 1_000
    samples = ccop.rvs(n)
    n_lower_empirical = sum([(sample < (0.5, 0.5)).all() for sample in samples])
    n_upper_empirical = sum([(sample > (0.5, 0.5)).all() for sample in samples])
    theoretical_ratio = 1 / 6 * n
    assert n_lower_empirical < 1.5 * theoretical_ratio
    assert n_upper_empirical < 1.5 * theoretical_ratio


def test_3d_check_pi_rvs():
    np.random.seed(1)
    ccop = CheckPi([[[1, 2], [2, 1]], [[1, 2], [2, 1]]])
    n = 1_000
    samples = ccop.rvs(n)
    n_lower_empirical = sum([(sample < (0.5, 0.5, 0.5)).all() for sample in samples])
    n_upper_empirical = sum([(sample > (0.5, 0.5, 0.5)).all() for sample in samples])
    theoretical_ratio = 1 / 12 * n
    assert 0.5 * theoretical_ratio < n_lower_empirical < 1.5 * theoretical_ratio
    assert 0.5 * theoretical_ratio < n_upper_empirical < 1.5 * theoretical_ratio


def test_initialization():
    """Test initialization with different types of matrices."""
    # Test with numpy array
    matr_np = np.array([[1, 2], [3, 4]])
    copula_np = CheckPi(matr_np)
    assert np.isclose(copula_np.matr.sum(), 1.0)

    # Test with list
    matr_list = [[1, 2], [3, 4]]
    copula_list = CheckPi(matr_list)
    assert np.isclose(copula_list.matr.sum(), 1.0)

    # Test with sympy Matrix
    matr_sympy = sympy.Matrix([[1, 2], [3, 4]])
    copula_sympy = CheckPi(matr_sympy)
    assert np.isclose(float(sum(copula_sympy.matr)), 1.0)


def test_cdf_boundary_cases():
    """Test boundary cases for CDF computation."""
    matr = np.array([[1, 2], [3, 4]])
    copula = CheckPi(matr)

    # Test when arguments are out of bounds
    assert copula.cdf(-0.1, 0.5) == 0
    assert copula.cdf(0.5, -0.1) == 0

    # Test when arguments exceed 1
    assert np.isclose(copula.cdf(1.1, 0.5), copula.cdf(1.0, 0.5))
    assert np.isclose(copula.cdf(0.5, 1.1), copula.cdf(0.5, 1.0))

    # Test at corners
    assert np.isclose(copula.cdf(0, 0), 0)
    assert np.isclose(copula.cdf(1, 1), 1)


def test_cdf_interpolation():
    """Test CDF interpolation for non-integer grid points."""
    # Create a simple 2x2 checkerboard with equal weights
    matr = np.array([[1, 1], [1, 1]])
    copula = CheckPi(matr)

    # Test at grid points
    assert np.isclose(copula.cdf(0.5, 0.5), 0.25)

    # Test proper interpolation between grid points
    assert np.isclose(copula.cdf(0.25, 0.25), 0.0625)  # 1/4 of the way in both dims
    assert np.isclose(copula.cdf(0.75, 0.75), 0.5625)  # 3/4 of the way in both dims

    # Test asymmetric interpolation
    assert np.isclose(copula.cdf(0.25, 0.75), 0.1875)  # 1/4 in x, 3/4 in y
    assert np.isclose(copula.cdf(0.75, 0.25), 0.1875)  # 3/4 in x, 1/4 in y


def test_pdf_behavior():
    """Test the behavior of the PDF function."""
    matr = np.array([[0.1, 0.2], [0.3, 0.4]])
    copula = CheckPi(matr)

    # Test at grid points - pdf returns the value at the cell containing the point
    assert np.isclose(copula.pdf(0.25, 0.25), 0.1)
    assert np.isclose(copula.pdf(0.25, 0.75), 0.2)
    assert np.isclose(copula.pdf(0.75, 0.25), 0.3)
    assert np.isclose(copula.pdf(0.75, 0.75), 0.4)

    # Test out of bounds
    assert copula.pdf(-0.1, 0.5) == 0
    assert copula.pdf(0.5, -0.1) == 0
    assert copula.pdf(1.1, 0.5) == 0
    assert copula.pdf(0.5, 1.1) == 0


def test_cond_distr():
    """Test conditional distribution computation."""
    matr = np.array([[0.1, 0.2], [0.3, 0.4]])
    copula = CheckPi(matr)

    # Test at various grid points
    # For an equal weight matrix, conditional distribution should match
    # theoretical expectations
    equal_matr = np.array([[0.25, 0.25], [0.25, 0.25]])
    equal_copula = CheckPi(equal_matr)

    # At (0.5, 0.5) with equal weights, conditional should be 0.5
    u_center = (0.5, 0.5)
    assert np.isclose(equal_copula.cond_distr(1, u_center), 0.5)
    assert np.isclose(equal_copula.cond_distr(2, u_center), 0.5)

    # For the asymmetric matrix, test specific points
    # At (0.25, 0.25), weight in cell is 0.1
    # Row sum is 0.1 + 0.2 = 0.3, column sum is 0.1 + 0.3 = 0.4
    u_lower = (0.25, 0.25)
    assert np.isclose(copula.cond_distr(1, u_lower), 0.1 / 0.4)  # 0.25
    assert np.isclose(copula.cond_distr(2, u_lower), 0.1 / 0.3)  # 0.33...

    # Test convenience methods
    assert np.isclose(copula.cond_distr_1(u_lower), 0.1 / 0.4)
    assert np.isclose(copula.cond_distr_2(u_lower), 0.1 / 0.3)

    # Test invalid dimension
    with pytest.raises(ValueError):
        copula.cond_distr(3, u_lower)  # Should raise error for 2D copula


def test_higher_dimensional():
    """Test operations on higher-dimensional checkerboard copulas."""
    # Create a 3x3x3 checkerboard
    matr = np.ones((3, 3, 3))
    copula = CheckPi(matr)

    # Test dimensions
    assert copula.d == 3
    assert copula.dim == (3, 3, 3)

    # Test CDF at various points
    assert np.isclose(copula.cdf(1 / 3, 1 / 3, 1 / 3), 1 / 27)
    assert np.isclose(copula.cdf(2 / 3, 2 / 3, 2 / 3), 8 / 27)
    assert np.isclose(copula.cdf(1, 1, 1), 1)

    # Test PDF
    assert np.isclose(copula.pdf(1 / 6, 1 / 6, 1 / 6), 1 / 27)
    assert np.isclose(copula.pdf(5 / 6, 5 / 6, 5 / 6), 1 / 27)


def test_rvs_distribution():
    """Test that random samples follow the expected distribution."""
    np.random.seed(42)

    # Create an asymmetric distribution
    matr = np.array([[0.8, 0.1], [0.05, 0.05]])
    copula = CheckPi(matr)

    # Generate samples
    n = 5000
    samples = copula.rvs(n)

    # Count samples in each quadrant
    q1 = sum([(sample < (0.5, 0.5)).all() for sample in samples])
    q2 = sum([(sample[0] < 0.5) & (sample[1] >= 0.5) for sample in samples])
    q3 = sum([(sample[0] >= 0.5) & (sample[1] < 0.5) for sample in samples])
    q4 = sum([(sample >= (0.5, 0.5)).all() for sample in samples])

    # Check proportions (with some tolerance for randomness)
    assert 0.75 * n <= q1 <= 0.85 * n  # Around 80%
    assert 0.07 * n <= q2 <= 0.13 * n  # Around 10%
    assert 0.03 * n <= q3 <= 0.07 * n  # Around 5%
    assert 0.03 * n <= q4 <= 0.07 * n  # Around 5%


def test_weighted_random_selection():
    """Test the weighted random selection method."""
    np.random.seed(42)

    # Create a matrix with very skewed weights
    matr = np.array([[100, 1], [1, 1]])

    # Select elements
    elements, indices = CheckPi._weighted_random_selection(matr, 1000)

    # Most elements should be from the (0,0) position
    count_00 = sum(1 for idx in indices if idx == (0, 0))
    assert count_00 > 900  # Should be around 97%


def test_lambda_functions():
    """Test the tail dependence functions."""
    matr = np.array([[1, 2], [3, 4]])
    copula = CheckPi(matr)

    # Currently these return 0 by default
    assert copula.lambda_L() == 0
    assert copula.lambda_U() == 0


def test_str_representation():
    """Test the string representation."""
    matr = np.array([[1, 2], [3, 4]])
    copula = CheckPi(matr)

    assert str(copula) == "CheckerboardCopula((2, 2))"


def test_is_absolutely_continuous():
    """Test the is_absolutely_continuous property."""
    matr = np.array([[1, 2], [3, 4]])
    copula = CheckPi(matr)

    assert copula.is_absolutely_continuous == True


def test_cdf_consistency():
    """Test that CDF is consistent with mathematical properties."""
    matr = np.array([[0.1, 0.2], [0.3, 0.4]])
    copula = CheckPi(matr)

    # CDF should be monotonically increasing
    assert copula.cdf(0.2, 0.2) <= copula.cdf(0.4, 0.2)
    assert copula.cdf(0.2, 0.2) <= copula.cdf(0.2, 0.4)
    assert copula.cdf(0.2, 0.2) <= copula.cdf(0.4, 0.4)

    # CDF should satisfy rectangle inequality
    # F(b1,b2) - F(a1,b2) - F(b1,a2) + F(a1,a2) >= 0
    a1, a2 = 0.2, 0.3
    b1, b2 = 0.7, 0.8
    rectangle_sum = (
            copula.cdf(b1, b2) -
            copula.cdf(a1, b2) -
            copula.cdf(b1, a2) +
            copula.cdf(a1, a2)
    )
    assert rectangle_sum >= 0

    # Sum of PDF over all cells should equal 1
    # For a 2x2 grid, we can check directly
    total_pdf = (
            copula.pdf(0.25, 0.25) +
            copula.pdf(0.25, 0.75) +
            copula.pdf(0.75, 0.25) +
            copula.pdf(0.75, 0.75)
    )
    assert np.isclose(total_pdf, 1.0)


def test_cdf_exact_values():
    """Test exact CDF values for a specific matrix."""
    # Create a matrix with specific values for testing
    matr = np.array([[0.1, 0.2], [0.3, 0.4]])
    copula = CheckPi(matr)

    # Calculate expected CDF values
    # F(0.5, 0.5) = sum of all values in the lower-left quadrant
    assert np.isclose(copula.cdf(0.5, 0.5), 0.1)

    # F(1.0, 0.5) = sum of the entire first column
    assert np.isclose(copula.cdf(1.0, 0.5), 0.4)

    # F(0.5, 1.0) = sum of the entire first row
    assert np.isclose(copula.cdf(0.5, 1.0), 0.3)

    # F(1.0, 1.0) = sum of all values
    assert np.isclose(copula.cdf(1.0, 1.0), 1.0)

    # Test interpolation at (0.25, 0.75)
    # This point is 1/4 into the first cell horizontally
    # and 3/4 into the first cell vertically
    # With bilinear interpolation:
    # F(0.25, 0.75) = 0.025
    assert np.isclose(copula.cdf(0.25, 0.75), 0.025)
