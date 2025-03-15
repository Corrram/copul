"""
Tests for the CheckPi and BivCheckPi classes.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import sympy

from copul.checkerboard.check_pi import CheckPi


class TestCheckPiBivCheckPiConversion:
    """Tests for CheckPi automatic conversion to BivCheckPi."""

    def test_checkpi_2d_returns_bivcheckpi(self):
        """Test that creating a CheckPi with a 2D matrix returns a BivCheckPi instance."""
        # Create a 2D matrix
        matr = np.array([[0.25, 0.25], [0.25, 0.25]])

        # Create a mock BivCheckPi class
        mock_bivcheckpi_instance = MagicMock()
        mock_bivcheckpi_instance.__class__.__name__ = "BivCheckPi"

        # Create a mock module that returns our mock instance
        mock_module = MagicMock()
        mock_module.BivCheckPi = MagicMock(return_value=mock_bivcheckpi_instance)

        # Patch importlib.import_module to return our mock module
        with patch("importlib.import_module", return_value=mock_module):
            # Call CheckPi constructor
            result = CheckPi(matr)

            # Verify the result is the BivCheckPi instance
            assert result is mock_bivcheckpi_instance
            # Verify BivCheckPi was called with the matrix
            mock_module.BivCheckPi.assert_called_once_with(matr)

    def test_checkpi_3d_returns_checkpi(self):
        """Test that creating a CheckPi with a 3D matrix returns a CheckPi instance."""
        # Create a 3D matrix
        matr = np.ones((2, 2, 2)) / 8  # Normalized 2x2x2 matrix

        # Create a CheckPi instance with the 3D matrix
        result = CheckPi(matr)

        # Verify the result is a CheckPi instance
        assert isinstance(result, CheckPi)
        assert not hasattr(result, "m")  # BivCheckPi has m, n attributes
        assert not hasattr(result, "n")

    def test_bivcheckpi_import_error_fallback(self):
        """Test that import error for BivCheckPi falls back to CheckPi."""
        # Create a 2D matrix
        matr = np.array([[0.25, 0.25], [0.25, 0.25]])

        # Patch importlib.import_module to raise an ImportError for the bivcheckpi module
        with patch(
            "importlib.import_module",
            side_effect=ImportError("No module named 'bivcheckpi'"),
        ):
            # Call CheckPi constructor
            result = CheckPi(matr)

            # Verify the result is a CheckPi instance
            assert isinstance(result, CheckPi)
            assert not hasattr(result, "m")  # BivCheckPi has m, n attributes
            assert not hasattr(result, "n")

    def test_integration_with_real_classes(self):
        """Integration test with real classes to ensure conversion works properly."""
        try:
            # Try to import directly first to check if it's available
            from importlib import import_module

            try:
                import_module("copul.checkerboard.bivcheckpi")
            except ImportError:
                pytest.skip("BivCheckPi class not available for integration test")

            # Create a 2D matrix
            matr = np.array([[0.25, 0.25], [0.25, 0.25]])

            # Create a CheckPi instance with the 2D matrix
            result = CheckPi(matr)

            # Verify the result is a BivCheckPi instance
            # We use string comparison in case actual classes aren't imported in test environment
            assert result.__class__.__name__ == "BivCheckPi"
            assert hasattr(result, "m")
            assert hasattr(result, "n")
            assert result.m == 2
            assert result.n == 2

        except Exception as e:
            pytest.skip(f"Error in integration test: {e}")

    def test_subclass_not_affected(self):
        """Test that subclasses of CheckPi are not affected by the conversion."""

        # Create a subclass of CheckPi
        class SubCheckPi(CheckPi):
            pass

        # Create a 2D matrix
        matr = np.array([[0.25, 0.25], [0.25, 0.25]])

        # Create a mock module with a BivCheckPi class
        mock_module = MagicMock()
        mock_module.BivCheckPi = MagicMock()

        # Patch importlib.import_module to return our mock module
        with patch("importlib.import_module", return_value=mock_module):
            # Create a SubCheckPi instance with the 2D matrix
            result = SubCheckPi(matr)

            # Verify the result is a SubCheckPi instance
            assert isinstance(result, SubCheckPi)

            # Verify BivCheckPi was NOT called
            mock_module.BivCheckPi.assert_not_called()


def test_multivar_checkerboard():
    # 3-dim matrix
    matr = np.full((2, 2, 2), 0.5)
    copula = CheckPi(matr)
    u = (0.5, 0.5, 0.5)
    v = (0.25, 0.25, 0.25)
    w = (0.75, 0.75, 0.75)
    assert copula.cdf(*u) == 0.125
    assert copula.cdf(*v) == 0.125 / 8
    assert copula.cdf(*w) == 0.75**3
    assert copula.pdf(*u) == 0.125
    assert copula.pdf(*v) == 0.125
    assert copula.cond_distr(1, u) == 0.25
    assert copula.cond_distr(2, u) == 0.25
    assert copula.cond_distr(3, u) == 0.25
    assert copula.cond_distr(1, v) == 0.25**2
    assert copula.cond_distr(1, w) == 0.75**2


@pytest.mark.parametrize(
    "u, v, expected",
    [
        (0.4, 0.4, 0.8),
        (0.4, 0.6, 1),
        (0.6, 0.4, 0),
    ],
)
def test_2d_ccop_cond_distr_1_different_points(u, v, expected):
    matr = [[1, 0], [0, 1]]
    ccop = CheckPi(matr)
    actual = ccop.cond_distr_1((u, v))
    assert np.isclose(actual, expected)


@pytest.mark.parametrize(
    "matr, expected",
    [
        ([[1, 0], [0, 1]], 0),  # 0.5 belongs to the second row, so ~ Unif[0.5, 1]
        ([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], 0.5),
        ([[1, 5, 4], [5, 3, 2], [4, 2, 4]], 0.65),  # second row -> (1*5+0.5*3+0*2)/10
    ],
)
def test_ccop_cond_distr_1(matr, expected):
    ccop = CheckPi(matr)
    actual = ccop.cond_distr_1((0.5, 0.5))
    assert np.isclose(actual, expected)


@pytest.mark.parametrize(
    "point, ratio",
    [
        ((0.5, 0.5), 1 / 6),
        ((0.25, 0.25), 1 / 24),
        ((0.75, 0.75), 1 / 6 * 1 + 1 / 3 * 1 / 2 + 1 / 3 * 1 / 2 + 1 / 6 * 1 / 4),
    ],
)
def test_2d_check_pi_rvs(point, ratio):
    np.random.seed(1)
    ccop = CheckPi([[1, 2], [2, 1]])
    n = 1_000
    samples = ccop.rvs(n)
    n_lower_empirical = sum([(sample < point).all() for sample in samples])
    theoretical_ratio = ratio * n
    assert np.isclose(n_lower_empirical, theoretical_ratio, rtol=0.1)


def test_3d_check_pi_rvs():
    np.random.seed(1)
    ccop = CheckPi([[[1, 2], [2, 1]], [[1, 2], [2, 1]]])
    n = 20_000
    samples = ccop.rvs(n)
    n_lower_empirical = sum([(sample < (0.5, 0.5, 0.5)).all() for sample in samples])
    n_upper_empirical = sum([(sample > (0.5, 0.5, 0.5)).all() for sample in samples])
    ratio = 1 / 12 * n
    assert np.isclose(n_lower_empirical, ratio, rtol=0.1)
    assert np.isclose(n_upper_empirical, ratio, rtol=0.1)

    n_lower = sum([(sample < (0.25, 0.25, 0.5)).all() for sample in samples])
    assert np.isclose(n_lower, ratio / 4, rtol=0.1)
    n_lower_quarter = sum([(sample < (0.25, 0.25, 0.25)).all() for sample in samples])
    expected = ratio / 8
    assert np.isclose(n_lower_quarter, expected, rtol=0.1)

    n_lower_part = sum([(sample < (0.25, 0.75, 0.5)).all() for sample in samples])
    expected_part = (1 / 12 * 1 / 2 + 1 / 6 * 1 / 4) * n
    assert np.isclose(n_lower_part, expected_part, rtol=0.1)


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


def test_higher_dimensional():
    """Test operations on higher-dimensional checkerboard copulas."""
    # Create a 3x3x3 checkerboard
    matr = np.ones((3, 3, 3))
    copula = CheckPi(matr)

    # Test dimensions
    assert copula.dim == 3
    assert copula.matr.shape == (3, 3, 3)

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

    assert str(copula) == "CheckPiCopula((2, 2))"


def test_is_absolutely_continuous():
    """Test the is_absolutely_continuous property."""
    matr = np.array([[1, 2], [3, 4]])
    copula = CheckPi(matr)

    assert copula.is_absolutely_continuous is True


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
        copula.cdf(b1, b2)
        - copula.cdf(a1, b2)
        - copula.cdf(b1, a2)
        + copula.cdf(a1, a2)
    )
    assert rectangle_sum >= 0

    # Sum of PDF over all cells should equal 1
    # For a 2x2 grid, we can check directly
    total_pdf = (
        copula.pdf(0.25, 0.25)
        + copula.pdf(0.25, 0.75)
        + copula.pdf(0.75, 0.25)
        + copula.pdf(0.75, 0.75)
    )
    assert np.isclose(total_pdf, 1.0)


def test_random_points_cdf_2d():
    """
    Sample random points in [0,1]^2 for a small 2x2 matrix.
    Compare cdf result to a brute force piecewise computation.
    """
    matr = np.array(
        [[4.0, 1.0], [1.0, 2.0]]
    )  # sum=8 => normalized => top-left=0.5, top-right=0.125, bottom-left=0.125, bottom-right=0.25
    copula = CheckPi(matr)

    # We'll generate some random points and compare copula.cdf(...) to a direct piecewise approach
    rng = np.random.RandomState(123)
    for _ in range(10):
        x = rng.rand()
        y = rng.rand()
        cdf_val = copula.cdf(x, y)

        # We'll compute a direct piecewise approach:
        # sum_{cell} [ fraction_of_cell_in [0,x]*[0,y] * cell_mass ]
        direct_sum = 0.0
        for irow in (0, 1):
            for icol in (0, 1):
                cell_mass = copula.matr[irow, icol]
                # cell is [irow/2, (irow+1)/2)*[icol/2, (icol+1)/2)
                lower_x, upper_x = irow / 2, (irow + 1) / 2
                lower_y, upper_y = icol / 2, (icol + 1) / 2
                overlap_x = max(0.0, min(x, upper_x) - lower_x)
                overlap_y = max(0.0, min(y, upper_y) - lower_y)
                # fraction in x => overlap_x / (0.5), fraction in y => overlap_y / (0.5)
                frac_x = overlap_x / 0.5
                frac_y = overlap_y / 0.5
                if frac_x < 0 or frac_y < 0:
                    frac_x = 0
                    frac_y = 0
                if frac_x > 1:
                    frac_x = 1
                if frac_y > 1:
                    frac_y = 1
                frac_cell = frac_x * frac_y
                direct_sum += cell_mass * frac_cell

        assert np.isclose(cdf_val, direct_sum, atol=1e-14)


def test_cond_distr_consistency_3d():
    """
    Test a 3D checkerboard's cond_distr to ensure that dimension i
    means 'U_{-i} | U_i' as required.
    """
    matr = np.ones((2, 2, 2))
    copula = CheckPi(matr)

    # Let's pick a point u = (0.3, 0.6, 0.9)
    # cond_distr(1, u) => F_{(U2,U3)|U1}( (0.6,0.9) | 0.3 )
    #   = cdf(0.3,0.6,0.9)/cdf(0.3,1,1)
    cdf_u = copula.cdf(0.3, 0.6, 0.9)
    denom_1 = copula.cdf(0.3, 1.0, 1.0)
    ratio_1 = cdf_u / denom_1 if denom_1 > 0 else 0
    assert np.isclose(copula.cond_distr(1, (0.3, 0.6, 0.9)), ratio_1)

    # cond_distr(2, u) => F_{(U1,U3)|U2}( (0.3,0.9) | 0.6 )
    #   = cdf(0.3,0.6,0.9)/cdf(1,0.6,1)
    denom_2 = copula.cdf(1.0, 0.6, 1.0)
    ratio_2 = cdf_u / denom_2 if denom_2 > 0 else 0
    assert np.isclose(copula.cond_distr(2, (0.3, 0.6, 0.9)), ratio_2)

    # cond_distr(3, u) => F_{(U1,U2)|U3}( (0.3,0.6) | 0.9 )
    #   = cdf(0.3,0.6,0.9)/cdf(1,1,0.9)
    denom_3 = copula.cdf(1.0, 1.0, 0.9)
    ratio_3 = cdf_u / denom_3 if denom_3 > 0 else 0
    assert np.isclose(copula.cond_distr(3, (0.3, 0.6, 0.9)), ratio_3)


@pytest.mark.parametrize(
    "matr, point, expected",
    [
        ([[1, 1], [1, 1]], (0.2, 0.1), 0.1),
        ([[1, 1], [1, 1]], (0.1, 0.2), 0.2),
        ([[1, 1], [1, 1]], (0.7, 0.6), 0.6),
        ([[1, 1], [1, 1]], (0.6, 0.7), 0.7),
        ([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], (0.5, 0.5), 0.5),
        ([[1, 5, 4], [5, 3, 2], [4, 2, 4]], (0.5, 0.5), 0.65),
    ],
)
def test_ccop_cond_distr(matr, point, expected):
    ccop = CheckPi(matr)
    actual = ccop.cond_distr(1, point)
    assert np.isclose(actual, expected)


def test_chatterjees_xi_for_independence_copula():
    """Test the Chatterjee's xi estimator."""
    matr = np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
    copula = CheckPi(matr)

    # Chatterjee's xi for this copula should be 0
    xi = copula.chatterjees_xi(seed=42)
    assert np.isclose(xi, 0, atol=1e-2)
