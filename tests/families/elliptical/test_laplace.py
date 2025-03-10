import numpy as np
import pytest
import sympy
from unittest.mock import patch

from copul.families.elliptical.laplace import Laplace, multivariate_laplace
from copul.families.other import LowerFrechet, UpperFrechet


@pytest.fixture
def laplace_copula():
    """Create a Laplace copula with rho=0.5 for testing."""
    copula = Laplace()
    copula.rho = 0.5
    return copula


@pytest.fixture
def laplace_family():
    """Create a symbolic Laplace copula family for testing."""
    return Laplace()


def test_laplace_init():
    """Test initialization of Laplace copula."""
    # Default initialization with symbol
    copula = Laplace()
    assert hasattr(copula, "rho")
    assert isinstance(copula.rho, sympy.Symbol)
    assert str(copula.rho) == "rho"

    # Initialization with parameter
    copula = Laplace()(0.5)
    assert copula.rho == 0.5


def test_laplace_parameter_bounds():
    """Test parameter bounds for Laplace copula."""
    copula = Laplace()

    # rho should be in [-1, 1]
    rho_interval = copula.intervals["rho"]
    assert rho_interval.left == -1
    assert rho_interval.right == 1
    assert not rho_interval.left_open  # Left bound is closed
    assert not rho_interval.right_open  # Right bound is closed


@pytest.mark.parametrize(
    "rho, expected_class",
    [(-1, LowerFrechet), (1, UpperFrechet)],
)
def test_laplace_special_cases(rho, expected_class):
    """Test special cases for rho = -1 and rho = 1."""
    copula = Laplace()
    result = copula(rho)
    assert isinstance(result, expected_class)


def test_laplace_properties(laplace_copula):
    """Test basic properties of Laplace copula."""
    # Test symmetry property
    assert laplace_copula.is_symmetric is True

    # Test absolute continuity property
    assert laplace_copula.is_absolutely_continuous is True


def test_laplace_rvs(laplace_copula):
    """Test random sampling from the Laplace copula."""
    # Mock the multivariate_laplace.rvs method
    mock_samples = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    with patch.object(
        multivariate_laplace, "rvs", return_value=mock_samples
    ) as mock_rvs:
        # Mock the stats.laplace.cdf method to handle vectorized calls
        # We need to use a class with a __call__ method to handle arrays
        class VectorizedMock:
            def __init__(self):
                self.call_count = 0

            def __call__(self, x):
                self.call_count += 1
                # Add 0.1 to each element in the array
                return x + 0.1

        vectorized_mock = VectorizedMock()

        with patch("scipy.stats.laplace.cdf", vectorized_mock):
            # Generate samples
            samples = laplace_copula.rvs(3)

            # Check that multivariate_laplace.rvs was called with correct parameters
            mock_rvs.assert_called_once()
            call_args = mock_rvs.call_args[1]
            assert np.array_equal(call_args["mean"], [0, 0])
            assert np.array_equal(call_args["cov"], [[1, 0.5], [0.5, 1]])
            assert call_args["size"] == 3

            # Check that the transform to uniform was called twice (once per column)
            assert vectorized_mock.call_count == 2

            # Check the shape and range of the output
            assert samples.shape == (3, 2)
            assert np.all(samples >= 0) and np.all(samples <= 1)

            # Expected result based on our mocked transformations
            expected = np.array(
                [[0.1 + 0.1, 0.2 + 0.1], [0.3 + 0.1, 0.4 + 0.1], [0.5 + 0.1, 0.6 + 0.1]]
            )
            assert np.array_equal(samples, expected)


def test_laplace_corr_matrix(laplace_copula):
    """Test the correlation matrix property."""
    # Get the correlation matrix
    corr_matrix = laplace_copula.corr_matrix

    # Check the structure
    assert isinstance(corr_matrix, sympy.Matrix)
    assert corr_matrix.shape == (2, 2)

    # Check the values
    assert corr_matrix[0, 0] == 1
    assert corr_matrix[1, 1] == 1
    assert corr_matrix[0, 1] == 0.5
    assert corr_matrix[1, 0] == 0.5


def test_laplace_cdf_not_implemented():
    """Test that cdf raises NotImplementedError."""
    copula = Laplace()(0.5)

    with pytest.raises(NotImplementedError) as excinfo:
        copula.cdf(0.3, 0.7)

    assert "not implemented" in str(excinfo.value).lower()


def test_laplace_pdf_not_implemented():
    """Test that pdf raises NotImplementedError."""
    copula = Laplace()(0.5)

    with pytest.raises(NotImplementedError) as excinfo:
        copula.pdf(0.3, 0.7)

    assert "not implemented" in str(excinfo.value).lower()


def test_laplace_call_method():
    """Test the __call__ method for creating new instances."""
    copula = Laplace()

    # Test with rho parameter
    result = copula(0.5)
    assert isinstance(result, Laplace)
    assert result.rho == 0.5

    # Test with keyword argument
    result = copula(rho=0.7)
    assert isinstance(result, Laplace)
    assert result.rho == 0.7


def test_laplace_edge_cases():
    """Test edge cases for Laplace copula parameters."""
    copula = Laplace()

    # Test with rho close to boundaries
    near_minus_one = copula(-0.999)
    assert near_minus_one.rho == -0.999

    near_one = copula(0.999)
    assert near_one.rho == 0.999


def test_laplace_characteristic_function_call():
    """Test that the characteristic_function method is inherited and works."""
    # This test verifies the behavior with a non-implemented generator
    copula = Laplace()(0.5)

    # Laplace doesn't define a generator, so this should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        copula.characteristic_function(1, 2)
