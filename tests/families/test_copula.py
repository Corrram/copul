"""
Tests for the Copula class.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from copul.families.core_copula import CoreCopula
from copul.families.copula import Copula


class TestCopula:
    """Tests for the Copula class."""

    def setup_method(self):
        """Setup test fixtures."""
        # Instead of patching inheritance, create a proper mock instance
        # We'll patch specific methods that are needed
        self.copula = MagicMock(spec=Copula)
        self.copula.dimension = 2

        # We need to manually add the rvs method from Copula to our mock
        # This is the method we're actually testing
        self.copula.rvs = Copula.rvs.__get__(self.copula)

    def test_inheritance(self):
        """Test that Copula inherits from CoreCopula."""
        assert issubclass(Copula, CoreCopula)

    @patch("copul.families.copula.Checkerboarder")
    def test_rvs_default_parameters(self, mock_checkerboarder_class):
        """Test the rvs method with default parameters."""
        # Create mock checkerboarder
        mock_checkerboarder = MagicMock()
        mock_checkerboarder_class.return_value = mock_checkerboarder

        # Create mock for the checkerboard copula
        mock_check_pi = MagicMock()
        mock_checkerboarder.compute_check_pi.return_value = mock_check_pi

        # Create sample return value
        sample_data = np.random.random((1, 2))
        mock_check_pi.rvs.return_value = sample_data

        # Call the method
        result = self.copula.rvs()

        # Verify the calls
        mock_checkerboarder_class.assert_called_once_with(100, dim=2)
        mock_checkerboarder.compute_check_pi.assert_called_once_with(self.copula)
        mock_check_pi.rvs.assert_called_once_with(1)

        # Verify the result
        assert np.array_equal(result, sample_data)

    @patch("copul.families.copula.Checkerboarder")
    def test_rvs_custom_parameters(self, mock_checkerboarder_class):
        """Test the rvs method with custom parameters."""
        # Create mock checkerboarder
        mock_checkerboarder = MagicMock()
        mock_checkerboarder_class.return_value = mock_checkerboarder

        # Create mock for the checkerboard copula
        mock_check_pi = MagicMock()
        mock_checkerboarder.compute_check_pi.return_value = mock_check_pi

        # Create sample return value
        sample_data = np.random.random((50, 2))
        mock_check_pi.rvs.return_value = sample_data

        # Call the method with custom parameters
        n_samples = 50
        precision = 3
        result = self.copula.rvs(n=n_samples, precision=precision)

        # Verify the calls
        mock_checkerboarder_class.assert_called_once_with(1000, dim=2)  # 10^3 = 1000
        mock_checkerboarder.compute_check_pi.assert_called_once_with(self.copula)
        mock_check_pi.rvs.assert_called_once_with(n_samples)

        # Verify the result
        assert np.array_equal(result, sample_data)

    @patch("copul.families.copula.Checkerboarder")
    def test_rvs_high_precision(self, mock_checkerboarder_class):
        """Test the rvs method with high precision."""
        # Create mock checkerboarder
        mock_checkerboarder = MagicMock()
        mock_checkerboarder_class.return_value = mock_checkerboarder

        # Create mock for the checkerboard copula
        mock_check_pi = MagicMock()
        mock_checkerboarder.compute_check_pi.return_value = mock_check_pi

        # Create sample return value
        sample_data = np.random.random((10, 2))
        mock_check_pi.rvs.return_value = sample_data

        # Call the method with high precision
        result = self.copula.rvs(n=10, precision=4)

        # Verify the calls
        mock_checkerboarder_class.assert_called_once_with(10000, dim=2)  # 10^4 = 10000
        mock_checkerboarder.compute_check_pi.assert_called_once_with(self.copula)
        mock_check_pi.rvs.assert_called_once_with(10)

        # Verify the result
        assert np.array_equal(result, sample_data)

    @patch("copul.families.copula.Checkerboarder")
    def test_rvs_higher_dimension(self, mock_checkerboarder_class):
        """Test the rvs method with higher dimension."""
        # Set higher dimension
        self.copula.dimension = 3

        # Create mock checkerboarder
        mock_checkerboarder = MagicMock()
        mock_checkerboarder_class.return_value = mock_checkerboarder

        # Create mock for the checkerboard copula
        mock_check_pi = MagicMock()
        mock_checkerboarder.compute_check_pi.return_value = mock_check_pi

        # Create sample return value
        sample_data = np.random.random((5, 3))
        mock_check_pi.rvs.return_value = sample_data

        # Call the method
        result = self.copula.rvs(n=5)

        # Verify the calls
        mock_checkerboarder_class.assert_called_once_with(100, dim=3)
        mock_checkerboarder.compute_check_pi.assert_called_once_with(self.copula)
        mock_check_pi.rvs.assert_called_once_with(5)

        # Verify the result
        assert np.array_equal(result, sample_data)

    @patch("copul.families.copula.Checkerboarder")
    def test_rvs_error_handling(self, mock_checkerboarder_class):
        """Test error handling in the rvs method."""
        # Create mock checkerboarder
        mock_checkerboarder = MagicMock()
        mock_checkerboarder_class.return_value = mock_checkerboarder

        # Make compute_check_pi raise an exception
        mock_checkerboarder.compute_check_pi.side_effect = ValueError("Test error")

        # Verify the exception is propagated
        with pytest.raises(ValueError, match="Test error"):
            self.copula.rvs()


@pytest.mark.parametrize(
    "n_samples,precision,expected_grid_size",
    [
        (1, 2, 100),  # Default case
        (10, 3, 1000),  # Higher precision
        (100, 1, 10),  # Lower precision
        (1000, 4, 10000),  # High samples and precision
    ],
)
def test_rvs_grid_size(n_samples, precision, expected_grid_size):
    """Parametrized test for different grid sizes in rvs method."""
    # Create mock copula
    mock_copula = MagicMock(spec=Copula)
    mock_copula.dimension = 2

    # Store original method to call later
    original_rvs = Copula.rvs.__get__(mock_copula)

    # Create mock checkerboarder
    with patch("copul.families.copula.Checkerboarder") as mock_checkerboarder_class:
        mock_checkerboarder = MagicMock()
        mock_checkerboarder_class.return_value = mock_checkerboarder

        # Create mock for the checkerboard copula
        mock_check_pi = MagicMock()
        mock_check_pi.rvs.return_value = np.random.random((n_samples, 2))
        mock_checkerboarder.compute_check_pi.return_value = mock_check_pi

        # Call the method using the bound method
        original_rvs(n=n_samples, precision=precision)

        # Verify the grid size is calculated correctly
        mock_checkerboarder_class.assert_called_once_with(expected_grid_size, dim=2)

        # Verify the other calls
        mock_checkerboarder.compute_check_pi.assert_called_once_with(mock_copula)
        mock_check_pi.rvs.assert_called_once_with(n_samples)


def test_integration_with_real_copula():
    """Integration test with a real CoreCopula implementation."""
    # Try to import a real copula implementation for integration testing
    from copul.families.elliptical.gaussian import Gaussian

    # Create a Gaussian copula with rho=0.5
    gaussian = Gaussian(rho=0.5)

    # Instead of creating a combined class, mock the Copula and use its rvs method
    mock_copula = MagicMock(spec=Copula)
    mock_copula.dimension = 2

    # Bind the real rvs method to our mock
    mock_copula.rvs = Copula.rvs.__get__(mock_copula)

    # Create a patch to return our gaussian when compute_check_pi is called
    with patch(
        "copul.checkerboard.checkerboarder.Checkerboarder.compute_check_pi",
        return_value=gaussian,
    ):
        # Generate samples
        samples = mock_copula.rvs(n=10, precision=1)

        # Verify basic properties
        assert samples.shape == (10, 2)
        assert np.all(samples >= 0) and np.all(samples <= 1)
