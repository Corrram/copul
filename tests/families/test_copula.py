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
        # Create a proper mock instance
        self.copula = MagicMock(spec=Copula)
        self.copula.dimension = 2

        # We need to manually add the rvs method from Copula to our mock
        # This is the method we're actually testing
        self.copula.rvs = Copula.rvs.__get__(self.copula)

    def test_inheritance(self):
        """Test that Copula inherits from CoreCopula."""
        assert issubclass(Copula, CoreCopula)

    @patch("copul.families.copula.CopulaSampler")
    def test_rvs_default_parameters(self, mock_sampler_class):
        """Test the rvs method with default parameters."""
        # Create mock sampler
        mock_sampler = MagicMock()
        mock_sampler_class.return_value = mock_sampler

        # Create sample return value
        sample_data = np.random.random((1, 2))
        mock_sampler.rvs.return_value = sample_data

        # Call the method
        result = self.copula.rvs()

        # Verify the calls
        mock_sampler_class.assert_called_once_with(self.copula, random_state=None)
        mock_sampler.rvs.assert_called_once_with(1, False)

        # Verify the result
        assert np.array_equal(result, sample_data)

    @patch("copul.families.copula.CopulaSampler")
    def test_rvs_custom_parameters(self, mock_sampler_class):
        """Test the rvs method with custom parameters."""
        # Create mock sampler
        mock_sampler = MagicMock()
        mock_sampler_class.return_value = mock_sampler

        # Create sample return value
        sample_data = np.random.random((50, 2))
        mock_sampler.rvs.return_value = sample_data

        # Call the method with custom parameters
        n_samples = 50
        random_state = 42
        result = self.copula.rvs(n=n_samples, random_state=random_state)

        # Verify the calls
        mock_sampler_class.assert_called_once_with(
            self.copula, random_state=random_state
        )
        mock_sampler.rvs.assert_called_once_with(n_samples, False)

        # Verify the result
        assert np.array_equal(result, sample_data)

    @patch("copul.families.copula.CopulaSampler")
    def test_rvs_with_approximate(self, mock_sampler_class):
        """Test the rvs method with approximate=True."""
        # Create mock sampler
        mock_sampler = MagicMock()
        mock_sampler_class.return_value = mock_sampler

        # Create sample return value
        sample_data = np.random.random((10, 2))
        mock_sampler.rvs.return_value = sample_data

        # Call the method with approximate=True
        result = self.copula.rvs(n=10, approximate=True)

        # Verify the calls
        mock_sampler_class.assert_called_once_with(self.copula, random_state=None)
        mock_sampler.rvs.assert_called_once_with(10, True)

        # Verify the result
        assert np.array_equal(result, sample_data)

    @patch("copul.families.copula.CopulaSampler")
    def test_rvs_with_random_state(self, mock_sampler_class):
        """Test the rvs method with a specific random_state."""
        # Create mock sampler
        mock_sampler = MagicMock()
        mock_sampler_class.return_value = mock_sampler

        # Create sample return value
        sample_data = np.random.random((5, 2))
        mock_sampler.rvs.return_value = sample_data

        # Call the method with a specific random_state
        result = self.copula.rvs(n=5, random_state=123)

        # Verify the calls
        mock_sampler_class.assert_called_once_with(self.copula, random_state=123)
        mock_sampler.rvs.assert_called_once_with(5, False)

        # Verify the result
        assert np.array_equal(result, sample_data)

    @patch("copul.families.copula.CopulaSampler")
    def test_rvs_error_handling(self, mock_sampler_class):
        """Test error handling in the rvs method."""
        # Create mock sampler
        mock_sampler = MagicMock()
        mock_sampler_class.return_value = mock_sampler

        # Make rvs raise an exception
        mock_sampler.rvs.side_effect = ValueError("Test error")

        # Verify the exception is propagated
        with pytest.raises(ValueError, match="Test error"):
            self.copula.rvs()


@pytest.mark.parametrize(
    "n_samples,random_state,approximate",
    [
        (1, None, False),  # Default case
        (10, 42, False),  # With random_state
        (100, None, True),  # With approximate
        (1000, 123, True),  # With all parameters
    ],
)
def test_rvs_parameter_combinations(n_samples, random_state, approximate):
    """Parametrized test for different parameter combinations in rvs method."""
    # Create mock copula
    mock_copula = MagicMock(spec=Copula)
    mock_copula.dimension = 2

    # Store original method to call later
    original_rvs = Copula.rvs.__get__(mock_copula)

    # Create mock sampler
    with patch("copul.families.copula.CopulaSampler") as mock_sampler_class:
        mock_sampler = MagicMock()
        mock_sampler.rvs.return_value = np.random.random((n_samples, 2))
        mock_sampler_class.return_value = mock_sampler

        # Call the method using the bound method
        original_rvs(n=n_samples, random_state=random_state, approximate=approximate)

        # Verify the calls
        mock_sampler_class.assert_called_once_with(
            mock_copula, random_state=random_state
        )
        mock_sampler.rvs.assert_called_once_with(n_samples, approximate)
