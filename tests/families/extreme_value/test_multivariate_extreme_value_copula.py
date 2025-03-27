import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import sympy as sp

from copul.families.extreme_value.multivariate_extreme_value_copula import (
    MultivariateExtremeValueCopula,
)


# Create a concrete subclass for testing
class SampleMultivariateEVCopula(MultivariateExtremeValueCopula):
    """Concrete implementation of MultivariateExtremeValueCopula for testing"""

    # Define parameters
    theta = sp.symbols("theta", positive=True)
    params = [theta]
    intervals = {str(theta): sp.Interval(1, float("inf"), False, True)}

    def __init__(self, dimension=2, theta=2.0):
        super().__init__(dimension, theta=theta)
        self.theta = theta
        self._free_symbols = {"theta": self.theta}

    def _compute_extreme_value_function(self, u_values):
        """Simple implementation for testing"""
        # Handle boundary cases
        if any(u <= 0 for u in u_values):
            return 0
        if all(u == 1 for u in u_values):
            return 1

        # Compute a simple extreme value function
        log_sum = sum((-np.log(u)) ** self.theta for u in u_values)
        return np.exp(-((log_sum) ** (1 / self.theta)))

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def is_symmetric(self) -> bool:
        return True


class TestMultivariateExtremeValueCopula:
    """Test class for MultivariateExtremeValueCopula"""

    @pytest.fixture
    def copula(self):
        """Create a test copula instance"""
        return SampleMultivariateEVCopula(dimension=3, theta=2.0)

    def test_init(self):
        """Test initialization"""
        copula = SampleMultivariateEVCopula(dimension=4, theta=2.5)
        assert copula.dim == 4
        assert copula.theta == 2.5
        assert len(copula.u_symbols) == 4

    def test_cdf(self, copula):
        """Test CDF computation"""
        # Test boundary cases
        assert float(copula.cdf(0, 0.5, 0.8)) == 0
        assert float(copula.cdf(0.5, 0, 0.8)) == 0
        assert float(copula.cdf(0.5, 0.8, 0)) == 0
        assert float(copula.cdf(1, 1, 1)) == 1

        # Test interior points
        cdf_value = float(copula.cdf(0.3, 0.4, 0.5))
        assert 0 < cdf_value < 1

        # CDF should be less than or equal to min of arguments (Fréchet-Hoeffding upper bound)
        assert cdf_value <= min(0.3, 0.4, 0.5)

    def test_cdf_vectorized(self, copula):
        """Test vectorized CDF computation"""
        # Create test data
        u = np.array([0, 0.3, 0.7, 1])
        v = np.array([0, 0.4, 0.6, 1])
        w = np.array([0, 0.5, 0.8, 1])

        # Compute vectorized CDF
        cdf_values = copula.cdf_vectorized(u, v, w)

        # Check shape
        assert cdf_values.shape == u.shape

        # Check boundary values
        assert cdf_values[0] == 0  # (0, 0, 0)
        assert cdf_values[-1] == 1  # (1, 1, 1)

        # Check interior points
        for i in range(1, len(u) - 1):
            manual_cdf = float(copula.cdf(u[i], v[i], w[i]))
            assert abs(cdf_values[i] - manual_cdf) < 1e-10

    def test_sample_parameters(self, copula):
        """Test parameter sampling"""
        samples = copula.sample_parameters(n=5)

        # Check that we get the expected parameters
        assert "theta" in samples

        # Check that we get the expected number of samples
        assert len(samples["theta"]) == 5

        # Check that samples are within bounds
        for theta in samples["theta"]:
            assert 1 <= theta <= 10  # Upper bound is min(10, inf)

    def test_rvs(self, copula):
        """Test random sample generation"""
        with patch("copul.copula_sampler.CopulaSampler") as mock_sampler_class:
            # Setup mock
            mock_sampler = MagicMock()
            mock_sampler_class.return_value = mock_sampler
            mock_sampler.rvs.return_value = np.random.random((100, 3))

            # Call rvs
            samples = copula.rvs(100, approximate=True)

            # Check result shape
            assert samples.shape == (100, 3)

    def test_abstract_methods(self):
        """Test that abstract methods need to be implemented"""

        # Create incomplete subclass
        class IncompleteEVCopula(MultivariateExtremeValueCopula):
            def __init__(self):
                super().__init__(dimension=2)

        # Should be able to instantiate
        incomplete = IncompleteEVCopula()

        # But calling abstract methods should fail
        with pytest.raises(NotImplementedError):
            _ = incomplete.is_absolutely_continuous

        with pytest.raises(NotImplementedError):
            _ = incomplete.is_symmetric

        with pytest.raises(NotImplementedError):
            _ = incomplete._compute_extreme_value_function([0.5, 0.5])

    def test_different_dimensions(self):
        """Test copulas with different dimensions"""
        # Test various dimensions
        for dim in [2, 3]:
            copula = SampleMultivariateEVCopula(dimension=dim)
            assert copula.dim == dim
            assert len(copula.u_symbols) == dim

            # Create arguments
            args = [0.5] * dim

            # Check CDF computation
            cdf_value = float(copula.cdf(*args))
            assert 0 < cdf_value < 1

            # Check random sampling
            with patch(
                "copul.families.core.copula_sampling_mixin.CopulaSampler"
            ) as mock_sampler_class:
                mock_sampler = MagicMock()
                mock_sampler_class.return_value = mock_sampler
                mock_sampler.rvs.return_value = np.random.random((10, dim))

                samples = copula.rvs(10)
                assert samples.shape == (10, dim)


def test_compute_extreme_value_function():
    """Test the _compute_extreme_value_function method"""
    # Create test copula
    copula = SampleMultivariateEVCopula(dimension=3, theta=2.0)

    # Test boundary cases
    assert copula._compute_extreme_value_function([0, 0.5, 0.8]) == 0
    assert copula._compute_extreme_value_function([1, 1, 1]) == 1

    # Test interior point
    ev_value = copula._compute_extreme_value_function([0.3, 0.4, 0.5])
    assert 0 < ev_value < 1

    # Value should be less than min of arguments
    assert ev_value <= min(0.3, 0.4, 0.5)


def test_context_manager():
    """Test the suppress_warnings context manager"""
    import warnings

    # Create test copula
    copula = SampleMultivariateEVCopula()

    # Test that warnings are suppressed
    with copula.suppress_warnings():
        # This should not show a warning
        warnings.warn("This warning should be suppressed")
