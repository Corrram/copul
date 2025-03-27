import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from copul.families.extreme_value.multivariate_gumbel_hougaard import (
    MultivariateGumbelHougaard,
)
from copul.families.other import BivIndependenceCopula
from copul.families.other.independence_copula import IndependenceCopula


class TestMultivariateGumbelHougaard:
    """Test class for MultivariateGumbelHougaard"""

    @pytest.fixture
    def copula(self):
        """Create a test copula instance"""
        return MultivariateGumbelHougaard(dimension=3, theta=2.0)

    def test_init(self):
        """Test initialization"""
        copula = MultivariateGumbelHougaard(dimension=4, theta=2.5)
        assert copula.dim == 4
        assert copula.theta == 2.5
        assert len(copula.u_symbols) == 4

    def test_independence_case(self):
        """Test special case when theta=1 (should return IndependenceCopula)"""
        # Test with positional arg
        copula1 = MultivariateGumbelHougaard(dimension=3, theta=1)
        assert isinstance(copula1, IndependenceCopula)
        assert copula1.dim == 3

        # Test with keyword arg
        copula2 = MultivariateGumbelHougaard(dimension=2, theta=1)
        assert isinstance(copula2, BivIndependenceCopula)
        assert copula2.dim == 2

    def test_call_method(self, copula):
        """Test __call__ method"""
        # Test normal call
        new_copula = copula(theta=3.0)
        assert new_copula.theta == 3.0
        assert new_copula.dim == copula.dim
        assert new_copula is not copula

        # Test call that should return IndependenceCopula
        indep_copula = copula(theta=1)
        assert isinstance(indep_copula, IndependenceCopula)
        assert indep_copula.dim == copula.dim

    def test_is_absolutely_continuous(self, copula):
        """Test is_absolutely_continuous property"""
        assert copula.is_absolutely_continuous is True

    def test_is_symmetric(self, copula):
        """Test is_symmetric property"""
        assert copula.is_symmetric is True

    def test_compute_extreme_value_function(self, copula):
        """Test _compute_extreme_value_function method"""
        # Test boundary cases
        assert copula._compute_extreme_value_function([0, 0.5, 0.8]) == 0
        assert copula._compute_extreme_value_function([1, 1, 1]) == 1

        # Test interior point
        ev_value = copula._compute_extreme_value_function([0.3, 0.4, 0.5])
        assert 0 < ev_value < 1

        # Value should be less than min of arguments
        assert ev_value <= min(0.3, 0.4, 0.5)

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

        # CDF should be less than or equal to min of arguments
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
        assert np.isclose(cdf_values[0], 0)  # (0, 0, 0)
        assert np.isclose(cdf_values[-1], 1)  # (1, 1, 1)

        # Check interior points
        for i in range(1, len(u) - 1):
            manual_cdf = float(copula.cdf(u[i], v[i], w[i]))
            assert abs(cdf_values[i] - manual_cdf) < 1e-10

    def test_tau(self, copula):
        """Test Kendall's tau computation"""
        # For theta=2, tau should be (2-1)/2 = 0.5
        tau = copula.tau()
        assert float(tau) == 0.5

        # Test with different theta values
        thetas = [1.5, 3, 5, 10]
        for theta in thetas:
            # Create new copula with this theta
            test_copula = MultivariateGumbelHougaard(dimension=3, theta=theta)

            # Compute tau
            tau = test_copula.tau()

            # Check against expected formula: (theta-1)/theta
            expected_tau = (theta - 1) / theta
            assert abs(float(tau) - expected_tau) < 1e-10

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

    def test_different_dimensions(self):
        """Test Gumbel-Hougaard copulas with different dimensions"""
        # Test various dimensions
        for dim in [2, 3, 5]:
            copula = MultivariateGumbelHougaard(dimension=dim, theta=2.0)
            assert copula.dim == dim
            assert len(copula.u_symbols) == dim

            # Create arguments
            args = [0.5] * dim

            # Check CDF computation
            cdf_value = float(copula.cdf(*args))
            assert 0 < cdf_value < 1

            # Check random sampling
            with patch("copul.copula_sampler.CopulaSampler") as mock_sampler_class:
                mock_sampler = MagicMock()
                mock_sampler_class.return_value = mock_sampler
                mock_sampler.rvs.return_value = np.random.random((10, dim))

                samples = copula.rvs(10, approximate=True)
                assert samples.shape == (10, dim)


@pytest.mark.parametrize(
    "theta,expected_tau",
    [
        (1.0, 0.0),  # Independence case
        (1.5, 1 / 3),  # (1.5-1)/1.5 = 0.333...
        (2.0, 0.5),  # (2-1)/2 = 0.5
        (4.0, 0.75),  # (4-1)/4 = 0.75
        (10.0, 0.9),  # (10-1)/10 = 0.9
        (100.0, 0.99),  # (100-1)/100 = 0.99
        # As theta → ∞, tau → 1 (perfect dependence)
    ],
)
def test_kendalls_tau_values(theta, expected_tau):
    """Test Kendall's tau for different theta values"""
    # Handle the special case where theta=1 returns IndependenceCopula
    if theta == 1.0:
        # Create independence copula directly for testing
        indep_copula = BivIndependenceCopula()
        tau = indep_copula.tau()
        assert float(tau) == expected_tau
    else:
        # Create Gumbel-Hougaard copula with specified theta
        copula = MultivariateGumbelHougaard(dimension=2, theta=theta)
        tau = copula.tau()
        assert abs(float(tau) - expected_tau) < 1e-10


def test_extreme_dependency_limits():
    """Test behavior at dependency parameter limits"""
    # Low theta (near 1) should be close to independence
    low_theta_copula = MultivariateGumbelHougaard(dimension=2, theta=1.001)
    indep_copula = BivIndependenceCopula(dimension=2)

    # Values should be close for interior points
    u, v = 0.3, 0.7
    low_theta_value = float(low_theta_copula.cdf(u, v))
    indep_value = float(indep_copula.cdf(u, v))
    assert abs(low_theta_value - indep_value) < 0.01

    # High theta should be close to perfect dependence (min)
    high_theta_copula = MultivariateGumbelHougaard(dimension=2, theta=100)
    high_theta_value = float(high_theta_copula.cdf(u, v))
    min_value = min(u, v)
    assert abs(high_theta_value - min_value) < 0.01
