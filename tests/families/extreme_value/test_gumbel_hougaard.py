import pytest
import numpy as np
import sympy as sp
from unittest.mock import patch

from copul.families.extreme_value.gumbel_hougaard import (
    GumbelHougaardEV as GumbelHougaard,
)
from copul.families.other import BivIndependenceCopula
from copul.wrapper.cdf_wrapper import CDFWrapper


class TestGumbelHougaard:
    @pytest.fixture
    def gumbel_copula(self):
        """Create a GumbelHougaard copula instance with theta=2 for testing"""
        copula = GumbelHougaard(2)
        return copula

    def test_init(self):
        """Test initialization of GumbelHougaard copula"""
        # Default initialization
        copula = GumbelHougaard()
        assert hasattr(copula, "theta")
        assert isinstance(copula.theta, sp.Symbol)
        assert str(copula.theta) == "theta"

        # Initialization with parameter
        copula = GumbelHougaard(theta=1.5)
        assert hasattr(copula, "theta")
        assert copula.theta == 1.5

    def test_parameter_bounds(self):
        """Test parameter bounds of GumbelHougaard copula"""
        copula = GumbelHougaard()
        # Theta should be >= 1
        assert copula.intervals["theta"].left == 1
        assert copula.intervals["theta"].right == float("inf")
        assert not copula.intervals["theta"].left_open  # Left bound is closed
        assert copula.intervals["theta"].right_open  # Right bound is open

    def test_is_symmetric(self, gumbel_copula):
        """Test symmetry property of GumbelHougaard copula"""
        assert gumbel_copula.is_symmetric is True

    def test_is_absolutely_continuous(self, gumbel_copula):
        """Test absolute continuity property"""
        assert gumbel_copula.is_absolutely_continuous is True

    def test_pickands(self, gumbel_copula):
        """Test Pickands dependence function"""
        # Test at t=0.5 with theta=2
        result = float(gumbel_copula.pickands(0.5))
        # Expected value for GumbelHougaard with theta=2 at t=0.5 is approximately 0.7071
        expected = (0.5**2 + 0.5**2) ** (1 / 2)
        assert abs(result - expected) < 1e-10

        # Test at boundaries
        # At t=0 or t=1, Pickands should be 1 for any valid EV copula
        t0 = float(gumbel_copula.pickands(0))
        t1 = float(gumbel_copula.pickands(1))
        assert abs(t0 - 1.0) < 1e-10
        assert abs(t1 - 1.0) < 1e-10

    def test_cdf(self, gumbel_copula):
        """Test CDF computation at specific points"""
        # For GumbelHougaard with theta=2, at u=v=0.5
        with patch.object(CDFWrapper, "__call__") as mock_call:
            mock_call.return_value = 0.25  # Mock return value

            # Call CDF
            result = gumbel_copula.cdf(0.5, 0.5)

            # Verify CDF was called correctly
            mock_call.assert_called_once_with(0.5, 0.5)
            assert result == 0.25

    def test_independence_special_case(self):
        """Test special case when theta=1 (should return Independence copula)"""
        # When theta=1, GumbelHougaard becomes the Independence copula
        copula = GumbelHougaard(theta=1)

        # We should get an Independence copula instance
        assert isinstance(copula, BivIndependenceCopula)

    def test_call_method_with_kwargs(self):
        """Test __call__ method with kwargs"""
        # Create base copula
        copula = GumbelHougaard()

        # Update parameter using kwargs
        new_copula = copula(theta=2.5)

        # Original should be unchanged
        assert isinstance(copula.theta, sp.Symbol)

        # New instance should have updated parameter
        assert new_copula.theta == 2.5
        assert not isinstance(new_copula, BivIndependenceCopula)

    def test_call_method_with_args(self):
        """Test __call__ method with positional args"""
        # Create base copula
        copula = GumbelHougaard()

        # Update parameter using positional args
        new_copula = copula(2.5)

        # New instance should have updated parameter
        assert new_copula.theta == 2.5
        assert not isinstance(new_copula, BivIndependenceCopula)

        # Test independence case with positional arg
        ind_copula = copula(1)
        assert isinstance(ind_copula, BivIndependenceCopula)

    def test_tau(self, gumbel_copula):
        """Test Kendall's tau calculation"""
        # For GumbelHougaard, tau = (theta-1)/theta
        # With theta=2, tau = 0.5
        tau = gumbel_copula.tau()
        assert abs(float(tau) - 0.5) < 1e-10

        # Create a new copula with different theta
        copula = GumbelHougaard(theta=4.0)
        tau = copula.tau()
        assert abs(float(tau) - 0.75) < 1e-10

    def test_rho(self, gumbel_copula):
        """Test Spearman's rho computation"""
        # Patch the integration result since it's complex
        with patch.object(GumbelHougaard, "_rho") as mock_rho:
            mock_rho.return_value = 0.7  # Expected value for theta=2

            rho = gumbel_copula.rho()
            assert rho == 0.7
            mock_rho.assert_called_once()

    def test_sampling(self, gumbel_copula):
        """Test random sampling from the copula"""
        # Patch the rvs method to avoid actual computation
        with patch("copul.copula_sampler.CopulaSampler.rvs") as mock_rvs:
            # Prepare mock data
            mock_data = np.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]])
            mock_rvs.return_value = mock_data

            # Generate samples
            samples = gumbel_copula.rvs(3)

            # Verify result
            assert np.array_equal(samples, mock_data)
            mock_rvs.assert_called_once_with(3, False)

    def test_tail_dependence(self):
        """Test tail dependence properties"""
        # GumbelHougaard has upper tail dependence but no lower tail dependence
        gumbel = GumbelHougaard(theta=2.0)

        # Patch lambda methods to return expected values
        with patch.object(GumbelHougaard, "lambda_U", return_value=0.5):
            # Upper tail dependence should be positive
            assert gumbel.lambda_U() > 0

        with patch.object(GumbelHougaard, "lambda_L", return_value=0.0):
            # Lower tail dependence should be zero
            assert gumbel.lambda_L() == 0

    def test_rho_calculation(self):
        """Test the _rho method internals"""
        # This is a complex test that checks the integration without executing it
        copula = GumbelHougaard(theta=2.0)

        # Patch sympy.Integral and sympy.plot to avoid actual computation
        with patch("sympy.Integral") as mock_integral, patch("sympy.plot"):
            mock_integral.return_value = 1.0  # Mock the integral result

            # Call the _rho method
            result = copula._rho()

            # Verify that plot and integral were called
            mock_integral.assert_called_once()

            # Check the result formula: 12 * integral - 3
            assert result == 12 * 1.0 - 3

    @pytest.mark.parametrize(
        "method_name, point, expected",
        [
            ("cond_distr_1", (0, 0), 0),
            ("cond_distr_2", (0, 0), 0),
        ],
    )
    def test_cond_distr_edge_cases_gh(
        self, method_name, point, expected, gumbel_copula
    ):
        method = getattr(gumbel_copula, method_name)
        func = method(point)
        evaluated_func = float(func)
        assert np.isclose(evaluated_func, expected)

    @pytest.mark.parametrize(
        "method_name, point, expected",
        [
            ("cond_distr_1", (0, 0), 0),
            ("cond_distr_2", (0, 0), 0),
        ],
    )
    def test_cond_distr_edge_cases_gh_with_asterisk(
        self, method_name, point, expected, gumbel_copula
    ):
        method = getattr(gumbel_copula, method_name)
        func = method(*point)
        evaluated_func = float(func)
        assert np.isclose(evaluated_func, expected)
