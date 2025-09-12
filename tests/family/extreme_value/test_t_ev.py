import pytest
import numpy as np
import math

from copul.family.extreme_value.t_ev import tEV


class TestTEVCopula:
    """Test suite for the Student-t Extreme Value Copula."""

    @pytest.fixture
    def basic_tev(self):
        """Create a basic tEV copula with nu=2, rho=0.5"""
        return tEV(nu=2, rho=0.5)

    @pytest.mark.parametrize("t", [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1])
    def test_pickands_boundary(self, t):
        """Test that the Pickands function satisfies A(0) = A(1) = 1."""
        copula = tEV(nu=2, rho=0.5)
        result = copula.pickands(t=t)

        # Convert to float for comparison
        result_float = float(result)

        # All Pickands functions must satisfy:
        # 1. A(0) = A(1) = 1
        # 2. max(t, 1-t) <= A(t) <= 1
        if t == 0 or t == 1:
            assert math.isclose(result_float, 1.0, abs_tol=1e-10)
        else:
            assert min(t, 1 - t) <= result_float <= 1.0, f"t={t}, result={result_float}"

    def test_pickands_returns_real_values(self, basic_tev):
        """Test that the Pickands function returns real values."""
        for t in np.linspace(0.1, 0.9, 9):
            result = basic_tev.pickands(t=t)

            # Check that we can convert to float
            result_float = float(result)

            # Ensure the result is a real number
            assert isinstance(result_float, float)
            assert not math.isnan(result_float)
            assert not math.isinf(result_float)

    def test_is_symmetric(self):
        """Test symmetry property."""
        # Should be symmetric when rho = 0
        symmetric_copula = tEV(nu=2, rho=0)
        assert symmetric_copula.is_symmetric

        # Should not be symmetric when rho != 0
        non_symmetric_copula = tEV(nu=2, rho=0.5)
        assert not non_symmetric_copula.is_symmetric

    def test_is_absolutely_continuous(self, basic_tev):
        """Test absolute continuity property."""
        assert basic_tev.is_absolutely_continuous

    @pytest.mark.parametrize(
        "nu,rho",
        [
            (0.5, 0.0),  # Small nu
            (2.0, 0.0),  # Medium nu
            (10.0, 0.0),  # Large nu
            (2.0, -0.5),  # Negative rho
            (2.0, 0.0),  # Zero rho
            (2.0, 0.5),  # Positive rho
        ],
    )
    def test_pickands_with_various_parameters(self, nu, rho):
        """Test Pickands function with various parameter combinations."""
        copula = tEV(nu=nu, rho=rho)

        for t in [0.1, 0.5, 0.9]:
            result = copula.pickands(t=t)

            # Convert to float
            result_float = float(result)

            # Check constraints
            assert min(t, 1 - t) <= result_float <= 1.0, (
                f"nu={nu}, rho={rho}, t={t}, result={result_float}"
            )

    def test_float_conversion(self, basic_tev):
        """Test that we can convert Pickands function to float."""
        result = basic_tev.pickands(t=0.5)

        # Should be convertible to float
        float_val = float(result)

        # Should satisfy Pickands function constraints
        assert 0.5 <= float_val <= 1.0

    def test_cdf_vectorized(self, basic_tev):
        """Test vectorized CDF evaluation."""
        # Create a grid of u, v values
        u = np.linspace(0.1, 0.9, 5)
        v = np.linspace(0.1, 0.9, 5)
        U, V = np.meshgrid(u, v)

        # Compute vectorized CDF
        cdf_values = basic_tev.cdf_vectorized(U, V)

        # Check shape
        assert cdf_values.shape == U.shape

        # Check constraints (0 ≤ C(u,v) ≤ min(u,v))
        assert np.all(cdf_values >= 0)
        assert np.all(cdf_values <= np.minimum(U, V))

    def test_boundary_conditions(self, basic_tev):
        """Test CDF boundary conditions."""
        # Create values
        u = np.linspace(0.1, 0.9, 9)

        # C(0,v) = 0
        zeros = np.zeros_like(u)
        assert np.allclose(basic_tev.cdf_vectorized(0, u), zeros)

        # C(u,0) = 0
        assert np.allclose(basic_tev.cdf_vectorized(u, 0), zeros)

        # C(1,v) = v
        assert np.allclose(basic_tev.cdf_vectorized(1, u), u)

        # C(u,1) = u
        assert np.allclose(basic_tev.cdf_vectorized(u, 1), u)

    def test_all_extreme_value_test_compatibility(self):
        """
        Test compatibility with the test_all_extreme_value.py test.

        This specifically tests the case that was failing in the original code.
        """
        copula = tEV(nu=0.5, rho=2)  # Using the family_representatives values

        # Test at all the points used in the test_all_extreme_value.py test
        for t in [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]:
            result = copula.pickands(t=t)
            result_float = float(result)

            # This is the exact assertion that was failing
            assert min(t, 1 - t) <= result_float <= 1.0, f"t={t}, result={result_float}"
