from unittest.mock import patch

import numpy as np
import pytest

from copul.family.archimedean.nelsen20 import Nelsen20
from copul.family.other.biv_independence_copula import BivIndependenceCopula


@pytest.fixture
def nelsen20_copula():
    """Fixture providing a Nelsen20 copula with theta=1."""
    return Nelsen20(1)


def test_special_cases():
    """Test that special cases return appropriate copula instances."""
    # Test theta = 0 returns IndependenceCopula
    independence = Nelsen20(0)
    assert isinstance(independence, BivIndependenceCopula)

    # Test via create factory method
    independence_create = Nelsen20.create(0)
    assert isinstance(independence_create, BivIndependenceCopula)

    # Test via call method
    base_copula = Nelsen20(1)
    independence_call = base_copula(0)
    assert isinstance(independence_call, BivIndependenceCopula)


def test_parameter_validation():
    """Test parameter validation for the Nelsen20 copula."""
    # Valid values should not raise errors
    Nelsen20(0)  # Lower bound
    Nelsen20(1)  # Interior point
    Nelsen20(10)  # Larger value

    # Negative values should raise ValueError
    with pytest.raises(ValueError, match="must be >= 0"):
        Nelsen20(-0.1)


def test_is_absolutely_continuous(nelsen20_copula):
    """Test the is_absolutely_continuous property."""
    # Nelsen20 is absolutely continuous
    assert nelsen20_copula.is_absolutely_continuous


def test_generator_function(nelsen20_copula):
    """Test the generator function of the Nelsen20 copula."""
    # Test generator at specific t values
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    for t in t_values:
        # Manual calculation using the formula: exp(t^(-theta)) - exp(1)
        theta = 1
        expected = np.exp(t ** (-theta)) - np.exp(1)
        actual = float(nelsen20_copula.generator(t))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_inverse_generator_function(nelsen20_copula):
    """Test the inverse generator function of the Nelsen20 copula."""
    # Test inverse generator at specific y values
    y_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    for y in y_values:
        # Manual calculation using the formula: log(y + e)^(-1/theta)
        theta = 1
        expected = np.log(y + np.e) ** (-1 / theta)
        actual = float(nelsen20_copula.inv_generator(y))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_cdf_function(nelsen20_copula):
    """Test the CDF function of the Nelsen20 copula."""
    # Test CDF at specific points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2), (0.9, 0.8)]

    for u, v in points:
        # Manual calculation using the formula
        theta = 1
        expected = np.log(np.exp(u ** (-theta)) + np.exp(v ** (-theta)) - np.e) ** (
            -1 / theta
        )
        actual = float(nelsen20_copula.cdf(u, v))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_cdf_boundary_conditions(nelsen20_copula):
    """Test boundary conditions for the Nelsen20 copula CDF."""
    # Test points along the boundaries
    u_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # C(u,0) = 0 for any u in [0,1]
    for u in u_values:
        assert np.isclose(float(nelsen20_copula.cdf(u, 0)), 0, atol=1e-10)
        assert np.isclose(float(nelsen20_copula.cdf(0, u)), 0, atol=1e-10)

    # C(u,1) = u and C(1,v) = v for any u,v in [0,1]
    for u in u_values:
        assert np.isclose(float(nelsen20_copula.cdf(u, 1)), u, rtol=1e-5)
        assert np.isclose(float(nelsen20_copula.cdf(1, u)), u, rtol=1e-5)


def test_conditional_distribution_2(nelsen20_copula):
    """Test the second conditional distribution function."""
    # Test conditional distribution at some valid points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2)]

    for u, v in points:
        # Just verify it returns a value in [0,1] without errors
        result = float(nelsen20_copula.cond_distr_2(u, v))
        assert 0 <= result <= 1


def test_theta_dependent_behavior():
    """Test how behavior changes with different theta values."""
    # Create copulas with different theta values
    copula1 = Nelsen20(0.5)
    copula2 = Nelsen20(1)
    copula3 = Nelsen20(2)

    # Test at a specific point
    u, v = 0.3, 0.4

    # Calculate CDF values
    cdf1 = float(copula1.cdf(u, v))
    cdf2 = float(copula2.cdf(u, v))
    cdf3 = float(copula3.cdf(u, v))

    # For Nelsen20, different theta should give different CDF values
    # Verify they are different
    assert cdf1 != cdf2
    assert cdf2 != cdf3

    # Verify they are all in valid range
    assert 0 <= cdf1 <= 1
    assert 0 <= cdf2 <= 1
    assert 0 <= cdf3 <= 1


def test_independence_case():
    """Test behavior at the special case (theta=0)."""
    # Create an IndependenceCopula instance directly
    direct_instance = BivIndependenceCopula()

    # Get an instance via Nelsen20 special case
    special_case = Nelsen20.create(0)

    # Test at some points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2)]

    for u, v in points:
        direct_result = float(direct_instance.cdf(u, v))
        special_result = float(special_case.cdf(u, v))
        assert np.isclose(direct_result, special_result, rtol=1e-5)


def test_large_theta_behavior():
    """Test behavior with a very large theta value."""
    # Create a copula with a large theta
    large_theta = Nelsen20(50)

    # Test at some points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2)]

    for u, v in points:
        # Calculate CDF value
        result = float(large_theta.cdf(u, v))

        # Result should be within [0,1]
        assert 0 <= result <= 1


def test_rvs_method(nelsen20_copula):
    """Test the rvs method inherited from HeavyComputeArch."""
    # Mock the _sample_values method to avoid actual computation
    with patch.object(nelsen20_copula, "_sample_values", return_value=(0.5, 0.6)):
        # Generate a small sample
        samples = nelsen20_copula.rvs(3)

        # Check shape and values
        assert samples.shape == (3, 2)
        assert np.allclose(samples, np.array([(0.5, 0.6), (0.5, 0.6), (0.5, 0.6)]))


def test_numerical_stability():
    """Test numerical stability at challenging points."""
    copula = Nelsen20(1)

    # Test at points very close to 0
    edge_points = [
        (0.001, 0.5),  # u close to 0
        (0.5, 0.001),  # v close to 0
    ]

    for u, v in edge_points:
        # Verify CDF computation doesn't raise errors
        result = float(copula.cdf(u, v))

        # Result should be within [0,1]
        assert 0 <= result <= 1

        # For very small u or v, result should be close to 0
        assert result < 0.1

    # Test at points very close to 1
    edge_points = [
        (0.999, 0.5),  # u close to 1
        (0.5, 0.999),  # v close to 1
    ]

    for u, v in edge_points:
        # Verify CDF computation doesn't raise errors
        result = float(copula.cdf(u, v))

        # Result should be within [0,1]
        assert 0 <= result <= 1

        # For u close to 1, result should be close to v
        if u > 0.99:
            assert np.isclose(result, v, rtol=1e-2)

        # For v close to 1, result should be close to u
        if v > 0.99:
            assert np.isclose(result, u, rtol=1e-2)
