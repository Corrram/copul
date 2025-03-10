import numpy as np
import pytest

from copul.families.archimedean.nelsen22 import Nelsen22
from copul.families.other.independence_copula import IndependenceCopula


@pytest.fixture
def nelsen22_copula():
    """Fixture providing a Nelsen22 copula with theta=0.5."""
    return Nelsen22(0.5)


def test_special_cases():
    """Test that special cases return appropriate copula instances."""
    # Test theta = 0 returns IndependenceCopula
    independence = Nelsen22(0)
    assert isinstance(independence, IndependenceCopula)

    # Test via create factory method
    independence_create = Nelsen22.create(0)
    assert isinstance(independence_create, IndependenceCopula)

    # Test via call method
    base_copula = Nelsen22(0.5)
    independence_call = base_copula(0)
    assert isinstance(independence_call, IndependenceCopula)


def test_parameter_validation():
    """Test parameter validation for the Nelsen22 copula."""
    # Valid values should not raise errors
    Nelsen22(0)  # Lower bound
    Nelsen22(0.5)  # Interior point
    Nelsen22(1)  # Upper bound

    # Values below lower bound should raise ValueError
    with pytest.raises(ValueError, match="must be >= 0"):
        Nelsen22(-0.1)

    # Values above upper bound should raise ValueError
    with pytest.raises(ValueError, match="must be <= 1"):
        Nelsen22(1.1)


def test_is_absolutely_continuous(nelsen22_copula):
    """Test the is_absolutely_continuous property."""
    # Nelsen22 is absolutely continuous
    assert nelsen22_copula.is_absolutely_continuous


def test_generator_function(nelsen22_copula):
    """Test the generator function of the Nelsen22 copula."""
    # Test generator at specific t values
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    for t in t_values:
        # Manual calculation using the formula: asin(1 - t^theta)
        theta = 0.5
        expected = np.arcsin(1 - t**theta)
        actual = float(nelsen22_copula.generator(t))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_inverse_generator_function(nelsen22_copula):
    """Test the inverse generator function of the Nelsen22 copula."""
    # Test inverse generator at specific y values
    y_values = [0.1, 0.5, 1.0, np.pi / 4]

    for y in y_values:
        # Manual calculation using the formula: (1 - sin(y))^(1/theta)
        theta = 0.5
        expected = (1 - np.sin(y)) ** (1 / theta)
        actual = float(nelsen22_copula.inv_generator(y))
        assert np.isclose(actual, expected, rtol=1e-5)

    # Test for y > pi/2, should return 0
    large_y = np.pi
    assert float(nelsen22_copula.inv_generator(large_y)) == 0


def test_cdf_function(nelsen22_copula):
    """Test the CDF function of the Nelsen22 copula."""
    # Test CDF at specific points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2), (0.9, 0.8)]

    for u, v in points:
        # Manual calculation using the formula
        theta = 0.5
        condition = np.arcsin(u**theta - 1) + np.arcsin(v**theta - 1) >= -np.pi / 2

        if condition:
            expected = (
                np.sin(np.arcsin(u**theta - 1) + np.arcsin(v**theta - 1)) + 1
            ) ** (1 / theta)
        else:
            expected = 0

        actual = float(nelsen22_copula.cdf(u, v))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_cdf_boundary_conditions(nelsen22_copula):
    """Test boundary conditions for the Nelsen22 copula CDF."""
    # Test points along the boundaries
    u_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # C(u,0) = 0 for any u in [0,1]
    for u in u_values:
        assert np.isclose(float(nelsen22_copula.cdf(u, 0)), 0, atol=1e-10)
        assert np.isclose(float(nelsen22_copula.cdf(0, u)), 0, atol=1e-10)

    # C(u,1) = u and C(1,v) = v for any u,v in [0,1]
    for u in u_values:
        assert np.isclose(float(nelsen22_copula.cdf(u, 1)), u, rtol=1e-5)
        assert np.isclose(float(nelsen22_copula.cdf(1, u)), u, rtol=1e-5)


def test_compute_gen_max(nelsen22_copula):
    """Test the compute_gen_max method."""
    # The maximum value of the generator is pi/2
    expected = np.pi / 2
    actual = float(nelsen22_copula.compute_gen_max())
    assert np.isclose(actual, expected, rtol=1e-10)


def test_lambda_l(nelsen22_copula):
    """Test lower tail dependence coefficient (lambda_L)."""
    # For Nelsen22, lambda_L = 0
    assert float(nelsen22_copula.lambda_L()) == 0


def test_lambda_u(nelsen22_copula):
    """Test upper tail dependence coefficient (lambda_U)."""
    # For Nelsen22, lambda_U = 0
    assert float(nelsen22_copula.lambda_U()) == 0


def test_theta_dependent_behavior():
    """Test how behavior changes with different theta values."""
    # Create copulas with different theta values
    copula1 = Nelsen22(0.2)
    copula2 = Nelsen22(0.5)
    copula3 = Nelsen22(0.8)

    # Test at a specific point
    u, v = 0.3, 0.4

    # Calculate CDF values
    cdf1 = float(copula1.cdf(u, v))
    cdf2 = float(copula2.cdf(u, v))
    cdf3 = float(copula3.cdf(u, v))

    # For Nelsen22, different theta should give different CDF values
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
    direct_instance = IndependenceCopula()

    # Get an instance via Nelsen22 special case
    special_case = Nelsen22.create(0)

    # Test at some points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2)]

    for u, v in points:
        direct_result = float(direct_instance.cdf(u, v))
        special_result = float(special_case.cdf(u, v))
        assert np.isclose(direct_result, special_result, rtol=1e-5)


def test_theta_max_case():
    """Test behavior at the maximum theta value (theta=1)."""
    # Create a Nelsen22 instance with theta=1
    copula = Nelsen22(1)

    # Test at some points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2)]

    for u, v in points:
        # Verify computation doesn't raise errors
        result = float(copula.cdf(u, v))

        # Result should be within [0,1]
        assert 0 <= result <= 1

        # For theta=1, the CDF has a specific form
        # Manual calculation for the theta=1 case
        condition = np.arcsin(u - 1) + np.arcsin(v - 1) >= -np.pi / 2
        if condition:
            expected = np.sin(np.arcsin(u - 1) + np.arcsin(v - 1)) + 1
            assert np.isclose(result, expected, rtol=1e-5)


def test_numerical_stability():
    """Test numerical stability at challenging points."""
    copula = Nelsen22(0.5)

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
