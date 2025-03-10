import numpy as np
import pytest

from copul.families.archimedean.nelsen12 import Nelsen12
from copul.families.other.pi_over_sigma_minus_pi import PiOverSigmaMinusPi


@pytest.fixture
def nelsen12_copula():
    """Fixture providing a Nelsen12 copula with theta=2."""
    return Nelsen12(2)


def test_special_cases():
    """Test that special cases return appropriate copula instances."""
    # Test theta = 1 returns PiOverSigmaMinusPi
    special_case = Nelsen12(1)
    assert isinstance(special_case, PiOverSigmaMinusPi)

    # Test via create factory method
    special_case_create = Nelsen12.create(1)
    assert isinstance(special_case_create, PiOverSigmaMinusPi)

    # Test via call method
    base_copula = Nelsen12(2)
    special_case_call = base_copula(1)
    assert isinstance(special_case_call, PiOverSigmaMinusPi)


def test_parameter_validation():
    """Test parameter validation for the Nelsen12 copula."""
    # Valid values should not raise errors
    Nelsen12(1)  # Lower bound
    Nelsen12(2)  # Interior point
    Nelsen12(10)  # Larger value

    # Values below lower bound should raise ValueError
    with pytest.raises(ValueError, match="must be >= 1"):
        Nelsen12(0.5)

    # Negative values should raise ValueError
    with pytest.raises(ValueError, match="must be >= 1"):
        Nelsen12(-1)


def test_is_absolutely_continuous(nelsen12_copula):
    """Test the is_absolutely_continuous property."""
    # Nelsen12 is absolutely continuous
    assert nelsen12_copula.is_absolutely_continuous


def test_generator_function(nelsen12_copula):
    """Test the generator function of the Nelsen12 copula."""
    # Test generator at specific t values
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    for t in t_values:
        # Manual calculation using the formula: (1/t - 1)^theta
        theta = 2
        expected = (1 / t - 1) ** theta
        actual = float(nelsen12_copula.generator(t))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_inverse_generator_function(nelsen12_copula):
    """Test the inverse generator function of the Nelsen12 copula."""
    # Test inverse generator at specific y values
    y_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    for y in y_values:
        # Manual calculation using the formula: 1/(y^(1/theta) + 1)
        theta = 2
        expected = 1 / (y ** (1 / theta) + 1)
        actual = float(nelsen12_copula.inv_generator(y))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_cdf_function(nelsen12_copula):
    """Test the CDF function of the Nelsen12 copula."""
    # Test CDF at specific points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2), (0.9, 0.8)]

    for u, v in points:
        # Manual calculation using the formula
        theta = 2
        term = ((u ** (-1) - 1) ** theta + (v ** (-1) - 1) ** theta) ** (1 / theta)
        expected = 1 / (1 + term)
        actual = float(nelsen12_copula.cdf(u, v))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_cdf_boundary_conditions(nelsen12_copula):
    """Test boundary conditions for the Nelsen12 copula CDF."""
    # Test points along the boundaries
    u_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # C(u,0) = 0 for any u in [0,1]
    for u in u_values:
        assert np.isclose(float(nelsen12_copula.cdf(u, 0)), 0, atol=1e-10)
        assert np.isclose(float(nelsen12_copula.cdf(0, u)), 0, atol=1e-10)

    # C(u,1) = u and C(1,v) = v for any u,v in [0,1]
    for u in u_values:
        assert np.isclose(float(nelsen12_copula.cdf(u, 1)), u, rtol=1e-5)
        assert np.isclose(float(nelsen12_copula.cdf(1, u)), u, rtol=1e-5)


def test_lambda_l(nelsen12_copula):
    """Test lower tail dependence coefficient (lambda_L)."""
    # For Nelsen12, lambda_L = 2^(-1/theta)
    theta = 2
    expected = 2 ** (-1 / theta)
    actual = float(nelsen12_copula.lambda_L())
    assert np.isclose(actual, expected, rtol=1e-5)

    # Test with different theta values
    for theta in [1.5, 3, 5]:
        copula = Nelsen12(theta)
        expected = 2 ** (-1 / theta)
        actual = float(copula.lambda_L())
        assert np.isclose(actual, expected, rtol=1e-5)


def test_lambda_u(nelsen12_copula):
    """Test upper tail dependence coefficient (lambda_U)."""
    # For Nelsen12, lambda_U = 2 - 2^(1/theta)
    theta = 2
    expected = 2 - 2 ** (1 / theta)
    actual = float(nelsen12_copula.lambda_U())
    assert np.isclose(actual, expected, rtol=1e-5)

    # Test with different theta values
    for theta in [1.5, 3, 5]:
        copula = Nelsen12(theta)
        expected = 2 - 2 ** (1 / theta)
        actual = float(copula.lambda_U())
        assert np.isclose(actual, expected, rtol=1e-5)


def test_theta_dependent_behavior():
    """Test how behavior changes with different theta values."""
    # Create copulas with different theta values
    copula1 = Nelsen12(1.5)
    copula2 = Nelsen12(3)
    copula3 = Nelsen12(5)

    # Test at a specific point
    u, v = 0.3, 0.4

    # Calculate CDF values
    cdf1 = float(copula1.cdf(u, v))
    cdf2 = float(copula2.cdf(u, v))
    cdf3 = float(copula3.cdf(u, v))

    # For Nelsen12, as theta increases, the copula should approach the minimum copula
    # So the CDF values should decrease as theta increases
    assert cdf1 < cdf2 < cdf3


def test_special_case_behavior():
    """Test behavior at the special case (theta=1)."""
    # Create a PiOverSigmaMinusPi instance directly
    direct_instance = PiOverSigmaMinusPi()

    # Get an instance via Nelsen12 special case
    special_case = Nelsen12.create(1)

    # Test at some points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2)]

    for u, v in points:
        direct_result = float(direct_instance.cdf(u, v))
        special_result = float(special_case.cdf(u, v))
        assert np.isclose(direct_result, special_result, rtol=1e-5)


def test_large_theta_behavior():
    """Test behavior with a very large theta value."""
    # With large theta, Nelsen12 approaches the minimum copula: C(u,v) = min(u,v)
    large_theta = Nelsen12(100)

    # Test at some points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2)]

    for u, v in points:
        result = float(large_theta.cdf(u, v))
        expected = min(u, v)
        assert np.isclose(result, expected, rtol=1e-2)  # Allow some tolerance


def test_numerical_stability():
    """Test numerical stability at challenging points."""
    copula = Nelsen12(2)

    # Test at points very close to 0 and 1
    edge_points = [
        (0.001, 0.5),  # u close to 0
        (0.5, 0.001),  # v close to 0
        (0.999, 0.5),  # u close to 1
        (0.5, 0.999),  # v close to 1
    ]

    for u, v in edge_points:
        # Verify CDF computation doesn't raise errors
        result = float(copula.cdf(u, v))

        # Result should be within [0,1]
        assert 0 <= result <= 1

        # Test the basic properties
        if u < 0.01 or v < 0.01:  # Very close to 0
            assert result < 0.01

        if u > 0.99:  # Very close to 1
            assert result > 0.4  # Should be close to v

        if v > 0.99:  # Very close to 1
            assert result > 0.4  # Should be close to u
