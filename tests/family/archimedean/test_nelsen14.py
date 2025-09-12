import numpy as np
import pytest

from copul.family.archimedean.nelsen14 import Nelsen14
from copul.family.other.pi_over_sigma_minus_pi import PiOverSigmaMinusPi


@pytest.fixture
def nelsen14_copula():
    """Fixture providing a Nelsen14 copula with theta=2."""
    return Nelsen14(2)


def test_special_cases():
    """Test that special cases return appropriate copula instances."""
    # Test theta = 1 returns PiOverSigmaMinusPi
    special_case = Nelsen14(1)
    assert isinstance(special_case, PiOverSigmaMinusPi)

    # Test via create factory method
    special_case_create = Nelsen14.create(1)
    assert isinstance(special_case_create, PiOverSigmaMinusPi)

    # Test via call method
    base_copula = Nelsen14(2)
    special_case_call = base_copula(1)
    assert isinstance(special_case_call, PiOverSigmaMinusPi)


def test_parameter_validation():
    """Test parameter validation for the Nelsen14 copula."""
    # Valid values should not raise errors
    Nelsen14(1)  # Lower bound
    Nelsen14(2)  # Interior point
    Nelsen14(10)  # Larger value

    # Values below lower bound should raise ValueError
    with pytest.raises(ValueError, match="must be >= 1"):
        Nelsen14(0.5)

    # Negative values should raise ValueError
    with pytest.raises(ValueError, match="must be >= 1"):
        Nelsen14(-1)


def test_is_absolutely_continuous(nelsen14_copula):
    """Test the is_absolutely_continuous property."""
    # Nelsen14 is absolutely continuous
    assert nelsen14_copula.is_absolutely_continuous


def test_generator_function(nelsen14_copula):
    """Test the generator function of the Nelsen14 copula."""
    # Test generator at specific t values
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    for t in t_values:
        # Manual calculation using the formula: (t^(-1/theta) - 1)^theta
        theta = 2
        expected = (t ** (-1 / theta) - 1) ** theta
        actual = float(nelsen14_copula.generator(t))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_inverse_generator_function(nelsen14_copula):
    """Test the inverse generator function of the Nelsen14 copula."""
    # Test inverse generator at specific y values
    y_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    for y in y_values:
        # Manual calculation using the formula: (y^(1/theta) + 1)^(-theta)
        theta = 2
        expected = (y ** (1 / theta) + 1) ** (-theta)
        actual = float(nelsen14_copula.inv_generator(y))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_cdf_function(nelsen14_copula):
    """Test the CDF function of the Nelsen14 copula."""
    # Test CDF at specific points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2), (0.9, 0.8)]

    for u, v in points:
        # Manual calculation using the formula
        theta = 2
        term = (
            (u ** (-1 / theta) - 1) ** theta + (v ** (-1 / theta) - 1) ** theta
        ) ** (1 / theta)
        expected = (1 + term) ** (-theta)
        actual = float(nelsen14_copula.cdf(u, v))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_cdf_boundary_conditions(nelsen14_copula):
    """Test boundary conditions for the Nelsen14 copula CDF."""
    # Test points along the boundaries
    u_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # C(u,0) = 0 for any u in [0,1]
    for u in u_values:
        assert np.isclose(float(nelsen14_copula.cdf(u, 0)), 0, atol=1e-10)
        assert np.isclose(float(nelsen14_copula.cdf(0, u)), 0, atol=1e-10)

    # C(u,1) = u and C(1,v) = v for any u,v in [0,1]
    for u in u_values:
        assert np.isclose(float(nelsen14_copula.cdf(u, 1)), u, rtol=1e-5)
        assert np.isclose(float(nelsen14_copula.cdf(1, u)), u, rtol=1e-5)


def test_lambda_l(nelsen14_copula):
    """Test lower tail dependence coefficient (lambda_L)."""
    # For Nelsen14, lambda_L = 1/2
    expected = 1 / 2
    actual = float(nelsen14_copula.lambda_L())
    assert np.isclose(actual, expected, rtol=1e-5)


def test_lambda_u(nelsen14_copula):
    """Test upper tail dependence coefficient (lambda_U)."""
    # For Nelsen14, lambda_U = 2 - 2^(1/theta)
    theta = 2
    expected = 2 - 2 ** (1 / theta)
    actual = float(nelsen14_copula.lambda_U())
    assert np.isclose(actual, expected, rtol=1e-5)

    # Test with different theta values
    for theta in [1.5, 3, 5]:
        copula = Nelsen14(theta)
        expected = 2 - 2 ** (1 / theta)
        actual = float(copula.lambda_U())
        assert np.isclose(actual, expected, rtol=1e-5)


def test_theta_dependent_behavior():
    """Test how behavior changes with different theta values."""
    # Create copulas with different theta values
    copula1 = Nelsen14(1.5)
    copula2 = Nelsen14(3)
    copula3 = Nelsen14(5)

    # Test at a specific point
    u, v = 0.3, 0.4

    # Calculate CDF values
    cdf1 = float(copula1.cdf(u, v))
    cdf2 = float(copula2.cdf(u, v))
    cdf3 = float(copula3.cdf(u, v))

    # For Nelsen14, different theta should give different CDF values
    # Verify they are different
    assert cdf1 != cdf2
    assert cdf2 != cdf3

    # Verify they are all in valid range
    assert 0 <= cdf1 <= 1
    assert 0 <= cdf2 <= 1
    assert 0 <= cdf3 <= 1


def test_special_case_behavior():
    """Test behavior at the special case (theta=1)."""
    # Create a PiOverSigmaMinusPi instance directly
    direct_instance = PiOverSigmaMinusPi()

    # Get an instance via Nelsen14 special case
    special_case = Nelsen14.create(1)

    # Test at some points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2)]

    for u, v in points:
        direct_result = float(direct_instance.cdf(u, v))
        special_result = float(special_case.cdf(u, v))
        assert np.isclose(direct_result, special_result, rtol=1e-5)


def test_large_theta_behavior():
    """Test behavior with a very large theta value."""
    # With large theta, check that computations remain stable
    large_theta = Nelsen14(50)

    # Test at some points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2)]

    for u, v in points:
        result = float(large_theta.cdf(u, v))

        # Result should be within [0,1]
        assert 0 <= result <= 1

        # For very large theta, the copula approaches a specific form
        # Calculate the expected value with large theta
        theta = 50
        term = (
            (u ** (-1 / theta) - 1) ** theta + (v ** (-1 / theta) - 1) ** theta
        ) ** (1 / theta)
        expected = (1 + term) ** (-theta)
        assert np.isclose(result, expected, rtol=1e-5)


def test_numerical_stability():
    """Test numerical stability at challenging points."""
    copula = Nelsen14(2)

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

        # For very small u or v, result should be close to 0
        if u < 0.01 or v < 0.01:
            assert result < 0.01

        # For u close to 1, result should be close to v
        if u > 0.99:
            assert result > 0.4  # Should approach v but might not be exactly v

        # For v close to 1, result should be close to u
        if v > 0.99:
            assert result > 0.4  # Should approach u but might not be exactly u
