import numpy as np
import pytest
import sympy

from copul.family.archimedean.nelsen19 import Nelsen19
from copul.family.other.pi_over_sigma_minus_pi import PiOverSigmaMinusPi


@pytest.fixture
def nelsen19_copula():
    """Fixture providing a Nelsen19 copula with theta=1."""
    return Nelsen19(1)


def test_special_cases():
    """Test that special cases return appropriate copula instances."""
    # Test theta = 0 returns PiOverSigmaMinusPi
    special_case = Nelsen19(0)
    assert isinstance(special_case, PiOverSigmaMinusPi)

    # Test via create factory method
    special_case_create = Nelsen19.create(0)
    assert isinstance(special_case_create, PiOverSigmaMinusPi)

    # Test via call method
    base_copula = Nelsen19(1)
    special_case_call = base_copula(0)
    assert isinstance(special_case_call, PiOverSigmaMinusPi)


def test_parameter_validation():
    """Test parameter validation for the Nelsen19 copula."""
    # Valid values should not raise errors
    Nelsen19(0)  # Lower bound
    Nelsen19(1)  # Interior point
    Nelsen19(10)  # Larger value

    # Negative values should raise ValueError
    with pytest.raises(ValueError, match="must be >= 0"):
        Nelsen19(-0.1)


def test_is_absolutely_continuous(nelsen19_copula):
    """Test the is_absolutely_continuous property."""
    # Nelsen19 is absolutely continuous
    assert nelsen19_copula.is_absolutely_continuous


def test_generator_function(nelsen19_copula):
    """Test the generator function of the Nelsen19 copula."""
    # Test generator at specific t values
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    for t in t_values:
        # Manual calculation using the formula: exp(theta/t) - exp(theta)
        theta = 1
        expected = np.exp(theta / t) - np.exp(theta)
        actual = float(nelsen19_copula.generator(t))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_inverse_generator_function(nelsen19_copula):
    """Test the inverse generator function of the Nelsen19 copula."""
    # Test inverse generator at specific y values
    y_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    for y in y_values:
        # Manual calculation using the formula: theta/log(y + exp(theta))
        theta = 1
        expected = theta / np.log(y + np.exp(theta))
        actual = float(nelsen19_copula.inv_generator(y))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_cdf_function(nelsen19_copula):
    """Test the CDF function of the Nelsen19 copula."""
    # Test CDF at specific points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2), (0.9, 0.8)]

    for u, v in points:
        # Manual calculation using the formula
        theta = 1
        expected = theta / np.log(
            -np.exp(theta) + np.exp(theta / u) + np.exp(theta / v)
        )
        actual = float(nelsen19_copula.cdf(u, v))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_cdf_boundary_conditions(nelsen19_copula):
    """Test boundary conditions for the Nelsen19 copula CDF."""
    # Test points along the boundaries
    u_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # C(u,0) = 0 for any u in [0,1]
    for u in u_values:
        assert np.isclose(float(nelsen19_copula.cdf(u, 0)), 0, atol=1e-10)
        assert np.isclose(float(nelsen19_copula.cdf(0, u)), 0, atol=1e-10)

    # C(u,1) = u and C(1,v) = v for any u,v in [0,1]
    for u in u_values:
        assert np.isclose(float(nelsen19_copula.cdf(u, 1)), u, rtol=1e-5)
        assert np.isclose(float(nelsen19_copula.cdf(1, u)), u, rtol=1e-5)


def test_theta_dependent_behavior():
    """Test how behavior changes with different theta values."""
    # Create copulas with different theta values
    copula1 = Nelsen19(0.5)
    copula2 = Nelsen19(1)
    copula3 = Nelsen19(2)

    # Test at a specific point
    u, v = 0.3, 0.4

    # Calculate CDF values
    cdf1 = float(copula1.cdf(u, v))
    cdf2 = float(copula2.cdf(u, v))
    cdf3 = float(copula3.cdf(u, v))

    # For Nelsen19, different theta should give different CDF values
    # Verify they are different
    assert cdf1 != cdf2
    assert cdf2 != cdf3

    # Verify they are all in valid range
    assert 0 <= cdf1 <= 1
    assert 0 <= cdf2 <= 1
    assert 0 <= cdf3 <= 1


def test_pi_over_sigma_minus_pi_case():
    """Test behavior at the special case (theta=0)."""
    # Create a PiOverSigmaMinusPi instance directly
    direct_instance = PiOverSigmaMinusPi()

    # Get an instance via Nelsen19 special case
    special_case = Nelsen19.create(0)

    # Test at some points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2)]

    for u, v in points:
        direct_result = float(direct_instance.cdf(u, v))
        special_result = float(special_case.cdf(u, v))
        assert np.isclose(direct_result, special_result, rtol=1e-3)


def test_large_theta_behavior():
    """Test behavior with a very large theta value."""
    # Create a copula with a large theta
    large_theta = Nelsen19(50)

    # Test at some points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2)]

    for u, v in points:
        # Calculate CDF value
        result = float(large_theta.cdf(u, v))

        # Result should be within [0,1]
        assert 0 <= result <= 1

        # For very large theta, the copula approaches a specific form
        # Just verify it's a valid value


def test_numerical_stability():
    """Test numerical stability at challenging points."""
    copula = Nelsen19(1)

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


def test_inherited_methods():
    """Test that inherited methods work correctly."""
    copula = Nelsen19(1)

    # Test that lambda_L and lambda_U exist and return values
    assert hasattr(copula, "lambda_L")
    assert hasattr(copula, "lambda_U")

    # These methods should return real numbers or sympy expressions
    lambda_l = copula.lambda_L()
    lambda_u = copula.lambda_U()

    # Verify we can convert to float if numerical, or that they're sympy expressions
    try:
        float(lambda_l)
    except TypeError:
        assert isinstance(lambda_l, sympy.Expr)

    try:
        float(lambda_u)
    except TypeError:
        assert isinstance(lambda_u, sympy.Expr)
