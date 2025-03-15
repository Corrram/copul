from unittest.mock import patch

import numpy as np
import pytest
import sympy

from copul.families.archimedean.nelsen16 import Nelsen16
from copul.families.other.lower_frechet import LowerFrechet


@pytest.fixture
def nelsen16_copula():
    """Fixture providing a Nelsen16 copula with theta=1."""
    return Nelsen16(1)


def test_special_cases():
    """Test that special cases return appropriate copula instances."""
    # Test theta = 0 returns LowerFrechet
    lower_frechet = Nelsen16(0)
    assert isinstance(lower_frechet, LowerFrechet)

    # Test via create factory method
    lower_frechet_create = Nelsen16.create(0)
    assert isinstance(lower_frechet_create, LowerFrechet)

    # Test via call method
    base_copula = Nelsen16(1)
    lower_frechet_call = base_copula(0)
    assert isinstance(lower_frechet_call, LowerFrechet)


def test_parameter_validation():
    """Test parameter validation for the Nelsen16 copula."""
    # Valid values should not raise errors
    Nelsen16(0)  # Lower bound
    Nelsen16(1)  # Interior point
    Nelsen16(10)  # Larger value

    # Negative values should raise ValueError
    with pytest.raises(ValueError, match="must be >= 0"):
        Nelsen16(-0.1)


def test_is_absolutely_continuous(nelsen16_copula):
    """Test the is_absolutely_continuous property."""
    # Nelsen16 is absolutely continuous
    assert nelsen16_copula.is_absolutely_continuous


def test_generator_function(nelsen16_copula):
    """Test the generator function of the Nelsen16 copula."""
    # Test generator at specific t values
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    for t in t_values:
        # Manual calculation using the formula: (theta/t + 1) * (1 - t)
        theta = 1
        expected = (theta / t + 1) * (1 - t)
        actual = float(nelsen16_copula.generator(t))
        assert np.isclose(actual, expected, rtol=1e-5)

def test_inverse_generator_function(nelsen16_copula):
    """Test the inverse generator function of the Nelsen16 copula."""
    # Test inverse generator at specific y values
    y_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    for y in y_values:
        # Manual calculation using the formula:
        # (1 - theta - y + sqrt((theta + y - 1)^2 + 4*theta))/2
        theta = 1
        expected = (1 - theta - y + np.sqrt((theta + y - 1) ** 2 + 4 * theta)) / 2
        actual = float(nelsen16_copula.inv_generator(y))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_cdf_function(nelsen16_copula):
    """Test the CDF function of the Nelsen16 copula."""
    # Test CDF at specific points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2), (0.9, 0.8)]

    for u, v in points:
        # Manual calculation using the formula
        theta = 1
        term1 = u * v * (1 - theta)
        term2 = u * (theta + v) * (v - 1)
        term3 = v * (theta + u) * (u - 1)
        term4 = np.sqrt(4 * theta * u**2 * v**2 + (term1 + term2 + term3) ** 2)
        expected = (term1 + term2 + term3 + term4) / (2 * u * v)
        actual = float(nelsen16_copula.cdf(u, v))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_cdf_boundary_conditions(nelsen16_copula):
    """Test boundary conditions for the Nelsen16 copula CDF."""
    # Test points along the boundaries
    u_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # C(u,0) = 0 for any u in [0,1]
    for u in u_values:
        assert np.isclose(float(nelsen16_copula.cdf(u, 0)), 0, atol=1e-10)
        assert np.isclose(float(nelsen16_copula.cdf(0, u)), 0, atol=1e-10)

    # C(u,1) = u and C(1,v) = v for any u,v in [0,1]
    for u in u_values:
        assert np.isclose(float(nelsen16_copula.cdf(u, 1)), u, rtol=1e-5)
        assert np.isclose(float(nelsen16_copula.cdf(1, u)), u, rtol=1e-5)


def test_derivative_functions(nelsen16_copula):
    """Test the derivative functions of the Nelsen16 copula."""
    # These are complex symbolic functions, so we'll just verify they can be called
    # without errors and return symbolic expressions

    # Patch the 'y' property
    with patch.object(
        nelsen16_copula, "y", sympy.Symbol("y", real=True, positive=True)
    ):
        # Call the derivative functions
        first_deriv = nelsen16_copula.first_deriv_of_ci_char()
        second_deriv = nelsen16_copula.second_deriv_of_ci_char()

        # Verify they return sympy expressions
        assert isinstance(first_deriv, sympy.Expr)
        assert isinstance(second_deriv, sympy.Expr)


def test_lambda_l(nelsen16_copula):
    """Test lower tail dependence coefficient (lambda_L)."""
    # For Nelsen16, lambda_L = 1/2
    expected = 1 / 2
    actual = float(nelsen16_copula.lambda_L())
    assert np.isclose(actual, expected, rtol=1e-5)


def test_lambda_u(nelsen16_copula):
    """Test upper tail dependence coefficient (lambda_U)."""
    # For Nelsen16, lambda_U = 0
    assert float(nelsen16_copula.lambda_U()) == 0


def test_theta_dependent_behavior():
    """Test how behavior changes with different theta values."""
    # Create copulas with different theta values
    copula1 = Nelsen16(0.5)
    copula2 = Nelsen16(1)
    copula3 = Nelsen16(2)

    # Test at a specific point
    u, v = 0.3, 0.4

    # Calculate CDF values
    cdf1 = float(copula1.cdf(u, v))
    cdf2 = float(copula2.cdf(u, v))
    cdf3 = float(copula3.cdf(u, v))

    # For Nelsen16, different theta should give different CDF values
    # Just verify they are different (the specific ordering depends on u,v)
    assert cdf1 != cdf2 or cdf2 != cdf3

    # Verify they are all in valid range
    assert 0 <= cdf1 <= 1
    assert 0 <= cdf2 <= 1
    assert 0 <= cdf3 <= 1


def test_lower_frechet_case():
    """Test behavior at the special case (theta=0)."""
    # Create a LowerFrechet instance directly
    direct_instance = LowerFrechet()

    # Get an instance via Nelsen16 special case
    special_case = Nelsen16.create(0)

    # Test at some points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2)]

    for u, v in points:
        direct_result = float(direct_instance.cdf(u, v))
        special_result = float(special_case.cdf(u, v))
        assert np.isclose(direct_result, special_result, rtol=1e-5)


def test_large_theta_behavior():
    """Test behavior with a very large theta value."""
    # With large theta, check that computations remain stable
    large_theta = Nelsen16(50)

    # Test at some points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2)]

    for u, v in points:
        result = float(large_theta.cdf(u, v))

        # Result should be within [0,1]
        assert 0 <= result <= 1

        # For very large theta, the copula has a specific behavior
        # We verify the result is well-defined and matches the formula
        theta = 50
        term1 = u * v * (1 - theta)
        term2 = u * (theta + v) * (v - 1)
        term3 = v * (theta + u) * (u - 1)
        term4 = np.sqrt(4 * theta * u**2 * v**2 + (term1 + term2 + term3) ** 2)
        expected = (term1 + term2 + term3 + term4) / (2 * u * v)
        assert np.isclose(result, expected, rtol=1e-5)


def test_numerical_stability():
    """Test numerical stability at challenging points."""
    copula = Nelsen16(1)

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
            assert result < 0.1

        # For u close to 1, result should be close to v
        if u > 0.99:
            assert np.isclose(result, v, rtol=1e-2)

        # For v close to 1, result should be close to u
        if v > 0.99:
            assert np.isclose(result, u, rtol=1e-2)
