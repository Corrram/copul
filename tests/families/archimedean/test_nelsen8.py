import numpy as np
import pytest

from copul.families.archimedean.nelsen8 import Nelsen8
from copul.families.other.lower_frechet import LowerFrechet


@pytest.fixture
def nelsen8_copula():
    """Fixture providing a Nelsen8 copula with theta=2."""
    return Nelsen8(2)


def test_special_cases():
    """Test that special cases return appropriate copula instances."""
    # Test theta = 1 returns LowerFrechet
    lower_frechet = Nelsen8(1)
    assert isinstance(lower_frechet, LowerFrechet)

    # Test via create factory method
    lower_frechet_create = Nelsen8.create(1)
    assert isinstance(lower_frechet_create, LowerFrechet)

    # Test via call method
    base_copula = Nelsen8(2)
    lower_frechet_call = base_copula(1)
    assert isinstance(lower_frechet_call, LowerFrechet)


def test_parameter_validation():
    """Test parameter validation for the Nelsen8 copula."""
    # Valid values should not raise errors
    Nelsen8(1)  # Lower bound
    Nelsen8(2)  # Interior point
    Nelsen8(10)  # Larger value

    # Values below lower bound should raise ValueError
    with pytest.raises(ValueError, match="must be >= 1"):
        Nelsen8(0.5)

    # Negative values should raise ValueError
    with pytest.raises(ValueError, match="must be >= 1"):
        Nelsen8(-1)


def test_is_absolutely_continuous(nelsen8_copula):
    """Test the is_absolutely_continuous property."""
    # Nelsen8 is not absolutely continuous
    assert not nelsen8_copula.is_absolutely_continuous


def test_generator_function(nelsen8_copula):
    """Test the generator function of the Nelsen8 copula."""
    # Test generator at specific t values
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    for t in t_values:
        # Manual calculation using the formula: (1 - t) / (1 + (theta - 1) * t)
        theta = 2
        expected = (1 - t) / (1 + (theta - 1) * t)
        actual = float(nelsen8_copula.generator(t))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_inverse_generator_function(nelsen8_copula):
    """Test the inverse generator function of the Nelsen8 copula."""
    # Test inverse generator at specific y values
    y_values = [0.1, 0.5, 0.9, 1.0]

    for y in y_values:
        # The inverse generator has a piecewise function condition
        theta = 2
        if y <= 1:
            expected = (1 - y) / (theta * y - y + 1)
        else:
            expected = 0

        actual = float(nelsen8_copula.inv_generator(y))
        assert np.isclose(actual, expected, rtol=1e-5)

    # Test y > 1 case
    y_greater_than_1 = 1.5
    assert float(nelsen8_copula.inv_generator(y_greater_than_1)) == 0


def test_cdf_function(nelsen8_copula):
    """Test the CDF function of the Nelsen8 copula."""
    # Test CDF at specific points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2), (0.9, 0.8)]

    for u, v in points:
        # Manual calculation using the formula
        theta = 2
        num = theta**2 * u * v - (1 - u) * (1 - v)
        den = theta**2 - (theta - 1) ** 2 * (1 - u) * (1 - v)
        expected = max(num / den, 0)
        actual = float(nelsen8_copula.cdf(u, v))
        assert np.isclose(actual, expected, rtol=1e-5)

    # Test with points that might yield negative values before Max is applied
    # Compute some edge cases where the num could be negative
    # These are just examples that might need adjustment
    edge_points = [(0.01, 0.01), (0.1, 0.1), (0.05, 0.1)]

    for u, v in edge_points:
        theta = 2
        num = theta**2 * u * v - (1 - u) * (1 - v)
        den = theta**2 - (theta - 1) ** 2 * (1 - u) * (1 - v)
        raw_value = num / den
        actual = float(nelsen8_copula.cdf(u, v))

        # If raw value is negative, result should be 0
        if raw_value < 0:
            assert actual == 0
        else:
            assert np.isclose(actual, raw_value, rtol=1e-5)


def test_cdf_boundary_conditions(nelsen8_copula):
    """Test boundary conditions for the Nelsen8 copula CDF."""
    # Test points along the boundaries
    u_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # C(u,0) = 0 for any u in [0,1]
    for u in u_values:
        assert np.isclose(float(nelsen8_copula.cdf(u, 0)), 0, atol=1e-10)
        assert np.isclose(float(nelsen8_copula.cdf(0, u)), 0, atol=1e-10)

    # C(u,1) = u and C(1,v) = v for any u,v in [0,1]
    for u in u_values:
        assert np.isclose(float(nelsen8_copula.cdf(u, 1)), u, rtol=1e-5)
        assert np.isclose(float(nelsen8_copula.cdf(1, u)), u, rtol=1e-5)


def test_conditional_distribution_1(nelsen8_copula):
    """Test the first conditional distribution function."""
    # Test conditional distribution at specific points
    points = [(0.7, 0.8), (0.8, 0.8), (0.9, 0.9)]

    for u, v in points:
        # Just check the result is within valid range
        result = float(nelsen8_copula.cond_distr_1(u, v))
        assert 0 <= result <= 1


def test_squared_cond_distr_1(nelsen8_copula):
    """Test the _squared_cond_distr_1 method."""
    # Test at specific points
    v, u = 0.8, 0.7

    # Since this is a complex expression used in calculations,
    # we'll just verify it returns a value without error
    # and that the value is non-negative (as it's squared)
    result = float(nelsen8_copula._squared_cond_distr_1(v, u))
    assert result >= 0


def test_lambda_l(nelsen8_copula):
    """Test lower tail dependence coefficient (lambda_L)."""
    # Nelsen8 has no lower tail dependence
    assert float(nelsen8_copula.lambda_L()) == 0


def test_lambda_u(nelsen8_copula):
    """Test upper tail dependence coefficient (lambda_U)."""
    # Nelsen8 has no upper tail dependence
    assert float(nelsen8_copula.lambda_U()) == 0


def test_theta_dependent_behavior():
    """Test how behavior changes with different theta values."""
    # Create copulas with different theta values
    copula1 = Nelsen8(2)
    copula2 = Nelsen8(5)
    copula3 = Nelsen8(10)

    # Test at a specific point
    u, v = 0.5, 0.5

    # For Nelsen8, larger theta should generally produce larger CDF values at the same (u,v)
    cdf1 = float(copula1.cdf(u, v))
    cdf2 = float(copula2.cdf(u, v))
    cdf3 = float(copula3.cdf(u, v))

    # As theta increases, the CDF should increase (or at least not decrease)
    # This might not be strictly true for all copulas, but works for Nelsen8
    assert cdf1 <= cdf2 <= cdf3


def test_nelsen8_with_large_theta():
    """Test Nelsen8 copula with a very large theta value."""
    # With very large theta, the copula should approach some limiting behavior
    large_theta = Nelsen8(1000)

    # Test CDF at a point
    u, v = 0.5, 0.5
    result = float(large_theta.cdf(u, v))

    # The result should be well-defined and within [0,1]
    assert 0 <= result <= 1

    # For Nelsen8, as theta gets very large, CDF(0.5, 0.5) should approach 1/3
    # (the value of PiOverSigmaMinusPi at this point).
    # This is a property of this specific copula
    assert np.isclose(result, 1 / 3, rtol=1e-2)
