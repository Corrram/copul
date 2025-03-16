import numpy as np
import pytest
import sympy

from copul.families.archimedean.nelsen10 import Nelsen10
from copul.families.other.biv_independence_copula import BivIndependenceCopula


@pytest.fixture
def nelsen10_copula():
    """Fixture providing a Nelsen10 copula with theta=0.5."""
    return Nelsen10(0.5)


def test_special_cases():
    """Test that special cases return appropriate copula instances."""
    # Test theta = 0 returns IndependenceCopula
    independence = Nelsen10(0)
    assert isinstance(independence, BivIndependenceCopula)

    # Test via create factory method
    independence_create = Nelsen10.create(0)
    assert isinstance(independence_create, BivIndependenceCopula)

    # Test via call method
    base_copula = Nelsen10(0.5)
    independence_call = base_copula(0)
    assert isinstance(independence_call, BivIndependenceCopula)


def test_parameter_validation():
    """Test parameter validation for the Nelsen10 copula."""
    # Valid values should not raise errors
    Nelsen10(0)  # Lower bound
    Nelsen10(0.5)  # Interior point
    Nelsen10(1)  # Upper bound

    # Values below lower bound should raise ValueError
    with pytest.raises(ValueError, match="must be >= 0"):
        Nelsen10(-0.1)

    # Values above upper bound should raise ValueError
    with pytest.raises(ValueError, match="must be <= 1"):
        Nelsen10(1.1)


def test_is_absolutely_continuous(nelsen10_copula):
    """Test the is_absolutely_continuous property."""
    # Nelsen10 is absolutely continuous
    assert nelsen10_copula.is_absolutely_continuous


def test_generator_function(nelsen10_copula):
    """Test the generator function of the Nelsen10 copula."""
    # Test generator at specific t values
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    for t in t_values:
        # Manual calculation using the formula: log(2 * t^(-theta) - 1)
        theta = 0.5
        expected = np.log(2 * t ** (-theta) - 1)
        actual = float(nelsen10_copula.generator(t))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_inverse_generator_function(nelsen10_copula):
    """Test the inverse generator function of the Nelsen10 copula."""
    # Test inverse generator at specific y values
    y_values = [0.1, 0.5, 1.0, 2.0]

    for y in y_values:
        # Manual calculation
        theta = 0.5
        expected = (2 / (np.exp(y) + 1)) ** (1 / theta)
        actual = float(nelsen10_copula.inv_generator(y))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_cdf_function(nelsen10_copula):
    """Test the CDF function of the Nelsen10 copula."""
    # Test CDF at specific points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2), (0.9, 0.8)]

    for u, v in points:
        # Manual calculation using the formula
        theta = 0.5
        numerator = 2 * u**theta * v**theta
        denominator = u**theta * v**theta + (u**theta - 2) * (v**theta - 2)
        expected = (numerator / denominator) ** (1 / theta)
        actual = float(nelsen10_copula.cdf(u, v))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_cdf_boundary_conditions(nelsen10_copula):
    """Test boundary conditions for the Nelsen10 copula CDF."""
    # Test points along the boundaries
    u_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # C(u,0) = 0 for any u in [0,1]
    for u in u_values:
        assert np.isclose(float(nelsen10_copula.cdf(u, 0)), 0, atol=1e-10)
        assert np.isclose(float(nelsen10_copula.cdf(0, u)), 0, atol=1e-10)

    # C(u,1) = u and C(1,v) = v for any u,v in [0,1]
    for u in u_values:
        assert np.isclose(float(nelsen10_copula.cdf(u, 1)), u, rtol=1e-5)
        assert np.isclose(float(nelsen10_copula.cdf(1, u)), u, rtol=1e-5)


def test_cdf_theta_effect():
    """Test the effect of theta on the CDF values."""
    # Create copulas with different theta values
    copula_small = Nelsen10(0.2)
    copula_medium = Nelsen10(0.5)
    copula_large = Nelsen10(0.8)

    # Test at specific point
    u, v = 0.5, 0.5

    # Compute CDF values
    cdf_small = float(copula_small.cdf(u, v))
    cdf_medium = float(copula_medium.cdf(u, v))
    cdf_large = float(copula_large.cdf(u, v))

    # For Nelsen10, the CDF values change with theta
    # Verify they are different (this is a less strict test)
    assert cdf_small != cdf_medium
    assert cdf_medium != cdf_large


def test_independence_case():
    """Test that theta=0 behaves like an independence copula."""
    # Get an independence copula via the special case
    independence = Nelsen10(0)

    # Test CDF at various points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2), (0.9, 0.8)]

    for u, v in points:
        # For independence copula, C(u,v) = u*v
        expected = u * v
        actual = float(independence.cdf(u, v))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_limiting_case():
    """Test the limiting case as theta approaches 1."""
    # Create a copula with theta close to 1
    copula_limit = Nelsen10(0.999)

    # Test at some points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2)]

    for u, v in points:
        # Calculate the CDF
        result = float(copula_limit.cdf(u, v))

        # Result should be within [0,1]
        assert 0 <= result <= 1

        # As theta approaches 1, the copula has a specific behavior
        # We verify the result is well-defined and within bounds
        theta = 0.999
        numerator = 2 * u**theta * v**theta
        denominator = u**theta * v**theta + (u**theta - 2) * (v**theta - 2)
        expected = (numerator / denominator) ** (1 / theta)
        assert np.isclose(result, expected, rtol=1e-5)


def test_inherited_methods():
    """Test that inherited methods work correctly."""
    copula = Nelsen10(0.5)

    # Test that common methods exist and return values
    assert hasattr(copula, "lambda_L")
    assert hasattr(copula, "lambda_U")

    # These methods should return real numbers or sympy expressions
    lambda_l = copula.lambda_L()
    lambda_u = copula.lambda_U()

    # Check we can convert to float if numerical, or that they're sympy expressions
    try:
        float(lambda_l)
    except TypeError:
        assert isinstance(lambda_l, sympy.Expr)

    try:
        float(lambda_u)
    except TypeError:
        assert isinstance(lambda_u, sympy.Expr)


def test_numerical_stability():
    """Test numerical stability at challenging points."""
    copula = Nelsen10(0.5)

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

        # Test the basic boundary conditions
        if u < 0.01:  # Very close to 0
            assert result < 0.01

        if v < 0.01:  # Very close to 0
            assert result < 0.01

        if u > 0.99:  # Very close to 1
            assert np.isclose(result, v, rtol=1e-2)

        if v > 0.99:  # Very close to 1
            assert np.isclose(result, u, rtol=1e-2)
