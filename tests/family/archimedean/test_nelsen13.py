import numpy as np
import pytest

from copul.family.archimedean.nelsen13 import Nelsen13
from copul.family.frechet.biv_independence_copula import BivIndependenceCopula


@pytest.fixture
def nelsen13_copula():
    """Fixture providing a Nelsen13 copula with theta=1."""
    return Nelsen13(1)


def test_special_cases():
    """Test that special cases return appropriate copula instances."""
    # Test theta = 0 returns IndependenceCopula
    independence = Nelsen13(0)
    assert isinstance(independence, BivIndependenceCopula)

    # Test via create factory method
    independence_create = Nelsen13.create(0)
    assert isinstance(independence_create, BivIndependenceCopula)

    # Test via call method
    base_copula = Nelsen13(1)
    independence_call = base_copula(0)
    assert isinstance(independence_call, BivIndependenceCopula)


def test_parameter_validation():
    """Test parameter validation for the Nelsen13 copula."""
    # Valid values should not raise errors
    Nelsen13(0)  # Lower bound
    Nelsen13(1)  # Interior point
    Nelsen13(10)  # Larger value

    # Negative values should raise ValueError
    with pytest.raises(ValueError, match="must be >= 0"):
        Nelsen13(-0.1)


def test_is_absolutely_continuous(nelsen13_copula):
    """Test the is_absolutely_continuous property."""
    # Nelsen13 is absolutely continuous
    assert nelsen13_copula.is_absolutely_continuous


def test_generator_function(nelsen13_copula):
    """Test the generator function of the Nelsen13 copula."""
    # Test generator at specific t values
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    for t in t_values:
        # Manual calculation using the formula: (1 - log(t))^theta - 1
        theta = 1
        expected = (1 - np.log(t)) ** theta - 1
        actual = float(nelsen13_copula.generator(t))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_inverse_generator_function(nelsen13_copula):
    """Test the inverse generator function of the Nelsen13 copula."""
    # Test inverse generator at specific y values
    y_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    for y in y_values:
        # Manual calculation using the formula: exp(1 - (y + 1)^(1/theta))
        theta = 1
        expected = np.exp(1 - (y + 1) ** (1 / theta))
        actual = float(nelsen13_copula.inv_generator(y))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_cdf_function(nelsen13_copula):
    """Test the CDF function of the Nelsen13 copula."""
    # Test CDF at specific points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2), (0.9, 0.8)]

    for u, v in points:
        # Manual calculation using the formula
        theta = 1
        term = ((1 - np.log(u)) ** theta + (1 - np.log(v)) ** theta - 1) ** (1 / theta)
        expected = np.exp(1 - term)
        actual = float(nelsen13_copula.cdf(u, v))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_cdf_boundary_conditions(nelsen13_copula):
    """Test boundary conditions for the Nelsen13 copula CDF."""
    # Test points along the boundaries
    u_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # C(u,0) = 0 for any u in [0,1]
    for u in u_values:
        assert np.isclose(float(nelsen13_copula.cdf(u, 0)), 0, atol=1e-10)
        assert np.isclose(float(nelsen13_copula.cdf(0, u)), 0, atol=1e-10)

    # C(u,1) = u and C(1,v) = v for any u,v in [0,1]
    for u in u_values:
        assert np.isclose(float(nelsen13_copula.cdf(u, 1)), u, rtol=1e-5)
        assert np.isclose(float(nelsen13_copula.cdf(1, u)), u, rtol=1e-5)


def test_lambda_l(nelsen13_copula):
    """Test lower tail dependence coefficient (lambda_L)."""
    # For Nelsen13, lambda_L = 0
    assert float(nelsen13_copula.lambda_L()) == 0


def test_lambda_u(nelsen13_copula):
    """Test upper tail dependence coefficient (lambda_U)."""
    # For Nelsen13, lambda_U = 0
    assert float(nelsen13_copula.lambda_U()) == 0


def test_theta_dependent_behavior():
    """Test how behavior changes with different theta values."""
    # Create copulas with different theta values
    copula1 = Nelsen13(0.5)
    copula2 = Nelsen13(1)
    copula3 = Nelsen13(2)

    # Test at a specific point
    u, v = 0.3, 0.4

    # Calculate CDF values
    cdf1 = float(copula1.cdf(u, v))
    cdf2 = float(copula2.cdf(u, v))
    cdf3 = float(copula3.cdf(u, v))

    # For Nelsen13, different theta should give different CDF values
    # Just verify they are different (the specific ordering depends on u,v)
    assert cdf1 != cdf2 or cdf2 != cdf3

    # Verify they are all in valid range
    assert 0 <= cdf1 <= 1
    assert 0 <= cdf2 <= 1
    assert 0 <= cdf3 <= 1


def test_independence_case():
    """Test that theta=0 behaves like an independence copula."""
    # Get an independence copula via the special case
    independence = Nelsen13(0)

    # Test CDF at various points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2), (0.9, 0.8)]

    for u, v in points:
        # For independence copula, C(u,v) = u*v
        expected = u * v
        actual = float(independence.cdf(u, v))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_large_theta_behavior():
    """Test behavior with a very large theta value."""
    # With large theta, check that computations remain stable
    large_theta = Nelsen13(50)

    # Test at some points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2)]

    for u, v in points:
        result = float(large_theta.cdf(u, v))

        # Result should be within [0,1]
        assert 0 <= result <= 1

        # For very large theta, the copula approaches a specific limiting form
        # We just verify it's a valid value without specific expectations
        theta = 50
        term = ((1 - np.log(u)) ** theta + (1 - np.log(v)) ** theta - 1) ** (1 / theta)
        expected = np.exp(1 - term)
        assert np.isclose(result, expected, rtol=1e-5)


def test_numerical_stability():
    """Test numerical stability at challenging points."""
    copula = Nelsen13(1)

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


def test_nelsen13_lower_orthant_ordered():
    nelsen = Nelsen13(0.5)
    nelsen2 = Nelsen13(1.5)

    def func(u, v):
        return (nelsen.cdf(u, v) - nelsen2.cdf(u, v)).evalf()

    linspace = np.linspace(0.01, 0.99, 10)
    grid2d = np.meshgrid(linspace, linspace)
    values = np.array(
        [func(u, v) for u, v in zip(grid2d[0].flatten(), grid2d[1].flatten())]
    )
    assert np.all(values <= 0)
