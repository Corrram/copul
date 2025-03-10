import numpy as np
import pytest

from copul.families.archimedean.nelsen15 import GenestGhoudi, Nelsen15
from copul.families.other.lower_frechet import LowerFrechet


@pytest.fixture
def genest_ghoudi():
    """Fixture providing a GenestGhoudi copula with theta=2."""
    return GenestGhoudi(2)


def test_special_cases():
    """Test that special cases return appropriate copula instances."""
    # Test theta = 1 returns LowerFrechet
    lower_frechet = GenestGhoudi(1)
    assert isinstance(lower_frechet, LowerFrechet)

    # Test via create factory method
    lower_frechet_create = GenestGhoudi.create(1)
    assert isinstance(lower_frechet_create, LowerFrechet)

    # Test via call method
    base_copula = GenestGhoudi(2)
    lower_frechet_call = base_copula(1)
    assert isinstance(lower_frechet_call, LowerFrechet)

    # Also verify Nelsen15 alias works the same
    nelsen15 = Nelsen15(1)
    assert isinstance(nelsen15, LowerFrechet)


def test_parameter_validation():
    """Test parameter validation for the GenestGhoudi copula."""
    # Valid values should not raise errors
    GenestGhoudi(1)  # Lower bound
    GenestGhoudi(2)  # Interior point
    GenestGhoudi(10)  # Larger value

    # Values below lower bound should raise ValueError
    with pytest.raises(ValueError, match="must be >= 1"):
        GenestGhoudi(0.5)

    # Negative values should raise ValueError
    with pytest.raises(ValueError, match="must be >= 1"):
        GenestGhoudi(-1)


def test_is_absolutely_continuous(genest_ghoudi):
    """Test the is_absolutely_continuous property."""
    # GenestGhoudi is not absolutely continuous
    assert not genest_ghoudi.is_absolutely_continuous


def test_generator_function(genest_ghoudi):
    """Test the generator function of the GenestGhoudi copula."""
    # Test generator at specific t values
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    for t in t_values:
        # Manual calculation using the formula: (1 - t^(1/theta))^theta
        theta = 2
        expected = (1 - t ** (1 / theta)) ** theta
        actual = float(genest_ghoudi.generator(t))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_inverse_generator_function(genest_ghoudi):
    """Test the inverse generator function of the GenestGhoudi copula."""
    # Test inverse generator at specific y values
    y_values = [0.1, 0.5, 0.9, 1.0]

    for y in y_values:
        # The inverse generator has a piecewise function condition
        theta = 2
        if y <= 1:
            expected = (1 - y ** (1 / theta)) ** theta
        else:
            expected = 0

        actual = float(genest_ghoudi.inv_generator(y))
        assert np.isclose(actual, expected, rtol=1e-5)

    # Test y > 1 case
    y_greater_than_1 = 1.5
    assert float(genest_ghoudi.inv_generator(y_greater_than_1)) == 0


def test_cdf_function(genest_ghoudi):
    """Test the CDF function of the GenestGhoudi copula."""
    # Test CDF at specific points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2), (0.9, 0.8)]

    for u, v in points:
        # Manual calculation using the formula
        theta = 2
        term = ((1 - u ** (1 / theta)) ** theta + (1 - v ** (1 / theta)) ** theta) ** (
            1 / theta
        )
        expected = max(0, (1 - term) ** theta)
        actual = float(genest_ghoudi.cdf(u, v))
        assert np.isclose(actual, expected, rtol=1e-3)

    # Test with points that might yield 0 due to the Max function
    edge_points = [(0.01, 0.01), (0.1, 0.1), (0.05, 0.1)]

    for u, v in edge_points:
        theta = 2
        term = ((1 - u ** (1 / theta)) ** theta + (1 - v ** (1 / theta)) ** theta) ** (
            1 / theta
        )
        raw_value = (1 - term) ** theta
        actual = float(genest_ghoudi.cdf(u, v))

        # If raw value is negative, result should be 0
        if term > 1:
            assert actual == 0
        else:
            assert np.isclose(actual, raw_value, rtol=1e-3)


def test_cdf_boundary_conditions(genest_ghoudi):
    """Test boundary conditions for the GenestGhoudi copula CDF."""
    # Test points along the boundaries
    u_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # C(u,0) = 0 for any u in [0,1]
    for u in u_values:
        assert np.isclose(float(genest_ghoudi.cdf(u, 0)), 0, atol=1e-10)
        assert np.isclose(float(genest_ghoudi.cdf(0, u)), 0, atol=1e-10)

    # C(u,1) = u and C(1,v) = v for any u,v in [0,1]
    for u in u_values:
        assert np.isclose(float(genest_ghoudi.cdf(u, 1)), u, rtol=1e-5)
        assert np.isclose(float(genest_ghoudi.cdf(1, u)), u, rtol=1e-5)


def test_lambda_l(genest_ghoudi):
    """Test lower tail dependence coefficient (lambda_L)."""
    # For GenestGhoudi, lambda_L = 0
    assert float(genest_ghoudi.lambda_L()) == 0


def test_lambda_u(genest_ghoudi):
    """Test upper tail dependence coefficient (lambda_U)."""
    # For GenestGhoudi, lambda_U = 2 - 2^(1/theta)
    theta = 2
    expected = 2 - 2 ** (1 / theta)
    actual = float(genest_ghoudi.lambda_U())
    assert np.isclose(actual, expected, rtol=1e-5)

    # Test with different theta values
    for theta in [1.5, 3, 5]:
        copula = GenestGhoudi(theta)
        expected = 2 - 2 ** (1 / theta)
        actual = float(copula.lambda_U())
        assert np.isclose(actual, expected, rtol=1e-5)


def test_theta_dependent_behavior():
    """Test how behavior changes with different theta values."""
    # Create copulas with different theta values
    copula1 = GenestGhoudi(1.5)
    copula2 = GenestGhoudi(3)
    copula3 = GenestGhoudi(5)

    # Test at a specific point
    u, v = 0.4, 0.6

    # Calculate CDF values
    cdf1 = float(copula1.cdf(u, v))
    cdf2 = float(copula2.cdf(u, v))
    cdf3 = float(copula3.cdf(u, v))

    # For GenestGhoudi, different theta should give different CDF values
    # Verify they are different
    assert cdf1 != cdf2
    assert cdf2 != cdf3

    # Verify they are all in valid range
    assert 0 <= cdf1 <= 1
    assert 0 <= cdf2 <= 1
    assert 0 <= cdf3 <= 1


def test_lower_frechet_case():
    """Test behavior at the special case (theta=1)."""
    # Create a LowerFrechet instance directly
    direct_instance = LowerFrechet()

    # Get an instance via GenestGhoudi special case
    special_case = GenestGhoudi.create(1)

    # Test at some points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2)]

    for u, v in points:
        direct_result = float(direct_instance.cdf(u, v))
        special_result = float(special_case.cdf(u, v))
        assert np.isclose(direct_result, special_result, rtol=1e-5)


def test_large_theta_behavior():
    """Test behavior with a very large theta value."""
    # With large theta, the copula should approach a specific form
    large_theta = GenestGhoudi(50)

    # Test at some points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2)]

    for u, v in points:
        result = float(large_theta.cdf(u, v))

        # Result should be within [0,1]
        assert 0 <= result <= 1

        # For very large theta, calculate the expected value
        theta = 50
        term = ((1 - u ** (1 / theta)) ** theta + (1 - v ** (1 / theta)) ** theta) ** (
            1 / theta
        )
        expected = max(0, (1 - term) ** theta)
        assert np.isclose(result, expected, rtol=1e-4)  # Use a bit more tolerance


def test_numerical_stability():
    """Test numerical stability at challenging points."""
    copula = GenestGhoudi(2)

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
