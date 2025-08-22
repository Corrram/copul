import numpy as np
import pytest

from copul.families.archimedean.nelsen6 import Joe
from copul.families.other.biv_independence_copula import BivIndependenceCopula


@pytest.fixture
def joe_copula():
    """Fixture providing a Joe copula with theta=2."""
    return Joe(2)


def test_special_case_independence():
    """Test that theta=1 creates an IndependenceCopula."""
    # Test direct instantiation
    copula = Joe(1)
    assert isinstance(copula, BivIndependenceCopula)

    # Test via create factory method
    copula_via_create = Joe.create(1)
    assert isinstance(copula_via_create, BivIndependenceCopula)

    # Test via call method
    base_copula = Joe(2)
    new_copula = base_copula(1)
    assert isinstance(new_copula, BivIndependenceCopula)


def test_parameter_validation():
    """Test parameter validation for the Joe copula."""
    # Valid values should not raise errors
    Joe(1)  # Lower bound
    Joe(2)  # Interior point
    Joe(10)  # Larger value

    # Values below lower bound should raise ValueError
    with pytest.raises(ValueError, match="must be >= 1"):
        Joe(0.5)

    # Negative values should raise ValueError
    with pytest.raises(ValueError, match="must be >= 1"):
        Joe(-1)


def test_generator_function(joe_copula):
    """Test the generator function of the Joe copula."""
    # Test generator at specific t values
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    for t in t_values:
        # Manual calculation using the formula: -log(1 - (1 - t)^theta)
        theta = 2
        expected = -np.log(1 - (1 - t) ** theta)
        actual = float(joe_copula.generator(t))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_inverse_generator_function(joe_copula):
    """Test the inverse generator function of the Joe copula."""
    # Test inverse generator at specific y values
    y_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    for y in y_values:
        # Manual calculation using the formula: 1 - (1 - exp(-y))^(1/theta)
        theta = 2
        expected = 1 - (1 - np.exp(-y)) ** (1 / theta)
        actual = float(joe_copula.inv_generator(y))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_cdf_function(joe_copula):
    """Test the CDF function of the Joe copula."""
    # Test CDF at specific points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2), (0.9, 0.8)]

    for u, v in points:
        # Manual calculation using the formula
        theta = 2
        expected = 1 - (-(((1 - u) ** theta - 1) * ((1 - v) ** theta - 1)) + 1) ** (
            1 / theta
        )
        actual = float(joe_copula.cdf(u, v))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_cdf_boundary_conditions(joe_copula):
    """Test boundary conditions for the Joe copula CDF."""
    # Test points along the boundaries
    u_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # C(u,0) = 0 for any u in [0,1]
    for u in u_values:
        assert np.isclose(float(joe_copula.cdf(u, 0)), 0, atol=1e-10)
        assert np.isclose(float(joe_copula.cdf(0, u)), 0, atol=1e-10)

    # C(u,1) = u and C(1,v) = v for any u,v in [0,1]
    for u in u_values:
        assert np.isclose(float(joe_copula.cdf(u, 1)), u, rtol=1e-5)
        assert np.isclose(float(joe_copula.cdf(1, u)), u, rtol=1e-5)


def test_conditional_distribution_1(joe_copula):
    """Test the first conditional distribution function."""
    # Test conditional distribution at specific points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2), (0.9, 0.8)]

    for u, v in points:
        # Test that result is between 0 and 1
        result = float(joe_copula.cond_distr_1(u, v))
        assert 0 <= result <= 1

        # Additional test for theta=1 (independence case)
        independence = BivIndependenceCopula()
        # For IndependenceCopula, the conditional distribution should equal v
        assert np.isclose(float(independence.cond_distr_1(u, v)), v, rtol=1e-5)


def test_conditional_distribution_2(joe_copula):
    """Test the second conditional distribution function."""
    # Test conditional distribution at specific points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2), (0.9, 0.8)]

    for u, v in points:
        # Test that result is between 0 and 1
        result = float(joe_copula.cond_distr_2(u, v))
        assert 0 <= result <= 1

        # Additional test for theta=1 (independence case)
        independence = BivIndependenceCopula()
        # For IndependenceCopula, the conditional distribution should equal u
        assert np.isclose(float(independence.cond_distr_2(u, v)), u, rtol=1e-5)


def test_lambda_l(joe_copula):
    """Test lower tail dependence coefficient (lambda_L)."""
    # Joe copula has no lower tail dependence
    assert float(joe_copula.lambda_L()) == 0


def test_lambda_u(joe_copula):
    """Test upper tail dependence coefficient (lambda_U)."""
    # For Joe copula, lambda_U = 2 - 2^(1/theta)
    theta = 2
    expected = 2 - 2 ** (1 / theta)
    actual = float(joe_copula.lambda_U())
    assert np.isclose(actual, expected, rtol=1e-5)

    # Test with different theta values
    for theta in [1.5, 3, 5]:
        copula = Joe(theta)
        expected = 2 - 2 ** (1 / theta)
        actual = float(copula.lambda_U())
        assert np.isclose(actual, expected, rtol=1e-5)

    # As theta approaches infinity, lambda_U approaches 1
    large_theta = Joe(100)
    assert np.isclose(float(large_theta.lambda_U()), 1, rtol=1e-2)


def test_is_absolutely_continuous():
    """Test the is_absolutely_continuous property."""
    # Joe copula is absolutely continuous for all valid theta
    for theta in [1, 2, 5, 10]:
        if theta == 1:
            # Special case returns IndependenceCopula
            copula = Joe.create(theta)
        else:
            copula = Joe(theta)
        assert copula.is_absolutely_continuous
