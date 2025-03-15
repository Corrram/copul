from unittest.mock import patch

import numpy as np
import pytest
import sympy

from copul.families.archimedean.nelsen11 import Nelsen11
from copul.families.other.independence_copula import IndependenceCopula


@pytest.fixture
def nelsen11_copula():
    """Fixture providing a Nelsen11 copula with theta=0.3."""
    return Nelsen11(0.3)


def test_special_cases():
    """Test that special cases return appropriate copula instances."""
    # Test theta = 0 returns IndependenceCopula
    independence = Nelsen11(0)
    assert isinstance(independence, IndependenceCopula)

    # Test via create factory method
    independence_create = Nelsen11.create(0)
    assert isinstance(independence_create, IndependenceCopula)

    # Test via call method
    base_copula = Nelsen11(0.3)
    independence_call = base_copula(0)
    assert isinstance(independence_call, IndependenceCopula)


def test_parameter_validation():
    """Test parameter validation for the Nelsen11 copula."""
    # Valid values should not raise errors
    Nelsen11(0)  # Lower bound
    Nelsen11(0.3)  # Interior point
    Nelsen11(0.5)  # Upper bound

    # Values below lower bound should raise ValueError
    with pytest.raises(ValueError, match="must be >= 0"):
        Nelsen11(-0.1)

    # Values above upper bound should raise ValueError
    with pytest.raises(ValueError, match="must be <= 0.5"):
        Nelsen11(0.6)


def test_is_absolutely_continuous(nelsen11_copula):
    """Test the is_absolutely_continuous property."""
    # Nelsen11 is not absolutely continuous
    assert not nelsen11_copula.is_absolutely_continuous


def test_generator_function(nelsen11_copula):
    """Test the generator function of the Nelsen11 copula."""
    # Test generator at specific t values
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    for t in t_values:
        # Manual calculation using the formula: log(2 - t^theta)
        theta = 0.3
        expected = np.log(2 - t**theta)
        actual = float(nelsen11_copula.generator(t))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_inverse_generator_function(nelsen11_copula):
    """Test the inverse generator function of the Nelsen11 copula."""
    # Test inverse generator at specific y values
    y_values = [0.1, 0.5, 0.69]  # All less than log(2) ≈ 0.693

    for y in y_values:
        # Manual calculation for y <= log(2)
        theta = 0.3
        expected = (2 - np.exp(y)) ** (1 / theta)
        actual = float(nelsen11_copula.inv_generator(y))
        assert np.isclose(actual, expected, rtol=1e-5)

    # Test y > log(2) case
    y_greater_than_log2 = 0.7  # Greater than log(2) ≈ 0.693
    assert float(nelsen11_copula.inv_generator(y_greater_than_log2)) == 0

def test_nelsen11_inverse_generator_at_infinity():
    """Test the inverse generator function of Nelsen3."""
    copula = Nelsen11()
    y = sympy.log(2)
    inv_gen = copula.inv_generator(y=y)
    actual = float(inv_gen)
    assert np.isclose(actual, 0, rtol=1e-10)


def test_cdf_function(nelsen11_copula):
    """Test the CDF function of the Nelsen11 copula."""
    # Test CDF at specific points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2), (0.9, 0.8)]

    for u, v in points:
        # Manual calculation using the formula
        theta = 0.3
        term = u**theta * v**theta - 2 * (1 - u**theta) * (1 - v**theta)
        if term > 0:
            expected = term ** (1 / theta)
        else:
            expected = 0
        actual = float(nelsen11_copula.cdf(u, v))
        assert np.isclose(actual, expected, rtol=1e-5)

    # Test with points that might yield 0 due to the Max function
    # These are just examples that might yield negative values before Max is applied
    edge_points = [(0.1, 0.1), (0.2, 0.1), (0.1, 0.2)]

    for u, v in edge_points:
        theta = 0.3
        term = u**theta * v**theta - 2 * (1 - u**theta) * (1 - v**theta)
        actual = float(nelsen11_copula.cdf(u, v))

        # If term is negative, result should be 0
        if term < 0:
            assert actual == 0
        else:
            expected = term ** (1 / theta)
            assert np.isclose(actual, expected, rtol=1e-5)


def test_cdf_boundary_conditions(nelsen11_copula):
    """Test boundary conditions for the Nelsen11 copula CDF."""
    # Test points along the boundaries
    u_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # C(u,0) = 0 for any u in [0,1]
    for u in u_values:
        assert np.isclose(float(nelsen11_copula.cdf(u, 0)), 0, atol=1e-10)
        assert np.isclose(float(nelsen11_copula.cdf(0, u)), 0, atol=1e-10)

    # C(u,1) = u and C(1,v) = v for any u,v in [0,1]
    for u in u_values:
        assert np.isclose(float(nelsen11_copula.cdf(u, 1)), u, rtol=1e-5)
        assert np.isclose(float(nelsen11_copula.cdf(1, u)), u, rtol=1e-5)


def test_conditional_distribution_2(nelsen11_copula):
    """Test the second conditional distribution function."""
    # Test conditional distribution at specific points where the Heaviside function is 1
    # (i.e., points where u^theta * v^theta - 2*(1-u^theta)*(1-v^theta) > 0)
    points = [(0.7, 0.8), (0.8, 0.8), (0.9, 0.9)]

    for u, v in points:
        # Just check the result is within valid range
        result = float(nelsen11_copula.cond_distr_2(u, v))
        assert 0 <= result <= 1

        # The exact formula is complex, but we can verify basic properties
        # For these points, the Heaviside function should be 1
        assert result > 0


def test_rho_int_1(nelsen11_copula):
    """Test the _rho_int_1 helper method."""
    # This method involves symbolic integration
    # We'll just verify it can be called with a symbolic variable without error
    v_sym = sympy.Symbol("v", real=True, positive=True)

    # Patch the necessary properties to use symbolic variables
    with patch.object(nelsen11_copula, "v", v_sym):
        with patch.object(
            nelsen11_copula, "u", sympy.Symbol("u", real=True, positive=True)
        ):
            # Just make sure it returns a symbolic expression without error
            result = nelsen11_copula._rho_int_1()
            assert isinstance(result, sympy.Expr)


def test_lambda_l(nelsen11_copula):
    """Test lower tail dependence coefficient (lambda_L)."""
    # Nelsen11 has no lower tail dependence
    assert float(nelsen11_copula.lambda_L()) == 0


def test_lambda_u(nelsen11_copula):
    """Test upper tail dependence coefficient (lambda_U)."""
    # Nelsen11 has no upper tail dependence
    assert float(nelsen11_copula.lambda_U()) == 0


def test_theta_dependent_behavior():
    """Test how behavior changes with different theta values."""
    # Create copulas with different theta values
    copula1 = Nelsen11(0.1)
    copula2 = Nelsen11(0.3)
    copula3 = Nelsen11(0.5)

    # Test at a specific point where CDF is positive for all theta
    u, v = 0.8, 0.8

    # For Nelsen11, different theta should give different CDF values
    cdf1 = float(copula1.cdf(u, v))
    cdf2 = float(copula2.cdf(u, v))
    cdf3 = float(copula3.cdf(u, v))

    # Verify they are different
    assert cdf1 != cdf2
    assert cdf2 != cdf3

    # Verify they are all in valid range
    assert 0 <= cdf1 <= 1
    assert 0 <= cdf2 <= 1
    assert 0 <= cdf3 <= 1


def test_independence_case():
    """Test that theta=0 behaves like an independence copula."""
    # Get an independence copula via the special case
    independence = Nelsen11(0)

    # Test CDF at various points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2), (0.9, 0.8)]

    for u, v in points:
        # For independence copula, C(u,v) = u*v
        expected = u * v
        actual = float(independence.cdf(u, v))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_numerical_stability():
    """Test numerical stability at challenging points."""
    copula = Nelsen11(0.3)

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
            assert result < 0.1

        if v < 0.01:  # Very close to 0
            assert result < 0.1

        if u > 0.99:  # Very close to 1
            assert result > 0.4  # Should approach v but might not be exactly v

        if v > 0.99:  # Very close to 1
            assert result > 0.4  # Should approach u but might not be exactly u


def test_max_theta_case():
    """Test the limiting case with maximum theta value (0.5)."""
    # Create a copula with maximum allowed theta
    max_theta_copula = Nelsen11(0.5)

    # Test at some points
    points = [(0.7, 0.8), (0.8, 0.8), (0.9, 0.9)]

    for u, v in points:
        # Calculate the CDF
        result = float(max_theta_copula.cdf(u, v))

        # Result should be within [0,1]
        assert 0 <= result <= 1

        # Calculate the expected value with theta = 0.5
        theta = 0.5
        term = u**theta * v**theta - 2 * (1 - u**theta) * (1 - v**theta)
        if term > 0:
            expected = term ** (1 / theta)
        else:
            expected = 0
        assert np.isclose(result, expected, rtol=1e-5)
