from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import sympy

from copul.families.archimedean.nelsen21 import Nelsen21
from copul.families.other.lower_frechet import LowerFrechet


@pytest.fixture
def nelsen21_copula():
    """Fixture providing a Nelsen21 copula with theta=2."""
    return Nelsen21(2)


def test_special_cases():
    """Test that special cases return appropriate copula instances."""
    # Test theta = 1 returns LowerFrechet
    lower_frechet = Nelsen21(1)
    assert isinstance(lower_frechet, LowerFrechet)

    # Test via create factory method
    lower_frechet_create = Nelsen21.create(1)
    assert isinstance(lower_frechet_create, LowerFrechet)

    # Test via call method
    base_copula = Nelsen21(2)
    lower_frechet_call = base_copula(1)
    assert isinstance(lower_frechet_call, LowerFrechet)


def test_parameter_validation():
    """Test parameter validation for the Nelsen21 copula."""
    # Valid values should not raise errors
    Nelsen21(1)  # Lower bound
    Nelsen21(2)  # Interior point
    Nelsen21(10)  # Larger value

    # Values below lower bound should raise ValueError
    with pytest.raises(ValueError, match="must be >= 1"):
        Nelsen21(0.5)

    # Negative values should raise ValueError
    with pytest.raises(ValueError, match="must be >= 1"):
        Nelsen21(-1)


def test_is_absolutely_continuous(nelsen21_copula):
    """Test the is_absolutely_continuous property."""
    # Nelsen21 is not absolutely continuous
    assert not nelsen21_copula.is_absolutely_continuous


def test_generator_function(nelsen21_copula):
    """Test the generator function of the Nelsen21 copula."""
    # Test generator at specific t values
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    for t in t_values:
        # Manual calculation using the formula: 1 - (1 - (1 - t)^theta)^(1/theta)
        theta = 2
        expected = 1 - (1 - (1 - t) ** theta) ** (1 / theta)
        actual = float(nelsen21_copula.generator(t))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_inverse_generator_function(nelsen21_copula):
    """Test the inverse generator function of the Nelsen21 copula."""
    # Test inverse generator at specific y values
    y_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    for y in y_values:
        # For this copula, the inverse generator has a piecewise condition
        # But all y values in [0,1] should satisfy the y <= pi/2 condition
        theta = 2
        expected = 1 - (1 - (1 - y) ** theta) ** (1 / theta)
        actual = float(nelsen21_copula.inv_generator(y))
        assert np.isclose(actual, expected, rtol=1e-5)

    # Test for y > pi/2, should return 0
    large_y = np.pi
    assert float(nelsen21_copula.inv_generator(large_y)) == 0


def test_cdf_function(nelsen21_copula):
    """Test the CDF function of the Nelsen21 copula."""
    # Test CDF at specific points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2), (0.9, 0.8)]

    for u, v in points:
        # Manual calculation using the formula
        theta = 2
        term = (
            (1 - (1 - u) ** theta) ** (1 / theta)
            + (1 - (1 - v) ** theta) ** (1 / theta)
            - 1
        )
        term = max(term, 0)
        expected = 1 - (1 - term**theta) ** (1 / theta)
        actual = float(nelsen21_copula.cdf(u, v))
        assert np.isclose(actual, expected, rtol=1e-5)

    # Test with points where the expression might be negative before max
    edge_points = [(0.1, 0.1), (0.2, 0.1), (0.1, 0.2)]

    for u, v in edge_points:
        theta = 2
        term = (
            (1 - (1 - u) ** theta) ** (1 / theta)
            + (1 - (1 - v) ** theta) ** (1 / theta)
            - 1
        )
        if term < 0:
            # If term < 0, then after max(term, 0), the result should be 0
            # and cdf should be 1 - (1 - 0^theta)^(1/theta) = 0
            expected = 0
        else:
            expected = 1 - (1 - term**theta) ** (1 / theta)
        actual = float(nelsen21_copula.cdf(u, v))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_cdf_boundary_conditions(nelsen21_copula):
    """Test boundary conditions for the Nelsen21 copula CDF."""
    # Test points along the boundaries
    u_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # C(u,0) = 0 for any u in [0,1]
    for u in u_values:
        assert np.isclose(float(nelsen21_copula.cdf(u, 0)), 0, atol=1e-10)
        assert np.isclose(float(nelsen21_copula.cdf(0, u)), 0, atol=1e-10)

    # C(u,1) = u and C(1,v) = v for any u,v in [0,1]
    for u in u_values:
        assert np.isclose(float(nelsen21_copula.cdf(u, 1)), u, rtol=1e-5)
        assert np.isclose(float(nelsen21_copula.cdf(1, u)), u, rtol=1e-5)


def test_conditional_distribution_2(nelsen21_copula):
    """Test the second conditional distribution function."""
    # Test conditional distribution at some valid points
    points = [(0.7, 0.8), (0.8, 0.8), (0.9, 0.9)]

    for u, v in points:
        # Just verify it returns a value in [0,1] without errors
        result = float(nelsen21_copula.cond_distr_2(u, v))
        assert 0 <= result <= 1


def test_lambda_l(nelsen21_copula):
    """Test lower tail dependence coefficient (lambda_L)."""
    # For Nelsen21, lambda_L = 0
    assert float(nelsen21_copula.lambda_L()) == 0


def test_lambda_u(nelsen21_copula):
    """Test upper tail dependence coefficient (lambda_U)."""
    # For Nelsen21, lambda_U = 2 - 2^(1/theta)
    theta = 2
    expected = 2 - 2 ** (1 / theta)
    actual = float(nelsen21_copula.lambda_U())
    assert np.isclose(actual, expected, rtol=1e-5)

    # Test with different theta values
    for theta in [1.5, 3, 5]:
        copula = Nelsen21(theta)
        expected = 2 - 2 ** (1 / theta)
        actual = float(copula.lambda_U())
        assert np.isclose(actual, expected, rtol=1e-5)


def test_rho_int_1_sympy(nelsen21_copula):
    """Test the _rho_int_1_sympy method."""
    # This method returns a symbolic integral, so we'll just verify it returns
    # the expected type

    # Patch print to avoid output
    with patch("builtins.print"):
        result = nelsen21_copula._rho_int_1_sympy()
        assert isinstance(result, sympy.Integral)


def test_integration_methods(nelsen21_copula):
    """Test the integration methods of Nelsen21."""
    # These methods involve numerical integration which can be slow and numerically sensitive
    # We'll just verify they run without errors for a simple case

    # Test _cdf method with a specific point
    result = nelsen21_copula._cdf(0.5, 0.5, 2)
    assert 0 <= result <= 1

    # Test _positive_cdf method
    result = nelsen21_copula._positive_cdf(0.5, 0.5, 2)
    assert 0 <= result <= 1

    # Test _lower_bound method
    result = nelsen21_copula._lower_bound(0.5, 2)
    assert 0 <= result <= 1

    # Mock the integration methods to avoid actual integration in tests
    with patch("scipy.integrate.quad", return_value=(0.3, 0.01)):
        result = nelsen21_copula._rho_int_1(0.5, 2)
        assert np.isclose(result, 0.3, atol=0.1)


def test_theta_dependent_behavior():
    """Test how behavior changes with different theta values."""
    # Create copulas with different theta values
    copula1 = Nelsen21(1.5)
    copula2 = Nelsen21(3)
    copula3 = Nelsen21(5)

    # Test at a specific point
    u, v = 0.5, 0.5

    # Calculate CDF values
    cdf1 = float(copula1.cdf(u, v))
    cdf2 = float(copula2.cdf(u, v))
    cdf3 = float(copula3.cdf(u, v))

    # For Nelsen21, different theta should give different CDF values
    # Verify they are different
    assert cdf1 != cdf2 and cdf2 != cdf3

    # Verify they are all in valid range
    assert 0 <= cdf1 <= 1
    assert 0 <= cdf2 <= 1
    assert 0 <= cdf3 <= 1


def test_lower_frechet_case():
    """Test behavior at the special case (theta=1)."""
    # Create a LowerFrechet instance directly
    direct_instance = LowerFrechet()

    # Get an instance via Nelsen21 special case
    special_case = Nelsen21.create(1)

    # Test at some points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2)]

    for u, v in points:
        direct_result = float(direct_instance.cdf(u, v))
        special_result = float(special_case.cdf(u, v))
        assert np.isclose(direct_result, special_result, rtol=1e-5)


def test_large_theta_behavior():
    """Test behavior with a very large theta value."""
    # With large theta, check that computations remain stable
    large_theta = Nelsen21(50)

    # Test at some points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2)]

    for u, v in points:
        result = float(large_theta.cdf(u, v))

        # Result should be within [0,1]
        assert 0 <= result <= 1

        # For very large theta, the copula approaches a specific limiting form
        # Just verify the result is a valid value


def test_numerical_stability():
    """Test numerical stability at challenging points."""
    copula = Nelsen21(2)

    # Test at points very close to 0 and 1
    edge_points = [
        (0.001, 0.5),  # u close to 0
        (0.5, 0.001),  # v close to 0
        (0.999, 0.5),  # u close to 1
        (0.5, 0.999),  # v close to 1
    ]

    for u, v in edge_points:
        # Just verify computation doesn't raise errors
        result = float(copula.cdf(u, v))

        # Result should be within [0,1]
        assert 0 <= result <= 1
