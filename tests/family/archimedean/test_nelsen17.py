from unittest.mock import patch

import numpy as np
import pytest
import sympy

from copul.family.archimedean.nelsen17 import Nelsen17
from copul.family.other.biv_independence_copula import BivIndependenceCopula


@pytest.fixture
def nelsen17_copula():
    """Fixture providing a Nelsen17 copula with theta=1."""
    return Nelsen17(1)


def test_special_cases():
    """Test that special cases return appropriate copula instances."""
    # Test theta = -1 returns IndependenceCopula
    independence = Nelsen17(-1)
    assert isinstance(independence, BivIndependenceCopula)

    # Test via create factory method
    independence_create = Nelsen17.create(-1)
    assert isinstance(independence_create, BivIndependenceCopula)

    # Test via call method
    base_copula = Nelsen17(1)
    independence_call = base_copula(-1)
    assert isinstance(independence_call, BivIndependenceCopula)


def test_invalid_parameters():
    """Test that invalid parameters raise ValueErrors."""
    # Test theta = 0 raises ValueError
    with pytest.raises(ValueError, match="Parameter theta cannot be 0"):
        Nelsen17(0)

    # Test via create factory method
    with pytest.raises(ValueError, match="Parameter theta cannot be 0"):
        Nelsen17.create(0)

    # Test via call method
    base_copula = Nelsen17(1)
    with pytest.raises(ValueError, match="Parameter theta cannot be 0"):
        base_copula(0)


def test_parameter_validation():
    """Test parameter validation for the Nelsen17 copula."""
    # Valid values should not raise errors
    Nelsen17(-2)  # Negative value
    Nelsen17(1)  # Positive value
    Nelsen17(10)  # Larger value

    # Only theta = 0 is invalid
    with pytest.raises(ValueError, match="Parameter theta cannot be 0"):
        Nelsen17(0)


def test_is_absolutely_continuous(nelsen17_copula):
    """Test the is_absolutely_continuous property."""
    # Nelsen17 is absolutely continuous
    assert nelsen17_copula.is_absolutely_continuous


def test_generator_function(nelsen17_copula):
    """Test the generator function of the Nelsen17 copula."""
    # Test generator at specific t values
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    for t in t_values:
        # Manual calculation using the formula: -log(((1+t)^(-theta) - 1)/(2^(-theta) - 1))
        theta = 1
        expected = -np.log(((1 + t) ** (-theta) - 1) / (2 ** (-theta) - 1))
        actual = float(nelsen17_copula.generator(t))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_inverse_generator_function(nelsen17_copula):
    """Test the inverse generator function of the Nelsen17 copula."""
    # Test inverse generator at specific y values
    y_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    for y in y_values:
        # Manual calculation using the formula
        theta = 1
        expected = (2**theta * np.exp(y) / (2**theta * np.exp(y) - 2**theta + 1)) ** (
            1 / theta
        ) - 1
        actual = float(nelsen17_copula.inv_generator(y))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_cdf_function(nelsen17_copula):
    """Test the CDF function of the Nelsen17 copula."""
    # Test CDF at specific points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2), (0.9, 0.8)]

    for u, v in points:
        # Manual calculation using the formula
        theta = 1
        term = (
            ((1 + u) ** (-theta) - 1) * ((1 + v) ** (-theta) - 1) / (2 ** (-theta) - 1)
        )
        expected = (1 + term) ** (-1 / theta) - 1
        actual = float(nelsen17_copula.cdf(u, v))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_pdf_function(nelsen17_copula):
    """Test the PDF function of the Nelsen17 copula."""
    # PDF is very complex, so we'll just verify it returns valid values at some points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2), (0.9, 0.8)]

    for u, v in points:
        result = float(nelsen17_copula.pdf(u, v))
        # PDF should be non-negative for a valid copula
        assert result >= 0


def test_cdf_boundary_conditions(nelsen17_copula):
    """Test boundary conditions for the Nelsen17 copula CDF."""
    # Test points along the boundaries
    u_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # C(u,0) = 0 for any u in [0,1]
    for u in u_values:
        assert np.isclose(float(nelsen17_copula.cdf(u, 0)), 0, atol=1e-10)
        assert np.isclose(float(nelsen17_copula.cdf(0, u)), 0, atol=1e-10)

    # C(u,1) = u and C(1,v) = v for any u,v in [0,1]
    for u in u_values:
        assert np.isclose(float(nelsen17_copula.cdf(u, 1)), u, rtol=1e-5)
        assert np.isclose(float(nelsen17_copula.cdf(1, u)), u, rtol=1e-5)


def test_derivative_functions(nelsen17_copula):
    """Test the derivative functions of the Nelsen17 copula."""
    # These are very complex symbolic functions
    # We'll just verify they can be called without errors

    # Patch the 'y' property for testing
    with patch.object(
        nelsen17_copula, "y", sympy.Symbol("y", real=True, positive=True)
    ):
        # Test first derivatives
        first_deriv_inv_gen = nelsen17_copula.first_deriv_of_inv_gen
        first_deriv_ci_char = nelsen17_copula.first_deriv_of_ci_char
        first_deriv_tp2_char = nelsen17_copula.first_deriv_of_tp2_char

        # Test second derivatives
        second_deriv_inv_gen = nelsen17_copula.second_deriv_of_inv_gen
        second_deriv_tp2_char = nelsen17_copula.second_deriv_of_tp2_char

        # Verify they return sympy expressions
        assert isinstance(first_deriv_inv_gen, sympy.Expr)
        assert isinstance(first_deriv_ci_char, sympy.Expr)
        assert isinstance(first_deriv_tp2_char(), sympy.Expr)
        assert isinstance(second_deriv_inv_gen, sympy.Expr)
        assert isinstance(second_deriv_tp2_char(), sympy.Expr)


def test_density_related_functions(nelsen17_copula):
    """Test the density-related functions of the Nelsen17 copula."""
    # These functions are extremely complex
    # We'll just verify they can be called without errors

    # Patch the 'u' and 'v' properties for testing
    with patch.object(
        nelsen17_copula, "u", sympy.Symbol("u", real=True, positive=True)
    ):
        with patch.object(
            nelsen17_copula, "v", sympy.Symbol("v", real=True, positive=True)
        ):
            # Test derivative of log density
            deriv_log_density = nelsen17_copula.deriv_of_log_density()

            # Test density of log density
            density_log_density = nelsen17_copula.density_of_log_density()

            # Verify they return sympy expressions
            assert isinstance(deriv_log_density, sympy.Expr)
            assert isinstance(density_log_density, sympy.Expr)


def test_lambda_l(nelsen17_copula):
    """Test lower tail dependence coefficient (lambda_L)."""
    # For Nelsen17, lambda_L = 0
    assert float(nelsen17_copula.lambda_L()) == 0


def test_lambda_u(nelsen17_copula):
    """Test upper tail dependence coefficient (lambda_U)."""
    # For Nelsen17, lambda_U = 0
    assert float(nelsen17_copula.lambda_U()) == 0


def test_theta_dependent_behavior():
    """Test how behavior changes with different theta values."""
    # Create copulas with different theta values
    copula1 = Nelsen17(-2)
    copula2 = Nelsen17(1)
    copula3 = Nelsen17(2)

    # Test at a specific point
    u, v = 0.3, 0.4

    # Calculate CDF values
    cdf1 = float(copula1.cdf(u, v))
    cdf2 = float(copula2.cdf(u, v))
    cdf3 = float(copula3.cdf(u, v))

    # For Nelsen17, different theta should give different CDF values
    # Verify they are different
    assert cdf1 != cdf2
    assert cdf2 != cdf3

    # Verify they are all in valid range
    assert 0 <= cdf1 <= 1
    assert 0 <= cdf2 <= 1
    assert 0 <= cdf3 <= 1


def test_independence_case():
    """Test behavior at the special case (theta=-1)."""
    # Create an IndependenceCopula instance directly
    direct_instance = BivIndependenceCopula()

    # Get an instance via Nelsen17 special case
    special_case = Nelsen17.create(-1)

    # Test at some points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2)]

    for u, v in points:
        direct_result = float(direct_instance.cdf(u, v))
        special_result = float(special_case.cdf(u, v))
        assert np.isclose(direct_result, special_result, rtol=1e-5)


def test_extreme_theta_values():
    """Test behavior with extreme theta values."""
    # Test with very large positive and negative theta
    large_pos_theta = Nelsen17(50)
    large_neg_theta = Nelsen17(-50)

    # Test at some points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2)]

    for u, v in points:
        # Calculate CDF values
        pos_result = float(large_pos_theta.cdf(u, v))
        neg_result = float(large_neg_theta.cdf(u, v))

        # Results should be in valid range
        assert 0 <= pos_result <= 1
        assert 0 <= neg_result <= 1


def test_numerical_stability():
    """Test numerical stability at challenging points."""
    copula = Nelsen17(1)

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

        # Test basic properties at boundaries
        if u < 0.01:  # Very close to 0
            assert result < 0.1

        if v < 0.01:  # Very close to 0
            assert result < 0.1

        if u > 0.99:  # Very close to 1
            assert result > 0.4  # Should approach v but might not be exactly v

        if v > 0.99:  # Very close to 1
            assert result > 0.4  # Should approach u but might not be exactly u
