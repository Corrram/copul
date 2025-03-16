import matplotlib

matplotlib.use("Agg")  # Use the 'Agg' backend to suppress the pop-up

from unittest.mock import patch

import numpy as np
import pytest
import sympy

from copul.families.archimedean.nelsen9 import GumbelBarnett, Nelsen9
from copul.families.other.biv_independence_copula import BivIndependenceCopula


@pytest.fixture
def gumbel_barnett():
    """Fixture providing a GumbelBarnett copula with theta=0.5."""
    return GumbelBarnett(0.5)


def test_special_cases():
    """Test that special cases return appropriate copula instances."""
    # Test theta = 0 returns IndependenceCopula
    independence = GumbelBarnett(0)
    assert isinstance(independence, BivIndependenceCopula)

    # Test via create factory method
    independence_create = GumbelBarnett.create(0)
    assert isinstance(independence_create, BivIndependenceCopula)

    # Test via call method
    base_copula = GumbelBarnett(0.5)
    independence_call = base_copula(0)
    assert isinstance(independence_call, BivIndependenceCopula)


def test_parameter_validation():
    """Test parameter validation for the GumbelBarnett copula."""
    # Valid values should not raise errors
    GumbelBarnett(0)  # Lower bound
    GumbelBarnett(0.5)  # Interior point
    GumbelBarnett(1)  # Upper bound

    # Values below lower bound should raise ValueError
    with pytest.raises(ValueError, match="must be >= 0"):
        GumbelBarnett(-0.1)

    # Values above upper bound should raise ValueError
    with pytest.raises(ValueError, match="must be <= 1"):
        GumbelBarnett(1.1)


def test_is_absolutely_continuous(gumbel_barnett):
    """Test the is_absolutely_continuous property."""
    # GumbelBarnett is absolutely continuous
    assert gumbel_barnett.is_absolutely_continuous


def test_generator_function(gumbel_barnett):
    """Test the generator function of the GumbelBarnett copula."""
    # Test generator at specific t values
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    for t in t_values:
        # Manual calculation using the formula: log(1 - theta * log(t))
        theta = 0.5
        expected = np.log(1 - theta * np.log(t))
        actual = float(gumbel_barnett.generator(t))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_inverse_generator_function(gumbel_barnett):
    """Test the inverse generator function of the GumbelBarnett copula."""
    # Test inverse generator at specific y values
    y_values = [0.1, 0.5, 1.0, 2.0]

    for y in y_values:
        # Manual calculation
        theta = 0.5
        expected = np.exp((1 - np.exp(y)) / theta)
        actual = float(gumbel_barnett.inv_generator(y))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_cdf_function(gumbel_barnett):
    """Test the CDF function of the GumbelBarnett copula."""
    # Test CDF at specific points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2), (0.9, 0.8)]

    for u, v in points:
        # Manual calculation using the formula
        theta = 0.5
        expected = u * v * np.exp(-theta * np.log(u) * np.log(v))
        actual = float(gumbel_barnett.cdf(u, v))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_cdf_boundary_conditions(gumbel_barnett):
    """Test boundary conditions for the GumbelBarnett copula CDF."""
    # Test points along the boundaries
    u_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # C(u,0) = 0 for any u in [0,1]
    for u in u_values:
        assert np.isclose(float(gumbel_barnett.cdf(u, 0)), 0, atol=1e-10)
        assert np.isclose(float(gumbel_barnett.cdf(0, u)), 0, atol=1e-10)

    # C(u,1) = u and C(1,v) = v for any u,v in [0,1]
    for u in u_values:
        assert np.isclose(float(gumbel_barnett.cdf(u, 1)), u, rtol=1e-5)
        assert np.isclose(float(gumbel_barnett.cdf(1, u)), u, rtol=1e-5)


def test_cdf_increasing_theta():
    """Test that increasing theta produces a different CDF value."""
    # Create copulas with different theta values
    gb_small = GumbelBarnett(0.2)
    gb_medium = GumbelBarnett(0.5)
    gb_large = GumbelBarnett(0.8)

    # Test at a specific point (u,v)
    u, v = 0.3, 0.4

    # Compute CDF values
    cdf_small = float(gb_small.cdf(u, v))
    cdf_medium = float(gb_medium.cdf(u, v))
    cdf_large = float(gb_large.cdf(u, v))

    # For GumbelBarnett, increasing theta should decrease the CDF value
    # This is because exp(-theta * log(u) * log(v)) decreases as theta increases
    # (assuming u,v < 1, which means log(u), log(v) < 0)
    assert cdf_small > cdf_medium > cdf_large


def test_xi_int_1(gumbel_barnett):
    """Test the _xi_int_1 helper method."""
    # Test with numerical values
    v_values = [0.3, 0.5, 0.7]

    for v in v_values:
        # Calculate result
        result = float(gumbel_barnett._xi_int_1(v))

        # The result should be a real number
        assert isinstance(result, float)

        # For this function with v < 1 and theta > 0, result should be positive
        assert result > 0


def test_xi_int_2(gumbel_barnett):
    """Test the _xi_int_2 helper method."""
    # This method does not take arguments, so just call it
    result = float(gumbel_barnett._xi_int_2())

    # The result should be a real number
    assert isinstance(result, float)


def test_rho_int_1(gumbel_barnett):
    """Test the _rho_int_1 helper method."""
    # Mock the 'v' property
    v_value = 0.5
    with patch.object(gumbel_barnett, "v", v_value):
        # Calculate result
        result = float(gumbel_barnett._rho_int_1())

        # The result should be a real number
        assert isinstance(result, float)

        # Manual calculation
        theta = 0.5
        expected = -v_value / (theta * np.log(v_value) - 2)
        assert np.isclose(result, expected, rtol=1e-5)


def test_rho_int_2(gumbel_barnett):
    """Test the _rho_int_2 helper method."""
    # Mock the 'v' property
    with patch.object(gumbel_barnett, "v", sympy.Symbol("v")):
        # This calls a symbolic integration, so we'll just check it runs
        # and returns a sympy expression
        result = gumbel_barnett._rho_int_2()
        assert isinstance(result, sympy.Expr)


def test_independence_case():
    """Test that theta=0 behaves like an independence copula."""
    # Get an independence copula via the special case
    independence = GumbelBarnett(0)

    # Test CDF at various points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2), (0.9, 0.8)]

    for u, v in points:
        # For independence copula, C(u,v) = u*v
        expected = u * v
        actual = float(independence.cdf(u, v))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_inherited_methods():
    """Test that inherited methods work correctly."""
    copula = GumbelBarnett(0.5)

    # Test that lambda_L and lambda_U exist and return values
    assert hasattr(copula, "lambda_L")
    assert hasattr(copula, "lambda_U")

    # These methods should return real numbers
    lambda_l = float(copula.lambda_L())
    lambda_u = float(copula.lambda_U())

    assert isinstance(lambda_l, float)
    assert isinstance(lambda_u, float)


def test_nelsen9_plot_cdf():
    nelsen = Nelsen9(0.5)

    # Call plot_cdf and simply ensure it does not raise an exception
    try:
        nelsen.plot(cond_distr_2=nelsen.cond_distr_2)
    except Exception as e:
        pytest.fail(f"plot_cdf() raised an exception: {e}")


def test_gumbel_barnett_cdf():
    nelsen = Nelsen9(0.5)
    result = nelsen.cdf(0.5, 0.5)
    assert np.isclose(result.evalf(), 0.19661242613985133)
