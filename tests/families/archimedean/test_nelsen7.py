from unittest.mock import patch

import numpy as np
import pytest
import sympy

from copul.families.archimedean.nelsen7 import Nelsen7
from copul.families.other.independence_copula import IndependenceCopula
from copul.families.other.lower_frechet import LowerFrechet


@pytest.fixture
def nelsen7_copula():
    """Fixture providing a Nelsen7 copula with theta=0.5."""
    return Nelsen7(0.5)


def test_special_cases():
    """Test that special cases return appropriate copula instances."""
    # Test theta = 0 returns LowerFrechet
    lower_frechet = Nelsen7(0)
    assert isinstance(lower_frechet, LowerFrechet)

    # Test theta = 1 returns IndependenceCopula
    independence = Nelsen7(1)
    assert isinstance(independence, IndependenceCopula)

    # Test via create factory method
    lower_frechet_create = Nelsen7.create(0)
    assert isinstance(lower_frechet_create, LowerFrechet)

    independence_create = Nelsen7.create(1)
    assert isinstance(independence_create, IndependenceCopula)

    # Test via call method
    base_copula = Nelsen7(0.5)
    lower_frechet_call = base_copula(0)
    assert isinstance(lower_frechet_call, LowerFrechet)

    independence_call = base_copula(1)
    assert isinstance(independence_call, IndependenceCopula)


def test_parameter_validation():
    """Test parameter validation for the Nelsen7 copula."""
    # Valid values should not raise errors
    Nelsen7(0)  # Lower bound
    Nelsen7(0.5)  # Interior point
    Nelsen7(1)  # Upper bound

    # Values below lower bound should raise ValueError
    with pytest.raises(ValueError, match="must be >= 0"):
        Nelsen7(-0.1)

    # Values above upper bound should raise ValueError
    with pytest.raises(ValueError, match="must be <= 1"):
        Nelsen7(1.1)


def test_is_absolutely_continuous(nelsen7_copula):
    """Test the is_absolutely_continuous property."""
    # Nelsen7 is not absolutely continuous
    assert not nelsen7_copula.is_absolutely_continuous


def test_generator_function(nelsen7_copula):
    """Test the generator function of the Nelsen7 copula."""
    # Test generator at specific t values
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    for t in t_values:
        # Manual calculation using the formula: -log(theta*t + 1 - theta)
        theta = 0.5
        expected = -np.log(theta * t + 1 - theta)
        actual = float(nelsen7_copula.generator(t))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_inverse_generator_function(nelsen7_copula):
    """Test the inverse generator function of the Nelsen7 copula."""
    # Test inverse generator at specific y values
    y_values = [0.1, 0.5, 1.0, 2.0]

    for y in y_values:
        # The inverse generator has a Heaviside function condition
        # We're only testing for y values where it should return non-zero
        theta = 0.5
        # Manual calculation for values where y > -log(1-theta)
        if y > -np.log(1 - theta):
            expected = 0  # Expected to be 0 due to Heaviside
        else:
            expected = ((theta * np.exp(y) - np.exp(y) + 1) * np.exp(-y)) / theta

        actual = float(nelsen7_copula.inv_generator(y))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_cdf_function(nelsen7_copula):
    """Test the CDF function of the Nelsen7 copula."""
    # Test CDF at specific points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2), (0.9, 0.8)]

    for u, v in points:
        # Manual calculation using the formula
        theta = 0.5
        expected = max(theta * u * v + (1 - theta) * (u + v - 1), 0)
        actual = float(nelsen7_copula.cdf(u, v))
        assert np.isclose(actual, expected, rtol=1e-5)

    # Test with points that should yield 0 due to the Max function
    negative_points = [(0.1, 0.2), (0.3, 0.1), (0.2, 0.3)]
    for u, v in negative_points:
        theta = 0.5
        raw_value = theta * u * v + (1 - theta) * (u + v - 1)
        # Only test points where the raw value is negative
        if raw_value < 0:
            assert float(nelsen7_copula.cdf(u, v)) == 0


def test_cdf_boundary_conditions(nelsen7_copula):
    """Test boundary conditions for the Nelsen7 copula CDF."""
    # Test points along the boundaries
    u_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # C(u,0) = 0 for any u in [0,1]
    for u in u_values:
        assert np.isclose(float(nelsen7_copula.cdf(u, 0)), 0, atol=1e-10)
        assert np.isclose(float(nelsen7_copula.cdf(0, u)), 0, atol=1e-10)

    # C(u,1) = u and C(1,v) = v for any u,v in [0,1]
    for u in u_values:
        assert np.isclose(float(nelsen7_copula.cdf(u, 1)), u, rtol=1e-5)
        assert np.isclose(float(nelsen7_copula.cdf(1, u)), u, rtol=1e-5)


def test_conditional_distribution_1(nelsen7_copula):
    """Test the first conditional distribution function."""
    # Test conditional distribution at specific points
    points = [(0.7, 0.8), (0.8, 0.8), (0.9, 0.9)]

    for u, v in points:
        # Test that result is between 0 and 1
        result = float(nelsen7_copula.cond_distr_1(u, v))
        assert 0 <= result <= 1

        # For these points, the Heaviside function should be 1
        theta = 0.5
        expected = theta * v - theta + 1
        assert np.isclose(result, expected, rtol=1e-5)


def test_conditional_distribution_2(nelsen7_copula):
    """Test the second conditional distribution function."""
    # Test conditional distribution at specific points
    points = [(0.7, 0.8), (0.8, 0.8), (0.9, 0.9)]

    for u, v in points:
        # Test that result is between 0 and 1
        result = float(nelsen7_copula.cond_distr_2(u, v))
        assert 0 <= result <= 1

        # For these points, the Heaviside function should be 1
        theta = 0.5
        expected = theta * u - theta + 1
        assert np.isclose(result, expected, rtol=1e-5)


def test_chatterjees_xi(nelsen7_copula):
    """Test Chatterjee's xi coefficient."""
    # For Nelsen7, xi = 1 - theta
    theta = 0.5
    expected = 1 - theta
    actual = float(nelsen7_copula.chatterjees_xi())
    assert np.isclose(actual, expected, rtol=1e-5)

    # Test with other theta values
    for theta in [0, 0.3, 0.7, 1]:
        if theta in [0, 1]:
            # For special cases, create the copula specifically
            copula = Nelsen7.create(theta)
        else:
            copula = Nelsen7(theta)

        expected = 1 - theta
        actual = float(copula.chatterjees_xi())
        assert np.isclose(actual, expected, rtol=1e-5)


def test_kendalls_tau(nelsen7_copula):
    """Test Kendall's tau coefficient."""
    theta = 0.5
    # Manual calculation using the formula for theta=0.5
    log_term = np.log(1 - theta)
    expected = 2 - 2 / theta - 2 * (theta - 1) ** 2 * log_term / theta**2
    actual = float(nelsen7_copula.kendalls_tau())
    assert np.isclose(actual, expected, rtol=1e-5)

    # Test special cases
    lower_frechet = Nelsen7.create(0)
    assert float(lower_frechet.kendalls_tau()) == -1

    independence = Nelsen7.create(1)
    assert float(independence.kendalls_tau()) == 0


def test_lambda_l(nelsen7_copula):
    """Test lower tail dependence coefficient (lambda_L)."""
    # Nelsen7 has no lower tail dependence
    assert float(nelsen7_copula.lambda_L()) == 0


def test_lambda_u(nelsen7_copula):
    """Test upper tail dependence coefficient (lambda_U)."""
    # Nelsen7 has no upper tail dependence
    assert float(nelsen7_copula.lambda_U()) == 0


def test_rho_method(nelsen7_copula):
    """Test the _rho method (related to Spearman's rho)."""
    theta = 0.5
    # Manual calculation using the formula for theta=0.5
    log_term = np.log(1 - theta)
    expected = (
        -3 + 9 / theta - 6 / theta**2 - 6 * (theta - 1) ** 2 * log_term / theta**3
    )
    actual = float(nelsen7_copula._rho())
    assert np.isclose(actual, expected, rtol=1e-5)

    # Test special cases
    lower_frechet = Nelsen7.create(0)
    assert float(lower_frechet._rho()) == -1

    independence = Nelsen7.create(1)
    assert float(independence._rho()) == 0


def test_helper_methods_existence(nelsen7_copula):
    """Test that helper methods exist and run without errors."""
    # Just verify these methods exist and can be called
    # Detailed testing of these would require symbolic calculations
    assert hasattr(nelsen7_copula, "_rho_int_1")
    assert hasattr(nelsen7_copula, "_tau_int_1")

    # Just make sure they don't raise errors when called
    # Since these are symbolic integrations, we just check they return something
    # but don't validate exact values
    v_sym = sympy.Symbol("v", real=True, positive=True)
    with patch.object(nelsen7_copula, "v", v_sym):
        result_rho = nelsen7_copula._rho_int_1()
        assert isinstance(result_rho, sympy.Expr)

        result_tau = nelsen7_copula._tau_int_1()
        assert isinstance(result_tau, sympy.Expr)


@pytest.mark.parametrize("theta, expected", [(0, -1), (1, 0)])
def test_nelsen7_rho(theta, expected):
    nelsen = Nelsen7()(theta)
    rho = nelsen.spearmans_rho()
    assert np.isclose(rho, expected)


@pytest.mark.parametrize("theta, expected", [(0, -1), (1, 0)])
def test_nelsen7_tau(theta, expected):
    nelsen = Nelsen7(theta)
    tau = nelsen.kendalls_tau()
    assert np.isclose(tau, expected)
