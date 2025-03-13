import numpy as np
import pytest
import sympy
from unittest.mock import patch

from copul.families.other import IndependenceCopula, LowerFrechet, UpperFrechet
from copul.families.elliptical.gaussian import Gaussian
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


@pytest.fixture
def gaussian_copula():
    """Create a Gaussian copula with rho=0.5 for testing."""
    return Gaussian(0.5)


@pytest.fixture
def gaussian_family():
    """Create a symbolic Gaussian copula family for testing."""
    return Gaussian()


@pytest.mark.parametrize(
    "rho, expected_class",
    [(-1, LowerFrechet), (0, IndependenceCopula), (1, UpperFrechet)],
)
def test_gaussian_edge_cases(rho, expected_class):
    cop = Gaussian()(rho)
    class_name = expected_class.__name__
    msg = f"Expected {class_name} for rho={rho}, but got {type(cop).__name__}"
    assert isinstance(cop, expected_class), msg

    # Now test with direct initialization as well
    cop2 = Gaussian(rho)
    assert isinstance(cop2, expected_class), msg


def test_gaussian_rvs():
    cop = Gaussian(0.5)
    assert cop.rvs(10).shape == (10, 2)


def test_gaussian_cdf():
    gaussian_family = Gaussian()
    cop = gaussian_family(0.5)
    assert np.isclose(cop.cdf(0.5, 0.5).evalf(), 1 / 3)


def test_gaussian_cd1():
    gaussian_family = Gaussian()
    cop = gaussian_family(0.5)
    cdf = cop.cond_distr_1(0.3, 0.4)
    assert np.isclose(cdf.evalf(), 0.504078212489690)


@pytest.mark.parametrize("rho, expected", [(-1, -1), (0, 0), (1, 1)])
def test_gaussian_tau(rho, expected):
    cop = Gaussian()(rho)
    assert cop.kendalls_tau() == expected


@pytest.mark.parametrize("rho, expected", [(-1, 1), (0, 0), (1, 1)])
def test_gaussian_xi(rho, expected):
    cop = Gaussian()(rho)
    assert cop.chatterjees_xi() == expected


# Extended tests


def test_gaussian_init():
    """Test initialization of Gaussian copula."""
    # Default initialization with symbol
    copula = Gaussian()
    assert hasattr(copula, "rho")
    assert isinstance(copula.rho, sympy.Symbol)
    assert str(copula.rho) == "rho"

    # Initialization with parameter (that isn't a special case)
    copula = Gaussian(0.5)
    assert hasattr(copula, "rho")
    assert copula.rho == 0.5


def test_gaussian_properties(gaussian_copula):
    """Test basic properties of Gaussian copula."""
    # Test symmetry property
    assert gaussian_copula.is_symmetric is True

    # Test absolute continuity property
    assert gaussian_copula.is_absolutely_continuous is True


def test_gaussian_generator():
    """Test the generator function is correctly defined."""
    # The generator should be exp(-t/2)
    t = Gaussian.t
    generator_expr = Gaussian.generator

    # Evaluate at t=1 and get numeric values for both
    result = float(generator_expr.subs(t, 1).evalf())
    expected = float(sympy.exp(-1 / 2).evalf())

    assert np.isclose(result, expected)


def test_gaussian_cond_distr_2():
    """Test second conditional distribution."""
    cop = Gaussian(0.5)

    # Test edge cases
    assert np.isclose(cop.cond_distr_2(0, 0.5).evalf(), 0)
    assert np.isclose(cop.cond_distr_2(1, 0.5).evalf(), 1)

    # Test regular case
    cdf = cop.cond_distr_2(0.4, 0.3)
    # Expected value based on the conditional distribution formula
    # Value may need adjustment if implementation details change
    expected_value = 0.504078212489690  # Same as cond_distr_1 with args swapped
    assert np.isclose(cdf.evalf(), expected_value)


def test_gaussian_pdf():
    """Test PDF calculation."""
    cop = Gaussian(0.5)

    # Mock the PDF calculation from statsmodels to isolate the test
    with patch(
        "statsmodels.distributions.copula.elliptical.GaussianCopula.pdf"
    ) as mock_pdf:
        mock_pdf.return_value = 1.25  # Arbitrary test value

        # Evaluate the PDF at a specific point
        result = cop.pdf(0.3, 0.7)

        # Check that the wrapper was called with correct arguments
        mock_pdf.assert_called_once_with([0.3, 0.7])
        assert isinstance(result, SymPyFuncWrapper)


def test_gaussian_spearmans_rho():
    """Test Spearman's rho calculation."""
    # For rho = 0.5, Spearman's rho should be 6/π * arcsin(0.5/2) ≈ 0.4886
    cop = Gaussian(0.5)
    rho = cop.spearmans_rho()
    expected = 6 / np.pi * np.arcsin(0.5 / 2)
    assert np.isclose(rho, expected)

    # Test with parameter passed to the method
    copula = Gaussian()
    rho = copula.spearmans_rho(0.5)
    assert np.isclose(rho, expected)


def test_gaussian_correlation_measures_consistency():
    """Test consistency between different correlation measures."""
    # Creating copulas with different rho values
    rho_values = [-0.8, -0.5, -0.2, 0.2, 0.5, 0.8]

    for rho in rho_values:
        cop = Gaussian(rho)

        # Calculate correlation measures
        tau = cop.kendalls_tau()
        spearman = cop.spearmans_rho()
        xi = cop.chatterjees_xi()

        # For Gaussian copula, certain relationships should hold:
        # - tau and rho have the same sign
        # - spearman and rho have the same sign
        # - xi should be non-negative
        assert np.sign(tau) == np.sign(rho) if rho != 0 else tau == 0
        assert np.sign(spearman) == np.sign(rho) if rho != 0 else spearman == 0
        assert xi >= 0

        # Specific relationships for Gaussian:
        # - tau = 2/π * arcsin(rho)
        # - spearman = 6/π * arcsin(rho/2)
        assert np.isclose(tau, 2 / np.pi * np.arcsin(rho))
        assert np.isclose(spearman, 6 / np.pi * np.arcsin(rho / 2))


def test_gaussian_conditional_distribution_function():
    """Test the _conditional_distribution method."""
    cop = Gaussian(0.5)

    # Test function with both arguments
    result = cop._conditional_distribution(0.3, 0.4)
    assert isinstance(result, float)

    # Test function with only first argument
    func = cop._conditional_distribution(0.3)
    assert callable(func)
    assert isinstance(func(0.4), float)

    # Test function with only second argument
    func = cop._conditional_distribution(v=0.4)
    assert callable(func)
    assert isinstance(func(0.3), float)

    # Test function with no arguments
    func = cop._conditional_distribution()
    assert callable(func)
    assert isinstance(func(0.3, 0.4), float)


def test_gaussian_characteristic_function():
    """Test the characteristic function with the Gaussian generator."""
    cop = Gaussian(0.5)
    # Calculate the argument for t1=t2=1 with rho=0.5
    # arg = 1^2 + 1^2 + 2*1*1*0.5 = 2 + 1 = 3
    arg_value = 3

    # Evaluate both expressions numerically
    result = float(cop.characteristic_function(1, 1).evalf())
    expected = float(sympy.exp(-arg_value / 2).evalf())

    assert np.isclose(result, expected)
