import pytest
import sympy as sp
import numpy as np
from unittest.mock import patch

from copul.families.elliptical.elliptical_copula import EllipticalCopula
from copul.families.other.lower_frechet import LowerFrechet
from copul.families.other.upper_frechet import UpperFrechet
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


# Create a concrete subclass of EllipticalCopula for testing
class ConcreteEllipticalCopula(EllipticalCopula):
    """Concrete implementation of EllipticalCopula for testing purposes."""

    # Define a simple generator function for testing
    generator = sp.exp

    @property
    def cdf(self):
        """Implement the abstract CDF method for testing."""
        # A simple placeholder CDF
        cdf = (self.u + self.v) * (1 + self.rho * (1 - self.u) * (1 - self.v)) / 2
        return SymPyFuncWrapper(cdf)


@pytest.fixture
def elliptical_copula():
    """Create a concrete elliptical copula for testing."""
    return ConcreteEllipticalCopula()


@pytest.fixture
def elliptical_copula_with_rho():
    """Create a concrete elliptical copula with rho=0.5 for testing."""
    copula = ConcreteEllipticalCopula()
    copula.rho = 0.5
    return copula


def test_init(elliptical_copula):
    """Test initialization of elliptical copula."""
    # Check that the copula has the expected attributes
    assert hasattr(elliptical_copula, "rho")
    assert hasattr(elliptical_copula, "params")
    assert hasattr(elliptical_copula, "intervals")
    assert hasattr(elliptical_copula, "generator")

    # Check that rho is a symbolic variable
    assert isinstance(elliptical_copula.rho, sp.Symbol)
    assert str(elliptical_copula.rho) == "rho"

    # Check that 'params' includes rho
    assert elliptical_copula.params == [elliptical_copula.rho]


def test_parameter_bounds(elliptical_copula):
    """Test parameter bounds for elliptical copula."""
    # Check that rho interval is [-1, 1]
    rho_interval = elliptical_copula.intervals["rho"]
    assert rho_interval.left == -1
    assert rho_interval.right == 1
    assert not rho_interval.left_open  # Left bound is closed
    assert not rho_interval.right_open  # Right bound is closed


def test_call_method_special_cases(elliptical_copula):
    """Test special cases of the __call__ method."""
    # Test rho = -1 (should return LowerFrechet)
    copula = elliptical_copula(rho=-1)
    assert isinstance(copula, LowerFrechet)

    # Test rho = 1 (should return UpperFrechet)
    copula = elliptical_copula(rho=1)
    assert isinstance(copula, UpperFrechet)


def test_call_method_normal_case(elliptical_copula):
    """Test normal case of the __call__ method."""
    # Test rho = 0.5 (should return a new instance with updated rho)
    copula = elliptical_copula(rho=0.5)
    assert isinstance(copula, ConcreteEllipticalCopula)
    assert copula.rho == 0.5


def test_corr_matrix(elliptical_copula_with_rho):
    """Test the corr_matrix property."""
    corr_matrix = elliptical_copula_with_rho.corr_matrix

    # Check that the matrix has the correct structure
    assert isinstance(corr_matrix, sp.Matrix)
    assert corr_matrix.shape == (2, 2)

    # Check the values
    assert corr_matrix[0, 0] == 1
    assert corr_matrix[1, 1] == 1
    assert corr_matrix[0, 1] == 0.5
    assert corr_matrix[1, 0] == 0.5


def test_characteristic_function(elliptical_copula_with_rho):
    """Test the characteristic_function method."""
    # Evaluate the characteristic function at specific points
    result = elliptical_copula_with_rho.characteristic_function(1, 2)

    # Calculate the expected result manually
    # arg = t1^2 + t2^2 + 2*t1*t2*rho = 1^2 + 2^2 + 2*1*2*0.5 = 1 + 4 + 2 = 7
    # generator = exp, so result = exp(7)
    expected = float(sp.exp(7).evalf())

    # Convert the result to a float for comparison
    if isinstance(result, sp.Expr):
        result_float = float(result.evalf())
    else:
        result_float = float(result)

    # Compare with tolerance for floating-point comparison
    assert np.isclose(result_float, expected)


def test_cdf_abstract_method():
    """Test that the cdf property is implemented in the concrete class."""
    copula = ConcreteEllipticalCopula()

    # Should not raise NotImplementedError
    cdf = copula.cdf
    assert isinstance(cdf, SymPyFuncWrapper)


def test_cdf_evaluation(elliptical_copula_with_rho):
    """Test evaluation of the CDF."""
    # Mock the wrapper __call__ method to isolate the test
    with patch.object(SymPyFuncWrapper, "__call__") as mock_call:
        mock_call.return_value = 0.6  # Arbitrary return value

        # Evaluate the CDF at a specific point
        result = elliptical_copula_with_rho.cdf(0.3, 0.7)

        # Check that the wrapper was called with correct arguments
        mock_call.assert_called_once_with(0.3, 0.7)
        assert result == 0.6


def test_edge_cases():
    """Test edge cases of the elliptical copula."""
    copula = ConcreteEllipticalCopula()

    # Test rho near -1
    near_minus_one = copula(rho=-0.999)
    assert isinstance(near_minus_one, ConcreteEllipticalCopula)
    assert near_minus_one.rho == -0.999

    # Test rho near 1
    near_one = copula(rho=0.999)
    assert isinstance(near_one, ConcreteEllipticalCopula)
    assert near_one.rho == 0.999

    # Test rho = 0 (independence in many elliptical copulas)
    independence = copula(rho=0)
    assert isinstance(independence, ConcreteEllipticalCopula)
    assert independence.rho == 0


def test_elliptical_copula_is_abstract():
    """Test that EllipticalCopula is an abstract class that cannot be instantiated directly."""
    with pytest.raises(TypeError):
        # Should raise TypeError due to abstract method 'cdf'
        EllipticalCopula()
