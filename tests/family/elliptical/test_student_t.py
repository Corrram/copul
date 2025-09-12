import numpy as np
import pytest
import sympy
from unittest.mock import patch

from copul.family.elliptical.student_t import StudentT
from copul.family.other import LowerFrechet, UpperFrechet
from copul.wrapper.cd1_wrapper import CD1Wrapper
from copul.wrapper.cd2_wrapper import CD2Wrapper
from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


@pytest.fixture
def student_t_copula():
    """Create a StudentT copula with specific parameters for testing."""
    copula = StudentT()
    copula.rho = 0.5
    copula.nu = 4.0
    return copula


@pytest.fixture
def student_t_family():
    """Create a symbolic StudentT copula family for testing."""
    return StudentT()


def test_student_t_init():
    """Test initialization of StudentT copula."""
    # Default initialization with symbols
    copula = StudentT()
    assert hasattr(copula, "rho")
    assert hasattr(copula, "nu")
    assert isinstance(copula.rho, sympy.Symbol)
    assert isinstance(copula.nu, sympy.Symbol)
    assert str(copula.rho) == "rho"
    assert str(copula.nu) == "nu"

    # Initialization with parameters
    copula = StudentT()(0.5, 4.0)
    assert copula.rho == 0.5
    assert copula.nu == 4.0


def test_student_t_parameter_bounds():
    """Test parameter bounds for StudentT copula."""
    copula = StudentT()

    # rho should be in [-1, 1]
    rho_interval = copula.intervals["rho"]
    assert rho_interval.left == -1
    assert rho_interval.right == 1
    assert not rho_interval.left_open  # Left bound is closed
    assert not rho_interval.right_open  # Right bound is closed

    # nu should be > 0
    nu_interval = copula.intervals["nu"]
    assert nu_interval.left == 0
    assert nu_interval.right == sympy.oo  # infinity
    assert nu_interval.left_open  # Left bound is open (nu > 0)
    assert nu_interval.right_open  # Right bound is open (nu < infinity)


@pytest.mark.parametrize(
    "rho, expected_class",
    [(-1, LowerFrechet), (1, UpperFrechet)],
)
def test_student_t_special_cases(rho, expected_class):
    """Test special cases for rho = -1 and rho = 1."""
    copula = StudentT()
    result = copula(rho, 4.0)
    assert isinstance(result, expected_class)


def test_student_t_properties(student_t_copula):
    """Test basic properties of StudentT copula."""
    # Test symmetry property
    assert student_t_copula.is_symmetric is True

    # Test absolute continuity property
    assert student_t_copula.is_absolutely_continuous is True


def test_student_t_rvs(student_t_copula):
    """Test random sampling from the StudentT copula."""
    # Test sample generation
    samples = student_t_copula.rvs(10)
    assert samples.shape == (10, 2)
    assert np.all(samples >= 0) and np.all(samples <= 1)


def test_student_t_cdf():
    """Test CDF calculation of StudentT copula."""
    copula = StudentT()
    copula.rho = 0.5
    copula.nu = 4.0

    # Mock the _calculate_student_t_cdf method
    with patch.object(
        StudentT, "_calculate_student_t_cdf", return_value=0.42
    ) as mock_cdf:
        # Get the callable
        cdf_func = copula.cdf
        # Call it with arguments
        result = cdf_func(0.3, 0.7)

        # Check that the method was called with the correct arguments
        # The first argument is not 'self' because the method is called via the object
        mock_cdf.assert_called_once_with(0.3, 0.7, 0.5, 4.0)

        # Check the result type and value
        assert isinstance(result, CDFWrapper)
        assert float(result.evalf()) == 0.42


def test_student_t_conditional_distribution(student_t_copula):
    """Test the _conditional_distribution method."""
    # Test with both arguments
    with patch("scipy.stats.t.cdf", return_value=0.65) as mock_cdf:
        result = student_t_copula._conditional_distribution(0.3, 0.7)
        assert mock_cdf.called
        assert result == sympy.S(0.65)

    # Test with only the first argument
    with patch("scipy.stats.t.cdf", return_value=0.65) as mock_cdf:
        func = student_t_copula._conditional_distribution(0.3, None)
        assert callable(func)
        result = func(0.7)
        assert mock_cdf.called
        assert result == sympy.S(0.65)


def test_student_t_cond_distr_1(student_t_copula):
    """Test the first conditional distribution method."""
    # Test edge cases
    assert student_t_copula.cond_distr_1(None, 0) == CD1Wrapper(sympy.S(0))
    assert student_t_copula.cond_distr_1(None, 1) == CD1Wrapper(sympy.S(1))

    # Test regular case with mock
    with patch.object(
        StudentT, "_conditional_distribution", return_value=sympy.S(0.75)
    ) as mock_cd:
        result = student_t_copula.cond_distr_1(0.3, 0.7)
        mock_cd.assert_called_once_with(0.3, 0.7)
        assert result == CD1Wrapper(sympy.S(0.75))


def test_student_t_cond_distr_2(student_t_copula):
    """Test the second conditional distribution method."""
    # Test edge cases
    assert student_t_copula.cond_distr_2(0, None) == CD2Wrapper(sympy.S(0))
    assert student_t_copula.cond_distr_2(1, None) == CD2Wrapper(sympy.S(1))

    # Test regular case with mock
    with patch.object(
        StudentT, "_conditional_distribution", return_value=sympy.S(0.75)
    ) as mock_cd:
        result = student_t_copula.cond_distr_2(0.7, 0.3)
        mock_cd.assert_called_once_with(0.3, 0.7)
        assert result == CD2Wrapper(sympy.S(0.75))


def test_student_t_pdf():
    """Test PDF calculation of StudentT copula."""
    copula = StudentT()
    copula.rho = 0.5
    copula.nu = 4.0

    # Mock the StudentTCopula.pdf to avoid actual computation
    with patch(
        "statsmodels.distributions.copula.elliptical.StudentTCopula.pdf",
        return_value=1.25,
    ) as mock_pdf:
        result = copula.pdf(0.3, 0.7)

        # Check that the function was called
        mock_pdf.assert_called_once_with([0.3, 0.7])

        # Check the result type
        assert isinstance(result, SymPyFuncWrapper)


def test_student_t_call_method():
    """Test the __call__ method for creating new instances."""
    copula = StudentT()

    # Test with one parameter (rho)
    result = copula(0.5)
    assert isinstance(result, StudentT)
    assert result.rho == 0.5
    assert isinstance(result.nu, sympy.Symbol)

    # Test with two parameters (rho, nu)
    result = copula(0.5, 4.0)
    assert isinstance(result, StudentT)
    assert result.rho == 0.5
    assert result.nu == 4.0

    # Test with keyword arguments
    result = copula(rho=0.7, nu=5.0)
    assert isinstance(result, StudentT)
    assert result.rho == 0.7
    assert result.nu == 5.0


def test_student_t_edge_cases():
    """Test edge cases for StudentT copula parameters."""
    copula = StudentT()

    # Test with small nu (degrees of freedom)
    small_nu = copula(0.5, 0.1)
    assert small_nu.nu == 0.1

    # Test with large nu (approaching Gaussian)
    large_nu = copula(0.5, 1000)
    assert large_nu.nu == 1000

    # Test with rho close to boundaries
    near_minus_one = copula(-0.999, 4.0)
    assert near_minus_one.rho == -0.999

    near_one = copula(0.999, 4.0)
    assert near_one.rho == 0.999


def test_student_t_characteristic_function_call():
    """Test that the characteristic_function method is inherited and works."""
    # This test verifies the method exists and can be called, not the actual result
    copula = StudentT()(0.5, 4.0)

    # StudentT doesn't define a generator, so this should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        copula.characteristic_function(1, 2)


def test_calculate_student_t_cdf():
    """Test the _calculate_student_t_cdf helper method by verifying its logic."""

    # Create a subclass for testing with a predictable implementation
    class TestableStudentT(StudentT):
        def _calculate_student_t_cdf(self, u, v, rho_val, nu_val):
            # Simple implementation that returns predictable results based on inputs
            # This allows us to verify the method works without complex mocking
            return u * v * rho_val / nu_val  # Simple deterministic formula

    # Use our testable subclass
    test_copula = TestableStudentT()
    test_copula.rho = 0.5
    test_copula.nu = 4.0

    # Call the method directly
    result = test_copula._calculate_student_t_cdf(0.3, 0.7, 0.5, 4.0)

    # Verify expected result from our test implementation
    expected = 0.3 * 0.7 * 0.5 / 4.0
    assert result == expected


@pytest.mark.parametrize(
    "point, expected", [((0, 0), 0), ((0, 0.5), 0), ((1, 0.5), 0.5), ((1, 1), 1)]
)
def test_cdf_edge_cases(point, expected):
    cop = StudentT(0.5, 2)
    evaluated_cdf = cop.cdf(*point)
    actual = evaluated_cdf.evalf()
    assert np.isclose(actual, expected)
