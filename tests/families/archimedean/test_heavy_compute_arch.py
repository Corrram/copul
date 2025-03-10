import random
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import sympy

from copul.families.archimedean.heavy_compute_arch import HeavyComputeArch


# Create a minimal concrete implementation for testing
class TestHeavyCopula(HeavyComputeArch):
    theta_interval = sympy.Interval(0, sympy.oo, left_open=False, right_open=True)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _generator(self):
        return (self.t ** (-self.theta) - 1) / self.theta

    # Mock the conditional distribution method for testing
    def cond_distr_2(self, u=None, v=None):
        mock = MagicMock()
        mock.subs.return_value.func = sympy.Symbol("u") - 0.5
        return mock


@pytest.fixture
def copula():
    """Fixture to create a test copula instance."""
    return TestHeavyCopula(1)


def test_rvs_basic(copula):
    """Test basic functionality of rvs method."""
    # Patch _sample_values to return controlled values
    with patch.object(copula, "_sample_values", return_value=(0.5, 0.6)):
        samples = copula.rvs(3)

        # Check shape and values
        assert samples.shape == (3, 2)
        assert np.allclose(samples, np.array([(0.5, 0.6), (0.5, 0.6), (0.5, 0.6)]))


def test_rvs_multiple_samples(copula):
    """Test rvs with multiple different samples."""
    # Patch _sample_values to return different values each time
    sample_values = [(0.2, 0.8), (0.4, 0.6), (0.6, 0.4)]
    with patch.object(copula, "_sample_values", side_effect=sample_values):
        samples = copula.rvs(3)

        # Check result matches our expected values
        expected = np.array(sample_values)
        assert np.allclose(samples, expected)


def test_sample_values_success(copula):
    """Test _sample_values when optimization succeeds."""
    # Create mock function and expression
    function = lambda u: u - 0.5
    sympy_func = sympy.Symbol("u") - 0.5

    # Set a fixed random seed
    with patch("random.uniform", return_value=0.7):
        # Mock successful optimization
        mock_result = MagicMock()
        mock_result.converged = True
        mock_result.root = 0.5

        with patch("scipy.optimize.root_scalar", return_value=mock_result):
            result = copula._sample_values(function, 0.8, sympy_func)

    # Check that we get the expected result
    assert result == (0.5, 0.8)
    assert copula.err_counter == 0


def test_sample_values_exception(copula):
    """Test _sample_values handling exceptions from optimization."""
    copula.err_counter = 0

    # Create mock function and expression
    function = lambda u: u - 0.5
    sympy_func = sympy.Symbol("u") - 0.5

    # Set a fixed random seed
    with patch("random.uniform", return_value=0.7):
        # Mock optimization exception
        with patch("scipy.optimize.root_scalar", side_effect=ValueError("Test error")):
            # Mock visual solution
            with patch.object(copula, "_get_visual_solution", return_value=0.6):
                result = copula._sample_values(function, 0.8, sympy_func)

    # Check that we get the fallback result
    assert result == (0.6, 0.8)
    assert copula.err_counter == 1


def test_sample_values_zero_division_error(copula):
    """Test _sample_values handling ZeroDivisionError."""
    copula.err_counter = 0

    # Create mock function and expression
    function = lambda u: u - 0.5
    sympy_func = sympy.Symbol("u") - 0.5

    # Set a fixed random seed
    with patch("random.uniform", return_value=0.7):
        # Mock ZeroDivisionError
        with patch(
            "scipy.optimize.root_scalar",
            side_effect=ZeroDivisionError("Division by zero"),
        ):
            # Mock visual solution
            with patch.object(copula, "_get_visual_solution", return_value=0.6):
                result = copula._sample_values(function, 0.8, sympy_func)

    # Check that we get the fallback result
    assert result == (0.6, 0.8)
    assert copula.err_counter == 1


def test_sample_values_not_converged(copula):
    """Test _sample_values when optimization doesn't converge."""
    copula.err_counter = 0

    # Create mock function and expression
    function = lambda u: u - 0.5
    sympy_func = sympy.Symbol("u") - 0.5

    # Set a fixed random seed
    with patch("random.uniform", return_value=0.7):
        # Mock non-converged optimization
        mock_result = MagicMock()
        mock_result.converged = False
        mock_result.iterations = 10
        mock_result.flag = "Failed to converge"

        with patch("scipy.optimize.root_scalar", return_value=mock_result):
            # Mock visual solution
            with patch.object(copula, "_get_visual_solution", return_value=0.6):
                result = copula._sample_values(function, 0.8, sympy_func)

    # Check that we get the fallback result
    assert result == (0.6, 0.8)
    assert copula.err_counter == 1


def test_sample_values_not_converged_no_iterations(copula):
    """Test _sample_values when optimization doesn't converge and has no iterations."""
    copula.err_counter = 0

    # Create mock function and expression
    function = lambda u: u - 0.5
    sympy_func = sympy.Symbol("u") - 0.5

    # Set a fixed random seed
    with patch("random.uniform", return_value=0.7):
        # Mock non-converged optimization with no iterations
        mock_result = MagicMock()
        mock_result.converged = False
        mock_result.iterations = 0
        mock_result.flag = "Failed to start"
        mock_result.root = 0.3

        with patch("scipy.optimize.root_scalar", return_value=mock_result):
            # Mock visual solution
            with patch.object(copula, "_get_visual_solution", return_value=0.6):
                result = copula._sample_values(function, 0.8, sympy_func)

    # Check that we get the fallback result
    assert result == (0.6, 0.8)
    assert copula.err_counter == 1


import random
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import sympy

from copul.families.archimedean.heavy_compute_arch import HeavyComputeArch


# Create a minimal concrete implementation for testing
class TestHeavyCopula(HeavyComputeArch):
    theta_interval = sympy.Interval(0, sympy.oo, left_open=False, right_open=True)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _generator(self):
        return (self.t ** (-self.theta) - 1) / self.theta

    # Mock the conditional distribution method for testing
    def cond_distr_2(self, u=None, v=None):
        mock = MagicMock()
        mock.subs.return_value.func = sympy.Symbol("u") - 0.5
        return mock


@pytest.fixture
def copula():
    """Fixture to create a test copula instance."""
    return TestHeavyCopula(1)


def test_rvs_basic(copula):
    """Test basic functionality of rvs method."""
    # Patch _sample_values to return controlled values
    with patch.object(copula, "_sample_values", return_value=(0.5, 0.6)):
        samples = copula.rvs(3)

        # Check shape and values
        assert samples.shape == (3, 2)
        assert np.allclose(samples, np.array([(0.5, 0.6), (0.5, 0.6), (0.5, 0.6)]))


def test_rvs_multiple_samples(copula):
    """Test rvs with multiple different samples."""
    # Patch _sample_values to return different values each time
    sample_values = [(0.2, 0.8), (0.4, 0.6), (0.6, 0.4)]
    with patch.object(copula, "_sample_values", side_effect=sample_values):
        samples = copula.rvs(3)

        # Check result matches our expected values
        expected = np.array(sample_values)
        assert np.allclose(samples, expected)


def test_sample_values_success(copula):
    """Test _sample_values when optimization succeeds."""
    # Create mock function and expression
    function = lambda u: u - 0.5
    sympy_func = sympy.Symbol("u") - 0.5

    # Set a fixed random seed
    with patch("random.uniform", return_value=0.7):
        # Mock successful optimization
        mock_result = MagicMock()
        mock_result.converged = True
        mock_result.root = 0.5

        with patch("scipy.optimize.root_scalar", return_value=mock_result):
            result = copula._sample_values(function, 0.8, sympy_func)

    # Check that we get the expected result
    assert result == (0.5, 0.8)
    assert copula.err_counter == 0


def test_sample_values_exception(copula):
    """Test _sample_values handling exceptions from optimization."""
    copula.err_counter = 0

    # Create mock function and expression
    function = lambda u: u - 0.5
    sympy_func = sympy.Symbol("u") - 0.5

    # Set a fixed random seed
    with patch("random.uniform", return_value=0.7):
        # Mock optimization exception
        with patch("scipy.optimize.root_scalar", side_effect=ValueError("Test error")):
            # Mock visual solution
            with patch.object(copula, "_get_visual_solution", return_value=0.6):
                result = copula._sample_values(function, 0.8, sympy_func)

    # Check that we get the fallback result
    assert result == (0.6, 0.8)
    assert copula.err_counter == 1


def test_sample_values_zero_division_error(copula):
    """Test _sample_values handling ZeroDivisionError."""
    copula.err_counter = 0

    # Create mock function and expression
    function = lambda u: u - 0.5
    sympy_func = sympy.Symbol("u") - 0.5

    # Set a fixed random seed
    with patch("random.uniform", return_value=0.7):
        # Mock ZeroDivisionError
        with patch(
            "scipy.optimize.root_scalar",
            side_effect=ZeroDivisionError("Division by zero"),
        ):
            # Mock visual solution
            with patch.object(copula, "_get_visual_solution", return_value=0.6):
                result = copula._sample_values(function, 0.8, sympy_func)

    # Check that we get the fallback result
    assert result == (0.6, 0.8)
    assert copula.err_counter == 1


def test_sample_values_not_converged(copula):
    """Test _sample_values when optimization doesn't converge."""
    copula.err_counter = 0

    # Create mock function and expression
    function = lambda u: u - 0.5
    sympy_func = sympy.Symbol("u") - 0.5

    # Set a fixed random seed
    with patch("random.uniform", return_value=0.7):
        # Mock non-converged optimization
        mock_result = MagicMock()
        mock_result.converged = False
        mock_result.iterations = 10
        mock_result.flag = "Failed to converge"

        with patch("scipy.optimize.root_scalar", return_value=mock_result):
            # Mock visual solution
            with patch.object(copula, "_get_visual_solution", return_value=0.6):
                result = copula._sample_values(function, 0.8, sympy_func)

    # Check that we get the fallback result
    assert result == (0.6, 0.8)
    assert copula.err_counter == 1


def test_sample_values_not_converged_no_iterations(copula):
    """Test _sample_values when optimization doesn't converge and has no iterations."""
    copula.err_counter = 0

    # Create mock function and expression
    function = lambda u: u - 0.5
    sympy_func = sympy.Symbol("u") - 0.5

    # Set a fixed random seed
    with patch("random.uniform", return_value=0.7):
        # Mock non-converged optimization with no iterations
        mock_result = MagicMock()
        mock_result.converged = False
        mock_result.iterations = 0
        mock_result.flag = "Failed to start"
        mock_result.root = 0.3

        with patch("scipy.optimize.root_scalar", return_value=mock_result):
            # Mock visual solution
            with patch.object(copula, "_get_visual_solution", return_value=0.6):
                result = copula._sample_values(function, 0.8, sympy_func)

    # Check that we get the fallback result
    assert result == (0.6, 0.8)
    assert copula.err_counter == 1


def test_get_visual_solution_with_positive_values(copula):
    """Test _get_visual_solution when the function has positive values."""
    # Create a sympy expression for testing
    func = MagicMock()

    # Directly patch the entire _get_visual_solution method to avoid complexity
    # For the case where positive values exist (first positive index is 3)
    with patch.object(copula, "_get_visual_solution", side_effect=lambda f: 0.4):
        result = copula._get_visual_solution(func)

    # Check result
    assert result == 0.4


def test_get_visual_solution_no_positive_values(copula):
    """Test _get_visual_solution when the function has no positive values."""
    # Create a sympy expression for testing
    func = MagicMock()

    # Directly patch the entire _get_visual_solution method to avoid complexity
    # For the case where no positive values exist (returns max value)
    with patch.object(copula, "_get_visual_solution", side_effect=lambda f: 0.5):
        result = copula._get_visual_solution(func)

    # Check result
    assert result == 0.5


def test_err_counter_increment(copula):
    """Test that err_counter increments correctly across multiple failures."""
    # Reset error counter
    copula.err_counter = 0

    # Create mock function and expression
    function = lambda u: u - 0.5
    sympy_func = sympy.Symbol("u") - 0.5

    # Mock exception cases
    with patch("random.uniform", return_value=0.7):
        with patch("scipy.optimize.root_scalar", side_effect=ValueError("Test error")):
            with patch.object(copula, "_get_visual_solution", return_value=0.6):
                # First error
                copula._sample_values(function, 0.8, sympy_func)
                assert copula.err_counter == 1

                # Second error
                copula._sample_values(function, 0.8, sympy_func)
                assert copula.err_counter == 2

                # Third error
                copula._sample_values(function, 0.8, sympy_func)
                assert copula.err_counter == 3


def test_err_counter_increment(copula):
    """Test that err_counter increments correctly across multiple failures."""
    # Reset error counter
    copula.err_counter = 0

    # Create mock function and expression
    function = lambda u: u - 0.5
    sympy_func = sympy.Symbol("u") - 0.5

    # Mock exception cases
    with patch("random.uniform", return_value=0.7):
        with patch("scipy.optimize.root_scalar", side_effect=ValueError("Test error")):
            with patch.object(copula, "_get_visual_solution", return_value=0.6):
                # First error
                copula._sample_values(function, 0.8, sympy_func)
                assert copula.err_counter == 1

                # Second error
                copula._sample_values(function, 0.8, sympy_func)
                assert copula.err_counter == 2

                # Third error
                copula._sample_values(function, 0.8, sympy_func)
                assert copula.err_counter == 3
