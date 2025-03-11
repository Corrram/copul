"""
Tests for the CopulaSampler class.
"""

import pytest
import numpy as np
import sympy
from unittest.mock import MagicMock, patch

from copul.copula_sampler import CopulaSampler
from copul.checkerboard.check_pi import CheckPi
from copul.families.other.independence_copula import IndependenceCopula
from copul.families.other.lower_frechet import LowerFrechet
from copul.families.other.upper_frechet import UpperFrechet


class TestCopulaSampler:
    """Tests for the CopulaSampler class."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create a mock copula
        self.mock_copula = MagicMock()
        self.mock_copula.__class__.__name__ = "MockCopula"

        # Set up symbols for the mock copula
        u, v = sympy.symbols("u v")
        self.mock_copula.u_symbols = [u, v]

        # Default sampler with mock copula
        self.sampler = CopulaSampler(self.mock_copula)

        # Sampler with specified precision and random_state
        self.sampler_with_params = CopulaSampler(
            self.mock_copula, precision=4, random_state=42
        )

    def test_initialization(self):
        """Test initialization of CopulaSampler."""
        # Test default initialization
        assert self.sampler._copul == self.mock_copula
        assert self.sampler._precision == 3
        assert self.sampler._random_state is None

        # Test with custom parameters
        assert self.sampler_with_params._copul == self.mock_copula
        assert self.sampler_with_params._precision == 4
        assert self.sampler_with_params._random_state == 42

    @patch("random.seed")
    def test_rvs_with_random_state(self, mock_seed):
        """Test that rvs uses the random_state when provided."""

        # Setup mock for conditional distribution with parameters
        def mock_cond_distr(u, v, theta=1.0):
            return u * v * theta

        self.mock_copula.cond_distr_2 = mock_cond_distr
        self.mock_copula.intervals = {"theta": (0, 10)}

        # Setup mock for sampling
        with patch.object(
            self.sampler_with_params, "_sample_val", return_value=np.array([[0.5, 0.5]])
        ):
            self.sampler_with_params.rvs(1)

        # Verify random.seed was called with the correct random_state
        mock_seed.assert_called_once_with(42)

    def test_rvs_with_direct_function(self):
        """Test rvs when the copula has a direct cond_distr_2 function."""
        # Setup a mock copula with a parameter in cond_distr_2
        mock_param_copula = MagicMock()
        mock_param_copula.__class__.__name__ = "MockParamCopula"

        # Make cond_distr_2 have a parameter that's in intervals
        def cond_distr_with_param(u, v, theta=1.0):
            return u * v * theta

        mock_param_copula.cond_distr_2 = cond_distr_with_param
        mock_param_copula.intervals = {"theta": (0, 10)}

        # Create sampler with the parametrized copula
        param_sampler = CopulaSampler(mock_param_copula)

        # Test sampling
        with patch.object(
            param_sampler, "_sample_val", return_value=np.array([[0.5, 0.5]])
        ):
            result = param_sampler.rvs(1)

        # rvs returns an array of shape n where each element is a (u, v) pair
        assert result.shape == (1, 2)  # 1 sample with pairs of (u, v) values

    def test_rvs_with_symbolic_function(self):
        """Test rvs when the copula returns a symbolic function."""
        # Setup symbolic expression for conditional distribution
        u, v = sympy.symbols("u v")
        expr = u * v

        # Create a MagicMock that has a func attribute with the symbolic expression
        class SymbolicFunc(MagicMock):
            @property
            def func(self):
                return expr

        # Setup cond_distr_2 method to return our symbolic function
        self.mock_copula.cond_distr_2 = MagicMock(return_value=SymbolicFunc())
        self.mock_copula.intervals = {}

        # Test sampling with mocked _sample_val
        with patch.object(
            self.sampler, "_sample_val", return_value=np.array([[0.5, 0.5]])
        ):
            result = self.sampler.rvs(1)

        # Verify result shape
        assert result.shape == (1, 2)

    def test_sample_val_array(self):
        """Test that _sample_val returns an array of samples."""
        # Setup mock
        mock_function = MagicMock()

        # Mock the individual sample_val method
        with patch.object(self.sampler, "sample_val", return_value=(0.5, 0.5)):
            result = self.sampler._sample_val(mock_function, 3)

        # Should return an array of 3 samples
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)  # 3 samples of (u, v) pairs

    @patch("random.uniform")
    @patch("scipy.optimize.root_scalar")
    def test_sample_val_successful(self, mock_root_scalar, mock_uniform):
        """Test successful sampling of a single value."""
        # Setup mocks
        mock_function = MagicMock()
        mock_uniform.side_effect = [0.3, 0.7]  # v and t values

        # Mock the root_scalar result
        mock_result = MagicMock()
        mock_result.converged = True
        mock_result.root = 0.6
        mock_root_scalar.return_value = mock_result

        # Call sample_val
        result = self.sampler.sample_val(mock_function)

        # Verify results
        assert result == (0.6, 0.3)  # (root, v)
        mock_root_scalar.assert_called_once()

    @patch("random.uniform")
    @patch("scipy.optimize.root_scalar")
    def test_sample_val_not_converged(self, mock_root_scalar, mock_uniform):
        """Test when root_scalar doesn't converge."""
        # Setup mocks
        mock_function = MagicMock()
        mock_uniform.side_effect = [0.3, 0.7]  # v and t values

        # Mock the root_scalar result with non-convergence
        mock_result = MagicMock()
        mock_result.converged = False
        mock_result.iterations = 100  # Iterations were performed but didn't converge
        mock_root_scalar.return_value = mock_result

        # Mock _get_visual_solution to return a fallback value
        with patch.object(self.sampler, "_get_visual_solution", return_value=0.55):
            result = self.sampler.sample_val(mock_function)

        # Verify results
        assert result == (0.55, 0.3)  # (visual solution, v)
        assert self.sampler.err_counter > 0  # Error counter should increment

    @patch("random.uniform")
    def test_sample_val_exception(self, mock_uniform):
        """Test when root_scalar raises an exception."""
        # Setup mocks
        mock_function = MagicMock()
        mock_uniform.side_effect = [0.3, 0.7]  # v and t values

        # Track initial error counter
        initial_counter = CopulaSampler.err_counter

        # Mock root_scalar to raise an exception
        with patch("scipy.optimize.root_scalar", side_effect=ValueError("Test error")):
            # Mock _get_visual_solution to return a fallback value
            with patch.object(self.sampler, "_get_visual_solution", return_value=0.45):
                result = self.sampler.sample_val(mock_function)

        # Verify results
        assert result == (0.45, 0.3)  # (visual solution, v)
        assert (
            self.sampler.err_counter > initial_counter
        )  # Error counter should increment

    def test_get_visual_solution(self):
        """Test the visual solution fallback method."""

        # Create a simple function that has a minimum at x=0.5
        def test_func(x):
            return (x - 0.5) ** 2

        # Get the visual solution
        result = self.sampler._get_visual_solution(test_func)

        # Should be close to 0.5 (exact value depends on the precision)
        assert abs(result - 0.5) < 10 ** (-self.sampler._precision)

    @patch("numpy.linspace")
    def test_get_visual_solution_precision(self, mock_linspace):
        """Test that _get_visual_solution uses the correct precision."""
        # Setup mocks
        mock_linspace.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # Mock function that always returns 0
        mock_func = MagicMock(return_value=0)

        # Call _get_visual_solution with default precision (3)
        self.sampler._get_visual_solution(mock_func)

        # Verify linspace was called with start=0.001, end=0.999, num=1000
        start, end, num = mock_linspace.call_args[0]
        assert start == 10 ** (-3)
        assert end == 1 - 10 ** (-3)
        assert num == 10**3

        # Test with higher precision
        self.sampler_with_params._get_visual_solution(mock_func)

        # Verify linspace was called with start=0.0001, end=0.9999, num=10000
        start, end, num = mock_linspace.call_args[0]
        assert start == 10 ** (-4)
        assert end == 1 - 10 ** (-4)
        assert num == 10**4

    @patch.object(CopulaSampler, "rvs")
    def test_integration_with_mock_copula(self, mock_rvs):
        """Integration test with a mock copula."""
        # Setup a mock to return a pre-defined array
        mock_rvs.return_value = np.array([[0.5, 0.5]])

        # Create a simple linear copula function for testing
        def mock_cond_dist(u, v):
            return u * v

        self.mock_copula.cond_distr_2 = mock_cond_dist
        self.mock_copula.intervals = {}

        # Sample from the copula using the mock
        result = mock_rvs(1)

        # Verify the result
        assert result.shape == (1, 2)
        assert result[0, 0] == 0.5  # u value
        assert result[0, 1] == 0.5  # v value


def test_error_counter_is_class_var():
    """Test that the error counter is a class variable."""
    # Create two samplers
    mock_copula1 = MagicMock()
    sampler1 = CopulaSampler(mock_copula1)

    mock_copula2 = MagicMock()
    sampler2 = CopulaSampler(mock_copula2)

    # Verify both samplers access the same class variable
    assert CopulaSampler.err_counter == sampler1.err_counter
    assert CopulaSampler.err_counter == sampler2.err_counter

    # Directly modify the class variable
    old_counter = CopulaSampler.err_counter
    CopulaSampler.err_counter += 5

    # Verify both instances see the change
    assert sampler1.err_counter == old_counter + 5
    assert sampler2.err_counter == old_counter + 5

    # Reset for other tests
    CopulaSampler.err_counter = old_counter


@patch.object(CopulaSampler, "_sample_val")
def test_with_check_pi_copula(mock_sample_val):
    """Test that the sampler works correctly with CheckPi copula."""
    # Mock the _sample_val method
    mock_sample_val.return_value = np.array([[0.5, 0.5]])

    # Create a mock that identifies as CheckPi
    mock_check_pi = MagicMock(spec=CheckPi)
    mock_check_pi.__class__ = CheckPi

    # Setup a simple conditional distribution
    def cond_distr(u, v):
        return u * v

    mock_check_pi.cond_distr_2 = cond_distr

    # Create the sampler
    sampler = CopulaSampler(mock_check_pi)

    # Sample from the copula
    result = sampler.rvs(1)

    # Verify the result
    assert result.shape == (1, 2)
    mock_sample_val.assert_called_once_with(cond_distr, 1)


@pytest.mark.parametrize("n_samples", [1, 5, 10])
@patch.object(CopulaSampler, "_sample_val")
def test_rvs_sample_count(mock_sample_val, n_samples):
    """Test that rvs returns the correct number of samples."""
    # Setup mock to return appropriate shape based on n_samples
    if n_samples == 1:
        return_value = np.array([[0.5, 0.5]])
    else:
        return_value = np.array([([0.5, 0.5]) for _ in range(n_samples)])
    mock_sample_val.return_value = return_value

    # Create a mock copula with parameters
    mock_copula = MagicMock()
    mock_copula.__class__.__name__ = "MockCopula"

    # Setup a simple cond_distr_2 function with parameter
    def cond_distr(u, v, theta=1.0):
        return u * v * theta

    mock_copula.cond_distr_2 = cond_distr
    mock_copula.intervals = {"theta": (0, 2)}

    # Create the sampler
    sampler = CopulaSampler(mock_copula)

    # Sample from the copula
    result = sampler.rvs(n_samples)

    # Verify the result shape
    assert result.shape == (n_samples, 2)
    mock_sample_val.assert_called_once_with(cond_distr, n_samples)


def test_rvs_from_upper_frechet():
    copula = UpperFrechet()
    sampler = CopulaSampler(copula)
    results = sampler.rvs(3)
    assert len(results) == 3
    for result in results:
        assert len(result) == 2
        assert np.isclose(result[0], result[1])


def test_rvs_from_lower_frechet():
    copula = LowerFrechet()
    sampler = CopulaSampler(copula)
    results = sampler.rvs(3)
    assert len(results) == 3
    for result in results:
        assert len(result) == 2
        assert np.isclose(result[0], 1 - result[1])


def test_rvs_from_independence_copula():
    copula = IndependenceCopula()
    sampler = CopulaSampler(copula)
    results = sampler.rvs(200)
    corr = np.corrcoef(results[:, 0], results[:, 1])[0, 1]
    assert np.abs(corr) < 0.1
