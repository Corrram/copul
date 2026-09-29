from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import sympy

from copul.family.archimedean.heavy_compute_arch import HeavyComputeArch


# Create a minimal concrete implementation for testing
class SampleHeavyCopula(HeavyComputeArch):
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
    return SampleHeavyCopula(1)


def test_rvs_uses_vectorized_sampler():
    """HeavyComputeArch families (Nelsen 20) sample by vectorized conditional inversion."""
    from copul.family.archimedean import Nelsen20

    cop = Nelsen20(0.5)
    samples = cop.rvs(200, random_state=0)
    assert samples.shape == (200, 2)
    assert np.all((samples >= 0) & (samples <= 1))
    assert np.array_equal(samples, cop.rvs(200, random_state=0))
    # V solves C_1(V | U) = W for the uniforms drawn by the generator
    rng = np.random.default_rng(0)
    u, w = rng.random(200), rng.random(200)
    assert np.allclose(samples[:, 0], u)
    assert np.allclose(cop.cond_distr_1(u, samples[:, 1]), w, atol=1e-8)


def test_sample_values_success(copula):
    """Test _sample_values when optimization succeeds."""

    # Create mock function and expression
    def function(u):
        return u - 0.5

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
    def function(u):
        return u - 0.5

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
    def function(u):
        return u - 0.5

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
    def function(u):
        return u - 0.5

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
    def function(u):
        return u - 0.5

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
