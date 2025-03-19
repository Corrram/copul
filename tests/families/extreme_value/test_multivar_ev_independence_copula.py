"""
Tests for the MultivariateExtremeValueCopula class.
"""

import numpy as np
import pytest

from copul.families.extreme_value.multivar_ev_independence_copula import (
    MultivariateExtremeIndependenceCopula,
)


@pytest.fixture
def copula2d():
    """Create a 2D test copula instance."""
    return MultivariateExtremeIndependenceCopula(dimension=2)


@pytest.fixture
def copula3d():
    """Create a 3D test copula instance."""
    return MultivariateExtremeIndependenceCopula(dimension=3)


@pytest.fixture
def copula4d():
    """Create a 4D test copula instance."""
    return MultivariateExtremeIndependenceCopula(dimension=4)


# Initialization tests
def test_default_init():
    """Test initialization with default parameters."""
    copula = MultivariateExtremeIndependenceCopula(2)
    assert copula.dim == 2


def test_custom_dimension_init():
    """Test initialization with custom dimension."""
    copula = MultivariateExtremeIndependenceCopula(dimension=4)
    assert copula.dim == 4
    assert len(copula.u_symbols) == 4


def test_call_method(copula3d):
    """Test __call__ method creates a new instance."""
    new_copula = copula3d()
    assert new_copula.dim == copula3d.dim
    assert new_copula is not copula3d  # Should be a new instance


# Property tests
def test_is_absolutely_continuous(copula3d):
    """Test is_absolutely_continuous property."""
    assert copula3d.is_absolutely_continuous is True


def test_is_symmetric(copula3d):
    """Test is_symmetric property."""
    assert copula3d.is_symmetric is True


# Extreme value function tests
def test_compute_extreme_value_function(copula3d):
    """Test _compute_extreme_value_function method."""
    # Test various points
    assert copula3d._compute_extreme_value_function([0.3, 0.4, 0.5]) == 0.3 * 0.4 * 0.5
    assert copula3d._compute_extreme_value_function([1, 1, 1]) == 1
    assert copula3d._compute_extreme_value_function([0, 0.5, 0.8]) == 0


def test_extreme_value_function_for_different_dimensions():
    """Test that extreme value function works correctly for different dimensions."""
    # Test higher dimensions
    for dim in [2, 3, 4, 5]:
        copula = MultivariateExtremeIndependenceCopula(dimension=dim)

        # Generate test values
        u_values = [0.5] * dim

        # Compute extreme value function
        ev_value = copula._compute_extreme_value_function(u_values)

        # Expected value is 0.5^dim
        expected = 0.5**dim
        assert abs(ev_value - expected) < 1e-10


# CDF tests
def test_cdf_boundary_cases(copula3d):
    """Test CDF computation at boundary cases."""
    # Test boundary cases
    assert float(copula3d.cdf(0, 0.5, 0.8)) == 0
    assert float(copula3d.cdf(0.5, 0, 0.8)) == 0
    assert float(copula3d.cdf(0.5, 0.8, 0)) == 0
    assert float(copula3d.cdf(1, 1, 1)) == 1


def test_cdf_interior_point(copula3d):
    """Test CDF computation at interior points."""
    # Test interior point
    assert abs(float(copula3d.cdf(0.3, 0.4, 0.5)) - (0.3 * 0.4 * 0.5)) < 1e-10


def test_cdf_different_dimensions():
    """Test that the CDF formula is correct for various dimensions."""
    # Test 2D
    copula2d = MultivariateExtremeIndependenceCopula(dimension=2)
    assert abs(float(copula2d.cdf(0.3, 0.4)) - (0.3 * 0.4)) < 1e-10

    # Test 3D
    copula3d = MultivariateExtremeIndependenceCopula(dimension=3)
    assert abs(float(copula3d.cdf(0.3, 0.4, 0.5)) - (0.3 * 0.4 * 0.5)) < 1e-10

    # Test 4D
    copula4d = MultivariateExtremeIndependenceCopula(dimension=4)
    assert (
        abs(float(copula4d.cdf(0.3, 0.4, 0.5, 0.6)) - (0.3 * 0.4 * 0.5 * 0.6)) < 1e-10
    )


# Vectorized CDF tests
def test_cdf_vectorized(copula3d):
    """Test vectorized CDF computation."""
    # Create test data
    u = np.array([0, 0.3, 0.7, 1])
    v = np.array([0, 0.4, 0.6, 1])
    w = np.array([0, 0.5, 0.8, 1])

    # Compute vectorized CDF
    cdf_values = copula3d.cdf_vectorized(u, v, w)

    # Check shape
    assert cdf_values.shape == u.shape

    # Check values
    expected_values = u * v * w
    np.testing.assert_allclose(cdf_values, expected_values)


def test_cdf_vectorized_error_on_wrong_args(copula3d):
    """Test that vectorized CDF raises error with wrong number of arguments."""
    # Create test data
    u = np.array([0, 0.3, 0.7, 1])
    v = np.array([0, 0.4, 0.6, 1])

    # Test wrong number of arguments
    with pytest.raises(ValueError):
        copula3d.cdf_vectorized(u, v)  # Missing one argument


# PDF tests
def test_pdf(copula3d):
    """Test PDF computation."""
    # PDF should be constant 1 everywhere
    assert float(copula3d.pdf) == 1


def test_pdf_vectorized(copula3d):
    """Test vectorized PDF computation."""
    # Create test data
    u = np.array([0.1, 0.3, 0.7, 0.9])
    v = np.array([0.2, 0.4, 0.6, 0.8])
    w = np.array([0.3, 0.5, 0.7, 0.9])

    # Compute vectorized PDF
    pdf_values = copula3d.pdf_vectorized(u, v, w)

    # Check shape
    assert pdf_values.shape == u.shape

    # Check values (all should be 1)
    np.testing.assert_allclose(pdf_values, np.ones_like(u))


# Dependence measures tests
def test_dependence_measures(copula3d):
    """Test all dependence measures."""
    # All dependence measures should be 0 for independence
    assert copula3d.tau() == 0
    assert copula3d.rho() == 0
    assert copula3d.lambda_L() == 0
    assert copula3d.lambda_U() == 0


# Random sampling tests
def test_rvs_shape(copula3d):
    """Test random sample generation shape."""
    # Test sample shape
    samples = copula3d.rvs(100)
    assert samples.shape == (100, 3)

    # All values should be in [0, 1]
    assert np.all(samples >= 0)
    assert np.all(samples <= 1)


def test_rvs_reproducibility(copula3d):
    """Test that random generation is reproducible with same seed."""
    # Test with random state for reproducibility
    samples1 = copula3d.rvs(10, random_state=42)
    samples2 = copula3d.rvs(10, random_state=42)
    np.testing.assert_array_equal(samples1, samples2)
