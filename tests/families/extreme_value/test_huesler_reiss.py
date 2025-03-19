import numpy as np
import pytest
import sympy as sp

from copul.families.extreme_value.huesler_reiss import HueslerReiss
from copul.families.other.biv_independence_copula import BivIndependenceCopula
from tests.family_representatives import family_representatives


@pytest.fixture
def huesler_reiss_copula():
    """Create a HueslerReiss copula with delta=1.0 for testing"""
    return HueslerReiss(1.0)


@pytest.fixture
def huesler_reiss_symbolic():
    """Create a HueslerReiss copula with symbolic delta for testing"""
    return HueslerReiss()


def test_hr_init():
    """Test initialization of HueslerReiss copula"""
    # Default initialization with symbol
    copula = HueslerReiss()
    assert hasattr(copula, "delta")
    assert isinstance(copula.delta, sp.Symbol)
    assert str(copula.delta) == "delta"

    # Initialization with parameter
    copula = HueslerReiss(2.0)
    assert hasattr(copula, "delta")
    assert copula.delta == 2.0


def test_hr_parameter_bounds():
    """Test parameter bounds of HueslerReiss copula"""
    copula = HueslerReiss()
    # Delta should be ≥ 0
    assert copula.intervals["delta"].left == 0
    assert copula.intervals["delta"].right == float("inf")
    assert not copula.intervals["delta"].left_open  # Left bound is closed
    assert copula.intervals["delta"].right_open  # Right bound is open


def test_hr_independence_special_case():
    """Test special case when delta=0 (should return Independence copula)"""
    # When delta=0, HueslerReiss becomes the Independence copula
    copula = HueslerReiss(delta=0)

    # Should return an Independence copula instance
    assert isinstance(copula, BivIndependenceCopula)

    # Test with positional argument too
    copula2 = HueslerReiss(0)
    assert isinstance(copula2, BivIndependenceCopula)


def test_hr_is_symmetric(huesler_reiss_copula):
    """Test symmetry property of HueslerReiss copula"""
    assert huesler_reiss_copula.is_symmetric is True


def test_hr_is_absolutely_continuous(huesler_reiss_copula):
    """Test absolute continuity property"""
    assert huesler_reiss_copula.is_absolutely_continuous is True


def test_hr_pickands_symbolic(huesler_reiss_symbolic):
    """Test Pickands dependence function structure with symbolic delta"""
    # The _pickands property should exist
    assert hasattr(huesler_reiss_symbolic, "_pickands")

    # Test that it's properly constructed with sympy expressions
    pickands_expr = huesler_reiss_symbolic._pickands
    assert isinstance(pickands_expr, sp.Expr)

    # For symbolic delta, the expression should include both t and delta symbols
    t = huesler_reiss_symbolic.t
    assert t in pickands_expr.free_symbols
    assert huesler_reiss_symbolic.delta in pickands_expr.free_symbols


def test_hr_pickands_concrete(huesler_reiss_copula):
    """Test Pickands dependence function structure with concrete delta value"""
    # The _pickands property should exist
    assert hasattr(huesler_reiss_copula, "_pickands")

    # Test that it's properly constructed with sympy expressions
    pickands_expr = huesler_reiss_copula._pickands
    assert isinstance(pickands_expr, sp.Expr)

    # For concrete delta, the expression should include t but not delta (it's substituted)
    t = huesler_reiss_copula.t
    assert t in pickands_expr.free_symbols
    # Delta should not be in free_symbols because it's a concrete value (1.0)
    assert len(pickands_expr.free_symbols) > 0


def test_hr_call_method():
    """Test __call__ method for creating new instances"""
    # Create base copula
    copula = HueslerReiss()

    # Update parameter using kwargs
    new_copula = copula(delta=2.0)

    # Original should be unchanged
    assert isinstance(copula.delta, sp.Symbol)

    # New instance should have updated parameter
    assert new_copula.delta == 2.0

    # Test with positional arg
    new_copula2 = copula(3.0)
    assert new_copula2.delta == 3.0

    # Test independence special case
    ind_copula = copula(0)
    assert isinstance(ind_copula, BivIndependenceCopula)


@pytest.mark.parametrize(
    "point, expected",
    [
        ((1, 0.5), 0.5),
        ((1, 1), 1),
        ((0, 0), 0),
        ((0, 0.5), 0),
    ],
)
def test_cdf_edge_cases_for_hr(point, expected):
    params = family_representatives["HueslerReiss"]
    cop = HueslerReiss(params)
    evaluated_cdf = cop.cdf(*point)
    assert np.isclose(float(evaluated_cdf), expected, atol=0)


def test_hr_pickands_values(huesler_reiss_copula):
    val1 = float(huesler_reiss_copula.pickands(0.1))
    val2 = float(huesler_reiss_copula.pickands(0.3))
    val3 = float(huesler_reiss_copula.pickands(0.5))
    val4 = float(huesler_reiss_copula.pickands(0.7))
    val5 = float(huesler_reiss_copula.pickands(0.9))
    assert val1 > val2 > val3
    assert val3 < val4 < val5
    assert val1 > 0.9
    assert val5 > 0.9
    assert val3 < 0.9


def test_cdf_close_to_one():
    cop = HueslerReiss(1)
    evaluated_cdf = float(cop.cdf(0.9, 0.9))
    assert np.isclose(evaluated_cdf, 0.9, atol=0.1)


def test_hr_edge_cases():
    """Test edge cases for parameter values"""
    # Very small delta (close to independence)
    copula_small = HueslerReiss(1e-10)
    assert copula_small.delta == 1e-10

    # Very large delta
    copula_large = HueslerReiss(1e10)
    assert copula_large.delta == 1e10


def test_hr_properties_inheritance():
    """Test that the copula inherits properties from ExtremeValueCopula"""
    copula = HueslerReiss(1.0)

    # Should have kendalls_tau and spearmans_rho methods
    assert hasattr(copula, "tau")
    assert hasattr(copula, "rho")
    assert callable(copula.tau)
    assert callable(copula.rho)

    # Should have CDF method
    assert hasattr(copula, "cdf")


def test_cdf_vectorized_scalar_inputs():
    """Test that cdf_vectorized works correctly with scalar inputs"""
    delta = 1.0
    copula = HueslerReiss(delta)

    # Test at some specific points
    u, v = 0.7, 0.3

    # Both methods should return the same result for scalar inputs
    cdf_result = float(copula.cdf(u, v))
    cdf_vec_result = float(copula.cdf_vectorized(u, v))

    assert np.isclose(cdf_result, cdf_vec_result, rtol=1e-10)


def test_cdf_vectorized_array_inputs():
    """Test that cdf_vectorized works correctly with array inputs"""
    delta = 1.5
    copula = HueslerReiss(delta)

    # Create arrays of u and v values
    u = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    v = np.array([0.2, 0.4, 0.6, 0.8, 0.95])

    # Calculate results with vectorized method
    vectorized_results = copula.cdf_vectorized(u, v)

    # Calculate results individually with standard cdf
    standard_results = np.array([float(copula.cdf(u_i, v_i)) for u_i, v_i in zip(u, v)])

    # Compare results
    assert np.allclose(vectorized_results, standard_results, rtol=1e-10)


def test_cdf_vectorized_broadcasting():
    """Test that cdf_vectorized correctly broadcasts scalar and array inputs"""
    delta = 2.0
    copula = HueslerReiss(delta)

    # Test broadcasting scalar u with array v
    u_scalar = 0.5
    v_array = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    vec_results1 = copula.cdf_vectorized(u_scalar, v_array)
    std_results1 = np.array([float(copula.cdf(u_scalar, v_i)) for v_i in v_array])

    assert np.allclose(vec_results1, std_results1, rtol=1e-10)

    # Test broadcasting array u with scalar v
    u_array = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    v_scalar = 0.5

    vec_results2 = copula.cdf_vectorized(u_array, v_scalar)
    std_results2 = np.array([float(copula.cdf(u_i, v_scalar)) for u_i in u_array])

    assert np.allclose(vec_results2, std_results2, rtol=1e-10)


@pytest.mark.parametrize(
    "point, expected",
    [
        ((1, 0.5), 0.5),  # u=1, v=0.5 -> C(u,v)=v
        ((0.5, 1), 0.5),  # u=0.5, v=1 -> C(u,v)=u
        ((1, 1), 1),  # u=1, v=1 -> C(u,v)=min(u,v)=1
        ((0, 0.5), 0),  # u=0, v=0.5 -> C(u,v)=0
        ((0.5, 0), 0),  # u=0.5, v=0 -> C(u,v)=0
        ((0, 0), 0),  # u=0, v=0 -> C(u,v)=0
    ],
)
def test_cdf_vectorized_edge_cases(point, expected):
    """Test edge cases for the vectorized CDF"""
    delta = 1.0
    copula = HueslerReiss(delta)

    # Test single point
    u, v = point
    result = copula.cdf_vectorized(u, v)
    assert np.isclose(result, expected, atol=1e-10)

    # Test vectorized with arrays containing edge cases
    u_array = np.array([point[0], 0.5])
    v_array = np.array([point[1], 0.6])

    results = copula.cdf_vectorized(u_array, v_array)
    assert np.isclose(results[0], expected, atol=1e-10)


def test_cdf_vectorized_independence():
    """Test the independence case (delta=0)"""
    # When delta=0, HueslerReiss returns IndependenceCopula
    # and cdf_vectorized should return u*v
    copula = HueslerReiss(0)

    u = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    v = np.array([0.2, 0.4, 0.6, 0.8, 0.95])

    results = copula.cdf_vectorized(u, v)
    expected = u * v

    assert np.allclose(results, expected, rtol=1e-10)


def test_cdf_vectorized_2d_array():
    """Test that cdf_vectorized works with 2D arrays"""
    delta = 1.0
    copula = HueslerReiss(delta)

    # Create 2D arrays
    u = np.array([[0.1, 0.5], [0.7, 0.9]])
    v = np.array([[0.2, 0.6], [0.8, 0.95]])

    # Calculate with vectorized method
    vectorized_results = copula.cdf_vectorized(u, v)

    # Expected shape
    assert vectorized_results.shape == u.shape

    # Calculate expected results
    expected = np.zeros_like(u)
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            expected[i, j] = float(copula.cdf(u[i, j], v[i, j]))

    # Compare
    assert np.allclose(vectorized_results, expected, rtol=1e-10)


def test_cdf_vectorized_numerical_stability():
    """Test numerical stability with values near boundaries"""
    delta = 1.0
    copula = HueslerReiss(delta)

    # Test with values near boundaries but not too extreme to avoid log(0) issues
    u_near_bounds = np.array([0.001, 0.999])
    v_near_bounds = np.array([0.001, 0.999])

    # This shouldn't raise any exceptions
    results = copula.cdf_vectorized(u_near_bounds, v_near_bounds)

    # Values should be finite
    assert np.all(np.isfinite(results))

    # For small u and v, result should be close to 0
    assert results[0] < 0.01

    # For u,v close to 1, result should be close to min(u,v)
    assert np.isclose(results[1], 0.999, rtol=1e-3)


def test_cdf_vectorized_mixed_scenarios():
    """Test a mix of boundary and interior values in the same array"""
    delta = 1.5
    copula = HueslerReiss(delta)

    # Mix of boundary and interior values
    u_mixed = np.array([0, 0.3, 1, 0.7, 0])
    v_mixed = np.array([0.2, 0, 0.8, 1, 0])

    # Expected results based on copula properties
    expected = np.zeros_like(u_mixed)
    # u=0 or v=0 -> C(u,v)=0
    expected[0] = 0  # u=0, v=0.2
    expected[1] = 0  # u=0.3, v=0
    expected[4] = 0  # u=0, v=0
    # u=1 -> C(u,v)=v
    expected[2] = 0.8  # u=1, v=0.8
    # v=1 -> C(u,v)=u
    expected[3] = 0.7  # u=0.7, v=1

    results = copula.cdf_vectorized(u_mixed, v_mixed)

    # Compare
    assert np.allclose(results, expected, rtol=1e-10)


def test_cdf_vectorized_large_arrays():
    """Test performance and stability with large arrays"""
    delta = 1.0
    copula = HueslerReiss(delta)

    # Create large arrays (sufficient to test batching)
    np.random.seed(42)  # For reproducibility
    size = 15000  # Large enough to test batching (> 10000)
    u_large = np.random.random(size)
    v_large = np.random.random(size)

    # This should run without memory issues
    results = copula.cdf_vectorized(u_large, v_large)

    # Basic sanity checks on results
    assert results.shape == u_large.shape
    assert np.all(results >= 0) and np.all(results <= 1)
    assert np.all(np.isfinite(results))

    # Test a few random points for accuracy
    indices = np.random.choice(size, 5, replace=False)
    for idx in indices:
        u_val, v_val = u_large[idx], v_large[idx]
        std_result = float(copula.cdf(u_val, v_val))
        vec_result = results[idx]
        assert np.isclose(vec_result, std_result, rtol=1e-8)


def test_pickands_for_hr():
    hr = HueslerReiss(1)
    result = hr.pickands(0)
    assert np.isclose(result, 1)
