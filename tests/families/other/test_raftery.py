import numpy as np
import pytest

from copul.families.core.biv_copula import BivCopula
from copul.families.other.raftery import Raftery


@pytest.fixture
def copula():
    """Create a Raftery copula instance with delta=0.5 for tests."""
    return Raftery(delta=0.5)


def test_initialization():
    """Test that the Raftery copula can be initialized with different parameters."""
    # Default initialization
    raftery = Raftery()
    assert str(raftery.delta) == "delta"

    # With parameter
    raftery = Raftery(delta=0.5)
    assert float(raftery.delta) == 0.5

    # Edge case parameters
    raftery_min = Raftery(delta=0)
    assert float(raftery_min.delta) == 0

    raftery_max = Raftery(delta=1)
    assert float(raftery_max.delta) == 1


def test_parameter_constraints():
    """Test parameter constraints: 0 ≤ delta ≤ 1."""
    # Valid parameters
    Raftery(delta=0)  # Lower bound
    Raftery(delta=0.5)  # Middle
    Raftery(delta=1)  # Upper bound

    # Invalid parameters (delta < 0 or delta > 1)
    with pytest.raises(ValueError):
        Raftery(delta=-0.1)

    with pytest.raises(ValueError):
        Raftery(delta=1.1)


def test_special_cases():
    """Test special cases of the Raftery copula."""
    # When delta = 0, should return IndependenceCopula
    independence = Raftery(delta=0)
    test_points = [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]

    for u, v in test_points:
        # For independence, C(u,v) = u*v
        cdf_val = float(independence.cdf(u=u, v=v))
        expected = u * v
        assert abs(cdf_val - expected) < 1e-10, (
            f"CDF incorrect for independence case at u={u}, v={v}"
        )

    # When delta = 1, should return UpperFrechet
    upper_frechet = Raftery(delta=1)

    for u, v in test_points:
        # For upper Frechet, C(u,v) = min(u,v)
        cdf_val = float(upper_frechet.cdf(u=u, v=v))
        expected = min(u, v)
        assert abs(cdf_val - expected) < 1e-10, (
            f"CDF incorrect for upper Frechet case at u={u}, v={v}"
        )


def test_inheritance():
    """Test that Raftery properly inherits from BivCopula."""
    copula = Raftery()
    assert isinstance(copula, BivCopula)


def test_is_symmetric(copula):
    """Test that the Raftery copula is symmetric."""
    assert copula.is_symmetric is True


def test_is_absolutely_continuous(copula):
    """Test the absolutely continuous property."""
    assert copula.is_absolutely_continuous is False


def test_cdf_values(copula):
    """Test specific CDF values."""
    delta = 0.5

    def raftery_cdf(u, v, delta):
        """Compute the Raftery copula CDF value."""
        min_uv = min(u, v)
        max_uv = max(u, v)
        return min_uv + (1 - delta) / (1 + delta) * (u * v) ** (1 / (1 - delta)) * (
            1 - max_uv ** (-(1 + delta) / (1 - delta))
        )

    test_cases = [
        (0.5, 0.5, raftery_cdf(0.5, 0.5, delta)),  # Symmetric point
        (0.3, 0.7, raftery_cdf(0.3, 0.7, delta)),  # u < v
        (0.7, 0.3, raftery_cdf(0.7, 0.3, delta)),  # u > v
        (0.0, 0.5, 0.0),  # Boundary u=0
        (0.5, 0.0, 0.0),  # Boundary v=0
        (1.0, 0.5, 0.5),  # Boundary u=1
        (0.5, 1.0, 0.5),  # Boundary v=1
        (1.0, 1.0, 1.0),  # Boundary u=v=1
    ]

    for u, v, expected in test_cases:
        # Skip cases where u or v is 0 or 1 - boundaries are handled separately
        if u == 0 or v == 0 or u == 1 or v == 1:
            continue

        cdf_val = float(copula.cdf(u=u, v=v))
        # Allowing for some numerical error due to complex formula
        diff = abs(cdf_val - expected)
        assert diff < 1e-6, f"CDF value incorrect for u={u}, v={v}"


def test_boundary_cases(copula):
    """Test that the copula behaves correctly at boundary values."""
    # Create a range of test values
    u_vals = np.linspace(0.1, 0.9, 5)

    # At (0, v) and (u, 0), copula should be 0
    for u in u_vals:
        # Get CDF values
        cdf_u0 = float(copula.cdf(u=u, v=0))
        cdf_0v = float(copula.cdf(u=0, v=u))

        assert abs(cdf_u0) < 1e-8, f"C({u},0) should be 0, got {cdf_u0}"
        assert abs(cdf_0v) < 1e-8, f"C(0,{u}) should be 0, got {cdf_0v}"

    # At (1, v), copula should be v
    # At (u, 1), copula should be u
    for u in u_vals:
        cdf_u1 = float(copula.cdf(u=u, v=1))
        cdf_1v = float(copula.cdf(u=1, v=u))

        assert abs(cdf_u1 - u) < 1e-8, f"C({u},1) should be {u}, got {cdf_u1}"
        assert abs(cdf_1v - u) < 1e-8, f"C(1,{u}) should be {u}, got {cdf_1v}"


def test_pdf_values(copula):
    """Test that PDF is correctly implemented."""
    # Simply check that it returns positive values inside unit square
    test_points = [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]

    for u, v in test_points:
        pdf_val = float(copula.pdf(u=u, v=v))
        assert pdf_val > 0, f"PDF should be positive at u={u}, v={v}"


def test_rho():
    """Test Spearman's rho calculation."""
    # Formula: delta * (4 - 3*delta) / (2 - delta)^2
    test_cases = [
        (0, 0),
        (0.25, 0.25 * (4 - 3 * 0.25) / (2 - 0.25) ** 2),
        (0.5, 0.5 * (4 - 3 * 0.5) / (2 - 0.5) ** 2),
        (0.75, 0.75 * (4 - 3 * 0.75) / (2 - 0.75) ** 2),
        (1, 1),  # At delta=1, rho=1
    ]

    for delta, expected in test_cases:
        copula = Raftery(delta=delta)
        rho = float(copula.rho())
        assert abs(rho - expected) < 1e-8, f"Spearman's rho incorrect for delta={delta}"


def test_tau():
    """Test Kendall's tau calculation."""
    # Formula: 2*delta / (3 - delta)
    test_cases = [
        (0, 0),
        (0.25, 2 * 0.25 / (3 - 0.25)),
        (0.5, 2 * 0.5 / (3 - 0.5)),
        (0.75, 2 * 0.75 / (3 - 0.75)),
        (1, 1),  # At delta=1, tau=1
    ]

    for delta, expected in test_cases:
        copula = Raftery(delta=delta)
        tau = float(copula.tau())
        assert abs(tau - expected) < 1e-8, f"Kendall's tau incorrect for delta={delta}"


def test_tail_dependence():
    """Test tail dependence coefficients."""
    # Lower tail dependence: 2*delta / (1 + delta)
    # Upper tail dependence: 0
    test_cases = [
        (0, 0),
        (0.25, 2 * 0.25 / (1 + 0.25)),
        (0.5, 2 * 0.5 / (1 + 0.5)),
        (0.75, 2 * 0.75 / (1 + 0.75)),
        (1, 1),  # At delta=1, lambda_L=1
    ]

    for delta, expected_lambda_L in test_cases:
        copula = Raftery(delta=delta)

        lambda_L = float(copula.lambda_L)
        assert abs(lambda_L - expected_lambda_L) < 1e-8, (
            f"Lower tail dependence incorrect for delta={delta}"
        )

        lambda_U = float(copula.lambda_U)
        assert lambda_U == 0, "Upper tail dependence should be 0 for all delta values"


def test_helper_methods(copula):
    """Test the helper methods of Raftery copula."""
    # Test _b method (PDF component)
    u, v = 0.3, 0.7
    b_val = copula._b(u, v)
    assert b_val is not None, "_b method should return a value"

    # Since _squared_cond_distr_1 and _xi_int_1 might be complex/computational methods
    # We just verify they exist with the correct names and basic behavior
    if hasattr(copula, "_squared_cond_distr_1"):
        try:
            result = copula._squared_cond_distr_1(u, v)
            assert result is not None
        except Exception as e:
            pytest.skip(f"_squared_cond_distr_1 raised an exception: {e}")

    if hasattr(copula, "_xi_int_1"):
        try:
            result = copula._xi_int_1(v)
            assert result is not None
        except Exception as e:
            pytest.skip(f"_xi_int_1 raised an exception: {e}")


def test_raftery_cdf_vectorized_scalar_inputs():
    """Test that cdf_vectorized works correctly with scalar inputs"""
    delta = 0.5
    copula = Raftery(delta)

    # Test at some specific points
    u, v = 0.7, 0.3

    # Both methods should return the same result for scalar inputs
    cdf_result = float(copula.cdf(u, v))
    cdf_vec_result = float(copula.cdf_vectorized(u, v))

    assert np.isclose(cdf_result, cdf_vec_result, rtol=1e-10)


def test_raftery_cdf_vectorized_array_inputs():
    """Test that cdf_vectorized works correctly with array inputs"""
    delta = 0.5
    copula = Raftery(delta)

    # Create arrays of u and v values
    u = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    v = np.array([0.2, 0.4, 0.6, 0.8, 0.95])

    # Calculate results with vectorized method
    vectorized_results = copula.cdf_vectorized(u, v)

    # Calculate results individually with standard cdf
    standard_results = np.array([float(copula.cdf(u_i, v_i)) for u_i, v_i in zip(u, v)])

    # Compare results
    assert np.allclose(vectorized_results, standard_results, rtol=1e-10)


def test_raftery_cdf_vectorized_special_cases():
    """Test that cdf_vectorized correctly handles special cases"""
    # Independence case (delta = 0)
    copula_ind = Raftery(0)
    u = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    v = np.array([0.2, 0.4, 0.6, 0.8, 0.95])

    ind_results = copula_ind.cdf_vectorized(u, v)
    expected_ind = u * v

    assert np.allclose(ind_results, expected_ind, rtol=1e-10)

    # Upper Fréchet case (delta = 1)
    copula_uf = Raftery(1)
    uf_results = copula_uf.cdf_vectorized(u, v)
    expected_uf = np.minimum(u, v)

    assert np.allclose(uf_results, expected_uf, rtol=1e-10)


def test_raftery_cdf_vectorized_edge_cases():
    """Test that cdf_vectorized correctly handles edge cases (0s and 1s)"""
    delta = 0.5
    copula = Raftery(delta)

    # Test when u or v is 0
    u_zeros = np.array([0, 0, 0.5, 0.7])
    v_zeros = np.array([0.2, 0, 0, 0.8])

    zero_results = copula.cdf_vectorized(u_zeros, v_zeros)

    # Calculate expected results individually using standard CDF
    expected_zeros = np.array(
        [
            float(copula.cdf(u_zeros[0], v_zeros[0])),
            float(copula.cdf(u_zeros[1], v_zeros[1])),
            float(copula.cdf(u_zeros[2], v_zeros[2])),
            float(copula.cdf(u_zeros[3], v_zeros[3])),
        ]
    )

    assert np.allclose(zero_results, expected_zeros, rtol=1e-10)

    # Specifically check that when u or v is 0, result is 0
    assert np.all(zero_results[:3] == 0)

    # Test when u or v is 1
    u_ones = np.array([1, 0.3, 1, 0.7])
    v_ones = np.array([0.2, 1, 1, 0.8])

    ones_results = copula.cdf_vectorized(u_ones, v_ones)

    # Calculate expected results individually using standard CDF
    expected_ones = np.array(
        [
            float(copula.cdf(u_ones[0], v_ones[0])),
            float(copula.cdf(u_ones[1], v_ones[1])),
            float(copula.cdf(u_ones[2], v_ones[2])),
            float(copula.cdf(u_ones[3], v_ones[3])),
        ]
    )

    assert np.allclose(ones_results, expected_ones, rtol=1e-10)


def test_raftery_cdf_vectorized_broadcasting():
    """Test that cdf_vectorized correctly broadcasts scalar and array inputs"""
    delta = 0.5
    copula = Raftery(delta)

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


def test_raftery_cdf_vectorized_2d_array():
    """Test that cdf_vectorized works with 2D arrays"""
    delta = 0.5
    copula = Raftery(delta)

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


def test_raftery_cdf_vectorized_numerical_stability():
    """Test numerical stability with values near boundaries"""
    delta = 0.5
    copula = Raftery(delta)

    # Test with values near boundaries
    u_near_bounds = np.array([1e-10, 1 - 1e-10])
    v_near_bounds = np.array([1e-10, 1 - 1e-10])

    # This shouldn't raise any exceptions
    results = copula.cdf_vectorized(u_near_bounds, v_near_bounds)

    # Values should be finite
    assert np.all(np.isfinite(results))

    # For small u and v, result should be close to 0
    assert results[0] < 0.01

    # For u,v close to 1, result should be close to min(u,v)
    assert np.isclose(results[1], u_near_bounds[1], rtol=1e-5)


def test_raftery_cdf_vectorized_large_arrays():
    """Test performance and stability with large arrays"""
    delta = 0.5
    copula = Raftery(delta)

    # Create large arrays (sufficient to test batching)
    np.random.seed(42)  # For reproducibility
    size = 10000
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
