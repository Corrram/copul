import numpy as np
import pytest
import sympy
from unittest.mock import patch

from copul.families.other import BivIndependenceCopula, LowerFrechet, UpperFrechet
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
    [(-1, LowerFrechet), (0, BivIndependenceCopula), (1, UpperFrechet)],
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

@pytest.mark.instable
@pytest.mark.parametrize("param, expected", [(-1, -1), (0, 0), (1, 1)])
def test_gaussian_tau(param, expected):
    cop = Gaussian()(param)
    assert cop.tau() == expected


@pytest.mark.parametrize("xi, expected", [(-1, 1), (0, 0), (1, 1)])
def test_gaussian_xi(xi, expected):
    cop = Gaussian()(xi)
    assert cop.xi() == expected


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


def test_gaussian_rho():
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
        tau = cop.tau()
        spearman = cop.spearmans_rho()
        xi = cop.xi()

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


def test_gaussian_cdf_vectorized_basic(gaussian_copula):
    """Test that cdf_vectorized gives same results as scalar evaluation."""
    import numpy as np

    # Define test points
    u_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    v_values = np.array([0.2, 0.4, 0.6, 0.8, 0.7])

    # Calculate expected results using scalar CDF
    expected_results = np.array(
        [
            float(gaussian_copula.cdf(u=u_values[i], v=v_values[i]).evalf())
            for i in range(len(u_values))
        ]
    )

    # Calculate results using vectorized CDF
    actual_results = gaussian_copula.cdf_vectorized(u_values, v_values)

    # Check that results match
    np.testing.assert_allclose(actual_results, expected_results, rtol=1e-10)


def test_gaussian_cdf_vectorized_broadcasting(gaussian_copula):
    """Test that cdf_vectorized correctly handles broadcasting."""
    import numpy as np

    # Test broadcasting: u is scalar, v is array
    u_scalar = 0.5
    v_array = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    # Calculate expected results using scalar CDF
    expected_results = np.array(
        [float(gaussian_copula.cdf(u=u_scalar, v=v).evalf()) for v in v_array]
    )

    # Calculate results using vectorized CDF
    actual_results = gaussian_copula.cdf_vectorized(u_scalar, v_array)

    # Check that results match
    np.testing.assert_allclose(actual_results, expected_results, rtol=1e-10)

    # Test broadcasting: u is array, v is scalar
    u_array = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    v_scalar = 0.5

    # Calculate expected results using scalar CDF
    expected_results = np.array(
        [float(gaussian_copula.cdf(u=u, v=v_scalar).evalf()) for u in u_array]
    )

    # Calculate results using vectorized CDF
    actual_results = gaussian_copula.cdf_vectorized(u_array, v_scalar)

    # Check that results match
    np.testing.assert_allclose(actual_results, expected_results, rtol=1e-10)


def test_gaussian_cdf_vectorized_grid(gaussian_copula):
    """Test cdf_vectorized with grid inputs."""
    import numpy as np

    # Create grid of values
    u_grid = np.linspace(0.1, 0.9, 5)
    v_grid = np.linspace(0.1, 0.9, 5)
    U, V = np.meshgrid(u_grid, v_grid)

    # Calculate expected results using scalar CDF
    expected_results = np.zeros_like(U)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            expected_results[i, j] = float(
                gaussian_copula.cdf(u=U[i, j], v=V[i, j]).evalf()
            )

    # Calculate results using vectorized CDF
    actual_results = gaussian_copula.cdf_vectorized(U, V)

    # Check that results match
    np.testing.assert_allclose(actual_results, expected_results, rtol=1e-10)


def test_gaussian_cdf_vectorized_boundary_values(gaussian_copula):
    """Test cdf_vectorized with boundary values (0 and 1)."""
    import numpy as np

    # Test boundary values
    u_values = np.array([0, 0, 1, 1, 0.5])
    v_values = np.array([0, 1, 0, 1, 0.5])

    # Calculate results using vectorized CDF
    results = gaussian_copula.cdf_vectorized(u_values, v_values)

    # Expected results for boundary cases:
    # C(0,v) = 0 for all v
    # C(u,0) = 0 for all u
    # C(1,v) = v for all v
    # C(u,1) = u for all u
    # For the (0.5, 0.5) case, we use the actual CDF value
    expected = np.array([0, 0, 0, 1, float(gaussian_copula.cdf(u=0.5, v=0.5).evalf())])

    # Check that results match
    np.testing.assert_allclose(results, expected, rtol=1e-10)


def test_gaussian_cdf_vectorized_input_validation(gaussian_copula):
    """Test that cdf_vectorized properly validates inputs."""
    import numpy as np

    # Test with invalid values (outside [0,1])
    with pytest.raises(ValueError, match="Marginals must be in"):
        gaussian_copula.cdf_vectorized(np.array([-0.1, 0.5]), np.array([0.2, 0.3]))

    with pytest.raises(ValueError, match="Marginals must be in"):
        gaussian_copula.cdf_vectorized(np.array([0.2, 0.5]), np.array([0.2, 1.1]))


def test_gaussian_cdf_vectorized_performance(gaussian_copula):
    """Test that cdf_vectorized is faster than scalar evaluation for large inputs."""
    import numpy as np
    import time

    # Create large test arrays (1000 points)
    np.random.seed(42)  # For reproducibility
    u_large = np.random.random(1000)
    v_large = np.random.random(1000)

    # Time scalar evaluation
    start_scalar = time.time()
    scalar_results = np.array(
        [
            float(gaussian_copula.cdf(u=u_large[i], v=v_large[i]).evalf())
            for i in range(len(u_large))
        ]
    )
    scalar_time = time.time() - start_scalar

    # Time vectorized evaluation
    start_vector = time.time()
    vector_results = gaussian_copula.cdf_vectorized(u_large, v_large)
    vector_time = time.time() - start_vector

    # Check that results match
    np.testing.assert_allclose(vector_results, scalar_results, rtol=1e-6)

    # Check that vectorized is faster (should be at least 5x faster)
    assert vector_time < scalar_time * 0.2, (
        f"Vectorized: {vector_time}s, Scalar: {scalar_time}s"
    )


@pytest.mark.parametrize(
    "rho, expected_function",
    [
        (-1, lambda u, v: np.maximum(u + v - 1, 0)),  # Lower Fréchet bound
        (0, lambda u, v: u * v),  # Independence
        (1, lambda u, v: np.minimum(u, v)),  # Upper Fréchet bound
    ],
)
def test_gaussian_cdf_vectorized_special_cases(rho, expected_function):
    """Test cdf_vectorized with special correlation values."""
    import numpy as np

    # Create a Gaussian copula with the special rho value
    # Need to create directly instead of using fixtures to get different rho values
    copula = Gaussian()(rho)  # This should handle the special cases internally

    # Create test points
    u_values = np.array([0.2, 0.4, 0.6, 0.8])
    v_values = np.array([0.3, 0.5, 0.7, 0.9])

    # Calculate expected results using the known formulas for special cases
    expected_results = expected_function(u_values, v_values)

    # Calculate results using vectorized CDF
    # Even though copula might be a special case class, it should have the cdf_vectorized method
    actual_results = copula.cdf_vectorized(u_values, v_values)

    # Check that results match
    np.testing.assert_allclose(actual_results, expected_results, rtol=1e-10)


def test_gaussian_cdf_vectorized_against_theoretical():
    """Test cdf_vectorized against theoretical properties of Gaussian copulas."""
    import numpy as np

    # Create a copula instance with rho = 0.5
    copula = Gaussian(0.5)

    # Property: Gaussian copula values should satisfy Fréchet bounds
    u_grid = np.linspace(0.1, 0.9, 9)
    v_grid = np.linspace(0.1, 0.9, 9)
    U, V = np.meshgrid(u_grid, v_grid)

    # Calculate results using vectorized CDF
    cdf_values = copula.cdf_vectorized(U, V)

    # Check against Fréchet lower bound: C(u,v) ≥ max(u+v-1, 0)
    lower_bound = np.maximum(U + V - 1, 0)
    assert np.all(cdf_values >= lower_bound - 1e-10)

    # Check against Fréchet upper bound: C(u,v) ≤ min(u,v)
    upper_bound = np.minimum(U, V)
    assert np.all(cdf_values <= upper_bound + 1e-10)

    # Check symmetry property: C(u,v) = C(v,u)
    cdf_transpose = copula.cdf_vectorized(V, U)
    np.testing.assert_allclose(cdf_values, cdf_transpose, rtol=1e-10)
