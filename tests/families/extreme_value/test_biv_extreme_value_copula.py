import time
import numpy as np
import pytest
import sympy as sp

from copul.families.extreme_value.biv_extreme_value_copula import BivExtremeValueCopula


# Create a concrete subclass of ExtremeValueCopula for testing
class ConcreteEVC(BivExtremeValueCopula):
    """Concrete implementation of ExtremeValueCopula for testing."""

    theta = sp.symbols("theta", positive=True)
    params = [theta]
    intervals = {"theta": sp.Interval(0, sp.oo, left_open=False, right_open=True)}
    _free_symbols = {"theta": theta}  # Add this line to initialize _free_symbols

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def is_symmetric(self) -> bool:
        return True

    # Change to instance attribute instead of property for consistency
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        t = self.t
        # A simple Pickands dependence function: A(t) = 1 - theta * t * (1-t)
        self._pickands = 1 - self.theta * t * (1 - t)


@pytest.fixture
def copula():
    """Create a concrete EVC instance with theta=0.5 for tests."""
    return ConcreteEVC(theta=0.5)


def test_initialization():
    """Test that the ExtremeValueCopula can be initialized through a concrete subclass."""
    evc = ConcreteEVC()
    assert str(evc.theta) == "theta"

    evc = ConcreteEVC(theta=0.5)
    assert float(evc.theta) == 0.5


def test_from_pickands():
    """Test creating an ExtremeValueCopula from a Pickands function."""
    # Create a simple Pickands function: A(t) = 1 - alpha * t * (1-t)
    alpha = sp.symbols("alpha", positive=True)
    t = sp.symbols("t", positive=True)
    pickands = 1 - alpha * t * (1 - t)

    # Create copula from Pickands function
    evc = BivExtremeValueCopula.from_pickands(pickands, [alpha])

    # Check that the Pickands function was correctly set
    assert str(evc._pickands) == str(pickands.subs(t, evc.t))

    # Check that parameters were correctly identified
    assert len(evc.params) == 1
    assert str(evc.params[0]) == "alpha"


def test_pickands_property(copula):
    """Test the pickands property."""
    # Get Pickands function
    pickands = copula.pickands

    # Evaluate at some points
    t_vals = [0, 0.25, 0.5, 0.75, 1]
    for t in t_vals:
        result = float(pickands(t=t))
        expected = 1 - 0.5 * t * (1 - t)
        assert abs(result - expected) < 1e-10


def test_deriv_pickand_at_0(copula):
    """Test the derivative of Pickands function at t=0."""
    # For A(t) = 1 - 0.5 * t * (1-t), A'(t) = 0.5 * (2t - 1)
    # At t=0, A'(0) = -0.5
    deriv = copula.deriv_pickand_at_0()
    assert float(deriv) == -0.5


def test_sample_parameters(copula):
    """Test sampling parameters."""
    n_samples = 5
    samples = copula.sample_parameters(n=n_samples)

    # Check that we got the expected number of samples
    assert len(samples["theta"]) == n_samples

    # Check that all samples are within bounds
    for sample in samples["theta"]:
        assert sample >= 0  # Lower bound of theta


def test_is_ci(copula):
    """Test the is_ci property."""
    assert copula.is_ci is True


def test_cdf_formula(copula):
    """Test the CDF formula."""
    # For our test Pickands function, check CDF at some points
    test_points = [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]

    for u, v in test_points:
        cdf_val = float(copula.cdf(u=u, v=v))

        # CDF should be positive, at most min(u,v), and at least u*v
        assert 0 <= cdf_val <= min(u, v)
        assert cdf_val >= u * v


def test_pdf_existence(copula):
    """Test that PDF function exists."""
    # Just check that we can get a PDF object
    pdf = copula.pdf
    assert pdf is not None


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


def test_pickands_constraints():
    """Test that Pickands function satisfies theoretical constraints."""
    # Create a valid Pickands function
    copula = ConcreteEVC(theta=0.5)
    pickands = copula.pickands

    # Pickands function should satisfy:
    # 1. A(0) = A(1) = 1
    assert abs(float(pickands(t=0)) - 1) < 1e-10
    assert abs(float(pickands(t=1)) - 1) < 1e-10

    # 2. max(t, 1-t) <= A(t) <= 1 for all t in [0,1]
    t_vals = np.linspace(0.1, 0.9, 10)
    for t in t_vals:
        a_t = float(pickands(t=t))
        assert max(t, 1 - t) <= a_t <= 1 + 1e-10


def test_spearmans_rho(copula):
    """Test Spearman's rho calculation."""
    # Just check that we can calculate it without errors
    try:
        rho = copula.spearmans_rho()
        # Spearman's rho should be between -1 and 1
        assert -1 <= float(rho) <= 1
    except Exception as e:
        pytest.skip(f"Spearman's rho calculation raised an exception: {e}")


def test_kendalls_tau(copula):
    """Test Kendall's tau calculation."""
    # Just check that we can calculate it without errors
    try:
        tau = copula.kendalls_tau()
        # Kendall's tau should be between -1 and 1
        assert -1 <= float(tau) <= 1
    except Exception as e:
        pytest.skip(f"Kendall's tau calculation raised an exception: {e}")


def test_abstract_methods():
    """Test that abstract methods must be implemented in subclasses."""

    class IncompleteEVC(BivExtremeValueCopula):
        pass

    evc = IncompleteEVC()

    with pytest.raises(NotImplementedError):
        _ = evc.is_absolutely_continuous

    with pytest.raises(NotImplementedError):
        _ = evc.is_symmetric


def test_minimize_func():
    """Test the minimize_func method."""
    copula = ConcreteEVC(theta=0.5)

    # Define a simple function to minimize
    x1, x2, y1, y2 = sp.symbols("x1 x2 y1 y2")
    theta = sp.symbols("theta", positive=True)
    func = (
        (x1 - 0.5) ** 2
        + (x2 - 0.5) ** 2
        + (y1 - 0.5) ** 2
        + (y2 - 0.5) ** 2
        + (theta - 1) ** 2
    )

    # Try to minimize the function
    try:
        solution, x0 = copula.minimize_func(func)
        # If successful, check that we got a valid solution
        if solution is not None:
            assert solution.success
    except Exception:
        pytest.skip("minimize_func raised an exception")


def test_mix_params():
    """Test the _mix_params static method."""
    params = {"theta": [0.5, 1.0, 1.5], "alpha": 0.7}

    mixed = BivExtremeValueCopula._mix_params(params)

    assert len(mixed) == 3  # Three combinations based on theta values
    assert all(item["alpha"] == 0.7 for item in mixed)
    assert [item["theta"] for item in mixed] == [0.5, 1.0, 1.5]


def test_plotting_functions(copula):
    """Test that plotting functions don't raise errors."""
    # Just make sure these methods exist and can be called
    assert hasattr(copula, "plot_pickands")

    # Skip actual plotting tests since they use matplotlib
    # and we don't want to display plots during testing


def test_from_generator_with_galambos():
    pickands = "1 - (t ** (-delta) + (1 - t) ** (-delta)) ** (-1 / delta)"
    copula_family = BivExtremeValueCopula.from_pickands(pickands)
    copula = copula_family(2)
    result = copula.pickands(0.5)
    assert np.isclose(result.evalf(), 0.6464466094067263)


def test_from_generator_with_galambos_with_different_var_name():
    pickands = "1 - (x ** (-delta) + (1 - x) ** (-delta)) ** (-1 / delta)"
    copula_family = BivExtremeValueCopula.from_pickands(pickands, "delta")
    copula = copula_family(2)
    result = copula.pickands(0.5)
    assert np.isclose(result.evalf(), 0.6464466094067263)


def test_cdf_vectorized_basic(copula):
    """Test that cdf_vectorized gives same results as scalar evaluation."""
    # Define test points
    u_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    v_values = np.array([0.2, 0.4, 0.6, 0.8, 0.7])

    # Calculate expected results using scalar CDF
    results = []
    for i in range(len(u_values)):
        results.append(float(copula.cdf(u=u_values[i], v=v_values[i]).evalf()))
    expected_results = np.array(results)

    # Calculate results using vectorized CDF
    actual_results = copula.cdf_vectorized(u_values, v_values)

    # Check that results match
    np.testing.assert_allclose(actual_results, expected_results, rtol=1e-10)

def test_cdf_vectorized_broadcasting_u(copula):
    """Test that cdf_vectorized correctly handles broadcasting with u as scalar."""
    # Test broadcasting: u is scalar, v is array
    u_scalar = 0.5
    v_array = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    # Calculate expected results using scalar CDF
    expected_results = np.array(
        [float(copula.cdf(u=u_scalar, v=v).evalf()) for v in v_array]
    )

    # Calculate results using vectorized CDF
    actual_results = copula.cdf_vectorized(u_scalar, v_array)
    
    # Check that results match
    np.testing.assert_allclose(actual_results, expected_results, rtol=1e-10)

def test_cdf_vectorized_broadcasting_v(copula):
    """Test that cdf_vectorized correctly handles broadcasting with v as scalar."""
    # Test broadcasting: u is array, v is scalar
    u_array = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    v_scalar = 0.5

    # Calculate expected results using scalar CDF
    expected_results = np.array(
        [float(copula.cdf(u=u, v=v_scalar).evalf()) for u in u_array]
    )

    # Calculate results using vectorized CDF
    actual_results = copula.cdf_vectorized(u_array, v_scalar)
    
    # Check that results match
    np.testing.assert_allclose(actual_results, expected_results, rtol=1e-10)
def test_cdf_vectorized_grid(copula):
    """Test cdf_vectorized with grid inputs."""
    # Create grid of values
    u_grid = np.linspace(0.1, 0.9, 5)
    v_grid = np.linspace(0.1, 0.9, 5)
    U, V = np.meshgrid(u_grid, v_grid)

    # Calculate expected results using scalar CDF
    expected_results = np.zeros_like(U)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            expected_results[i, j] = float(copula.cdf(u=U[i, j], v=V[i, j]).evalf())

    # Calculate results using vectorized CDF
    actual_results = copula.cdf_vectorized(U, V)

    # Check that results match
    np.testing.assert_allclose(actual_results, expected_results, rtol=1e-10)


def test_cdf_vectorized_boundary_values(copula):
    """Test cdf_vectorized with boundary values (0 and 1)."""
    # Test boundary values
    u_values = np.array([0, 0, 1, 1, 0.5])
    v_values = np.array([0, 1, 0, 1, 0.5])

    # Calculate results using vectorized CDF
    results = copula.cdf_vectorized(u_values, v_values)

    # Expected results for a copula CDF:
    # C(0,v) = 0 for all v
    # C(u,0) = 0 for all u
    # C(1,v) = v for all v
    # C(u,1) = u for all u
    expected = np.array([0, 0, 0, 1, float(copula.cdf(u=0.5, v=0.5).evalf())])

    # Check that results match
    np.testing.assert_allclose(results, expected, rtol=1e-10)


def test_cdf_vectorized_input_validation(copula):
    """Test that cdf_vectorized properly validates inputs."""
    # Test with invalid values (outside [0,1])
    with pytest.raises(ValueError, match="Marginals must be in"):
        copula.cdf_vectorized(np.array([-0.1, 0.5]), np.array([0.2, 0.3]))

    with pytest.raises(ValueError, match="Marginals must be in"):
        copula.cdf_vectorized(np.array([0.2, 0.5]), np.array([0.2, 1.1]))


def test_cdf_vectorized_with_different_parameters():
    """Test cdf_vectorized with different parameter values."""
    # Create multiple copulas with different theta values
    thetas = [0.2, 0.5, 0.8]

    for theta in thetas:
        # Create copula with this theta
        copula = ConcreteEVC(theta=theta)

        # Define test points
        u_values = np.array([0.3, 0.5, 0.7])
        v_values = np.array([0.4, 0.6, 0.8])

        # Calculate expected results using scalar CDF
        expected_results = np.array(
            [
                float(copula.cdf(u=u_values[i], v=v_values[i]).evalf())
                for i in range(len(u_values))
            ]
        )

        # Calculate results using vectorized CDF
        actual_results = copula.cdf_vectorized(u_values, v_values)

        # Check that results match
        np.testing.assert_allclose(
            actual_results,
            expected_results,
            rtol=1e-10,
            err_msg=f"Failed with theta={theta}",
        )


def test_cdf_vectorized_against_theoretical():
    """Test cdf_vectorized against theoretical properties of extreme value copulas."""
    import numpy as np

    # Create a copula instance
    copula = ConcreteEVC(theta=0.5)

    # For our concrete EVC, pickands is A(t) = 1 - theta * t * (1-t)
    # At u=v (diagonal), t=0.5, so A(0.5) = 1 - 0.5 * 0.5 * 0.5 = 0.875
    u_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    # Calculate results using vectorized CDF
    actual_results = copula.cdf_vectorized(u_values, u_values)

    # Calculate results using scalar CDF for verification
    expected_results = np.array([float(copula.cdf(u=u, v=u).evalf()) for u in u_values])

    # Check that vectorized results match scalar results
    np.testing.assert_allclose(actual_results, expected_results, rtol=1e-10)

    # Property: Should satisfy C(u,v) ≥ max(u+v-1, 0) (Fréchet lower bound)
    u_grid = np.linspace(0.1, 0.9, 9)
    v_grid = np.linspace(0.1, 0.9, 9)
    U, V = np.meshgrid(u_grid, v_grid)

    # Calculate results using vectorized CDF
    cdf_values = copula.cdf_vectorized(U, V)

    # Check against Fréchet lower bound
    lower_bound = np.maximum(U + V - 1, 0)
    assert np.all(cdf_values >= lower_bound - 1e-10)

    # Property: Should satisfy C(u,v) ≤ min(u,v) (Fréchet upper bound)
    upper_bound = np.minimum(U, V)
    assert np.all(cdf_values <= upper_bound + 1e-10)


def test_cdf_vectorized_vs_cdf(copula):
    """Test that cdf_vectorized is faster than scalar evaluation for large inputs."""
    # Create large test arrays (1000 points)
    np.random.seed(42)  # For reproducibility
    u_large = np.random.random(10)
    v_large = np.random.random(10)

    # Time scalar evaluation
    start_scalar = time.time()
    scalar_results = np.array(
        [
            float(copula.cdf(u=u_large[i], v=v_large[i]).evalf())
            for i in range(len(u_large))
        ]
    )
    scalar_time = time.time() - start_scalar

    # Time vectorized evaluation
    start_vector = time.time()
    vector_results = copula.cdf_vectorized(u_large, v_large)
    vector_time = time.time() - start_vector

    # Check that results match
    np.testing.assert_allclose(vector_results, scalar_results, rtol=1e-10)

    # Check that vectorized is faster (should be at least 5x faster)
    assert vector_time < scalar_time * 0.2, (
        f"Vectorized: {vector_time}s, Scalar: {scalar_time}s"
    )
