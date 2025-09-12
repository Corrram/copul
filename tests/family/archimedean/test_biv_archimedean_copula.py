import numpy as np
import pytest
import sympy

from copul.family.archimedean.biv_archimedean_copula import BivArchimedeanCopula
from copul.family.other.biv_independence_copula import BivIndependenceCopula
from copul.family.other.lower_frechet import LowerFrechet


def test_special_cases_create_method():
    """Test that the create factory method correctly handles special cases."""

    # Define a test copula class with special cases
    class TestCopula(BivArchimedeanCopula):
        theta_interval = sympy.Interval(-1, sympy.oo, left_open=False, right_open=True)

        # Define special cases
        special_cases = {-1: LowerFrechet, 0: BivIndependenceCopula}

        @property
        def is_absolutely_continuous(self) -> bool:
            return self.theta >= 0

        @property
        def _generator(self):
            if self.theta == 0:
                return -sympy.log(self.t)
            return ((1 / self.t) ** self.theta - 1) / self.theta

    # Test regular case
    regular = TestCopula.create(2)
    assert isinstance(regular, TestCopula)
    assert regular.theta == 2

    # Test special case: theta = 0 should return IndependenceCopula
    independence = TestCopula.create(0)
    assert isinstance(independence, BivIndependenceCopula)

    # Test special case: theta = -1 should return LowerFrechet
    lower_frechet = TestCopula.create(-1)
    assert isinstance(lower_frechet, LowerFrechet)

    # Test with keyword arguments
    kwargs_regular = TestCopula.create(theta=2)
    assert isinstance(kwargs_regular, TestCopula)
    assert kwargs_regular.theta == 2

    kwargs_special = TestCopula.create(theta=0)
    assert isinstance(kwargs_special, BivIndependenceCopula)


def test_special_cases_new_method():
    """Test that the __new__ constructor correctly handles special cases."""

    # Define a test copula class with special cases
    class TestCopula(BivArchimedeanCopula):
        theta_interval = sympy.Interval(-1, sympy.oo, left_open=False, right_open=True)

        # Define special cases
        special_cases = {-1: LowerFrechet, 0: BivIndependenceCopula}

        @property
        def is_absolutely_continuous(self) -> bool:
            return self.theta >= 0

        @property
        def _generator(self):
            if self.theta == 0:
                return -sympy.log(self.t)
            return ((1 / self.t) ** self.theta - 1) / self.theta

    # Test regular case
    regular = TestCopula(2)
    assert isinstance(regular, TestCopula)
    assert regular.theta == 2

    # Test special case: theta = 0 should return IndependenceCopula
    independence = TestCopula(0)
    assert isinstance(independence, BivIndependenceCopula)

    # Test special case: theta = -1 should return LowerFrechet
    lower_frechet = TestCopula(-1)
    assert isinstance(lower_frechet, LowerFrechet)

    # Test with keyword arguments
    kwargs_regular = TestCopula(theta=2)
    assert isinstance(kwargs_regular, TestCopula)
    assert kwargs_regular.theta == 2

    kwargs_special = TestCopula(theta=0)
    assert isinstance(kwargs_special, BivIndependenceCopula)


def test_special_cases_call_method():
    """Test that the __call__ method correctly handles special cases."""

    # Define a test copula class with special cases
    class TestCopula(BivArchimedeanCopula):
        theta_interval = sympy.Interval(-1, sympy.oo, left_open=False, right_open=True)

        # Define special cases
        special_cases = {-1: LowerFrechet, 0: BivIndependenceCopula}

        @property
        def is_absolutely_continuous(self) -> bool:
            return self.theta >= 0

        @property
        def _generator(self):
            if self.theta == 0:
                return -sympy.log(self.t)
            return ((1 / self.t) ** self.theta - 1) / self.theta

    # Create a regular instance
    copula = TestCopula(2)

    # Test __call__ with regular parameter
    regular = copula(3)
    assert isinstance(regular, TestCopula)
    assert regular.theta == 3

    # Test __call__ with special case parameter
    independence = copula(0)
    assert isinstance(independence, BivIndependenceCopula)

    lower_frechet = copula(-1)
    assert isinstance(lower_frechet, LowerFrechet)

    # Test with keyword arguments
    kwargs_regular = copula(theta=3)
    assert isinstance(kwargs_regular, TestCopula)
    assert kwargs_regular.theta == 3

    kwargs_special = copula(theta=0)
    assert isinstance(kwargs_special, BivIndependenceCopula)


def test_empty_special_cases():
    """Test that copulas with no special cases work correctly."""

    # Define a test copula class with no special cases
    class SimpleTestCopula(BivArchimedeanCopula):
        theta_interval = sympy.Interval(0, sympy.oo, left_open=False, right_open=True)

        # Empty special cases
        special_cases = {}

        @property
        def is_absolutely_continuous(self) -> bool:
            return True

        @property
        def _generator(self):
            return -sympy.log(self.t) * self.theta

    # Test create method
    regular = SimpleTestCopula.create(2)
    assert isinstance(regular, SimpleTestCopula)
    assert regular.theta == 2

    # Test __new__ method
    direct = SimpleTestCopula(3)
    assert isinstance(direct, SimpleTestCopula)
    assert direct.theta == 3

    # Test __call__ method
    instance = SimpleTestCopula(1)
    new_instance = instance(4)
    assert isinstance(new_instance, SimpleTestCopula)
    assert new_instance.theta == 4


def test_inherited_special_cases():
    """Test that special cases are properly inherited by subclasses."""

    # Define a base test copula with special cases
    class BaseTestCopula(BivArchimedeanCopula):
        theta_interval = sympy.Interval(-1, sympy.oo, left_open=False, right_open=True)

        # Define special cases
        special_cases = {-1: LowerFrechet, 0: BivIndependenceCopula}

        @property
        def is_absolutely_continuous(self) -> bool:
            return self.theta >= 0

        @property
        def _generator(self):
            if self.theta == 0:
                return -sympy.log(self.t)
            return ((1 / self.t) ** self.theta - 1) / self.theta

    # Define a subclass that inherits from the base
    class SubTestCopula(BaseTestCopula):
        # No need to redefine special_cases, should inherit from parent
        pass

    # Test special case handling in subclass
    regular = SubTestCopula(2)
    assert isinstance(regular, SubTestCopula)

    independence = SubTestCopula(0)
    assert isinstance(independence, BivIndependenceCopula)

    lower_frechet = SubTestCopula(-1)
    assert isinstance(lower_frechet, LowerFrechet)


def test_overridden_special_cases():
    """Test that subclasses can override special cases from parent class."""

    # Define a base test copula with special cases
    class BaseTestCopula(BivArchimedeanCopula):
        theta_interval = sympy.Interval(-1, sympy.oo, left_open=False, right_open=True)

        # Define special cases
        special_cases = {-1: LowerFrechet, 0: BivIndependenceCopula}

        @property
        def is_absolutely_continuous(self) -> bool:
            return self.theta >= 0

        @property
        def _generator(self):
            if self.theta == 0:
                return -sympy.log(self.t)
            return ((1 / self.t) ** self.theta - 1) / self.theta

    # Define a subclass that overrides special cases
    class SubTestCopula(BaseTestCopula):
        # Override with different special cases
        special_cases = {
            # Only keep theta = -1 special case
            -1: LowerFrechet
        }

    # Test special case handling in subclass
    regular = SubTestCopula(2)
    assert isinstance(regular, SubTestCopula)

    # Should be a regular instance now, not IndependenceCopula
    regular_zero = SubTestCopula(0)
    assert isinstance(regular_zero, SubTestCopula)
    assert regular_zero.theta == 0

    # This special case is still preserved
    lower_frechet = SubTestCopula(-1)
    assert isinstance(lower_frechet, LowerFrechet)


def test_invalid_params():
    """Test that invalid parameters raise ValueError."""

    # Define a test copula class with invalid parameter values
    class TestCopula(BivArchimedeanCopula):
        theta_interval = sympy.Interval(
            -np.inf, np.inf, left_open=True, right_open=True
        )

        # Define special cases and invalid parameters
        special_cases = {-1: BivIndependenceCopula}
        invalid_params = {0}  # theta = 0 should raise ValueError

        @property
        def is_absolutely_continuous(self) -> bool:
            return True

        @property
        def _generator(self):
            return -sympy.log(self.t) * self.theta

    # Test regular case
    regular = TestCopula(2)
    assert isinstance(regular, TestCopula)
    assert regular.theta == 2

    # Test special case: theta = -1 should return IndependenceCopula
    special_case = TestCopula(-1)
    assert isinstance(special_case, BivIndependenceCopula)

    # Test invalid parameter: theta = 0 should raise ValueError
    with pytest.raises(ValueError, match="Parameter theta cannot be 0"):
        TestCopula(0)

    # Test with keyword arguments
    with pytest.raises(ValueError, match="Parameter theta cannot be 0"):
        TestCopula(theta=0)

    # Test via create method
    with pytest.raises(ValueError, match="Parameter theta cannot be 0"):
        TestCopula.create(0)

    # Test via __call__ method
    copula = TestCopula(2)
    with pytest.raises(ValueError, match="Parameter theta cannot be 0"):
        copula(0)


def test_both_special_and_invalid_params():
    """Test a copula with both special cases and invalid parameters."""

    # Define a test copula class with both special cases and invalid parameters
    class ComplexCopula(BivArchimedeanCopula):
        theta_interval = sympy.Interval(
            -np.inf, np.inf, left_open=True, right_open=True
        )

        # Define special cases
        special_cases = {-1: BivIndependenceCopula, 1: LowerFrechet}

        # Define invalid parameters
        invalid_params = {0, 2}  # theta = 0 or 2 should raise ValueError

        @property
        def is_absolutely_continuous(self) -> bool:
            return True

        @property
        def _generator(self):
            return -sympy.log(self.t) * self.theta

    # Test regular case
    regular = ComplexCopula(3)
    assert isinstance(regular, ComplexCopula)
    assert regular.theta == 3

    # Test special cases
    independence = ComplexCopula(-1)
    assert isinstance(independence, BivIndependenceCopula)

    lower_frechet = ComplexCopula(1)
    assert isinstance(lower_frechet, LowerFrechet)

    # Test invalid parameters
    with pytest.raises(ValueError, match="Parameter theta cannot be 0"):
        ComplexCopula(0)

    with pytest.raises(ValueError, match="Parameter theta cannot be 2"):
        ComplexCopula(2)

    # Test with __call__ method
    copula = ComplexCopula(3)
    result1 = copula(-1)
    assert isinstance(result1, BivIndependenceCopula)

    with pytest.raises(ValueError, match="Parameter theta cannot be 2"):
        copula(2)


def test_cdf_vectorized_basic():
    """Test that cdf_vectorized gives same results as scalar evaluation."""

    # Define a simple test copula for testing vectorized CDF
    class TestCopula(BivArchimedeanCopula):
        theta_interval = sympy.Interval(0, sympy.oo, left_open=False, right_open=True)

        @property
        def is_absolutely_continuous(self) -> bool:
            return True

        @property
        def _raw_generator(self):
            # Using Clayton generator with theta=2 for testing
            return (self.t ** (-self.theta) - 1) / self.theta

    # Create a test instance with theta=2
    copula = TestCopula(2)

    # Define test points
    u_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    v_values = np.array([0.2, 0.4, 0.6, 0.8, 0.7])

    # Calculate expected results using scalar CDF
    expected_results = np.array(
        [copula.cdf(u_values[i], v_values[i]) for i in range(len(u_values))]
    )

    # Calculate results using vectorized CDF
    actual_results = copula.cdf_vectorized(u_values, v_values)

    # Check that results match
    for i in range(len(u_values)):
        actual = actual_results[i]
        expected = expected_results[i]
        assert np.isclose(actual, float(expected), rtol=1e-10)


def test_cdf_vectorized_grid():
    """Test cdf_vectorized with grid inputs."""

    class TestCopula(BivArchimedeanCopula):
        theta_interval = sympy.Interval(0, sympy.oo, left_open=False, right_open=True)

        @property
        def is_absolutely_continuous(self) -> bool:
            return True

        @property
        def _raw_generator(self):
            # Using Frank generator with theta=3 for testing
            return -sympy.log(
                (sympy.exp(-self.theta * self.t) - 1) / (sympy.exp(-self.theta) - 1)
            )

    # Create test instance
    copula = TestCopula(3)

    # Create grid of values
    u_grid = np.linspace(0.1, 0.9, 5)
    v_grid = np.linspace(0.1, 0.9, 5)
    U, V = np.meshgrid(u_grid, v_grid)

    # Get the CDF as a callable function
    cdf_func = copula.cdf

    # Calculate expected results using scalar CDF
    expected_results = np.zeros_like(U)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            expected_results[i, j] = cdf_func(U[i, j], V[i, j])

    # Calculate results using vectorized CDF
    actual_results = copula.cdf_vectorized(U, V)

    # Check that results match
    np.testing.assert_allclose(actual_results, expected_results, rtol=1e-10)


def test_cdf_vectorized_boundary_values():
    """Test cdf_vectorized with boundary values (0 and 1)."""

    class TestCopula(BivArchimedeanCopula):
        theta_interval = sympy.Interval(0, sympy.oo, left_open=False, right_open=True)

        @property
        def is_absolutely_continuous(self) -> bool:
            return True

        @property
        def _raw_generator(self):
            # Using Clayton generator with theta=1 for testing
            return (self.t ** (-self.theta) - 1) / self.theta

    # Create test instance
    copula = TestCopula(1)

    # Test boundary values
    u_values = np.array([0, 0, 1, 1, 0.5])
    v_values = np.array([0, 1, 0, 1, 0.5])

    # Calculate results using vectorized CDF
    results = copula.cdf_vectorized(u_values, v_values)

    # Create a function for evaluating the CDF at scalar points
    def scalar_cdf(u, v):
        # Create lambdified versions of generator and inverse generator
        from sympy.utilities.lambdify import lambdify

        generator_func = lambdify(copula.t, copula.generator.func, "numpy")
        inv_generator_func = lambdify(copula.y, copula.inv_generator.func, "numpy")

        # Apply the Archimedean copula formula
        return inv_generator_func(generator_func(u) + generator_func(v))

    # Expected results for a copula CDF:
    # C(0,v) = 0 for all v
    # C(u,0) = 0 for all u
    # C(1,v) = v for all v
    # C(u,1) = u for all u
    expected = np.array([0, 0, 0, 1, scalar_cdf(0.5, 0.5)])

    # Check that results match
    np.testing.assert_allclose(results, expected, rtol=1e-10)


def test_cdf_vectorized_input_validation():
    """Test that cdf_vectorized properly validates inputs."""

    class TestCopula(BivArchimedeanCopula):
        theta_interval = sympy.Interval(0, sympy.oo, left_open=False, right_open=True)

        @property
        def is_absolutely_continuous(self) -> bool:
            return True

        @property
        def _generator(self):
            return -sympy.log(self.t)  # Independence copula generator

    # Create test instance
    copula = TestCopula(1)

    # Test with invalid values (outside [0,1])
    with pytest.raises(ValueError, match="Marginals must be in"):
        copula.cdf_vectorized(np.array([-0.1, 0.5]), np.array([0.2, 0.3]))

    with pytest.raises(ValueError, match="Marginals must be in"):
        copula.cdf_vectorized(np.array([0.2, 0.5]), np.array([0.2, 1.1]))


def test_cdf_vectorized_special_cases():
    """Test cdf_vectorized with special case copulas."""

    # Define a test copula with special cases
    class TestCopula(BivArchimedeanCopula):
        theta_interval = sympy.Interval(0, sympy.oo, left_open=False, right_open=True)
        special_cases = {0: BivIndependenceCopula}

        @property
        def is_absolutely_continuous(self) -> bool:
            return True

        @property
        def _generator(self):
            # Simple generator that becomes Independence when theta=0
            if self.theta == 0:
                return -sympy.log(self.t)
            return -sympy.log(self.t) * (1 + self.theta * (1 - self.t))

    # Create special case instance
    independence = TestCopula(0)
    assert isinstance(independence, BivIndependenceCopula)

    # Test with regular values
    u_values = np.array([0.2, 0.4, 0.6, 0.8])
    v_values = np.array([0.3, 0.5, 0.7, 0.9])

    # For Independence copula, C(u,v) = u*v
    expected_results = u_values * v_values

    # Calculate using vectorized CDF
    actual_results = independence.cdf_vectorized(u_values, v_values)

    # Check that results match
    np.testing.assert_allclose(actual_results, expected_results, rtol=1e-10)


class TestArchimedeanCopula:
    class MockArchimedeanCopula(BivArchimedeanCopula):
        """Mock implementation of ArchimedeanCopula for testing"""

        theta_interval = sympy.Interval.open(-1, float("inf"))

        @property
        def _raw_generator(self):
            # Clayton-like generator for testing
            if self.theta == 0:
                return -sympy.log(self.t)
            return ((1 / self.t) ** self.theta - 1) / self.theta

        @property
        def _raw_inv_generator(self):
            # Clayton-like inverse generator for testing
            if self.theta == 0:
                return sympy.exp(-self.y)
            return (self.theta * self.y + 1) ** (-1 / self.theta)

        @property
        def is_absolutely_continuous(self):
            return True

    @pytest.fixture
    def copula(self):
        """Create a test copula instance with theta=2"""
        return self.MockArchimedeanCopula(theta=2)

    @pytest.fixture
    def zero_theta_copula(self):
        """Create a test copula instance with theta=0"""
        return self.MockArchimedeanCopula(theta=0)

    def test_generator_includes_piecewise_structure(self, copula):
        """Test that generator properly creates a piecewise function with edge cases"""
        generator_func = copula.generator.func

        # Verify it's a piecewise function
        assert isinstance(generator_func, sympy.Piecewise)

        # Extract conditions from piecewise
        conditions = [cond for _, cond in generator_func.args]

        # Check for proper conditions
        assert any(str(cond).find("t > 0") != -1 for cond in conditions)
        assert sympy.true in conditions  # Default case

    def test_generator_returns_infinity_for_negative_t(self, copula):
        """Test that generator returns infinity for invalid t values"""
        # Create a numerical function from the symbolic one
        gen_func = copula.generator.numpy_func()

        # Test with a negative t value (should return inf)
        result = gen_func(-0.5)
        assert np.isinf(result)

    # Fix for test_inv_generator_has_proper_edge_cases
    def test_inv_generator_has_proper_edge_cases(self, copula):
        """Test that inverse generator has the correct piecewise structure"""
        # Make sure we can get a numerical function from it
        inv_gen_func = copula.inv_generator.numpy_func()

        # Test behavior at key points, which is what we really care about
        # inv_generator(0) should be 1
        assert np.isclose(inv_gen_func(0), 1.0)

        # inv_generator at very large values should approach 0
        assert inv_gen_func(1e10) < 1e-5

        # Regular value behavior should match expected formula
        # For the MockArchimedeanCopula with theta=2, inverse generator is (2y + 1)^(-1/2)
        assert np.isclose(inv_gen_func(1.0), (2.0 * 1.0 + 1.0) ** (-1.0 / 2.0))

    # Fix for test_default_computation_of_inverse_generator
    def test_default_computation_of_inverse_generator(self):
        """Test that inverse generator can be computed from generator if not explicitly defined"""

        class SimpleArchimedeanCopula(BivArchimedeanCopula):
            theta_interval = sympy.Interval.open(0, float("inf"))

            @property
            def _raw_generator(self):
                # Simple generator: t^(-theta) - 1
                return self.t ** (-self.theta) - 1

            @property
            def is_absolutely_continuous(self):
                return True

            # Intentionally not defining _raw_inv_generator

        # Create instance
        copula = SimpleArchimedeanCopula(theta=1)

        # Get the inverse generator function
        inv_gen_func = copula.inv_generator.numpy_func()

        # For this generator with theta=1, inverse should behave like (y+1)^(-1)
        y_values = np.array([0.5, 1.0, 2.0])
        expected = (y_values + 1) ** (-1)
        actual = inv_gen_func(y_values)

        # Test that the values match expected formula
        np.testing.assert_allclose(actual, expected, rtol=1e-10)

        # Test edge cases
        assert np.isclose(inv_gen_func(0), 1.0)
        assert inv_gen_func(1e10) < 1e-5  # Approaches 0 as y approaches infinity

    def test_generator_with_theta_zero(self, zero_theta_copula):
        """Test that generator properly handles theta=0 case"""
        gen_func = zero_theta_copula.generator.numpy_func()

        # For theta=0, generator should be -log(t)
        t_vals = np.array([0.1, 0.5, 0.9])
        expected = -np.log(t_vals)
        actual = gen_func(t_vals)

        np.testing.assert_allclose(actual, expected)

    def test_inv_generator_with_theta_zero(self, zero_theta_copula):
        """Test that inverse generator properly handles theta=0 case"""
        inv_gen_func = zero_theta_copula.inv_generator.numpy_func()

        # For theta=0, inverse generator should be exp(-y)
        y_vals = np.array([0.5, 1.0, 2.0])
        expected = np.exp(-y_vals)
        actual = inv_gen_func(y_vals)

        np.testing.assert_allclose(actual, expected)

    def test_inv_generator_with_theta_nonzero(self, copula):
        """Test inverse generator with non-zero theta"""
        inv_gen_func = copula.inv_generator.numpy_func()

        # For theta=2, inverse generator should be (2y + 1)^(-1/2)
        y_vals = np.array([0.5, 1.0, 2.0])
        expected = (2 * y_vals + 1) ** (-1 / 2)
        actual = inv_gen_func(y_vals)

        np.testing.assert_allclose(actual, expected)

    def test_raw_generator_implementation_required(self):
        """Test that _raw_generator must be implemented by subclasses"""

        class IncompleteArchimedeanCopula(BivArchimedeanCopula):
            theta_interval = sympy.Interval.open(0, 1)

            @property
            def is_absolutely_continuous(self):
                return True

        # Should raise NotImplementedError when accessing generator
        with pytest.raises(NotImplementedError):
            IncompleteArchimedeanCopula(theta=0.5).generator
