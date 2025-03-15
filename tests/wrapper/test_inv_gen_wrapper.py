import numpy as np
import pytest
import sympy

from copul.wrapper.inv_gen_wrapper import InvGenWrapper


# Mock classes for testing
class MockCopula:
    """Mock copula class for testing the InvGenWrapper"""

    t = sympy.symbols("t", nonnegative=True)
    y = sympy.symbols("y", nonnegative=True)
    theta = sympy.symbols("theta", nonnegative=True)
    _generator_at_0 = sympy.oo  # Default value

    def __init__(self, theta_val=1.0):
        self.theta = theta_val


class MockNelsen11Copula(MockCopula):
    """Mock Nelsen11 copula for testing special cases"""

    _generator_at_0 = sympy.log(2)

    def __init__(self, theta_val=0.25):
        super().__init__(theta_val)

    @property
    def __class__(self):
        # Create a mock class with the correct name for type checking
        class MockClass:
            __name__ = "Nelsen11"

        return MockClass


class TestInvGenWrapper:
    """Test suite for the InvGenWrapper class"""

    @pytest.fixture
    def y_symbol(self):
        """Fixture for the y symbol"""
        return sympy.symbols("y", nonnegative=True)

    @pytest.fixture
    def basic_expr(self):
        """Fixture for a basic expression: (1 + 2*y)^(-1/2)"""
        y = sympy.symbols("y", nonnegative=True)
        return (1 + 2 * y) ** (-0.5)

    @pytest.fixture
    def mock_copula(self):
        """Fixture for a standard mock copula"""
        return MockCopula()

    @pytest.fixture
    def nelsen11_copula(self):
        """Fixture for a Nelsen11 mock copula"""
        return MockNelsen11Copula()

    @pytest.fixture
    def basic_wrapper(self, basic_expr, y_symbol, mock_copula):
        """Fixture for a basic InvGenWrapper instance"""
        return InvGenWrapper(basic_expr, y_symbol, mock_copula)

    @pytest.fixture
    def nelsen_wrapper(self, y_symbol, nelsen11_copula):
        """Fixture for a Nelsen11-specific wrapper with the special case expression"""
        # For Nelsen11 at y=log(2), expression is 0**(1/theta)
        return InvGenWrapper(
            sympy.Pow(0, 1 / nelsen11_copula.theta), y_symbol, nelsen11_copula
        )

    def test_init(self, basic_expr, y_symbol, mock_copula):
        """Test initialization of InvGenWrapper"""
        wrapper = InvGenWrapper(basic_expr, y_symbol, mock_copula)

        assert wrapper._func == basic_expr
        assert wrapper.y_symbol == y_symbol
        assert wrapper.copula == mock_copula
        assert wrapper.theta_val == mock_copula.theta
        assert wrapper.generator_at_0 == mock_copula._generator_at_0

    def test_call_basic(self, basic_wrapper, y_symbol):
        """Test the __call__ method for basic cases"""
        # Test with y=0
        result = basic_wrapper(y=0)
        assert isinstance(result, InvGenWrapper)
        assert float(result) == 1.0

        # Test with a regular value
        result = basic_wrapper(y=1.0)
        assert isinstance(result, InvGenWrapper)
        assert abs(float(result) - (1 + 2 * 1.0) ** (-0.5)) < 1e-10

    def test_call_special_cases(self, basic_wrapper, y_symbol):
        """Test special cases in the __call__ method"""
        # Test with y=infinity
        result = basic_wrapper(y=sympy.oo)
        assert isinstance(result, InvGenWrapper)
        assert float(result) == 0.0

    def test_call_nelsen11_log2(self, nelsen_wrapper, y_symbol):
        """Test the Nelsen11 special case with y=log(2)"""
        # Test with y=log(2)
        result = nelsen_wrapper(y=sympy.log(2))
        assert isinstance(result, InvGenWrapper)
        assert float(result) == 0.0

    def test_float_nelsen11(self, nelsen_wrapper):
        """Test float conversion for Nelsen11 special case"""
        # The wrapper contains 0**(1/theta) which should convert to 0.0
        assert float(nelsen_wrapper) == 0.0

    def test_subs_special_cases(self, basic_wrapper, y_symbol):
        """Test special cases in the subs method"""
        # Test with y=0
        result = basic_wrapper.subs(y_symbol, 0)
        assert float(result) == 1.0

        # Test with y=infinity
        result = basic_wrapper.subs(y_symbol, sympy.oo)
        assert float(result) == 0.0

    def test_subs_nelsen11_log2(self, nelsen_wrapper, y_symbol):
        """Test the Nelsen11 special case in the subs method"""
        # Test with y=log(2)
        result = nelsen_wrapper.subs(y_symbol, sympy.log(2))
        assert float(result) == 0.0

    def test_numpy_func_basic(self, basic_wrapper):
        """Test the numpy_func method for basic cases"""
        func = basic_wrapper.numpy_func()

        # Test with regular values
        y_vals = np.array([0.5, 1.0, 2.0])
        expected = (1 + 2 * y_vals) ** (-0.5)
        actual = func(y_vals)

        np.testing.assert_allclose(actual, expected)

    def test_numpy_func_edge_cases(self, basic_wrapper):
        """Test the numpy_func method with edge cases"""
        func = basic_wrapper.numpy_func()

        # Test with y=0
        assert func(0.0) == 1.0

        # Test with y=infinity
        assert func(np.inf) == 0.0

    def test_numpy_func_vectorized(self, basic_wrapper):
        """Test that numpy_func properly handles vectorized operations"""
        func = basic_wrapper.numpy_func()

        # Create a grid of values
        y_grid = np.linspace(0.1, 2.0, 5)
        Y = y_grid.reshape(-1, 1) * np.ones((5, 5))

        # Expected values
        expected = (1 + 2 * Y) ** (-0.5)

        # Actual values
        actual = func(Y)

        # Check shape and values
        assert actual.shape == Y.shape
        np.testing.assert_allclose(actual, expected)

    def test_numpy_func_mixed_edge_cases(self, basic_wrapper):
        """Test that numpy_func handles arrays with mixed regular and edge cases"""
        func = basic_wrapper.numpy_func()

        # Create array with regular values, zeros, and infinity
        y_vals = np.array([0.0, 0.5, 1.0, np.inf])

        # Expected values
        expected = np.array(
            [1.0, (1 + 2 * 0.5) ** (-0.5), (1 + 2 * 1.0) ** (-0.5), 0.0]
        )

        # Actual values
        actual = func(y_vals)

        np.testing.assert_allclose(actual, expected)

    def test_numpy_func_nelsen11(self, nelsen_wrapper):
        """Test numpy_func with Nelsen11 critical value"""
        func = nelsen_wrapper.numpy_func()

        # Test with values around log(2)
        log2_val = float(sympy.log(2))
        y_vals = np.array([log2_val - 0.1, log2_val, log2_val + 0.1])

        # Expected: values less than log(2) follow regular formula,
        # values >= log(2) should be 0
        expected = np.array([func(log2_val - 0.1), 0.0, 0.0])

        actual = func(y_vals)

        np.testing.assert_allclose(actual, expected)
