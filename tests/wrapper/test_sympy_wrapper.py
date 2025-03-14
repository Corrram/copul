"""
Tests for the SymPyFuncWrapper class.
"""

import time
import pytest
import numpy as np
import sympy
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class TestSymPyFuncWrapper:
    """Tests for the SymPyFuncWrapper class."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create symbolic variables
        self.x, self.y, self.z = sympy.symbols("x y z")

        # Create some test expressions
        self.expr1 = self.x**2 + self.y
        self.expr2 = sympy.sin(self.x) * self.y
        self.expr3 = sympy.exp(self.x + self.y)

        # Create SymPyFuncWrapper instances
        self.func1 = SymPyFuncWrapper(self.expr1)
        self.func2 = SymPyFuncWrapper(self.expr2)
        self.func3 = SymPyFuncWrapper(self.expr3)

    def test_initialization(self):
        """Test initialization of SymPyFuncWrapper."""
        # Test with a sympy expression
        func = SymPyFuncWrapper(self.expr1)
        assert isinstance(func, SymPyFuncWrapper)
        assert func.func == self.expr1

        # Test with another SymPyFuncWrapper
        func2 = SymPyFuncWrapper(func)
        assert isinstance(func2, SymPyFuncWrapper)
        assert func2.func == func.func

        # Test with a float
        func3 = SymPyFuncWrapper(3.14)
        assert isinstance(func3, SymPyFuncWrapper)
        assert func3.func == sympy.Number(3.14)

        # Test with invalid type
        with pytest.raises(AssertionError):
            SymPyFuncWrapper("not a sympy expression")

    def test_str_repr(self):
        """Test string representation."""
        assert str(self.func1) == str(self.expr1)
        assert repr(self.func1) == repr(self.expr1)

    def test_float_conversion(self):
        """Test float conversion."""
        # Convert a numeric expression to float
        numeric_expr = SymPyFuncWrapper(sympy.Number(2.5))
        assert float(numeric_expr) == 2.5

        # Should raise an exception for expressions with free symbols
        with pytest.raises(Exception):
            float(self.func1)

    def test_call_with_args(self):
        """Test __call__ method with positional arguments."""
        # Call with positional args matching the order of free symbols
        result = self.func1(2, 3)  # x=2, y=3
        assert isinstance(result, SymPyFuncWrapper)
        assert result.func == 2**2 + 3

        # Try to get float with handling potential TypeError
        try:
            result_value = float(result.evalf())
            assert abs(result_value - 7) < 1e-10
        except TypeError:
            # If conversion fails, check the expression directly
            assert result.func == 7

    def test_call_with_kwargs(self):
        """Test __call__ method with keyword arguments."""
        # Call with keyword args
        result = self.func1(x=2, y=3)
        assert isinstance(result, SymPyFuncWrapper)
        assert result.func == 2**2 + 3
        assert abs(float(result.evalf()) - 7) < 1e-10

        # Call with partial substitution
        result = self.func1(x=2)
        assert isinstance(result, SymPyFuncWrapper)
        assert result.func == 4 + self.y

        # Call with irrelevant kwargs
        result = self.func1(z=5)
        assert result.func == self.expr1  # No substitution

    def test_call_with_mixed_args(self):
        """Test __call__ method with mixed args and kwargs."""
        # This should raise a ValueError according to the implementation
        with pytest.raises(ValueError):
            self.func1(2, y=3)

    def test_prepare_call(self):
        """Test _prepare_call method."""
        # Test with args
        vars_, kwargs = self.func1._prepare_call([2, 3], {})
        assert vars_ == {self.x: 2, self.y: 3}
        assert kwargs == {str(self.x): 2, str(self.y): 3}

        # Test with kwargs
        vars_, kwargs = self.func1._prepare_call([], {"x": 2, "y": 3})
        assert vars_ == {self.x: 2, self.y: 3}
        assert kwargs == {"x": 2, "y": 3}

        # Test with None values in kwargs (should be filtered out)
        vars_, kwargs = self.func1._prepare_call([], {"x": 2, "y": None})
        assert vars_ == {self.x: 2}
        assert kwargs == {"x": 2}

        # Test with mixed args and kwargs (this would raise an error in __call__)
        with pytest.raises(ValueError):
            self.func1._prepare_call([2], {"y": 3})

    def test_func_property(self):
        """Test func property."""
        assert self.func1.func == self.expr1

    def test_subs_method(self):
        """Test subs method."""
        # Substitute x with 2
        result = self.func1.subs(self.x, 2)
        assert isinstance(result, SymPyFuncWrapper)
        assert result.func == 4 + self.y
        assert result is not self.func1  # subs does not modify in place

        # Substitute y with 3
        result = result.subs(self.y, 3)
        assert result.func == 7
        assert result is not self.func1  # subs does not modify in place

    def test_diff_method(self):
        """Test diff method."""
        # Differentiate with respect to x
        result = SymPyFuncWrapper(self.expr1).diff(self.x)
        assert isinstance(result, SymPyFuncWrapper)
        assert result.func == 2 * self.x

        # Differentiate with respect to y
        result = SymPyFuncWrapper(self.expr1).diff(self.y)
        assert result.func == 1

    def test_to_latex(self):
        """Test to_latex method."""
        latex_repr = self.func1.to_latex()
        expected = sympy.latex(self.expr1)
        assert latex_repr == expected

    def test_evalf(self):
        """Test evalf method."""
        # For a numeric expression
        numeric_expr = SymPyFuncWrapper(sympy.Number(2.5))
        assert numeric_expr.evalf() == 2.5

        # For a symbolic expression with substituted values
        expr_with_values = self.func1.subs({self.x: 2, self.y: 3})
        # Use approximate comparison instead of exact equality
        assert abs(float(expr_with_values.evalf()) - 7) < 1e-10

        # For a symbolic expression without values
        assert isinstance(self.func1.evalf(), sympy.core.expr.Expr)

    def test_equality(self):
        """Test equality comparison."""
        # Same expression should be equal
        func1a = SymPyFuncWrapper(self.x**2 + self.y)
        func1b = SymPyFuncWrapper(self.x**2 + self.y)
        assert func1a == func1b

        # Different expressions should not be equal
        assert self.func1 != self.func2

        # Compare with raw sympy expression
        assert self.func1 == self.expr1
        assert self.func1 != self.expr2

    def test_inequality(self):
        """Test inequality comparison."""
        # Same expression should not be unequal
        func1a = SymPyFuncWrapper(self.x**2 + self.y)
        func1b = SymPyFuncWrapper(self.x**2 + self.y)
        assert not (func1a != func1b)

        # Different expressions should be unequal
        assert self.func1 != self.func2

    def test_arithmetic_operations(self):
        """Test arithmetic operations."""
        # Addition
        result = self.func1 + self.func2
        assert isinstance(result, SymPyFuncWrapper)
        assert result.func == self.expr1 + self.expr2

        # Addition with a sympy expression
        result = self.func1 + self.expr2
        assert isinstance(result, SymPyFuncWrapper)
        assert result.func == self.expr1 + self.expr2

        # Subtraction
        result = self.func1 - self.func2
        assert isinstance(result, SymPyFuncWrapper)
        assert result.func == self.expr1 - self.expr2

        # Multiplication
        result = self.func1 * self.func2
        assert isinstance(result, SymPyFuncWrapper)
        assert result.func == self.expr1 * self.expr2

        # Division
        result = self.func1 / self.func2
        assert isinstance(result, SymPyFuncWrapper)
        assert result.func == self.expr1 / self.expr2

        # Power
        result = self.func1**2
        assert isinstance(result, SymPyFuncWrapper)
        assert result.func == self.expr1**2

        # Power with another wrapper
        result = self.func1**self.func2
        assert isinstance(result, SymPyFuncWrapper)
        assert result.func == self.expr1**self.expr2

    def test_isclose(self):
        """Test isclose method."""
        # Create two expressions that evaluate to the same value
        expr1 = SymPyFuncWrapper(self.x**2)
        expr2 = SymPyFuncWrapper(self.x * self.x)

        # Substitute values
        expr1_eval = expr1(x=2)
        expr2_eval = expr2(x=2)

        # Should be close
        assert expr1_eval.isclose(expr2_eval)
        assert expr1_eval.isclose(4)

        # Should not be close
        assert not expr1_eval.isclose(4.1)

        # Test with non-SymPyFuncWrapper value
        assert expr1_eval.isclose(4.0)
        assert not expr1_eval.isclose(3.9)


def test_persistence_of_orig_func():
    x = sympy.symbols("x")
    func = x**2
    wrapped_func = SymPyFuncWrapper(func)
    assert wrapped_func(2).isclose(4)
    assert wrapped_func(1).isclose(1)


def test_evalf():
    x = sympy.symbols("x")
    func = x**2
    wrapped_func = SymPyFuncWrapper(func)
    assert np.isclose(float(wrapped_func(2).evalf()), 4)


def test_numpy_func():
    """Test the numpy_func method for vectorized evaluation."""
    # Create symbolic variables
    x, y = sympy.symbols("x y")

    # Test with a univariate expression
    expr1 = x**2 + 3
    func1 = SymPyFuncWrapper(expr1)
    numpy_func1 = func1.numpy_func()

    # Test with scalar input
    result1 = numpy_func1(2.0)
    assert np.isclose(result1, 7.0)

    # Test with array input
    x_values = np.array([1.0, 2.0, 3.0, 4.0])
    expected1 = x_values**2 + 3
    result1_array = numpy_func1(x_values)
    np.testing.assert_allclose(result1_array, expected1)

    # Test with a bivariate expression
    expr2 = x**2 + y**2
    func2 = SymPyFuncWrapper(expr2)
    numpy_func2 = func2.numpy_func()

    # Test with scalar inputs
    result2 = numpy_func2(2.0, 3.0)
    assert np.isclose(result2, 13.0)

    # Test with array inputs of the same shape
    x_values = np.array([1.0, 2.0, 3.0, 4.0])
    y_values = np.array([5.0, 4.0, 3.0, 2.0])
    expected2 = x_values**2 + y_values**2
    result2_array = numpy_func2(x_values, y_values)
    np.testing.assert_allclose(result2_array, expected2)

    # Test with broadcasting
    x_values = np.array([1.0, 2.0, 3.0, 4.0])
    y_scalar = 5.0
    expected2_broadcast = x_values**2 + y_scalar**2
    result2_broadcast = numpy_func2(x_values, y_scalar)
    np.testing.assert_allclose(result2_broadcast, expected2_broadcast)

    # Test with a constant expression
    expr3 = sympy.Number(7.5)
    func3 = SymPyFuncWrapper(expr3)
    numpy_func3 = func3.numpy_func()

    # Should return the constant value regardless of input
    result3_scalar = numpy_func3()
    assert np.isclose(result3_scalar, 7.5)

    # Should broadcast the constant to match input shape
    result3_array = numpy_func3(np.ones(5))
    np.testing.assert_allclose(result3_array, np.full(5, 7.5))


def test_numpy_func_with_mathematical_functions():
    """Test numpy_func with mathematical functions like sin, exp, etc."""

    # Create symbolic variables
    x = sympy.symbols("x")

    # Test with sin function
    expr1 = sympy.sin(x)
    func1 = SymPyFuncWrapper(expr1)
    numpy_func1 = func1.numpy_func()

    # Test with array input
    x_values = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    expected1 = np.sin(x_values)
    result1 = numpy_func1(x_values)
    np.testing.assert_allclose(result1, expected1, rtol=1e-10)

    # Test with exp function
    expr2 = sympy.exp(x)
    func2 = SymPyFuncWrapper(expr2)
    numpy_func2 = func2.numpy_func()

    # Test with array input
    x_values = np.array([0, 1, 2])
    expected2 = np.exp(x_values)
    result2 = numpy_func2(x_values)
    np.testing.assert_allclose(result2, expected2, rtol=1e-10)

    # Test with more complex expression
    expr3 = sympy.sin(x) ** 2 + sympy.cos(x) ** 2
    func3 = SymPyFuncWrapper(expr3)
    numpy_func3 = func3.numpy_func()

    # This should always be approximately 1 (trigonometric identity)
    x_values = np.linspace(0, 2 * np.pi, 100)
    expected3 = np.ones_like(x_values)
    result3 = numpy_func3(x_values)
    np.testing.assert_allclose(result3, expected3, rtol=1e-10)

def test_numpy_func_with_piecewise():
    """Test numpy_func with piecewise expressions."""

    # Create symbolic variables
    x = sympy.symbols("x")

    # Create a piecewise function
    expr = sympy.Piecewise(
        (1/x, x > 0),
        (1+ x**2, True)
    )
    func = SymPyFuncWrapper(expr)
    numpy_func = func.numpy_func()

    # Test with scalar input
    result1 = numpy_func(0)
    assert result1 == 1

    result2 = numpy_func(2)
    assert result2 == 0.5

    # Test with array input
    x_values = np.array([-1, 0, 1, 2])
    expected = np.array([2, 1, 1, 0.5])
    result = numpy_func(x_values)
    np.testing.assert_allclose(result, expected)

def test_numpy_func_performance():
    """Test that numpy_func provides performance benefits for vectorized operations."""

    # Create a moderately complex expression
    x, y = sympy.symbols("x y")
    expr = sympy.sin(x) * sympy.exp(y / 2) + sympy.cos(x * y)
    func = SymPyFuncWrapper(expr)

    # Create large arrays of values
    np.random.seed(42)  # For reproducibility
    x_values = np.random.random(1000)
    y_values = np.random.random(1000)

    # Get numpy function
    numpy_func = func.numpy_func()

    # Measure time for loop-based evaluation
    start_loop = time.time()
    loop_results = np.zeros(len(x_values))
    for i in range(len(x_values)):
        loop_results[i] = float(func(x_values[i], y_values[i]).evalf())
    loop_time = time.time() - start_loop

    # Measure time for vectorized evaluation
    start_vector = time.time()
    vector_results = numpy_func(x_values, y_values)
    vector_time = time.time() - start_vector

    # Check that results match
    np.testing.assert_allclose(vector_results, loop_results, rtol=1e-10)

    # Vectorized operation should be significantly faster
    assert vector_time < loop_time * 0.5, (
        f"Vectorized: {vector_time}s, Loop: {loop_time}s"
    )
