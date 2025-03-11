"""
Tests for the SymPyFuncWrapper class.
"""

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
