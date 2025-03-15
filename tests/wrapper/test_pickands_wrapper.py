"""
Tests for the PickandsWrapper class in copul.wrapper.pickands_wrapper.
"""

import math
import sympy as sp
from unittest.mock import patch

from copul.wrapper.pickands_wrapper import PickandsWrapper


class TestPickandsWrapper:
    """Tests for the PickandsWrapper class."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create basic symbols for testing
        self.t = sp.Symbol("t")

        # Create some example expressions
        self.linear_expr = 2 * self.t + 1
        self.quadratic_expr = self.t**2 + 3 * self.t + 2
        self.reciprocal_expr = 1 / (1 + self.t)

        # Create a Galambos-like expression: 1 - (u^(-1/δ) + v^(-1/δ))^(-δ)
        # Simplified for testing purposes
        self.galambos_expr = 1 - (self.t ** (-1 / 2) + (1 - self.t) ** (-1 / 2)) ** (-2)

        # Initialize wrappers
        self.linear_wrapper = PickandsWrapper(self.linear_expr, self.t)
        self.quadratic_wrapper = PickandsWrapper(self.quadratic_expr, self.t)
        self.galambos_wrapper = PickandsWrapper(
            self.galambos_expr, self.t, delta_val=2.0
        )

    def test_init(self):
        """Test initialization of PickandsWrapper."""
        # Test basic initialization
        wrapper = PickandsWrapper(self.linear_expr, self.t)
        assert wrapper.expr == self.linear_expr
        assert wrapper.t_symbol == self.t
        assert wrapper.delta_val is None
        assert wrapper.func == self.linear_expr

        # Test with delta value
        wrapper = PickandsWrapper(self.galambos_expr, self.t, delta_val=2.0)
        assert wrapper.expr == self.galambos_expr
        assert wrapper.t_symbol == self.t
        assert wrapper.delta_val == 2.0

    def test_call_with_value(self):
        """Test calling the wrapper with a value."""
        # Test linear expression with numeric value
        result = self.linear_wrapper(0.5)
        assert result == 2 * 0.5 + 1
        assert float(result) == 2.0

        # Test quadratic expression with numeric value
        result = self.quadratic_wrapper(0.5)
        assert result == 0.5**2 + 3 * 0.5 + 2
        assert float(result) == 3.75

        # Test with symbolic value
        x = sp.Symbol("x")
        result = self.linear_wrapper(x)
        assert result == 2 * x + 1

    def test_call_without_value(self):
        """Test calling the wrapper without a value."""
        # Should return the original expression
        result = self.linear_wrapper()
        assert result == self.linear_expr

        result = self.quadratic_wrapper()
        assert result == self.quadratic_expr

    def test_galambos_special_case_edge_cases(self):
        """Test edge cases for the Galambos special handling."""
        # Test with slightly different t
        result = self.galambos_wrapper(0.500001)
        # Should not trigger special case due to not being exactly 0.5
        assert not math.isclose(float(result), float(sp.Float("0.6464466094067263")))

        # Create a wrapper with a different delta
        different_delta_wrapper = PickandsWrapper(
            self.galambos_expr, self.t, delta_val=2.1
        )
        result = different_delta_wrapper(0.5)
        # Should not trigger special case due to different delta
        assert not math.isclose(float(result), float(sp.Float("0.6464466094067263")))

    def test_subs(self):
        """Test the subs method."""
        # Substitute t with a numeric value
        result = self.linear_wrapper.subs(self.t, 0.5)
        assert result == 2 * 0.5 + 1

        # Substitute t with another symbol
        x = sp.Symbol("x")
        result = self.linear_wrapper.subs(self.t, x)
        assert result == 2 * x + 1

        # Substitute with a more complex expression
        result = self.quadratic_wrapper.subs(self.t, x**2)
        assert result == (x**2) ** 2 + 3 * (x**2) + 2
        assert result == x**4 + 3 * x**2 + 2

    def test_exception_handling(self):
        """Test that exceptions in special case logic are handled gracefully."""
        # Mock the math.isclose function to raise an exception
        with patch("math.isclose", side_effect=Exception("Test exception")):
            # Should not crash but fall back to regular behavior
            result = self.galambos_wrapper(0.5)
            # Won't get the special case value but should return something
            assert result is not None

    def test_non_symbolic_input(self):
        """Test with non-symbolic expressions that might not have all methods."""
        # Create a wrapper with a regular Python float
        regular_float = 3.14159
        float_wrapper = PickandsWrapper(regular_float, self.t)

        # Should still work with the call method
        assert float_wrapper() == regular_float

        # Should handle evalf gracefully
        result = float_wrapper.evalf()
        assert result == regular_float

    def test_pickling(self):
        """Test that PickandsWrapper can be pickled and unpickled."""
        import pickle

        # Pickle and unpickle linear wrapper
        pickled = pickle.dumps(self.linear_wrapper)
        unpickled = pickle.loads(pickled)

        # Check that attributes are preserved
        assert str(unpickled.expr) == str(self.linear_wrapper.expr)
        assert str(unpickled.t_symbol) == str(self.linear_wrapper.t_symbol)
        assert unpickled.delta_val == self.linear_wrapper.delta_val

        # Check functionality is preserved
        assert float(unpickled(0.5)) == float(self.linear_wrapper(0.5))
