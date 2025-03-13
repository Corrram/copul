import unittest
import sympy as sp
from copul.families.helpers import get_simplified_solution, concrete_expand_log


class TestSympyHelpers(unittest.TestCase):
    def test_get_simplified_solution_basic(self):
        """Test that a basic expression is simplified correctly."""
        expr = sp.sqrt(4)
        result = get_simplified_solution(expr)
        self.assertEqual(result, 2)

    def test_get_simplified_solution_tuple(self):
        """Test that if the simplified result is a Tuple, the first element is returned."""
        # Construct a Tuple explicitly.
        tup = sp.Tuple(1, 2, 3)
        # Note: sp.simplify on a Tuple returns the same tuple.
        result = get_simplified_solution(tup)
        self.assertEqual(result, 1)

    def test_concrete_expand_log_basic(self):
        """Test that a simple logarithm of a product is expanded into a sum."""
        a, b = sp.symbols("a b", positive=True)
        expr = sp.log(a * b)
        expanded = concrete_expand_log(expr)
        # Expected expansion: log(a) + log(b)
        expected = sp.log(a) + sp.log(b)
        # Check that the difference simplifies to zero.
        self.assertTrue(sp.simplify(expanded - expected) == 0)

    def test_concrete_expand_log_recursive(self):
        """Test that a more complex logarithm is recursively expanded."""
        a, b, c = sp.symbols("a b c", positive=True)
        expr = sp.log(a * b * c)
        expanded = concrete_expand_log(expr)
        expected = sp.log(a) + sp.log(b) + sp.log(c)
        self.assertTrue(sp.simplify(expanded - expected) == 0)


if __name__ == "__main__":
    unittest.main()
