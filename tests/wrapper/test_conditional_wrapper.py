import unittest
import sympy as sp
from sympy import symbols
import numpy as np

# Import the wrapper classes being tested
from copul.wrapper.conditional_wrapper import ConditionalWrapper

class TestConditionalWrapper(unittest.TestCase):
    """
    Tests for the ConditionalWrapper base class functionality
    """
    
    def setUp(self):
        """Set up test fixtures"""
        self.u, self.v = symbols("u v", positive=True)
        self.u1, self.u2, self.u3 = symbols("u1 u2 u3", positive=True)
        
        # Create a simple expression for testing
        self.expr = self.u * self.v
        self.multi_expr = self.u1 * self.u2 * self.u3
        
        # Create a custom ConditionalWrapper subclass for testing
        class TestWrapper(ConditionalWrapper):
            def _check_boundary_conditions(self, u_symbols, vars_dict, kwargs):
                return None
        
        self.TestWrapper = TestWrapper
        
    def test_initialization(self):
        """Test initialization of ConditionalWrapper"""
        wrapper = self.TestWrapper(self.expr, condition_index=1)
        self.assertEqual(wrapper.condition_index, 1)
        self.assertEqual(str(wrapper.func), str(self.expr))
        
    def test_get_u_symbols(self):
        """Test extraction of u symbols"""
        wrapper = self.TestWrapper(self.expr, condition_index=1)
        u_symbols = wrapper._get_u_symbols()
        
        self.assertEqual(len(u_symbols), 2)
        self.assertTrue("u" in u_symbols)
        self.assertTrue("v" in u_symbols)
        self.assertEqual(str(u_symbols["u"]), "u")
        self.assertEqual(str(u_symbols["v"]), "v")
        
    def test_get_u_symbols_multivariate(self):
        """Test extraction of u symbols in multivariate case"""
        wrapper = self.TestWrapper(self.multi_expr, condition_index=1)
        u_symbols = wrapper._get_u_symbols()
        
        self.assertEqual(len(u_symbols), 3)
        self.assertTrue("u1" in u_symbols)
        self.assertTrue("u2" in u_symbols)
        self.assertTrue("u3" in u_symbols)
        self.assertEqual(str(u_symbols["u1"]), "u1")
        self.assertEqual(str(u_symbols["u2"]), "u2")
        self.assertEqual(str(u_symbols["u3"]), "u3")
        
    def test_call_with_substitution(self):
        """Test calling wrapper with value substitutions"""
        wrapper = self.TestWrapper(self.expr, condition_index=1)
        
        # Test substitution using positional args
        result = wrapper(0.5, 0.3)
        self.assertEqual(float(result), 0.5 * 0.3)
        
        # Test substitution using kwargs
        result = wrapper(u=0.7, v=0.4)
        self.assertEqual(float(result), 0.7 * 0.4)
        
    def test_call_with_symbolic_params(self):
        """Test calling wrapper with symbolic parameters"""
        a, b = symbols("a b", positive=True)
        expr = a * self.u * self.v
        wrapper = self.TestWrapper(expr, condition_index=1)
        
        # Substitute for u and v but leave a symbolic
        result = wrapper(a, 0.5, 0.4)
        self.assertEqual(str(result.func), "0.2*a")
        
        # Substitute for a as well
        result = result.subs(a, 2)
        self.assertEqual(float(result.func), 0.4)


class TestConditionalWrapperInheritance(unittest.TestCase):
    """
    Tests for checking that the ConditionalWrapper inheritance works correctly.
    """
    
    def setUp(self):
        """Set up test fixtures"""
        self.u, self.v = symbols("u v", positive=True)
        # Create a simple expression for testing
        self.cdf = self.u * self.v
        
        # Create a custom subclass with a specific boundary condition
        class CustomWrapper(ConditionalWrapper):
            def _check_boundary_conditions(self, u_symbols, vars_dict, kwargs):
                u_sym = u_symbols.get("u", symbols("u"))
                if vars_dict.get(u_sym) == 0.5 or kwargs.get("u") == 0.5:
                    return sp.sympify(99)  # Return a special value for testing
                return None
                
        self.CustomWrapper = CustomWrapper
    
    def test_boundary_condition_check(self):
        """Test that boundary conditions are properly checked and returned"""
        wrapper = self.CustomWrapper(self.cdf)
        
        # Test normal case
        result = wrapper(0.3, 0.7)
        self.assertEqual(float(result.func), 0.3 * 0.7)
        
        # Test boundary condition
        result = wrapper(0.5, 0.7)
        self.assertEqual(float(result), 99)
        
        # Test boundary with kwargs
        result = wrapper(u=0.5, v=0.9)
        self.assertEqual(float(result), 99)