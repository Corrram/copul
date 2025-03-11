"""
Tests for the CD1Wrapper class (partial derivative with respect to first argument).
"""

import sympy
from sympy import simplify
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper
from copul.wrapper.cd1_wrapper import CD1Wrapper


class TestCD1Wrapper:
    """Tests for the CD1Wrapper class."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create symbolic variables
        self.u, self.v, self.theta = sympy.symbols("u v theta")

        # Create some test expressions representing partial derivatives of copulas
        # For independence copula C(u,v) = u*v, the derivative ∂C/∂u = v
        self.indep_deriv = self.v

        # For Clayton copula, partial derivative with respect to u
        # This is a complex expression, simplified for testing
        theta_val = 2
        self.clayton_deriv = self.u ** (-theta_val - 1) * (
            self.u ** (-theta_val) + self.v ** (-theta_val) - 1
        ) ** (-(1 / theta_val) - 1)

        # Create CD1Wrapper instances
        self.indep_cd1 = CD1Wrapper(self.indep_deriv)
        self.clayton_cd1 = CD1Wrapper(self.clayton_deriv)

    def test_initialization(self):
        """Test initialization of CD1Wrapper."""
        # Test initialization with a SymPy expression
        cd1 = CD1Wrapper(self.v)
        assert isinstance(cd1, CD1Wrapper)
        assert isinstance(cd1, SymPyFuncWrapper)
        assert str(cd1.func) == str(self.v)

        # Test initialization with another CD1Wrapper
        cd1_2 = CD1Wrapper(cd1)
        assert isinstance(cd1_2, CD1Wrapper)
        assert cd1_2.func == cd1.func

    def test_boundary_conditions(self):
        """Test boundary conditions for conditional distributions."""
        # When v=0, CD1 should be 0
        assert self.indep_cd1(v=0) == SymPyFuncWrapper(sympy.S.Zero)

        # When v=1, CD1 should be 1
        assert self.indep_cd1(v=1) == SymPyFuncWrapper(sympy.S.One)

        # Clayton should also follow these rules
        assert self.clayton_cd1(v=0) == SymPyFuncWrapper(sympy.S.Zero)
        assert self.clayton_cd1(v=1) == SymPyFuncWrapper(sympy.S.One)

    def test_substitution(self):
        """Test variable substitution."""
        # Test with independence copula derivative
        result = self.indep_cd1(u=0.5, v=0.7)
        assert isinstance(result, CD1Wrapper)
        assert result.evalf() == 0.7

        # Test with Clayton copula derivative
        clayton_at_point = self.clayton_cd1(u=0.5, v=0.7)
        assert isinstance(clayton_at_point, CD1Wrapper)

        # The exact value depends on the formula, we just check it's a number
        assert isinstance(float(clayton_at_point.evalf()), float)

    def test_partial_substitution(self):
        """Test partial substitution of variables."""
        # Substitute only u
        indep_u_05 = self.indep_cd1(u=0.5)
        assert isinstance(indep_u_05, CD1Wrapper)
        assert self.u not in indep_u_05.func.free_symbols
        assert self.v in indep_u_05.func.free_symbols

        # Should still be equal to v
        assert indep_u_05.func == self.v

        # Substitute only v
        indep_v_05 = self.indep_cd1(v=0.5)
        assert isinstance(indep_v_05, CD1Wrapper)
        assert self.v not in indep_v_05.func.free_symbols
        assert indep_v_05.evalf() == 0.5

    def test_operations(self):
        """Test basic operations."""
        # Addition
        sum_cd1 = self.indep_cd1 + self.indep_cd1
        assert isinstance(sum_cd1, SymPyFuncWrapper)
        assert sum_cd1.func == 2 * self.v

        # Subtraction
        diff_cd1 = self.indep_cd1 - CD1Wrapper(self.v * 0.5)
        assert isinstance(diff_cd1, SymPyFuncWrapper)
        assert diff_cd1.func == 0.5 * self.v

        # Multiplication
        prod_cd1 = self.indep_cd1 * 2
        assert isinstance(prod_cd1, SymPyFuncWrapper)
        assert prod_cd1.func == 2 * self.v

        # Division
        div_cd1 = self.indep_cd1 / 2
        assert isinstance(div_cd1, SymPyFuncWrapper)
        # Use simplify to compare expressions that are mathematically equivalent
        assert simplify(div_cd1.func - 0.5 * self.v) == 0

    def test_methods(self):
        """Test inherited methods from SymPyFuncWrapper."""
        # Test diff method
        deriv = self.indep_cd1.diff(self.v)
        assert isinstance(deriv, SymPyFuncWrapper)
        assert deriv.func == 1

        # Test to_latex method
        latex_repr = self.indep_cd1.to_latex()
        # More flexible check since LaTeX representation might vary
        assert "v" in latex_repr

        # Test subs method
        subbed = self.indep_cd1.subs(self.v, 0.5)
        assert isinstance(subbed, SymPyFuncWrapper)
        assert subbed.func == 0.5

    def test_special_cases(self):
        """Test special cases and edge conditions."""
        # Create a derivative with both u and v
        mixed_deriv = CD1Wrapper(self.u * self.v)

        # When v=0, result should be 0 regardless of u
        assert mixed_deriv(u=0.5, v=0) == SymPyFuncWrapper(sympy.S.Zero)
        assert mixed_deriv(u=1, v=0) == SymPyFuncWrapper(sympy.S.Zero)

        # When v=1, result should be 1 regardless of u
        assert mixed_deriv(u=0.5, v=1) == SymPyFuncWrapper(sympy.S.One)
        assert mixed_deriv(u=1, v=1) == SymPyFuncWrapper(sympy.S.One)

        # Test with u replaced first
        u_replaced = mixed_deriv(u=0.5)
        assert u_replaced.func == 0.5 * self.v
        # Now test boundaries
        assert u_replaced(v=0) == SymPyFuncWrapper(sympy.S.Zero)
        assert u_replaced(v=1) == SymPyFuncWrapper(sympy.S.One)

    def test_chain_calls(self):
        """Test chained calls to __call__."""
        mixed_deriv = CD1Wrapper(self.u**2 * self.v)

        # Chain substitutions
        result = mixed_deriv(u=0.5)(v=0.7)
        assert isinstance(result, CD1Wrapper)
        assert abs(float(result.evalf()) - 0.175) < 1e-10

        # Check that the boundary conditions still work after chaining
        result = mixed_deriv(u=0.5)(v=0)
        assert result == SymPyFuncWrapper(sympy.S.Zero)

        result = mixed_deriv(u=0.5)(v=1)
        assert result == SymPyFuncWrapper(sympy.S.One)
