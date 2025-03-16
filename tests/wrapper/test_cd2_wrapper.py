"""
Tests for the CD2Wrapper class (partial derivative with respect to second argument).
"""

import sympy
from sympy import simplify
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper
from copul.wrapper.cd2_wrapper import CD2Wrapper


class TestCD2Wrapper:
    """Tests for the CD2Wrapper class."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create symbolic variables
        self.u, self.v, self.theta = sympy.symbols("u v theta")

        # Create some test expressions representing partial derivatives of copulas
        # For independence copula C(u,v) = u*v, the derivative ∂C/∂v = u
        self.indep_deriv = self.u

        # For Clayton copula, partial derivative with respect to v
        # This is a complex expression, simplified for testing
        theta_val = 2
        self.clayton_deriv = self.v ** (-theta_val - 1) * (
            self.u ** (-theta_val) + self.v ** (-theta_val) - 1
        ) ** (-(1 / theta_val) - 1)

        # Create CD2Wrapper instances
        self.indep_cd2 = CD2Wrapper(self.indep_deriv)
        self.clayton_cd2 = CD2Wrapper(self.clayton_deriv)

    def test_initialization(self):
        """Test initialization of CD2Wrapper."""
        # Test initialization with a SymPy expression
        cd2 = CD2Wrapper(self.u)
        assert isinstance(cd2, CD2Wrapper)
        assert isinstance(cd2, SymPyFuncWrapper)
        assert str(cd2.func) == str(self.u)

        # Test initialization with another CD2Wrapper
        cd2_2 = CD2Wrapper(cd2)
        assert isinstance(cd2_2, CD2Wrapper)
        assert cd2_2.func == cd2.func

    def test_boundary_conditions(self):
        """Test boundary conditions for conditional distributions."""
        # When u=0, CD2 should be 0
        assert self.indep_cd2(u=0) == SymPyFuncWrapper(sympy.S.Zero)

        # When u=1, CD2 should be 1
        assert self.indep_cd2(u=1) == SymPyFuncWrapper(sympy.S.One)

        # Clayton should also follow these rules
        assert self.clayton_cd2(u=0) == SymPyFuncWrapper(sympy.S.Zero)
        assert self.clayton_cd2(u=1) == SymPyFuncWrapper(sympy.S.One)

    def test_substitution(self):
        """Test variable substitution."""
        # Test with independence copula derivative
        result = self.indep_cd2(u=0.5, v=0.7)
        assert isinstance(result, CD2Wrapper)
        assert result.evalf() == 0.5

        # Test with Clayton copula derivative
        clayton_at_point = self.clayton_cd2(u=0.5, v=0.7)
        assert isinstance(clayton_at_point, CD2Wrapper)

        # The exact value depends on the formula, we just check it's a number
        assert isinstance(float(clayton_at_point.evalf()), float)

    def test_partial_substitution(self):
        """Test partial substitution of variables."""
        # Substitute only v
        indep_v_05 = self.indep_cd2(v=0.5)
        assert isinstance(indep_v_05, CD2Wrapper)
        assert self.v not in indep_v_05.func.free_symbols
        assert self.u in indep_v_05.func.free_symbols

        # Should still be equal to u
        assert indep_v_05.func == self.u

        # Substitute only u
        indep_u_05 = self.indep_cd2(u=0.5)
        assert isinstance(indep_u_05, CD2Wrapper)
        assert self.u not in indep_u_05.func.free_symbols
        assert indep_u_05.evalf() == 0.5

    def test_operations(self):
        """Test basic operations."""
        # Addition
        sum_cd2 = self.indep_cd2 + self.indep_cd2
        assert isinstance(sum_cd2, SymPyFuncWrapper)
        assert sum_cd2.func == 2 * self.u

        # Subtraction
        diff_cd2 = self.indep_cd2 - CD2Wrapper(self.u * 0.5)
        assert isinstance(diff_cd2, SymPyFuncWrapper)
        assert diff_cd2.func == 0.5 * self.u

        # Multiplication
        prod_cd2 = self.indep_cd2 * 2
        assert isinstance(prod_cd2, SymPyFuncWrapper)
        assert prod_cd2.func == 2 * self.u

        # Division
        div_cd2 = self.indep_cd2 / 2
        assert isinstance(div_cd2, SymPyFuncWrapper)
        # Use simplify to compare expressions that are mathematically equivalent
        assert simplify(div_cd2.func - 0.5 * self.u) == 0

    def test_methods(self):
        """Test inherited methods from SymPyFuncWrapper."""
        # Test diff method
        deriv = self.indep_cd2.diff(self.u)
        assert isinstance(deriv, SymPyFuncWrapper)
        assert deriv.func == 1

        # Test to_latex method
        latex_repr = self.indep_cd2.to_latex()
        # More flexible check since LaTeX representation might vary
        assert "u" in latex_repr or "1" in latex_repr

        # Test subs method
        subbed = self.indep_cd2.subs(self.u, 0.5)
        assert isinstance(subbed, SymPyFuncWrapper)
        assert subbed.func == 0.5

    def test_chain_calls(self):
        """Test chained calls to __call__."""
        mixed_deriv = CD2Wrapper(self.u * self.v**2)

        # Chain substitutions
        result = mixed_deriv(v=0.5)(u=0.7)
        assert isinstance(result, CD2Wrapper)
        assert abs(float(result.evalf()) - 0.175) < 1e-10

        # Check that the boundary conditions still work after chaining
        result = mixed_deriv(v=0.5)(u=0)
        assert result == SymPyFuncWrapper(sympy.S.Zero)

        result = mixed_deriv(v=0.5)(u=1)
        assert result == SymPyFuncWrapper(sympy.S.One)

    def test_different_variable_names(self):
        """Test behavior with different variable names."""
        # Create symbolic variables with different names
        u1, u2 = sympy.symbols("u1 u2")

        # Create a derivative with different variable names
        alt_deriv = CD2Wrapper(u1 * u2)

        # For CD2Wrapper with non-standard names, the special conditions
        # should only apply when u and v are part of the free symbols
        # So regular substitution behavior is expected
        result = alt_deriv(u1=0.5, u2=0.7)
        assert isinstance(result, CD2Wrapper)
        assert abs(float(result.evalf()) - 0.35) < 1e-10
