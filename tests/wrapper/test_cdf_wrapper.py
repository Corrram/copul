"""
Tests for the CDFWrapper class.
"""

import sympy
from sympy import simplify
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper
from copul.wrapper.cdf_wrapper import CDFWrapper


class TestCDFWrapper:
    """Tests for the CDFWrapper class."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create symbolic variables
        self.u, self.v, self.theta = sympy.symbols("u v theta")
        self.u1, self.u2, self.alpha = sympy.symbols("u1 u2 alpha")

        # Create some common copula CDF expressions
        # Product copula (independence): C(u,v) = u*v
        self.product_copula = self.u * self.v

        # Clayton copula: C(u,v) = (u^(-theta) + v^(-theta) - 1)^(-1/theta)
        self.clayton_copula = (
            self.u ** (-self.theta) + self.v ** (-self.theta) - 1
        ) ** (-1 / self.theta)

        # Frank copula: C(u,v) = -1/theta * log(1 + (exp(-theta*u)-1)*(exp(-theta*v)-1)/(exp(-theta)-1))
        self.frank_copula = (-1 / self.theta) * sympy.log(
            1
            + (sympy.exp(-self.theta * self.u) - 1)
            * (sympy.exp(-self.theta * self.v) - 1)
            / (sympy.exp(-self.theta) - 1)
        )

        # Alternative variable names
        self.alt_product_copula = self.u1 * self.u2

        # Create CDFWrapper instances
        self.product_cdf = CDFWrapper(self.product_copula)
        self.clayton_cdf = CDFWrapper(self.clayton_copula)
        self.frank_cdf = CDFWrapper(self.frank_copula)
        self.alt_product_cdf = CDFWrapper(self.alt_product_copula)

    def test_initialization(self):
        """Test initialization of CDFWrapper."""
        # Test initialization with a SymPy expression
        cdf = CDFWrapper(self.u * self.v)
        assert isinstance(cdf, CDFWrapper)
        assert isinstance(cdf, SymPyFuncWrapper)
        assert str(cdf.func) == str(self.u * self.v)

        # Test initialization with another CDFWrapper
        cdf2 = CDFWrapper(cdf)
        assert isinstance(cdf2, CDFWrapper)
        assert cdf2.func == cdf.func

    def test_boundary_conditions_uv(self):
        """Test boundary conditions for CDFs with u,v variables."""
        # When u=0 or v=0, copula CDF should be 0
        assert self.product_cdf(u=0) == SymPyFuncWrapper(sympy.S.Zero)
        assert self.product_cdf(v=0) == SymPyFuncWrapper(sympy.S.Zero)

        # When u=1, copula CDF should be v
        u1_result = self.product_cdf(u=1)
        assert u1_result.func == self.v

        # When v=1, copula CDF should be u
        v1_result = self.product_cdf(v=1)
        assert v1_result.func == self.u

    def test_boundary_conditions_u1u2(self):
        """Test boundary conditions for CDFs with u1,u2 variables."""
        # When u1=0 or u2=0, copula CDF should be 0
        assert self.alt_product_cdf(u1=0) == SymPyFuncWrapper(sympy.S.Zero)
        assert self.alt_product_cdf(u2=0) == SymPyFuncWrapper(sympy.S.Zero)

    def test_substitution(self):
        """Test variable substitution."""
        # Test with product copula
        result = self.product_cdf(u=0.5, v=0.7)
        assert isinstance(result, SymPyFuncWrapper)
        assert result.evalf() == 0.35

        # Test with Clayton copula
        clayton_with_theta = self.clayton_cdf(theta=2)
        assert isinstance(clayton_with_theta, CDFWrapper)

        # Evaluate at specific points
        clayton_at_point = clayton_with_theta(u=0.5, v=0.7)
        # The exact value depends on the formula, but we can check it's a CDFWrapper
        assert isinstance(clayton_at_point, SymPyFuncWrapper)

        # For Clayton with theta=2, u=0.5, v=0.7, the correct value is approximately 0.4454
        assert abs(float(clayton_at_point.evalf()) - 0.4454) < 1e-4

    def test_partial_substitution(self):
        """Test partial substitution of variables."""
        # Substitute only theta
        clayton_theta_2 = self.clayton_cdf(theta=2)
        assert isinstance(clayton_theta_2, CDFWrapper)
        assert self.u in clayton_theta_2.func.free_symbols
        assert self.v in clayton_theta_2.func.free_symbols
        assert self.theta not in clayton_theta_2.func.free_symbols

        # Substitute only u
        clayton_u_05 = self.clayton_cdf(u=0.5)
        assert isinstance(clayton_u_05, CDFWrapper)
        assert self.u not in clayton_u_05.func.free_symbols
        assert self.v in clayton_u_05.func.free_symbols
        assert self.theta in clayton_u_05.func.free_symbols

    def test_operations(self):
        """Test basic operations."""
        # Addition
        sum_cdf = self.product_cdf + self.product_cdf
        assert isinstance(sum_cdf, SymPyFuncWrapper)
        assert sum_cdf.func == 2 * self.u * self.v

        # Subtraction
        diff_cdf = self.product_cdf - CDFWrapper(self.u * self.v * 0.5)
        assert isinstance(diff_cdf, SymPyFuncWrapper)
        assert diff_cdf.func == 0.5 * self.u * self.v

        # Multiplication
        prod_cdf = self.product_cdf * 2
        assert isinstance(prod_cdf, SymPyFuncWrapper)
        assert prod_cdf.func == 2 * self.u * self.v

        # Division
        div_cdf = self.product_cdf / 2
        assert isinstance(div_cdf, SymPyFuncWrapper)
        # Use simplify to compare expressions that are mathematically equivalent
        assert simplify(div_cdf.func - 0.5 * self.u * self.v) == 0

        # Power
        pow_cdf = self.product_cdf**2
        assert isinstance(pow_cdf, SymPyFuncWrapper)
        assert pow_cdf.func == (self.u * self.v) ** 2

    def test_methods(self):
        """Test inherited methods from SymPyFuncWrapper."""
        # Test diff method
        deriv = self.product_cdf.diff(self.u)
        assert isinstance(deriv, SymPyFuncWrapper)
        assert deriv.func == self.v

        # Test to_latex method
        latex_repr = self.product_cdf.to_latex()
        # More flexible check that just verifies variables appear in the output
        assert "v" in latex_repr

        # Test subs method
        subbed = self.product_cdf.subs(self.u, 0.5)
        assert isinstance(subbed, SymPyFuncWrapper)
        assert subbed.func == 0.5 * self.v

    def test_special_cases(self):
        """Test special cases and edge conditions."""
        # Both u=0 and v=0
        assert self.product_cdf(u=0, v=0) == SymPyFuncWrapper(sympy.S.Zero)

        # Both u=1 and v=1
        result = self.product_cdf(u=1, v=1)
        # Handle potential float conversion issues
        assert float(result) == 1.0

        # Mixed boundary conditions
        assert self.product_cdf(u=0, v=1) == SymPyFuncWrapper(sympy.S.Zero)
        assert self.product_cdf(u=1, v=0) == SymPyFuncWrapper(sympy.S.Zero)

    def test_equality(self):
        """Test equality comparison."""
        # Same expression should be equal
        cdf1 = CDFWrapper(self.u * self.v)
        cdf2 = CDFWrapper(self.u * self.v)
        assert cdf1 == cdf2

        # Different expressions should not be equal
        cdf3 = CDFWrapper(self.u + self.v)
        assert cdf1 != cdf3

        # Compare with raw SymPy expression
        assert cdf1 == self.u * self.v

    def test_isclose(self):
        """Test isclose method."""
        # Create expressions that evaluate to the same value
        cdf1 = CDFWrapper(self.u * self.v)
        cdf2 = CDFWrapper(self.u * self.v * 1.0)

        # Substitute values
        cdf1_eval = cdf1(u=0.5, v=0.5)
        cdf2_eval = cdf2(u=0.5, v=0.5)

        # Should be close
        assert cdf1_eval.isclose(cdf2_eval)
        assert cdf1_eval.isclose(0.25)

        # Should not be close
        assert not cdf1_eval.isclose(0.3)
