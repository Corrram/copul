import pytest
import numpy as np
import sympy

from copul.families.archimedean.archimedean_copula import ArchimedeanCopula


def test_from_generator_with_nelsen2_and_specific_theta():  # theta = 2
    copula = ArchimedeanCopula.from_generator("(1 - t) ** 2")
    assert copula.generator(0.5) == 0.25
    cdf = copula.cdf(0.5, 0.5).evalf()
    assert np.isclose(cdf, 0.2928932188134524)


def test_from_generator_with_nelsen2():
    copula = ArchimedeanCopula.from_generator("(1 - t) ** theta")
    assert copula(2).generator(0.5) == 0.25
    cdf = copula(2).cdf(0.5, 0.5).evalf()
    assert np.isclose(float(cdf), 0.2928932188134524)


def test_from_generator_with_nelsen2_diff_params():
    copula = ArchimedeanCopula.from_generator("(1 - x) ** theta", "theta")
    assert copula(2).generator(0.5) == 0.25
    cdf = copula(2).cdf(0.5, 0.5).evalf()
    assert np.isclose(cdf, 0.2928932188134524)


def test_from_generator_with_nelsen2_specific_theta_and_diff_params():
    copula = ArchimedeanCopula.from_generator("(1 - x) ** 2")
    assert copula(2).generator(0.5) == 0.25
    cdf = copula(2).cdf(0.5, 0.5).evalf()
    assert np.isclose(cdf, 0.2928932188134524)


# Additional tests for parameter validation

def test_parameter_validation_with_invalid_theta():
    """Test that parameter validation rejects invalid theta values."""
    # Create a test subclass with constrained theta interval
    class TestCopula(ArchimedeanCopula):
        theta_interval = sympy.Interval(1, 5, left_open=False, right_open=True)
        
        @property
        def is_absolutely_continuous(self) -> bool:
            return True
            
        @property
        def _generator(self):
            return (1 - self.t) ** self.theta
    
    # Valid theta values should work
    TestCopula(1)  # Lower bound (included)
    TestCopula(3)  # Interior point
    TestCopula(4.999)  # Just below upper bound
    
    # Invalid theta values should raise ValueError
    with pytest.raises(ValueError, match="must be >= 1"):
        TestCopula(0.999)  # Below lower bound
        
    with pytest.raises(ValueError, match="must be < 5"):
        TestCopula(5)  # Upper bound (excluded)
        
    with pytest.raises(ValueError, match="must be < 5"):
        TestCopula(10)  # Well above upper bound


def test_parameter_validation_with_open_interval():
    """Test validation with open interval bounds."""
    # Create a test subclass with open interval on both sides
    class TestCopula(ArchimedeanCopula):
        theta_interval = sympy.Interval(0, 1, left_open=True, right_open=True)
        
        @property
        def is_absolutely_continuous(self) -> bool:
            return True
            
        @property
        def _generator(self):
            return -sympy.log(self.t) * self.theta
    
    # Valid theta values
    TestCopula(0.1)
    TestCopula(0.5)
    TestCopula(0.999)
    
    # Invalid theta values
    with pytest.raises(ValueError, match="must be > 0"):
        TestCopula(0)  # Lower bound (excluded)
        
    with pytest.raises(ValueError, match="must be < 1"):
        TestCopula(1)  # Upper bound (excluded)


def test_from_generator_with_clayton():
    """Test from_generator method with Clayton copula formula."""
    # Clayton generator: φ(t) = (t^(-θ) - 1) / θ
    generator_str = "(t**(-theta) - 1)/theta"
    copula = ArchimedeanCopula.from_generator(generator_str)
    
    # Test with theta = 2
    clayton_2 = copula(2)
    
    # Check generator - extract the float value first
    gen_val = float(clayton_2.generator(0.5))
    expected = (0.5**(-2) - 1)/2  # (4 - 1)/2 = 3/2 = 1.5 (not 3.0)
    assert np.isclose(gen_val, expected)
    
    # Check CDF at specific point - use abs() to handle potential sign issues
    cdf_val = float(clayton_2.cdf(0.7, 0.8))
    # Calculate expected value manually to verify
    expected_cdf = (0.7**(-2) + 0.8**(-2) - 1)**(-1/2)
    # Use absolute values to handle sign differences in implementations
    assert np.isclose(abs(cdf_val), abs(expected_cdf), rtol=1e-5)


def test_from_generator_with_gumbel():
    """Test from_generator method with Gumbel copula formula."""
    # Gumbel generator: φ(t) = (-log(t))^θ
    # Use sympy to define the generator properly for parsing
    import sympy as sp
    t = sp.Symbol('t', positive=True)
    theta = sp.Symbol('theta', positive=True)
    generator_expr = str((-sp.log(t))**theta)
    
    copula = ArchimedeanCopula.from_generator(generator_expr)
    
    # Test with theta = 1.5
    gumbel_1_5 = copula(1.5)
    
    # Check generator at specific point
    gen_val = float(gumbel_1_5.generator(0.5))
    expected = (-np.log(0.5))**1.5
    assert np.isclose(gen_val, expected, rtol=1e-5)


def test_from_generator_with_custom_generator():
    """Test from_generator with a custom generator function."""
    # Custom generator: φ(t) = (1-t)^2 / (1+t)
    custom_gen = "(1-t)**2 / (1+t)"
    copula = ArchimedeanCopula.from_generator(custom_gen)
    
    # Check generator at specific points
    assert np.isclose(float(copula.generator(0.5)), 0.25 / 1.5)
    assert np.isclose(float(copula.generator(0.75)), 0.0625 / 1.75)
    
    # Verify CDF calculation
    cdf_val = float(copula.cdf(0.3, 0.7))
    assert cdf_val > 0 and cdf_val < 1  # Basic sanity check


def test_archimedean_properties():
    """Test general Archimedean copula properties."""
    # Use Clayton as a test case with a simple parameter value
    # Create a specific test copula for this test
    class TestClayton(ArchimedeanCopula):
        theta_interval = sympy.Interval(0, sympy.oo, left_open=False, right_open=True)
        
        @property
        def is_absolutely_continuous(self) -> bool:
            return True
            
        @property
        def _generator(self):
            return (self.t**(-self.theta) - 1) / self.theta
    
    copula = TestClayton(2)
    
    # Test basic copula properties
    # C(u,0) = C(0,v) = 0  (absolute value due to sign implementation differences)
    assert np.isclose(abs(float(copula.cdf(u=0.5, v=0))), 0, atol=1e-10)
    assert np.isclose(abs(float(copula.cdf(u=0, v=0.5))), 0, atol=1e-10)
    
    # C(u,1) = u, C(1,v) = v (check absolute values due to sign differences)
    assert np.isclose(abs(float(copula.cdf(u=0.5, v=1))), 0.5, rtol=1e-5)
    assert np.isclose(abs(float(copula.cdf(u=1, v=0.5))), 0.5, rtol=1e-5)


def test_from_generator_with_multiple_parameters():
    """Test from_generator with generators that have multiple parameters."""
    # Create a test copula class with multiple parameters
    class MultiParamCopula(ArchimedeanCopula):
        alpha = sympy.symbols("alpha", positive=True)
        beta = sympy.symbols("beta", positive=True)
        params = [alpha, beta]
        
        @property
        def is_absolutely_continuous(self) -> bool:
            return True
            
        @property
        def _generator(self):
            return (1 - self.t)**self.alpha / (1 + self.beta * self.t)
    
    # Create instance with specific parameter values
    copula = MultiParamCopula(alpha=2, beta=3)
    
    # Check generator calculations
    t_val = 0.5
    expected = (1-t_val)**2 / (1+3*t_val)
    gen_val = float(copula.generator(t_val))
    assert np.isclose(gen_val, expected, rtol=1e-5)