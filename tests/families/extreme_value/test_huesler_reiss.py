import pytest
import sympy as sp

from copul.families.extreme_value.huesler_reiss import HueslerReiss
from copul.families.other.independence_copula import IndependenceCopula


@pytest.fixture
def huesler_reiss_copula():
    """Create a HueslerReiss copula with delta=1.0 for testing"""
    return HueslerReiss(1.0)


@pytest.fixture
def huesler_reiss_symbolic():
    """Create a HueslerReiss copula with symbolic delta for testing"""
    return HueslerReiss()


def test_hr_init():
    """Test initialization of HueslerReiss copula"""
    # Default initialization with symbol
    copula = HueslerReiss()
    assert hasattr(copula, "delta")
    assert isinstance(copula.delta, sp.Symbol)
    assert str(copula.delta) == "delta"

    # Initialization with parameter
    copula = HueslerReiss(2.0)
    assert hasattr(copula, "delta")
    assert copula.delta == 2.0


def test_hr_parameter_bounds():
    """Test parameter bounds of HueslerReiss copula"""
    copula = HueslerReiss()
    # Delta should be ≥ 0
    assert copula.intervals["delta"].left == 0
    assert copula.intervals["delta"].right == float("inf")
    assert not copula.intervals["delta"].left_open  # Left bound is closed
    assert copula.intervals["delta"].right_open  # Right bound is open


def test_hr_independence_special_case():
    """Test special case when delta=0 (should return Independence copula)"""
    # When delta=0, HueslerReiss becomes the Independence copula
    copula = HueslerReiss(delta=0)

    # Should return an Independence copula instance
    assert isinstance(copula, IndependenceCopula)

    # Test with positional argument too
    copula2 = HueslerReiss(0)
    assert isinstance(copula2, IndependenceCopula)


def test_hr_is_symmetric(huesler_reiss_copula):
    """Test symmetry property of HueslerReiss copula"""
    assert huesler_reiss_copula.is_symmetric is True


def test_hr_is_absolutely_continuous(huesler_reiss_copula):
    """Test absolute continuity property"""
    assert huesler_reiss_copula.is_absolutely_continuous is True


def test_hr_z_function(huesler_reiss_copula):
    """Test the _z helper function"""
    # Test at boundaries
    assert huesler_reiss_copula._z(0) == 0
    assert huesler_reiss_copula._z(1) == 1

    # Test at t=0.5 with delta=1.0
    # z(0.5) = 1/1 + 1/2 * ln(0.5/(1-0.5)) = 1 + 0 = 1
    result = huesler_reiss_copula._z(0.5)
    # Since ln(1) = 0, we expect z(0.5) = 1/delta
    assert result == 1.0

    # Test with different delta value
    copula = HueslerReiss(2.0)
    # z(0.5) = 1/2 + 2/2 * ln(0.5/(1-0.5)) = 0.5 + 0 = 0.5
    result = copula._z(0.5)
    assert result == 0.5


def test_hr_pickands_symbolic(huesler_reiss_symbolic):
    """Test Pickands dependence function structure with symbolic delta"""
    # The _pickands property should exist
    assert hasattr(huesler_reiss_symbolic, "_pickands")

    # Test that it's properly constructed with sympy expressions
    pickands_expr = huesler_reiss_symbolic._pickands
    assert isinstance(pickands_expr, sp.Expr)

    # For symbolic delta, the expression should include both t and delta symbols
    t = huesler_reiss_symbolic.t
    assert t in pickands_expr.free_symbols
    assert huesler_reiss_symbolic.delta in pickands_expr.free_symbols


def test_hr_pickands_concrete(huesler_reiss_copula):
    """Test Pickands dependence function structure with concrete delta value"""
    # The _pickands property should exist
    assert hasattr(huesler_reiss_copula, "_pickands")

    # Test that it's properly constructed with sympy expressions
    pickands_expr = huesler_reiss_copula._pickands
    assert isinstance(pickands_expr, sp.Expr)

    # For concrete delta, the expression should include t but not delta (it's substituted)
    t = huesler_reiss_copula.t
    assert t in pickands_expr.free_symbols
    # Delta should not be in free_symbols because it's a concrete value (1.0)
    assert len(pickands_expr.free_symbols) > 0


def test_hr_call_method():
    """Test __call__ method for creating new instances"""
    # Create base copula
    copula = HueslerReiss()

    # Update parameter using kwargs
    new_copula = copula(delta=2.0)

    # Original should be unchanged
    assert isinstance(copula.delta, sp.Symbol)

    # New instance should have updated parameter
    assert new_copula.delta == 2.0

    # Test with positional arg
    new_copula2 = copula(3.0)
    assert new_copula2.delta == 3.0

    # Test independence special case
    ind_copula = copula(0)
    assert isinstance(ind_copula, IndependenceCopula)


def test_hr_edge_cases():
    """Test edge cases for parameter values"""
    # Very small delta (close to independence)
    copula_small = HueslerReiss(1e-10)
    assert copula_small.delta == 1e-10

    # Very large delta
    copula_large = HueslerReiss(1e10)
    assert copula_large.delta == 1e10


def test_hr_properties_inheritance():
    """Test that the copula inherits properties from ExtremeValueCopula"""
    copula = HueslerReiss(1.0)

    # Should have kendalls_tau and spearmans_rho methods
    assert hasattr(copula, "kendalls_tau")
    assert hasattr(copula, "spearmans_rho")
    assert callable(copula.kendalls_tau)
    assert callable(copula.spearmans_rho)

    # Should have CDF method
    assert hasattr(copula, "cdf")
