import pytest
import sympy as sp
from unittest.mock import patch

from copul.families.extreme_value import GumbelHougaardEV as GumbelHougaard, Tawn
from copul.families.extreme_value.marshall_olkin import MarshallOlkin
from copul.families.other.biv_independence_copula import BivIndependenceCopula
from copul.wrapper.cdf_wrapper import CDFWrapper


@pytest.fixture
def tawn_copula():
    """Create a Tawn copula with non-trivial parameters for testing"""
    return Tawn(0.5, 0.7, 2.0)


@pytest.fixture
def symmetric_tawn_copula():
    """Create a symmetric Tawn copula for testing"""
    return Tawn(0.6, 0.6, 2.0)


def test_tawn_init():
    """Test initialization of Tawn copula"""
    # Default initialization with symbols
    copula = Tawn()
    assert hasattr(copula, "alpha_1")
    assert hasattr(copula, "alpha_2")
    assert hasattr(copula, "theta")
    assert isinstance(copula.alpha_1, sp.Symbol)
    assert isinstance(copula.alpha_2, sp.Symbol)
    assert isinstance(copula.theta, sp.Symbol)

    # Initialization with parameters
    copula = Tawn(0.3, 0.7, 1.5)
    assert copula.alpha_1 == 0.3
    assert copula.alpha_2 == 0.7
    assert copula.theta == 1.5


def test_tawn_parameter_bounds():
    """Test parameter bounds of Tawn copula"""
    copula = Tawn()

    # alpha_1 and alpha_2 should be in [0, 1]
    assert copula.intervals["alpha_1"].left == 0
    assert copula.intervals["alpha_1"].right == 1
    assert not copula.intervals["alpha_1"].left_open  # Left bound is closed
    assert not copula.intervals["alpha_1"].right_open  # Right bound is closed

    assert copula.intervals["alpha_2"].left == 0
    assert copula.intervals["alpha_2"].right == 1
    assert not copula.intervals["alpha_2"].left_open
    assert not copula.intervals["alpha_2"].right_open

    # theta should be in [1, inf)
    assert copula.intervals["theta"].left == 1
    assert copula.intervals["theta"].right == float("inf")
    assert not copula.intervals["theta"].left_open
    assert copula.intervals["theta"].right_open


def test_tawn_special_case_gumbel():
    """Test special case when alpha_1 = alpha_2 = 1 (should return GumbelHougaard)"""
    # Test with initialization
    copula = Tawn(1, 1, 2.0)
    assert isinstance(copula, GumbelHougaard)
    assert copula.theta == 2.0

    # Test with __call__ method
    base = Tawn()
    copula = base(alpha_1=1, alpha_2=1, theta=2.0)
    assert isinstance(copula, GumbelHougaard)
    assert copula.theta == 2.0

    # Test partial case with alpha_1=1
    copula = Tawn(1, 0.5, 2.0)
    base = Tawn()
    result = base(alpha_1=1, alpha_2=0.5, theta=2.0)
    assert result.alpha_1 == 1
    assert result.alpha_2 == 0.5
    assert result.theta == 2.0

    # Test partial case with alpha_2=1
    copula = Tawn(0.5, 1, 2.0)
    base = Tawn()
    result = base(alpha_1=0.5, alpha_2=1, theta=2.0)
    assert result.alpha_1 == 0.5
    assert result.alpha_2 == 1
    assert result.theta == 2.0


def test_tawn_special_case_independence():
    """Test special case when theta=1 (should return IndependenceCopula)"""
    # Test with __call__ method
    base = Tawn()
    copula = base(alpha_1=0.5, alpha_2=0.7, theta=1)
    assert isinstance(copula, BivIndependenceCopula)


def test_tawn_special_case_marshall_olkin():
    """Test special case when theta=infinity (should return MarshallOlkin)"""
    # Test with __call__ method using sympy.oo
    base = Tawn()
    copula = base(alpha_1=0.3, alpha_2=0.7, theta=sp.oo)
    assert isinstance(copula, MarshallOlkin)
    assert copula.alpha_1 == 0.3
    assert copula.alpha_2 == 0.7


def test_tawn_is_symmetric(symmetric_tawn_copula, tawn_copula):
    """Test symmetry property of Tawn copula"""
    # Symmetric case
    assert symmetric_tawn_copula.is_symmetric is True

    # Non-symmetric case
    assert tawn_copula.is_symmetric is False


def test_tawn_is_absolutely_continuous(tawn_copula):
    """Test absolute continuity property"""
    assert tawn_copula.is_absolutely_continuous is True


def test_tawn_pickands(tawn_copula):
    """Test Pickands dependence function"""
    # For Tawn with alpha_1=0.5, alpha_2=0.7, theta=2.0, at t=0.5
    result = tawn_copula._pickands.subs(tawn_copula.t, 0.5)

    # Calculate expected value manually:
    # A(t) = (1-alpha_1)*(1-t) + (1-alpha_2)*t + ((alpha_1*(1-t))^theta + (alpha_2*t)^theta)^(1/theta)
    # A(0.5) = (1-0.5)*(1-0.5) + (1-0.7)*0.5 + ((0.5*0.5)^2 + (0.7*0.5)^2)^(1/2)
    # = 0.25 + 0.15 + (0.0625 + 0.1225)^0.5
    # = 0.4 + 0.4135 = 0.8135
    expected = 0.25 + 0.15 + (0.0625 + 0.1225) ** (1 / 2)
    assert abs(float(result) - expected) < 1e-10

    # Test at boundaries
    t0 = float(tawn_copula._pickands.subs(tawn_copula.t, 0))
    t1 = float(tawn_copula._pickands.subs(tawn_copula.t, 1))

    # Calculate expected values:
    # A(0) = (1-0.5)*(1-0) + (1-0.7)*0 + ((0.5*1)^2 + (0.7*0)^2)^(1/2)
    # = 0.5 + 0 + 0.5 = 1.0
    # A(1) = (1-0.5)*(1-1) + (1-0.7)*1 + ((0.5*0)^2 + (0.7*1)^2)^(1/2)
    # = 0 + 0.3 + 0.7 = 1.0
    assert abs(t0 - 1.0) < 1e-10
    assert abs(t1 - 1.0) < 1e-10


def test_tawn_cdf():
    """Test CDF computation"""
    copula = Tawn(0.3, 0.7, 1.5)

    # Mock the CDFWrapper to avoid actual computation
    with patch.object(CDFWrapper, "__call__") as mock_call:
        mock_call.return_value = 0.42  # Mock return value

        # Call CDF
        result = copula.cdf(0.5, 0.6)

        # Verify it was called with correct params
        mock_call.assert_called_once_with(0.5, 0.6)
        assert result == 0.42


def test_tawn_call_method():
    """Test __call__ method for creating new instances"""
    # Create base copula
    copula = Tawn()

    # Update parameters using kwargs
    new_copula = copula(alpha_1=0.2, alpha_2=0.8, theta=2.5)

    # Original should be unchanged
    assert isinstance(copula.alpha_1, sp.Symbol)
    assert isinstance(copula.alpha_2, sp.Symbol)
    assert isinstance(copula.theta, sp.Symbol)

    # New instance should have updated parameters
    assert new_copula.alpha_1 == 0.2
    assert new_copula.alpha_2 == 0.8
    assert new_copula.theta == 2.5

    # Test with positional args
    new_copula2 = copula(0.3, 0.7, 1.5)
    assert new_copula2.alpha_1 == 0.3
    assert new_copula2.alpha_2 == 0.7
    assert new_copula2.theta == 1.5


def test_tawn_call_with_invalid_args():
    """Test __call__ method with incorrect number of arguments"""
    copula = Tawn()

    # Should raise ValueError when not providing all three parameters
    with pytest.raises(ValueError):
        copula(0.5, 0.7)  # Only two parameters

    with pytest.raises(ValueError):
        copula(0.5)  # Only one parameter


def test_tawn_correlation_properties():
    """Test correlation properties are inherited from ExtremeValueCopula"""
    # Create a simple instance for testing
    copula = Tawn(0.5, 0.7, 2.0)

    # Check that it has the expected correlation methods
    assert hasattr(copula, "tau")
    assert hasattr(copula, "rho")
    assert callable(copula.tau)
    assert callable(copula.rho)


def test_tawn_edge_cases():
    """Test edge cases for parameter values"""
    # alpha_1 = alpha_2 = 0
    copula = Tawn(0, 0, 2.0)
    # This is still a valid Tawn copula
    assert copula.alpha_1 == 0
    assert copula.alpha_2 == 0
    assert copula.theta == 2.0

    # theta = 1 (just above)
    copula = Tawn(0.5, 0.7, 1.001)
    assert copula.theta == 1.001

    # Check that parameters are properly passed to special case copulas
    base = Tawn()
    # When theta → ∞, should become Marshall-Olkin with correct alphas
    mo_copula = base(alpha_1=0.4, alpha_2=0.6, theta=sp.oo)
    assert isinstance(mo_copula, MarshallOlkin)
    assert mo_copula.alpha_1 == 0.4
    assert mo_copula.alpha_2 == 0.6
