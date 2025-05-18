import pytest
import sympy as sp
from unittest.mock import patch

from copul.families.core.biv_copula import BivCopula


@pytest.fixture(scope="function")
def copula_fam():
    """Create a simple BivCopula instance for testing"""

    class SimpleBivCopula(BivCopula):
        # Define a simple copula with a parameter theta
        theta = sp.symbols("theta", positive=True)
        params = [theta]
        intervals = {str(theta): sp.Interval(0, float("inf"))}

        @property
        def _cdf_expr(self):
            # Simple product copula with a parameter influence
            expr = self.u * self.v * (1 + self.theta * (1 - self.u) * (1 - self.v))
            return expr

    return SimpleBivCopula


def test_init(copula_fam):
    """Test initialization of BivCopula"""
    copula = copula_fam(0.5)
    assert copula.dim == 2
    assert copula.theta == 0.5
    assert len(copula.params) == 0  # params used in init are removed from list
    assert copula.intervals == {}  # intervals for used params are removed


def test_segregate_symbols():
    """Test the _segregate_symbols static method"""
    # Create a simple expression with function variable and parameter
    t, a = sp.symbols("t a", positive=True)
    expr = a * t**2 + t

    # Test with explicit parameter
    func_vars, params = BivCopula._segregate_symbols(expr, params=[a])
    assert func_vars == [t]
    assert params == [a]

    # Test with function variable name
    func_vars, params = BivCopula._segregate_symbols(expr, func_var_name="t")
    assert func_vars == [t]
    assert params == [a]

    # Test with no guidance
    # Note: The actual behavior is to use the first symbol in expr.free_symbols as the function variable
    # Get the actual symbols as they appear in free_symbols (order depends on SymPy internals)
    all_symbols = list(expr.free_symbols)
    func_vars, params = BivCopula._segregate_symbols(expr)

    # Verify that we get the first symbol as function variable and the rest as parameters
    assert func_vars == [all_symbols[0]]
    assert set(params) == set(all_symbols[1:])  # Use a set to ignore order


def test_from_string():
    """Test the _from_string class method"""
    # Create with string parameters
    biv_copula = BivCopula._from_string(params=["alpha", "beta"])

    # Check if parameters were properly set
    assert len(biv_copula.params) == 2
    assert str(biv_copula.params[0]) == "alpha"
    assert str(biv_copula.params[1]) == "beta"
    assert hasattr(biv_copula, "alpha")
    assert hasattr(biv_copula, "beta")


def test_rank_correlations(copula_fam):
    """Test rank correlation calculations"""
    copula = copula_fam(0.5)
    # These methods can be complex to test directly, so just mock them

    # Mock the _tau method to return a simple value
    with patch.object(BivCopula, "_tau", return_value=0.3):
        tau = copula.tau()
        assert tau == 0.3

    # Mock the _rho method
    with patch.object(BivCopula, "_rho", return_value=0.5):
        rho = copula.rho()
        assert rho == 0.5


@patch("copul.families.tp2_verifier.TP2Verifier.is_tp2")
def test_is_tp2(mock_is_tp2, copula_fam):
    """Test TP2 property verification"""
    copula = copula_fam(0.5)
    mock_is_tp2.return_value = True

    assert copula.is_tp2() is True
    mock_is_tp2.assert_called_once_with(copula)


@patch("copul.families.core.biv_core_copula.CISVerifier.is_cis")
def test_is_cis(mock_is_cis, copula_fam):
    """Test CIS property verification"""
    copula = copula_fam(0.5)
    mock_is_cis.return_value = True

    assert copula.is_cis() is True
    mock_is_cis.assert_called_once_with(copula)
