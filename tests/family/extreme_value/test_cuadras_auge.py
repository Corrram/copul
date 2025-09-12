import numpy as np
import pytest
import sympy as sp
from unittest.mock import patch

from copul.family.extreme_value import CuadrasAuge
from copul.family.other.biv_independence_copula import BivIndependenceCopula
from copul.family.other.upper_frechet import UpperFrechet
from copul.exceptions import PropertyUnavailableException
from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.wrapper.cd1_wrapper import CD1Wrapper


@pytest.fixture
def cuadras_auge_copula():
    """Create a CuadrasAuge copula with delta=0.5 for testing"""
    return CuadrasAuge(0.5)


def test_cuadras_auge():
    cop = CuadrasAuge(0.5)
    xi = cop.chatterjees_xi()
    assert np.isclose(xi, 1 / 6)


def test_ca_init():
    """Test initialization of CuadrasAuge copula"""
    # Default initialization with symbol
    copula = CuadrasAuge()
    assert hasattr(copula, "delta")
    assert isinstance(copula.delta, sp.Symbol)
    assert str(copula.delta) == "delta"

    # Initialization with parameter
    copula = CuadrasAuge(0.3)
    assert hasattr(copula, "delta")
    assert copula.delta == 0.3


def test_ca_parameter_bounds():
    """Test parameter bounds of CuadrasAuge copula"""
    copula = CuadrasAuge()
    # Delta should be in [0, 1]
    assert copula.intervals["delta"].left == 0
    assert copula.intervals["delta"].right == 1
    assert not copula.intervals["delta"].left_open  # Left bound is closed
    assert not copula.intervals["delta"].right_open  # Right bound is closed


def test_ca_independence_special_case():
    """Test special case when delta=0 (should return Independence copula)"""
    # When delta=0, CuadrasAuge becomes the Independence copula
    # Test with positional argument
    copula1 = CuadrasAuge(0)
    assert isinstance(copula1, BivIndependenceCopula)

    # Test with keyword argument
    copula2 = CuadrasAuge(delta=0)
    assert isinstance(copula2, BivIndependenceCopula)


def test_ca_upper_frechet_special_case():
    """Test special case when delta=1 (should return UpperFrechet copula)"""
    # When delta=1, CuadrasAuge becomes the UpperFrechet copula
    # Test with positional argument
    copula1 = CuadrasAuge(1)
    assert isinstance(copula1, UpperFrechet)

    # Test with keyword argument
    copula2 = CuadrasAuge(delta=1)
    assert isinstance(copula2, UpperFrechet)


def test_ca_is_symmetric(cuadras_auge_copula):
    """Test symmetry property of CuadrasAuge copula"""
    assert cuadras_auge_copula.is_symmetric is True


def test_ca_is_absolutely_continuous():
    """Test absolute continuity property"""
    # When delta=0, it should be absolutely continuous
    copula1 = CuadrasAuge(
        0
    )  # Use near-zero instead of zero to get a CuadrasAuge instance
    assert copula1.is_absolutely_continuous is True

    copula1 = CuadrasAuge(
        0.01
    )  # Use near-zero instead of zero to get a CuadrasAuge instance
    assert copula1.is_absolutely_continuous is False

    # When delta>0, it should not be absolutely continuous
    copula2 = CuadrasAuge(0.5)
    assert copula2.is_absolutely_continuous is False


def test_ca_correlation_measures(cuadras_auge_copula):
    """Test correlation measures with delta=0.5"""
    # Chatterjee's xi
    xi = cuadras_auge_copula.chatterjees_xi()
    assert np.isclose(xi, 0.5**2 / (2 - 0.5))
    assert np.isclose(xi, 1 / 6)

    # Spearman's rho
    rho = cuadras_auge_copula.spearmans_rho()
    assert np.isclose(rho, 3 * 0.5 / (4 - 0.5))
    assert np.isclose(rho, 3 / 7)

    # Kendall's tau
    tau = cuadras_auge_copula.kendalls_tau()
    assert np.isclose(tau, 0.5 / (2 - 0.5))
    assert np.isclose(tau, 1 / 3)


def test_ca_correlation_with_arguments():
    """Test correlation measures using arguments"""
    # Base copula without specified parameter
    copula = CuadrasAuge()

    # Chatterjee's xi
    xi = copula.chatterjees_xi(0.5)
    assert np.isclose(xi, 1 / 6)

    # Spearman's rho
    rho = copula.spearmans_rho(0.5)
    assert np.isclose(rho, 3 / 7)

    # Kendall's tau
    tau = copula.kendalls_tau(0.5)
    assert np.isclose(tau, 1 / 3)


def test_ca_pickands(cuadras_auge_copula):
    """Test Pickands dependence function"""
    # For CuadrasAuge with delta=0.5, at t=0.5
    # A(t) = 1 - delta * min(1-t, t)
    # At t=0.5: A(0.5) = 1 - 0.5 * min(0.5, 0.5) = 1 - 0.5 * 0.5 = 0.75
    result = cuadras_auge_copula._pickands.subs(cuadras_auge_copula.t, 0.5)
    expected = 0.75
    assert float(result) == expected

    # Test at boundaries
    # At t=0: A(0) = 1 - 0.5 * min(1, 0) = 1 - 0.5 * 0 = 1
    # At t=1: A(1) = 1 - 0.5 * min(0, 1) = 1 - 0.5 * 0 = 1
    t0 = float(cuadras_auge_copula._pickands.subs(cuadras_auge_copula.t, 0))
    t1 = float(cuadras_auge_copula._pickands.subs(cuadras_auge_copula.t, 1))
    assert t0 == 1.0
    assert t1 == 1.0


def test_ca_cdf():
    """Test CDF computation"""
    copula = CuadrasAuge(0.3)

    # Mock the SymPyFuncWrapper to avoid actual computation
    with patch.object(CDFWrapper, "__call__") as mock_call:
        mock_call.return_value = 0.42  # Mock return value

        # Call CDF
        result = copula.cdf(0.5, 0.6)

        # Verify it was called with correct params
        mock_call.assert_called_once_with(0.5, 0.6)
        assert result == 0.42


def test_ca_cond_distr_1():
    """Test conditional distribution 1"""
    copula = CuadrasAuge(0.3)

    # Test cond_distr_1
    with patch.object(CD1Wrapper, "__call__") as mock_cd1:
        mock_cd1.return_value = 0.55  # Mock return value
        result = copula.cond_distr_1(0.5, 0.6)
        mock_cd1.assert_called_once_with(0.5, 0.6)
        assert result == 0.55


def test_ca_pdf_unavailable():
    """Test that PDF raises PropertyUnavailableException"""
    copula = CuadrasAuge(0.3)

    with pytest.raises(PropertyUnavailableException) as excinfo:
        _ = copula.pdf

    assert "Cuadras-Auge copula does not have a pdf" in str(excinfo.value)


def test_ca_call_method():
    """Test __call__ method for creating new instances"""
    # Create base copula
    copula = CuadrasAuge()

    # Update parameter using kwargs
    new_copula = copula(delta=0.4)

    # Original should be unchanged
    assert isinstance(copula.delta, sp.Symbol)

    # New instance should have updated parameter
    assert new_copula.delta == 0.4

    # Test with positional arg
    new_copula2 = copula(0.6)
    assert new_copula2.delta == 0.6

    # Test the independence special case
    ind_copula = copula(0)
    assert isinstance(ind_copula, BivIndependenceCopula)

    # Test the upper Fréchet special case
    uf_copula = copula(1)
    assert isinstance(uf_copula, UpperFrechet)


def test_ca_edge_cases():
    """Test correlation measures at edge cases"""
    # Create copulas with extreme values
    copula_low = CuadrasAuge(0.001)  # Almost independence
    copula_high = CuadrasAuge(0.999)  # Almost upper Fréchet

    # For delta near 0, correlation measures should be near 0
    assert copula_low.kendalls_tau() < 0.001
    assert copula_low.spearmans_rho() < 0.001
    assert copula_low.chatterjees_xi() < 0.001

    # For delta near 1, values should approach:
    # kendall's tau: 1/1 = 1
    # spearman's rho: 3/3 = 1
    # chatterjee's xi: 1/1 = 1
    assert np.isclose(copula_high.kendalls_tau(), 0.999 / (2 - 0.999))
    assert np.isclose(copula_high.spearmans_rho(), 3 * 0.999 / (4 - 0.999))
    assert np.isclose(copula_high.chatterjees_xi(), 0.999**2 / (2 - 0.999))
    assert copula_high.kendalls_tau() > 0.9
    assert copula_high.spearmans_rho() > 0.9
    assert copula_high.chatterjees_xi() > 0.9


def test_ca_with_invalid_parameter():
    """Test behavior with invalid parameter values"""
    # Test with parameter outside bounds (should raise ValueError)
    with pytest.raises(ValueError):
        CuadrasAuge(-0.1)  # Negative delta

    with pytest.raises(ValueError):
        CuadrasAuge(1.1)  # Delta > 1

    # Test with invalid parameter in __call__
    copula = CuadrasAuge()
    with pytest.raises(ValueError):
        copula(-0.1)

    with pytest.raises(ValueError):
        copula(1.1)
