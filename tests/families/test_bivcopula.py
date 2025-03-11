import pytest
import numpy as np
import sympy as sp
from unittest.mock import patch, MagicMock

from copul.families.bivcopula import BivCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class TestBivCopula:
    @pytest.fixture
    def simple_biv_copula(self):
        """Create a simple BivCopula instance for testing"""

        class SimpleBivCopula(BivCopula):
            # Define a simple copula with a parameter theta
            theta = sp.symbols("theta", positive=True)
            params = [theta]
            intervals = {str(theta): sp.Interval(0, float("inf"))}

            @property
            def cdf(self):
                # Simple product copula with a parameter influence
                expr = self.u * self.v * (1 + self.theta * (1 - self.u) * (1 - self.v))
                return SymPyFuncWrapper(expr)

        return SimpleBivCopula(theta=0.5)

    def test_init(self, simple_biv_copula):
        """Test initialization of BivCopula"""
        assert simple_biv_copula.dimension == 2
        assert simple_biv_copula.theta == 0.5
        assert (
            len(simple_biv_copula.params) == 0
        )  # params used in init are removed from list
        assert (
            simple_biv_copula.intervals == {}
        )  # intervals for used params are removed

    def test_segregate_symbols(self):
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

    def test_from_string(self):
        """Test the _from_string class method"""
        # Create with string parameters
        biv_copula = BivCopula._from_string(params=["alpha", "beta"])

        # Check if parameters were properly set
        assert len(biv_copula.params) == 2
        assert str(biv_copula.params[0]) == "alpha"
        assert str(biv_copula.params[1]) == "beta"
        assert hasattr(biv_copula, "alpha")
        assert hasattr(biv_copula, "beta")

    def test_pdf(self, simple_biv_copula):
        """Test the pdf property"""
        pdf = simple_biv_copula.pdf
        assert isinstance(pdf, SymPyFuncWrapper)

        # Verify the PDF at a specific point
        u, v = 0.5, 0.5
        expected = 1 + 0.5 * (1 - 2 * u) * (1 - 2 * v)  # Analytical derivative
        result = float(pdf(u, v))  # Convert to float
        assert abs(result - expected) < 1e-10

    def test_cond_distr_methods(self, simple_biv_copula):
        """Test conditional distribution methods"""
        # Test cond_distr_1
        cd1 = simple_biv_copula.cond_distr_1(u=0.5, v=0.5)
        assert cd1 is not None

        # Test cond_distr_2
        cd2 = simple_biv_copula.cond_distr_2(u=0.5, v=0.5)
        assert cd2 is not None

    @patch("copul.copula_sampler.CopulaSampler")
    def test_rvs(self, mock_sampler, simple_biv_copula):
        """Test random variable sampling"""
        # Setup mock
        mock_instance = MagicMock()
        mock_sampler.return_value = mock_instance
        mock_instance.rvs.return_value = np.array([[0.5, 0.5]])

        # Call rvs - use a with statement to patch the import path
        with patch("copul.families.bivcopula.CopulaSampler", mock_sampler):
            result = simple_biv_copula.rvs(1)

            # Verify
            mock_sampler.assert_called_once_with(simple_biv_copula, random_state=None)
            mock_instance.rvs.assert_called_once_with(1)
            assert np.array_equal(result, np.array([[0.5, 0.5]]))

    def test_rank_correlations(self, simple_biv_copula):
        """Test rank correlation calculations"""
        # These methods can be complex to test directly, so just mock them

        # Mock the _tau method to return a simple value
        with patch.object(BivCopula, "_tau", return_value=0.3):
            tau = simple_biv_copula.kendalls_tau()
            assert tau == 0.3

        # Mock the _rho method
        with patch.object(BivCopula, "_rho", return_value=0.5):
            rho = simple_biv_copula.spearmans_rho()
            assert rho == 0.5

    def test_tail_dependence(self, simple_biv_copula):
        """Test tail dependence coefficients"""
        # Lower tail dependence
        lambda_l = simple_biv_copula.lambda_L()
        assert lambda_l is not None

        # Upper tail dependence
        lambda_u = simple_biv_copula.lambda_U()
        assert lambda_u is not None

    @patch("copul.families.tp2_verifier.TP2Verifier.is_tp2")
    def test_is_tp2(self, mock_is_tp2, simple_biv_copula):
        """Test TP2 property verification"""
        mock_is_tp2.return_value = True

        assert simple_biv_copula.is_tp2() is True
        mock_is_tp2.assert_called_once_with(simple_biv_copula)

    @patch("copul.families.cis_verifier.CISVerifier.is_cis")
    def test_is_cis(self, mock_is_cis, simple_biv_copula):
        """Test CIS property verification"""
        mock_is_cis.return_value = True

        assert simple_biv_copula.is_cis() is True
        mock_is_cis.assert_called_once_with(simple_biv_copula)
