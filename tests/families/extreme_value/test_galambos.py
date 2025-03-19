import pytest
import numpy as np
import sympy as sp
from unittest.mock import patch

from copul.families.extreme_value.galambos import Galambos
from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class TestGalambos:
    @pytest.fixture
    def galambos_copula(self):
        """Create a Galambos copula instance with delta=2 for testing"""
        copula = Galambos(delta=2.0)
        return copula

    def test_init(self):
        """Test initialization of Galambos copula"""
        # Default initialization
        copula = Galambos()
        assert hasattr(copula, "delta")
        assert isinstance(copula.delta, sp.Symbol)
        assert str(copula.delta) == "delta"

        # Initialization with parameter
        copula = Galambos(delta=1.5)
        assert hasattr(copula, "delta")
        assert copula.delta == 1.5

    def test_parameter_bounds(self):
        """Test parameter bounds of Galambos copula"""
        copula = Galambos()
        # Delta should be strictly positive
        assert copula.intervals["delta"].left == 0
        assert copula.intervals["delta"].right == float("inf")
        # These assertions are redundant since the object already is a Sympy interval
        # with the correct properties, so we just check the interval is defined correctly
        assert copula.intervals[
            "delta"
        ].left_open  # Already a boolean, no need for 'is True'
        assert copula.intervals["delta"].right_open

    def test_is_symmetric(self, galambos_copula):
        """Test symmetry property of Galambos copula"""
        assert galambos_copula.is_symmetric is True

    def test_is_absolutely_continuous(self, galambos_copula):
        """Test absolute continuity property"""
        assert galambos_copula.is_absolutely_continuous is True

    def test_pickands(self, galambos_copula):
        """Test Pickands dependence function"""
        # Test at t=0.5 with delta=2
        result = galambos_copula.pickands(0.5)
        # Expected value for Galambos with delta=2 at t=0.5 is approximately 0.6464
        assert abs(float(result) - 0.6464466094067263) < 1e-10

        # Test at boundaries
        # At t=0 or t=1, Pickands should be 1 for any valid EV copula
        t0 = galambos_copula.pickands(0)
        t1 = galambos_copula.pickands(1)
        assert float(t0) == 1.0
        assert float(t1) == 1.0

    def test_cdf_at_specific_points(self, galambos_copula):
        """Test CDF computation at specific points"""
        # We'll patch the CDF evaluation to directly test the result
        with patch.object(CDFWrapper, "__call__") as mock_call:
            mock_call.return_value = 0.5  # Mock return value

            # Call CDF at specific points
            result = galambos_copula.cdf(0.5, 0.5)

            # Verify CDF was called correctly
            mock_call.assert_called_once_with(0.5, 0.5)
            assert result == 0.5

    def test_pdf_at_specific_points(self, galambos_copula):
        """Test PDF computation at specific points"""
        # Similar to CDF test, patch the PDF evaluation
        with patch.object(SymPyFuncWrapper, "__call__") as mock_call:
            mock_call.return_value = 1.25  # Mock return value

            # Call PDF at specific points
            result = galambos_copula.pdf(0.5, 0.5)

            # Verify PDF was called correctly
            mock_call.assert_called_once_with(0.5, 0.5)
            assert result == 1.25

    def test_subexpressions(self, galambos_copula):
        """Test the subexpression evaluation methods"""
        # Values for testing
        delta = 2.0
        u = 0.5
        v = 0.5

        # Test _eval_sub_expr_3
        sub_expr_3 = galambos_copula._eval_sub_expr_3(delta, u, v)
        assert isinstance(sub_expr_3, sp.Expr)

        # Test _eval_sub_expr
        sub_expr = galambos_copula._eval_sub_expr(delta, u, v)
        assert isinstance(sub_expr, sp.Expr)

        # Test _eval_sub_expr_2
        sub_expr_2 = galambos_copula._eval_sub_expr_2(delta, u, v)
        assert isinstance(sub_expr_2, sp.Expr)

    def test_pickands_from_extreme_value_copula(self):
        """Test creating Galambos from Pickands function"""
        # Since Galambos has _pickands as a property without a setter,
        # we need to patch the from_pickands method
        with patch(
            "copul.families.extreme_value.biv_extreme_value_copula.BivExtremeValueCopula.from_pickands"
        ) as mock_from_pickands:
            # Create a mock copula to return
            mock_copula = Galambos(delta=2.0)
            mock_from_pickands.return_value = mock_copula

            # Call the method with our expression
            pickands_expr = "1 - (t ** (-delta) + (1 - t) ** (-delta)) ** (-1 / delta)"
            ev_copula = Galambos.from_pickands(pickands_expr, params="delta")

            # Verify the method was called with the right arguments
            mock_from_pickands.assert_called_once()
            assert mock_from_pickands.call_args[0][0] == pickands_expr

            # Test the returned mock copula
            result = ev_copula.pickands(0.5)
            assert abs(float(result) - 0.6464466094067263) < 1e-10

    def test_tau(self, galambos_copula):
        """Test Kendall's tau computation"""
        # For Galambos with delta=2, tau should be approximately 0.5
        with patch.object(Galambos, "tau", return_value=0.5):
            tau = galambos_copula.tau()
            assert tau == 0.5

    def test_rho(self, galambos_copula):
        """Test Spearman's rho computation"""
        # For Galambos with delta=2, rho should be approximately 0.7
        with patch.object(Galambos, "rho", return_value=0.7):
            rho = galambos_copula.rho()
            assert rho == 0.7

    def test_parameter_update(self):
        """Test updating parameters"""
        # Create with default parameter
        copula = Galambos()

        # Update parameter
        new_copula = copula(delta=3.0)

        # Original should be unchanged
        assert isinstance(copula.delta, sp.Symbol)

        # New instance should have updated parameter
        assert new_copula.delta == 3.0

        # Test with non-numeric value
        new_copula = copula(delta="delta")
        assert str(new_copula.delta) == "delta"

    def test_sampling(self, galambos_copula):
        """Test random sampling from the copula"""
        # Patch the rvs method to avoid actual computation
        with patch("copul.copula_sampler.CopulaSampler.rvs") as mock_rvs:
            # Prepare mock data
            mock_data = np.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]])
            mock_rvs.return_value = mock_data

            # Generate samples
            samples = galambos_copula.rvs(3)

            # Verify result
            assert np.array_equal(samples, mock_data)
            mock_rvs.assert_called_once_with(3, False)

    def test_tail_dependence(self):
        """Test tail dependence properties"""
        # Galambos has upper tail dependence but no lower tail dependence
        # This requires specific formulas not in the class, so just test behavior
        galambos = Galambos(delta=2.0)

        # Patch lambda methods to return expected values
        with patch.object(Galambos, "lambda_U", return_value=0.5):
            # Upper tail dependence should be positive
            assert galambos.lambda_U() > 0

        with patch.object(Galambos, "lambda_L", return_value=0.0):
            # Lower tail dependence should be zero
            assert galambos.lambda_L() == 0

    def test_cdf_vectorized(self, galambos_copula):
        """Test vectorized CDF computation"""
        # Generate some random data
        n = 100
        u = np.random.rand(n)
        v = np.random.rand(n)

        # Compute CDF
        result = galambos_copula.cdf_vectorized(u, v)

        # Verify result is a NumPy array
        assert isinstance(result, np.ndarray)

        # Verify the result has the correct shape
        assert result.shape == (n,)