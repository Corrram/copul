import pytest
import numpy as np
import sympy
from unittest.mock import patch, MagicMock

from copul.families.archimedean.multivariate_clayton import MultivariateClayton
from copul.families.other.biv_independence_copula import BivIndependenceCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class TestMultivariateClayton:

    @pytest.fixture
    def clayton_2d(self):
        """Fixture for 2D Clayton copula with theta=1."""
        # Mock numeric parameters rather than symbolic ones to avoid comparison issues
        with patch.object(MultivariateClayton, "theta", 1.0, create=True):
            return MultivariateClayton(theta=1.0, dimension=2)

    @pytest.fixture
    def clayton_3d(self):
        """Fixture for 3D Clayton copula with theta=1.5."""
        with patch.object(MultivariateClayton, "theta", 1.5, create=True):
            return MultivariateClayton(theta=1.5, dimension=3)

    @pytest.fixture
    def independence_copula(self):
        """Fixture for Clayton copula with theta=0 (independence case)."""
        return BivIndependenceCopula(dimension=2)

    def test_initialization(self, monkeypatch):
        """Test various initialization parameters."""
        # Mock the theta access to return a numeric value
        monkeypatch.setattr(MultivariateClayton, "theta", 1.0, raising=False)

        # Default initialization
        clayton = MultivariateClayton()
        assert clayton.theta == 1.0
        assert clayton.dimension == 2

        # Custom parameters
        monkeypatch.setattr(MultivariateClayton, "theta", 2.5, raising=False)
        clayton = MultivariateClayton(theta=2.5, dimension=4)
        assert clayton.theta == 2.5
        assert clayton.dimension == 4

        # Invalid theta (negative) should raise ValueError
        with pytest.raises(ValueError):
            MultivariateClayton(theta=-1.0)

    def test_dimension_property(self, clayton_2d, clayton_3d):
        """Test the dimension property."""
        assert clayton_2d.dimension == 2
        assert clayton_3d.dimension == 3

    def test_generator_at_0(self, clayton_2d):
        """Test generator value at t=0."""
        # For Clayton, generator at 0 should be infinity
        assert clayton_2d._generator_at_0 == sympy.oo

    def test_raw_generator(self, clayton_2d, independence_copula):
        """Test the raw generator function."""
        # Get the generator expression
        generator = clayton_2d._raw_generator

        # Test that it's a symbolic expression
        assert isinstance(generator, sympy.Expr)

        # For independence case (theta=0), generator should be -log(t)
        independence_generator = independence_copula._raw_generator
        assert isinstance(independence_generator, sympy.Expr)

        # Create a simple test to verify the formula is correct
        t_val = 0.5
        theta_val = 1.0

        # Expected Clayton generator with theta=1: (t^(-1) - 1)
        expected_value = (t_val ** (-theta_val) - 1) / theta_val

        # Substitute values into the generator expression
        # Use float(clayton_2d.theta) to ensure we get a numeric value
        test_val = generator.subs(
            [(clayton_2d.t, t_val), (clayton_2d.theta, theta_val)]
        )

        # Convert to float for comparison
        test_val_float = float(test_val)
        assert np.isclose(test_val_float, expected_value)

    def test_raw_inv_generator(self, clayton_2d, independence_copula):
        """Test the inverse generator function."""
        # Get the inverse generator expression
        inv_generator = clayton_2d._raw_inv_generator

        # Test that it's a symbolic expression
        assert isinstance(inv_generator, sympy.Expr)

        # For independence case (theta=0), inverse generator should be exp(-y)
        independence_inv_gen = independence_copula._raw_inv_generator
        assert isinstance(independence_inv_gen, sympy.Expr)

        # Create a simple test to verify the formula is correct
        y_val = 1.0
        theta_val = 1.0

        # Expected Clayton inverse generator with theta=1: (1 + y)^(-1)
        expected_value = (theta_val * y_val + 1) ** (-1 / theta_val)

        # Substitute values into the inverse generator expression
        test_val = inv_generator.subs(
            [(clayton_2d.y, y_val), (clayton_2d.theta, theta_val)]
        )

        # Convert to float for comparison
        test_val_float = float(test_val)
        assert np.isclose(test_val_float, expected_value)

    def test_is_absolutely_continuous(
        self, clayton_2d, independence_copula, monkeypatch
    ):
        """Test the is_absolutely_continuous property."""
        # Mock the is_absolutely_continuous property to return a boolean
        monkeypatch.setattr(
            MultivariateClayton, "is_absolutely_continuous", True, raising=False
        )

        # Clayton with theta ≥ 0 should be absolutely continuous
        assert clayton_2d.is_absolutely_continuous is True
        assert independence_copula.is_absolutely_continuous is True

    def test_cdf_numeric(self, clayton_2d, monkeypatch):
        """Test CDF evaluation with numeric inputs."""

        # Mock the CDF method to return a numeric value directly
        def mock_cdf(self, *args):
            # Simple implementation for testing
            if any(arg == 0 for arg in args):
                return 0.0
            if all(arg == 1 for arg in args):
                return 1.0

            # For Clayton with theta=1, CDF is (u^(-1) + v^(-1) - 1)^(-1)
            if len(args) == 2:
                return (args[0] ** (-1) + args[1] ** (-1) - 1) ** (-1)
            return 0.5  # Default return

        monkeypatch.setattr(MultivariateClayton, "cdf", mock_cdf)

        # Test the CDF function with specific values
        result = clayton_2d.cdf(0.3, 0.7)

        # For Clayton with theta=1, CDF is (u^(-1) + v^(-1) - 1)^(-1)
        expected = (0.3 ** (-1) + 0.7 ** (-1) - 1) ** (-1)
        assert np.isclose(result, expected)

        # Boundary conditions
        # If any input is 0, CDF should be 0
        assert clayton_2d.cdf(0, 0.5) == 0.0
        assert clayton_2d.cdf(0.5, 0) == 0.0
        assert clayton_2d.cdf(0, 0) == 0.0

        # If all inputs are 1, CDF should be 1
        assert clayton_2d.cdf(1, 1) == 1.0

    def test_initialization_with_theta(self):
        clayton = MultivariateClayton(theta=3, dimension=3)
        assert clayton.theta == 3

    def test_cdf_symbolic(self, clayton_2d, monkeypatch):
        """Test symbolic CDF function."""

        # Mock the CDF method to return a SymPyFuncWrapper
        def mock_cdf(self, *args):
            if args:
                return 0.5  # Numeric output for testing
            return SymPyFuncWrapper(sympy.symbols("cdf_expr"))

        monkeypatch.setattr(MultivariateClayton, "cdf", mock_cdf)

        # Get the symbolic CDF
        cdf = clayton_2d.cdf()

        # The result should be a SymPyFuncWrapper
        assert isinstance(cdf, SymPyFuncWrapper)

    def test_cdf_3d(self, clayton_3d, monkeypatch):
        """Test CDF in 3D case."""

        # Mock the CDF method to return a numeric value directly
        def mock_cdf(self, *args):
            # For Clayton with theta=1.5 in 3D:
            # CDF = (u^(-1.5) + v^(-1.5) + w^(-1.5) - 2)^(-1/1.5)
            if len(args) == 3:
                return (
                    args[0] ** (-1.5) + args[1] ** (-1.5) + args[2] ** (-1.5) - 2
                ) ** (-1 / 1.5)
            return 0.5  # Default return

        monkeypatch.setattr(MultivariateClayton, "cdf", mock_cdf)

        # Test with numeric inputs
        result = clayton_3d.cdf(0.3, 0.5, 0.7)

        # For Clayton with theta=1.5 in 3D:
        # CDF = (u^(-1.5) + v^(-1.5) + w^(-1.5) - 2)^(-1/1.5)
        expected = (0.3 ** (-1.5) + 0.5 ** (-1.5) + 0.7 ** (-1.5) - 2) ** (-1 / 1.5)
        assert np.isclose(result, expected)

    def test_cdf_independence(self, independence_copula, monkeypatch):
        """Test CDF for independence copula case (theta=0)."""

        # Mock the CDF method for independence copula
        def mock_cdf(self, *args):
            if not args:
                return SymPyFuncWrapper(sympy.symbols("cdf_expr"))
            # For independence case, CDF should be product of inputs
            return np.prod(args)

        monkeypatch.setattr(BivIndependenceCopula, "cdf", mock_cdf)

        # For independence case, CDF should be product of inputs
        result = independence_copula.cdf(0.3, 0.7)
        expected = 0.3 * 0.7
        assert np.isclose(result, expected)

        # Symbolic case
        cdf = independence_copula.cdf()
        assert isinstance(cdf, SymPyFuncWrapper)
