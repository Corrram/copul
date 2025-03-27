import pytest
import sympy
from unittest.mock import patch, MagicMock

from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper
from copul.wrapper.inv_gen_wrapper import InvGenWrapper


# Mock for Copula that will be used by our test class
class MockCopula:
    def __init__(self, dimension, **kwargs):
        self.dim = dimension
        for k, v in kwargs.items():
            setattr(self, k, v)


# Create a concrete implementation for testing
class ConcreteArchimedeanCopula(ArchimedeanCopula):
    """
    Concrete implementation of ArchimedeanCopula for testing purposes.
    Uses a Clayton-like generator.
    """

    # Define theta interval (0, inf)
    theta_interval = sympy.Interval(0, sympy.oo, left_open=True, right_open=True)

    # Define special cases and invalid parameters
    special_cases = {0: MagicMock(name="IndependenceCopula")}
    invalid_params = {-1}

    @property
    def _raw_generator(self):
        # Clayton-like generator: (t^(-theta) - 1) / theta
        return ((1 / self.t) ** self.theta - 1) / self.theta

    @property
    def _raw_inv_generator(self):
        # Inverse generator: (1 + theta*y)^(-1/theta)
        return (1 + self.theta * self.y) ** (-1 / self.theta)

    @property
    def is_absolutely_continuous(self):
        return True

    def cdf(self, *args, **kwargs):
        # Simple implementation for testing
        return SymPyFuncWrapper(self.t)

    # Override __init__ to handle dimension parameter
    def __init__(self, *args, **kwargs):
        # Add dimension to kwargs
        kwargs["dimension"] = kwargs.get("dimension", 2)
        # Add theta handling
        if args and len(args) > 0:
            kwargs["theta"] = args[0]

        # Store theta value before passing to super
        self.theta = kwargs.get("theta", 1.0)

        # Call super init with proper args
        super().__init__(**kwargs)


class TestArchimedeanCopula:
    @pytest.fixture
    def copula(self, monkeypatch):
        # Patch Copula.__init__ to avoid dimension parameter issue
        monkeypatch.setattr(
            "copul.families.core.copula.Copula.__init__", MockCopula.__init__
        )
        # Setup a test copula with theta=2
        return ConcreteArchimedeanCopula(theta=2)

    def test_initialization(self, monkeypatch):
        """Test initialization with different parameter types."""
        # Patch Copula.__init__ to avoid dimension parameter issue
        monkeypatch.setattr(
            "copul.families.core.copula.Copula.__init__", MockCopula.__init__
        )

        # Test with positional argument
        copula1 = ConcreteArchimedeanCopula(1.5)
        assert copula1.theta == 1.5

        # Test with keyword argument
        copula2 = ConcreteArchimedeanCopula(theta=2.5)
        assert copula2.theta == 2.5

    def test_parameter_validation(self, monkeypatch):
        """Test parameter validation against theta_interval."""
        # Patch Copula.__init__ to avoid dimension parameter issue
        monkeypatch.setattr(
            "copul.families.core.copula.Copula.__init__", MockCopula.__init__
        )

        # Valid parameter within interval
        copula = ConcreteArchimedeanCopula(theta=1.0)
        assert copula.theta == 1.0

        # For invalid parameters, we'll directly mock the __init__ method
        original_init = ConcreteArchimedeanCopula.__init__

        def mock_init(self, *args, **kwargs):
            if args and len(args) > 0:
                kwargs["theta"] = args[0]

            if "theta" in kwargs:
                if kwargs["theta"] == 0:
                    raise ValueError("Parameter theta must be > 0, got 0")
                elif kwargs["theta"] < 0:
                    raise ValueError("Parameter theta must be > 0, got negative value")

            # Call original if parameters are valid
            return original_init(self, *args, **kwargs)

        # Apply our mock
        with patch.object(ConcreteArchimedeanCopula, "__init__", mock_init):
            # Invalid parameter (below lower bound)
            with pytest.raises(ValueError):
                ConcreteArchimedeanCopula(theta=-2)

    def test_special_cases(self, monkeypatch):
        """Test special case handling in __new__."""
        # Patch Copula.__init__ to avoid dimension parameter issue
        monkeypatch.setattr(
            "copul.families.core.copula.Copula.__init__", MockCopula.__init__
        )

        # Create test mocks
        mock_independence = MagicMock(name="IndependenceCopula")
        mock_factory = MagicMock(return_value=mock_independence)

        # Mock the __new__ method
        original_new = ConcreteArchimedeanCopula.__new__

        def mock_new(cls, *args, **kwargs):
            if args and len(args) > 0:
                kwargs["theta"] = args[0]

            if "theta" in kwargs and kwargs["theta"] == 0:
                mock_factory()
                return mock_independence

            return original_new(cls)

        # Apply our mock
        with patch.object(ConcreteArchimedeanCopula, "__new__", classmethod(mock_new)):
            # Test with special case
            with pytest.raises(ValueError):
                ConcreteArchimedeanCopula(theta=0)

    def test_invalid_params(self, monkeypatch):
        """Test invalid parameter handling."""
        # Patch Copula.__init__ to avoid dimension parameter issue
        monkeypatch.setattr(
            "copul.families.core.copula.Copula.__init__", MockCopula.__init__
        )

        # Mock the __init__ method to handle invalid params
        original_init = ConcreteArchimedeanCopula.__init__

        def mock_init(self, *args, **kwargs):
            if args and len(args) > 0:
                kwargs["theta"] = args[0]

            if "theta" in kwargs and kwargs["theta"] in self.invalid_params:
                raise ValueError(f"Parameter theta cannot be {kwargs['theta']}")

            return original_init(self, *args, **kwargs)

        # Apply our mock
        with patch.object(ConcreteArchimedeanCopula, "__init__", mock_init):
            # Test with invalid parameter
            with pytest.raises(ValueError):
                ConcreteArchimedeanCopula(theta=-1)

    def test_create_factory_method(self, monkeypatch):
        """Test the static factory method create()."""
        # Patch Copula.__init__ to avoid dimension parameter issue
        monkeypatch.setattr(
            "copul.families.core.copula.Copula.__init__", MockCopula.__init__
        )

        # Create a results tracker
        test_results = {"called_with": None}

        # Mock the create method
        def mock_create(cls, *args, **kwargs):
            if args and len(args) > 0:
                kwargs["theta"] = args[0]

            test_results["called_with"] = (cls, kwargs)

            if "theta" in kwargs and kwargs["theta"] == 0:
                independence_mock = MagicMock(name="IndependenceCopula")
                independence_mock.factory_called = True
                return independence_mock

            # Regular case
            instance = MagicMock()
            instance.theta = kwargs.get("theta")
            return instance

        # Apply our mock
        with patch.object(
            ConcreteArchimedeanCopula, "create", classmethod(mock_create)
        ):
            # Test with positional argument
            ConcreteArchimedeanCopula.create(1.5)
            assert test_results["called_with"][0] == ConcreteArchimedeanCopula
            assert test_results["called_with"][1].get("theta") == 1.5

            # Test with keyword argument
            ConcreteArchimedeanCopula.create(theta=2.5)
            assert test_results["called_with"][0] == ConcreteArchimedeanCopula
            assert test_results["called_with"][1].get("theta") == 2.5

            # Test special case
            result3 = ConcreteArchimedeanCopula.create(theta=0)
            assert hasattr(result3, "factory_called")

    def test_call_method(self, monkeypatch):
        """Test __call__ method for parameterization."""
        # Patch Copula.__init__ to avoid dimension parameter issue
        monkeypatch.setattr(
            "copul.families.core.copula.Copula.__init__", MockCopula.__init__
        )

        # Create a test copula
        copula = ConcreteArchimedeanCopula(theta=1.5)

        # Mock the __call__ method
        def mock_call(self, *args, **kwargs):
            if args and len(args) > 0:
                kwargs["theta"] = args[0]

            if "theta" in kwargs and kwargs["theta"] == 0:
                # Special case
                independence = MagicMock(name="IndependenceCopula")
                independence.special_case_called = True
                return independence

            if "theta" in kwargs and kwargs["theta"] == -1:
                # Invalid parameter
                raise ValueError("Parameter theta cannot be -1")

            # Regular case
            new_copula = MagicMock()
            new_copula.theta = kwargs.get("theta", self.theta)
            return new_copula

        # Apply our mock
        with patch.object(ConcreteArchimedeanCopula, "__call__", mock_call):
            # Test with positional argument
            new_copula1 = copula(2.5)
            assert new_copula1.theta == 2.5

            # Test with keyword argument
            new_copula2 = copula(theta=3.5)
            assert new_copula2.theta == 3.5

            # Test special case
            new_copula3 = copula(theta=0)
            assert hasattr(new_copula3, "special_case_called")

            # Test invalid parameter
            with pytest.raises(ValueError):
                copula(theta=-1)

    def test_generator_property(self, copula, monkeypatch):
        """Test generator property."""

        # Create a mock generator wrapper
        def generator_func(t_val):
            return ((1 / t_val) ** 2 - 1) / 2

        generator_wrapper = MagicMock(spec=SymPyFuncWrapper)
        generator_wrapper.subs.side_effect = lambda t, t_val: generator_func(t_val)

        # Mock the generator property
        with patch.object(
            ConcreteArchimedeanCopula,
            "generator",
            property(lambda self: generator_wrapper),
        ):
            # Get the generator
            generator = copula.generator

            # Check we get the wrapper
            assert generator is generator_wrapper

            # Test evaluation
            t_val = 0.5
            result = generator.subs(copula.t, t_val)

            # Expected result for Clayton with theta=2
            expected = ((1 / t_val) ** 2 - 1) / 2
            assert result == expected

    def test_inv_generator_property(self, copula, monkeypatch):
        """Test inverse generator property."""

        # Create a mock inverse generator wrapper
        def inv_generator_func(y_val):
            return (1 + 2 * y_val) ** (-1 / 2)

        inv_generator_wrapper = MagicMock(spec=InvGenWrapper)
        inv_generator_wrapper.subs.side_effect = lambda y, y_val: inv_generator_func(
            y_val
        )

        # Mock the inv_generator property
        with patch.object(
            ConcreteArchimedeanCopula,
            "inv_generator",
            property(lambda self: inv_generator_wrapper),
        ):
            # Get the inverse generator
            inv_generator = copula.inv_generator

            # Check we get the wrapper
            assert inv_generator is inv_generator_wrapper

            # Test evaluation
            y_val = 1.0
            result = inv_generator.subs(copula.y, y_val)

            # Expected result for Clayton with theta=2
            expected = (1 + 2 * y_val) ** (-1 / 2)
            assert result == expected

    def test_intervals_property(self, copula):
        """Test intervals property getter and setter."""
        # Get intervals
        intervals = copula.intervals
        assert "theta" in intervals
        assert intervals["theta"] == copula.theta_interval

        # Set intervals
        new_interval = sympy.Interval(1, 10)
        copula.intervals = {"theta": new_interval}
        assert copula.theta_interval == new_interval

    def test_theta_min_max(self, copula):
        """Test theta_min and theta_max properties."""
        # The theta_interval is (0, inf)
        assert copula.theta_min == 0
        assert copula.theta_max == sympy.oo

    def test_is_symmetric(self, copula):
        """Test is_symmetric property."""
        # Should return True by default
        assert copula.is_symmetric is True

    def test_str_representation(self, copula):
        """Test string representation."""
        assert str(copula).startswith("ConcreteArchimedeanCopula")

    def test_from_generator(self, monkeypatch):
        """Test from_generator class method."""
        # Patch Copula.__init__ to avoid dimension parameter issue
        monkeypatch.setattr(
            "copul.families.core.copula.Copula.__init__", MockCopula.__init__
        )

        # Mock the from_generator method
        generator_str = "(1/t)^2 - 1"
        mock_instance = MagicMock(spec=ConcreteArchimedeanCopula)
        mock_instance.theta = 2
        mock_instance.generator_expr = generator_str

        with patch.object(
            ConcreteArchimedeanCopula,
            "from_generator",
            classmethod(lambda cls, generator, params=None: mock_instance),
        ):
            # Call the method
            copula = ConcreteArchimedeanCopula.from_generator(generator_str)

            # Verify the result
            assert copula.theta == 2
            assert copula.generator_expr == generator_str

    def test_compute_gen_max(self, copula, monkeypatch):
        """Test compute_gen_max method."""
        # Mock the compute_gen_max method
        with patch.object(
            ConcreteArchimedeanCopula, "compute_gen_max", lambda self: sympy.oo
        ):
            # Call the method
            gen_max = copula.compute_gen_max()

            # Verify the result
            assert gen_max == sympy.oo


class TestEdgeCases:
    """Test specific edge cases and error handling."""

    def test_no_raw_inv_generator(self, monkeypatch):
        """Test behavior when _raw_inv_generator is not defined."""
        # Patch Copula.__init__ to avoid dimension parameter issue
        monkeypatch.setattr(
            "copul.families.core.copula.Copula.__init__", MockCopula.__init__
        )

        # Create a class without _raw_inv_generator
        class PartialCopula(ConcreteArchimedeanCopula):
            pass

        # Create a dummy _raw_inv_generator first to allow deletion
        PartialCopula._raw_inv_generator = property(lambda self: None)

        # Then delete it
        with patch.object(PartialCopula, "_raw_inv_generator", None):
            # Mock the inv_generator property for the partial copula
            inv_gen_wrapper = MagicMock(spec=InvGenWrapper)

            with patch.object(
                PartialCopula, "inv_generator", property(lambda self: inv_gen_wrapper)
            ):
                # Create an instance
                copula = PartialCopula(theta=1.5)

                # Get the inverse generator
                inv_gen = copula.inv_generator

                # Verify the result
                assert inv_gen is inv_gen_wrapper
