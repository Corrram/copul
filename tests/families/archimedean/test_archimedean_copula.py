import numpy as np
import pytest
import sympy

from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.families.other.independence_copula import IndependenceCopula
from copul.families.other.lower_frechet import LowerFrechet


def test_special_cases_create_method():
    """Test that the create factory method correctly handles special cases."""

    # Define a test copula class with special cases
    class TestCopula(ArchimedeanCopula):
        theta_interval = sympy.Interval(-1, sympy.oo, left_open=False, right_open=True)

        # Define special cases
        special_cases = {-1: LowerFrechet, 0: IndependenceCopula}

        @property
        def is_absolutely_continuous(self) -> bool:
            return self.theta >= 0

        @property
        def _generator(self):
            if self.theta == 0:
                return -sympy.log(self.t)
            return ((1 / self.t) ** self.theta - 1) / self.theta

    # Test regular case
    regular = TestCopula.create(2)
    assert isinstance(regular, TestCopula)
    assert regular.theta == 2

    # Test special case: theta = 0 should return IndependenceCopula
    independence = TestCopula.create(0)
    assert isinstance(independence, IndependenceCopula)

    # Test special case: theta = -1 should return LowerFrechet
    lower_frechet = TestCopula.create(-1)
    assert isinstance(lower_frechet, LowerFrechet)

    # Test with keyword arguments
    kwargs_regular = TestCopula.create(theta=2)
    assert isinstance(kwargs_regular, TestCopula)
    assert kwargs_regular.theta == 2

    kwargs_special = TestCopula.create(theta=0)
    assert isinstance(kwargs_special, IndependenceCopula)


def test_special_cases_new_method():
    """Test that the __new__ constructor correctly handles special cases."""

    # Define a test copula class with special cases
    class TestCopula(ArchimedeanCopula):
        theta_interval = sympy.Interval(-1, sympy.oo, left_open=False, right_open=True)

        # Define special cases
        special_cases = {-1: LowerFrechet, 0: IndependenceCopula}

        @property
        def is_absolutely_continuous(self) -> bool:
            return self.theta >= 0

        @property
        def _generator(self):
            if self.theta == 0:
                return -sympy.log(self.t)
            return ((1 / self.t) ** self.theta - 1) / self.theta

    # Test regular case
    regular = TestCopula(2)
    assert isinstance(regular, TestCopula)
    assert regular.theta == 2

    # Test special case: theta = 0 should return IndependenceCopula
    independence = TestCopula(0)
    assert isinstance(independence, IndependenceCopula)

    # Test special case: theta = -1 should return LowerFrechet
    lower_frechet = TestCopula(-1)
    assert isinstance(lower_frechet, LowerFrechet)

    # Test with keyword arguments
    kwargs_regular = TestCopula(theta=2)
    assert isinstance(kwargs_regular, TestCopula)
    assert kwargs_regular.theta == 2

    kwargs_special = TestCopula(theta=0)
    assert isinstance(kwargs_special, IndependenceCopula)


def test_special_cases_call_method():
    """Test that the __call__ method correctly handles special cases."""

    # Define a test copula class with special cases
    class TestCopula(ArchimedeanCopula):
        theta_interval = sympy.Interval(-1, sympy.oo, left_open=False, right_open=True)

        # Define special cases
        special_cases = {-1: LowerFrechet, 0: IndependenceCopula}

        @property
        def is_absolutely_continuous(self) -> bool:
            return self.theta >= 0

        @property
        def _generator(self):
            if self.theta == 0:
                return -sympy.log(self.t)
            return ((1 / self.t) ** self.theta - 1) / self.theta

    # Create a regular instance
    copula = TestCopula(2)

    # Test __call__ with regular parameter
    regular = copula(3)
    assert isinstance(regular, TestCopula)
    assert regular.theta == 3

    # Test __call__ with special case parameter
    independence = copula(0)
    assert isinstance(independence, IndependenceCopula)

    lower_frechet = copula(-1)
    assert isinstance(lower_frechet, LowerFrechet)

    # Test with keyword arguments
    kwargs_regular = copula(theta=3)
    assert isinstance(kwargs_regular, TestCopula)
    assert kwargs_regular.theta == 3

    kwargs_special = copula(theta=0)
    assert isinstance(kwargs_special, IndependenceCopula)


def test_empty_special_cases():
    """Test that copulas with no special cases work correctly."""

    # Define a test copula class with no special cases
    class SimpleTestCopula(ArchimedeanCopula):
        theta_interval = sympy.Interval(0, sympy.oo, left_open=False, right_open=True)

        # Empty special cases
        special_cases = {}

        @property
        def is_absolutely_continuous(self) -> bool:
            return True

        @property
        def _generator(self):
            return -sympy.log(self.t) * self.theta

    # Test create method
    regular = SimpleTestCopula.create(2)
    assert isinstance(regular, SimpleTestCopula)
    assert regular.theta == 2

    # Test __new__ method
    direct = SimpleTestCopula(3)
    assert isinstance(direct, SimpleTestCopula)
    assert direct.theta == 3

    # Test __call__ method
    instance = SimpleTestCopula(1)
    new_instance = instance(4)
    assert isinstance(new_instance, SimpleTestCopula)
    assert new_instance.theta == 4


def test_inherited_special_cases():
    """Test that special cases are properly inherited by subclasses."""

    # Define a base test copula with special cases
    class BaseTestCopula(ArchimedeanCopula):
        theta_interval = sympy.Interval(-1, sympy.oo, left_open=False, right_open=True)

        # Define special cases
        special_cases = {-1: LowerFrechet, 0: IndependenceCopula}

        @property
        def is_absolutely_continuous(self) -> bool:
            return self.theta >= 0

        @property
        def _generator(self):
            if self.theta == 0:
                return -sympy.log(self.t)
            return ((1 / self.t) ** self.theta - 1) / self.theta

    # Define a subclass that inherits from the base
    class SubTestCopula(BaseTestCopula):
        # No need to redefine special_cases, should inherit from parent
        pass

    # Test special case handling in subclass
    regular = SubTestCopula(2)
    assert isinstance(regular, SubTestCopula)

    independence = SubTestCopula(0)
    assert isinstance(independence, IndependenceCopula)

    lower_frechet = SubTestCopula(-1)
    assert isinstance(lower_frechet, LowerFrechet)


def test_overridden_special_cases():
    """Test that subclasses can override special cases from parent class."""

    # Define a base test copula with special cases
    class BaseTestCopula(ArchimedeanCopula):
        theta_interval = sympy.Interval(-1, sympy.oo, left_open=False, right_open=True)

        # Define special cases
        special_cases = {-1: LowerFrechet, 0: IndependenceCopula}

        @property
        def is_absolutely_continuous(self) -> bool:
            return self.theta >= 0

        @property
        def _generator(self):
            if self.theta == 0:
                return -sympy.log(self.t)
            return ((1 / self.t) ** self.theta - 1) / self.theta

    # Define a subclass that overrides special cases
    class SubTestCopula(BaseTestCopula):
        # Override with different special cases
        special_cases = {
            # Only keep theta = -1 special case
            -1: LowerFrechet
        }

    # Test special case handling in subclass
    regular = SubTestCopula(2)
    assert isinstance(regular, SubTestCopula)

    # Should be a regular instance now, not IndependenceCopula
    regular_zero = SubTestCopula(0)
    assert isinstance(regular_zero, SubTestCopula)
    assert regular_zero.theta == 0

    # This special case is still preserved
    lower_frechet = SubTestCopula(-1)
    assert isinstance(lower_frechet, LowerFrechet)


def test_invalid_params():
    """Test that invalid parameters raise ValueError."""

    # Define a test copula class with invalid parameter values
    class TestCopula(ArchimedeanCopula):
        theta_interval = sympy.Interval(
            -np.inf, np.inf, left_open=True, right_open=True
        )

        # Define special cases and invalid parameters
        special_cases = {-1: IndependenceCopula}
        invalid_params = {0}  # theta = 0 should raise ValueError

        @property
        def is_absolutely_continuous(self) -> bool:
            return True

        @property
        def _generator(self):
            return -sympy.log(self.t) * self.theta

    # Test regular case
    regular = TestCopula(2)
    assert isinstance(regular, TestCopula)
    assert regular.theta == 2

    # Test special case: theta = -1 should return IndependenceCopula
    special_case = TestCopula(-1)
    assert isinstance(special_case, IndependenceCopula)

    # Test invalid parameter: theta = 0 should raise ValueError
    with pytest.raises(ValueError, match="Parameter theta cannot be 0"):
        TestCopula(0)

    # Test with keyword arguments
    with pytest.raises(ValueError, match="Parameter theta cannot be 0"):
        TestCopula(theta=0)

    # Test via create method
    with pytest.raises(ValueError, match="Parameter theta cannot be 0"):
        TestCopula.create(0)

    # Test via __call__ method
    copula = TestCopula(2)
    with pytest.raises(ValueError, match="Parameter theta cannot be 0"):
        copula(0)


def test_both_special_and_invalid_params():
    """Test a copula with both special cases and invalid parameters."""

    # Define a test copula class with both special cases and invalid parameters
    class ComplexCopula(ArchimedeanCopula):
        theta_interval = sympy.Interval(
            -np.inf, np.inf, left_open=True, right_open=True
        )

        # Define special cases
        special_cases = {-1: IndependenceCopula, 1: LowerFrechet}

        # Define invalid parameters
        invalid_params = {0, 2}  # theta = 0 or 2 should raise ValueError

        @property
        def is_absolutely_continuous(self) -> bool:
            return True

        @property
        def _generator(self):
            return -sympy.log(self.t) * self.theta

    # Test regular case
    regular = ComplexCopula(3)
    assert isinstance(regular, ComplexCopula)
    assert regular.theta == 3

    # Test special cases
    independence = ComplexCopula(-1)
    assert isinstance(independence, IndependenceCopula)

    lower_frechet = ComplexCopula(1)
    assert isinstance(lower_frechet, LowerFrechet)

    # Test invalid parameters
    with pytest.raises(ValueError, match="Parameter theta cannot be 0"):
        ComplexCopula(0)

    with pytest.raises(ValueError, match="Parameter theta cannot be 2"):
        ComplexCopula(2)

    # Test with __call__ method
    copula = ComplexCopula(3)
    result1 = copula(-1)
    assert isinstance(result1, IndependenceCopula)

    with pytest.raises(ValueError, match="Parameter theta cannot be 2"):
        copula(2)
