import numpy as np
import sympy
from sympy import stats
from copul.families.extreme_value.extreme_value_copula import ExtremeValueCopula
from copul.families.other.independence_copula import IndependenceCopula
from copul.wrapper.cdf_wrapper import CDFWrapper


class HueslerReiss(ExtremeValueCopula):
    """
    Huesler-Reiss extreme value copula.

    The Huesler-Reiss copula is a parametric model for bivariate extreme values,
    characterized by a single dependence parameter delta ≥ 0.
    When delta = 0, it corresponds to the independence copula.
    """

    def __new__(cls, *args, **kwargs):
        # Handle special case during initialization
        if len(args) == 1 and args[0] == 0:
            return IndependenceCopula()
        elif len(args) == 0 and "delta" in kwargs and kwargs["delta"] == 0:
            # Remove delta so it doesn't conflict with IndependenceCopula init
            del kwargs["delta"]
            return IndependenceCopula(**kwargs)

        # Default case - proceed with normal initialization
        return super().__new__(cls)

    @property
    def is_symmetric(self) -> bool:
        return True

    delta = sympy.symbols("delta", nonnegative=True)
    params = [delta]
    intervals = {"delta": sympy.Interval(0, np.inf, left_open=False, right_open=True)}

    def __call__(self, *args, **kwargs):
        if args is not None and len(args) > 0:
            kwargs["delta"] = args[0]

        if "delta" in kwargs and kwargs["delta"] == 0:
            del kwargs["delta"]
            return IndependenceCopula()(**kwargs)

        return super().__call__(**kwargs)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _pickands(self):
        std_norm = stats.cdf(stats.Normal("x", 0, 1))
        return (1 - self.t) * std_norm(self._z(1 - self.t)) + self.t * std_norm(
            self._z(self.t)
        )

    def _z(self, t):
        if t == 0:
            return 0
        elif t == 1:
            return 1
        return 1 / self.delta + self.delta / 2 * sympy.log(t / (1 - t))

    @property
    def cdf(self):
        """
        Compute the cumulative distribution function of the Huesler-Reiss copula.

        The CDF is defined based on the Pickands dependence function.
        """
        u = self.u
        v = self.v

        # Compute the CDF based on Pickands representation
        std_norm = stats.cdf(stats.Normal("x", 0, 1))

        # Convert u, v to standard Fréchet scale
        x = -1 / sympy.log(u)
        y = -1 / sympy.log(v)

        # Compute the CDF using the representation in terms of standard normal CDF
        result = sympy.exp(
            -x * std_norm(self.delta / 2 + 1 / self.delta * sympy.log(y / x))
            - y * std_norm(self.delta / 2 + 1 / self.delta * sympy.log(x / y))
        )

        return CDFWrapper(result)
