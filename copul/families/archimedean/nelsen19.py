import numpy as np
import sympy

from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.families.other.pi_over_sigma_minus_pi import PiOverSigmaMinusPi
from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class Nelsen19(ArchimedeanCopula):
    ac = ArchimedeanCopula
    theta = sympy.symbols("theta", nonnegative=True)
    theta_interval = sympy.Interval(0, np.inf, left_open=False, right_open=True)
    special_cases = {0: PiOverSigmaMinusPi}

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _generator(self):
        expr = sympy.exp(self.theta / self.t) - sympy.exp(self.theta)
        return sympy.Piecewise(
            (expr, self.t > 0),
            (sympy.oo, True)
        )

    @property
    def inv_generator(self):
        theta = self.theta
        y = self.y

        # Regular case expression
        regular_expr = theta / sympy.log(y + sympy.exp(theta))

        # Define piecewise function to handle edge cases
        inv_gen = sympy.Piecewise(
            (0, y == sympy.oo),  # When y is infinity
            (1, y == 0),  # When y is 0
            (regular_expr, True)  # Regular case
        )

        return SymPyFuncWrapper(inv_gen)
    @property
    def cdf(self):
        cdf = self.theta / sympy.log(
            -sympy.exp(self.theta)
            + sympy.exp(self.theta / self.u)
            + sympy.exp(self.theta / self.v)
        )
        return CDFWrapper(cdf)
