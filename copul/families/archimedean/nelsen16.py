import numpy as np
import sympy

from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.families.other.lower_frechet import LowerFrechet
from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class Nelsen16(ArchimedeanCopula):
    ac = ArchimedeanCopula
    theta = sympy.symbols("theta", nonnegative=True)
    theta_interval = sympy.Interval(0, np.inf, left_open=False, right_open=True)
    special_cases = {0: LowerFrechet}

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def generator(self):
        expr = (self.theta / self.t + 1) * (1 - self.t)
        gen = sympy.Piecewise(
            (expr, self.t > 0), (1, self.theta == 0), (sympy.oo, True)
        )
        return SymPyFuncWrapper(gen)

    @property
    def _raw_inv_generator(self):
        theta = self.theta
        y = self.y
        return (1 - theta - y + sympy.sqrt((theta + y - 1) ** 2 + 4 * theta)) / 2

    @property
    def cdf(self):
        th = self.theta
        v = self.v
        u = self.u

        # Define the original calculation for non-zero u and v
        regular_case = (
            u * v * (1 - th)
            + u * (th + v) * (v - 1)
            + v * (th + u) * (u - 1)
            + sympy.sqrt(
                4 * th * u**2 * v**2
                + (u * v * (1 - th) + u * (th + v) * (v - 1) + v * (th + u) * (u - 1))
                ** 2
            )
        ) / (2 * u * v)

        # Use Piecewise to handle the edge cases
        cdf = sympy.Piecewise((regular_case, sympy.And(u > 0, v > 0)), (0, True))
        return CDFWrapper(cdf)

    def first_deriv_of_ci_char(self):
        theta = self.theta
        y = self.y
        return (
            4
            * theta
            / (4 * theta + (theta + y - 1) ** 2)
            * (theta + y - sympy.sqrt(4 * theta - (theta + y - 1) ** 2) - 1)
        )

    def second_deriv_of_ci_char(self):
        theta = self.theta
        y = self.y
        return (
            4
            * theta
            * (
                -2
                * sympy.sqrt(4 * theta - (theta + y - 1) ** 2)
                * (theta + y - 1)
                * (theta + y - sympy.sqrt(4 * theta - (theta + y - 1) ** 2) - 1)
                + (4 * theta + (theta + y - 1) ** 2)
                * (theta + y + sympy.sqrt(4 * theta - (theta + y - 1) ** 2) - 1)
            )
            / (
                sympy.sqrt(4 * theta - (theta + y - 1) ** 2)
                * (4 * theta + (theta + y - 1) ** 2) ** 2
            )
        )

    def lambda_L(self):
        return 1 / 2

    def lambda_U(self):
        return 0
