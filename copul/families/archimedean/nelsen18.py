import numpy as np
import sympy

from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.wrapper.cd2_wrapper import CD2Wrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class Nelsen18(ArchimedeanCopula):
    ac = ArchimedeanCopula
    theta = sympy.symbols("theta", positive=True)
    theta_interval = sympy.Interval(2, np.inf, left_open=False, right_open=True)

    @property
    def is_absolutely_continuous(self) -> bool:
        return False

    @property
    def _generator(self):
        return sympy.exp(self.theta / (self.t - 1))

    @property
    def inv_generator(self) -> SymPyFuncWrapper:
        ind = sympy.Heaviside(self.y - sympy.exp(-self.theta))
        gen = ind * self.theta / sympy.log(self.y) + 1
        return SymPyFuncWrapper(gen)

    @property
    def cdf(self) -> SymPyFuncWrapper:
        cdf = sympy.Max(
            (
                1
                + (
                    self.theta
                    / sympy.log(
                        (
                            sympy.Piecewise(
                                (0, self.u >= 1),
                                (sympy.exp(self.theta / (self.u - 1)), True),
                            )
                            + sympy.Piecewise(
                                (0, self.v >= 1),
                                (sympy.exp(self.theta / (self.v - 1)), True),
                            )
                        )
                    )
                )
            ),
            0,
        )
        return SymPyFuncWrapper(cdf)

    def cond_distr_2(self, u=None, v=None):
        cond_distr = (
            self.theta**2
            * sympy.exp(self.theta / (self.v - 1))
            * sympy.Heaviside(
                self.theta
                / sympy.log(
                    sympy.exp(self.theta / (self.u - 1))
                    + sympy.exp(self.theta / (self.v - 1))
                )
                + 1
            )
            / (
                (self.v - 1) ** 2
                * (
                    sympy.exp(self.theta / (self.u - 1))
                    + sympy.exp(self.theta / (self.v - 1))
                )
                * sympy.log(
                    sympy.exp(self.theta / (self.u - 1))
                    + sympy.exp(self.theta / (self.v - 1))
                )
                ** 2
            )
        )
        return CD2Wrapper(cond_distr)(u, v)

    def lambda_L(self):
        return 0

    def lambda_U(self):
        return 1
