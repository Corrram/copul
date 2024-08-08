import numpy as np
import sympy

from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.sympy_wrapper import SymPyFunctionWrapper


class Nelsen12(ArchimedeanCopula):
    ac = ArchimedeanCopula
    theta = sympy.symbols("theta", positive=True)
    theta_interval = sympy.Interval(1, np.inf, left_open=False, right_open=True)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _generator(self):
        return (1 / self.t - 1) ** self.theta

    @property
    def inv_generator(self):
        gen = 1 / (self.y ** (1 / self.theta) + 1)
        return SymPyFunctionWrapper(gen)

    @property
    def cdf(self):
        cdf = (
            1
            + ((self.u ** (-1) - 1) ** self.theta + (self.v ** (-1) - 1) ** self.theta)
            ** (1 / self.theta)
        ) ** (-1)
        return SymPyFunctionWrapper(cdf)

    def lambda_L(self):
        return 2 ** (-1 / self.theta)

    def lambda_U(self):
        return 2 - 2 ** (1 / self.theta)
