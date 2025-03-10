import numpy as np
import sympy

from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.families.other.lower_frechet import LowerFrechet
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class GenestGhoudi(ArchimedeanCopula):
    ac = ArchimedeanCopula
    theta = sympy.symbols("theta", positive=True)
    theta_interval = sympy.Interval(1, np.inf, left_open=False, right_open=True)
    special_cases = {
        1: LowerFrechet,
    }

    @property
    def is_absolutely_continuous(self) -> bool:
        return False

    @property
    def _generator(self):
        return (1 - self.t ** (1 / self.theta)) ** self.theta

    @property
    def inv_generator(self):
        ind = sympy.Piecewise((1, self.y <= 1), (0, True))
        gen = (1 - self.y ** (1 / self.theta)) ** self.theta * ind
        return SymPyFuncWrapper(gen)

    @property
    def cdf(self):
        cdf = (
            sympy.Max(
                1
                - (
                    (1 - self.u ** (1 / self.theta)) ** self.theta
                    + (1 - self.v ** (1 / self.theta)) ** self.theta
                )
                ** (1 / self.theta),
                0,
            )
            ** self.theta
        )
        return SymPyFuncWrapper(cdf)

    def lambda_L(self):
        return 0

    def lambda_U(self):
        return 2 - 2 ** (1 / self.theta)


Nelsen15 = GenestGhoudi
