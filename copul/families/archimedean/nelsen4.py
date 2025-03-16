import numpy as np
import sympy

from copul.families.archimedean.biv_archimedean_copula import BivArchimedeanCopula
from copul.families.other.biv_independence_copula import BivIndependenceCopula
from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class GumbelHougaard(BivArchimedeanCopula):
    ac = BivArchimedeanCopula
    theta = sympy.symbols("theta", positive=True)
    theta_interval = sympy.Interval(1, np.inf, left_open=False, right_open=True)
    special_cases = {1: BivIndependenceCopula}

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _raw_generator(self):
        return (-sympy.log(self.t)) ** self.theta

    @property
    def _raw_inv_generator(self):
        return sympy.exp(-(self.y ** (1 / self.theta)))

    @property
    def cdf(self):
        if self.u == 0 or self.v == 0:
            return SymPyFuncWrapper(0)
        gen = sympy.exp(
            -(
                (
                    (-sympy.log(self.u)) ** self.theta
                    + (-sympy.log(self.v)) ** self.theta
                )
                ** (1 / self.theta)
            )
        )
        return CDFWrapper(gen)

    def lambda_L(self):
        return 0

    def lambda_U(self):
        return 2 - 2 ** (1 / self.theta)


Nelsen4 = GumbelHougaard

# B6 = GumbelHougaard
