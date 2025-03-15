import numpy as np
import sympy

from copul.families.archimedean.biv_archimedean_copula import BivArchimedeanCopula
from copul.families.other.independence_copula import IndependenceCopula
from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class Nelsen13(BivArchimedeanCopula):
    ac = BivArchimedeanCopula
    theta = sympy.symbols("theta", nonnegative=True)
    theta_interval = sympy.Interval(0, np.inf, left_open=False, right_open=True)
    special_cases = {0: IndependenceCopula}

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _raw_generator(self):
        return (1 - sympy.log(self.t)) ** self.theta - 1

    @property
    def _raw_inv_generator(self) -> SymPyFuncWrapper:
        return sympy.exp(1 - (self.y + 1) ** (1 / self.theta))

    @property
    def cdf(self):
        cdf = sympy.exp(
            1
            - (
                (1 - sympy.log(self.u)) ** self.theta
                + (1 - sympy.log(self.v)) ** self.theta
                - 1
            )
            ** (1 / self.theta)
        )
        return CDFWrapper(cdf)

    def lambda_L(self):
        return 0

    def lambda_U(self):
        return 0
