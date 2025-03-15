import logging

import numpy as np
import sympy

from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.families.archimedean.heavy_compute_arch import HeavyComputeArch
from copul.families.other.independence_copula import IndependenceCopula
from copul.wrapper.cd2_wrapper import CD2Wrapper
from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper

log = logging.getLogger(__name__)


class Nelsen20(HeavyComputeArch):
    ac = ArchimedeanCopula
    theta = sympy.symbols("theta", nonnegative=True)
    theta_interval = sympy.Interval(0, np.inf, left_open=False, right_open=True)
    special_cases = {0: IndependenceCopula}

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _raw_generator(self):
        return sympy.exp(self.t ** (-self.theta)) - sympy.exp(1)

    @property
    def _raw_inv_generator(self):
        return sympy.log(self.y + sympy.E) ** (-1 / self.theta)

    @property
    def cdf(self):
        cdf = sympy.log(
            sympy.exp(self.u ** (-self.theta))
            + sympy.exp(self.v ** (-self.theta))
            - np.e
        ) ** (-1 / self.theta)
        return CDFWrapper(cdf)

    def cond_distr_2(self, u=None, v=None):
        theta = self.theta
        cond_distr = 1 / (
            self.v ** (theta + 1)
            * (
                sympy.exp(self.u ** (-theta) - self.v ** (-theta))
                + 1
                - np.e * sympy.exp(-(self.v ** (-theta)))
            )
            * sympy.log(
                sympy.exp(self.u ** (-theta)) + sympy.exp(self.v ** (-theta)) - np.e
            )
            ** ((theta + 1) / theta)
        )
        return CD2Wrapper(cond_distr)(u, v)
