import numpy as np
import sympy

from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.families.archimedean.heavy_compute_arch import HeavyComputeArch
from copul.sympy_wrapper import SymPyFunctionWrapper


class Joe(HeavyComputeArch):
    ac = ArchimedeanCopula
    theta = sympy.symbols("theta", positive=True)
    theta_interval = sympy.Interval(1, np.inf, left_open=False, right_open=True)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _generator(self):
        return -sympy.log(1 - (1 - self.t) ** self.theta)

    @property
    def inv_generator(self):
        gen = 1 - (1 - sympy.exp(-self.y)) ** (1 / self.theta)
        return SymPyFunctionWrapper(gen)

    @property
    def cdf(self):
        theta = self.theta
        gen = 1 - (-((1 - self.u) ** theta - 1) * ((1 - self.v) ** theta - 1) + 1) ** (
            1 / theta
        )
        return SymPyFunctionWrapper(gen)

    def cond_distr_1(self, u=None, v=None):
        theta = self.theta
        cond_distr_1 = (
            -((1 - self.u) ** theta)
            * ((1 - (1 - self.u) ** theta) * ((1 - self.v) ** theta - 1) + 1)
            ** (1 / theta)
            * ((1 - self.v) ** theta - 1)
            / (
                (1 - self.u)
                * ((1 - (1 - self.u) ** theta) * ((1 - self.v) ** theta - 1) + 1)
            )
        )
        return SymPyFunctionWrapper(cond_distr_1)(u, v)

    def cond_distr_2(self, u=None, v=None):
        theta = self.theta
        cond_distr_2 = (
            (1 - self.v) ** theta
            * (1 - (1 - self.u) ** theta)
            * ((1 - (1 - self.u) ** theta) * ((1 - self.v) ** theta - 1) + 1)
            ** (1 / theta)
            / (
                (1 - self.v)
                * ((1 - (1 - self.u) ** theta) * ((1 - self.v) ** theta - 1) + 1)
            )
        )
        return SymPyFunctionWrapper(cond_distr_2)(u, v)

    def lambda_L(self):
        return 0

    def lambda_U(self):
        return 2 - 2 ** (1 / self.theta)


Nelsen6 = Joe

B5 = Joe
