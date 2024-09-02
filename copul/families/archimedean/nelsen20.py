import logging
import random
import warnings

import numpy as np
import sympy

import scipy.optimize as opt

from copul.wrapper.cd2_wrapper import CD2Wrapper
from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.families.archimedean.heavy_compute_arch import HeavyComputeArch
from copul.families.archimedean.nelsen1 import PiOverSigmaMinusPi
from copul.families.other.independence_copula import IndependenceCopula
from copul.wrapper.sympy_wrapper import SymPyFunctionWrapper

log = logging.getLogger(__name__)


class Nelsen20(HeavyComputeArch):
    ac = ArchimedeanCopula
    theta = sympy.symbols("theta", nonnegative=True)
    theta_interval = sympy.Interval(0, np.inf, left_open=False, right_open=True)

    def __call__(self, **kwargs):
        if "theta" in kwargs and kwargs["theta"] == 0:
            del kwargs["theta"]
            return IndependenceCopula()(**kwargs)
        if "theta" in kwargs and kwargs["theta"] == 1:
            del kwargs["theta"]
            return PiOverSigmaMinusPi()(**kwargs)
        return super().__call__(**kwargs)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _generator(self):
        return sympy.exp(self.t ** (-self.theta)) - sympy.exp(1)

    @property
    def inv_generator(self):
        gen = sympy.log(self.y + sympy.E) ** (-1 / self.theta)
        return SymPyFunctionWrapper(gen)

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
