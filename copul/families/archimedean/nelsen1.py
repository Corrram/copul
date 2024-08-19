import numpy as np
import sympy

from copul.cd1_wrapper import CD1Wrapper
from copul.cd2_wrapper import CD2Wrapper
from copul.cdf_wrapper import CDFWrapper
from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.families.other.independence_copula import IndependenceCopula
from copul.families.other.lower_frechet import LowerFrechet
from copul.sympy_wrapper import SymPyFunctionWrapper


class Clayton(ArchimedeanCopula):
    ac = ArchimedeanCopula
    theta_interval = sympy.Interval(-1, np.inf, left_open=False, right_open=True)

    @property
    def _generator(self):
        return ((1 / self.t) ** self.theta - 1) / self.theta

    def __call__(self, *args, **kwargs):
        if args is not None and len(args) > 0:
            kwargs["theta"] = args[0]
        if "theta" in kwargs and kwargs["theta"] == -1:
            del kwargs["theta"]
            return LowerFrechet()(**kwargs)
        if "theta" in kwargs and kwargs["theta"] == 0:
            del kwargs["theta"]
            return IndependenceCopula()(**kwargs)
        return super().__call__(**kwargs)

    @property
    def inv_generator(self):
        ind = sympy.Piecewise(
            (1, (self.y < -1 / self.theta) | (self.theta > 0)), (0, True)
        )
        cdf = ind * (self.theta * self.y + 1) ** (-1 / self.theta)
        return SymPyFunctionWrapper(cdf)

    @property
    def cdf(self):
        u = self.u
        theta = self.theta
        v = self.v
        cdf = sympy.Max((u ** (-theta) + v ** (-theta) - 1), 0) ** (-1 / theta)
        return CDFWrapper(cdf)

    def cond_distr_1(self, u=None, v=None):
        theta = self.theta
        cond_distr = sympy.Heaviside(-1 + self.u ** (-theta) + self.v ** (-theta)) / (
            self.u
            * self.u**theta
            * (-1 + self.u ** (-theta) + self.v ** (-theta))
            * (-1 + self.u ** (-theta) + self.v ** (-theta)) ** (1 / theta)
        )
        wrapped_cd1 = CD1Wrapper(cond_distr)
        evaluated_cd1 = wrapped_cd1(u, v)
        return evaluated_cd1

    def cond_distr_2(self, u=None, v=None):
        theta = self.theta
        cond_distr = sympy.Heaviside(
            (-1 + self.v ** (-theta) + self.u ** (-theta)) ** (-1 / theta)
        ) / (
            self.v
            * self.v**theta
            * (-1 + self.v ** (-theta) + self.u ** (-theta))
            * (-1 + self.v ** (-theta) + self.u ** (-theta)) ** (1 / theta)
        )
        return CD2Wrapper(cond_distr)(u, v)

    def _squared_cond_distr_1(self, u, v):
        theta = self.theta
        return sympy.Heaviside((-1 + v ** (-theta) + u ** (-theta)) ** (-1 / theta)) / (
            u**2
            * u ** (2 * theta)
            * (-1 + v ** (-theta) + u ** (-theta)) ** 2
            * (-1 + v ** (-theta) + u ** (-theta)) ** (2 / theta)
        )

    @property
    def pdf(self):
        theta = self.theta
        return (
            (self.u ** (-theta) + self.v ** (-theta) - 1) ** (-2 - 1 / theta)
            * self.u ** (-theta - 1)
            * self.v ** (-theta - 1)
            * (theta + 1)
        )

    @property
    def is_absolutely_continuous(self) -> bool:
        return self.theta >= 0

    def lambda_L(self):
        return 2 ** (-1 / self.theta)

    def lambda_U(self):
        return 0


Nelsen1 = Clayton

B4 = Clayton

PiOverSigmaMinusPi = Clayton(1)
