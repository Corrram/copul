import numpy as np
import sympy

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

    def __call__(self, **kwargs):
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
        # if u == 0 or v == 0:
        #     return SymPyFunctionWrapper(0)
        # cdf = (u ** (-theta) + v ** (-theta) - 1) ** (-1 / theta)
        cdf = sympy.Max((u ** (-theta) + v ** (-theta) - 1), 0) ** (-1 / theta)
        return SymPyFunctionWrapper(cdf)

    def cond_distr_1(self) -> SymPyFunctionWrapper:
        v = self.v
        u = self.u
        theta = self.theta
        cond_distr = sympy.Heaviside(-1 + u ** (-theta) + v ** (-theta)) / (
            u
            * u**theta
            * (-1 + u ** (-theta) + v ** (-theta))
            * (-1 + u ** (-theta) + v ** (-theta)) ** (1 / theta)
        )
        return SymPyFunctionWrapper(cond_distr)

    def cond_distr_2(self):
        v = self.v
        u = self.u
        theta = self.theta
        cond_distr = sympy.Heaviside(
            (-1 + v ** (-theta) + u ** (-theta)) ** (-1 / theta)
        ) / (
            v
            * v**theta
            * (-1 + v ** (-theta) + u ** (-theta))
            * (-1 + v ** (-theta) + u ** (-theta)) ** (1 / theta)
        )
        return SymPyFunctionWrapper(cond_distr)

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

PiOverSigmaMinusPi = Clayton()(theta=1)
