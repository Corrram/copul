import numpy as np
import sympy

from copul.families.other.independence_copula import IndependenceCopula
from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.sympy_wrapper import SymPyFunctionWrapper


class GumbelHougaard(ArchimedeanCopula):
    ac = ArchimedeanCopula
    theta = sympy.symbols("theta", positive=True)
    theta_interval = sympy.Interval(1, np.inf, left_open=False, right_open=True)

    def __call__(self, *args, **kwargs):
        if args is not None and len(args) > 0:
            kwargs["theta"] = args[0]
        if "theta" in kwargs and kwargs["theta"] == 1:
            del kwargs["theta"]
            return IndependenceCopula()(**kwargs)
        return super().__call__(**kwargs)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _generator(self):
        return (-sympy.log(self.t)) ** self.theta

    @property
    def inv_generator(self):
        gen = sympy.exp(-self.y ** (1 / self.theta))
        return SymPyFunctionWrapper(gen)

    @property
    def cdf(self):
        if self.u == 0 or self.v == 0:
            return SymPyFunctionWrapper(0)
        gen = sympy.exp(
            -(
                (
                    (-sympy.log(self.u)) ** self.theta
                    + (-sympy.log(self.v)) ** self.theta
                )
                ** (1 / self.theta)
            )
        )
        return SymPyFunctionWrapper(gen)

    def lambda_L(self):
        return 0

    def lambda_U(self):
        return 2 - 2 ** (1 / self.theta)


Nelsen4 = GumbelHougaard

B6 = GumbelHougaard
