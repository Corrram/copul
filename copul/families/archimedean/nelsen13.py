import numpy as np
import sympy

from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.families.other.independence_copula import IndependenceCopula
from copul.wrapper.sympy_wrapper import SymPyFunctionWrapper


class Nelsen13(ArchimedeanCopula):
    ac = ArchimedeanCopula
    theta = sympy.symbols("theta", nonnegative=True)
    theta_interval = sympy.Interval(0, np.inf, left_open=False, right_open=True)

    def __call__(self, *args, **kwargs):
        if args is not None and len(args) > 0:
            kwargs["theta"] = args[0]
        if "theta" in kwargs and kwargs["theta"] == 0:
            del kwargs["theta"]
            return IndependenceCopula()(**kwargs)
        return super().__call__(**kwargs)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _generator(self):
        return (1 - sympy.log(self.t)) ** self.theta - 1

    @property
    def inv_generator(self) -> SymPyFunctionWrapper:
        gen = sympy.exp(1 - (self.y + 1) ** (1 / self.theta))
        return SymPyFunctionWrapper(gen)

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
