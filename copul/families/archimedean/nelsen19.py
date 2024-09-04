import numpy as np
import sympy

from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.families.archimedean.nelsen1 import PiOverSigmaMinusPi
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class Nelsen19(ArchimedeanCopula):
    ac = ArchimedeanCopula
    theta = sympy.symbols("theta", nonnegative=True)
    theta_interval = sympy.Interval(0, np.inf, left_open=False, right_open=True)

    def __call__(self, *args, **kwargs):
        if args is not None and len(args) > 0:
            self.theta = args[0]
        if "theta" in kwargs and kwargs["theta"] == 0:
            del kwargs["theta"]
            return PiOverSigmaMinusPi()(**kwargs)
        return super().__call__(**kwargs)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _generator(self):
        return sympy.exp(self.theta / self.t) - sympy.exp(self.theta)

    @property
    def inv_generator(self):
        gen = self.theta / sympy.log(self.y + sympy.exp(self.theta))
        return SymPyFuncWrapper(gen)

    @property
    def cdf(self):
        cdf = self.theta / sympy.log(
            -sympy.exp(self.theta)
            + sympy.exp(self.theta / self.u)
            + sympy.exp(self.theta / self.v)
        )
        return CDFWrapper(cdf)
