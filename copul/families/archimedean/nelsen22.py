import numpy as np
import sympy

from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.families.other.independence_copula import IndependenceCopula
from copul.wrapper.sympy_wrapper import SymPyFunctionWrapper


class Nelsen22(ArchimedeanCopula):
    ac = ArchimedeanCopula
    theta = sympy.symbols("theta", nonnegative=True)
    theta_interval = sympy.Interval(0, 1, left_open=False, right_open=False)

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
        return sympy.asin(1 - self.t**self.theta)

    @property
    def inv_generator(self) -> SymPyFunctionWrapper:
        indicator = sympy.Piecewise((1, self.y <= sympy.pi / 2), (0, True))
        gen = (1 - sympy.sin(self.y)) ** (1 / self.theta) * indicator
        return SymPyFunctionWrapper(gen)

    @property
    def cdf(self) -> SymPyFunctionWrapper:
        u = self.u
        theta = self.theta
        v = self.v
        cdf = sympy.Piecewise(
            (
                (sympy.sin(sympy.asin(u**theta - 1) + sympy.asin(v**theta - 1)) + 1)
                ** (1 / theta),
                sympy.asin(u**theta - 1) + sympy.asin(v**theta - 1) >= -sympy.pi / 2,
            ),
            (0, True),
        )
        return SymPyFunctionWrapper(cdf)

    def compute_gen_max(self):
        return np.pi / 2

    def lambda_L(self):
        return 0

    def lambda_U(self):
        return 0
